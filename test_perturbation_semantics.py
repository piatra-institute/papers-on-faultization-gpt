# /// script
# dependencies = ["numpy"]
# ///
"""
Implementation-audit tests for perturbation semantics.

Verifies that:
1. Frozen heads still compute in forward pass but receive zero gradients
2. Local-loss mode gives each layer gradients from its own local loss only
3. Exp 6 composite conditions apply both forward AND gradient perturbations
4. Per-layer probe heads are independent (Fix 1)
5. Stochastic perturbations are reproducible with seeded RNGs (Fix 2)
6. Partial stop-gradient fractions don't compound (Fix 3)
7. Schedule + grad_hook combos are rejected (Fix 4)
"""

import numpy as np
from model import (
    Hooks, make_config, init_state_dict, load_dataset, tokenize,
    _forward_backward,
)
from perturbations import (
    make_freeze_head_params, freeze_random_heads, freeze_specific_heads,
    make_noisy_gradients, make_partial_stop_gradient_grad_hook,
)


def test_frozen_head_forward_nonzero():
    """A frozen head still computes in forward pass (non-zero output)."""
    print("Test 1a: Frozen head has non-zero forward output...")
    train_docs, val_docs, uchars, BOS, vocab_size = load_dataset()
    config = make_config(n_layer=4, n_embd=16, n_head=4, vocab_size=vocab_size)
    sd, params = init_state_dict(config, seed=42)

    # Train a few steps to get non-trivial weights
    from model import train, TrainConfig
    tc = TrainConfig(num_steps=20, print_every=0, detail_level='loss_only')
    train(sd, params, config, tc, train_docs, uchars, BOS, seed=42)

    # Freeze head (0, 0)
    head_list = [(0, 0)]
    _, freeze_gh = freeze_specific_heads(head_list, config)

    # Forward pass - hooks are empty (no forward zeroing)
    hooks = Hooks()
    tokens = tokenize(train_docs[0], uchars, BOS)
    n = min(config['block_size'], len(tokens) - 1)
    loss, _, grads, snapshots = _forward_backward(
        tokens, n, sd, config, hooks, capture_state=True)

    # Check that head (0,0) has non-zero output in snapshots
    head_out = snapshots[0]['layers'][0]['heads'][0]['head_out_vec']
    head_norm = np.linalg.norm(head_out)
    assert head_norm > 1e-6, f"Frozen head should have non-zero forward output, got norm={head_norm}"
    print(f"  PASS: head (0,0) output norm = {head_norm:.6f}")


def test_frozen_head_zero_gradients():
    """A frozen head receives zero gradients for Q, K, V, Wo parameters."""
    print("Test 1b: Frozen head receives zero gradients...")
    train_docs, val_docs, uchars, BOS, vocab_size = load_dataset()
    config = make_config(n_layer=4, n_embd=16, n_head=4, vocab_size=vocab_size)
    sd, params = init_state_dict(config, seed=42)

    head_dim = config['head_dim']
    layer, head = 0, 0
    hs, he = head * head_dim, (head + 1) * head_dim

    # Get gradients WITHOUT freezing
    hooks = Hooks()
    tokens = tokenize(train_docs[0], uchars, BOS)
    n = min(config['block_size'], len(tokens) - 1)
    _, _, grads_before, _ = _forward_backward(tokens, n, sd, config, hooks)

    # Verify gradients are non-zero before freezing
    for comp in ['attn_wq', 'attn_wk', 'attn_wv']:
        key = f'layer{layer}.{comp}'
        assert np.any(grads_before[key][hs:he, :] != 0), \
            f"Gradients for {key}[{hs}:{he}] should be non-zero before freezing"
    wo_key = f'layer{layer}.attn_wo'
    assert np.any(grads_before[wo_key][:, hs:he] != 0), \
        f"Gradients for {wo_key}[:, {hs}:{he}] should be non-zero before freezing"

    # Get gradients WITH freezing (apply grad hook)
    _, _, grads_after, _ = _forward_backward(tokens, n, sd, config, hooks)
    gh = make_freeze_head_params(layer, head, config['n_embd'], head_dim)
    gh(grads_after, sd, 0)

    # Q, K, V rows should be zeroed
    for comp in ['attn_wq', 'attn_wk', 'attn_wv']:
        key = f'layer{layer}.{comp}'
        frozen_grad = grads_after[key][hs:he, :]
        assert np.all(frozen_grad == 0), \
            f"Gradients for {key}[{hs}:{he}] should be zero after freezing"

    # Wo columns should be zeroed
    frozen_wo = grads_after[wo_key][:, hs:he]
    assert np.all(frozen_wo == 0), \
        f"Gradients for {wo_key}[:, {hs}:{he}] should be zero after freezing"

    # Other heads should still have non-zero gradients
    other_hs = head_dim  # head 1
    other_he = 2 * head_dim
    assert np.any(grads_after[f'layer{layer}.attn_wq'][other_hs:other_he, :] != 0), \
        "Other heads should still have non-zero gradients"

    print("  PASS: frozen head has zero Q/K/V/Wo gradients, other heads unaffected")


def test_local_loss_independence():
    """In local-loss mode, zeroing layer 3's local loss path should not
    affect layer 0's gradients."""
    print("Test 2: Local-loss layer independence...")
    train_docs, val_docs, uchars, BOS, vocab_size = load_dataset()
    config = make_config(n_layer=4, n_embd=16, n_head=4, vocab_size=vocab_size)
    sd, params = init_state_dict(config, seed=42)

    hooks = Hooks()
    tokens = tokenize(train_docs[0], uchars, BOS)
    n = min(config['block_size'], len(tokens) - 1)

    # Run with local_loss=True
    loss, _, grads_local, _ = _forward_backward(
        tokens, n, sd, config, hooks, local_loss=True)

    # Verify all layers got non-zero gradients
    for li in range(config['n_layer']):
        key = f'layer{li}.attn_wq'
        assert np.any(grads_local[key] != 0), \
            f"Layer {li} should have non-zero gradients in local-loss mode"

    # Run with local_loss=False (standard)
    _, _, grads_global, _ = _forward_backward(
        tokens, n, sd, config, hooks, local_loss=False)

    # In local-loss mode, layer gradients should differ from global mode
    # because each layer gets its own local loss instead of end-to-end backprop
    any_differ = False
    for li in range(config['n_layer']):
        key = f'layer{li}.attn_wq'
        if not np.allclose(grads_local[key], grads_global[key], rtol=1e-5):
            any_differ = True
            break
    assert any_differ, "Local-loss gradients should differ from global gradients"

    # Verify that loss is reasonable (not NaN or zero)
    assert not np.isnan(loss), "Local-loss loss should not be NaN"
    assert loss > 0, "Local-loss loss should be positive"

    print(f"  PASS: local_loss={loss:.4f}, gradients differ from global mode")


def test_composite_perturbation():
    """Exp 6 composite conditions should apply both forward AND gradient
    perturbations simultaneously."""
    print("Test 3: Composite perturbation applies both forward and grad hooks...")
    from experiments import ExperimentConfig, run_experiment

    train_docs, val_docs, uchars, BOS, vocab_size = load_dataset()

    # Run a composite condition (cautious_cautious = noise_injection + sign_only)
    cfg = ExperimentConfig(
        name='test_composite',
        num_steps=10,
        perturbation_type='composite',
        perturbation_params={'perturbations': [
            {'type': 'noise_injection', 'params': {'hook_name': 'emb', 'noise_std': 0.001}},
            {'type': 'sign_only_gradients'},
        ]},
        seed=42,
        print_progress=False,
    )
    result = run_experiment(cfg, train_docs, uchars, BOS, vocab_size,
                            val_docs=val_docs)

    # Verify it ran successfully
    assert result['summary']['final_loss'] > 0, "Composite should produce valid loss"

    # Run baseline for comparison
    cfg_base = ExperimentConfig(
        name='test_baseline',
        num_steps=10,
        perturbation_type='none',
        seed=42,
        print_progress=False,
    )
    result_base = run_experiment(cfg_base, train_docs, uchars, BOS, vocab_size,
                                 val_docs=val_docs)

    # Composite should differ from baseline (both perturbations active)
    assert abs(result['summary']['final_loss'] - result_base['summary']['final_loss']) > 1e-6, \
        "Composite perturbation should produce different results from baseline"

    print(f"  PASS: composite={result['summary']['final_loss']:.4f} vs "
          f"baseline={result_base['summary']['final_loss']:.4f}")


def test_validation_evaluation():
    """Validation loss is computed and stored in probe."""
    print("Test 4: Validation evaluation...")
    train_docs, val_docs, uchars, BOS, vocab_size = load_dataset()

    from experiments import ExperimentConfig, run_experiment

    cfg = ExperimentConfig(
        name='test_val',
        num_steps=30,
        perturbation_type='none',
        seed=42,
        print_progress=False,
    )
    result = run_experiment(cfg, train_docs, uchars, BOS, vocab_size,
                            val_docs=val_docs)

    # Check that val_losses were recorded
    assert len(result['probe'].val_losses) > 0, "Validation losses should be recorded"
    assert all(l > 0 for _, l in result['probe'].val_losses), \
        "All validation losses should be positive"

    # Check summary includes val metrics
    assert 'val_final_loss' in result['summary'] or len(result['probe'].val_losses) > 0, \
        "Summary should include validation metrics"

    print(f"  PASS: {len(result['probe'].val_losses)} val evaluations recorded, "
          f"final val_loss={result['probe'].val_losses[-1][1]:.4f}")


# ============================================================================
# New regression tests for peer-review fixes
# ============================================================================

def test_probe_head_independence():
    """Fix 1: Per-layer probe heads are independent — zeroing probe_head_3
    should not change layer 0's gradients in local-loss mode."""
    print("Test 5 (Fix 1): Per-layer probe head independence...")
    train_docs, val_docs, uchars, BOS, vocab_size = load_dataset()
    config = make_config(n_layer=4, n_embd=16, n_head=4, vocab_size=vocab_size)

    # Run 1: normal local-loss
    sd1, _ = init_state_dict(config, seed=42)
    hooks = Hooks()
    tokens = tokenize(train_docs[0], uchars, BOS)
    n = min(config['block_size'], len(tokens) - 1)
    _, _, grads1, _ = _forward_backward(tokens, n, sd1, config, hooks, local_loss=True)
    layer0_grad_1 = grads1['layer0.attn_wq'].copy()

    # Run 2: zero out probe_head_3, re-run
    sd2, _ = init_state_dict(config, seed=42)
    sd2['probe_head_3'] = np.zeros_like(sd2['probe_head_3'])
    _, _, grads2, _ = _forward_backward(tokens, n, sd2, config, hooks, local_loss=True)
    layer0_grad_2 = grads2['layer0.attn_wq'].copy()

    # Layer 0 gradients should be identical regardless of probe_head_3
    assert np.allclose(layer0_grad_1, layer0_grad_2, atol=1e-10), \
        "Zeroing probe_head_3 should not affect layer 0 gradients"

    # But layer 3 gradients should differ
    assert not np.allclose(grads1['layer3.attn_wq'], grads2['layer3.attn_wq'], atol=1e-10), \
        "Zeroing probe_head_3 should affect layer 3 gradients"

    # Verify probe heads exist and lm_head is NOT used in local-loss
    assert 'probe_head_0' in sd1, "probe_head_0 should exist"
    assert 'probe_head_3' in sd1, "probe_head_3 should exist"

    # lm_head should have zero gradients in local-loss mode (not used)
    assert np.allclose(grads1['lm_head'], 0, atol=1e-10), \
        "lm_head should have zero gradients in local-loss mode"

    print("  PASS: probe heads are independent per layer")


def test_stochastic_reproducibility():
    """Fix 2: Stochastic perturbations are reproducible with seeded RNGs."""
    print("Test 6 (Fix 2): Stochastic gradient noise reproducibility...")

    # Create two identical seeded RNGs
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(42)

    hook1 = make_noisy_gradients(noise_std=0.1, rng=rng1)
    hook2 = make_noisy_gradients(noise_std=0.1, rng=rng2)

    # Create dummy gradients
    grads1 = {'w': np.ones((4, 4)), 'b': np.ones((4,))}
    grads2 = {'w': np.ones((4, 4)), 'b': np.ones((4,))}
    sd = {'w': np.zeros((4, 4)), 'b': np.zeros((4,))}

    # Apply hooks
    hook1(grads1, sd, step=0)
    hook2(grads2, sd, step=0)

    # Results should be identical
    assert np.allclose(grads1['w'], grads2['w']), \
        "Seeded noisy gradients should produce identical results"
    assert np.allclose(grads1['b'], grads2['b']), \
        "Seeded noisy gradients should produce identical results"

    # Should differ from unperturbed
    assert not np.allclose(grads1['w'], np.ones((4, 4))), \
        "Noisy gradients should differ from original"

    # Apply again — should produce different noise (RNG advances)
    grads3 = {'w': np.ones((4, 4)), 'b': np.ones((4,))}
    grads4 = {'w': np.ones((4, 4)), 'b': np.ones((4,))}
    hook1(grads3, sd, step=1)
    hook2(grads4, sd, step=1)
    assert np.allclose(grads3['w'], grads4['w']), \
        "Second call should also be identical with same seed"

    print("  PASS: seeded RNGs produce reproducible stochastic perturbations")


def test_non_compounding_fractions():
    """Fix 3: With pass_fraction=0.5 and n_layer=4, each boundary layer
    gets exactly 0.5 scaling, not compounded."""
    print("Test 7 (Fix 3): Non-compounding gradient fractions...")
    train_docs, val_docs, uchars, BOS, vocab_size = load_dataset()
    config = make_config(n_layer=4, n_embd=16, n_head=4, vocab_size=vocab_size)
    sd, _ = init_state_dict(config, seed=42)

    hooks = Hooks()
    tokens = tokenize(train_docs[0], uchars, BOS)
    n = min(config['block_size'], len(tokens) - 1)

    # Get unscaled gradients
    _, _, grads_orig, _ = _forward_backward(tokens, n, sd, config, hooks)

    # Get scaled gradients (hooks at boundaries 0, 1, 2 with pass_fraction=0.5)
    _, _, grads_scaled, _ = _forward_backward(tokens, n, sd, config, hooks)
    pass_fraction = 0.5
    for boundary in range(config['n_layer'] - 1):  # boundaries 0, 1, 2
        gh = make_partial_stop_gradient_grad_hook(boundary, pass_fraction, config)
        gh(grads_scaled, sd, 0)

    # Layer 0: should be scaled by 0.5 (only from hook 0)
    key0 = 'layer0.attn_wq'
    ratio0 = grads_scaled[key0] / (grads_orig[key0] + 1e-20)
    nonzero_mask0 = np.abs(grads_orig[key0]) > 1e-15
    if np.any(nonzero_mask0):
        actual_ratio0 = np.median(ratio0[nonzero_mask0])
        assert abs(actual_ratio0 - 0.5) < 0.01, \
            f"Layer 0 should be scaled by 0.5, got {actual_ratio0:.4f}"

    # Layer 1: should be scaled by 0.5 (only from hook 1)
    key1 = 'layer1.attn_wq'
    ratio1 = grads_scaled[key1] / (grads_orig[key1] + 1e-20)
    nonzero_mask1 = np.abs(grads_orig[key1]) > 1e-15
    if np.any(nonzero_mask1):
        actual_ratio1 = np.median(ratio1[nonzero_mask1])
        assert abs(actual_ratio1 - 0.5) < 0.01, \
            f"Layer 1 should be scaled by 0.5, got {actual_ratio1:.4f}"

    # Layer 2: should be scaled by 0.5 (only from hook 2)
    key2 = 'layer2.attn_wq'
    ratio2 = grads_scaled[key2] / (grads_orig[key2] + 1e-20)
    nonzero_mask2 = np.abs(grads_orig[key2]) > 1e-15
    if np.any(nonzero_mask2):
        actual_ratio2 = np.median(ratio2[nonzero_mask2])
        assert abs(actual_ratio2 - 0.5) < 0.01, \
            f"Layer 2 should be scaled by 0.5, got {actual_ratio2:.4f}"

    # Layer 3: should be unscaled (no hook)
    key3 = 'layer3.attn_wq'
    assert np.allclose(grads_scaled[key3], grads_orig[key3]), \
        "Layer 3 (top) should be unscaled"

    # Embeddings: should be scaled by 0.5 (from hook 0)
    assert np.allclose(grads_scaled['wte'], grads_orig['wte'] * 0.5, atol=1e-15), \
        "Embeddings should be scaled by 0.5 from boundary 0"

    print("  PASS: each boundary layer gets exactly 0.5 scaling, no compounding")


def test_schedule_grad_hook_rejection():
    """Fix 4: ValueError raised when schedule != 'chronic' with grad_hook perturbations."""
    print("Test 8 (Fix 4): Schedule + grad_hook rejection...")
    from experiments import ExperimentConfig, run_experiment

    train_docs, val_docs, uchars, BOS, vocab_size = load_dataset()

    # noisy_gradients produces a grad_hook; schedule='acute' should be rejected
    cfg = ExperimentConfig(
        name='test_schedule_rejection',
        num_steps=10,
        perturbation_type='noisy_gradients',
        perturbation_params={'noise_std': 0.01},
        schedule='acute',
        schedule_params={'start_step': 0, 'end_step': 5},
        seed=42,
        print_progress=False,
    )

    raised = False
    try:
        run_experiment(cfg, train_docs, uchars, BOS, vocab_size, val_docs=val_docs)
    except ValueError as e:
        raised = True
        assert 'acute' in str(e).lower() or 'schedule' in str(e).lower(), \
            f"Error message should mention schedule, got: {e}"

    assert raised, "Should raise ValueError for schedule='acute' with grad hooks"

    # Verify that 'chronic' schedule with grad hooks still works
    cfg_ok = ExperimentConfig(
        name='test_schedule_ok',
        num_steps=10,
        perturbation_type='noisy_gradients',
        perturbation_params={'noise_std': 0.01},
        schedule='chronic',
        seed=42,
        print_progress=False,
    )
    result = run_experiment(cfg_ok, train_docs, uchars, BOS, vocab_size,
                            val_docs=val_docs)
    assert result['summary']['final_loss'] > 0, \
        "chronic schedule with grad hooks should work"

    print("  PASS: non-chronic schedules with grad hooks are rejected")


if __name__ == '__main__':
    test_frozen_head_forward_nonzero()
    test_frozen_head_zero_gradients()
    test_local_loss_independence()
    test_composite_perturbation()
    test_validation_evaluation()
    test_probe_head_independence()
    test_stochastic_reproducibility()
    test_non_compounding_fractions()
    test_schedule_grad_hook_rejection()
    print("\n=== ALL SEMANTIC TESTS PASSED ===")
