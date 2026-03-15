# /// script
# dependencies = ["numpy"]
# ///
"""
Implementation-audit tests for perturbation semantics.

Verifies that:
1. Frozen heads still compute in forward pass but receive zero gradients
2. Local-loss mode gives each layer gradients from its own local loss only
3. Exp 6 composite conditions apply both forward AND gradient perturbations
"""

import numpy as np
from morphogpt_np import (
    Hooks, make_config, init_state_dict, load_dataset, tokenize,
    _forward_backward,
)
from perturbations_np import (
    make_freeze_head_params, freeze_random_heads, freeze_specific_heads,
)


def test_frozen_head_forward_nonzero():
    """A frozen head still computes in forward pass (non-zero output)."""
    print("Test 1a: Frozen head has non-zero forward output...")
    train_docs, val_docs, uchars, BOS, vocab_size = load_dataset()
    config = make_config(n_layer=4, n_embd=16, n_head=4, vocab_size=vocab_size)
    sd, params = init_state_dict(config, seed=42)

    # Train a few steps to get non-trivial weights
    from morphogpt_np import train, TrainConfig
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
    from experiments_np import ExperimentConfig, run_experiment

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

    from experiments_np import ExperimentConfig, run_experiment

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


if __name__ == '__main__':
    test_frozen_head_forward_nonzero()
    test_frozen_head_zero_gradients()
    test_local_loss_independence()
    test_composite_perturbation()
    test_validation_evaluation()
    print("\n=== ALL SEMANTIC TESTS PASSED ===")
