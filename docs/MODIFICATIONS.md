# What Can We Break? A Systematic Inventory of microgpt's Assumptions

Levin's method is not "damage a system and measure degradation." It is: **identify every assumption the algorithm depends on, break each one, and observe what competencies survive or emerge.**

Levin broke two assumptions of classical sorting:
1. A central controller executes the algorithm (→ replaced with autonomous cells)
2. Every operation succeeds (→ introduced frozen/damaged cells)

And discovered:
- Cell-view algorithms are more robust than centralized ones
- Damaged systems exhibit delayed gratification (rerouting)
- Chimeric arrays still sort, and exhibit unexpected aggregation

The question: **what are ALL the assumptions microgpt makes, and what happens when we break each one?**


---


## The Assumptions (and Their Violations)


### A1. SEQUENTIAL PROCESSING: Tokens are processed one at a time, left to right

microgpt's training loop:
```python
for pos_id in range(n):
    token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
    logits = gpt(token_id, pos_id, keys, values)
```

This is **top-down sequential control**. The loop dictates the order. Each token is passive — it gets processed when the controller says so.

#### Violations:

**A1a. Random order processing.** Shuffle the position order: process position 5, then 2, then 0, then 7... The KV cache still accumulates, but in a non-sequential order. The model must attend to keys/values that arrived in random order. Does it still learn?

**A1b. Parallel processing (cell-view tokens).** Process ALL positions simultaneously. Each token computes its own query and attends only to tokens that have already been "placed" (or to all tokens in parallel with a mask). This is the direct analog of Levin's cell-view sorting: each cell/token acts in parallel based on local information.

**A1c. Right-to-left processing.** Reverse the loop. Process the last token first. Structurally, this is a different language model (predicts the previous token from future context). But the weights are the same. Does a model trained left-to-right have any competence when run right-to-left?

**A1d. Bidirectional processing.** Process left-to-right, then right-to-left, accumulating information from both directions. Each position gets context from both past and future. This breaks the causal assumption but tests whether the model's representations are directional or symmetric.

**What we learn**: Is the sequential order an essential part of the GPT algorithm, or is it just a convention? If the model works with random-order processing, the competence lies in the attention mechanism itself (local policy), not in the imposed order (top-down control). This directly parallels Levin's finding that distributed sorting works as well as sequential sorting.


### A2. CAUSAL ATTENTION: Each token can only see past tokens

In microgpt, causality is enforced structurally: the KV cache at position t contains only keys/values from positions 0..t. There's no explicit causal mask — the structure IS the mask.

#### Violations:

**A2a. Full attention (no causality).** Pre-fill the KV cache with ALL positions' keys/values before computing any attention. Every token can attend to every other token, including future ones. This breaks the autoregressive assumption.

**A2b. Windowed attention.** Each token can only see the K most recent tokens (K < current position). This tests short-range vs. long-range dependency.

**A2c. Sparse/random attention.** Each token attends to a random subset of past tokens (not all of them). How much of the full attention pattern is actually needed?

**What we learn**: How much of the model's competence is due to having full causal context vs. just local context? If windowed attention (K=2) works almost as well as full attention, the model's knowledge is primarily local — like a cell-view sorting algorithm that only sees its immediate neighbors.


### A3. SHARED PARAMETERS: All positions and all documents use the same weights

microgpt has one set of weights. Position 0 and position 15 use the same Wq, Wk, Wv matrices. Document "alice" and document "bob" use the same model.

#### Violations:

**A3a. Position-specific weights.** Each position gets its own copy of the attention weights. Position 0 has Wq_0, position 1 has Wq_1, etc. This is extremely parameter-inefficient but tests whether weight sharing is compressing fundamentally different computations into shared matrices.

**A3b. Document-conditioned weights.** The weights are modulated by some function of the input document. A simple version: multiply each weight by a scalar that depends on the first token (or the document length, or a hash of the document). This tests whether the model needs document-specific adaptation.

**A3c. Gradually diverging weights.** Start with shared weights. During training, slowly clone them into position-specific copies and allow them to diverge. Track HOW MUCH they diverge — this measures the pressure for specialization.

**What we learn**: Weight sharing is THE key efficiency assumption of neural networks. If unshared weights produce qualitatively different behavior (not just more overfitting), it reveals that shared weights are compressing multiple distinct computations into one set of parameters — and the model's competence includes this compression.


### A4. DETERMINISTIC FORWARD PASS: Given the same input and weights, the output is always the same

microgpt's forward pass is deterministic. Randomness only enters during sampling (inference) and document selection (training).

#### Violations:

**A4a. Stochastic neurons.** Each ReLU activation has a probability p of firing even when it shouldn't (or not firing when it should). This is biological: neurons are noisy.

**A4b. Dropout during training.** Randomly zero some activations during forward pass. (This is well-studied, but testing it in the Levin framework — measuring DG index, robustness curves, rerouting — is new.)

**A4c. Stochastic attention.** Instead of deterministic softmax, sample from the attention distribution. Each attention step is a random draw from the weight distribution. This makes the forward pass fundamentally stochastic.

**A4d. Temperature in the forward pass (not just inference).** Scale attention logits by a temperature parameter that varies during training. High temperature = uniform attention (noisy). Low temperature = sharp attention (deterministic).

**What we learn**: Levin notes that biology excels with noisy hardware, and "noisiness of the local environment is a feature, not a bug." Does adding noise to the forward pass make the model MORE robust (as biological noise makes organisms more adaptable)?


### A5. GLOBAL BACKPROPAGATION: One loss, gradients flow through the entire graph

This is the big one. microgpt has:
```python
loss = (1 / n) * sum(losses)
loss.backward()
```

One loss, one backward pass, one optimization step. This is the ultimate top-down controller: the loss function has complete knowledge of the entire computation and sends precise instructions (gradients) to every parameter.

#### Violations:

**A5a. Block-local losses.** Each layer has its own loss (a linear probe that predicts next-token from that layer's output). No gradients flow between layers.

**A5b. Stop-gradient boundaries.** Gradients are cut at specific points (between layers, between attention and MLP, between the residual stream and the component).

**A5c. Random feedback (feedback alignment).** Instead of backpropagating through the actual weights, use random fixed matrices for the backward pass. Gradients are "approximate" — they point in roughly the right direction but are not computed correctly.

**A5d. Forward-only learning (Hebbian / perturbation-based).** No backward pass at all. Instead:
- Perturb a random parameter slightly
- Measure if loss went up or down
- Update in the direction that helped
This is extremely slow (one parameter at a time) but tests whether the STRUCTURE of backprop (the precise credit assignment) is essential.

**A5e. Delayed gradients.** Use gradients from N steps ago instead of the current step. The parameter update is based on stale information. How stale can it be before learning fails?

**A5f. Sign-only gradients.** Replace each gradient with just its sign (+1 or -1). All information about gradient magnitude is lost. Only the direction of each parameter's update is preserved.

**What we learn**: This is the most direct test of "top-down control vs. distributed competence." If the model learns with local losses or random feedback, it means the ARCHITECTURE ITSELF (the arrangement of attention + MLP + residual connections) contains enough inductive bias that precise credit assignment is not needed. The competence is in the structure, not in the optimizer. This is exactly Levin's finding: cell-view sorting works because the STRUCTURE of the algorithm (local comparison + swap) inherently moves toward the goal, without needing a top-down controller.


### A6. SYNCHRONOUS UPDATES: All parameters update at the same time, at the same rate

Adam updates all parameters at every step with the same learning rate schedule.

#### Violations:

**A6a. Asynchronous updates.** Different components update at different frequencies:
- Embeddings: every step
- Layer 0: every step
- Layer 1: every 2 steps
- Layer 2: every 5 steps
- Layer 3: every 10 steps

This creates a hierarchy of time scales, like biological systems where some processes are fast (neural signaling) and others slow (gene expression).

**A6b. Update budgets.** At each step, only K% of parameters are updated (the ones with the largest gradients, or random K%). This is a form of "metabolic constraint" — the system has limited energy for change.

**A6c. Momentum diversity.** Different components have different Adam momentum parameters. Some are "fast-adapting" (low momentum, high learning rate). Others are "slow-adapting" (high momentum, low learning rate). They coexist in the same model.

**What we learn**: Do different time scales create a natural hierarchy of function? Do fast-adapting components handle local/surface features while slow-adapting components handle global/structural features? This parallels developmental biology where morphogenesis operates across multiple time scales.


### A7. FIXED ARCHITECTURE: The model structure never changes

microgpt has n_layer layers, n_head heads per layer, fixed n_embd, forever.

#### Violations:

**A7a. Growth (neurogenesis).** Start with a minimal model (1 layer, 1 head). Periodically add new heads or layers (initialized randomly). The model must integrate new components into its existing computation.

**A7b. Pruning (apoptosis).** Start with a large model. Track each head's contribution. When a head's contribution falls below a threshold for N steps, remove it permanently. The model loses components over time, keeping only what's useful.

**A7c. Growth + Pruning (morphogenesis).** Combine: add random components, prune useless ones. The model's architecture EVOLVES during training. The final architecture is an emergent property of the training dynamics, not a design choice.

**A7d. Splitting.** A single head "divides" into two heads (like cell division). The new head starts as a copy of the old one, then they diverge. Does this create useful specialization?

**A7e. Merging.** Two heads merge into one (their weights are averaged). This is the opposite of splitting. When is merging harmful vs. neutral?

**What we learn**: This is the most direct analog of morphogenesis. The question is not "what is the best architecture?" but "can the training process BUILD the architecture?" If growth + pruning produces a model that's more robust or more efficient than a fixed architecture, it means there's a sense in which the model "develops" — it undergoes morphogenesis in architecture space. The parallels to Levin's developmental biology work are direct.


### A8. ONE OBJECTIVE: The model minimizes next-token cross-entropy loss

#### Violations:

**A8a. Multi-objective.** Different layers optimize different losses:
- Layer 0: next-token prediction
- Layer 1: reconstruct the input embedding
- Layer 2: predict the token two steps ahead
- Layer 3: next-token prediction

The model must serve multiple masters.

**A8b. Adversarial objective.** Some components are trained to MAXIMIZE loss while others minimize it. This is an internal adversarial game.

**A8c. Self-supervised auxiliary losses.** Add losses that don't directly relate to next-token prediction:
- Contrastive loss: embeddings of similar tokens should be close
- Reconstruction: the model must be able to reconstruct its input from its intermediate representations
- Consistency: the model's prediction should be stable under small input perturbations

**A8d. No explicit loss at all.** Train without backprop. Instead, use evolutionary strategies: randomly perturb all weights, keep perturbations that reduce loss on a batch of documents. The "loss" is only used for selection, not for gradient computation.

**What we learn**: Is next-token prediction the "goal" of the model, or is it just one way of specifying the goal? If auxiliary losses improve robustness, the model is capable of pursuing multiple objectives simultaneously (like a biological system that maintains homeostasis while also growing). If adversarial objectives reach equilibrium, we see the chimeric competition that Levin observed with opposing sorting directions.


### A9. PASSIVE DATA: Tokens are static input, parameters are the only learnable things

In microgpt, tokens are integers. They get embedded, processed, and predicted. They never change, they never "act."

#### Violations:

**A9a. Learnable token embeddings that update per-document.** The embedding for each character adapts during the processing of a single document (fast weights). After the document, the fast weights reset. The model has a form of "working memory" in its embeddings.

**A9b. Token agents.** Each token in the input "negotiates" its own representation. After the standard forward pass, each token adjusts its embedding based on the output (a form of iterative refinement or recurrence). Multiple rounds of this produce a stable representation.

**A9c. The input modifies the model.** During training, the current document doesn't just produce gradients for the parameters — it directly modifies the computation for the NEXT document. For example: the attention patterns from document N are used to modulate the weights for document N+1. This creates a form of inter-document learning that's faster than gradient descent.

**What we learn**: In biology, the substrate is not passive. Cells are "agential materials" (to use Levin's phrase). They don't just get pushed around by genes — they actively respond, adapt, and self-organize. If we make tokens or embeddings "active" (self-modifying), do new competencies emerge?


### A10. THE AUTOGRAD IS SACRED: Gradients are computed exactly via the chain rule

microgpt's Value class implements exact reverse-mode automatic differentiation. Every gradient is mathematically correct.

#### Violations:

**A10a. Noisy gradients.** Add Gaussian noise to every gradient: `p.grad += noise`. The signal is still there, but corrupted. How much noise can the system tolerate?

**A10b. Quantized gradients.** Round gradients to {-1, 0, +1} or to K bits of precision. How much gradient information is actually needed?

**A10c. Truncated backprop through positions.** In microgpt, gradients flow through the KV cache — position 5's loss sends gradients back to position 0's key/value computations. What if we cut this? Each position only sends gradients to its own parameters, not to earlier positions' cached values.

**A10d. Shuffled gradients.** Randomly reassign gradients among parameters of the same type. Head 0's gradient goes to head 3, and vice versa. The total gradient information is preserved, but the credit assignment is scrambled.

**A10e. No autograd at all — just the forward pass.** Remove the backward() call entirely. Instead, learn using:
- Random perturbation (evolutionary)
- Local Hebbian rules (correlations between activations)
- Reward-based (REINFORCE-style: evaluate output quality, use that scalar signal to update)

**What we learn**: Backpropagation is the "top-down controller" of learning. It tells each parameter exactly how to change. Levin showed that removing top-down control from sorting can make it MORE robust. If we degrade or remove backprop and the model still learns (even less efficiently), it means the architecture has intrinsic competence for organizing itself toward the goal — the competence is in the STRUCTURE, not in the optimization algorithm.


### A11. THE KV CACHE IS TRANSPARENT: Every past token is remembered perfectly

Each token's key and value vectors are cached and accessible to all future tokens, unmodified.

#### Violations:

**A11a. Decaying cache.** Older entries are gradually "forgotten" — multiplied by a decay factor (0.9^age, for example). Recent tokens are vivid, old tokens are faded.

**A11b. Limited cache.** Only the K most recent entries are kept. Older entries are evicted. This forces the model to work with limited memory.

**A11c. Noisy cache.** Add noise to cached entries over time (as if memory degrades). The longer an entry is cached, the noisier it gets.

**A11d. Shared/compressed cache.** Instead of storing exact K,V for each position, compress old entries (average them, or keep only a summary). This tests whether the model can work with lossy memory.

**A11e. Persistent cache across documents.** DON'T clear the cache between documents. The model processes document after document with a growing cache. It must deal with irrelevant past context. Does it learn to ignore stale cache entries? Does cross-document context help or hurt?

**What we learn**: The KV cache is the model's "memory" of the sequence. Degrading it is like damaging a biological organism's memory. Does the model exhibit robustness to memory degradation? Can it compensate (reroute through other mechanisms)?


### A12. THE DATASET IS FIXED AND EXTERNAL: Training data comes from a static file

#### Violations:

**A12a. Self-generated data.** After initial training, the model generates its own training data (sample from itself, then train on those samples). This is a form of self-play or bootstrapping.

**A12b. Adversarial data.** Intentionally create training examples that the model is worst at (hard mining). The dataset adapts to the model, not just the model to the dataset.

**A12c. Curriculum.** Start with easy examples (short names, common patterns), gradually introduce harder ones. The "environment" changes in response to the model's competence.

**A12d. Data poisoning.** Mix in corrupted or random data at various rates. At what rate does data noise overwhelm learning? How does this interact with architectural damage?

**What we learn**: The interaction between a learning system and its environment is itself a form of collective behavior. A model that generates its own data is the beginning of autonomy — a system that shapes its own learning context.


---


## Summary: The Assumption Inventory

| # | Assumption | Category | Levin Parallel |
|---|---|---|---|
| A1 | Sequential L→R processing | Execution order | Top-down control |
| A2 | Causal (past-only) attention | Information flow | Cell visibility range |
| A3 | Shared parameters across positions | Resource sharing | All cells same hardware |
| A4 | Deterministic forward pass | Reliability | Reliable hardware |
| A5 | Global backpropagation | Credit assignment | Top-down controller |
| A6 | Synchronous parameter updates | Coordination | Synchronous cell actions |
| A7 | Fixed architecture | Structure | Fixed body plan |
| A8 | Single objective (next-token loss) | Goals | Single sorting direction |
| A9 | Passive tokens/data | Agency of substrate | Passive vs. agential material |
| A10 | Exact gradient computation | Precision of control | Perfect instruction execution |
| A11 | Perfect KV cache memory | Memory | Perfect cell memory |
| A12 | Fixed external dataset | Environment | Fixed input array |


---


## Recommended Priority (most Levin-faithful, most tractable, most novel)

### Tier 1: Direct Levin analogs (MUST DO)

**A4 + A10: Unreliable hardware + imprecise control** (frozen/noisy components, noisy/truncated gradients)
- This is the frozen cell experiment. Already planned but should include gradient perturbations too.

**A5: Breaking top-down control** (local learning rules)
- The cell-view GPT. The most fundamental Levin experiment.

**A8: Mixed/opposing objectives** (chimeric GPT)
- Different components with different goals. Already planned.

### Tier 2: High novelty, tractable (SHOULD DO)

**A1: Breaking execution order** (random order, parallel processing)
- Very Levin-like (cell-view = parallel execution). Easy to implement. Novel for GPT.

**A7: Architecture morphogenesis** (growth + pruning during training)
- Direct morphogenesis analog. Moderate implementation effort. High novelty.

**A6: Asynchronous updates** (multi-timescale learning)
- Easy to implement. Biologically grounded. Unexplored in this context.

### Tier 3: Speculative but interesting (COULD DO)

**A11: Degraded memory** (decaying/limited/noisy KV cache)
- Easy to implement. Tests memory robustness.

**A9: Agential tokens** (iterative refinement, self-modifying embeddings)
- Hard to implement cleanly. Very novel.

**A3: Unshared parameters** (position-specific weights)
- Memory-expensive. Tests the necessity of weight sharing.

### Tier 4: Radical (EXPLORATORY)

**A12: Self-generated data** (the model trains on its own outputs)
- Tests autonomy. Interesting but may be unstable.

**A10e: No autograd** (evolutionary learning)
- Extremely slow. But the ultimate test of architectural competence.


---


## What This Changes About the Architecture

The original ARCHITECTURE.md had three experiment categories:
1. Frozen components
2. Cell-view GPT (local learning)
3. Chimeric GPT

This analysis reveals that these are just THREE of TWELVE assumption-violations. The full experimental program is much richer. The most impactful additions:

1. **Execution order experiments** (A1): process tokens in random/reverse/parallel order
2. **Architecture morphogenesis** (A7): grow and prune during training
3. **Multi-timescale learning** (A6): different update frequencies per component
4. **Gradient degradation** (A10): noisy, quantized, shuffled, or absent gradients
5. **Memory degradation** (A11): decaying, limited, or noisy KV cache

Each of these is a distinct "assumption violation" experiment in the Levin style, producing its own robustness curves, DG indices, and competence trajectories.

The full MorphoGPT experimental program is not "freeze some heads and measure loss." It is: **systematically question every assumption of the GPT algorithm and characterize the system's competence when each assumption is violated.**

This is the perturbation methodology taken to its fullest extent: not breaking one thing, but methodically interrupting the entire structure to reveal what holds it together.
