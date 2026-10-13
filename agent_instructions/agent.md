````markdown
# agent.md

# Porting RL Repository from MuJoCo to MuJoCo XLA (MJX)

## Objective

Port an existing reinforcement learning repository built on  

The final system must:

- Support batched parallel simulation
- Run fully on GPU
- Use functional programming style
- Be compatible with **:contentReference[oaicite:2]{index=2}**
- Preserve training stability and reproducibility

---

# 1. Core Migration Philosophy

Do NOT rewrite the RL algorithm logic unless necessary.

Instead:

1. Replace physics backend
2. Convert environment to pure functional form
3. Vectorize environment using `vmap`
4. Replace rollout loops with `lax.scan`
5. JIT-compile training

Migration must be incremental.

---

# 2. Critical Architectural Differences

| Classic MuJoCo | MJX |
|---------------|------|
| Imperative | Functional |
| Mutable state | Immutable state |
| CPU stepping | GPU/TPU via XLA |
| Single env common | Batched envs standard |

All environment logic must become **pure functions**.

No global state.
No mutation.
No side effects.

---

# 3. Environment Refactor

## 3.1 Reset

Original:
```python
obs = env.reset()
````

New:

```python
state = env.reset(key)
```

Requirements:

* Accept JAX PRNG key
* Return state object
* No internal randomness without key

---

## 3.2 Step

Original:

```python
obs, reward, done, info = env.step(action)
```

New:

```python
new_state, obs, reward, done, info = env.step(state, action)
```

Rules:

* Must be pure
* No in-place mutation
* No Python side effects
* Must operate on JAX arrays only

---

# 4. RNG Handling

All randomness must use JAX keys.

Bad:

```python
np.random.randn()
```

Correct:

```python
key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, shape)
```

Keys must be threaded through:

* reset
* step
* policy sampling
* exploration noise

---

# 5. Batched Environment

Single environment:

```python
state = env.reset(key)
```

Batched:

```python
keys = jax.random.split(master_key, batch_size)
states = jax.vmap(env.reset)(keys)
```

Batched step:

```python
states, obs, reward, done, info = jax.vmap(env.step)(states, actions)
```

Batch dimension must always be leading dimension.

---

# 6. Rollout Refactor (MANDATORY)

Remove Python loops over timesteps.

Replace:

```python
for t in range(T):
    ...
```

With:

```python
def rollout_fn(carry, _):
    state, key = carry
    key, subkey = jax.random.split(key)

    action = policy(...)
    state, obs, reward, done, info = env.step(state, action)

    return (state, key), (obs, reward, done)

(final_state, _), trajectory = jax.lax.scan(
    rollout_fn,
    init_carry,
    None,
    length=T
)
```

All rollout logic must use `lax.scan`.

---

# 7. Neural Network Migration

If the repo uses PyTorch:

Option A (Recommended):

* Rewrite networks using Flax
* Use Optax for optimizers

Option B:

* Keep PyTorch networks and only port environment (not recommended)

For full GPU acceleration:

* Policy
* Value network
* Loss
* Update step
  must all be JAX-based.

---

# 8. Training Step Structure

Example:

```python
@jax.jit
def train_step(params, opt_state, batch, key):

    def loss_fn(params):
        loss = compute_loss(params, batch, key)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss
```

Guidelines:

* JIT entire train step
* Avoid JITing tiny helpers
* Avoid host-device transfers

---

# 9. Replay Buffer (Off-Policy Only)

Rules:

* No Python lists
* No object storage
* Use preallocated JAX arrays
* Use functional updates

Sampling must:

* Use JAX RNG
* Be vectorized
* Avoid Python indexing loops

If buffer too large:

* Store on host
* Move minibatches to device only when needed

---

# 10. Memory Constraints (8GB GPU Target)

Required settings:

```python
jax.config.update("jax_enable_x64", False)
```

Guidelines:

* Use float32
* Avoid storing full trajectories unnecessarily
* Avoid large replay buffers on GPU
* Increase batch size gradually

Test memory scaling with batch sizes:

* 32
* 64
* 128
* 256

---

# 11. Performance Optimization

After basic migration:

1. JIT rollout + training
2. Ensure no Python loops remain
3. Profile device memory
4. Measure:

   * Steps per second
   * GPU utilization
   * Compile time

If slow:

* Check for hidden Python control flow
* Remove dynamic shapes
* Avoid unnecessary `.block_until_ready()`

---

# 12. Determinism & Validation

You must verify:

* Reset reproducibility
* Reward scale similarity
* Learning curve similarity
* No exploding gradients after port

Validation process:

1. Run 1 env unbatched
2. Compare reward trajectory
3. Enable batching
4. Increase batch gradually

---

# 13. Migration Order (STRICT)

Follow this exact order:

1. Replace MuJoCo backend with MJX
2. Convert environment to functional
3. Validate single environment
4. Add batching via `vmap`
5. Replace rollout with `lax.scan`
6. Port neural networks to JAX
7. JIT train step
8. Optimize memory
9. Benchmark

Do NOT skip steps.

---

# 14. Success Criteria

Migration is successful if:

* Entire training loop runs on GPU
* Batch size ≥ 128 supported
* Throughput exceeds CPU MuJoCo version
* Learning curve remains stable
* No Python loops inside jitted sections
* No host-device transfer bottlenecks

---

# 15. Deliverables

* Functional MJX environment module
* Batched training implementation
* JIT-compiled rollout
* JAX-based policy + value networks
* Benchmark script
* Memory profiling script
* Reproducibility test script

---

End of agent.md

```
```
