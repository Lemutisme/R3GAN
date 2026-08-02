Now I have complete understanding of both codebases. Here is the detailed audit report.

---

# R3GAN PyTorch vs JAX Drifting: Audit Report

## Executive Summary

The PyTorch implementation is **not a line-for-line port** of the JAX reference — it is a **parallel re-implementation** with significant algorithmic divergences. The existing parity tests in [test_drift_parity.py](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/tests/test_drift_parity.py) validate the PyTorch code against an intermediate `drifting_models` PyTorch package,  **not against the JAX code directly** . I found **7 critical algorithmic discrepancies** and  **9 performance issues** .

---

## 1. Mathematical & Algorithmic Alignment

### CRITICAL-1: Multi-Temperature Force Aggregation Strategy

**JAX** ([drift_loss.py:88-121](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/drifting/drift_loss.py#L88-L121)): All R-values produce forces that are independently normalized then  **summed into one force vector** , and a **single MSE loss** is computed from the summed goal.

```python
# JAX: sum forces, then MSE
for R in R_list:
    total_force_R = ...
    f_norm_val = (total_force_R ** 2).mean()
    force_scale = sqrt(clip(f_norm_val, min=1e-8))
    force_across_R += total_force_R / force_scale  # normalized sum
goal_scaled = old_gen_scaled + force_across_R
loss = mean((gen_scaled - goal_scaled) ** 2)  # single MSE
```

**PyTorch** ([drift_loss.py:81-129](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/drift_loss.py#L81-L129)): Default `temperature_aggregation="per_temperature_mse"` computes  **independent MSE losses per temperature** , then averages. Cross-temperature force interaction is lost.

```python
# PyTorch default: MSE per temperature, then average
for temperature in temperatures:
    loss, _, stats = drifting_stopgrad_loss(...)  # independent MSE each
    loss_terms.append(loss)
total_loss = torch.stack(loss_terms).mean()
```

 **Impact** : The gradient directions are fundamentally different. JAX allows force vectors from different scales to jointly determine the target position; PyTorch independently pulls toward each temperature's target.

**Fix at** [drift_loss.py:144](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/drift_loss.py#L144) — change default:

```python
temperature_aggregation: str = "sum_drifts_then_mse"  # was "per_temperature_mse"
```

And verify that the `sum_drifts_then_mse` path in [drift_loss.py:282-300](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/drift_loss.py#L282-L300) includes per-R normalization matching JAX's `total_force_R / force_scale` pattern. Currently the per-temperature drift normalization (`normalize_drifts`) normalizes by a global RMS, whereas JAX normalizes each R's force by its own mean-squared norm. The fix:

```python
# In _normalize_drifts or the sum_drifts_then_mse path:
# Normalize per-temperature drift by its own force scale (matching JAX)
drift_sq_mean = (drift_tau_full ** 2).mean()
force_scale = torch.sqrt(torch.clamp(drift_sq_mean, min=1e-8))
drift_tau_full = drift_tau_full / force_scale
```

---

### CRITICAL-2: Weighting Mechanism — Post-Softmax Multiplicative vs Pre-Softmax Log-Additive

**JAX** ([drift_loss.py:93-97](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/drifting/drift_loss.py#L93-L97)): Weights are applied **after** the geometric-mean affinity:

```python
affinity = sqrt(softmax(logits, -1) * softmax(logits, -2))
affinity = affinity * targets_w  # post-softmax multiplicative
```

**PyTorch** ([drift_field.py:113-114](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/drift_field.py#L113-L114)): Weights enter as  **log-additive terms before softmax** :

```python
logit_neg = logit_neg + negative_log_weights.view(1, -1)  # pre-softmax
logits = cat([logit_pos, logit_neg], dim=1)
affinity = sqrt(softmax(logits, -1) * softmax(logits, -2))
```

 **Impact** : These are NOT mathematically equivalent. Pre-softmax log-weighting is `softmax(l_i + log(w_i)) = w_i * exp(l_i) / Σ(w_j * exp(l_j))` — the weight enters the normalization denominator. Post-softmax multiplication does not renormalize.

For the JAX code specifically: `weight_neg = repeat(uncond_w, 'b -> (b f) k', f=..., k=n_uncond)` at [train.py:131](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/drifting/train.py#L131), where `uncond_w = (cfg - 1) * (gen_per_label - 1) / n_uncond`. This is a per-sample weight for the unconditional negatives.

 **Fix** : To exactly match JAX, the affinity computation should apply weights post-softmax. In [drift_field.py:84-126](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/drift_field.py#L84-L126):

```python
def compute_affinity_matrices(x, y_pos, y_neg, *, config, negative_log_weights=None,
                               generated_negative_count=None):
    dist_pos = torch.cdist(x, y_pos)
    dist_neg = torch.cdist(x, y_neg)
    # ... masking unchanged ...
    logit_pos = -(dist_pos / config.temperature)
    logit_neg = -(dist_neg / config.temperature)
    # DO NOT add log_weights to logits — apply post-softmax instead
    logits = torch.cat([logit_pos, logit_neg], dim=1)
    row_affinity = torch.softmax(logits, dim=-1)
    if config.normalize_over_x:
        col_affinity = torch.softmax(logits, dim=-2)
        affinity = torch.sqrt(torch.clamp(row_affinity * col_affinity, min=config.eps))
    else:
        affinity = row_affinity
    # Apply weights AFTER geometric mean (matching JAX)
    if negative_log_weights is not None:
        weights = torch.exp(negative_log_weights).view(1, -1)
        n_pos = y_pos.shape[0]
        weight_vector = torch.cat([torch.ones(1, n_pos, device=x.device), weights], dim=1)
        affinity = affinity * weight_vector
    n_pos = y_pos.shape[0]
    return affinity[:, :n_pos], affinity[:, n_pos:]
```

---

### CRITICAL-3: Scale Normalization — Coordinate Space vs Feature Space

**JAX** ([drift_loss.py:66-78](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/drifting/drift_loss.py#L66-L78)): Normalizes **all inputs** to order-1 coordinates, then uses **raw distance** for the kernel separately normalized by `scale`:

```python
scale = weighted_dist.mean() / targets_w.mean()
scale_inputs = clip(scale / sqrt(S), min=1e-3)
old_gen_scaled = old_gen / scale_inputs        # for target computation
targets_scaled = targets / scale_inputs        # for target computation
dist_normed = dist / clip(scale, min=1e-3)     # for kernel logits
logits = -dist_normed / R                      # kernel uses dist/scale/R
```

**PyTorch pixel-space** ([drift_training_loop.py:722-751](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/drift_training_loop.py#L722-L751)): Normalizes features, then scales temperature:

```python
feature_scale = clamp(mean_dist / sqrt(dim), min=eps)
x_grouped /= feature_scale                     # normalize features
effective_temperature = temperature * sqrt(dim) # scale temperature
logits = -dist(x_norm, y_norm) / effective_temperature
```

 **Analysis** : After substitution, both produce `logits ≈ -dist / (mean_dist * temperature)`, so the logit computation is equivalent. However the **target/goal MSE** differs:

* JAX: MSE is in `scaled` space (divided by `scale_inputs = scale/sqrt(S)`)
* PyTorch: MSE is in `feature_scale` normalized space (divided by `mean_dist/sqrt(dim)`)

These scales differ because JAX's `scale` uses **weighted** distances while PyTorch's `mean_dist` is unweighted.

**Fix at** [drift_training_loop.py:727-731](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/drift_training_loop.py#L727-L731): Use weighted distances matching JAX:

```python
with torch.no_grad():
    all_dists = torch.cdist(x_grouped, torch.cat([x_grouped, y_neg_grouped, y_pos_grouped], dim=1))
    if neg_log_weights_grouped is not None:
        w = torch.cat([torch.ones(G, negatives_per_group, device=Device),
                       torch.exp(neg_log_weights_grouped[:, negatives_per_group:]),
                       torch.ones(G, positives_per_group, device=Device)], dim=1)
        weighted_dist = (all_dists * w.unsqueeze(1)).mean() / w.mean()
    else:
        weighted_dist = all_dists.mean()
    feature_scale = torch.clamp(weighted_dist / sqrt(dim), min=normalization_eps)
```

---

### CRITICAL-4: Positional Embedding — Sinusoidal (JAX) vs Learnable Random (PyTorch)

**JAX** ([generator.py:389-394](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/drifting/models/generator.py#L389-L394)):

```python
pos_embed = self.param('pos_embed', sincos_init(hidden_size, num_patches), ...)
x = (x + pos_embed)  # frozen sinusoidal 2D
```

**PyTorch** ([dit_like.py:54-55](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/models/dit_like.py#L54-L55)):

```python
self.patch_positional_embedding = nn.Parameter(
    torch.randn(1, self.num_patches, config.hidden_dim) * 0.02  # learnable random
)
```

 **Impact** : Different initialization and trainability of positional embeddings changes how spatial structure is encoded. The sinusoidal embedding is deterministic and non-trainable; the random one is learnable with a very different starting distribution.

**Fix at** [dit_like.py:52-58](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/models/dit_like.py#L52-L58):

```python
if config.use_patch_positional_embedding:
    pos_embed = self._build_sincos_pos_embed(config.hidden_dim, self.num_patches)
    self.register_buffer("patch_positional_embedding", pos_embed)
else:
    self.patch_positional_embedding = None

@staticmethod
def _build_sincos_pos_embed(embed_dim, num_patches):
    grid_size = int(num_patches ** 0.5)
    half = embed_dim // 2
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    omega = 1.0 / (10000 ** (torch.arange(half // 2, dtype=torch.float64) / (half / 2)))
    pos_h = torch.outer(grid_h.repeat_interleave(grid_size), omega.float())
    pos_w = torch.outer(grid_w.repeat(grid_size), omega.float())
    emb = torch.cat([pos_h.sin(), pos_h.cos(), pos_w.sin(), pos_w.cos()], dim=1)
    return emb.unsqueeze(0)  # [1, num_patches, embed_dim]
```

---

### CRITICAL-5: Patch Embedding — Conv2d (PyTorch) vs Linear Projection (JAX)

**JAX** ([generator.py:384-387](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/drifting/models/generator.py#L384-L387)):

```python
# Reshape to patches then linear project
x = x.reshape(B, grid_h, effective_p, grid_w, effective_p, C)
x = transpose(x, (0,1,3,2,4,5)).reshape(B, num_patches, effective_p*effective_p*C)
x = TorchLinear(hidden_size)(x)  # xavier_uniform init
```

**PyTorch** ([dit_like.py:46-51](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/models/dit_like.py#L46-L51)):

```python
self.patch_embed = nn.Conv2d(
    in_channels, out_channels=hidden_dim,
    kernel_size=patch_size, stride=patch_size,
)  # kaiming_uniform init (PyTorch default)
```

 **Impact** :

1. **Different initialization** : JAX uses `xavier_uniform`; PyTorch Conv2d defaults to `kaiming_uniform`.
2. **Numerically different** even with same weights due to im2col + matmul (conv) vs explicit reshape + linear.

**Fix at** [dit_like.py:46-51](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/models/dit_like.py#L46-L51):

```python
self.patch_embed = nn.Conv2d(
    in_channels=config.in_channels,
    out_channels=config.hidden_dim,
    kernel_size=config.patch_size,
    stride=config.patch_size,
)
# Match JAX xavier_uniform initialization
nn.init.xavier_uniform_(self.patch_embed.weight.view(
    config.hidden_dim, -1).T.reshape(self.patch_embed.weight.shape))
nn.init.zeros_(self.patch_embed.bias)
```

---

### CRITICAL-6: CFG Conditioning — Different Embedding Architecture

**JAX** ([generator.py:578-595](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/drifting/models/generator.py#L578-L595)):

```python
cond = class_embed(c)  # class embedding
for i in range(noise_coords):
    cond += noise_embeds[i](noise_labels[:, i])  # additive noise coord embeds
cfg_scale_t = cfg_norm(cfg_embedder(cfg_scale))  # sinusoidal → MLP → RMSNorm
cond = cond + cfg_scale_t * 0.02  # tiny scale for cfg
```

**PyTorch** ([dit_like.py:153-176](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/models/dit_like.py#L153-L176)):

```python
class_cond = class_embedding(class_labels)  # class embedding
alpha_cond = alpha_embedding(alpha)  # MLP or fourier_MLP (no norm, no 0.02 scale)
style_cond = style_embedding(style_indices).sum(1) / sqrt(n_styles)
return class_cond + alpha_cond + style_cond  # additive, equal scale
```

 **Impact** : JAX multiplies CFG embedding by 0.02 and applies RMSNorm, giving it tiny influence relative to class embedding. PyTorch's alpha embedding has equal magnitude to class embedding — a very different conditioning balance.

**Fix at** [dit_like.py:153-176](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/models/dit_like.py#L153-L176) — if matching JAX behavior is desired, add RMSNorm and scale:

```python
def _build_conditioning(self, *, class_labels, alpha, ...):
    class_cond = self.class_embedding(class_labels)
    alpha_cond = self.alpha_norm(self.alpha_embedding(alpha)) * 0.02
    # ... style handling ...
    return class_cond + alpha_cond + style_cond
```

---

### CRITICAL-7: AdaLN Modulation — fp32 Precision Guard

**JAX** ([generator.py:291-296](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/drifting/models/generator.py#L291-L296)):

```python
# Explicit fp32 for modulation MLP
TorchLinear(out_dim, dtype=jnp.float32, ...)  # fp32 projection
chunks = adaLN_mod(c.astype(jnp.float32)).astype(self.dtype)
```

**PyTorch** ([dit_like.py:214-217](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/models/dit_like.py#L214-L217)):

```python
self.modulation = nn.Sequential(
    nn.SiLU(),
    nn.Linear(hidden_dim, hidden_dim * 6),  # runs in whatever autocast provides
)
```

 **Impact** : Under bf16 autocast, the PyTorch modulation runs in bf16, causing precision loss in shift/scale/gate values that compound through the transformer blocks.

**Fix at** [dit_like.py:221-223](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/models/dit_like.py#L221-L223):

```python
def forward(self, tokens, condition):
    with torch.amp.autocast(device_type=tokens.device.type, enabled=False):
        modulation = self.modulation(condition.float())
    shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
    # rest unchanged
```

---

### Minor Discrepancy 1: Epsilon Values

| Parameter       | JAX       | PyTorch                                    |
| --------------- | --------- | ------------------------------------------ |
| cdist eps       | `1e-8`  | N/A (torch.cdist uses L2 default)          |
| affinity clamp  | `1e-6`  | `1e-12` (DriftFieldConfig.eps)           |
| scale clamp     | `1e-3`  | `1e-8` (normalization_eps)               |
| self-mask value | `100.0` | `1e6` (DriftFieldConfig.self_mask_value) |
| RMSNorm eps     | `1e-6`  | `1e-8` (dit_like.py RMSNorm)             |

The `1e-12` affinity clamp is more aggressive than JAX's `1e-6`, which could cause numerical issues under bf16.

**Fix at** [drift_field.py:14](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/drift_field.py#L14):

```python
eps: float = 1e-6  # was 1e-12
```

**Fix at** [dit_like.py:312](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/models/dit_like.py#L312):

```python
def __init__(self, hidden_dim: int, eps: float = 1e-6, ...):  # was 1e-8
```

---

### Minor Discrepancy 2: Alpha Sampling Distribution

**JAX** ([train.py:74-79](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/drifting/train.py#L74-L79)): Power-law distribution for CFG scale:

```python
cfg = (cfg_min^pw + frac * (cfg_max^pw - cfg_min^pw))^(1/pw)
```

**PyTorch** ([drift_training_loop.py:685-690](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/drift_training_loop.py#L685-L690)): Uniform distribution:

```python
alpha = rand * (alpha_max - alpha_min) + alpha_min
```

This means PyTorch samples alpha uniformly in `[1,4]` while JAX samples with a power-law bias toward higher values (when `neg_cfg_pw > 1`). This changes the training distribution substantially.

---

### Minor Discrepancy 3: SwiGLU Inner Dimension Rounding

**JAX** ([generator.py:270-271](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/drifting/models/generator.py#L270-L271)):

```python
hid_size = int(2/3 * mlp_hidden_dim)
hid_size = (hid_size + 31) // 32 * 32  # round up to multiple of 32
```

**PyTorch** ([dit_like.py:245](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/models/dit_like.py#L245)):

```python
inner_dim = int(hidden_dim * mlp_ratio * (2.0 / 3.0))  # no rounding
```

For `hidden_dim=1024, mlp_ratio=4.0`: JAX gives `2752` (rounded), PyTorch gives `2730`.

**Fix at** [dit_like.py:245](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/models/dit_like.py#L245):

```python
inner_dim = int(hidden_dim * mlp_ratio * (2.0 / 3.0))
inner_dim = ((inner_dim + 31) // 32) * 32  # round to multiple of 32
```

---

### Minor Discrepancy 4: SwiGLU Gate Order

**JAX** ([generator.py:143-147](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/drifting/models/generator.py#L143-L147)):

```python
w1 = Linear(intermediate)(x)
w3 = Linear(intermediate)(x)  # separate projections
out = silu(w1) * w3
return Linear(hidden)(out)
```

**PyTorch** ([dit_like.py:249-251](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/models/dit_like.py#L249-L251)):

```python
value, gate = self.linear_in(tokens).chunk(2, dim=-1)  # fused projection
return self.linear_out(value * F.silu(gate))
```

JAX applies `silu` to `w1` (first projection), PyTorch applies `silu` to `gate` (second half of fused projection). This means `silu(w1) * w3` vs `value * silu(gate)`. With different weight assignments, the roles are swapped.

**Fix at** [dit_like.py:249-251](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/models/dit_like.py#L249-L251):

```python
def forward(self, tokens):
    gate, value = self.linear_in(tokens).chunk(2, dim=-1)  # swap order
    return self.linear_out(F.silu(gate) * value)
```

---

## 2. PyTorch Efficiency & High-Performance Optimization

### PERF-1: Python Loop Over Groups — Graph Break

 **File** : [drift_stage2.py:101](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/drift_stage2.py#L101)

```python
for group_index in range(groups):  # Python loop = graph break for torch.compile
    generated_group = generated_grouped[group_index]
    # ... per-group loss computation
```

 **Fix** : Vectorize using the already-batched `_batched_drifting_stopgrad_loss` pattern from [drift_training_loop.py:693-808](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/drift_training_loop.py#L693-L808), which operates on `[G, N, D]` tensors without Python loops.

---

### PERF-2: Unnecessary `.clone()` for Self-Masking

 **Files** : [drift_field.py:108](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/drift_field.py#L108), [drift_loss.py:403](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/drift_loss.py#L403), [drift_training_loop.py:745](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/drift_training_loop.py#L745)

```python
dist_neg = dist_neg.clone()  # full tensor copy just to modify diagonal
dist_neg[diagonal, diagonal] += self_mask_value
```

 **Fix** : Use `scatter_` or construct an additive mask instead:

```python
mask = torch.zeros_like(dist_neg)
mask[..., diagonal, diagonal] = config.self_mask_value
dist_neg = dist_neg + mask  # no clone needed, autograd-safe
```

This saves one full tensor allocation per forward pass.

---

### PERF-3: Redundant `.clone()` for Unconditional Log-Weights

 **File** : [drift_training_loop.py:404-410](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/drift_training_loop.py#L404-L410)

```python
safe_weights = AlphaWeights.clone()       # clone 1
zero_mask = safe_weights == 0.0
safe_weights[zero_mask] = 1.0
unc_log_w = torch.log(safe_weights).unsqueeze(1).expand(...)
if zero_mask.any():
    unc_log_w = unc_log_w.clone()         # clone 2
    unc_log_w[zero_mask] = finfo.min
```

 **Fix** : Use `torch.where` to avoid both clones:

```python
log_w = torch.where(
    AlphaWeights > 0,
    torch.log(AlphaWeights),
    torch.tensor(torch.finfo(torch.float32).min, device=Device),
)
unc_log_w = log_w.unsqueeze(1).expand(Groups, unconditional_per_group)
neg_log_weights_grouped = torch.cat([gen_zeros, unc_log_w], dim=1)
```

---

### PERF-4: Queue Sampling Creates Individual Python Lists

 **File** : [drift_queue.py:214-224](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/drift_queue.py#L214-L224)

```python
def _sample_from_queue(queue, count, ...):
    Permutation = torch.randperm(QueueLength)[:count]
    return [queue[Index] for Index in Permutation.tolist()]  # Python list comprehension
```

Each call creates a Python list of individual tensors then `torch.stack`s them. For large queues this is slow.

 **Fix** :

```python
def _sample_from_queue(queue, count, ...):
    QueueLength = len(queue)
    if QueueLength >= count:
        indices = torch.randperm(QueueLength)[:count]
    else:
        indices = torch.randint(0, QueueLength, (count,))
    # Stack once from deque as tensor
    buffer = torch.stack(list(queue))  # materialize once
    return buffer[indices]  # single index operation
```

Better yet, consider using a pre-allocated ring buffer tensor instead of a deque, similar to JAX's `ArrayMemoryBank`.

---

### PERF-5: `F.mse_loss` Default Reduction vs Manual Reduction

 **File** : [drift_loss.py:76](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/drift_loss.py#L76)

```python
loss = F.mse_loss(x, target)  # reduction='mean' over all elements
```

**JAX** ([drift_loss.py:131](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/drifting/drift_loss.py#L131)):

```python
loss = jnp.mean(diff ** 2, axis=(-1, -2))  # per-batch loss [B]
```

JAX computes per-sample loss then averages, PyTorch flattens everything. For equal batch sizes this is the same, but for uneven accumulation this matters.

 **Fix** : Use explicit reduction matching JAX:

```python
loss = (x - target).pow(2).mean(dim=-1)  # per-sample MSE [B]
loss = loss.mean()  # then average over batch
```

---

### PERF-6: Missing `inplace=True` on Activation Functions

 **File** : [dit_like.py:215](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/models/dit_like.py#L215)

```python
nn.SiLU(),  # not inplace
```

 **Fix** :

```python
nn.SiLU(inplace=True),
```

This saves one tensor allocation per modulation projection. Similarly in `AlphaConditioningEmbedding` at [dit_like.py:424-427](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/models/dit_like.py#L424-L427).

---

### PERF-7: Repeated `.contiguous()` in Unpatchify

 **File** : [dit_like.py:183](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/models/dit_like.py#L183)

```python
patch_values = patch_values.permute(0, 5, 1, 3, 2, 4).contiguous()
return patch_values.view(batch, out_channels, ...)  # view requires contiguous
```

The `.contiguous()` + `.view()` can be replaced with a single `.reshape()`:

```python
patch_values = patch_values.permute(0, 5, 1, 3, 2, 4)
return patch_values.reshape(batch, out_channels, self.config.image_size, self.config.image_size)
```

This lets PyTorch decide the optimal strategy and is `torch.compile`-friendly.

---

### PERF-8: Device Transfer in Log-Weights Construction

 **File** : [drift_loss.py:409-412](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/drift_loss.py#L409-L412)

```python
if log_weights.device != dist_neg.device:
    log_weights = log_weights.to(device=dist_neg.device)  # host-device sync
if log_weights.dtype != dist_neg.dtype:
    log_weights = log_weights.to(dtype=dist_neg.dtype)
```

These defensive transfers can cause implicit synchronization. The log_weights should be constructed on the correct device from the start.

 **Fix** : Ensure `build_negative_log_weights` always receives the correct device/dtype, then remove these guards.

---

### PERF-9: Per-Slot Python Loop in Drift Normalization

 **File** : [drift_loss.py:569-573](vscode-webview://18pr22f35gd3qa7eu6tpi2p8anmg3gccjrqoiqitm6q7u9982id1/training/drift_loss.py#L569-L573)

```python
for vector_index in range(drifts.shape[1]):  # Python loop over vector slots
    scale = _drift_scale(drifts[:, vector_index, :], eps=eps)
    scales.append(scale)
```

 **Fix** : Vectorize:

```python
# Vectorized per-slot scale
per_slot_sq = drifts.pow(2).sum(dim=-1) / float(drifts.shape[-1])  # [B, V]
scale_tensor = torch.sqrt(per_slot_sq.mean(dim=0))  # [V]
scale_tensor = torch.clamp(scale_tensor, min=eps).detach()
return drifts / scale_tensor.view(1, -1, 1), scale_tensor
```

---

## 3. Summary Table

| #  | Category     | Severity           | File:Line                  | Description                                                                             |
| -- | ------------ | ------------------ | -------------------------- | --------------------------------------------------------------------------------------- |
| C1 | Algorithm    | **Critical** | drift_loss.py:144          | Multi-temp aggregation: sum forces then MSE (JAX) vs independent MSE per temp (PyTorch) |
| C2 | Algorithm    | **Critical** | drift_field.py:113         | Weighting: post-softmax multiply (JAX) vs pre-softmax log-add (PyTorch)                 |
| C3 | Algorithm    | **Critical** | drift_training_loop.py:727 | Scale normalization uses unweighted distances; JAX uses weighted                        |
| C4 | Architecture | **Critical** | dit_like.py:54             | Positional embedding: sinusoidal frozen (JAX) vs learnable random (PyTorch)             |
| C5 | Architecture | **Critical** | dit_like.py:46             | Patch embed init: xavier_uniform (JAX) vs kaiming_uniform (PyTorch Conv2d)              |
| C6 | Architecture | **Critical** | dit_like.py:163            | CFG conditioning: RMSNorm + 0.02 scale (JAX) vs raw addition (PyTorch)                  |
| C7 | Precision    | **Critical** | dit_like.py:214            | AdaLN modulation: fp32 guarded (JAX) vs bf16 autocast (PyTorch)                         |
| M1 | Numerical    | Medium             | drift_field.py:14          | eps values differ (1e-12 vs 1e-6 for affinity clamp)                                    |
| M2 | Algorithm    | Medium             | drift_training_loop.py:685 | Alpha: uniform (PT) vs power-law (JAX)                                                  |
| M3 | Architecture | Medium             | dit_like.py:245            | SwiGLU inner dim not rounded to 32                                                      |
| M4 | Architecture | Medium             | dit_like.py:250            | SwiGLU gate/value swap                                                                  |
| P1 | Performance  | High               | drift_stage2.py:101        | Python loop over groups → graph break                                                  |
| P2 | Performance  | Medium             | drift_field.py:108         | Unnecessary `.clone()` for diagonal mask                                              |
| P3 | Performance  | Medium             | drift_training_loop.py:404 | Double `.clone()` for log-weights                                                     |
| P4 | Performance  | Medium             | drift_queue.py:220         | Per-element Python list in queue sampling                                               |
| P5 | Performance  | Low                | drift_loss.py:76           | `F.mse_loss` default reduction misaligns with JAX per-sample                          |
| P6 | Performance  | Low                | dit_like.py:215            | Missing `inplace=True` on SiLU                                                        |
| P7 | Performance  | Low                | dit_like.py:183            | `.contiguous()` + `.view()` instead of `.reshape()`                               |
| P8 | Performance  | Low                | drift_loss.py:409          | Defensive device transfers cause sync                                                   |
| P9 | Performance  | Medium             | drift_loss.py:569          | Python loop in per-slot drift normalization                                             |

---

 **Bottom line** : The core drift field and loss modules (C1, C2, C3) are the highest-priority fixes. The generator architecture differences (C4-C7) are intentional design choices unless exact weight-transfer from JAX checkpoints is required. The performance items (P1-P9) are straightforward wins for training throughput without affecting numerics.
