# GPU Utilization Fixes for Drift Training Loop

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate GPU pipeline stalls and unnecessary data transfers that keep GPU utilization at ~35% during drift training.

**Architecture:** Five independent fixes targeting: (1) vectorize scalar alpha computation, (2) defer `.item()` GPU syncs in loss stats, (3) fix redundant copies in data loading, (4) move queue storage to GPU, (5) increase DataLoader workers. Each fix is tested independently via a short training run measuring `sec/kimg`.

**Tech Stack:** PyTorch, CUDA, existing R3GAN training infrastructure

---

## Baseline

Before any fix, capture baseline performance:

```bash
CUDA_VISIBLE_DEVICES=2,3 TOTAL_KIMG=20 GPUS=2 BATCH=64 bash scripts/train_drift_cifar10_pixel.sh
```

Record `sec/kimg` from the last tick's log output.

---

### Task 1: Vectorize Alpha Weight Computation

**Problem:** `cfg_alpha_to_unconditional_weight` is called per-group in a Python list comprehension with `.item()`, causing G GPU-CPU syncs per step.

**Files:**
- Modify: `training/drift_field.py:17-28`
- Modify: `training/drift_training_loop.py:285-296`
- Test: `tests/test_drift_training.py` (run existing parity tests)

**Step 1: Add vectorized alpha weight function in drift_field.py**

Add after the existing `cfg_alpha_to_unconditional_weight` function:

```python
def cfg_alpha_to_unconditional_weight_vectorized(
    alpha: torch.Tensor,
    n_generated_negatives: int,
    n_unconditional_negatives: int,
) -> torch.Tensor:
    """Vectorized version: alpha is a 1-D tensor, returns a 1-D tensor of weights."""
    if n_generated_negatives <= 1:
        raise ValueError("n_generated_negatives must be > 1")
    if n_unconditional_negatives <= 0:
        raise ValueError("n_unconditional_negatives must be > 0")
    return ((alpha - 1.0) * (n_generated_negatives - 1)) / n_unconditional_negatives
```

**Step 2: Update training loop to use vectorized version**

In `drift_training_loop.py`, replace lines 285-296:

```python
# OLD: list comprehension with .item()
AlphaWeights = torch.tensor([
    cfg_alpha_to_unconditional_weight(
        alpha=float(Value.item()), ...
    ) for Value in AlphaGrouped
], device=Device, dtype=torch.float32)

# NEW: single tensor op, no .item()
AlphaWeights = cfg_alpha_to_unconditional_weight_vectorized(
    AlphaGrouped, negatives_per_group, unconditional_per_group,
)
```

**Step 3: Run existing tests**

```bash
cd /workspace/R3GAN && python -m pytest tests/test_drift_parity.py tests/test_drift_training.py -x -v
```

**Step 4: Verify with short training run, compare sec/kimg to baseline**

---

### Task 2: Defer .item() Calls in Loss Stats

**Problem:** `compute_weighted_drift` and `drifting_stopgrad_loss` call `.item()` on every stat, forcing GPU-CPU sync inside the per-group loop (G groups × 4 stats = many syncs per step).

**Files:**
- Modify: `training/drift_loss.py:46-52` (compute_weighted_drift stats)
- Modify: `training/drift_loss.py:77` (drifting_stopgrad_loss stats)
- Modify: `training/drift_training_loop.py:340-350` (DriftStats construction)
- Test: `tests/test_drift_parity.py`

**Step 1: Return detached tensors instead of floats in compute_weighted_drift**

```python
# OLD (line 46-52):
stats = {
    "drift_norm": float(drift.norm(dim=-1).mean().item()),
    "drift_pos_norm": float(drift_pos.norm(dim=-1).mean().item()),
    "drift_neg_norm": float(drift_neg.norm(dim=-1).mean().item()),
    "attraction_scale": float(config.attraction_scale),
    "repulsion_scale": float(config.repulsion_scale),
}

# NEW: keep as detached tensors, no .item()
stats = {
    "drift_norm": drift.norm(dim=-1).mean().detach(),
    "drift_pos_norm": drift_pos.norm(dim=-1).mean().detach(),
    "drift_neg_norm": drift_neg.norm(dim=-1).mean().detach(),
    "attraction_scale": float(config.attraction_scale),
    "repulsion_scale": float(config.repulsion_scale),
}
```

**Step 2: Same for drifting_stopgrad_loss**

```python
# OLD (line 77):
stats["loss"] = float(loss.item())

# NEW:
stats["loss"] = loss.detach()
```

**Step 3: Update training loop DriftStats to defer .item()**

```python
# OLD (lines 344-350):
Loss = torch.stack(GroupLosses).mean()
DriftStats = {
    'loss': float(Loss.item()),
    'mean_drift_norm': sum(GroupDriftNorms) / len(GroupDriftNorms),
    ...
}

# NEW: stats are already tensors, stack and mean them
Loss = torch.stack(GroupLosses).mean()
DriftStats = {
    'loss': Loss.detach(),
    'mean_drift_norm': torch.stack(GroupDriftNorms).mean(),
    'mean_drift_pos_norm': torch.stack(GroupDriftPosNorms).mean(),
    'mean_drift_neg_norm': torch.stack(GroupDriftNegNorms).mean(),
}
```

Also update lines 372-373 where DriftStats values are consumed:

```python
# OLD:
LastAlphaMean = float(AlphaGrouped.mean().item())
LastDriftNorm = float(DriftStats['mean_drift_norm'])

# NEW: defer .item() to tick boundary (lines 402-403)
LastAlphaMean = AlphaGrouped.mean().detach()
LastDriftNorm = DriftStats['mean_drift_norm']
```

And at the tick boundary (lines 402-403), convert to float:

```python
training_stats.report0('Progress/alpha_mean', float(LastAlphaMean))
training_stats.report0('Progress/drift_norm', float(LastDriftNorm))
```

**Step 4: Run parity tests**

```bash
cd /workspace/R3GAN && python -m pytest tests/test_drift_parity.py -x -v
```

**Step 5: Short training run, compare sec/kimg**

---

### Task 3: Fix _next_queue_batch Transfers

**Problem:** `.detach().clone()` is redundant on DataLoader output (already detached, already a new tensor). Missing `non_blocking=True` makes transfers synchronous.

**Files:**
- Modify: `training/drift_training_loop.py:591-595`

**Step 1: Simplify and add non_blocking**

```python
# OLD (lines 592-594):
Images = Images[:local_queue_push_batch].detach().clone().to(device).to(torch.float32) / 127.5 - 1
Labels = Labels[:local_queue_push_batch].detach().clone().to(device)

# NEW:
Images = Images[:local_queue_push_batch].to(device, non_blocking=True).to(torch.float32) / 127.5 - 1
Labels = Labels[:local_queue_push_batch].to(device, non_blocking=True)
```

**Step 2: Short training run, compare sec/kimg**

---

### Task 4: Queue Store on GPU

**Problem:** Queue stores on CPU, causing GPU→CPU on push() and CPU→GPU on sample(). Plus per-element Python loops with .item() in push/sample.

**Files:**
- Modify: `training/drift_queue.py` (push, sample_positive_grouped, sample_unconditional_grouped)
- Modify: `training/drift_training_loop.py:132` (store_device='cpu' → Device)
- Test: `tests/test_drift_training.py`

**Step 1: Change store_device to GPU in training loop**

```python
# OLD (line 132):
store_device='cpu',

# NEW:
store_device=Device,
```

**Step 2: Batch push() — eliminate per-element .item() loop**

```python
# OLD (lines 49-58):
StoredImages = images.detach().to(self.config.store_device)
StoredLabels = labels.detach().to('cpu').long()
for Index in range(StoredImages.shape[0]):
    Label = int(StoredLabels[Index].item())
    ...
    Sample = StoredImages[Index]
    self._class_queues[Label].append(Sample)
    self._global_queue.append(Sample)
    self._global_labels.append(Label)

# NEW:
StoredImages = images.detach().to(self.config.store_device)
StoredLabels = labels.detach().long()
LabelList = StoredLabels.tolist()  # single bulk sync
for Index in range(StoredImages.shape[0]):
    Label = LabelList[Index]
    if Label < 0 or Label >= self.config.num_classes:
        raise ValueError(f'label out of range: {Label}')
    Sample = StoredImages[Index]
    self._class_queues[Label].append(Sample)
    self._global_queue.append(Sample)
    self._global_labels.append(Label)
```

**Step 3: Batch sample_positive_grouped() — eliminate per-group .item()**

```python
# OLD (lines 66-79):
Outputs = []
for GroupIndex in range(class_ids.shape[0]):
    Label = int(class_ids[GroupIndex].item())
    ...

# NEW:
LabelList = class_ids.tolist()  # single bulk sync
Outputs = []
for GroupIndex in range(class_ids.shape[0]):
    Label = LabelList[GroupIndex]
    ...
```

Also remove `.to(device)` on the return since queue is already on GPU:
```python
# OLD:
return torch.stack(Outputs, dim=0).to(device)
# NEW:
Stacked = torch.stack(Outputs, dim=0)
return Stacked if Stacked.device == device else Stacked.to(device)
```

**Step 4: Update state_dict/load_state_dict for GPU-stored queue**

In `state_dict()`, the `.to('cpu')` on line 115 already handles serialization.
In `load_state_dict()`, the `push()` method will move to `store_device` automatically.

**Step 5: Run tests**

```bash
cd /workspace/R3GAN && python -m pytest tests/test_drift_training.py -x -v
```

**Step 6: Short training run, compare sec/kimg**

---

### Task 5: Increase DataLoader Workers

**Problem:** Default `num_workers=3` may bottleneck data supply for 2-GPU training with double-fetch pattern.

**Files:**
- Modify: `training/drift_training_loop.py:111` (add `persistent_workers`)
- Modify: `train.py:952` (default 3 → 8)

**Step 1: Change default workers**

```python
# OLD (train.py line 952):
default=3,

# NEW:
default=8,
```

**Step 2: Add persistent_workers to DataLoader in drift_training_loop.py**

After line 111, the DataLoader kwargs should include `persistent_workers=True` when num_workers > 0. This is best done in train.py where data_loader_kwargs is built:

In `train.py` around line 1215:
```python
c.data_loader_kwargs.num_workers = opts.workers
c.data_loader_kwargs.persistent_workers = opts.workers > 0
```

**Step 3: Short training run, compare sec/kimg**

---

## Testing Protocol

For each fix, run:

```bash
CUDA_VISIBLE_DEVICES=2,3 TOTAL_KIMG=20 GPUS=2 BATCH=64 \
    OUTDIR=outputs/drift/perf_test_fixN \
    bash scripts/train_drift_cifar10_pixel.sh
```

Record from last tick log line:
- `sec/kimg` — primary metric
- `gpumem` — should increase for Fix 4 (queue on GPU)
- GPU utilization via `nvidia-smi`

Compare each fix's `sec/kimg` against baseline.
