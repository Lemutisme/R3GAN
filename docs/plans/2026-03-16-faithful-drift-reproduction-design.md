# Faithful drift_models Reproduction in R3GAN — Design Document

**Date:** 2026-03-16
**Scope:** Phases 1–5 — inline drift_models into R3GAN's drift branch as a self-contained implementation
**Approach:** Mirror drift_models file structure, flat in `training/`, clean break from sibling repo

---

## Goal

Make `R3GAN/tree/drift` a **self-contained, faithful reproduction** of drift_models' stable latent pipeline. No dependency on the sibling `drift_models` package.

**Definition of done:**
1. Kernel parity tests pass (toy.py three-way + drift_models function-level)
2. Faithful latent smoke test runs from scratch
3. Default drift lane walks the feature-space latent path
4. Fixed-batch parity test numerically aligns R3GAN with drift_models

---

## File Layout

### New / Rewritten

| File | Action | Source |
|------|--------|--------|
| `training/drift_field.py` | NEW | `drift_models/.../drift_field.py` |
| `training/drift_loss.py` | REWRITE | `drift_models/.../drift_loss.py` |
| `training/drift_grouped.py` | NEW | `drift_models/.../train/grouped.py` |
| `training/drift_stage2.py` | NEW | `drift_models/.../train/stage2.py` |
| `training/drift_queue.py` | REWRITE | `drift_models/.../data/queue.py` |
| `training/drift_training_loop.py` | REWRITE | Delegates to drift_stage2, supports raw + faithful lanes |
| `training/features/__init__.py` | NEW | Empty |
| `training/features/vectorize.py` | NEW | `drift_models/.../features/vectorize.py` |
| `training/features/extractors.py` | NEW | `drift_models/.../features/extractors.py` |
| `training/models/__init__.py` | NEW | Empty |
| `training/models/dit_like.py` | NEW | `drift_models/.../models/dit_like.py` |
| `training/networks.py` | UPDATE | Use inlined dit_like, remove drift_reference import |

### Deleted

| File | Reason |
|------|--------|
| `training/drift_reference.py` | Sibling repo wrapper — replaced by inlined code |
| `training/drift_research.py` | Delegates to sibling repo — replaced by drift_stage2 |
| `training/drift_diagnostics.py` | Depends on drift_reference — can be re-added later |

### New Tests

| File | Coverage |
|------|----------|
| `tests/test_drift_parity.py` | Three-way kernel parity, API-level parity with drift_models |
| `tests/test_drift_integration.py` | Toy training smoke, image-shaped smoke, checkpoint round-trip |

---

## Data Flow (Faithful Latent Lane)

```
drift_training_loop.py
  ├─ Init: DiTLikeDriftGenerator (models/dit_like.py via networks.py)
  ├─ Init: TinyFeatureEncoder (features/extractors.py)
  ├─ Init: ClassConditionalSampleQueue (drift_queue.py)
  └─ Per step:
       ├─ Push real batch → queue
       ├─ Sample: positives, unconditional from queue
       ├─ Sample: noise, class_labels, alpha
       └─► drift_stage2.grouped_drift_training_step()
             ├─ Generator forward (flattened grouped noise)
             ├─ Per group:
             │    ├─ feature_input_transform (optional)
             │    ├─ feature_extractor → extract_feature_maps()
             │    ├─ vectorize_feature_maps() → dict[str, [B,V,C]]
             │    └─ feature_space_drifting_loss()
             │         ├─ _normalize_features()
             │         ├─ _compute_weighted_drifts_slot_batched_multi_temperature()
             │         ├─ _normalize_drifts()
             │         └─ MSE(x, stopgrad(x + drift))
             ├─ Stack losses → mean → backward
             └─ Return stats dict
```

---

## Design Decisions

1. **Config naming matches drift_models exactly** — `DriftFieldConfig`, `DriftingLossConfig`, `FeatureDriftingConfig`, `DiTLikeConfig`, `FeatureVectorizationConfig`
2. **Frozen dataclasses for all configs** — `@dataclass(frozen=True)`
3. **New modules use snake_case** — only `networks.py` keeps R3GAN PascalCase
4. **Queue is representation-agnostic** — stores `[B,C,H,W]` whether pixel or latent; feature_input_transform applied at step time
5. **DiTLikeDriftGenerator wrapper stays in networks.py** — adapts dit_like.py's forward contract to R3GAN conventions
6. **Two lanes via config** — raw-debug (flattened pixels, no feature extractor) and faithful-latent (DiT + feature encoder + multi-temp)

---

## Testing Strategy

### Layer 1: Kernel Parity (toy.py as oracle)
- Fixed seed 2D inputs, verify `drift_field.compute_v` matches toy.py `compute_drift` within atol=1e-6

### Layer 2: API Parity (drift_models as oracle)
- Function-level tests for `cfg_alpha_to_unconditional_weight`, `build_negative_log_weights`, `compute_affinity_matrices`, `drifting_stopgrad_loss`, `drifting_stopgrad_loss_multi_temperature`, `feature_space_drifting_loss`

### Layer 3: Toy Training Smoke
- Train MLP on swiss roll using R3GAN's inlined drift functions, verify loss decreases

### Layer 4: Image Integration Smoke
- Tiny DiT + tiny feature encoder + tiny queue, 2–5 steps, verify finite loss + non-zero grads + checkpoint round-trip
