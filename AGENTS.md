# AGENTS.md — R3GAN Coding Agent Guide

## Project Overview

Official PyTorch implementation of "The GAN is dead; long live the GAN! A Modern Baseline GAN"
(NeurIPS 2024). Proposes a regularized relativistic GAN loss (R3GAN) with convergence guarantees,
replacing ad-hoc GAN training tricks. Forked from NVIDIA's StyleGAN3 codebase.

**Languages:** Python, CUDA/C++ (custom ops), Bash (build scripts)
**Key dependencies:** PyTorch, NumPy, SciPy, Pillow, Click, psutil, tqdm

---

## Build & Run Commands

### Training

```bash
# CIFAR-10 example
python train.py --outdir=runs --data=datasets/cifar10.zip --gpus=1 --batch=64 --mirror=1
```

### Pre-build CUDA extensions (optional, speeds up first run)

```bash
bash scripts/build_custom_ops.sh
```

### Generate images from a trained model

```bash
python gen_images.py --outdir=out --seeds=0-35 --network=<path-to-pkl>
```

### Compute quality metrics

```bash
python calc_metrics.py --metrics=fid50k_full --data=datasets/cifar10.zip --network=<path-to-pkl>
```

### Tests

Tests use Python's built-in `unittest` framework. There is no pytest config.

```bash
# Run all tests
python -m unittest discover -s tests -v

# Run a single test file
python -m unittest tests.test_adv_losses -v

# Run a single test method
python -m unittest tests.test_adv_losses.TestAdversarialLosses.test_softmargin_margin_zero_matches_rpgan -v

# Alternative: run test file directly
python tests/test_adv_losses.py
```

There is no CI pipeline, no linter config, no type-checker config, and no pre-commit hooks.

---

## Repository Structure

```
R3GAN/                  # Core library (PascalCase filenames)
  Trainer.py            #   AdversarialTraining class (loss + gradient accumulation)
  Networks.py           #   Generator & Discriminator (ResNeXt-style)
  FusedOperators.py     #   BiasedActivation (CUDA fused + fallback)
  Resamplers.py         #   Up/Down samplers (CUDA fused + fallback)
training/               # Training infrastructure (adapted from StyleGAN3)
  training_loop.py      #   Main training loop with cosine-decay schedulers
  loss.py               #   R3GANLoss wrapper (adds ranking losses)
  networks.py           #   Thin wrappers adapting R3GAN.Networks to training loop
  augment.py            #   AugmentPipe (StyleGAN2-ADA augmentation)
  dataset.py            #   Dataset & ImageFolderDataset classes
metrics/                # Quality metrics (FID, KID, precision/recall, IS)
dnnlib/                 # NVIDIA deep learning utilities (EasyDict, Logger, etc.)
torch_utils/            # PyTorch utilities (custom ops, persistence, stats)
  ops/                  #   Fused CUDA operators (bias_act, upfirdn2d, conv2d_gradfix)
scripts/                # Shell scripts (build_custom_ops.sh)
tests/                  # Unit tests (unittest framework)
train.py                # CLI entry point for training
gen_images.py           # CLI entry point for image generation
calc_metrics.py         # CLI entry point for metric computation
dataset_tool.py         # CLI entry point for dataset preparation
legacy.py               # Convert legacy TF pickles to PyTorch
```

---

## Code Style Guidelines

### Two Style Families

This codebase has **two distinct naming conventions** that MUST be followed based on which
file you are editing:

**R3GAN module files** (`R3GAN/*.py`, `training/networks.py`):

- Functions, methods, parameters, local variables: **PascalCase**
- Example: `AccumulateGeneratorGradients(self, Noise, RealSamples, ...)`
- Instance attributes: **PascalCase** — `self.Generator`, `self.LinearLayer`, `self.DataType`
- Module filenames: **PascalCase** — `Trainer.py`, `Networks.py`, `Resamplers.py`

**NVIDIA-derived files** (`training/training_loop.py`, `training/loss.py`, `dnnlib/`, `torch_utils/`, `train.py`):

- Functions, methods, parameters, local variables: **snake_case**
- Example: `cosine_decay_with_warmup(cur_nimg, base_value, ...)`
- Instance attributes: **snake_case** — `self.dataset`, `self.rank`
- Module filenames: **snake_case** — `training_loop.py`, `metric_main.py`

**Both families share:**

- Classes: **PascalCase** — `AdversarialTraining`, `EasyDict`, `InfiniteSampler`
- Private methods/variables: underscore prefix with **snake_case** — `_as_vector`, `_constant_cache`
- Constants at module level: **PascalCase** in R3GAN code — `WidthPerStage`, `NoiseDimension`

### Imports

- No strict grouping enforced; stdlib, third-party, and local imports are loosely ordered
- Standard library first, then third-party (`torch`, `numpy`), then local (`dnnlib`, `training`)
- Use `import module` for top-level packages, `from module import symbol` for specific items
- Optional dependencies use try/except guards:
  ```python
  try:
      import pyspng
  except ImportError:
      pyspng = None
  ```

### Formatting

- **Indentation:** 4 spaces, no tabs
- **Line length:** No strict limit; lines commonly exceed 120 chars for long signatures
- **String quotes:** Single quotes preferred; double quotes acceptable
- **String formatting:** f-strings preferred in new code; `.format()` exists in legacy code
- **Section dividers:** NVIDIA-derived files use `#----------------------------------------------------------------------------`
  between top-level definitions. Follow this pattern when editing those files.
- **Column alignment:** Used in Click decorators and `training_loop` parameter defaults:
  ```python
  run_dir                 = '.',      # Output directory.
  training_set_kwargs     = {},       # Options for training set.
  ```
- **Trailing commas:** Used inconsistently; include them in multi-line function calls

### Type Annotations

- Minimal; only `dnnlib/util.py` and some functions in `training/loss.py` use annotations
- `from typing import Any, List, Tuple, Union` when annotations are used
- No inline variable annotations anywhere
- When editing files that already have annotations, continue the pattern

### Docstrings

- Single-line, triple double-quote format: `"""Brief description."""`
- NVIDIA-derived files have module-level docstrings (e.g., `"""Loss functions."""`)
- R3GAN module files have **no docstrings** — do not add them unless asked
- `torch_utils/misc.py` uses block comments above functions instead of docstrings

### Error Handling

- `raise ValueError(f'...')` for invalid arguments in R3GAN module code
- `raise click.ClickException(...)` for CLI validation errors in `train.py`
- `assert` statements for preconditions in `torch_utils/` and `dnnlib/`
- No custom exception classes; use built-in exceptions
- Chain exceptions with `from err` when re-raising

### Class Patterns

- Neural network classes inherit from `nn.Module`
- Use `super(ClassName, self).__init__()` (explicit class name, not bare `super()`)
- Non-module classes (e.g., `AdversarialTraining`, `R3GANLoss`) have no base class
- `@staticmethod` used for utility methods in `Trainer.py`
- Factory pattern via `dnnlib.util.construct_class_by_name(**kwargs)` with string class names
- `hasattr` checks for optional layers: `x = self.LinearLayer(x) if hasattr(self, 'LinearLayer') else x`

#### Other Conventions

- `dnnlib.EasyDict` used as configuration container throughout training code
- Method chaining on PyTorch modules: `G.train().requires_grad_(False).to(device)`
- Detach pattern for returning losses: `return [x.detach() for x in [Loss1, Loss2]]`
- Lambda defaults for optional preprocessors: `Preprocessor=lambda x: x`
- `torch.no_grad()` context for inference during training
- `torch.autograd.profiler.record_function` for profiling sections
- Copyright headers on NVIDIA-derived files only; do not add them to R3GAN module files
- `if __name__ == "__main__":` guard on all CLI entry points and test files
- `# pylint: disable=no-value-for-parameter` on Click command invocations
- 回答都要以 喵～ 结尾
