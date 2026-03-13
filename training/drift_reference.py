from __future__ import annotations

from pathlib import Path
import sys


def _ensure_reference_repo_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2] / "drift_models"
    if not repo_root.exists():
        raise ImportError(
            "drift_models reference repo not found at "
            f"{repo_root}. The drift parity trainer requires the sibling drift_models checkout."
        )
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


REFERENCE_REPO_ROOT = _ensure_reference_repo_on_path()

from drifting_models.data import (  # noqa: E402
    ClassConditionalSampleQueue,
    GroupedSamplingConfig,
    QueueConfig,
    RealBatchProvider,
    RealBatchProviderConfig,
    dataset_manifest_fingerprint,
    sample_grouped_real_batches,
)
from drifting_models.drift_field import (  # noqa: E402
    DriftFieldConfig,
    build_negative_log_weights,
    cfg_alpha_to_unconditional_weight,
)
from drifting_models.drift_loss import (  # noqa: E402
    DriftingLossConfig,
    FeatureDriftingConfig,
    drifting_stopgrad_loss,
    drifting_stopgrad_loss_multi_temperature,
    feature_space_drifting_loss,
)
from drifting_models.eval import (  # noqa: E402
    frechet_distance,
    gaussian_statistics,
    inception_score_from_logits,
)
from drifting_models.features import (  # noqa: E402
    FeatureVectorizationConfig,
    HookedFeatureAdapter,
    HookedFeatureAdapterConfig,
    LatentResNetMAE,
    LatentResNetMAEConfig,
    TinyFeatureEncoder,
    TinyFeatureEncoderConfig,
)
from drifting_models.models import DiTLikeConfig, DiTLikeGenerator  # noqa: E402
from drifting_models.sampling import postprocess_images, sample_pixel_generator  # noqa: E402
from drifting_models.train import GroupedDriftStepConfig, grouped_drift_training_step  # noqa: E402
from drifting_models.utils import (  # noqa: E402
    ModelEMA,
    codebase_fingerprint,
    environment_fingerprint,
    environment_snapshot,
    file_sha256,
    maybe_compile_callable,
    payload_sha256,
    save_training_checkpoint,
    load_training_checkpoint,
    write_json,
)
from drifting_models.utils.alpha import sample_alpha  # noqa: E402
from scripts import train_pixel as reference_train_pixel  # noqa: E402

build_feature_extractor = reference_train_pixel._build_feature_extractor
build_lr_scheduler = reference_train_pixel._build_lr_scheduler
attach_loss_scale_metrics = reference_train_pixel._attach_loss_scale_metrics
prime_queue = reference_train_pixel._prime_queue
ensure_queue_has_labels = reference_train_pixel._ensure_queue_has_labels
build_queue_warmup_report = reference_train_pixel._build_queue_warmup_report
build_real_provider_sanity_report = reference_train_pixel._build_real_provider_sanity_report
build_periodic_eval_state = reference_train_pixel._build_periodic_eval_state
run_periodic_eval = reference_train_pixel._run_periodic_eval
append_jsonl = reference_train_pixel._append_jsonl
metric_jsonl_entry = reference_train_pixel._metric_jsonl_entry
