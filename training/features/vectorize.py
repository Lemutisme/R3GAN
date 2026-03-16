from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureVectorizationConfig:
    include_per_location: bool = True
    include_global_stats: bool = True
    include_patch2_stats: bool = True
    include_patch4_stats: bool = True
    include_input_x2_mean: bool = False
    selected_stages: tuple[int, ...] | None = None
