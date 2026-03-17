# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Queue helpers for the prototype drift trainer."""

from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

#----------------------------------------------------------------------------


@dataclass(frozen=True)
class QueueConfig:
    num_classes: int = 10
    per_class_capacity: int = 256
    global_capacity: int = 4096
    store_device: str = 'cpu'
    strict_without_replacement: bool = False


class ClassConditionalSampleQueue:
    def __init__(self, config):
        if config.num_classes <= 0:
            raise ValueError('num_classes must be > 0')
        if config.per_class_capacity <= 0:
            raise ValueError('per_class_capacity must be > 0')
        if config.global_capacity <= 0:
            raise ValueError('global_capacity must be > 0')

        self.config = config
        self._class_queues = [deque(maxlen=config.per_class_capacity) for _idx in range(config.num_classes)]
        self._global_queue = deque(maxlen=config.global_capacity)
        self._global_labels = deque(maxlen=config.global_capacity)

    def push(self, images, labels):
        if images.ndim != 4:
            raise ValueError(f'images must be [B, C, H, W], got {tuple(images.shape)}')
        if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
            raise ValueError('labels must be [B] and aligned with images')

        StoredImages = images.detach().to(self.config.store_device)
        StoredLabels = labels.detach().to('cpu').long()
        for Index in range(StoredImages.shape[0]):
            Label = int(StoredLabels[Index].item())
            if Label < 0 or Label >= self.config.num_classes:
                raise ValueError(f'label out of range: {Label}')
            Sample = StoredImages[Index]
            self._class_queues[Label].append(Sample)
            self._global_queue.append(Sample)
            self._global_labels.append(Label)

    def sample_positive_grouped(self, class_ids, samples_per_group, device):
        if class_ids.ndim != 1:
            raise ValueError('class_ids must be [G]')
        if samples_per_group <= 0:
            raise ValueError('samples_per_group must be > 0')

        Outputs = []
        for GroupIndex in range(class_ids.shape[0]):
            Label = int(class_ids[GroupIndex].item())
            ClassQueue = self._class_queues[Label]
            if len(ClassQueue) == 0:
                raise RuntimeError(f'class queue {Label} is empty')
            Sampled = _sample_from_queue(
                ClassQueue,
                samples_per_group,
                strict_without_replacement=bool(self.config.strict_without_replacement),
                queue_name=f'class queue {Label}',
            )
            Outputs.append(torch.stack(Sampled, dim=0))
        return torch.stack(Outputs, dim=0).to(device)

    def sample_unconditional_grouped(self, groups, samples_per_group, device):
        if groups <= 0:
            raise ValueError('groups must be > 0')
        if samples_per_group <= 0:
            raise ValueError('samples_per_group must be > 0')
        if len(self._global_queue) == 0:
            raise RuntimeError('global queue is empty')

        Outputs = []
        for _idx in range(groups):
            Sampled = _sample_from_queue(
                self._global_queue,
                samples_per_group,
                strict_without_replacement=bool(self.config.strict_without_replacement),
                queue_name='global queue',
            )
            Outputs.append(torch.stack(Sampled, dim=0))
        return torch.stack(Outputs, dim=0).to(device)

    def class_count(self, label):
        if label < 0 or label >= self.config.num_classes:
            raise ValueError(f'label out of range: {label}')
        return len(self._class_queues[label])

    def class_counts(self):
        return [len(Queue) for Queue in self._class_queues]

    def global_count(self):
        return len(self._global_queue)

    def state_dict(self):
        GlobalImages = None
        GlobalLabels = None
        if len(self._global_queue) > 0:
            GlobalImages = torch.stack(list(self._global_queue), dim=0).to('cpu')
            GlobalLabels = torch.tensor(list(self._global_labels), dtype=torch.long)
        return {
            'version': 1,
            'config': {
                'num_classes': int(self.config.num_classes),
                'per_class_capacity': int(self.config.per_class_capacity),
                'global_capacity': int(self.config.global_capacity),
                'store_device': str(self.config.store_device),
                'strict_without_replacement': bool(self.config.strict_without_replacement),
            },
            'global_images': GlobalImages,
            'global_labels': GlobalLabels,
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            raise TypeError('queue state must be a dictionary')

        ConfigState = state.get('config', {})
        Expected = {
            'num_classes': int(self.config.num_classes),
            'per_class_capacity': int(self.config.per_class_capacity),
            'global_capacity': int(self.config.global_capacity),
            'store_device': str(self.config.store_device),
            'strict_without_replacement': bool(self.config.strict_without_replacement),
        }
        for Key, ExpectedValue in Expected.items():
            ActualValue = ConfigState.get(Key)
            if Key == 'strict_without_replacement' and ActualValue is None:
                ActualValue = False
            if ActualValue != ExpectedValue:
                raise ValueError(f'queue config mismatch for {Key}: expected {ExpectedValue}, found {ActualValue}')

        for Queue in self._class_queues:
            Queue.clear()
        self._global_queue.clear()
        self._global_labels.clear()

        GlobalImages = state.get('global_images')
        GlobalLabels = state.get('global_labels')
        if GlobalImages is None and GlobalLabels is None:
            return
        if not isinstance(GlobalImages, torch.Tensor) or not isinstance(GlobalLabels, torch.Tensor):
            raise TypeError('queue state must include tensor global_images and global_labels')
        self.push(GlobalImages, GlobalLabels.long())


#----------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupedSamplingConfig:
    positives_per_group: int
    unconditional_per_group: int


def sample_grouped_real_batches(
    *,
    queue: ClassConditionalSampleQueue,
    class_labels: torch.Tensor,
    config: GroupedSamplingConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    positives = queue.sample_positive_grouped(
        class_labels,
        config.positives_per_group,
        device,
    )
    unconditional = queue.sample_unconditional_grouped(
        class_labels.shape[0],
        config.unconditional_per_group,
        device,
    )
    return positives, unconditional


def ensure_class_coverage(queue, class_ids, refill_fn, required_count=1, max_attempts=128):
    if class_ids.ndim != 1:
        raise ValueError('class_ids must be [G]')
    if required_count <= 0:
        raise ValueError('required_count must be > 0')

    Attempts = 0
    NeededLabels = {int(Label.item()) for Label in class_ids.detach().to('cpu')}
    while any(queue.class_count(Label) < required_count for Label in NeededLabels):
        if Attempts >= max_attempts:
            raise RuntimeError('could not backfill enough samples to satisfy class coverage')
        Images, Labels = refill_fn()
        queue.push(Images, Labels)
        Attempts += 1
    return Attempts


def _sample_from_queue(queue, count, strict_without_replacement, queue_name):
    QueueLength = len(queue)
    if QueueLength == 0:
        raise RuntimeError('cannot sample from empty queue')
    if QueueLength >= count:
        Permutation = torch.randperm(QueueLength)[:count]
        return [queue[Index] for Index in Permutation.tolist()]
    if strict_without_replacement:
        raise RuntimeError(f'{queue_name} has {QueueLength} samples but {count} requested')
    Indices = torch.randint(0, QueueLength, [count])
    return [queue[Index] for Index in Indices.tolist()]
