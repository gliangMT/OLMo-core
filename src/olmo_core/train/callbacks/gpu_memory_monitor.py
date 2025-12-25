import logging
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class GPUMemoryMonitorCallback(Callback):
    """
    Adds metrics for GPU memory statistics.
    """

    priority: ClassVar[int] = -1
    device_id: Optional[int] = None
    _num_alloc_retries: int = 0

    @property
    def device(self) -> torch.device:
        return (
            torch.device("musa")
            if self.device_id is None
            else torch.device(f"musa:{self.device_id}")
        )

    @property
    def device_name(self) -> str:
        return torch.musa.get_device_name(self.device)

    @property
    def device_capacity(self) -> int:
        return torch.musa.get_device_properties(self.device).total_memory

    def pre_train(self):
        torch.musa.reset_peak_memory_stats()
        torch.musa.empty_cache()
        log.info(
            f"GPU capacity: {self.device_name} with {self._to_gib(self.device_capacity):.2f}GiB memory "
            f"of which {self._to_gib(torch.musa.memory_allocated()):.2f}GiB is currently allocated and "
            f"{self._to_gib(torch.musa.memory_reserved()):.2f}GiB is currently reserved."
        )

    def post_step(self):
        musa_info = torch.musa.memory_stats(self.device)

        max_active = musa_info["active_bytes.all.peak"]
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)
        self.trainer.record_metric("gpu_memory/GPU active mem (GiB)", max_active_gib)
        self.trainer.record_metric("gpu_memory/GPU active mem (%)", max_active_pct)

        max_reserved = musa_info["reserved_bytes.all.peak"]
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)
        self.trainer.record_metric("gpu_memory/GPU reserved mem (GiB)", max_reserved_gib)
        self.trainer.record_metric("gpu_memory/GPU reserved mem (%)", max_reserved_pct)

        num_retries = musa_info["num_alloc_retries"]
        if num_retries > self._num_alloc_retries:
            log.warning(f"{num_retries} MUSA memory allocation retries.")
            self._num_alloc_retries = num_retries

        num_ooms = musa_info["num_ooms"]
        if num_ooms > 0:
            log.warning(f"{num_ooms} MUSA OOM errors thrown.")

        torch.musa.reset_peak_memory_stats()

    def _to_pct(self, memory: float) -> float:
        return 100 * memory / self.device_capacity

    def _to_gib(self, memory_in_bytes: int) -> float:
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib
