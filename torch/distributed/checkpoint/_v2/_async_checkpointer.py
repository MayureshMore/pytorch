import abc

import io
import json
from concurrent.futures import Future
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, TypeAlias

import torch
import torch.multiprocessing as mp
from torch.types import FileLike

from torch.distributed.checkpoint._v2._checkpointer import (
    Checkpointer,
    StorageWriter,
    StagingMethod,
    CheckpointContext,
    CheckpointingConfig,
    RankInfo,
)


class A0CStagingMethod(StagingMethod):
    def __init__(self, rank_info: RankInfo, config: CheckpointingConfig):
        self._rank_info = rank_info
        self._config = config

    def initiate_staging(
        self,
        state_dict: Dict[str, Any],
        context: CheckpointContext,
    ) -> Any:
        pass

    def wait_for_staging(self):
        pass

    def close(self) -> None:
        pass


class AsyncCheckpointer(Checkpointer):

    def __init__(
        self,
        config: CheckpointingConfig,
        rank_info: RankInfo,
        staging_method: StagingMethod,
        storage_writer: StorageWriter,
    ):
        self._config = config
        self._rank_info = rank_info
        self._staging_method = staging_method
        self._storage_writer = storage_writer
        

    def save(
        self,
        state_dict: Dict[str, Any],
        context: CheckpointContext,
        root_dir: str,
        use_cached_metadata: bool = False,
    ) -> Optional[tuple[Future[None], Future[None]]]:

#         if not self.staging_method.init_in_thread:
#             self._invalidate_checkpoint()
#             latency_tracker.track_step("invalidate_checkpoint")
#             try:
#                 staged_state = self.staging_method.initiate_staging(
#                     checkpoint=checkpoint, use_shared_memory=self._use_shared_memory()
#                 )
#             finally:
#                 self.staging_method._release_staging_write_lock()
#             latency_tracker.track_step("initiate_staging")
#         else:
#             staged_state = checkpoint  # initialization done in thread
#         self.current_checkpoint_type = checkpoint_type
#         self.current_future: Future = self.executor.submit(
#             self._checkpointing_thread,
#             staged_state=staged_state,
#             config=self.config,
#             step=checkpoint.step,
#             dataloader_step=checkpoint.dataloader_step,
#             latency_tracker=OdsLatencyTracker(
#                 "train.checkpoint_write.execute.subprocess_comm"
#             ),
#             checkpoint_type=checkpoint_type,
#             log_ctx=structured_logger_context.export_context(),
#         )

#         latency_tracker.track_step("executor_submit")
#         logger.info(
#             f"Done: launch async checkpointing step {checkpoint.step}, dataloader_step={checkpoint.dataloader_step}",
#             extra=event_extra(EventType.CHECKPOINT_END, step=checkpoint.step),
#         )
#         latency_tracker.track_e2e("e2e")

# def _checkpointing_thread(
#         self,
#         staged_state: Any,
#         config: CheckpointingConfig,
#         step: int,
#         dataloader_step: int,
#         latency_tracker: OdsLatencyTracker,
#         checkpoint_type: CheckpointType = CheckpointType.PERSISTENT,
#         log_ctx: Any = None,
#     ):
#         if torch.cuda.is_available():
#             torch.cuda.set_device(get_local_rank())

#         logger.info(f"Checkpointing communication thread started for step {step}")
#         set_thread_name_safe(f"ckpt-comm-{config.rank_info.global_rank}")

#         # initiate staging here
#         if self.staging_method.init_in_thread:
#             # mark in-memory checkpoint as invalid to avoid race conditions
#             self._invalidate_checkpoint()
#             latency_tracker.track_step("invalidate_checkpoint")
#             try:
#                 staged_state = self.staging_method.initiate_staging(
#                     checkpoint=staged_state,
#                     use_shared_memory=True,
#                 )
#             finally:
#                 # lock should have been acquired in save_latest before launching the
#                 # checkpoint thread to avoid race conditions
#                 self.staging_method._release_staging_write_lock()

#         # else: this is done in save_latest

#         with self.staging_method.read_lock():
#             self.staging_method._assert_correct_step(step)

#             latency_tracker.track_step("wait_for_tensor_copy")
#             logger.info(
#                 "Checkpoint staging copy done.",
#                 extra=event_extra(EventType.CHECKPOINT_STAGING_END, step=step),
#             )

#             # model/optimized state will be passed as shared memory handles:
#             self.tx.put(
#                 {
#                     "op": "start-checkpointing",
#                     "staged_state": staged_state,
#                     "step": step,
#                     "dataloader_step": dataloader_step,
#                     "checkpoint_type": checkpoint_type,
#                     "log_ctx": log_ctx,
#                 }
#             )
#             latency_tracker.track_step("put")
#             msg = self.rx.get()
#             latency_tracker.track_e2e("e2e")
#             if msg != f"checkpoint-completed: #{step}":
#                 raise RuntimeError(f"Checkpointing failed for step {step}: {msg}")

#             logger.info(f"Checkpointing communication thread finished for step {step}")
