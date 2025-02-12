from typing import Any, Dict, Optional
from concurrent.futures import Future

from torch.distributed.checkpoint._v2._checkpointer import (
    CheckpointContext,
    CheckpointingConfig,
    Checkpointer,
    RankInfo,
    StorageWriter,
    ManifestBuilder,
)
from torch.distributed.checkpoint.metadata import Metadata


class SyncCheckpointer(Checkpointer):

    def __init__(
        self,
        config: CheckpointingConfig,
        rank_info: RankInfo,
        storage_writer: StorageWriter,
        manifest_builder: Optional[ManifestBuilder] = None,
    ):
        self._config = config
        self._rank_info = rank_info
        self._storage_writer = storage_writer
        self._cached_manifest: Optional[Metadata] = None
        self.manifest_builder = manifest_builder

    def save(
        self,
        state_dict: Dict[str, Any],
        context: CheckpointContext,
        root_dir: str,
        use_cached_metadata: bool = False,
    ) -> Optional[tuple[Future[None], Future[None]]]:

        if self._config.save_manifest_with_checkpoint and self.manifest_builder is not None:
            if not use_cached_metadata or self._cached_manifest is None:
                manifest = self.manifest_builder.buid_manifest(
                    state_dict=state_dict,
                    context=context,
                )
                self._cached_manifest = manifest

        self._storage_writer.write_checkpoint(state_dict, self._cached_manifest, context, root_dir)
