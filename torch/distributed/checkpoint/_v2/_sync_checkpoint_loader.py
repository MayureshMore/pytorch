import abc
from typing import Callable, Optional, Union, Dict, Any, List

from torch.distributed.checkpoint._v2._checkpointer import (
    CheckpointContext,
    CheckpointingConfig,
    RankInfo,
    Storage,
    CheckpointLayout,
    SerializationFormat,
    _STORAGE,
)

class SyncCheckpointLoader(abc.ABC):
    """
        Example for a sync checkpoint loader.
    """
    def __init__(
        self,
        config: CheckpointingConfig,
        rank_info: RankInfo,
        storage: Storage,
        checkpoint_layout: CheckpointLayout,
        serialization_format: SerializationFormat,
    ):
        self._config = config
        self._rank_info = rank_info
        self._storage = storage
        self._layout = checkpoint_layout
        self._serialization_format = serialization_format
    
    def load(
        self,
        path: str,
        context: CheckpointContext,
        *,
        storage_location: Optional[Union[Callable[[str], _STORAGE], Dict[str, Any]]] = None,
        reshard_if_needed: bool = True,
    ) -> Dict[str, Any]:

        return {}
    
    @abc.abstractmethod
    def load_fqns(
        self,
        path: str,
        context: CheckpointContext,
        fqns_to_load: List[str],
        storage_location: Optional[Union[Callable[[str], _STORAGE], Dict[str, Any]]] = None,
        reshard_if_needed: bool = True,
    ) -> Dict[str, Any]:
        """ 
            returns a dict of {fqn: obj} for the fqns specified in fqns_to_load.

            see load() for details on storage_location and reshard_if_needed.
        """
        pass
