from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch


@dataclass
class TensorMetadata:
    """
        Dataclass which holds information about a tensor.
    """
    dtype: torch.dtype
    device: torch.device
    sizes: List[int]
    shard_offsets: List[int]
    shard_lengths: List[int]


@dataclass
class Param:
    fqn: str
    type_name: str  # type(obj).__name__
    tensor_metadata: Optional[TensorMetadata] = (
        None  # this is populated only for tensors
    )


@dataclass
class Metadata:
    """
        Manifest for a checkpoint with FQNs.
    """
    manifest: Dict[str, List[Param]]

    dcp_checkpointer_version: int = 0
