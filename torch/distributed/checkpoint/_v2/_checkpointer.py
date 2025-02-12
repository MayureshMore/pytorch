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


_STORAGE: TypeAlias = Union[torch.TypedStorage, torch.UntypedStorage]

@dataclass
class CheckpointingConfig:
    save_manifest_with_checkpoint: bool = True
    filter_replicated_tensors_on_save: bool = False
    use_barrier_for_save_completion: bool = True
    barrier_timeout_on_save: int = 3600

@dataclass
class RankInfo:
    global_rank: int
    global_world_size: int
    rank_rank: int
    role_world_size: int
    role_index: int
    role_replica_count: int  # to support PAFT with HSDP

@dataclass
class CheckpointContext:
    step: int
    extra_context: Dict[str, Any]


class Checkpointer(abc.ABC):
    """
    This is a checkpointing solution to store and load models with parallelism at scale.

    Note: If you working with single rank models and do not need asynchronous checkpointing, we recommend
    using `torch.save` and `torch.load` for its simplicity.

    This class provides extension points for users to customize individual components as they
    need.

    .. warning::
        This feature is experimental and subject to removal/change.

    """

    @abc.abstractmethod
    def save(
        self,
        state_dict: Dict[str, Any],
        context: CheckpointContext,
        root_dir: str,
        use_cached_metadata: bool = False,
    ) -> Optional[tuple[Future[None], Future[None]]]:
        """
        Save a checkpoint to a given path on storage. This optionally saves the metadata aggregated
        across all ranks to make resharding on load efficient and generic.

        Using module names as top level keys is optional but see the below example for typical usage
        with a state dict containing module names as keys and module state dicts as values.
            {
                "model": model_state_dict,
                "optimizer": optimizer_state_dict,
                "dataloader": dataloader_state_dict,
                "metrics": metrics_state_dict,
                "extra_state": extra_state_dict,
            }

        We expect most users to use a variant of async checkpointing implementation and so the API is
        designed to be easy to use for this case. A synchronous checkpointing implementation
        can still be implemented by on top of this API. EG: synchronous version can return a tuple of
        two futures which are set with a result in the save() method.

        Args:
            state_dict (Dict[str, Any]): The state_dict to save.
            context (CheckpointContext): The context to save the checkpoint.
            root_dir (str): The path to save the checkpoint.
            use_cached_metadata (bool): Whether to use cached metadata. (Default: ``False``). If
                use_cached_metadata is True, it will use the compute the metadata to save the checkpoint.
                Otherwise, it will compute the metadata and save the checkpoint. If the checkpoint is saved
                successfully, it will return a tuple of two futures. The first future is a future for the
                metadata and the second future is a future for the checkpoint.

        Returns:
            tuple[Future, Future]: A tuple of two futures. The first future can be awaited on for completion
            of staging (D2H) the state_dict and the second for completion of checkpoint.
        """
        pass

    @abc.abstractmethod
    def load_manifest(self, path: str) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def close(self):
        pass


class CheckpointLoader(abc.ABC):
    """
    This is a checkpointing solution to load models with parallelism at scale.

    Note: If you working with single rank models and do not need asynchronous checkpointing, we recommend
    using `torch.save` and `torch.load` for its simplicity.

    This class provides extension points for users to customize individual components as they
    need.

    .. warning::
        This feature is experimental and subject to removal/change.

    """
    
    @abc.abstractmethod
    def load(
        self,
        path: str,
        context: CheckpointContext,
        *,
        storage_location: Optional[Union[Callable[[str], _STORAGE], Dict[str, Any]]] = None,
        reshard_if_needed: bool = True,
    ) -> Dict[str, Any]:
        """ 
            Loads a checkpoint from a given path.

            When a storage_location is provided, the checkpoint loader will attempt to load 
            the checkpoint into the provided storage location. If storage_location = None, laoder
            will return a state_dict constructed on "cpu" device which can be used to load the 
            checkpoint into the module using module.load_state_dict().

            When reshard_if_needed = True, the checkpoint loader will look up the shards to load
            across files from all ranks and reshard the checkpoint to the current rank. Note 
            resharding currently only works for checkpoints saved with manifest.

                
        """
        pass
    
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





# A base class for storage backends
class Storage(abc.ABC):
    """
    Acts as an adaptor for storage backends and is also responsible for parallelizing
    writes/reads to storage backend if needed.
    """

    @abc.abstractmethod
    def ls(self, path: str) -> List[Path]:
        pass

    @abc.abstractmethod
    def mkdir(self, path: str, recursive: bool, exists_ok: bool):
        pass

    @abc.abstractmethod
    def rmdir(self, path: str):
        pass

    @abc.abstractmethod
    def write(self, path: str, data: io.BytesIO):
        pass

    @abc.abstractmethod
    def read(self, path: str) -> io.BytesIO:
        pass

    @abc.abstractmethod
    def delete_obj(self, path: str):
        pass


class CheckpointLayout(abc.ABC):
    """
        This class is responsible for deciding the layout of the checkpoint on storage.

        TODO: How do we ensure checkpoint loader uses the same layout impl to load? Can 
        we say this responsibility is on the user? Can write metadata about the options 
        we used to save the checkpoint and use that to validate if needed?

        should we write metadata at file level?
    """
    @abc.abstractmethod
    def get_metadata_path(
        self, config: CheckpointingConfig, rank_info: RankInfo, context: Any
    ) -> str:
        return "metadata.json"

    @abc.abstractmethod
    def get_file_mappings_for_write(
        self, rank_info: int, state_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Examples usecases:

            1. To save state_dict from one rank in one file and all ranks in a single directory.
                `return {f"checkpoint_{rank_info.global_rank}.pt": state_dict}`
            2. To save state_dict from one rank in one file and but use a separate directory
                for every 1000 ranks.
                `return {f"{rank_info.global_rank/1000}/checkpoint_{rank_info.global_rank}.pt": state_dict}`
            2. To save each module in a separate file.
                `
                return {
                    "model.pt": state_dict["model"],
                    "optimizer.pt": state_dict["optimizer"],
                    "dataloader.pt": state_dict["dataloader"],
                }
                `
            3. To save each param in a separate file.
                `
                return {
                    "model.weights.param1.pt": state_dict["model"]["weights"]["param1"],
                    "model.weights.param2.pt": state_dict["model"]["weights"]["param2"],
                }
                `
        """
        pass

    @abc.abstractmethod
    def get_all_file_mappings_to_read(
        self, rank_info: int
    ) -> List[str]:
        """
            Return all files to be read for this rank.
        """
        # return f"checkpoint_{rank_info.global_rank}.pt"
        pass


    @abc.abstractmethod
    def get_file_mappings_to_read(
        self, rank_info: int, fqns_to_load: List[str]
    ) -> Dict[str, List[str]]:
        """
        """
        pass


class SerializationFormat(abc.ABC):
    # Can this be a callable? do we need an interface for this?
    @abc.abstractmethod
    def serialize(self, obj: object, f: FileLike) -> None:
        pass


class TorchSerializationFormat(abc.ABC):
    @abc.abstractmethod
    def serialize(self, obj: object, f: FileLike) -> None:
        return torch.save(obj, f)

    @abc.abstractmethod
    def deserialize(self, f: FileLike) -> object:
        return torch.load(f)

class Barrier(abc.ABC):
    def __init__(self, world_size: int, timeout: int):
        self.world_size = world_size

    def wait(self, timeout):
        pass


class StorageWriter:
    def __init__(
        self,
        config: Any,
        rank_info: RankInfo,
        storage: Storage,
        checkpoint_layout: CheckpointLayout,
        serialization_format: SerializationFormat,
        barrier: Barrier,
    ):
        """
        Writes the state_dict to storage.

        Args:
            config (Any): The config to use for the checkpoint.
            rank_info (RankInfo): The rank info to use for the checkpoint.
            storage (Storage): The storage to use for the checkpoint.
            checkpoint_layout (CheckpointLayout): The layout to use for the checkpoint.
            serialization_format (SerializationFormat): The serialization format to use for the checkpoint.
        """

        self._config = config
        self._rank_info = rank_info
        self._storage = storage
        self._layout = checkpoint_layout
        self._serialization_format = serialization_format
        self._barrier = barrier

    def write_checkpoint(
        self,
        state_dict: Dict[str, Any],
        manifest: Optional[Manifest],
        context: Any,
        root_dir: str,
    ) -> str:
        """
        Writes the state_dict to storage.

        Args:
            state_dict (Dict[str, Any]): The state_dict to write.
            manifest (Optional[Manifest]): The manifest to write.
            context (Any): The context to write.
            path (str): The path to write the checkpoint to.

        Returns:
            str: The path to the checkpoint.
        """
        # naive example for now
        manifest_path = self._layout.get_manifest_path(
            self._config, self._rank_info, context
        )
        if self._config.save_manifest_with_checkpoint:
            assert manifest is not None
            self._storage.write(
                os.path.join(Path(root_dir) / manifest_path, "manifest.pt"),
                BytesIO(json.dumps(asdict(manifest)).encode("utf-8")),
            )

        file_paths = self._layout.get_file_paths(self._rank_info, state_dict)
        for file_path, obj in file_paths.items():
            file_path = os.path.join(dest_path, file_path)
            with storage.open(file_path, "wb") as f:
                self._serialization_format.serialize(obj, f)

        if self._config.use_barrier_for_save_completion:
            self._barrier.wait(self._config.barrier_timeout)

        return dest_path


class StorageReader:
    def __init__(
        self,
        config: Any,
        rank_info: RankInfo,
        storage: Storage,
        checkpoint_layout: CheckpointLayout,
        serialization_format: SerializationFormat,
    ):
        """
        Writes the state_dict to storage.

        Args:
            config (Any): The config to use for the checkpoint.
            rank_info (RankInfo): The rank info to use for the checkpoint.
            storage (Storage): The storage to use for the checkpoint.
            checkpoint_layout (CheckpointLayout): The layout to use for the checkpoint.
            serialization_format (SerializationFormat): The serialization format to use for the checkpoint.
        """

        self._config = config
        self._rank_info = rank_info
        self._storage = storage
        self._layout = checkpoint_layout
        self._serialization_format = serialization_format

    def read_checkpoint(
        self,
        state_dict: Dict[str, Any],
        manifest: Optional[Manifest],
        context: Any,
        root_dir: str,
    ) -> str:
        """
        Writes the state_dict to storage.

        Args:
            state_dict (Dict[str, Any]): The state_dict to write.
            manifest (Optional[Manifest]): The manifest to write.
            context (Any): The context to write.
            path (str): The path to write the checkpoint to.

        Returns:
            str: The path to the checkpoint.
        """
        # naive example for now
        manifest_path = self._layout.get_manifest_path(
            self._config, self._rank_info, context
        )
        if self._config.save_manifest_with_checkpoint:
            assert manifest is not None
            self._storage.write(
                os.path.join(Path(root_dir) / manifest_path, "manifest.pt"),
                BytesIO(json.dumps(asdict(manifest)).encode("utf-8")),
            )

        file_paths = self._layout.get_file_paths(self._rank_info, state_dict)
        for file_path, obj in file_paths.items():
            file_path = os.path.join(dest_path, file_path)
            with storage.open(file_path, "wb") as f:
                self._serialization_format.serialize(obj, f)

        if self._config.use_barrier_for_save_completion:
            self._barrier.wait(self._config.barrier_timeout)

        return dest_path


class CheckpointWorker:
    def __init__(self, parent: Connection):
        """
        Initialize the SubProcessWorker with a connection object.

        :param conn: A connection object for communication.
        """
        self._parent = parent

    def run(self):
        """Run the worker to process messages from the parent."""
        while True:
            message = self._parent.recv()
            if message == "exit":
                print("Exiting subprocess.")
                break
            elif message == "save":
                print(f"Subprocess received: {message}")
                response = f"Processed: {message}"
                self._parent.send(response)

class ManifestBuilder:
    """
    This class is responsible for building the manifest for a checkpoint. 
    
    TODO alernate plan: 
    We can also avoid this class by just providing a utility function to build the manifest from
    ShardedTensor and DTensor. If the user has a state_dict with custom impls of parallelisms, they
    can write their own logic. 
    
    TODO: is this a problem? Allowing the user to build the manifest or override the builder is 
    flexible but has downside of not being able to ensure that manifest is built as expected which 
    might affect resharding.

    TODO alernate plan: 
    The alternate plan is to do the collective on load_with_resharding where each rank
    reads its own manifest (or from data.pkl in file saved with torch.save) in file and 
    does the collective to build the global manifest. This is a bit more expensive if 
    load_with_resharding is used by default.
    """

    def __init__(self, config: CheckpointingConfig, rank_info: RankInfo):
        self._config = config
        self._rank_info = rank_info
        self._manifest: Manifest = Manifest(items={})

    def buid_manifest(
        self,
        state_dict: Dict[str, Any],
        context: CheckpointContext,
    ) -> Manifest:
        assert(torch.distributed.is_initialized())
        # TODO 

        # prepare global manifest
        # do torch.distributed.all_reduce()
        return Manifest({})
        