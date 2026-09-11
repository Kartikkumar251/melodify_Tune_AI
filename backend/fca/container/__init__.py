"""Container package."""

from fca.container.header import FCAHeader, ContainerHeaderError, MAGIC_V1
from fca.container.blocks import EncodedBlockRecord, BlockCorruptError, BLOCK_SYNC_MARKER
from fca.container.writer import FCAContainerWriter
from fca.container.reader import FCAContainerReader

__all__ = [
    "FCAHeader",
    "ContainerHeaderError",
    "MAGIC_V1",
    "EncodedBlockRecord",
    "BlockCorruptError",
    "BLOCK_SYNC_MARKER",
    "FCAContainerWriter",
    "FCAContainerReader",
]
