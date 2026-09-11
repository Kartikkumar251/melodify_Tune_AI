"""FCA file container reader and validator."""

from __future__ import annotations

import os
from fca.container.header import FCAHeader, ContainerHeaderError
from fca.container.blocks import EncodedBlockRecord, BlockCorruptError


class FCAContainerReader:
    """Reads, parses, and validates FCA container files."""

    @staticmethod
    def read(file_path: str | os.PathLike) -> tuple[FCAHeader, list[EncodedBlockRecord]]:
        """Read and validate all headers and block records from an FCA file.

        Raises ContainerHeaderError on header corruption or BlockCorruptError on block corruption.
        """
        file_path = str(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FCA file not found: {file_path}")

        with open(file_path, "rb") as f:
            data = f.read()

        header, offset = FCAHeader.unpack(data)
        blocks: list[EncodedBlockRecord] = []

        while offset < len(data):
            block, offset = EncodedBlockRecord.unpack(data, offset)
            blocks.append(block)

        if len(blocks) != header.block_count:
            raise BlockCorruptError(
                f"Block count mismatch: header expects {header.block_count}, found {len(blocks)}"
            )

        return header, blocks
