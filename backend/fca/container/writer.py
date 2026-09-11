"""FCA file container writer."""

from __future__ import annotations

import os
from fca.container.header import FCAHeader
from fca.container.blocks import EncodedBlockRecord


class FCAContainerWriter:
    """Writes FCA binary format container."""

    @staticmethod
    def write(
        file_path: str | os.PathLike,
        header: FCAHeader,
        blocks: list[EncodedBlockRecord],
    ) -> int:
        """Write header and blocks to file. Returns total bytes written."""
        file_path = str(file_path)
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        # Update block count in header to match exactly
        header.block_count = len(blocks)
        header_bytes = header.pack()

        with open(file_path, "wb") as f:
            f.write(header_bytes)
            bytes_written = len(header_bytes)
            for block in blocks:
                block_bytes = block.pack()
                f.write(block_bytes)
                bytes_written += len(block_bytes)

        return bytes_written
