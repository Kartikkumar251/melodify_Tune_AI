"""Entropy coding package."""

from __future__ import annotations

from fca.entropy.base import EntropyCoder
from fca.entropy.raw import RawEntropyCoder
from fca.entropy.rice import RiceCoder
from fca.entropy.golomb import GolombCoder
from fca.entropy.huffman import HuffmanCoder
from fca.entropy.range_coder import RangeCoder

_CODERS: dict[int, type[EntropyCoder]] = {
    RawEntropyCoder.CODER_ID: RawEntropyCoder,
    RiceCoder.CODER_ID: RiceCoder,
    HuffmanCoder.CODER_ID: HuffmanCoder,
    GolombCoder.CODER_ID: GolombCoder,
    RangeCoder.CODER_ID: RangeCoder,
}


def get_entropy_coder(coder_id: int) -> EntropyCoder:
    """Retrieve an instantiated entropy coder by ID."""
    if coder_id not in _CODERS:
        raise ValueError(f"Unknown entropy coder ID: {coder_id}")
    return _CODERS[coder_id]()


__all__ = [
    "EntropyCoder",
    "RawEntropyCoder",
    "RiceCoder",
    "GolombCoder",
    "HuffmanCoder",
    "RangeCoder",
    "get_entropy_coder",
]
