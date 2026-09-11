"""FCA Audio Decoder implementing bit-exact FCA-L and FCA-P reconstruction."""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass
import numpy as np

from fca.pcm.samples import AudioData
from fca.blocks.block_manager import BlockManager, BlockData
from fca.channel.independent import IndependentChannelCoder
from fca.channel.mid_side import MidSideChannelCoder
from fca.transform.reversible import get_transform
from fca.prediction.fixed import FixedPredictor
from fca.entropy import get_entropy_coder
from fca.container.reader import FCAContainerReader
from fca.container.blocks import EncodedBlockRecord
from fca.container.header import FCAHeader
from fca.verify.hash import calculate_pcm_sha256


@dataclass
class DecodeResult:
    """Summary metrics of the decoding process."""

    total_samples: int
    output_bytes: int
    decode_time_seconds: float
    mode: str
    sha256_pcm: str
    audio: AudioData


class FCADecoder:
    """Decodes FCA container format (.fca) back to PCM AudioData."""

    def decode_file(self, fca_path: str | os.PathLike) -> DecodeResult:
        """Read .fca file and decode all blocks into an AudioData instance."""
        t0 = time.perf_counter()
        header, encoded_blocks = FCAContainerReader.read(fca_path)

        target_dtype = np.int16 if header.bit_depth in (8, 16) else np.int32
        decoded_blocks: list[BlockData] = []
        sample_offset = 0

        for block in encoded_blocks:
            block_data = self._decode_block(block, header, target_dtype, sample_offset)
            decoded_blocks.append(block_data)
            sample_offset += block.sample_count

        audio = BlockManager.assemble(
            blocks=decoded_blocks,
            sample_rate=header.sample_rate,
            channels=header.channels,
            bit_depth=header.bit_depth,
        )

        # Slice to exact total_samples if any slight padding occurred
        if audio.num_samples > header.total_samples:
            audio.samples = audio.samples[: header.total_samples]

        elapsed = time.perf_counter() - t0
        sha256_pcm = calculate_pcm_sha256(audio)
        mode_str = "lossless" if header.mode == 0 else "perceptual"

        return DecodeResult(
            total_samples=audio.num_samples,
            output_bytes=len(audio.to_pcm_bytes()),
            decode_time_seconds=round(elapsed, 4),
            mode=mode_str,
            sha256_pcm=sha256_pcm,
            audio=audio,
        )

    def _decode_block(
        self,
        block: EncodedBlockRecord,
        header: FCAHeader,
        target_dtype: np.dtype,
        sample_offset: int,
    ) -> BlockData:
        """Decode a single EncodedBlockRecord into a BlockData object."""
        channels = header.channels
        transform = get_transform(block.transform_id)

        if channels == 1:
            coder = get_entropy_coder(block.entropy_coder_id)
            residuals = coder.decode(block.payload, count=block.sample_count)
            coeffs = FixedPredictor.restore_samples(residuals, order=block.prediction_order)
            channel_samples = transform.inverse(coeffs)
            samples = channel_samples.reshape(-1, 1).astype(target_dtype)
        elif channels == 2:
            # Multi-channel payload parsing
            ch0_len = struct.unpack("<I", block.payload[:4])[0]
            ch0_payload = block.payload[4 : 4 + ch0_len]
            ch1_payload = block.payload[4 + ch0_len :]

            coder_id0 = block.entropy_coder_id
            pred_order0 = block.prediction_order

            pred_order1 = (block.aux_param >> 16) & 0xFF
            coder_id1 = (block.aux_param >> 8) & 0xFF

            coder0 = get_entropy_coder(coder_id0)
            coder1 = get_entropy_coder(coder_id1)

            res0 = coder0.decode(ch0_payload, count=block.sample_count)
            res1 = coder1.decode(ch1_payload, count=block.sample_count)

            coeffs0 = FixedPredictor.restore_samples(res0, order=pred_order0)
            coeffs1 = FixedPredictor.restore_samples(res1, order=pred_order1)

            ch0_samples = transform.inverse(coeffs0)
            ch1_samples = transform.inverse(coeffs1)

            if block.channel_mode == 1:  # Mid/Side
                samples = MidSideChannelCoder.decode([ch0_samples, ch1_samples], target_dtype=target_dtype)
            else:
                interleaved = IndependentChannelCoder.decode([ch0_samples, ch1_samples])
                samples = interleaved.astype(target_dtype)
        else:
            raise NotImplementedError(f"Channel counts > 2 ({channels}) not yet implemented.")

        return BlockData(
            block_index=block.block_index,
            sample_start=sample_offset,
            sample_count=block.sample_count,
            samples=samples,
        )
