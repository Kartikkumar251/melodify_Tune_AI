"""FCA Audio Encoder implementing FCA-L (lossless) and FCA-P (perceptual) pipelines."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
import numpy as np

from fca.pcm.samples import AudioData
from fca.blocks.block_manager import BlockManager, BlockData
from fca.channel.independent import IndependentChannelCoder
from fca.channel.mid_side import MidSideChannelCoder
from fca.transform.reversible import get_transform, IdentityTransform, IntegerLiftingDWT
from fca.prediction.fixed import FixedPredictor
from fca.entropy.rice import RiceCoder
from fca.entropy.raw import RawEntropyCoder
from fca.entropy.huffman import HuffmanCoder
from fca.container.header import FCAHeader
from fca.container.blocks import EncodedBlockRecord
from fca.container.writer import FCAContainerWriter
from fca.perceptual.quality import get_quality_profile, QualityProfile
from fca.transform.stft import STFTProcessor
from fca.perceptual.quantizer import PerceptualQuantizer


@dataclass
class EncodeResult:
    """Summary metrics of the encoding process."""

    input_samples: int
    input_bytes: int
    output_bytes: int
    compression_ratio: float
    space_saved_percent: float
    encode_time_seconds: float
    mode: str
    sha256_pcm: str


class FCAEncoder:
    """Encodes PCM audio into FCA container format (.fca)."""

    def __init__(self, block_size: int = 4096) -> None:
        self.block_size = block_size
        self.rice_coder = RiceCoder()
        self.raw_coder = RawEntropyCoder()
        self.huffman_coder = HuffmanCoder()

    def encode_file(
        self,
        audio: AudioData,
        output_path: str | os.PathLike,
        mode: str = "lossless",
        quality: int = 5,
        entropy_method: str = "auto",
    ) -> EncodeResult:
        """Encode AudioData and write directly to an .fca file."""
        t0 = time.perf_counter()
        is_lossless = (mode.lower() == "lossless")
        fca_mode = 0 if is_lossless else 1

        blocks = BlockManager.split(audio, block_size=self.block_size)
        encoded_blocks: list[EncodedBlockRecord] = []

        for block in blocks:
            if is_lossless:
                rec = self._encode_lossless_block(block, entropy_method=entropy_method)
            else:
                rec = self._encode_perceptual_block(block, quality=quality, sample_rate=audio.sample_rate)
            encoded_blocks.append(rec)

        sha256_hash = audio.compute_sha256()
        header = FCAHeader(
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            bit_depth=audio.bit_depth,
            total_samples=audio.num_samples,
            block_count=len(encoded_blocks),
            mode=fca_mode,
            metadata={"source_sha256": sha256_hash, "mode": mode, "quality": quality},
        )

        bytes_written = FCAContainerWriter.write(output_path, header, encoded_blocks)
        elapsed = time.perf_counter() - t0

        raw_bytes = len(audio.to_pcm_bytes())
        ratio = raw_bytes / float(max(bytes_written, 1))
        saving = max(0.0, (1.0 - (bytes_written / float(max(raw_bytes, 1)))) * 100.0)

        return EncodeResult(
            input_samples=audio.num_samples,
            input_bytes=raw_bytes,
            output_bytes=bytes_written,
            compression_ratio=round(ratio, 4),
            space_saved_percent=round(saving, 2),
            encode_time_seconds=round(elapsed, 4),
            mode=mode,
            sha256_pcm=sha256_hash,
        )

    def _encode_lossless_block(
        self,
        block: BlockData,
        entropy_method: str = "auto",
    ) -> EncodedBlockRecord:
        """Encode a single block for bit-exact reconstruction."""
        channels = block.samples.shape[1]
        best_payload: bytes | None = None
        best_record: EncodedBlockRecord | None = None
        min_payload_len = float("inf")

        # Candidates for channel modes
        ch_modes = [0]  # Independent
        if channels == 2:
            ch_modes.append(1)  # Mid/Side

        # Reversible transforms to evaluate: Identity (0)
        transform_ids = [0]

        for ch_mode in ch_modes:
            if ch_mode == 1:
                ch_channels = MidSideChannelCoder.encode(block.samples)
            else:
                ch_channels = IndependentChannelCoder.encode(block.samples)

            for trans_id in transform_ids:
                transform = get_transform(trans_id)

                # Process each channel through transform, prediction, and entropy coding
                block_payloads: list[bytes] = []
                pred_orders: list[int] = []
                coder_ids: list[int] = []
                aux_params: list[int] = []

                for ch_data in ch_channels:
                    coeffs = transform.forward(ch_data)

                    # Find optimal prediction order (0..4)
                    pred_order, residuals = FixedPredictor.find_best_order(coeffs)
                    pred_orders.append(pred_order)

                    # Entropy coding
                    if entropy_method == "rice" or entropy_method == "auto":
                        rice_payload, rice_meta = self.rice_coder.encode(residuals)
                        cand_payload = rice_payload
                        coder_id = RiceCoder.CODER_ID
                        aux = rice_meta.get("k", 0)
                    elif entropy_method == "huffman":
                        huff_payload, _ = self.huffman_coder.encode(residuals)
                        cand_payload = huff_payload
                        coder_id = HuffmanCoder.CODER_ID
                        aux = 0
                    else:
                        raw_payload, _ = self.raw_coder.encode(residuals)
                        cand_payload = raw_payload
                        coder_id = RawEntropyCoder.CODER_ID
                        aux = 0

                    # Check if raw is smaller than entropy coder
                    raw_payload, _ = self.raw_coder.encode(residuals)
                    if len(raw_payload) < len(cand_payload):
                        cand_payload = raw_payload
                        coder_id = RawEntropyCoder.CODER_ID
                        aux = 0

                    block_payloads.append(cand_payload)
                    coder_ids.append(coder_id)
                    aux_params.append(aux)

                # Combine channel payloads: store length of channel 0 payload if multi-channel
                if len(block_payloads) == 1:
                    combined_payload = block_payloads[0]
                else:
                    # 4 bytes for ch0 payload length, then ch0 payload, then ch1 payload
                    ch0_len = len(block_payloads[0])
                    combined_payload = np.uint32(ch0_len).tobytes() + block_payloads[0] + block_payloads[1]

                if len(combined_payload) < min_payload_len:
                    min_payload_len = len(combined_payload)
                    best_record = EncodedBlockRecord(
                        block_index=block.block_index,
                        sample_count=block.sample_count,
                        transform_id=trans_id,
                        channel_mode=ch_mode,
                        entropy_coder_id=coder_ids[0],
                        prediction_order=pred_orders[0],
                        aux_param=aux_params[0] if len(aux_params) == 1 else (pred_orders[1] << 16) | (coder_ids[1] << 8) | aux_params[1],
                        payload=combined_payload,
                    )

        assert best_record is not None
        return best_record

    def _encode_perceptual_block(
        self,
        block: BlockData,
        quality: int,
        sample_rate: int,
    ) -> EncodedBlockRecord:
        """Encode block with controlled perceptual loss using STFT and coefficient reduction."""
        profile = get_quality_profile(quality)
        stft = STFTProcessor(n_fft=profile.n_fft, hop_length=profile.hop_length)

        # For perceptual mode, quantize integer samples directly with quality step
        # and encode residuals efficiently
        q_step = max(1, int(round(profile.step_size * 100)))
        quantized_samples = np.round(block.samples / q_step).astype(np.int64) * q_step

        quant_block = BlockData(
            block_index=block.block_index,
            sample_start=block.sample_start,
            sample_count=block.sample_count,
            samples=quantized_samples,
        )
        rec = self._encode_lossless_block(quant_block, entropy_method="rice")
        # Overwrite aux_param with quality step for decoder reference
        rec.aux_param = quality
        return rec
