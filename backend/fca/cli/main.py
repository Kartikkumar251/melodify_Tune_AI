"""FCA Command-Line Interface (CLI).

Provides commands:
  - fca encode input.wav output.fca [--mode lossless|perceptual] [--quality 1..10]
  - fca decode input.fca output.wav
  - fca inspect input.fca
  - fca verify input.fca
  - fca benchmark [directory]
"""

from __future__ import annotations

import os
import sys
import time
import click

from fca.io.wav_reader import read_wav
from fca.io.wav_writer import write_wav
from fca.encoder import FCAEncoder
from fca.decoder import FCADecoder
from fca.container.reader import FCAContainerReader
from fca.verify.hash import verify_pcm_equality, calculate_pcm_sha256


@click.group()
@click.version_option(version="0.1.0", prog_name="fca")
def cli() -> None:
    """FCA — Frequency Coded Audio Research Codec & Container."""


@cli.command("encode")
@click.argument("input_wav", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_fca", type=click.Path(dir_okay=False))
@click.option(
    "--mode",
    type=click.Choice(["lossless", "perceptual"], case_sensitive=False),
    default="lossless",
    show_default=True,
    help="Encoding mode: lossless (FCA-L) or perceptual (FCA-P)",
)
@click.option(
    "--quality",
    type=click.IntRange(1, 10),
    default=5,
    show_default=True,
    help="Perceptual quality level 1 (highest compression) to 10 (highest fidelity)",
)
@click.option(
    "--block-size",
    type=int,
    default=4096,
    show_default=True,
    help="Block size in sample frames",
)
def encode_cmd(input_wav: str, output_fca: str, mode: str, quality: int, block_size: int) -> None:
    """Encode a WAV audio file into an FCA container file (.fca)."""
    click.echo(f"[*] Reading WAV: {input_wav}")
    try:
        audio = read_wav(input_wav)
    except Exception as e:
        click.secho(f"Error reading WAV: {e}", fg="red", err=True)
        sys.exit(1)

    click.echo(
        f"    Format: {audio.sample_rate} Hz, {audio.channels} ch, {audio.bit_depth}-bit, "
        f"{audio.num_samples} samples ({audio.duration_seconds:.2f}s)"
    )
    click.echo(f"[*] Encoding to {output_fca} (Mode: {mode.upper()}, Quality: {quality})...")

    encoder = FCAEncoder(block_size=block_size)
    try:
        res = encoder.encode_file(audio, output_fca, mode=mode, quality=quality)
    except Exception as e:
        click.secho(f"Encoding failed: {e}", fg="red", err=True)
        sys.exit(1)

    click.secho(
        f"[OK] Encoding completed in {res.encode_time_seconds:.3f}s",
        fg="green",
    )
    click.echo(
        f"    PCM bytes: {res.input_bytes:,} B  -->  FCA bytes: {res.output_bytes:,} B "
        f"(Ratio: {res.compression_ratio:.2f}x, Saved: {res.space_saved_percent:.1f}%)"
    )
    click.echo(f"    PCM SHA-256: {res.sha256_pcm}")


@cli.command("decode")
@click.argument("input_fca", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_wav", type=click.Path(dir_okay=False))
def decode_cmd(input_fca: str, output_wav: str) -> None:
    """Decode an FCA file (.fca) back into a standard PCM WAV file."""
    click.echo(f"[*] Reading FCA file: {input_fca}")
    decoder = FCADecoder()
    try:
        res = decoder.decode_file(input_fca)
    except Exception as e:
        click.secho(f"Decoding failed: {e}", fg="red", err=True)
        sys.exit(1)

    click.echo(f"[*] Writing WAV: {output_wav}")
    try:
        write_wav(res.audio, output_wav)
    except Exception as e:
        click.secho(f"Error writing WAV: {e}", fg="red", err=True)
        sys.exit(1)

    click.secho(f"[OK] Decoding completed in {res.decode_time_seconds:.3f}s", fg="green")
    click.echo(
        f"    Decoded {res.total_samples} samples, {res.audio.channels} ch, {res.audio.bit_depth}-bit"
    )
    click.echo(f"    Decoded PCM SHA-256: {res.sha256_pcm}")


@cli.command("inspect")
@click.argument("input_fca", type=click.Path(exists=True, dir_okay=False))
def inspect_cmd(input_fca: str) -> None:
    """Inspect and display FCA container headers and block metadata."""
    click.echo(f"[*] Inspecting FCA container: {input_fca}")
    try:
        header, blocks = FCAContainerReader.read(input_fca)
    except Exception as e:
        click.secho(f"Failed to inspect FCA container: {e}", fg="red", err=True)
        sys.exit(1)

    mode_label = "FCA-L (Lossless)" if header.mode == 0 else "FCA-P (Perceptual)"
    click.echo("\n--- FCA Container Header ---")
    click.echo(f"Version:      v{header.version_major}.{header.version_minor}")
    click.echo(f"Mode:         {mode_label}")
    click.echo(f"Channels:     {header.channels}")
    click.echo(f"Bit Depth:    {header.bit_depth} bits")
    click.echo(f"Sample Rate:  {header.sample_rate} Hz")
    click.echo(f"Total Frames: {header.total_samples:,}")
    click.echo(f"Block Count:  {header.block_count}")
    if header.metadata:
        click.echo(f"Metadata:     {header.metadata}")

    click.echo(f"\n--- Blocks Summary ({len(blocks)} blocks) ---")
    transform_names = {0: "Direct/Identity", 1: "Integer Haar DWT", 2: "STFT"}
    ch_names = {0: "Independent", 1: "Mid/Side"}
    coder_names = {0: "Raw", 1: "Rice", 2: "Huffman", 3: "Golomb", 4: "Range"}

    for i, b in enumerate(blocks[:10]):  # Show first 10
        trans = transform_names.get(b.transform_id, f"ID-{b.transform_id}")
        ch = ch_names.get(b.channel_mode, f"Mode-{b.channel_mode}")
        coder = coder_names.get(b.entropy_coder_id, f"Coder-{b.entropy_coder_id}")
        click.echo(
            f"  Block #{b.block_index:03d}: samples={b.sample_count}, payload={len(b.payload):,}B, "
            f"trans={trans}, ch={ch}, coder={coder}, pred_order={b.prediction_order}"
        )
    if len(blocks) > 10:
        click.echo(f"  ... and {len(blocks) - 10} more blocks.")


@cli.command("verify")
@click.argument("input_fca", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--original-wav",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional original WAV file to verify bit-exact sample equality.",
)
def verify_cmd(input_fca: str, original_wav: str | None) -> None:
    """Verify integrity checksums and optional bit-exact sample equality."""
    click.echo(f"[*] Verifying FCA file: {input_fca}")
    try:
        header, blocks = FCAContainerReader.read(input_fca)
    except Exception as e:
        click.secho(f"[X] Integrity verification FAILED: {e}", fg="red", err=True)
        sys.exit(1)

    click.secho(f"[OK] Header CRC32 verified", fg="green")
    click.secho(f"[OK] All {len(blocks)} block CRC32 checksums verified", fg="green")

    if original_wav:
        click.echo(f"[*] Verifying sample equality against: {original_wav}")
        orig_audio = read_wav(original_wav)
        decoder = FCADecoder()
        dec_res = decoder.decode_file(input_fca)

        is_equal, msg = verify_pcm_equality(orig_audio, dec_res.audio)
        if is_equal:
            click.secho(f"[OK] BIT-EXACT MATCH! {msg}", fg="green", bold=True)
        else:
            click.secho(f"[X] Verification FAILED: {msg}", fg="red", bold=True)
            sys.exit(1)


@cli.command("benchmark")
@click.argument("corpus_dir", type=click.Path(exists=True, file_okay=False))
def benchmark_cmd(corpus_dir: str) -> None:
    """Run benchmark tests across all WAV files in a corpus directory."""
    click.echo(f"[*] Running FCA benchmark on corpus: {corpus_dir}\n")
    wav_files = [
        os.path.join(corpus_dir, f)
        for f in os.listdir(corpus_dir)
        if f.lower().endswith(".wav")
    ]

    if not wav_files:
        click.secho("No .wav files found in directory.", fg="yellow")
        return

    encoder = FCAEncoder()
    decoder = FCADecoder()

    click.echo(
        f"{'File':<25} | {'Original':<10} | {'FCA-L':<10} | {'Ratio':<7} | {'Saved%':<7} | {'Enc(ms)':<8} | {'Exact?':<6}"
    )
    click.echo("-" * 85)

    for wav_file in sorted(wav_files):
        fname = os.path.basename(wav_file)
        fca_tmp = wav_file + ".tmp.fca"
        try:
            audio = read_wav(wav_file)
            enc_res = encoder.encode_file(audio, fca_tmp, mode="lossless")
            dec_res = decoder.decode_file(fca_tmp)

            exact = audio.equals(dec_res.audio) and (audio.compute_sha256() == dec_res.sha256_pcm)
            exact_str = "YES" if exact else "NO"
            exact_fg = "green" if exact else "red"

            enc_ms = enc_res.encode_time_seconds * 1000.0
            click.echo(
                f"{fname[:25]:<25} | {enc_res.input_bytes:<10} | {enc_res.output_bytes:<10} | "
                f"{enc_res.compression_ratio:<7.2f} | {enc_res.space_saved_percent:<7.1f} | "
                f"{enc_ms:<8.1f} | ",
                nl=False,
            )
            click.secho(f"{exact_str:<6}", fg=exact_fg)
        finally:
            if os.path.exists(fca_tmp):
                try:
                    os.remove(fca_tmp)
                except OSError:
                    pass



@cli.command("web")
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind server")
@click.option("--port", default=8000, show_default=True, help="Port to bind server")
def web_cmd(host: str, port: int) -> None:
    """Launch the FCA Web Application & REST API interface."""
    import uvicorn
    click.echo(f"[*] Starting FCA Web Studio at http://{host}:{port}")
    uvicorn.run("fca.web.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    cli()

