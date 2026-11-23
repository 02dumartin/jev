"""H.264 writer for Retail dashboard frames."""

from pathlib import Path


def h264_writer(path: Path, width: int, height: int, fps: int):
    """Generator that takes RGB frames and encodes H.264 / yuv420p with the ffmpeg bundled in imageio-ffmpeg."""
    import imageio_ffmpeg

    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        quality=7,
        macro_block_size=2,
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)  # start the generator
    return writer
