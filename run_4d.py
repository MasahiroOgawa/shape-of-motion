"""
End-to-end 4D reconstruction from a single video.

Usage:
    uv run python run_4d.py --input my_video.mp4
    uv run python run_4d.py --input my_video.mp4 --name MyScene --gpu 0
    uv run python run_4d.py --input my_video.mp4 --skip-to train   # resume from training
    uv run python run_4d.py --input my_video.mp4 --skip-to view    # just launch viewer
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREPROC = ROOT / "preproc"


def run(cmd: str, cwd: str | None = None, check: bool = True):
    print(f"\n{'='*60}")
    print(f"[run_4d] {cmd}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True, executable="/bin/bash", cwd=cwd)
    if check and result.returncode != 0:
        print(f"\n[run_4d] ERROR: command failed with code {result.returncode}")
        sys.exit(1)


def step_extract_frames(video_path: Path, img_dir: Path, fps: int | None):
    """Extract frames from video using ffmpeg."""
    if img_dir.exists() and any(img_dir.iterdir()):
        print(f"[run_4d] Frames already exist at {img_dir}, skipping extraction.")
        return
    img_dir.mkdir(parents=True, exist_ok=True)
    fps_filter = f"-vf fps={fps}" if fps else ""
    run(f"ffmpeg -i {video_path} {fps_filter} -q:v 2 {img_dir}/%05d.png")


def step_create_masks(data_dir: Path, gpu: int):
    """Launch interactive mask annotation GUI. User must manually annotate."""
    mask_dir = data_dir / "masks"
    if mask_dir.exists() and any(mask_dir.iterdir()):
        print(f"[run_4d] Masks already exist at {mask_dir}, skipping.")
        return

    print("\n" + "=" * 60)
    print("[run_4d] MANUAL STEP: Mask annotation")
    print("=" * 60)
    print("A Gradio UI will open. You need to:")
    print("  1. Click on foreground objects (positive points)")
    print("  2. Right-click on background (negative points)")
    print("  3. Run tracker to propagate masks")
    print("  4. Save masks and close the browser tab")
    print("Press Ctrl+C in the terminal when done.")
    print("=" * 60 + "\n")

    run(
        f"CUDA_VISIBLE_DEVICES={gpu} python mask_app.py --root_dir {data_dir}",
        cwd=str(PREPROC),
        check=False,  # User exits with Ctrl+C
    )

    if not mask_dir.exists() or not any(mask_dir.iterdir()):
        print("[run_4d] ERROR: No masks found. Please re-run and annotate masks.")
        sys.exit(1)


def step_preprocess(data_dir: Path, gpu: int):
    """Run depth estimation, SLAM, and tracking."""
    img_dir = data_dir / "images"
    run(
        f"python process_custom.py --img-dirs {img_dir} --gpus {gpu}",
        cwd=str(PREPROC),
    )


def step_train(data_dir: Path, work_dir: Path, num_epochs: int, port: int | None):
    """Train the 4D Gaussian model."""
    port_arg = f"--port {port}" if port else ""
    run(
        f"uv run python run_training.py "
        f"--work-dir {work_dir} "
        f"--num-epochs {num_epochs} "
        f"{port_arg} "
        f"data:custom --data.data-dir {data_dir}",
        cwd=str(ROOT),
    )


def step_view(work_dir: Path, port: int):
    """Launch interactive 4D viewer."""
    print(f"\n[run_4d] Launching viewer at http://localhost:{port}")
    run(
        f"uv run python run_rendering.py --work-dir {work_dir} --port {port}",
        cwd=str(ROOT),
    )


def main():
    parser = argparse.ArgumentParser(description="End-to-end 4D reconstruction")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument(
        "--name",
        default=None,
        help="Scene name (default: video filename without extension)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id (default: 0)")
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Extract frames at this FPS (default: use all frames)",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=500, help="Training epochs (default: 500)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Viewer port (default: 8080)"
    )
    parser.add_argument(
        "--skip-to",
        choices=["masks", "preprocess", "train", "view"],
        default=None,
        help="Skip to a specific step (assumes prior steps are done)",
    )
    args = parser.parse_args()

    video_path = Path(args.input).resolve()
    if not video_path.exists():
        print(f"[run_4d] ERROR: Video not found: {video_path}")
        sys.exit(1)

    scene_name = args.name or video_path.stem
    data_dir = ROOT / "data" / "custom" / scene_name
    work_dir = ROOT / "work_dir" / scene_name

    print(f"[run_4d] Video:    {video_path}")
    print(f"[run_4d] Scene:    {scene_name}")
    print(f"[run_4d] Data dir: {data_dir}")
    print(f"[run_4d] Work dir: {work_dir}")
    print(f"[run_4d] GPU:      {args.gpu}")

    steps = ["frames", "masks", "preprocess", "train", "view"]
    skip_to = args.skip_to
    if skip_to:
        # Map skip_to to step index
        skip_map = {"masks": 1, "preprocess": 2, "train": 3, "view": 4}
        start = skip_map[skip_to]
    else:
        start = 0

    img_dir = data_dir / "images"

    if start <= 0:
        step_extract_frames(video_path, img_dir, args.fps)
    if start <= 1:
        step_create_masks(data_dir, args.gpu)
    if start <= 2:
        step_preprocess(data_dir, args.gpu)
    if start <= 3:
        step_train(data_dir, work_dir, args.num_epochs, args.port)
    if start <= 4:
        step_view(work_dir, args.port)


if __name__ == "__main__":
    main()
