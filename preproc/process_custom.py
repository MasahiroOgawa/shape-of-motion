import subprocess
from concurrent.futures import ProcessPoolExecutor

import tyro


def main(
    img_dirs: list[str],
    gpus: list[int],
    img_name: str = "images",
    mask_name: str = "masks",
    metric_depth_name: str = "unidepth_disp",
    intrins_name: str = "unidepth_intrins",
    mono_depth_model: str = "depth-anything",
    slam_name: str = "droid_recon",
    track_model: str = "bootstapir",
    tapir_torch: bool = True,
):
    if len(img_dirs) > 0 and img_name not in img_dirs[0]:
        raise ValueError(f"Expecting {img_name} in {img_dirs[0]}")

    mono_depth_name = mono_depth_model.replace("-", "_")
    futures = []
    with ProcessPoolExecutor(max_workers=len(gpus)) as exc:
        for i, img_dir in enumerate(img_dirs):
            gpu = gpus[i % len(gpus)]
            img_dir = img_dir.rstrip("/")
            futures.append(
                exc.submit(
                    process_sequence,
                    gpu,
                    img_dir,
                    img_dir.replace(img_name, mask_name),
                    img_dir.replace(img_name, metric_depth_name),
                    img_dir.replace(img_name, intrins_name),
                    img_dir.replace(img_name, mono_depth_name),
                    img_dir.replace(img_name, f"aligned_{mono_depth_name}"),
                    img_dir.replace(img_name, slam_name),
                    img_dir.replace(img_name, track_model),
                    mono_depth_model,
                    track_model,
                    tapir_torch,
                )
            )
    for f in futures:
        f.result()  # re-raise any exception from subprocess


def process_sequence(
    gpu: int,
    img_dir: str,
    mask_dir: str,
    metric_depth_dir: str,
    intrins_name: str,
    mono_depth_dir: str,
    aligned_depth_dir: str,
    slam_path: str,
    track_dir: str,
    depth_model: str = "depth-anything",
    track_model: str = "bootstapir",
    tapir_torch: bool = True,
):
    torch_lib = subprocess.check_output(
        ["python", "-c", "import torch; print(torch.utils.cmake_prefix_path.replace('/share/cmake', '/lib'))"],
        text=True,
    ).strip()
    dev_arg = (
        f"CUDA_VISIBLE_DEVICES={gpu} "
        f"LD_LIBRARY_PATH={torch_lib}:$LD_LIBRARY_PATH "
        f"PYTHONPATH=$(pwd)/UniDepth:$PYTHONPATH"
    )

    def run_step(cmd, name):
        print(f"\n[process] Running: {name}")
        print(cmd)
        ret = subprocess.call(cmd, shell=True, executable="/bin/bash")
        if ret != 0:
            raise RuntimeError(f"[process] FAILED ({ret}): {name}\n  cmd: {cmd}")

    run_step(
        f"{dev_arg} uv run python compute_metric_depth.py --img-dir {img_dir} "
        f"--depth-dir {metric_depth_dir} --intrins-file {intrins_name}.json",
        "metric depth (UniDepth)",
    )

    run_step(
        f"{dev_arg} uv run python compute_depth.py --img_dir {img_dir} "
        f"--out_raw_dir {mono_depth_dir} --out_aligned_dir {aligned_depth_dir} "
        f"--model {depth_model} --metric_dir {metric_depth_dir}",
        "mono depth (Depth Anything)",
    )

    run_step(
        f"{dev_arg} uv run python recon_with_depth.py --img_dir {img_dir} "
        f"--calib {intrins_name}.json --depth_dir {aligned_depth_dir} --out_path {slam_path}",
        "SLAM (DROID-SLAM)",
    )

    track_script = "compute_tracks_torch.py" if tapir_torch else "compute_tracks_jax.py"
    run_step(
        f"{dev_arg} uv run python {track_script} --image_dir {img_dir} "
        f"--mask_dir {mask_dir} --out_dir {track_dir} --model_type {track_model}",
        "tracking (BootsTAPIR)",
    )


if __name__ == "__main__":
    tyro.cli(main)
