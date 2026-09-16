"""Export per-frame rendered depth from a fitted scene, as .npy keyed by frame name.

Shape-of-Motion fits one video at a time, so it cannot be dropped into a harness that
feeds short clips: it must be fitted per source sequence and then queried at the frames
a consumer cares about. This writes that query result to disk so the consumer scores it
through its own metrics, rather than this repo growing a second evaluation protocol.

Output::

    <out>/<frame_stem>.npy     float32 (H, W) depth at the training camera
    <out>/meta.json            img_wh, scene scale, frame list, checkpoint step

SCALE: the dataset normalises the scene (``w2cs[:, :3, 3] /= scale``), so the depth here
is in scene-normalised units, NOT metres. ``scale`` is recorded in meta.json; any
consumer comparing against metric GT must either apply it or use a scale-invariant
metric such as median-aligned AbsRel.

    uv run python scripts/export_depth_npy.py --work-dir <fit dir> --out <dir>
"""

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import tyro
import yaml
from loguru import logger as guru

from flow3d.data import CustomDataConfig, get_train_val_datasets
from flow3d.scene_model import SceneModel

torch.set_float32_matmul_precision("high")


@dataclass
class ExportConfig:
    work_dir: str
    out: str


def main(cfg: ExportConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
    assert os.path.exists(ckpt_path), ckpt_path

    with open(f"{cfg.work_dir}/cfg.yaml") as f:
        train_cfg = yaml.safe_load(f)
    data_cfg = CustomDataConfig(**train_cfg["data"])
    # load_val=False: only the train cameras are needed, and building the val keypoint
    # loader pulls in annotations a generic sequence does not have.
    train_dataset, _, _, _ = get_train_val_datasets(data_cfg, load_val=False)
    ds = getattr(train_dataset, "dataset", train_dataset)

    ckpt = torch.load(ckpt_path, weights_only=False)
    model = SceneModel.init_from_state_dict(ckpt["model"]).to(device)
    model.use_2dgs = train_cfg.get("use_2dgs", False)

    w2cs, Ks = ds.get_w2cs().to(device), ds.get_Ks().to(device)
    H, W = ds.get_image(0).shape[:2]
    names = list(ds.frame_names)
    os.makedirs(cfg.out, exist_ok=True)

    with torch.no_grad():
        for t, name in enumerate(names):
            rendered = model.render(t, w2cs[t : t + 1], Ks[t : t + 1], (W, H),
                                    return_depth=True)
            d = rendered["depth"][0, ..., 0].float().cpu().numpy()
            np.save(os.path.join(cfg.out, f"{name}.npy"),
                    np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0))

    with open(os.path.join(cfg.out, "meta.json"), "w") as f:
        json.dump({"img_wh": [W, H], "frames": names,
                   "scene_scale": float(getattr(ds, "scale", 1.0)),
                   "global_step": int(ckpt.get("global_step", -1)),
                   "data": asdict(data_cfg)}, f, indent=1)
    guru.info(f"exported {len(names)} depth maps ({W}x{H}) -> {cfg.out}")


if __name__ == "__main__":
    main(tyro.cli(ExportConfig))
