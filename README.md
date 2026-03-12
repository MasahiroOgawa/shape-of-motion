# Shape of Motion: 4D Reconstruction from a Single Video
**[Project Page](https://shape-of-motion.github.io/) | [Arxiv](https://arxiv.org/abs/2407.13764)**

[Qianqian Wang](https://qianqianwang68.github.io/)<sup>1,2</sup>*, [Vickie Ye](https://people.eecs.berkeley.edu/~vye/)<sup>1</sup>\*, [Hang Gao](https://hangg7.com/)<sup>1</sup>\*, [Weijia Zeng](https://fantasticoven2.github.io/)<sup>1</sup>\*, [Jake Austin](https://www.linkedin.com/in/jakeaustin4701)<sup>1</sup>, [Zhengqi Li](https://zhengqili.github.io/)<sup>2</sup>, [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/)<sup>1</sup>

<sup>1</sup>UC Berkeley   &nbsp;  <sup>2</sup>Google Research

\* Equal Contribution

ICCV 2025 (Highlight)

## *New
We have preprocessed nvidia dataset and custom dataset which can be found [here](https://drive.google.com/drive/folders/1xzn-Mu_jyr-JTsrERRU-Mh2hQ-NWdfv8). We used [MegaSaM](https://mega-sam.github.io/) to get cameras and depths for custom dataset.
### Training
To train nvidia dataset
```
python scripts/run_training.py \
  --work-dir <OUTPUT_DIR> \
  data:nvidia \
  --data.data-dir </path/to/data>
```

To train custom dataset
```
python scripts/run_training.py \
  --work-dir <OUTPUT_DIR> \
  data:custom \
  --data.data-dir </path/to/data>
```

### Train with 2D Gaussian Splatting
To get better scene geometry, we use 2D Gaussian Splatting:

```
python scripts/run_training.py \
  --work-dir <OUTPUT_DIR> \
  --use_2dgs
  data:custom \
  --data.data-dir </path/to/data>
```

## Installation

```bash
git clone --recurse-submodules https://github.com/vye16/shape-of-motion
cd shape-of-motion/

# Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv venv --python 3.10
uv sync

# gsplat requires torch at build time, install separately
uv pip install --no-build-isolation "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git"
```

## Download Preprocessing Checkpoints

Before running the pipeline on custom videos, download the required model checkpoints:

```bash
cd preproc
mkdir -p checkpoints/saves

# SAM (Segment Anything) - for interactive mask annotation
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# XMem - for mask propagation across frames
wget -P checkpoints/saves/ https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth

# DROID-SLAM - for camera pose estimation
uv pip install gdown
gdown -O checkpoints/ 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh

# BootsTAPIR - for 2D point tracking
wget -P checkpoints/ https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt

cd ..
```

## Usage

### End-to-End Pipeline (Video to 4D)

The easiest way to run the full pipeline from a single video:

```bash
uv run python scripts/run_4d.py --input my_video.mp4
```

This runs all steps automatically:
1. **Extract frames** from the video
2. **Create masks** — opens an interactive Gradio GUI where you click on foreground objects (the only manual step; follow the on-screen instructions)
3. **Preprocess** — depth estimation (UniDepth + DepthAnything), camera poses (DROID-SLAM), 2D tracking (BootsTAPIR)
4. **Train** the 4D Gaussian model
5. **Launch viewer** — interactive 4D viewer in the browser

Options:
```bash
uv run python scripts/run_4d.py --input my_video.mp4 --name MyScene --gpu 0 --fps 10 --port 8080

# Resume from a specific step (if prior steps are already done)
uv run python scripts/run_4d.py --input my_video.mp4 --skip-to preprocess
uv run python scripts/run_4d.py --input my_video.mp4 --skip-to train
uv run python scripts/run_4d.py --input my_video.mp4 --skip-to view
```

### Preprocessing

We depend on the third-party libraries in `preproc` to generate depth maps, object masks, camera estimates, and 2D tracks.
Please follow the guide in the [preprocessing README](./preproc/README.md).

### Interactive 4D Viewer

After training, launch an interactive viewer in the browser:

```bash
uv run python scripts/run_rendering.py --work-dir <OUTPUT_DIR> --port 8080
```

Then open `http://localhost:8080`. You can orbit the camera, scrub through time, toggle canonical view, and render tracks.

## Evaluation on iPhone Dataset
First, download our processed iPhone dataset from [this](https://drive.google.com/drive/folders/1xJaFS_3027crk7u36cue7BseAX80abRe?usp=sharing) link. To train on a sequence, e.g., *paper-windmill*, run:

```python
python scripts/run_training.py \
  --work-dir <OUTPUT_DIR> \
  --port <PORT> \
  data:iphone \
  --data.data-dir </path/to/paper-windmill/>
```

After optimization, the numerical result can be evaluated via:
```
PYTHONPATH='.' python scripts/evaluate_iphone.py \
  --data_dir </path/to/paper-windmill/> \
  --result_dir <OUTPUT_DIR> \
  --seq_names paper-windmill
```


## Citation
```
@inproceedings{som2024,
  title     = {Shape of Motion: 4D Reconstruction from a Single Video},
  author    = {Wang, Qianqian and Ye, Vickie and Gao, Hang and Zeng, Weijia and Austin, Jake and Li, Zhengqi and Kanazawa, Angjoo},
  booktitle   = {International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```
