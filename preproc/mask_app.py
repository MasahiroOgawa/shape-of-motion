import colorsys
import datetime
import os
import subprocess

import cv2
import gradio as gr
import imageio.v2 as iio
import numpy as np
import torch
from loguru import logger as guru
from mask_utils import init_sam_model, init_tracker, track_masks


STEP_INSTRUCTIONS = {
    "start": (
        "== STEP 1 / 7 == Select images.\n"
        "If images are already loaded below, skip to Step 2.\n"
        "Otherwise, select a sequence from 'Image directories' dropdown (middle column),\n"
        "or select a video from 'Video files' dropdown (left column) and click 'Extract frames'."
    ),
    "loaded": (
        "== STEP 2 / 7 == Choose a reference frame, then load SAM.\n"
        "1. Use the 'Frame index' slider to pick a frame where the foreground object is clearly visible.\n"
        "2. Click >>> 'Get SAM features' <<< (this loads the segmentation model; wait for Step 3 to appear).\n"
        "NOTE: You MUST click 'Get SAM features' before clicking on the image!"
    ),
    "sam_ready": (
        "== STEP 3 / 7 == SAM is ready! Now click on the foreground object.\n"
        "Click directly on the object in the 'Input Frame' image below.\n"
        "  - Green dots = include (positive, default mode)\n"
        "  - Red dots = exclude (click 'Toggle negative' first)\n"
        "A mask preview appears in 'Current selection' (right column).\n"
        "Click 'Clear points' to retry. For multiple objects, click 'Add new mask' between them.\n"
        "When the mask looks good, click >>> 'Submit mask for tracking' <<<."
    ),
    "tracked": (
        "== STEP 6 / 7 == Review the tracked video.\n"
        "Check the 'Masked video' panel. If it looks good, click >>> 'Save masks' <<<.\n"
        "If not, click 'Clear points' and redo from Step 3."
    ),
    "saved": (
        "== STEP 7 / 7 == Done!\n"
        "Masks saved. Go back to the terminal and press Ctrl+C to continue the pipeline."
    ),
}


class PromptGUI(object):
    def __init__(self, checkpoint_dir, sam_model_type, device):
        self.checkpoint_dir = checkpoint_dir
        self.sam_model_type = sam_model_type
        self.device = device
        self.sam_model = None
        self.tracker = None

        self.selected_points = []
        self.selected_labels = []
        self.cur_label_val = 1.0

        self.frame_index = 0
        self.image = None
        self.cur_mask_idx = 0
        # can store multiple object masks
        # saves the masks and logits for each mask index
        self.cur_masks = {}
        self.cur_logits = {}
        self.index_masks_all = []
        self.color_masks_all = []

        self.img_dir = ""
        self.img_paths = []

    def lazy_init_sam_model(self):
        if self.sam_model is None:
            self.sam_model = init_sam_model(
                self.checkpoint_dir, self.sam_model_type, self.device
            )

    def lazy_init_tracker(self):
        if self.tracker is None:
            self.tracker = init_tracker(self.checkpoint_dir, self.device)

    def clear_points(self) -> tuple[None, None, str]:
        self.selected_points.clear()
        self.selected_labels.clear()
        return None, None, STEP_INSTRUCTIONS["sam_ready"]

    def add_new_mask(self):
        self.cur_mask_idx += 1
        self.clear_points()
        message = (
            f"== STEP 4 / 7 == Adding mask #{self.cur_mask_idx}.\n"
            "Click on the next object. When done, click >>> 'Submit mask for tracking' <<<."
        )
        return None, message

    def make_index_mask(self):
        assert len(self.cur_masks) > 0
        idcs = list(self.cur_masks.keys())
        idx_mask = self.cur_masks[idcs[0]].astype("uint8")
        for i in idcs:
            mask = self.cur_masks[i]
            idx_mask[mask] = i + 1
        return idx_mask

    def _clear_image(self):
        self.image = None
        self.cur_mask_idx = 0
        self.frame_index = 0
        self.cur_masks = {}
        self.cur_logits = {}
        self.index_masks_all = []
        self.color_masks_all = []

    def set_img_dir(self, img_dir: str) -> int:
        self._clear_image()
        self.img_dir = img_dir
        self.img_paths = [
            f"{img_dir}/{p}" for p in sorted(os.listdir(img_dir)) if isimage(p)
        ]
        return len(self.img_paths)

    def set_input_image(self, i: int = 0) -> np.ndarray | None:
        guru.debug(f"Setting frame {i} / {len(self.img_paths)}")
        if i < 0 or i >= len(self.img_paths):
            return self.image
        self.clear_points()
        self._clear_image()
        self.frame_index = i
        image = iio.imread(self.img_paths[i])
        self.image = image
        return image

    @property
    def sam_ready(self) -> bool:
        return self.sam_model is not None and self.sam_model.is_image_set

    def get_sam_features(self) -> tuple[str, np.ndarray | None]:
        if self.image is None:
            return "ERROR: No image loaded. Use the Frame index slider first.", None
        try:
            self.lazy_init_sam_model()
        except Exception as e:
            return (
                f"ERROR: Failed to load SAM model: {e}\n"
                "Make sure checkpoints exist: run preproc/setup_dependencies.sh"
            ), None
        assert self.sam_model is not None
        self.sam_model.set_image(self.image)
        return STEP_INSTRUCTIONS["sam_ready"], self.image

    def set_positive(self) -> str:
        self.cur_label_val = 1.0
        return "Mode: POSITIVE (green). Click on the object to include."

    def set_negative(self) -> str:
        self.cur_label_val = 0.0
        return "Mode: NEGATIVE (red). Click on background to exclude."

    def add_point(self, img, i, j):
        self.selected_points.append([j, i])
        self.selected_labels.append(self.cur_label_val)
        mask, logit = self.get_sam_mask(
            img, np.array(self.selected_points), np.array(self.selected_labels)
        )
        self.cur_masks[self.cur_mask_idx] = mask
        self.cur_logits[self.cur_mask_idx] = logit
        idx_mask = self.make_index_mask()
        return idx_mask

    def get_sam_mask(self, img, input_points, input_labels):
        self.lazy_init_sam_model()
        assert self.sam_model is not None
        if self.sam_model.is_image_set is False:
            self.sam_model.set_image(img)

        logits = self.cur_logits.get(self.cur_mask_idx, None)
        mask_input = None if logits is None else logits[None]
        masks, scores, logits = self.sam_model.predict(
            point_coords=input_points,
            point_labels=input_labels,
            mask_input=mask_input,
            multimask_output=True,
        )
        idx_sel = np.argmax(scores)
        return masks[idx_sel], logits[idx_sel]

    def run_tracker(self) -> tuple[str, str]:
        idx_mask = self.make_index_mask()
        self.lazy_init_tracker()
        assert self.tracker is not None
        self.tracker.clear_memory()

        images = [iio.imread(p)[:, :, :3] for p in self.img_paths]

        self.index_masks_all = track_masks(
            self.tracker, images, idx_mask, self.frame_index
        )

        out_frames, self.color_masks_all = colorize_masks(images, self.index_masks_all)
        out_vidpath = "tracked_colors.mp4"
        iio.mimwrite(out_vidpath, out_frames)
        return out_vidpath, STEP_INSTRUCTIONS["tracked"]

    def save_masks_to_dir(self, output_dir: str) -> str:
        assert self.color_masks_all is not None
        os.makedirs(output_dir, exist_ok=True)
        for img_path, clr_mask in zip(self.img_paths, self.color_masks_all):
            name = os.path.basename(img_path)
            out_path = f"{output_dir}/{name}"
            iio.imwrite(out_path, clr_mask)
        guru.debug(f"Saved masks to {output_dir}!")
        return STEP_INSTRUCTIONS["saved"]


def isimage(p):
    ext = os.path.splitext(p.lower())[-1]
    return ext in [".png", ".jpg", ".jpeg"]


def has_images_directly(dir_path):
    """Check if a directory contains image files directly (not in subdirs)."""
    if not os.path.isdir(dir_path):
        return False
    return any(isimage(f) for f in os.listdir(dir_path))


def list_subdirs(dir_path):
    """List only subdirectories (not files) in a directory."""
    if dir_path is None or not os.path.isdir(dir_path):
        return []
    return sorted(
        d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))
    )


def draw_points(img, points, labels):
    out = img.copy()
    for p, label in zip(points, labels):
        x, y = int(p[0]), int(p[1])
        color = (0, 255, 0) if label == 1.0 else (255, 0, 0)
        out = cv2.circle(out, (x, y), 10, color, -1)
    return out


def get_hls_palette(
    n_colors: int,
    lightness: float = 0.5,
    saturation: float = 0.7,
) -> np.ndarray:
    hues = np.linspace(0, 1, int(n_colors) + 1)[1:-1]
    palette = [(0.0, 0.0, 0.0)] + [
        colorsys.hls_to_rgb(h_i, lightness, saturation) for h_i in hues
    ]
    return (255 * np.asarray(palette)).astype("uint8")


def colorize_masks(images, index_masks, fac: float = 0.5):
    max_idx = max([m.max() for m in index_masks])
    guru.debug(f"{max_idx=}")
    palette = get_hls_palette(max_idx + 1)
    color_masks = []
    out_frames = []
    for img, mask in zip(images, index_masks):
        clr_mask = palette[mask.astype("int")]
        color_masks.append(clr_mask)
        out_u = compose_img_mask(img, clr_mask, fac)
        out_frames.append(out_u)
    return out_frames, color_masks


def compose_img_mask(img, color_mask, fac: float = 0.5):
    out_f = fac * img / 255 + (1 - fac) * color_mask / 255
    out_u = (255 * out_f).astype("uint8")
    return out_u


def listdir(vid_dir):
    if vid_dir is not None and os.path.isdir(vid_dir):
        return sorted(os.listdir(vid_dir))
    return []


def make_demo(
    checkpoint_dir,
    sam_model_type,
    device,
    root_dir,
    vid_name: str = "videos",
    img_name: str = "images",
    mask_name: str = "masks",
):
    prompts = PromptGUI(checkpoint_dir, sam_model_type, device)

    vid_root = f"{root_dir}/{vid_name}"
    img_root = f"{root_dir}/{img_name}"

    # Auto-detect: images directly in img_root (no subdirectories)?
    direct_images = has_images_directly(img_root)
    if direct_images:
        # Images are directly in root_dir/images/ (e.g. from run_4d.py)
        num_imgs = prompts.set_img_dir(img_root)
        auto_loaded = num_imgs > 0
        start_instructions = (
            f"== STEP 2 / 7 == {num_imgs} images auto-loaded.\n"
            "1. Use the 'Frame index' slider to pick a frame where the foreground object is clearly visible.\n"
            "2. Click >>> 'Get SAM features' <<< (wait for Step 3 to appear).\n"
            "NOTE: You MUST click 'Get SAM features' before clicking on the image!"
        )
        # For direct layout, masks go in root_dir/masks/ (no seq subdir)
        auto_mask_dir = f"{root_dir}/{mask_name}"
    else:
        auto_loaded = False
        start_instructions = STEP_INSTRUCTIONS["start"]
        auto_mask_dir = None

    with gr.Blocks(title="Mask Annotation Tool") as demo:
        instruction = gr.Textbox(
            start_instructions,
            label=">>> INSTRUCTION (follow these steps) <<<",
            interactive=False,
            lines=4,
        )
        with gr.Row():
            root_dir_field = gr.Text(root_dir, label="Dataset root directory")
            vid_name_field = gr.Text(vid_name, label="Video subdirectory name")
            img_name_field = gr.Text(img_name, label="Image subdirectory name")
            mask_name_field = gr.Text(mask_name, label="Mask subdirectory name")
            seq_name_field = gr.Text(None, label="Sequence name", interactive=False)

        with gr.Row():
            with gr.Column():
                vid_files = listdir(vid_root)
                vid_files_field = gr.Dropdown(label="Video files", choices=vid_files)
                input_video_field = gr.Video(label="Input Video")

                with gr.Row():
                    start_time = gr.Number(0, label="Start time (s)")
                    end_time = gr.Number(0, label="End time (s)")
                    sel_fps = gr.Number(30, label="FPS")
                    sel_height = gr.Number(540, label="Height")
                    extract_button = gr.Button("Extract frames")

            with gr.Column():
                img_dirs = list_subdirs(img_root) if not direct_images else []
                img_dirs_field = gr.Dropdown(
                    label="Image directories", choices=img_dirs,
                    visible=not direct_images,
                )
                img_dir_field = gr.Text(
                    img_root if direct_images else None,
                    label="Input directory",
                    interactive=False,
                )
                frame_index = gr.Slider(
                    label="Frame index",
                    minimum=0,
                    maximum=max(len(prompts.img_paths) - 1, 0),
                    value=0,
                    step=1,
                )
                sam_button = gr.Button(
                    ">>> Get SAM features <<<",
                    variant="primary",
                )
                input_image = gr.Image(
                    prompts.set_input_image(0) if auto_loaded else None,
                    label="Input Frame",
                    every=1,
                )
                with gr.Row():
                    pos_button = gr.Button("Toggle positive")
                    neg_button = gr.Button("Toggle negative")
                clear_button = gr.Button("Clear points")

            with gr.Column():
                output_img = gr.Image(label="Current selection")
                add_button = gr.Button("Add new mask")
                submit_button = gr.Button(
                    ">>> Submit mask for tracking <<<",
                    variant="primary",
                )
                final_video = gr.Video(label="Masked video")
                mask_dir_field = gr.Text(
                    auto_mask_dir,
                    label="Path to save masks",
                    interactive=False,
                )
                save_button = gr.Button(
                    ">>> Save masks <<<",
                    variant="primary",
                )

        def update_vid_root(root_dir, vid_name):
            vid_root = f"{root_dir}/{vid_name}"
            vid_paths = listdir(vid_root)
            guru.debug(f"Updating video paths: {vid_paths=}")
            return vid_paths

        def update_img_root(root_dir, img_name):
            img_root = f"{root_dir}/{img_name}"
            img_dirs = list_subdirs(img_root)
            guru.debug(f"Updating img dirs: {img_dirs=}")
            return img_root, img_dirs

        def update_mask_dir(root_dir, mask_name, seq_name):
            if seq_name:
                return f"{root_dir}/{mask_name}/{seq_name}"
            return f"{root_dir}/{mask_name}"

        def update_root_paths(root_dir, vid_name, img_name, mask_name, seq_name):
            return (
                update_vid_root(root_dir, vid_name),
                update_img_root(root_dir, img_name),
                update_mask_dir(root_dir, mask_name, seq_name),
            )

        def select_video(root_dir, vid_name, seq_file):
            seq_name = os.path.splitext(seq_file)[0]
            guru.debug(f"Selected video: {seq_file=}")
            vid_path = f"{root_dir}/{vid_name}/{seq_file}"
            return seq_name, vid_path

        def extract_frames(
            root_dir, vid_name, img_name, vid_file, start, end, fps, height, ext="png"
        ):
            seq_name = os.path.splitext(vid_file)[0]
            vid_path = f"{root_dir}/{vid_name}/{vid_file}"
            out_dir = f"{root_dir}/{img_name}/{seq_name}"
            guru.debug(f"Extracting frames to {out_dir}")
            os.makedirs(out_dir, exist_ok=True)

            def make_time(seconds):
                return datetime.time(
                    seconds // 3600, (seconds % 3600) // 60, seconds % 60
                )

            start_time = make_time(start).strftime("%H:%M:%S")
            end_time = make_time(end).strftime("%H:%M:%S")
            cmd = (
                f"ffmpeg -ss {start_time} -to {end_time} -i {vid_path} "
                f"-vf 'scale=-1:{height},fps={fps}' {out_dir}/%05d.{ext}"
            )
            print(cmd)
            subprocess.call(cmd, shell=True)
            img_root = f"{root_dir}/{img_name}"
            img_dirs = list_subdirs(img_root)
            return out_dir, img_dirs

        def select_image_dir(root_dir, img_name, seq_name):
            img_dir = f"{root_dir}/{img_name}/{seq_name}"
            guru.debug(f"Selected image dir: {img_dir}")
            return seq_name, img_dir

        def update_image_dir(root_dir, img_name, seq_name):
            if seq_name:
                img_dir = f"{root_dir}/{img_name}/{seq_name}"
            else:
                img_dir = f"{root_dir}/{img_name}"
            num_imgs = prompts.set_img_dir(img_dir)
            slider = gr.Slider(minimum=0, maximum=num_imgs - 1, value=0, step=1)
            return slider, STEP_INSTRUCTIONS["loaded"]

        def get_select_coords(img, evt: gr.SelectData):
            if not prompts.sam_ready:
                gr.Warning(
                    "Click 'Get SAM features' first, then click on the image."
                )
                return img
            i = evt.index[1]  # type: ignore
            j = evt.index[0]  # type: ignore
            index_mask = prompts.add_point(img, i, j)
            guru.debug(f"{index_mask.shape=}")
            palette = get_hls_palette(index_mask.max() + 1)
            color_mask = palette[index_mask]
            out_u = compose_img_mask(img, color_mask)
            out = draw_points(out_u, prompts.selected_points, prompts.selected_labels)
            return out

        # update the root directory
        root_dir_field.submit(
            update_root_paths,
            [
                root_dir_field,
                vid_name_field,
                img_name_field,
                mask_name_field,
                seq_name_field,
            ],
            outputs=[vid_files_field, img_dirs_field, mask_dir_field],
        )
        vid_name_field.submit(
            update_vid_root,
            [root_dir_field, vid_name_field],
            outputs=[vid_files_field],
        )
        img_name_field.submit(
            update_img_root,
            [root_dir_field, img_name_field],
            outputs=[img_dirs_field],
        )
        mask_name_field.submit(
            update_mask_dir,
            [root_dir_field, mask_name_field, seq_name_field],
            outputs=[mask_dir_field],
        )

        # selecting a video file
        vid_files_field.select(
            select_video,
            [root_dir_field, vid_name_field, vid_files_field],
            outputs=[seq_name_field, input_video_field],
        )

        # when the img_dir_field changes
        img_dir_field.change(
            update_image_dir,
            [root_dir_field, img_name_field, seq_name_field],
            [frame_index, instruction],
        )
        seq_name_field.change(
            update_mask_dir,
            [root_dir_field, mask_name_field, seq_name_field],
            outputs=[mask_dir_field],
        )

        # selecting an image directory
        img_dirs_field.select(
            select_image_dir,
            [root_dir_field, img_name_field, img_dirs_field],
            [seq_name_field, img_dir_field],
        )

        # extracting frames from video
        extract_button.click(
            extract_frames,
            [
                root_dir_field,
                vid_name_field,
                img_name_field,
                vid_files_field,
                start_time,
                end_time,
                sel_fps,
                sel_height,
            ],
            outputs=[img_dir_field, img_dirs_field],
        )

        frame_index.change(prompts.set_input_image, [frame_index], [input_image])
        input_image.select(get_select_coords, [input_image], [output_img])

        sam_button.click(prompts.get_sam_features, outputs=[instruction, input_image])
        clear_button.click(
            prompts.clear_points, outputs=[output_img, final_video, instruction]
        )
        pos_button.click(prompts.set_positive, outputs=[instruction])
        neg_button.click(prompts.set_negative, outputs=[instruction])

        add_button.click(prompts.add_new_mask, outputs=[output_img, instruction])
        submit_button.click(prompts.run_tracker, outputs=[final_video, instruction])
        save_button.click(
            prompts.save_masks_to_dir, [mask_dir_field], outputs=[instruction]
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--sam_model_type", type=str, default="vit_h")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--vid_name", type=str, default="videos")
    parser.add_argument("--img_name", type=str, default="images")
    parser.add_argument("--mask_name", type=str, default="masks")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    demo = make_demo(
        args.checkpoint_dir,
        args.sam_model_type,
        device,
        args.root_dir,
        args.vid_name,
        args.img_name,
    )
    demo.launch(server_port=args.port)
