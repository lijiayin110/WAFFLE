"""Batch inference script for wall segmentation masks.

This utility wraps :class:`WallDetection` from ``wall_detection_inf.py`` and
applies it to all images inside a directory.  The defaults in this script are
configured for a Windows environment in which the fine-tuned Stable Diffusion
and ControlNet checkpoints are stored under ``E:\\我的项目\\空间建模\\WAFFLE\\models``
and the test images live in ``E:\\我的项目\\test``.

Example usage
-------------

```bash
python run_wall_detection_batch.py \
    --controlnet-path "E:\\我的项目\\空间建模\\WAFFLE\\models\\ft_controlnet-wall-detection" \
    --stable-diffusion-path "E:\\我的项目\\空间建模\\WAFFLE\\models\\ft_stable_diffusion" \
    --input-dir "E:\\我的项目\\test" \
    --output-dir "E:\\我的项目\\test\\wall_masks"
```
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

from .wall_detection_inf import WallDetection


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the batch inference script."""

    parser = argparse.ArgumentParser(description="Batch wall segmentation")
    parser.add_argument(
        "--controlnet-path",
        type=Path,
        required=True,
        help=(
            "Path to the fine-tuned ControlNet checkpoint directory. "
            "Example: E:/我的项目/空间建模/WAFFLE/models/ft_controlnet-wall-detection"
        ),
    )
    parser.add_argument(
        "--stable-diffusion-path",
        type=Path,
        default=Path("E:/我的项目/空间建模/WAFFLE/models/ft_stable_diffusion"),
        help=(
            "Path to the fine-tuned Stable Diffusion checkpoint directory. "
            "Defaults to the ft_stable_diffusion folder shipped with the project."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("E:/我的项目/test"),
        help="Directory containing floor-plan images to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("E:/我的项目/test/wall_masks"),
        help="Directory where the predicted wall masks will be written.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A floor plan",
        help="Prompt to condition the diffusion model on.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=16,
        help=(
            "Number of diffusion samples to aggregate for each input image. "
            "Higher values yield smoother masks at the cost of inference time."
        ),
    )
    return parser.parse_args()


def iter_images(input_dir: Path) -> Iterable[Path]:
    """Yield valid image files from ``input_dir`` sorted alphabetically."""

    supported_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in supported_suffixes:
            yield path


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    LOGGER.info("Loading models...")

    detector = WallDetection(
        ckpt_path=str(args.controlnet_path),
        stable_diffusion_ckpt=str(args.stable_diffusion_path),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in iter_images(args.input_dir):
        LOGGER.info("Processing %s", image_path)
        mask = detector.infer(
            image_path=str(image_path),
            prompt=args.prompt,
            num_images=args.num_images,
        )
        output_path = args.output_dir / f"{image_path.stem}_mask.png"
        mask.save(output_path)
        LOGGER.info("Saved mask to %s", output_path)

    LOGGER.info("Completed processing %d image(s).", sum(1 for _ in iter_images(args.input_dir)))


if __name__ == "__main__":
    main()

