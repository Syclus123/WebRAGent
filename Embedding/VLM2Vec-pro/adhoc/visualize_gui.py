import os
import PIL
import glob
import argparse
from pathlib import Path

from transformers.image_utils import ChannelDimension
from src.model.vlm_backbone.qwen2_vl_tokenselection.processing_qwen2_vl import Qwen2VLProcessor
from src.model.vlm_backbone.qwen2_vl_tokenselection.image_processing_qwen2_vl import Qwen2VLImageProcessor
from src.model.vlm_backbone.qwen2_vl_tokenselection.tokenization_qwen2_fast import Qwen2TokenizerFast

def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode text and images with Qwen2VLProcessor"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="pretrained model identifier or local path"
    )
    parser.add_argument(
        "--image_pattern",
        type=str,
        required=True,
        help="directory containing image files"
    )
    parser.add_argument(
        "--resize_use_processor",
        type=str,
        default=True,
        help="resize visual inputs insides processor, e.g. Qwen2VLImageProcessor"
    )
    parser.add_argument(
        "--resize_min_pixels",
        type=int,
        default=28*28*4,
        help="the min pixels of the image to resize the image."
    )
    parser.add_argument(
        "--resize_max_pixels",
        type=int,
        default=28*28*1280,
        help="the max pixels of the image to resize the image."
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        default=None,
        help="directory for visualization outputs (optional)"
    )
    parser.add_argument(
        "--uigraph_use",
        action="store_true",
        default=True,
        help="enable UI graph tokens"
    )
    parser.add_argument(
        "--uigraph_diff",
        type=int,
        default=1,
        help="pixel diff threshold for UI graph"
    )
    parser.add_argument(
        "--uigraph_rand",
        action="store_true",
        default=False,
        help="enable random UI graph construction"
    )
    parser.add_argument(
        "--uimask_ratio",
        type=float,
        default=0.5,
        help="ratio of patch tokens to mask"
    )
    parser.add_argument(
        "--uimask_rand",
        action="store_true",
        default=False,
        help="enable random token masking"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) load tokenizer and image processor
    tokenizer = Qwen2TokenizerFast.from_pretrained(args.model_name)
    image_processor = Qwen2VLImageProcessor.from_pretrained(args.model_name)

    image_processor.do_resize = args.resize_use_processor
    image_processor.min_pixels = args.resize_min_pixels
    image_processor.max_pixels = args.resize_max_pixels

    # 2) build the fused processor
    processor = Qwen2VLProcessor.from_pretrained(
        args.model_name,
        image_processor=image_processor,
        tokenizer=tokenizer,
        uigraph_use=args.uigraph_use,
        uigraph_diff=args.uigraph_diff,
        uigraph_rand=args.uigraph_rand,
        uimask_ratio=args.uimask_ratio,
        uimask_rand=args.uimask_rand
    )

    # 3) load images from directory
    img_paths = sorted(glob.glob(args.image_pattern))
    if not img_paths:
        raise FileNotFoundError(f"No images found in directory: {args.image_pattern}")
    images = [PIL.Image.open(p) for p in img_paths]
    save_dir = os.path.join(args.vis_dir, os.path.basename(img_paths[0]).split('_step')[0])
    os.makedirs(save_dir, exist_ok=True)

    # 4) run the processor
    inputs = processor(
        text="<|image_pad|>" * len(images),
        images=images,
        return_tensors="np",
        max_length=None,
        truncation=False,
        input_data_format=ChannelDimension.LAST,
        vis_dir=save_dir
    )

if __name__ == "__main__":
    main()