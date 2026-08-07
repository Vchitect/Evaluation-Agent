#!/usr/bin/env python3
"""
Simple inference script for CogVideoX model.
Usage: python inference.py --prompt "A beautiful sunset over the ocean" --output sunset.mp4
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from eval_agent.eval_models.cogvideox.cogvideox_predict import (
    CogVideoX2B,
    CogVideoX5B,
)


def main():
    parser = argparse.ArgumentParser(description="CogVideoX Video Generation Inference")
    parser.add_argument(
        "--prompt", type=str, required=True, help="Text prompt for video generation"
    )
    parser.add_argument(
        "--output", type=str, default="output_video.mp4", help="Output video file path"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["2b", "5b"],
        default="2b",
        help="Model variant: 2b or 5b",
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument(
        "--frames", type=int, default=49, help="Number of frames (must be 1 + 8*k)"
    )
    parser.add_argument("--guidance", type=float, default=6.0, help="Guidance scale")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=720, help="Video width")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Initializing CogVideoX-{args.model.upper()} model...")

    # Initialize the model
    if args.model == "2b":
        model = CogVideoX2B()
    else:
        model = CogVideoX5B()

    # Generate video
    print(f"\nGenerating video with prompt: '{args.prompt}'")
    print(f"Output will be saved to: {args.output}")

    try:
        model.predict(
            prompt=args.prompt,
            save_name=args.output,
            num_inference_steps=args.steps,
            num_frames=args.frames,
            guidance_scale=args.guidance,
            seed=args.seed,
            height=args.height,
            width=args.width,
        )
        print("\n✅ Video generation completed successfully!")
        print(f"Video saved at: {os.path.abspath(args.output)}")

    except Exception as e:
        print(f"\n❌ Error during video generation: {e}")
        return 1

    finally:
        # Clean up
        model.unload_model()

    return 0


if __name__ == "__main__":
    exit(main())
