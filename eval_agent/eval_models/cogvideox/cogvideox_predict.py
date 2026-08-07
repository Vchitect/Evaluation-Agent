"""CogVideoX-2B and CogVideoX-5B adapters for video generation."""

import gc
from pathlib import Path

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

CUR_DIR = Path(__file__).resolve().parent


class CogVideoX:
    def __init__(self, model_variant="2b", device=None, dtype=None):
        """
        Initialize CogVideoX model.

        Args:
            model_variant: "2b" or "5b" for different model sizes
            device: torch device to use (defaults to cuda if available)
            dtype: torch dtype (defaults to float16 for 2B, bfloat16 for 5B)
        """
        self.model_variant = model_variant.lower()
        if self.model_variant not in {"2b", "5b"}:
            raise ValueError("model_variant must be either '2b' or '5b'")

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Set appropriate dtype based on model variant
        if dtype is None:
            if self.model_variant == "5b":
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.dtype = dtype

        if self.device.type == "cpu" and self.dtype in {torch.float16, torch.bfloat16}:
            self.dtype = torch.float32

        # Model paths
        self.model_name = self._get_model_name()
        local_model_path = CUR_DIR / "checkpoints" / self.model_name

        # Check if local checkpoint exists, otherwise use HuggingFace hub
        if not local_model_path.is_dir():
            print(f"Local model not found at {local_model_path}")
            print(f"Using HuggingFace model: THUDM/{self.model_name}")
            self.model_path = f"THUDM/{self.model_name}"
        else:
            self.model_path = str(local_model_path)

        # Initialize the pipeline
        self._load_model()

    def _get_model_name(self):
        """Get the correct model name based on variant."""
        if self.model_variant == "5b":
            return "CogVideoX-5b"
        else:
            return "CogVideoX-2b"

    def _load_model(self):
        """Load the CogVideoX model pipeline."""
        print(f"Loading {self.model_name} model...")

        # Load pipeline with appropriate settings
        self.pipe = CogVideoXPipeline.from_pretrained(
            self.model_path, torch_dtype=self.dtype
        )

        # Install offload hooks before moving any component to CUDA. Moving the
        # full 5B pipeline first can exceed the memory of a 24 GB device.
        if self.device.type == "cuda" and hasattr(
            self.pipe, "enable_model_cpu_offload"
        ):
            self.pipe.enable_model_cpu_offload(device=self.device)
        elif self.device.type == "cuda" and hasattr(
            self.pipe, "enable_sequential_cpu_offload"
        ):
            self.pipe.enable_sequential_cpu_offload(device=self.device)
        else:
            self.pipe = self.pipe.to(self.device)

        # Enable VAE slicing for memory efficiency
        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()
        if hasattr(self.pipe, "enable_vae_tiling"):
            self.pipe.enable_vae_tiling()

        print(f"Model loaded successfully on {self.device}")

    def predict(
        self,
        prompt,
        save_name,
        num_inference_steps=50,
        num_frames=49,
        guidance_scale=6.0,
        seed=42,
        height=480,
        width=720,
    ):
        """
        Generate video from text prompt using official CogVideoX defaults.

        Args:
            prompt: Text prompt for video generation
            save_name: Path to save the generated video
            num_inference_steps: Number of denoising steps (default: 50)
            num_frames: Number of frames to generate (default: 49 for 2B/5B, must be 1 + 8*k)
            guidance_scale: Guidance scale for classifier-free guidance (default: 6.0)
            seed: Random seed for reproducibility (default: 42)
            height: Video height in pixels (default: 480 for 2B/5B)
            width: Video width in pixels (default: 720 for 2B/5B)
        """
        # Set default values based on model variant (matching official defaults)
        if num_frames is None:
            # Official defaults: 49 frames for CogVideoX-2B/5B (6 seconds at 8 fps)
            num_frames = 49

        if height is None:
            # Official default resolution for CogVideoX-2B/5B
            height = 480

        if width is None:
            # Official default resolution for CogVideoX-2B/5B
            width = 720

        if num_frames < 1:
            raise ValueError("num_frames must be at least 1")

        # Validate num_frames (must be 1 + 8*k for CogVideoX-2B/5B)
        if (num_frames - 1) % 8 != 0:
            valid_frames = 1 + 8 * ((num_frames - 1) // 8)
            print(
                f"Warning: num_frames must be 1 + 8*k. Adjusting from {num_frames} to {valid_frames}"
            )
            num_frames = valid_frames

        # Set random seed if provided
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        print(f"Generating video with prompt: {prompt}")
        print(
            f"Parameters: steps={num_inference_steps}, frames={num_frames}, "
            f"guidance={guidance_scale}, size={width}x{height}"
        )

        try:
            # Generate video
            output = self.pipe(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                generator=generator,
                height=height,
                width=width,
            )

            Path(save_name).parent.mkdir(parents=True, exist_ok=True)
            export_to_video(output.frames[0], save_name, fps=8)

            print(f"Video saved to: {save_name}")

        except Exception as e:
            print(f"Error during video generation: {e}")
            raise
        finally:
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def predict_with_options(self, prompt, save_name, **kwargs):
        """
        Generate video with custom options.

        This method allows passing additional parameters to the predict method.
        """
        return self.predict(prompt, save_name, **kwargs)

    def unload_model(self):
        """Unload model from memory."""
        del self.pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Model unloaded from memory")


class CogVideoX2B(CogVideoX):
    """Convenience class for CogVideoX-2B model."""

    def __init__(self, device=None):
        super().__init__(model_variant="2b", device=device, dtype=torch.float16)


class CogVideoX5B(CogVideoX):
    """Convenience class for CogVideoX-5B model."""

    def __init__(self, device=None):
        super().__init__(model_variant="5b", device=device, dtype=torch.bfloat16)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Generate video using CogVideoX")
    parser.add_argument(
        "--prompt", type=str, required=True, help="Text prompt for video generation"
    )
    parser.add_argument(
        "--output", type=str, default="output_video.mp4", help="Output video path"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["2b", "5b"],
        default="2b",
        help="Model variant to use",
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument(
        "--frames", type=int, default=49, help="Number of frames (must be 1 + 8*k)"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Initialize model
    model = CogVideoX(model_variant=args.model)

    # Generate video
    model.predict(
        prompt=args.prompt,
        save_name=args.output,
        num_inference_steps=args.steps,
        num_frames=args.frames,
        seed=args.seed,
    )
