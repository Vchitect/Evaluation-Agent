import sys
from pathlib import Path

import imageio
import torch
from diffusers import IFSuperResolutionPipeline, VideoToVideoSDPipeline
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

# Add the pinned Show-1 submodule to the import path.
CUR_DIR = Path(__file__).resolve().parent
SHOW1_REPO = CUR_DIR / "Show-1"
if not (SHOW1_REPO / "showone").is_dir():
    raise ImportError(
        "The Show-1 submodule is not initialized. From the repository root, "
        "run `git submodule update --init --recursive -- "
        "eval_agent/eval_models/show1/Show-1`, then install the dependencies "
        "described in "
        f"{CUR_DIR / 'README.md'}."
    )
sys.path.insert(0, str(SHOW1_REPO))

from showone.pipelines import TextToVideoIFInterpPipeline, TextToVideoIFPipeline
from showone.pipelines.pipeline_t2v_base_pixel import tensor2vid
from showone.pipelines.pipeline_t2v_sr_pixel_cond import (
    TextToVideoIFSuperResolutionPipeline_Cond,
)


class Show1:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("The Show-1 adapter requires a CUDA-capable GPU.")
        self.device = "cuda"
        self.dtype = torch.float16

        # Initialize all pipelines
        # Base Model
        self.pipe_base = TextToVideoIFPipeline.from_pretrained(
            "showlab/show-1-base", torch_dtype=self.dtype, variant="fp16"
        )
        self.pipe_base.enable_model_cpu_offload()

        # Interpolation Model
        self.pipe_interp = TextToVideoIFInterpPipeline.from_pretrained(
            "showlab/show-1-interpolation", torch_dtype=self.dtype, variant="fp16"
        )
        self.pipe_interp.enable_model_cpu_offload()

        # Super-Resolution Model 1 - Image SR
        self.pipe_sr_1_image = IFSuperResolutionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0",
            text_encoder=None,
            torch_dtype=self.dtype,
            variant="fp16",
        )
        self.pipe_sr_1_image.enable_model_cpu_offload()

        # Super-Resolution Model 1 - Conditional
        self.pipe_sr_1_cond = TextToVideoIFSuperResolutionPipeline_Cond.from_pretrained(
            "showlab/show-1-sr1", torch_dtype=self.dtype
        )
        self.pipe_sr_1_cond.enable_model_cpu_offload()

        # Super-Resolution Model 2
        self.pipe_sr_2 = VideoToVideoSDPipeline.from_pretrained(
            "showlab/show-1-sr2", torch_dtype=self.dtype
        )
        self.pipe_sr_2.enable_model_cpu_offload()
        self.pipe_sr_2.enable_vae_slicing()

    def predict(
        self,
        prompt,
        save_name,
        seed=42,
        negative_prompt="low resolution, blur",
        num_inference_steps_base=75,
        guidance_scale_base=9.0,
        enable_interpolation=True,
        enable_sr1=True,
        enable_sr2=True,
    ):
        """
        Generate video from text prompt.

        Args:
            prompt: Text prompt for video generation
            save_name: Path to save the generated video
            seed: Random seed for generation
            negative_prompt: Negative prompt for generation
            num_inference_steps_base: Number of inference steps for base model
            guidance_scale_base: Guidance scale for base model
            enable_interpolation: Enable frame interpolation
            enable_sr1: Enable first super-resolution pass
            enable_sr2: Enable second super-resolution pass
        """
        # Set random seed
        generator = torch.manual_seed(seed)

        # Text embeds
        prompt_embeds, negative_embeds = self.pipe_base.encode_prompt(prompt)

        # Keyframes generation (8x64x40, 2fps)
        print(f"Generating base video for: {prompt}")
        video_frames = self.pipe_base(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            num_frames=8,
            height=40,
            width=64,
            num_inference_steps=num_inference_steps_base,
            guidance_scale=guidance_scale_base,
            generator=generator,
            output_type="pt",
        ).frames

        # Frame interpolation (8x64x40, 2fps -> 29x64x40, 7.5fps)
        if enable_interpolation:
            print("Performing frame interpolation...")
            bsz, channel, num_frames, height, width = video_frames.shape
            new_num_frames = 3 * (num_frames - 1) + num_frames
            new_video_frames = torch.zeros(
                (bsz, channel, new_num_frames, height, width),
                dtype=video_frames.dtype,
                device=video_frames.device,
            )
            new_video_frames[:, :, torch.arange(0, new_num_frames, 4), ...] = (
                video_frames
            )
            init_noise = randn_tensor(
                (bsz, channel, 5, height, width),
                dtype=video_frames.dtype,
                device=video_frames.device,
                generator=torch.manual_seed(seed),
            )

            for i in range(num_frames - 1):
                batch_i = torch.zeros(
                    (bsz, channel, 5, height, width),
                    dtype=video_frames.dtype,
                    device=video_frames.device,
                )
                batch_i[:, :, 0, ...] = video_frames[:, :, i, ...]
                batch_i[:, :, -1, ...] = video_frames[:, :, i + 1, ...]
                batch_i = self.pipe_interp(
                    pixel_values=batch_i,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    num_frames=batch_i.shape[2],
                    height=40,
                    width=64,
                    num_inference_steps=75,
                    guidance_scale=4.0,
                    generator=torch.manual_seed(seed),
                    output_type="pt",
                    init_noise=init_noise,
                    cond_interpolation=True,
                ).frames
                new_video_frames[:, :, i * 4 : i * 4 + 5, ...] = batch_i

            video_frames = new_video_frames

        # Super-resolution 1 (29x64x40 -> 29x256x160)
        if enable_sr1:
            print("Applying first super-resolution pass...")
            bsz, channel, num_frames, height, width = video_frames.shape
            window_size, stride = 8, 7
            new_video_frames = torch.zeros(
                (bsz, channel, num_frames, height * 4, width * 4),
                dtype=video_frames.dtype,
                device=video_frames.device,
            )

            for i in range(0, num_frames - window_size + 1, stride):
                batch_i = video_frames[:, :, i : i + window_size, ...]

                if i == 0:
                    first_frame_cond = self.pipe_sr_1_image(
                        image=video_frames[:, :, 0, ...],
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_embeds,
                        height=height * 4,
                        width=width * 4,
                        num_inference_steps=70,
                        guidance_scale=4.0,
                        noise_level=150,
                        generator=torch.manual_seed(seed),
                        output_type="pt",
                    ).images
                    first_frame_cond = first_frame_cond.unsqueeze(2)
                else:
                    first_frame_cond = new_video_frames[:, :, i : i + 1, ...]

                batch_i = self.pipe_sr_1_cond(
                    image=batch_i,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    first_frame_cond=first_frame_cond,
                    height=height * 4,
                    width=width * 4,
                    num_inference_steps=125,
                    guidance_scale=7.0,
                    noise_level=250,
                    generator=torch.manual_seed(seed),
                    output_type="pt",
                ).frames
                new_video_frames[:, :, i : i + window_size, ...] = batch_i

            video_frames = new_video_frames

        # Super-resolution 2 (29x256x160 -> 29x576x320)
        if enable_sr2:
            print("Applying second super-resolution pass...")
            video_frames_pil = [
                Image.fromarray(frame).resize((576, 320))
                for frame in tensor2vid(video_frames.clone())
            ]
            video_frames = self.pipe_sr_2(
                prompt,
                negative_prompt=negative_prompt,
                video=video_frames_pil,
                strength=0.8,
                num_inference_steps=50,
                generator=torch.manual_seed(seed),
                output_type="pt",
            ).frames

        # Save video
        print(f"Saving video to {save_name}")
        Path(save_name).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(save_name, tensor2vid(video_frames.clone()), fps=8)
        print("Video saved successfully!")
