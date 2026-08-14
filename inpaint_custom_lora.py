import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
import transformers
if not hasattr(transformers, 'EncoderDecoderCache'):
    class DummyCache:
        pass
    transformers.EncoderDecoderCache = DummyCache

import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image
import os
import argparse
from pathlib import Path
import traceback
import sys

def load_ai_pipeline(diffusion_model, use_lora = True, load_ip_adapter = False):
    print("Initializing pipeline...")
    # 1. The Geometry Guide (Tile ControlNet)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1e_sd15_tile",
        torch_dtype=torch.float16
    )

    pipe = None
    try:
        if diffusion_model.lower().endswith(".safetensors"):
            pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
                diffusion_model,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None
            )
        else:
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                diffusion_model,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None
            )
    except Exception as e:
        print("\n[DIAGNOSTIC CRASH] Failed to load Stable Diffusion Pipeline:")
        traceback.print_exc()
        sys.exit(1)

    print("Configuring scheduler for high-definition (DPM++ 2M Karras)...")
    # We bypass the FrozenDict issues entirely by initializing a fresh DPM++ scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(dict(pipe.scheduler.config), use_karras_sigmas=True)

    print(f"load_ip_adapter")
    if load_ip_adapter:
        print(f"load_ip_adapter ip-adapter_sd15.bin")
        # 3. The Texture Injection (IP-Adapter)
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

    if use_lora:
        print(f"use_lora")
        mouth_lora_path = "lora/QMULHD_mouth_v2.safetensors"
        eye_lora_path = "lora/QMULHD_eyes_v2.safetensors"

        # 1. Bulletproof check: Ensure the files actually exist on the hard drive
        if not os.path.exists(mouth_lora_path):
            raise FileNotFoundError(f"\nCRITICAL ERROR: Could not find the Mouth LoRA at:\n{os.path.abspath(mouth_lora_path)}\nPlease verify the exact filename and folder location.")
        if not os.path.exists(eye_lora_path):
            raise FileNotFoundError(f"\nCRITICAL ERROR: Could not find the Eye LoRA at:\n{os.path.abspath(eye_lora_path)}\nPlease verify the exact filename and folder location.")

        # 2. Add local_files_only=True to completely block HuggingFace network fallbacks
        pipe.load_lora_weights(mouth_lora_path, weight_name="mouth_lora", adapter_name="mouth_lora", local_files_only=True)
        pipe.load_lora_weights(eye_lora_path, weight_name="eye_lora", adapter_name="eye_lora", local_files_only=True)
        # Disable them globally by default; we will activate them selectively in synthesize_patch
        pipe.set_adapters([])

    # Crucial for maintaining the VRAM margin
    pipe.enable_model_cpu_offload()

    return pipe

import torch
import torch.fft

class SpectralGuidanceCallback:
    def __init__(self, pipe, target_images, cutoff_freq=8.0, stop_step_ratio=0.8):
        self.pipe = pipe
        self.cutoff_freq = cutoff_freq
        self.stop_step_ratio = stop_step_ratio
        self.total_steps = None

        target_device = pipe._execution_device if hasattr(pipe, "_execution_device") else pipe.device
        image_tensor = pipe.image_processor.preprocess(target_images).to(target_device, dtype=pipe.dtype)

        with torch.no_grad():
            self.target_latents = pipe.vae.encode(image_tensor).latent_dist.sample()
            self.target_latents = self.target_latents * pipe.vae.config.scaling_factor

        self.target_latents = self.target_latents.to(target_device, dtype=pipe.dtype)

    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]

        if self.total_steps is None:
            self.total_steps = len(pipe.scheduler.timesteps)

        if step_index < int(self.total_steps * self.stop_step_ratio):
            batch_size = latents.shape[0]
            t_val = timestep.item() if hasattr(timestep, "item") else timestep
            ts = torch.tensor([t_val] * batch_size, device=latents.device, dtype=torch.long)

            noise = torch.randn_like(self.target_latents)
            target_noisy = pipe.scheduler.add_noise(self.target_latents, noise, ts)

            orig_dtype = latents.dtype
            latents_f32 = latents.to(torch.float32)
            target_noisy_f32 = target_noisy.to(torch.float32)

            fft_pred = torch.fft.fftshift(torch.fft.fft2(latents_f32))
            fft_target = torch.fft.fftshift(torch.fft.fft2(target_noisy_f32))

            _, _, h, w = fft_pred.shape
            center_y, center_x = h // 2, w // 2
            Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            dist = torch.sqrt((X - center_x)**2 + (Y - center_y)**2).to(latents.device)
            low_pass_mask = (dist <= self.cutoff_freq).float()

            fft_blended = (fft_target * low_pass_mask) + (fft_pred * (1.0 - low_pass_mask))
            latents = torch.fft.ifft2(torch.fft.ifftshift(fft_blended)).real.to(orig_dtype)

            callback_kwargs["latents"] = latents

        return callback_kwargs

def synthesize_patch(pipe, inference_steps, region_name, image_paths, mask_paths, ip_image_in, use_lora, use_spectral_hook, load_ip_adapter,
                     mouth_lora_weight=0.45, eye_lora_weight=0.45, lora_attn_scale = 1.0,
                     mouth_guidance_scale=7.0, eye_guidance_scale=6.0, mouth_strength=0.85, eye_strength=0.7, mouth_controlnet_scale=0.7, eye_controlnet_scale=0.85):
    """Processes the patch using dual-guidance (IP-Adapter + ControlNet) and Custom LoRAs."""
    batch_size = len(image_paths)

    init_images = [load_image(path) for path in image_paths]
    mask_images = [load_image(path) if path else None for path in mask_paths]

    strength=0.5
    guidance_scale=eye_guidance_scale
    controlnet_conditioning_scale=eye_controlnet_scale
    negative_prompt = "muscular, cartoon, 3d render, spots, dots, speckled, bumps, disease, saliva, wet, bubbles, excessive specular highlights, white noise, textured tongue, plastic, blurry, yellow reflection, colored lights, unnatural specular highlights, low quality, low resolution, jpeg artifacts, overexposed, underexposed, distorted, unrealistic, ribbed tongue, hairy tongue"

    # 1. ADDED TRIGGER WORDS AND DYNAMIC ADAPTER SWITCHING
    if "mouth" in region_name.lower():
        prompt = "QMULHD, photorealistic lips, realistic teeth, high quality, 8k resolution, mouth"
        ip_adapter_scale = 0.2
        strength = mouth_strength
        guidance_scale = mouth_guidance_scale
        controlnet_conditioning_scale=mouth_controlnet_scale
        if use_lora:
            # Activate the mouth LoRA for this batch
            pipe.set_adapters(["mouth_lora"], adapter_weights=[mouth_lora_weight])

    elif "eye" in region_name.lower():
        prompt = "QMULHD, photorealistic eyes, high quality, 8k resolution, sharp eyelashes"
        ip_adapter_scale = 0.6
        strength = eye_strength
        if use_lora:
            # Activate the eye LoRA for this batch
            pipe.set_adapters(["eye_lora"], adapter_weights=[eye_lora_weight])
    else:
        prompt = "QMULHD, photorealistic human skin texture, high quality, 8k resolution"
        if use_lora:
            pipe.set_adapters([]) # Turn off all LoRAs if neither

    pipe.set_ip_adapter_scale(ip_adapter_scale)

    pipe_kwargs = {
        "prompt": [prompt] * batch_size,
        "negative_prompt": [negative_prompt] * batch_size,
        "image": init_images,
        "mask_image": mask_images,
        "control_image": init_images,
        "num_inference_steps": inference_steps,
        "strength": strength,
        "controlnet_conditioning_scale": controlnet_conditioning_scale,
        "guidance_scale": guidance_scale,
        "cross_attention_kwargs": {"scale": lora_attn_scale} if use_lora else None,
    }

    if load_ip_adapter or getattr(pipe, "_qmul_ip_adapter_loaded", True):
        if ip_image_in:
            ip_images = [ip_image_in] * batch_size if not isinstance(ip_image_in, list) else ip_image_in
        else:
            ip_images = init_images
            ip_adapter_scale = 0.0

        pipe.set_ip_adapter_scale(ip_adapter_scale)
        pipe_kwargs["ip_adapter_image"] = [ip_images]

    if use_spectral_hook:
        spectral_hook = SpectralGuidanceCallback(
            pipe=pipe,
            target_images=init_images,
            cutoff_freq=8.0,
            stop_step_ratio=0.8,
        )
        pipe_kwargs["callback_on_step_end"] = spectral_hook
        pipe_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

    output = pipe(**pipe_kwargs).images
    return output


def batch_data(data_list, batch_size):
    """Yields successive n-sized chunks from a list."""
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]

def do_inpainting(diffusion_model, source_file_path, driver_file_path, 
                use_lora=False, load_ip_adapter=True, use_spectral_hook=True,
                mouth_lora_weight=0.45,
                eye_lora_weight=0.45,
                lora_attn_scale=1.0,
                mouth_guidance_scale=5.5, eye_guidance_scale=5.0,
                mouth_strength=0.7, eye_strength=0.55,
                mouth_controlnet_scale=0.8, eye_controlnet_scale=0.85,
                mouth_steps=30, eye_steps=22):
    print(f"Processing inpainting for {source_file_path} and {driver_file_path}")
    pipe = load_ai_pipeline(diffusion_model, use_lora=use_lora, load_ip_adapter=load_ip_adapter)
    print(f"Loaded diffusion model: {diffusion_model}")
    output_folder = "output/tmp"

    source_name = Path(source_file_path).stem
    driving_name = Path(driver_file_path).stem
    file_label = f"{source_name}-{driving_name}"
    print(f"Source: {source_file_path} | Driving: {driver_file_path} | Label: {file_label}")
    ip_eye_image = load_image(f"{output_folder}/eye_reference-{source_name}.png")
    print(f"\nLoaded IP-Adapter reference images for {source_name}:")
    ip_mouth_image = load_image(f"{output_folder}/mouth_reference-{source_name}.png")
    print(f"\nLoaded IP-Adapter reference images for {source_name}:")
    print(f"  - Eye reference: {ip_eye_image.size}")
    print(f"  - Mouth reference: {ip_mouth_image.size}")

    if driver_file_path.endswith(".mp4"):
        import cv2
        cap = cv2.VideoCapture(f"animations/{source_name}--{driving_name}.mp4")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_steps_multiplier = 1.0
        batch_size = 6
        # in case we only want to re-render just eyes or just mouth, don't initialize
        patches_mouth = []
        patches_eye = []
        pipe.enable_vae_slicing()
        for i in range(0,total_frames):
            frame_label = f"{file_label}-{i:04d}"
            if patches_mouth is not None:
                patches_mouth.append({
                        "name": "Mouth_Cavity",
                        "image_in": f"{output_folder}/04_plate_4K_Warp_mouth-{frame_label}.png",
                        "mask_in": f"{output_folder}/03_plate_Alpha_Mask_mouth-{frame_label}.png",
                        "output": f"{output_folder}/05_synthesized_patch_mouth-{frame_label}.png"
                    })
            if patches_eye is not None:
                patches_eye.append({
                        "name": "Eye_Band",
                        "image_in": f"{output_folder}/04_plate_4K_Warp_eyes-{frame_label}.png",
                        "mask_in": f"{output_folder}/03_plate_Alpha_Mask_eyes-{frame_label}.png",
                        "output": f"{output_folder}/05_synthesized_patch_eyes-{frame_label}.png"
                    })
        for batch in batch_data(patches_mouth, batch_size):
            image_paths = [p["image_in"] for p in batch]
            mask_paths = [p["mask_in"] for p in batch]
            output_paths = [p["output"] for p in batch]
            region_name = batch[0]["name"]
            print(f"\nProcessing {image_paths[0]} +{batch_size-1} out of {total_frames} for {region_name}...")
            outputs = synthesize_patch(
                pipe=pipe,
                inference_steps=int(mouth_steps * video_steps_multiplier),
                region_name=region_name,
                image_paths=image_paths,
                mask_paths=mask_paths,
                ip_image_in=ip_mouth_image,
                use_lora=use_lora,
                use_spectral_hook=use_spectral_hook,
                load_ip_adapter=load_ip_adapter,
                mouth_lora_weight=mouth_lora_weight,
                eye_lora_weight=eye_lora_weight,
                lora_attn_scale=lora_attn_scale,
                mouth_guidance_scale=mouth_guidance_scale,
                eye_guidance_scale=eye_guidance_scale,
                mouth_strength=mouth_strength,
                eye_strength=eye_strength,
                mouth_controlnet_scale=mouth_controlnet_scale,
                eye_controlnet_scale=eye_controlnet_scale,)
            for out_img, out_path in zip(outputs, output_paths):
                print(f"Saving {out_path}...")
                out_img.save(out_path)
        for batch in batch_data(patches_eye, batch_size):
            image_paths = [p["image_in"] for p in batch]
            mask_paths = [p["mask_in"] for p in batch]
            output_paths = [p["output"] for p in batch]
            region_name = batch[0]["name"]
            print(f"\nProcessing {image_paths[0]} +{batch_size-1} out of {total_frames} for {region_name}...")
            outputs = synthesize_patch(
                pipe=pipe,
                inference_steps=int(eye_steps * video_steps_multiplier),
                region_name=region_name,
                image_paths=image_paths,
                mask_paths=mask_paths,
                ip_image_in=ip_eye_image,
                use_lora=use_lora,
                use_spectral_hook=use_spectral_hook,
                load_ip_adapter=False, # override load_ip_adapter since metrics show it did not help eyes
                mouth_lora_weight=mouth_lora_weight,
                eye_lora_weight=eye_lora_weight,
                lora_attn_scale=lora_attn_scale,
                mouth_guidance_scale=mouth_guidance_scale,
                eye_guidance_scale=eye_guidance_scale,
                mouth_strength=mouth_strength,
                eye_strength=eye_strength,
                mouth_controlnet_scale=mouth_controlnet_scale,
                eye_controlnet_scale=eye_controlnet_scale,)
            for out_img, out_path in zip(outputs, output_paths):
                print(f"Saving {out_path}...")
                out_img.save(out_path)
    else:
        print(f"\nProcessing {source_file_path} + {driver_file_path} for inpainting...")
        patches = [
            {
                "name": "Mouth_Cavity",
                "inference_steps": mouth_steps,
                "image_in": f"{output_folder}/04_plate_4K_Warp_mouth-{file_label}.png",
                "mask_in": f"{output_folder}/03_plate_Alpha_Mask_mouth-{file_label}.png",
                "output": f"{output_folder}/05_synthesized_patch_mouth-{file_label}.png"
            },
            {
                "name": "Eye_Band",
                "inference_steps": eye_steps,
                "image_in": f"{output_folder}/04_plate_4K_Warp_eyes-{file_label}.png",
                "mask_in": f"{output_folder}/03_plate_Alpha_Mask_eyes-{file_label}.png",
                "output": f"{output_folder}/05_synthesized_patch_eyes-{file_label}.png"
            }
        ]
        for patch in patches:
            if os.path.exists(patch["image_in"]):
                is_eye_patch = "eye" in patch["name"].lower()
                output = synthesize_patch(
                    pipe=pipe,
                    inference_steps=patch["inference_steps"],
                    region_name=patch["name"],
                    image_paths=[patch["image_in"]],
                    mask_paths=[patch["mask_in"]],
                    ip_image_in=ip_eye_image if is_eye_patch else ip_mouth_image,
                    use_lora=use_lora,
                    use_spectral_hook=use_spectral_hook,
                    load_ip_adapter=False if is_eye_patch else load_ip_adapter, # override load_ip_adapter since metrics show it did not help eyes
                    mouth_lora_weight=mouth_lora_weight,
                    eye_lora_weight=eye_lora_weight,
                    lora_attn_scale=lora_attn_scale,
                    mouth_guidance_scale=mouth_guidance_scale,
                    eye_guidance_scale=eye_guidance_scale,
                    mouth_strength=mouth_strength,
                    eye_strength=eye_strength,
                    mouth_controlnet_scale=mouth_controlnet_scale,
                    eye_controlnet_scale=eye_controlnet_scale,)
                output[0].save(patch["output"])
                print(f" -> Saved {patch['output']}")
            else:
                print(f"\n[WARNING] Skipping {patch['name']}: Could not find input files. Please check the paths {patch['image_in']}.")

def test_img(diffusion_model):
    from diffusers import StableDiffusionPipeline
    pipe = None
    if diffusion_model.lower().endswith(".safetensors"):
        pipe = StableDiffusionPipeline.from_single_file(diffusion_model,
            torch_dtype=torch.float16,
            safety_checker=None
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(diffusion_model,
            torch_dtype=torch.float16,
            safety_checker=None
        )
    pipe.enable_model_cpu_offload()

    prompt = "a portrait of a university professor"
    negative_prompt = "cartoon, 3d render, plastic, blurry, reflection, colored lights, unnatural specular highlights, low quality, low resolution, jpeg artifacts, overexposed, underexposed, distorted, unrealistic"

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.0
    ).images[0]

    output.save("test_img.png")
    print("Success! Saved as test_img.png")


if __name__ == "__main__":
    # diffusion_model = "S:/mods/SD15/realisticVisionV60B1_v51VAE.safetensors"
    # diffusion_model = "S:/mods/SD15/cyberrealistic_final.safetensors"
    # diffusion_model = "https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1-inpainting.safetensors"
    diffusion_model = "S:/mods/SD15/cyberrealistic_v50-inpainting.safetensors"

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, help="The source argument value")
    parser.add_argument('-d', type=str, help="The driver argument value")
    parser.add_argument('-lora', type=bool, help="Use custom LoRA", default=False)
    parser.add_argument('-ipa', type=bool, help="Use IP-Adapter", default=True)
    parser.add_argument('-spectral', type=bool, help="Use spectral hook", default=True)
    args,unknown_args = parser.parse_known_args()
    if not args.s:
        print("No -s source file argument was provided.")
        exit(1)
    source_file_path = args.s
    if not args.d:
        print("No -d driver file argument was provided.")
        exit(1)
    driver_file_path = args.d

    load_ip_adapter = args.ipa
    use_lora = args.lora
    use_spectral_hook = args.spectral

    do_inpainting(diffusion_model, source_file_path, driver_file_path, use_lora=use_lora, load_ip_adapter=load_ip_adapter, use_spectral_hook=use_spectral_hook)

    print("Inpaint synthesis finished.")
