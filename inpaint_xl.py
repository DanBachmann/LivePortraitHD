import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
import transformers
if not hasattr(transformers, 'EncoderDecoderCache'):
    class DummyCache:
        pass
    transformers.EncoderDecoderCache = DummyCache

import torch
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, DPMSolverMultistepScheduler
from diffusers.utils import load_image
import os
import argparse
from pathlib import Path

def load_ai_pipeline(diffusion_model, load_ip_adapter=False):
    # 1. The Geometry Guide (SDXL Tile ControlNet)
# 1. The Geometry Guide (SDXL Tile ControlNet)
    print("Loading ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        "xinsir/controlnet-tile-sdxl-1.0", 
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    print(f"Loading Base Model: {diffusion_model}...")
    pipe = None
    if diffusion_model.lower().endswith(".safetensors"):    
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
            diffusion_model,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None
        )
    else:
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            diffusion_model, 
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None
        )
    
    if load_ip_adapter:
        # 3. The Texture Injection (SDXL IP-Adapter)
        print("Loading IP-Adapter...")
        pipe.load_ip_adapter(
            "h94/IP-Adapter", 
            subfolder="sdxl_models", 
            weight_name="ip-adapter_sdxl.bin"
        )
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, 
        use_karras_sigmas=True
    )    

    # Removed the SD1.5 Detail Tweaker LoRA as it is incompatible with SDXL.
    # SDXL natively produces much higher detail at 1024x1024 anyway.
    
    print("Applying VRAM Optimizations for 8GB GPU...")
    # CRITICAL VRAM optimizations for 8GB cards running SDXL
    pipe.enable_model_cpu_offload() 
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    
    # Optional: Enable xformers if installed for further memory reduction
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print("Xformers not enabled. Continuing without it.")
        
    return pipe

def synthesize_patch(pipe, inference_steps, region_name, image_paths, mask_paths, ip_image_in):
    """Processes the patch using dual-guidance (IP-Adapter + ControlNet)."""
    batch_size = len(image_paths)
    
    init_images = [load_image(path) for path in image_paths]
    mask_images = [load_image(path) if path else None for path in mask_paths]
        
    strength = 0.4
    guidance_scale = 6.0
    controlnet_conditioning_scale = 0.85
    negative_prompt = "muscular, cartoon, 3d render, spots, dots, speckled, bumps, disease, saliva, wet, bubbles, excessive specular highlights, white noise, textured tongue, plastic, blurry, yellow reflection, colored lights, unnatural specular highlights, low quality, low resolution, jpeg artifacts, overexposed, underexposed, distorted, unrealistic"
    
    if "mouth" in region_name.lower():
        prompt = "photorealistic lips, realistic teeth, high quality, 8k resolution"
        ip_adapter_scale = 0.2
        strength = 0.5
    elif "eye" in region_name.lower():
        prompt = "photorealistic eyes, high quality, 8k resolution"
        ip_adapter_scale = 0.6
        strength = 0.7
    else:
        prompt = "photorealistic human skin texture, high quality, 8k resolution"

    if ip_image_in:
        if not isinstance(ip_image_in, list):
            ip_images = [ip_image_in] * batch_size
        else:
            ip_images = ip_image_in
    else:
        ip_images = init_images
        ip_adapter_scale = 0.0
        
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    output = pipe(
        prompt=[prompt] * batch_size,
        negative_prompt=[negative_prompt] * batch_size,
        image=init_images,            
        mask_image=mask_images,
        control_image=init_images,    
        ip_adapter_image=[ip_images],   
        num_inference_steps=inference_steps,      
        strength=strength,               
        controlnet_conditioning_scale=controlnet_conditioning_scale, 
        guidance_scale=guidance_scale,   
    ).images
    return output

def batch_data(data_list, batch_size):
    """Yields successive n-sized chunks from a list."""
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]
    
def do_inpainting(diffusion_model, source_file_path, driver_file_path):
    # 1. Initialize the pipeline
    pipe = load_ai_pipeline(diffusion_model, True)
    output_folder = "output/tmp"

    source_name = Path(source_file_path).stem
    driving_name = Path(driver_file_path).stem
    file_label = f"{source_name}-{driving_name}"
    
    ip_eye_image = load_image(f"{output_folder}/eye_reference-{source_name}.png")
    ip_mouth_image = load_image(f"{output_folder}/mouth_reference-{source_name}.png")

    if driver_file_path.endswith(".mp4"):
        import cv2
        cap = cv2.VideoCapture(f"animations/{source_name}--{driving_name}.mp4")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # FORCED BATCH SIZE 1 FOR 8GB VRAM
        batch_size = 1 
        
        patches_mouth = []
        patches_eye = []
        
        for i in range(0, total_frames):
            frame_label = f"{file_label}-{i:04d}"            
            patches_mouth.append({
                    "name": "Mouth_Cavity",
                    "image_in": f"{output_folder}/04_plate_4K_Warp_mouth-{frame_label}.png",
                    "mask_in": f"{output_folder}/03_plate_Alpha_Mask_mouth-{frame_label}.png",
                    "output": f"{output_folder}/05_synthesized_patch_mouth-{frame_label}.png"
                })
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
            print(f"\nProcessing {image_paths[0]} for {region_name}...")
            outputs = synthesize_patch(
                pipe=pipe,
                inference_steps=18,
                region_name=region_name,
                image_paths=image_paths, 
                mask_paths=mask_paths, 
                ip_image_in=ip_mouth_image)
            for out_img, out_path in zip(outputs, output_paths):
                print(f"Saving {out_path}...")
                out_img.save(out_path)
                
        for batch in batch_data(patches_eye, batch_size):
            image_paths = [p["image_in"] for p in batch]
            mask_paths = [p["mask_in"] for p in batch]
            output_paths = [p["output"] for p in batch]
            region_name = batch[0]["name"]
            print(f"\nProcessing {image_paths[0]} for {region_name}...")
            outputs = synthesize_patch(
                pipe=pipe,
                inference_steps=18,
                region_name=region_name,
                image_paths=image_paths, 
                mask_paths=mask_paths, 
                ip_image_in=ip_eye_image)
            for out_img, out_path in zip(outputs, output_paths):
                print(f"Saving {out_path}...")
                out_img.save(out_path)
    else:
        patches = [
            {
                "name": "Mouth_Cavity",
                "image_in": f"{output_folder}/04_plate_4K_Warp_mouth-{file_label}.png",
                "mask_in": f"{output_folder}/03_plate_Alpha_Mask_mouth-{file_label}.png",
                "output": f"{output_folder}/05_synthesized_patch_mouth-{file_label}.png"
            },
            {
                "name": "Eye_Band",
                "image_in": f"{output_folder}/04_plate_4K_Warp_eyes-{file_label}.png",
                "mask_in": f"{output_folder}/03_plate_Alpha_Mask_eyes-{file_label}.png",
                "output": f"{output_folder}/05_synthesized_patch_eyes-{file_label}.png"
            }
        ]    
        for patch in patches:
            if os.path.exists(patch["image_in"]):
                output = synthesize_patch(
                    pipe=pipe,
                    inference_steps=22,
                    region_name=patch["name"],
                    image_paths=[patch["image_in"]], 
                    mask_paths=[patch["mask_in"]], 
                    ip_image_in=ip_eye_image if "eye" in patch["name"].lower() else ip_mouth_image
                )
                output[0].save(patch["output"])
            else:
                print(f"\n[WARNING] Skipping {patch['name']}: Could not find input files. Please check the paths.")


if __name__ == "__main__":
    # Pointed to your downloaded SDXL Safetensors file
    # UPDATE THIS PATH if it is not in the same directory as the script
    diffusion_model = "S:/mods/SDXL/cyberrealisticXL_v100.safetensors"

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, help="The source argument value")
    parser.add_argument('-d', type=str, help="The driver argument value")
    args, unknown_args = parser.parse_known_args()
    
    if not args.s:
        print("No -s source file argument was provided.")
        exit(1)
    source_file_path = args.s
    
    if not args.d:
        print("No -d driver file argument was provided.")
        exit(1)
    driver_file_path = args.d
    
    do_inpainting(diffusion_model, source_file_path, driver_file_path)
            
    print("Inpaint synthesis finished.")
    