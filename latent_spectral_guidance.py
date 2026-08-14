import torch
import torch.fft

def apply_spectral_guidance(pred_z0, target_z0, cutoff_freq, device):
    """
    Applies frequency-domain guidance to the predicted clean latent (z0).
    Forces the low frequencies of the generation to match the blurry input,
    while allowing Stable Diffusion to hallucinate the high frequencies (lips, eyelashes).
    """
    # 1. Move both latents into the frequency domain using 2D Fast Fourier Transform
    # pred_z0: What SD thinks the final image will look like at step t
    # target_z0: The latent representation of your blurry crop (eyes/mouth)
    fft_pred = torch.fft.fftshift(torch.fft.fft2(pred_z0))
    fft_target = torch.fft.fftshift(torch.fft.fft2(target_z0))
    
    # 2. Create a 2D Low-Pass Filter Mask
    _, _, h, w = fft_pred.shape
    center_y, center_x = h // 2, w // 2
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    
    # Calculate distance from the center (low frequencies are in the center after fftshift)
    dist = torch.sqrt((X - center_x)**2 + (Y - center_y)**2).to(device)
    
    # Create mask: 1.0 for low frequencies (inside cutoff), 0.0 for high frequencies
    low_pass_mask = (dist <= cutoff_freq).float()
    
    # 3. Swap the Frequencies
    # Keep the structural low frequencies from your blurry input (the shape of the smile/eye)
    # Keep the textural high frequencies from SD's prediction (lip wrinkles, specular highlights)
    fft_blended = (fft_target * low_pass_mask) + (fft_pred * (1 - low_pass_mask))
    
    # 4. Inverse FFT back to the spatial latent domain
    fft_blended_unshifted = torch.fft.ifftshift(fft_blended)
    z0_guided = torch.fft.ifft2(fft_blended_unshifted).real
    
    return z0_guided

def custom_diffusion_loop(pipe, blurry_crop_image, mask_image, prompt_embeds, num_inference_steps=50):
    """
    Custom inference loop with Spectral Guidance and proper Inpainting Masking.
    
    prompt_embeds: This is where your TEXTURE guidance comes from! 
                   (e.g., "8k resolution, highly detailed skin pores")
    """
    device = pipe.device
    
    # 1. Encode Target & Mask
    # target_latent: The underlying structure we want to keep
    target_latent = encode_image_to_latent(pipe, blurry_crop_image)
    
    # Resize mask to match latent dimensions (usually 64x64 or 128x128)
    mask_latent = torch.nn.functional.interpolate(
        mask_image, size=target_latent.shape[-2:], mode="bilinear"
    ).to(device)
    
    # 2. Standard setup
    pipe.scheduler.set_timesteps(num_inference_steps)
    timesteps = pipe.scheduler.timesteps
    
    # Start with pure noise
    latent = torch.randn_like(target_latent) 
    
    for i, t in enumerate(timesteps):
        # A. Predict the noise residual using the UNet
        # Here is where the text prompt guides the TEXTURE generation
        with torch.no_grad():
            noise_pred = pipe.unet(latent, t, encoder_hidden_states=prompt_embeds).sample
            
        # B. Scheduler step
        step_output = pipe.scheduler.step(noise_pred, t, latent)
        latent = step_output.prev_sample 
        pred_z0 = step_output.pred_original_sample 
        
        # C. INJECT CUSTOM MATH (Spectral Guidance)
        # Force the low frequencies of the generation to match the blurry target
        if i < int(num_inference_steps * 0.8):
            guided_z0 = apply_spectral_guidance(
                pred_z0=pred_z0, 
                target_z0=target_latent, 
                cutoff_freq=8.0, 
                device=device
            )
            # Re-add noise to our guided structure to continue the loop
            latent = pipe.scheduler.add_noise(guided_z0, torch.randn_like(guided_z0), t)

        # D. APPLY INPAINTING MASK (The step that was missing)
        # We must also calculate what the original target image would look like at this exact noise level 't'
        target_noisy_latent = pipe.scheduler.add_noise(target_latent, torch.randn_like(target_latent), t)
        
        # Blend: Keep 'latent' inside the mask, force 'target_noisy_latent' outside the mask
        latent = (mask_latent * latent) + ((1.0 - mask_latent) * target_noisy_latent)
            
    # Decode final latent to image
    return decode_latent_to_image(pipe, latent)

# (Helper functions encode_image_to_latent and decode_latent_to_image omitted for brevity)