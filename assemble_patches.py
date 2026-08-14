import cv2
import numpy as np
import json
import os
import argparse
from pathlib import Path
from PIL import Image

def add_noise(image_input, intensity_percent=1.0, monochrome=False):
    is_pil = False

    # 1. Handle pipeline input types and safely extract Alpha if it exists
    if isinstance(image_input, str):
        # IMREAD_UNCHANGED forces OpenCV to load the alpha channel if the file has one
        cv_img = cv2.imread(image_input, cv2.IMREAD_UNCHANGED)
    elif isinstance(image_input, Image.Image):
        is_pil = True
        if image_input.mode == 'RGBA':
            cv_img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGBA2BGRA)
        else:
            cv_img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        cv_img = image_input.copy()
    else:
        raise ValueError("Unsupported image input type.")

    h, w, c = cv_img.shape

    # 2. Isolate the RGB/BGR channels from the Alpha channel
    has_alpha = (c == 4)
    if has_alpha:
        alpha_channel = cv_img[:, :, 3]
        color_img = cv_img[:, :, :3]
    else:
        color_img = cv_img

    # Calculate the maximum pixel value offset
    max_delta = (intensity_percent / 100.0) * 255.0

    # 3. Generate Noise strictly for the 3 color channels
    if monochrome:
        noise_channel = np.random.uniform(-max_delta, max_delta, (h, w, 1)).astype(np.float32)
        noise = np.repeat(noise_channel, 3, axis=2)
    else:
        noise = np.random.uniform(-max_delta, max_delta, (h, w, 3)).astype(np.float32)

    # 4. Add noise using float32 to prevent standard 8-bit clipping errors
    noisy_color = color_img.astype(np.float32) + noise
    noisy_color = np.clip(noisy_color, 0, 255).astype(np.uint8)

    # 5. Re-attach the untouched Alpha channel
    if has_alpha:
        # Stack the 3 noisy color channels with the 1 pristine alpha channel
        final_img = np.dstack((noisy_color, alpha_channel))
    else:
        final_img = noisy_color

    # 6. Return to PIL Image to match your pipeline's synthesis outputs
    if is_pil:
        if has_alpha:
            return Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_BGRA2RGBA))
        else:
            return Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))

    return final_img

def combine_images(output_path, layer_data):
    """
    Combines multiple images into a single flattened image.

    :param output_path: Where to save the final image (e.g., 'output.png').
    :param layer_data: A list of tuples containing (image_path, optional_mask_path).
    """
    if not layer_data:
        print("Error: No image layers provided.")
        return

    # Initialize a transparent base canvas using the dimensions of the first image
    first_img_input = layer_data[0][0]
    first_img = Image.open(first_img_input) if isinstance(first_img_input, str) else first_img_input
    canvas = Image.new("RGBA", first_img.size, (0, 0, 0, 0))

    def add_image_to_canvas(base_canvas, img_input, mask_input=None):
        # 1. Load the main image
        if isinstance(img_input, str):
            img = Image.open(img_input)
        else:
            img = img_input.copy()

        active_mask = None

        # 2. Determine Mask Logic
        if mask_input is not None:
            if isinstance(mask_input, str):
                active_mask = Image.open(mask_input).convert("L")
            else:
                active_mask = mask_input.convert("L")
            img = img.convert("RGB")

        elif img.mode in ('RGBA', 'LA') or len(img.getbands()) == 4:
            active_mask = img.split()[-1]
            img = img.convert("RGB")

        else:
            img = img.convert("RGB")

        # 3. Handle resizing
        if img.size != base_canvas.size:
            img = img.resize(base_canvas.size, Image.Resampling.LANCZOS)
            if active_mask:
                active_mask = active_mask.resize(base_canvas.size, Image.Resampling.LANCZOS)

        # 4. Paste onto the canvas
        base_canvas.paste(img, (0, 0), active_mask)
        return base_canvas

    for data in layer_data:
        img_in = data[0]
        mask_in = data[1] if len(data) > 1 else None
        canvas = add_image_to_canvas(canvas, img_in, mask_in)

    if output_path.lower().endswith(('.jpg', '.jpeg')):
        canvas = canvas.convert("RGB")

    canvas.save(output_path)
    print(f"Successfully saved combined image to: {output_path}")


def paste_patch_with_bounds(canvas, patch, center_x, center_y, target_w, target_h):
    """
    Resizes the patch, converts it to RGBA if necessary, and pastes it
    onto the transparent canvas using the center coordinates.
    """
    current_h, current_w = patch.shape[:2]
    if current_w != target_w or current_h != target_h:
        print(f" -> Resizing patch from {current_w}x{current_h} to metadata spec: {target_w}x{target_h}")
        patch = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    if patch.shape[2] == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2BGRA)

    half_w = target_w // 2
    half_h = target_h // 2
    canvas_h, canvas_w = canvas.shape[:2]

    y1_canvas = max(0, center_y - half_h)
    y2_canvas = min(canvas_h, center_y + half_h)
    x1_canvas = max(0, center_x - half_w)
    x2_canvas = min(canvas_w, center_x + half_w)

    y1_patch = max(0, half_h - center_y)
    y2_patch = y1_patch + (y2_canvas - y1_canvas)
    x1_patch = max(0, half_w - center_x)
    x2_patch = x1_patch + (x2_canvas - x1_canvas)

    canvas[y1_canvas:y2_canvas, x1_canvas:x2_canvas] = patch[y1_patch:y2_patch, x1_patch:x2_patch]
    return canvas


def fade_patch_corners(patch, corner_radius=200, blur_size=151, left_fade=0, right_fade=0, top_fade=0, bottom_fade=0):
    """
    Fades patch edges using precise, distance-based smoothstep ramps starting directly
    from the outer borders, eliminating Mach-banding (hard edges) at the transition points.
    """
    h, w = patch.shape[:2]

    # Ensure patch has an alpha channel
    if patch.shape[2] == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2BGRA)

    # 1. Base Geometry: Optional rounded top corners
    if corner_radius > 0:
        corner_mask_u8 = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(corner_mask_u8, (0, corner_radius), (w, h), 255, -1)
        cv2.rectangle(corner_mask_u8, (corner_radius, 0), (w - corner_radius, corner_radius), 255, -1)
        cv2.circle(corner_mask_u8, (corner_radius, corner_radius), corner_radius, 255, -1)
        cv2.circle(corner_mask_u8, (w - corner_radius, corner_radius), corner_radius, 255, -1)

        # Smooth only the top rounded corner cutouts if blur_size is specified
        if blur_size > 0:
            if blur_size % 2 == 0:
                blur_size += 1
            corner_mask_u8 = cv2.GaussianBlur(corner_mask_u8, (blur_size, blur_size), 0)

        base_mask = corner_mask_u8.astype(np.float32) / 255.0
    else:
        base_mask = np.ones((h, w), dtype=np.float32)

    def smoothstep(t):
        # 3t^2 - 2t^3: Eases in from 0 and eases out into 1
        return t * t * (3.0 - 2.0 * t)

    # 2. Precise Directional Ramps (0.0 at outer edge -> 1.0 at fade_distance)
    y_indices = np.arange(h, dtype=np.float32)
    x_indices = np.arange(w, dtype=np.float32)

    # Apply the linear clip, then pass it through the Smoothstep function
    top_ramp = smoothstep(np.clip(y_indices / top_fade, 0.0, 1.0))[:, None] if top_fade > 0 else 1.0
    bottom_ramp = smoothstep(np.clip((h - 1 - y_indices) / bottom_fade, 0.0, 1.0))[:, None] if bottom_fade > 0 else 1.0
    left_ramp = smoothstep(np.clip(x_indices / left_fade, 0.0, 1.0))[None, :] if left_fade > 0 else 1.0
    right_ramp = smoothstep(np.clip((w - 1 - x_indices) / right_fade, 0.0, 1.0))[None, :] if right_fade > 0 else 1.0

    # Combine edge ramps (multiplication creates smooth 2D falloffs in corners)
    edge_gradient = top_ramp * bottom_ramp * left_ramp * right_ramp

    # 3. Combine corner geometry with directional edge gradients
    final_gradient = base_mask * edge_gradient

    # 4. Multiply gradient against the patch's existing alpha channel
    current_alpha = patch[:, :, 3].astype(np.float32) / 255.0
    patch[:, :, 3] = np.clip(current_alpha * final_gradient * 255.0, 0, 255).astype(np.uint8)

    return patch


def add_patch(canvas, patch_data, heavy_fade, alpha_path, patch_path):
    # Load with UNCHANGED to preserve any existing alpha channel
    if not os.path.exists(patch_path):
        raise FileNotFoundError(f"Missing file: {patch_path}. Run the extraction script first.")
    # add some noise for the illusion of texture on the inpainting to help with the blending with the other details
    inpainted_patch = add_noise(cv2.imread(patch_path, cv2.IMREAD_UNCHANGED), intensity_percent=2, monochrome=True)
    if inpainted_patch.shape[2] == 3 and alpha_path:
        alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
        if alpha is None:
            raise FileNotFoundError(f"CRITICAL: Missing external alpha mask at '{alpha}'")
        b, g, r = cv2.split(inpainted_patch)
        inpainted_patch = cv2.merge((b, g, r, alpha))

    if heavy_fade:
        inpainted_patch = fade_patch_corners(inpainted_patch, corner_radius=220, blur_size=151,
                                            left_fade=80, right_fade=80,
                                            top_fade=120, bottom_fade=120)
    else:
        inpainted_patch = fade_patch_corners(inpainted_patch, corner_radius=220, blur_size=151,
                                            left_fade=70, right_fade=70,
                                            top_fade=120, bottom_fade=50)

    canvas = paste_patch_with_bounds(
        canvas=canvas,
        patch=inpainted_patch,
        center_x=patch_data["center_x"],
        center_y=patch_data["center_y"],
        target_w=patch_data["width"],
        target_h=patch_data["height"]
    )
    print(f" -> added at canvas center {patch_path} ({patch_data['center_x']}, {patch_data})")
    return canvas

def assemble_patches(file_label, heavy_fade_eyes = True, heavy_fade_mouth = True, inpaint_eyes = True, inpaint_mouth = True, use_alpha_eyes = True, use_alpha_mouth = True):
    output_path = "output/tmp"
    json_path = f"{output_path}/metadata/patch_metadata-{file_label}.json"
    eyes_patch_path = f"{output_path}/05_synthesized_patch_eyes-{file_label}.png" if inpaint_eyes else None
    mouth_patch_path = f"{output_path}/05_synthesized_patch_mouth-{file_label}.png" if inpaint_mouth else None

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing metadata file: {json_path}. Run the extraction script first.")

    print(f"Loading metadata from {json_path}...")
    with open(json_path, "r") as f:
        metadata = json.load(f)

    canvas = np.zeros((4096, 4096, 4), dtype=np.uint8)

    # Reassemble Eyes Patch
    if inpaint_eyes and metadata.get("eyes") and os.path.exists(eyes_patch_path):
        canvas = add_patch(canvas, metadata["eyes"], heavy_fade_eyes, f"{output_path}/03_plate_Alpha_Mask_eyes-{file_label}.png" if use_alpha_eyes else None, eyes_patch_path)
    else:
        print("[SKIPPED] Eye patch skipped")

    # Reassemble Mouth Patch
    if inpaint_mouth and metadata.get("mouth") and os.path.exists(mouth_patch_path):
        canvas = add_patch(canvas, metadata["mouth"], heavy_fade_mouth, f"{output_path}/03_plate_Alpha_Mask_mouth-{file_label}.png" if use_alpha_mouth else None, mouth_patch_path)
    else:
        print("[SKIPPED] Mouth patch skipped")

    # Save the final reassembled 4K map - we don't need to do this anymore
    output_canvas_path = f"{output_path}/06_reassembled_ai_plate-{file_label}.png"
    os.makedirs(os.path.dirname(output_canvas_path), exist_ok=True)
    cv2.imwrite(output_canvas_path, canvas)
    print(f"Reassembled transparent 4K plate saved to: {output_canvas_path}")

    return canvas


def local_color_transfer(source_4k, target_lp, kernel_size=15):
    """
    Forces the LivePortrait base to match the local color and lighting
    gradients of the 4K plate using Gaussian blurring, without hard seams.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    src_f = source_4k.astype(np.float32)
    tgt_f = target_lp.astype(np.float32)

    src_blur = cv2.GaussianBlur(src_f, (kernel_size, kernel_size), 0)
    tgt_blur = cv2.GaussianBlur(tgt_f, (kernel_size, kernel_size), 0)

    epsilon = 1e-5
    local_ratio = src_blur / (tgt_blur + epsilon)

    matched_lp = tgt_f * local_ratio

    return np.clip(matched_lp, 0, 255).astype(np.uint8)

def straight_alpha_composite(img_4k, img_inpainted_rgba, blur_sigma=15, lf_tolerance=0.05):
    # 1. Split Alpha and RGB (Assuming input is already unmultiplied)
    alpha = img_inpainted_rgba[:, :, 3].astype(np.float32) / 255.0
    img_inpainted = img_inpainted_rgba[:, :, :3].astype(np.float32)
    img_4k = img_4k.astype(np.float32)

    # 2. Detail Layer Selection (The HF sanity check)
    base_4k = cv2.GaussianBlur(img_4k, (0, 0), sigmaX=blur_sigma)
    base_inpaint = cv2.GaussianBlur(img_inpainted, (0, 0), sigmaX=blur_sigma)
    detail_4k = img_4k - base_4k
    detail_inpaint = img_inpainted - base_inpaint
    hf_selection = (np.mean(np.abs(detail_4k), axis=2) > np.mean(np.abs(detail_inpaint), axis=2)).astype(np.float32)

    # 3. Straight Alpha Blend
    # This ignores the 'fringing' and just performs a clean cut.
    # We expand the alpha slightly to ensure the AI 'repair' covers the warp tear completely.
    alpha_kernel = np.ones((5,5), np.uint8)
    alpha = cv2.dilate(alpha, alpha_kernel, iterations=1)

    # Final Composite using straight alpha interpolation
    final_img = (img_inpainted * alpha[..., None]) + (img_4k * (1 - alpha[..., None]))

    return final_img.astype(np.uint8)

def assemble_and_warp_simple1(source_img, patches, skin_rgba_path, M_c2o_512, final_output_path):
    skin_rgb = cv2.imread(skin_rgba_path, cv2.IMREAD_UNCHANGED)[:, :, :3]
    print("skin_rgb.shape",skin_rgb.shape)
    if patches is not None and patches.shape[2] == 4:
        alpha_patches = patches[:, :, 3] / 255.0
        print("alpha_patches.shape",alpha_patches.shape)
        alpha_patches = np.expand_dims(alpha_patches, axis=-1)
        print("alpha_patches.shape",alpha_patches.shape)
        sandwich = (skin_rgb * (1 - alpha_patches) + patches[:, :, :3] * alpha_patches).astype(np.uint8)
        warp_sandwich(source_img, sandwich, M_c2o_512, final_output_path)
    else:
        warp_sandwich(source_img, skin_rgb, M_c2o_512, final_output_path)

def assemble_and_warp_simple2(source_img, patches, skin_rgba_path, M_c2o_512, final_output_path):
    background_rgb = cv2.imread(skin_rgba_path, cv2.IMREAD_UNCHANGED)[:, :, :3]
    # 1. Extract the RGB channels and the Alpha channel from the foreground
    fg_rgb = patches[:, :, :3]
    alpha_channel = patches[:, :, 3]
    # 2. Normalize the alpha channel to a float between 0.0 and 1.0
    # Expand dimensions so it can broadcast mathematically against the 3 color channels
    alpha_f = alpha_channel.astype(np.float32) / 255.0
    alpha_f = np.expand_dims(alpha_f, axis=-1)
    # 3. Convert image arrays to float32 to prevent 8-bit math overflow/clipping
    bg_f = background_rgb.astype(np.float32)
    fg_f = fg_rgb.astype(np.float32)
    # 4. The Alpha Blending Equation:
    # Output = (Foreground * Alpha) + (Background * (1.0 - Alpha))
    composited_f = (fg_f * alpha_f) + (bg_f * (1.0 - alpha_f))
    # 5. Clip the results to the valid 0-255 range and lock back to 8-bit uint8
    composited_8u = np.clip(composited_f, 0, 255).astype(np.uint8)
    warp_sandwich(source_img, composited_8u, M_c2o_512, final_output_path)

def assemble_laplacian(patches, skin_path):
    """
    Assembles foreground patches onto the background skin plate using
    Laplacian Pyramid Blending in the LAB color space. This eliminates
    RGB saturation banding and preserves true skin tones at transition seams.
    """
    background_rgb = cv2.imread(skin_path, cv2.IMREAD_UNCHANGED)[:, :, :3]

    # 1. Extract RGB and Alpha from the foreground (Source)
    fg_rgb = patches[:, :, :3]
    mask_8u = patches[:, :, 3]

    # Quick optimization: If mask is entirely empty, return background
    if cv2.countNonZero(mask_8u) == 0:
        return background_rgb.copy()

    # 2. Convert to float32 [0.0, 1.0] and shift to LAB color space
    # LAB separates Lightness from Color, preventing weird saturation spikes during blending
    bg_f = background_rgb.astype(np.float32) / 255.0
    fg_f = fg_rgb.astype(np.float32) / 255.0

    bg_lab = cv2.cvtColor(bg_f, cv2.COLOR_BGR2Lab)
    fg_lab = cv2.cvtColor(fg_f, cv2.COLOR_BGR2Lab)

    # Convert the mask to a 3-channel image to preserve depth during pyrDown
    mask_3c = cv2.cvtColor(mask_8u, cv2.COLOR_GRAY2BGR)
    mask_f = mask_3c.astype(np.float32) / 255.0

    # 3. Reduce pyramid depth from 6 to 4 to prevent the lip color from spreading too far
    num_levels = 4

    # 4. Build Gaussian Pyramids
    gp_fg = [fg_lab]
    gp_bg = [bg_lab]
    gp_mask = [mask_f]

    for i in range(num_levels):
        gp_fg.append(cv2.pyrDown(gp_fg[-1]))
        gp_bg.append(cv2.pyrDown(gp_bg[-1]))
        gp_mask.append(cv2.pyrDown(gp_mask[-1]))

    # 5. Build Laplacian Pyramids for Foreground and Background
    lp_fg = [gp_fg[-1]]
    lp_bg = [gp_bg[-1]]

    for i in range(num_levels - 1, -1, -1):
        size = (gp_fg[i].shape[1], gp_fg[i].shape[0])
        L_fg = gp_fg[i] - cv2.pyrUp(gp_fg[i+1], dstsize=size)
        L_bg = gp_bg[i] - cv2.pyrUp(gp_bg[i+1], dstsize=size)
        lp_fg.append(L_fg)
        lp_bg.append(L_bg)

    # 6. Blend the Laplacian Pyramids
    LS = []
    for i in range(num_levels + 1):
        mask_level = gp_mask[num_levels - i]
        blended_level = (lp_fg[i] * mask_level) + (lp_bg[i] * (1.0 - mask_level))
        LS.append(blended_level)

    # 7. Reconstruct the final image
    reconstructed_lab = LS[0]
    for i in range(1, num_levels + 1):
        size = (LS[i].shape[1], LS[i].shape[0])
        reconstructed_lab = cv2.pyrUp(reconstructed_lab, dstsize=size)
        reconstructed_lab = cv2.add(reconstructed_lab, LS[i])

    # 8. Convert back to BGR and clip to 8-bit
    reconstructed_bgr = cv2.cvtColor(reconstructed_lab, cv2.COLOR_Lab2BGR)
    composited_8u = np.clip(reconstructed_bgr * 255.0, 0, 255).astype(np.uint8)

    return composited_8u


def assemble_laplacian_rgb(patches, skin_path):
    """
    Assembles foreground patches onto the background skin plate using
    Laplacian Pyramid Blending in the LAB color space. This eliminates
    RGB saturation banding and preserves true skin tones at transition seams.
    """
    background_rgb = cv2.imread(skin_path, cv2.IMREAD_UNCHANGED)[:, :, :3]

    # 1. Extract RGB and Alpha from the foreground (Source)
    fg_rgb = patches[:, :, :3]
    mask_8u = patches[:, :, 3]

    # Quick optimization: If mask is entirely empty, return background
    if cv2.countNonZero(mask_8u) == 0:
        return background_rgb.copy()

    # 2. Convert to float32 [0.0, 1.0] and shift to LAB color space
    # LAB separates Lightness from Color, preventing weird saturation spikes during blending
    bg_f = background_rgb.astype(np.float32) / 255.0
    fg_f = fg_rgb.astype(np.float32) / 255.0

    bg_lab = cv2.cvtColor(bg_f, cv2.COLOR_BGR2Lab)
    fg_lab = cv2.cvtColor(fg_f, cv2.COLOR_BGR2Lab)

    # Convert the mask to a 3-channel image to preserve depth during pyrDown
    mask_3c = cv2.cvtColor(mask_8u, cv2.COLOR_GRAY2BGR)
    mask_f = mask_3c.astype(np.float32) / 255.0

    # 3. Reduce pyramid depth from 6 to 4 to prevent the lip color from spreading too far
    num_levels = 4

    # 4. Build Gaussian Pyramids
    gp_fg = [fg_lab]
    gp_bg = [bg_lab]
    gp_mask = [mask_f]

    for i in range(num_levels):
        gp_fg.append(cv2.pyrDown(gp_fg[-1]))
        gp_bg.append(cv2.pyrDown(gp_bg[-1]))
        gp_mask.append(cv2.pyrDown(gp_mask[-1]))

    # 5. Build Laplacian Pyramids for Foreground and Background
    lp_fg = [gp_fg[-1]]
    lp_bg = [gp_bg[-1]]

    for i in range(num_levels - 1, -1, -1):
        size = (gp_fg[i].shape[1], gp_fg[i].shape[0])
        L_fg = gp_fg[i] - cv2.pyrUp(gp_fg[i+1], dstsize=size)
        L_bg = gp_bg[i] - cv2.pyrUp(gp_bg[i+1], dstsize=size)
        lp_fg.append(L_fg)
        lp_bg.append(L_bg)

    # 6. Blend the Laplacian Pyramids
    LS = []
    for i in range(num_levels + 1):
        mask_level = gp_mask[num_levels - i]
        blended_level = (lp_fg[i] * mask_level) + (lp_bg[i] * (1.0 - mask_level))
        LS.append(blended_level)

    # 7. Reconstruct the final image
    reconstructed_lab = LS[0]
    for i in range(1, num_levels + 1):
        size = (LS[i].shape[1], LS[i].shape[0])
        reconstructed_lab = cv2.pyrUp(reconstructed_lab, dstsize=size)
        reconstructed_lab = cv2.add(reconstructed_lab, LS[i])

    # 8. Convert back to BGR and clip to 8-bit
    reconstructed_bgr = cv2.cvtColor(reconstructed_lab, cv2.COLOR_Lab2BGR)
    composited_8u = np.clip(reconstructed_bgr * 255.0, 0, 255).astype(np.uint8)

    return composited_8u

def assemble_simple(patches, skin_path):
    background_rgb = cv2.imread(skin_path, cv2.IMREAD_UNCHANGED)[:, :, :3]

    # 1. Extract RGB and Alpha from the foreground (Source)
    fg_rgb = patches[:, :, :3].copy()
    mask = patches[:, :, 3]  # Your alpha channel is now your binary mask

    composited_8u = background_rgb.copy()
    if cv2.countNonZero(mask) != 0:
        # Find distinct, separated shapes in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            # Skip microscopic noise
            if cv2.contourArea(cnt) < 50:
                continue
            # Create an isolated mask for just this one feature (e.g., just the mouth)
            single_mask = np.zeros_like(mask)
            cv2.drawContours(single_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            if cv2.countNonZero(single_mask) == 0:
                print("Warning: Mask is completely blank. Skipping seamless clone.")
                continue
            # Find the center of this specific feature's bounding box
            x, y, w_box, h_box = cv2.boundingRect(single_mask)
            center = (x + w_box // 2, y + h_box // 2)
            # Clone this feature directly onto the evolving background
            try:
                composited_8u = cv2.seamlessClone(fg_rgb, composited_8u, single_mask, center, cv2.NORMAL_CLONE)
            except cv2.error as e:
                print(f"OpenCV error during seamlessClone: {e}. Skipping. Mask may not have enough pixels to be cloned.")
                # cv2.imwrite(f"06_composited_8u{file_label}.png", composited_8u)
                # cv2.imwrite(f"06_fg_rgb{file_label}.png", fg_rgb)
                # cv2.imwrite(f"06_single_mask{file_label}.png", single_mask)
    return composited_8u

def assemble_and_warp_sandwich(source_img, base_ai_path, patches, skin_rgba_path, M_c2o_512, final_output_path, transfer_levels=False):
    skin_rgba = cv2.imread(skin_rgba_path, cv2.IMREAD_UNCHANGED)
    base_ai = cv2.imread(base_ai_path)
    if transfer_levels and skin_rgba is not None:
        base_ai = local_color_transfer(skin_rgba[:, :, :3], base_ai)
    # Build the Compositing Sandwich
    print("Stacking AI Patches over LivePortrait Base...")
    if patches is not None and patches.shape[2] == 4:
        alpha_patches = patches[:, :, 3] / 255.0
        alpha_patches = np.expand_dims(alpha_patches, axis=-1)
        sandwich = (base_ai * (1 - alpha_patches) + patches[:, :, :3] * alpha_patches).astype(np.uint8)
    else:
        sandwich = base_ai.copy()
    print("Applying 4K Skin Overlay with Difference Holes...")
    if skin_rgba is not None and skin_rgba.shape[2] == 4:
        alpha_skin = skin_rgba[:, :, 3] / 255.0
        alpha_skin = np.expand_dims(alpha_skin, axis=-1)
        sandwich = (sandwich * (1 - alpha_skin) + skin_rgba[:, :, :3] * alpha_skin).astype(np.uint8)
    warp_sandwich(source_img, sandwich, M_c2o_512, final_output_path)

def warp_sandwich(source_img, sandwich, M_c2o_512):
    source_h, source_w = source_img.shape[:2]
    # Scale Math (4096 to Original)
    CANVAS_SIZE = 4096
    MAX_DIM = 1280.0
    s_up = max(source_h, source_w) / MAX_DIM if max(source_h, source_w) > MAX_DIM else 1.0
    s_canvas = CANVAS_SIZE / 512.0

    M_c2o_highres = np.zeros((2, 3), dtype=np.float64)
    M_c2o_highres[:, 0:2] = M_c2o_512[:, 0:2] * (s_up / s_canvas)
    M_c2o_highres[:, 2] = M_c2o_512[:, 2] * s_up

    # 5. Warp the entire perfect sandwich back to the tilted camera angle
    print("Executing Affine Warp (Rotation, Scale, Translation)...")
    warped_sandwich = cv2.warpAffine(
        sandwich, M_c2o_highres, (source_w, source_h),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)
    )

    # Create a valid canvas mask
    # This prevents the black borders of the 4096 canvas from overwriting the original 4K background
    canvas_mask = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
    warped_mask = cv2.warpAffine(
        canvas_mask, M_c2o_highres, (source_w, source_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    warped_mask = np.expand_dims(warped_mask, axis=-1)

    # Paste onto Original Plate
    return (source_img * (1 - warped_mask) + warped_sandwich * warped_mask).astype(np.uint8)

def load_Mc2o(matrix_file_path):
    if not os.path.exists(matrix_file_path):
        print(f"\nCould not find Affine Matrix at {matrix_file_path}.")
        exit(1)
    M_c2o_loaded = np.loadtxt(matrix_file_path)
    if len(M_c2o_loaded.shape) == 1:
        M_c2o_loaded = M_c2o_loaded.reshape(-1, 3)
    return M_c2o_loaded[:2, :].copy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, help="The source argument value")
    parser.add_argument('-d', type=str, help="The driver argument value")
    parser.add_argument('-nip', type=bool, help="No in-painting")
    parser.add_argument('-over', type=bool, help="Overwrite some pre-existing assembled video frames")
    parser.add_argument('-startframe', type=int, default=0, help="Start frame")
    args, unknown_args = parser.parse_known_args()

    use_inpaint = not args.nip
    overwrite = args.over
    startframe = args.startframe
    if not args.s:
        print("No -s source file argument was provided.")
        exit(1)
    source_file_path = args.s
    if not args.d:
        print("No -d driver file argument was provided.")
        exit(1)
    driver_file_path = args.d

    M_c2o_512 = None
    source_img = None

    source_name = Path(source_file_path).stem
    driving_name = Path(driver_file_path).stem
    file_label = f"{source_name}-{driving_name}"
    output_file_suffix = 'hd' if use_inpaint else 'nip'
    output_dir_tmp = "output/tmp"

    if not source_file_path.lower().endswith(".mp4"):
        matrix_file_path = f"{output_dir_tmp}/metadata/M_c2o-{file_label}.txt"
        M_c2o_512 = load_Mc2o(matrix_file_path)
        if M_c2o_512 is None:
            print(f"Could not find matrix_c2o file {matrix_file_path}")
        source_img=cv2.imread(source_file_path)
        if source_img is None:
            print(f"Could not find source image {source_file_path}")

    os.makedirs(f"{output_dir_tmp}/animations", exist_ok=True)

    if driver_file_path.lower().endswith(".mp4"):
        import cv2
        import subprocess
        cap = cv2.VideoCapture(f"animations/{source_name}--{driving_name}.mp4")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if source_file_path.lower().endswith(".mp4"):
            cap = cv2.VideoCapture(source_file_path)
        for i in range(startframe, total_frames):
            print(f"\nProcessing Frame {i:04d}...")
            frame_label = f"{file_label}-{i:04d}"
            final_output_filename = f"{output_dir_tmp}/animations/{frame_label}-{output_file_suffix}.png"
            if overwrite or not os.path.exists(final_output_filename):
                skin_path = f"{output_dir_tmp}/04_plate_4K_Warp_RGB-{frame_label}.tiff"
                if source_file_path.lower().endswith(".mp4"):
                    matrix_file_path = f"{output_dir_tmp}/metadata/M_c2o-{frame_label}.txt"
                    M_c2o_512 = load_Mc2o(matrix_file_path)
                    ret, source_img = cap.read()
                base_ai_path = f"{output_dir_tmp}/01_plate_AI_Base-{frame_label}.png"
                if use_inpaint:
                    patches = assemble_patches(frame_label, False)
                    #intermediate = assemble_simple(patches=patches, skin_path=skin_path)
                    intermediate = assemble_laplacian(patches=patches, skin_path=skin_path)
                else:
                    intermediate = cv2.imread(skin_path, cv2.IMREAD_UNCHANGED)[:, :, :3]
                final_composite = warp_sandwich(source_img, intermediate, M_c2o_512)
                cv2.imwrite(final_output_filename, final_composite)
                print(f"Final composite saved to: {final_output_filename}")
        print("\n--- Compiling still frame sequence to video ---")
        #cmd1 = f'ffmpeg -y -framerate {fps} -i "output/tmp/animations/{file_label}-%04d-hd.png" -vf "scale=-2:\'min(ih,2160)\'" -c:v libx264 -pix_fmt yuv420p "animations/{file_label}-4k.mp4"' # combine frames
        #cmd2 = f'ffmpeg -y -i "animations/{file_label}-4k.mp4" -i "{driver_file_path}" -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac "animations/{file_label}-4k_with_audio.mp4"' # add original audio
        cmd_combined = f'ffmpeg -y -framerate {fps} -i "{output_dir_tmp}/animations/{file_label}-%04d-{output_file_suffix}.png" -i "{driver_file_path}" -map 0:v:0 -map 1:a:0 -vf "scale=-2:\'min(ih,2160)\'" -c:v libx264 -pix_fmt yuv420p -c:a aac "animations/{file_label}-{output_file_suffix}.mp4"'
        subprocess.run(cmd_combined, shell=True)
        print("\n--- Generating Side-by-Side Comparison ---")
        cmd_comp = f'ffmpeg -y -i "animations/{source_name}--{driving_name}.mp4" -i "animations/{file_label}-{output_file_suffix}.mp4" -filter_complex "[0:v]scale=-2:2160[v0];[1:v]scale=-2:2160[v1];[v0][v1]hstack=inputs=2" -c:v libx264 "animations/{file_label}-{output_file_suffix}_comp.mp4"'
        subprocess.run(cmd_comp, shell=True)
    else:
        skin_path = f"{output_dir_tmp}/04_plate_4K_Warp_RGB-{file_label}.tiff"
        if use_inpaint:
            patches = assemble_patches(file_label) if use_inpaint else None
            #intermediate = assemble_simple(patches=patches, skin_path=skin_path)
            intermediate = assemble_laplacian(patches=patches, skin_path=skin_path)
            intermediate_output_filename = f"{output_dir_tmp}/06_assembled{file_label}.png"
            cv2.imwrite(intermediate_output_filename, intermediate)
        else:
            intermediate = cv2.imread(skin_path, cv2.IMREAD_UNCHANGED)[:, :, :3]
        final_composite = warp_sandwich(source_img, intermediate, M_c2o_512)
        final_output_filename = f"animations/{source_name}--{driving_name}-{output_file_suffix}.png"
        cv2.imwrite(final_output_filename, final_composite)
        print(f"Final composite saved to: {final_output_filename}")
