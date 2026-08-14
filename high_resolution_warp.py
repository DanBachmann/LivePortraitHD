import warnings
warnings.filterwarnings("ignore")
import argparse
import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
from src.utils.helper import is_video


def get_warp_groups():
    # --- [Keep Eyes, Brows, Lips arrays exactly the same] ---
    outer_lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40]
    inner_lips = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80]
    eye_corners = [33, 133, 362, 263]
    right_iris = [468, 469, 470, 471, 472]
    left_iris = [473, 474, 475, 476, 477]
    face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    # --- NEW: DECONSTRUCTED NOSE ---
    # The rigid upper bridge
    nose_bridge = [168, 195, 197, 4] 
    
    # The flexible tip and outer nostrils (Alar base)
    # 98 (Right nostril corner), 327 (Left nostril corner), 1, 2, 5 (Tip)
    nose_base = [1, 2, 98, 327, 64, 294, 5, 6] 

    # --- ASSEMBLE GROUPS ---
    moving_groups = [
        # Lips: Kept circular (isotropic), intensity boosted
        {'indices': outer_lips, 'sigma': (120.0, 120.0), 'intensity': 1.5},
        {'indices': inner_lips, 'sigma': (50.0, 50.0), 'intensity': 1.5},
        
        # Nostrils: Narrow horizontally so it doesn't grab cheeks, taller vertically to stretch
        {'indices': nose_base, 'sigma': (35.0, 65.0), 'intensity': 1.5}
    ]

    pinned_groups = [
        {'indices': right_iris + left_iris + eye_corners, 'sigma': (25.0, 25.0), 'intensity': 100.0},
        
        # Pin the bridge firmly so the upward pull stretches the flesh away from the bone
        {'indices': nose_bridge, 'sigma': (80.0, 80.0), 'intensity': 100.0},
        
        {'indices': face_oval, 'sigma': (300.0, 300.0), 'intensity': 100.0}
    ]

    return moving_groups, pinned_groups

def get_filtered_landmarks(image, target_indices=None):
    """
    Extracts specific MediaPipe landmarks from an image.
    Returns an (N, 2) array of pixel coordinates.
    """
    mp_face_mesh = mp.solutions.face_mesh
    
    # Initialize FaceMesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5
    ) as face_mesh:
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            print("Warning: No face detected in the provided image.")
            return None
            
        h, w = image.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        
        # If no specific indices are provided, return all 478 points
        if target_indices is None:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        else:
            pts = []
            for idx in target_indices:
                try:
                    lm = landmarks[idx]
                    pts.append((int(lm.x * w), int(lm.y * h)))
                except IndexError:
                    print(f"Warning: Landmark index {idx} not found. Skipping.")
                    return None
                    
        return np.array(pts, dtype=np.float32)

import torch
def calculate_gaussian_flow_backward_gpu(img_shape, src_pts, dst_pts, sigmas, intensities, device='cuda'):
    h, w = img_shape[:2]
    src = torch.tensor(np.array(src_pts), dtype=torch.float32, device=device)
    dst = torch.tensor(np.array(dst_pts), dtype=torch.float32, device=device)
    # Format sigmas for X and Y components
    sigmas_np = np.array(sigmas)
    # Fallback: if a standard circular float was passed, duplicate it for X and Y
    if len(sigmas_np.shape) == 1:
        sigmas_np = np.column_stack((sigmas_np, sigmas_np))
    sigmas_t = torch.tensor(sigmas_np, dtype=torch.float32, device=device)
    intensities_t = torch.tensor(np.array(intensities), dtype=torch.float32, device=device)
    vectors = src - dst
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing='ij'
    )
    flow_x = torch.zeros((h, w), dtype=torch.float32, device=device)
    flow_y = torch.zeros((h, w), dtype=torch.float32, device=device)
    sum_weight = torch.full((h, w), 1e-6, dtype=torch.float32, device=device)
    max_weight = torch.zeros((h, w), dtype=torch.float32, device=device)
    for i in range(len(src)):
        tx, ty = dst[i, 0], dst[i, 1]
        # THE FIX: Elliptical (Anisotropic) Distance Calculation
        # Divides the X distance by sigma_X, and the Y distance by sigma_Y
        dist_sq = ((x_coords - tx)**2 / (sigmas_t[i, 0] ** 2)) + ((y_coords - ty)**2 / (sigmas_t[i, 1] ** 2))
        weight = torch.exp(-dist_sq) * intensities_t[i]
        flow_x += weight * vectors[i, 0]
        flow_y += weight * vectors[i, 1]
        sum_weight += weight
        max_weight = torch.maximum(max_weight, weight)
    flow_x /= sum_weight
    flow_y /= sum_weight
    flow_x *= max_weight
    flow_y *= max_weight
    return torch.stack((flow_x, flow_y), dim=-1).cpu().numpy()


def apply_unified_landmark_warp(source_4k, target_lp, moving_groups, pinned_groups):
    flat_src = []
    flat_dst = []
    flat_sigmas = []
    flat_intensities = [] # NEW
    
    # YOUR IDEA: Blur the 4K plate so MediaPipe sees apples-to-apples
    blur_radius = 21 # Adjust if needed to match AI softness
    source_4k_soft = cv2.GaussianBlur(source_4k, (blur_radius, blur_radius), 0)
    
    # --- PROCESS MOVING FEATURES ---
    for group in moving_groups:
        indices = group['indices']
        sigma_val = group['sigma']
        intensity_val = group.get('intensity', 1.0)
        
        src_pts = get_filtered_landmarks(source_4k_soft, indices)
        dst_pts = get_filtered_landmarks(target_lp, indices)
        
        if src_pts is not None and dst_pts is not None:
            flat_src.extend(src_pts)
            flat_dst.extend(dst_pts)
            flat_sigmas.extend([sigma_val] * len(src_pts))
            flat_intensities.extend([intensity_val] * len(src_pts))
            
    # --- PROCESS PINNED FEATURES (THE ANCHORS) ---
    for group in pinned_groups:
        indices = group['indices']
        sigma_val = group['sigma']
        intensity_val = group.get('intensity', 100.0) # Massive density for anchors
        
        src_pts = get_filtered_landmarks(source_4k_soft, indices)
        
        if src_pts is not None:
            flat_src.extend(src_pts)
            flat_dst.extend(src_pts.copy()) 
            flat_sigmas.extend([sigma_val] * len(src_pts))
            flat_intensities.extend([intensity_val] * len(src_pts))
            
    if not flat_src:
        print("ERROR: No landmarks extracted.")
        return source_4k
        
    master_flow = calculate_gaussian_flow_backward_gpu(
        source_4k.shape, flat_src, flat_dst, flat_sigmas, flat_intensities
    )
    
    h, w = source_4k.shape[:2]
    center_x, center_y = w // 2, h // 2
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    
    dist_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
    dampening_mask = np.exp(-dist_sq / (1500**2)).astype(np.float32)
    
    master_flow[..., 0] *= dampening_mask
    master_flow[..., 1] *= dampening_mask
    
    remap_x = (x_coords + master_flow[..., 0]).astype(np.float32)
    remap_y = (y_coords + master_flow[..., 1]).astype(np.float32)
    
    warped_img = cv2.remap(
        source_4k, remap_x, remap_y, 
        interpolation=cv2.INTER_LANCZOS4, 
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return warped_img


def run_alignment_diagnostic(source_4k, target_lp, target_indices):
    # 1. Extract Landmarks
    src_pts = get_filtered_landmarks(source_4k, target_indices)
    dst_pts = get_filtered_landmarks(target_lp, target_indices)
    
    if src_pts is None or dst_pts is None:
        print("Diagnostic failed: Missing landmarks.")
        return
        
    # 2. Calculate Distances
    vectors = dst_pts - src_pts
    distances = np.linalg.norm(vectors, axis=1)
    
    max_dist = np.max(distances)
    avg_dist = np.mean(distances)
    
    print(f"Average Vector Movement: {avg_dist:.2f} pixels")
    print(f"Maximum Vector Movement: {max_dist:.2f} pixels")
    
    if avg_dist > 50:
        print("\n🚨 CRITICAL ERROR: The faces are not globally aligned.")
        print("The algorithm is trying to drag the face across the screen.")
    elif avg_dist < 2:
        print("\n⚠️ WARNING: The movement is almost zero. Is the squint actually there?")
        
    # 3. Draw the Overlay Map
    # We will use the 4K source image as our canvas and darken it so the dots pop out
    debug_canvas = cv2.addWeighted(source_4k, 0.3, np.zeros_like(source_4k), 0.7, 0)
    
    for i in range(len(src_pts)):
        p1 = (int(src_pts[i][0]), int(src_pts[i][1])) # Source (Where we are)
        p2 = (int(dst_pts[i][0]), int(dst_pts[i][1])) # Target (Where we want to go)
        
        # Source = RED
        cv2.circle(debug_canvas, p1, radius=4, color=(0, 0, 255), thickness=-1)
        # Target = BLUE
        cv2.circle(debug_canvas, p2, radius=4, color=(255, 0, 0), thickness=-1)
        # Connect them with a thin GREEN line
        cv2.line(debug_canvas, p1, p2, color=(0, 255, 0), thickness=1)

    cv2.imwrite("alignment_diagnostic.png", debug_canvas)
    print("\nSaved 'alignment_diagnostic.png'.")
    print("Red Dots = Source (02_plate)")
    print("Blue Dots = Target (LivePortrait)")
    print("--- 🔬 DIAGNOSTIC COMPLETE ---\n")
    
    return debug_canvas


def create_chromatic_tear_mask(source_4k_warped, target_lp, blur_k=31):
    """
    Generates an exposure-immune mask by comparing the ratio of Red 
    to total light. Triggers exclusively when skin structurally tears into teeth/eyes.
    """
    src_f = source_4k_warped.astype(np.float32)
    tgt_f = target_lp.astype(np.float32)

    # Calculate total pixel luminosity (add epsilon to prevent division by zero)
    src_sum = src_f[:,:,0] + src_f[:,:,1] + src_f[:,:,2] + 1e-5
    tgt_sum = tgt_f[:,:,0] + tgt_f[:,:,1] + tgt_f[:,:,2] + 1e-5

    # Calculate the Red proportion (Channel 2 in OpenCV's BGR format)
    src_red_ratio = src_f[:,:,2] / src_sum
    tgt_red_ratio = tgt_f[:,:,2] / tgt_sum

    # Calculate the absolute shift in chromaticity
    ratio_shift = np.abs(src_red_ratio - tgt_red_ratio)

    # Smooth the detection to create a soft VFX feather
    if blur_k % 2 == 0: 
        blur_k += 1
    smoothed_shift = cv2.GaussianBlur(ratio_shift, (blur_k, blur_k), 0)

    # Normalize the mask. 
    # A 6% (0.06) shift in the red ratio mathematically indicates we hit teeth, sclera, or background.
    tear_mask = np.clip(smoothed_shift / 0.06, 0.0, 1.0)

    # Invert: 1.0 keeps 4K texture (stable skin), 0.0 removes it (tearing teeth/eyes)
    attenuation_mask = 1.0 - tear_mask
    return np.expand_dims(attenuation_mask, axis=-1)

def frequency_matched_transfer(source_4k, target_lp, freq_radius=91):
    """
    Separates the 4K image into high/low frequencies and rebuilds it using 
    the LivePortrait low-frequency geometry, safely preserving any Alpha channels.
    """
    if freq_radius % 2 == 0: 
        freq_radius += 1

    # 1. Alpha Channel Management (Extraction)
    has_alpha = source_4k.shape[2] == 4
    if has_alpha:
        alpha_channel = source_4k[:, :, 3] 
        # Keep as 8-bit uint8 for the median filter!
        src_rgb_8u = source_4k[:, :, :3] 
    else:
        src_rgb_8u = source_4k

    # Ensure LivePortrait base is strictly 3-channel RGB 8-bit
    tgt_rgb_8u = target_lp[:, :, :3] if target_lp.shape[2] == 4 else target_lp

    # 2. Extract Low Frequencies (Running Median Blur on the 8-bit arrays)
    low_4k_8u = cv2.medianBlur(src_rgb_8u, freq_radius)
    low_lp_8u = cv2.medianBlur(tgt_rgb_8u, freq_radius)

    # 3. Convert to 32-bit float AFTER the blur to allow for negative math
    src_rgb_f = src_rgb_8u.astype(np.float32)
    low_4k_f = low_4k_8u.astype(np.float32)
    low_lp_f = low_lp_8u.astype(np.float32)

    # 4. Extract High Frequencies from 4K
    high_4k = src_rgb_f - low_4k_f
    
    # (If you are using the chromatic tear mask, apply it to high_4k here)

    # 5. The Recombination
    recombined = low_lp_f + high_4k
    recombined_8bit = np.clip(recombined, 0, 255).astype(np.uint8)

    # 6. Alpha Channel Management (Re-attachment)
    if has_alpha:
        b, g, r = cv2.split(recombined_8bit)
        return cv2.merge((b, g, r, alpha_channel))
        
    return recombined_8bit

def frequency_matched_transfer_Gaussian(source_4k, target_lp, freq_radius=31):
    """
    Separates the 4K image into high/low frequencies and rebuilds it using 
    the LivePortrait low-frequency geometry, safely preserving any Alpha channels.
    """
    if freq_radius % 2 == 0: 
        freq_radius += 1

    # 1. Alpha Channel Management (Extraction)
    has_alpha = source_4k.shape[2] == 4
    if has_alpha:
        # Save the exact transparency mask for later
        alpha_channel = source_4k[:, :, 3] 
        # Isolate just the RGB pixels for the math
        src_rgb = source_4k[:, :, :3].astype(np.float32)
    else:
        src_rgb = source_4k.astype(np.float32)

    # Ensure LivePortrait base is strictly 3-channel RGB just in case
    tgt_rgb = target_lp[:, :, :3].astype(np.float32) if target_lp.shape[2] == 4 else target_lp.astype(np.float32)

    # 2. Extract Low Frequencies (Using only the isolated RGB channels)
    low_4k = cv2.GaussianBlur(src_rgb, (freq_radius, freq_radius), 0)
    low_lp = cv2.GaussianBlur(tgt_rgb, (freq_radius, freq_radius), 0)

    # 3. Extract High Frequencies from 4K
    high_4k = src_rgb - low_4k

    # 4. The Recombination
    # --- NEW: Apply Chromatic Attenuation ---
    # Fade the 4K texture to 0% wherever the skin structurally tears
    texture_mask = create_chromatic_tear_mask(src_rgb, tgt_rgb)
    attenuated_high_4k = high_4k * texture_mask

    # 4. The Recombination
    recombined = low_lp + attenuated_high_4k

    #recombined = low_lp + high_4k
    recombined_8bit = np.clip(recombined, 0, 255).astype(np.uint8)

    # 5. Alpha Channel Management (Re-attachment)
    if has_alpha:
        # Split the recombined image and merge it back together with the saved alpha
        b, g, r = cv2.split(recombined_8bit)
        return cv2.merge((b, g, r, alpha_channel))
        
    return recombined_8bit


def conform_base_to_plate(ai_base, warp_plate, error_mask):
    """
    Computes dense optical flow to dynamically warp the LivePortrait base (01) 
    to match the rigid geometry of the 4K Warped Plate (02).
    Uses Scharr gradients to bypass the Brightness Constancy failure between AI and Sensor plates.
    """
    print("\n   -> Calculating structural optical flow via gradient matching...")
    
    h, w = ai_base.shape[:2]
    
    # 1. Downscale
    scale_factor = 0.25
    small_ai = cv2.resize(ai_base, (0,0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    small_warp = cv2.resize(warp_plate, (0,0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    
    gray_ai = cv2.cvtColor(small_ai, cv2.COLOR_BGR2GRAY)
    gray_warp = cv2.cvtColor(small_warp, cv2.COLOR_BGR2GRAY)
    
    # 2. Blur to destroy 4K micro-textures
    gray_ai = cv2.GaussianBlur(gray_ai, (15, 15), 0)
    gray_warp = cv2.GaussianBlur(gray_warp, (15, 15), 0)

    # ==========================================
    # 3. STRUCTURE EXTRACTION (The Fix)
    # ==========================================
    # Extract the geometric ridges using a Scharr operator, ignoring all lighting/color differences
    grad_x_ai = cv2.Scharr(gray_ai, cv2.CV_32F, 1, 0)
    grad_y_ai = cv2.Scharr(gray_ai, cv2.CV_32F, 0, 1)
    mag_ai = cv2.magnitude(grad_x_ai, grad_y_ai)

    grad_x_warp = cv2.Scharr(gray_warp, cv2.CV_32F, 1, 0)
    grad_y_warp = cv2.Scharr(gray_warp, cv2.CV_32F, 0, 1)
    mag_warp = cv2.magnitude(grad_x_warp, grad_y_warp)

    # Normalize back to 8-bit for the Farneback calculator
    mag_ai = cv2.normalize(mag_ai, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    mag_warp = cv2.normalize(mag_warp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 4. Calculate Flow on the isolated structural ridges
    small_flow = cv2.calcOpticalFlowFarneback(
        mag_warp, mag_ai, None, 
        pyr_scale=0.5, levels=5, winsize=25, 
        iterations=3, poly_n=7, poly_sigma=1.5, flags=0
    )
    
    # 5. Upscale Flow Vectors
    flow = cv2.resize(small_flow, (w, h), interpolation=cv2.INTER_LINEAR)
    flow = flow * (1.0 / scale_factor)

    # 6. Attenuate with the Error Mask
    smooth_mask = cv2.GaussianBlur(error_mask, (151, 151), 0)
    mask_f = (smooth_mask / 255.0).astype(np.float32)
    if len(mask_f.shape) == 2:
        mask_f = np.expand_dims(mask_f, axis=-1)
        
    masked_flow = flow * mask_f
    
    # 7. Execute Remap
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = x_coords + masked_flow[..., 0]
    map_y = y_coords + masked_flow[..., 1]
    
    aligned_ai_base = cv2.remap(
        ai_base, map_x, map_y, 
        interpolation=cv2.INTER_LANCZOS4, 
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return aligned_ai_base


def refine_st_map1(ai_base, warp_plate, map_x, map_y, error_mask):
    """
    Calculates residual structural flow and modifies the ST-maps directly.
    Allows a single-pass remap of the original 4K photo to match AI geometry
    without suffering double-interpolation blur.
    """
    print("\n   -> Calculating residual flow to update ST-map coordinates...")
    
    h, w = ai_base.shape[:2]
    
    # 1. Downscale for Macro-Geometry
    scale_factor = 0.25
    small_ai = cv2.resize(ai_base, (0,0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    small_warp = cv2.resize(warp_plate, (0,0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    
    gray_ai = cv2.cvtColor(small_ai, cv2.COLOR_BGR2GRAY)
    gray_warp = cv2.cvtColor(small_warp, cv2.COLOR_BGR2GRAY)
    
    gray_ai = cv2.GaussianBlur(gray_ai, (15, 15), 0)
    gray_warp = cv2.GaussianBlur(gray_warp, (15, 15), 0)

    # 2. Extract Structural Ridges (Scharr)
    grad_x_ai = cv2.Scharr(gray_ai, cv2.CV_32F, 1, 0)
    grad_y_ai = cv2.Scharr(gray_ai, cv2.CV_32F, 0, 1)
    mag_ai = cv2.magnitude(grad_x_ai, grad_y_ai)

    grad_x_warp = cv2.Scharr(gray_warp, cv2.CV_32F, 1, 0)
    grad_y_warp = cv2.Scharr(gray_warp, cv2.CV_32F, 0, 1)
    mag_warp = cv2.magnitude(grad_x_warp, grad_y_warp)

    mag_ai = cv2.normalize(mag_ai, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    mag_warp = cv2.normalize(mag_warp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 3. Calculate Flow (DIRECTION: AI to Warp)
    # We want to know where the AI pixels should pull 4K data from.
    # winsize=35 allows the algorithm to reach further for the remaining misalignment.
    small_flow = cv2.calcOpticalFlowFarneback(
        mag_ai, mag_warp, None, 
        pyr_scale=0.5, levels=5, winsize=35, 
        iterations=3, poly_n=7, poly_sigma=1.5, flags=0
    )
    
    # 4. Upscale Flow Vectors
    flow = cv2.resize(small_flow, (w, h), interpolation=cv2.INTER_LINEAR)
    flow = flow * (1.0 / scale_factor)

    # 5. Attenuate with the Error Mask
    smooth_mask = cv2.GaussianBlur(error_mask, (151, 151), 0)
    mask_f = (smooth_mask / 255.0).astype(np.float32)
    if len(mask_f.shape) == 2:
        mask_f = np.expand_dims(mask_f, axis=-1)
        
    masked_flow = flow * mask_f
    
    # 6. Apply residual flow to the Coordinate Maps
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # The new lookup coordinates
    remap_x = x_coords + masked_flow[..., 0]
    remap_y = y_coords + masked_flow[..., 1]
    
    # Remap the actual ST-maps so they now point to the newly stretched locations
    updated_map_x = cv2.remap(map_x, remap_x, remap_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    updated_map_y = cv2.remap(map_y, remap_x, remap_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return updated_map_x, updated_map_y

def refine_st_map2(ai_base, warp_plate, map_x, map_y, error_mask):
    """
    Calculates residual structural flow and modifies the ST-maps directly.
    Allows a single-pass remap of the original 4K photo to match AI geometry
    without suffering double-interpolation blur.
    """
    print("\n   -> Calculating residual flow to update ST-map coordinates...")    
    h, w = ai_base.shape[:2]
    
    # 1. Downscale for Macro-Geometry
    scale_factor = 0.25
    small_ai = cv2.resize(ai_base, (0,0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    small_warp = cv2.resize(warp_plate, (0,0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    gray_ai = cv2.cvtColor(small_ai, cv2.COLOR_BGR2GRAY)
    gray_warp = cv2.cvtColor(small_warp, cv2.COLOR_BGR2GRAY)
    blur_amount = 15
    gray_ai = cv2.GaussianBlur(gray_ai, (blur_amount, blur_amount), 0)
    gray_warp = cv2.GaussianBlur(gray_warp, (blur_amount, blur_amount), 0)

    # 2. Extract Structural Ridges (Scharr)
    grad_x_ai = cv2.Scharr(gray_ai, cv2.CV_32F, 1, 0)
    grad_y_ai = cv2.Scharr(gray_ai, cv2.CV_32F, 0, 1)
    mag_ai = cv2.magnitude(grad_x_ai, grad_y_ai)
    grad_x_warp = cv2.Scharr(gray_warp, cv2.CV_32F, 1, 0)
    grad_y_warp = cv2.Scharr(gray_warp, cv2.CV_32F, 0, 1)
    mag_warp = cv2.magnitude(grad_x_warp, grad_y_warp)
    mag_ai = cv2.normalize(mag_ai, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    mag_warp = cv2.normalize(mag_warp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 3. Calculate Flow (DIRECTION: AI to Warp)
    # We want to know where the AI pixels should pull 4K data from.
    small_flow = cv2.calcOpticalFlowFarneback(
        mag_ai, mag_warp, None, 
        pyr_scale=0.5, levels=5, winsize=95, # 35 and lower, bad. 95 good 
        iterations=3, poly_n=7, poly_sigma=1.5, flags=0
    )
    
    # 4. Upscale Flow Vectors
    flow = cv2.resize(small_flow, (w, h), interpolation=cv2.INTER_LINEAR)
    flow = flow * (1.0 / scale_factor)

    # 5. Attenuate with the Error Mask (Solid Core Method)
    # Expand the pure white areas outward by ~35 pixels to fully engulf the nostril 
    # and jawline gaps, ensuring the flow vectors apply at 100% strength in these zones.
    dilate_kernel = np.ones((35, 35), np.uint8)
    dilated_mask = cv2.dilate(error_mask, dilate_kernel, iterations=1)
    # Blur the expanded mask for a smooth transition back to the unwarped 4K plate.
    # Because it was dilated first, the problem areas remain safely at 1.0 (255) beneath the blur.
    smooth_mask = cv2.GaussianBlur(dilated_mask, (91, 91), 0)
    mask_f = (smooth_mask / 255.0).astype(np.float32)
    if len(mask_f.shape) == 2:
        mask_f = np.expand_dims(mask_f, axis=-1)
    masked_flow = flow * mask_f
    
    # 6. Apply residual flow to the Coordinate Maps
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    # The new lookup coordinates
    remap_x = x_coords + masked_flow[..., 0]
    remap_y = y_coords + masked_flow[..., 1]
    # Remap the actual ST-maps so they now point to the newly stretched locations
    updated_map_x = cv2.remap(map_x, remap_x, remap_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    updated_map_y = cv2.remap(map_y, remap_x, remap_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return updated_map_x, updated_map_y

def modify_mask(mask, expansion_radius, closing_radius=30, feather_radius=10):
    if expansion_radius > 0:
        # Create a circular or square kernel to define how the mask grows
        kernel = np.ones((expansion_radius, expansion_radius), np.uint8)
        # Dilate (expand) the clean mask
        mask = cv2.dilate(mask, kernel, iterations=1)

    if closing_radius > 0:
        # 1. Define a large structural element (kernel)
        # The size dictates how large of a black gap it will bridge. 
        # A 25x25 or 35x35 kernel is usually required for a 4K canvas.
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_radius,closing_radius))
        # 2. Execute the Morphological Close
        # This acts like a shrink-wrap, fusing the archipelago into a solid landmass
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)

    # Modify the feathering line to use the newly expanded_mask
    return cv2.GaussianBlur(mask, (feather_radius, feather_radius), 0) if feather_radius > 0 else mask

    

def generate_difference_mask(ai_highres, warped_calibrated,
                                difference_threshold = 23,
                                threshold_mode = "LAB" # must be "LAB", "weightedRGB", "maxRGB", "mono" or "hybrid"
                                ):
    PRE_BLUR_RADIUS = 29  # Must be odd. Blurs away 4K skin pores before comparing.
    
    # Square Pixels: How large must a cluster of difference be to avoid being ignored?
    # (Removes skin pore noise or micro-flickers)
    MIN_CLUSTER_AREA = 700 # good for mono and maxRGB threshold_mode
    #MIN_CLUSTER_AREA = 1200

    # 1. Low-Pass Filter: Blur the 4K images to melt the skin pores and stubble 
    # while preserving the solid color volumes of the lips and teeth.
    ai_blur = cv2.GaussianBlur(ai_highres, (PRE_BLUR_RADIUS, PRE_BLUR_RADIUS), 0)
    #ai_blur = cv2.GaussianBlur(local_color_transfer(warped_calibrated,ai_highres), (PRE_BLUR_RADIUS, PRE_BLUR_RADIUS), 0)
    warp_blur = cv2.GaussianBlur(warped_calibrated, (PRE_BLUR_RADIUS, PRE_BLUR_RADIUS), 0)
    #ai_blur = local_color_transfer(warp_blur,ai_blur)
    if threshold_mode == "mono":    
        abs_diff = cv2.absdiff(ai_blur, warp_blur)
        diff_gray = cv2.cvtColor(abs_diff, cv2.COLOR_BGR2GRAY)
        # 0-255: How mathematically different must a pixel be to trigger the mask?
        # (Higher = ignores more subtle shading changes; Lower = grabs everything)
        #difference_threshold = 12 # good for mono threshold_mode
        _, raw_mask = cv2.threshold(diff_gray, difference_threshold//2, 255, cv2.THRESH_BINARY)
    elif threshold_mode == "weightedRGB":
        abs_diff = cv2.absdiff(ai_blur, warp_blur)    
        # Define channel weights (Assuming OpenCV's default BGR format)
        WEIGHT_B = 0.2
        WEIGHT_G = 0.6
        WEIGHT_R = 0.3
        # Calculate the weighted sum of the channels. 
        # Cast to float32 to prevent 8-bit overflow during the addition.
        weighted_diff = (
            abs_diff[:, :, 0].astype(np.float32) * WEIGHT_B + 
            abs_diff[:, :, 1].astype(np.float32) * WEIGHT_G + 
            abs_diff[:, :, 2].astype(np.float32) * WEIGHT_R
        )
        # Clip the results to ensure no pixel exceeds 255, then lock it back to uint8
        weighted_diff_uint8 = np.clip(weighted_diff, 0, 255).astype(np.uint8)        
        # Threshold based on the new weighted difference
        #difference_threshold = 20
        _, raw_mask = cv2.threshold(weighted_diff_uint8, difference_threshold, 255, cv2.THRESH_BINARY)
    elif threshold_mode == "maxRGB":
        abs_diff = cv2.absdiff(ai_blur, warp_blur)    
        # Instead of converting to gray, find the maximum difference across B, G, and R
        diff_max = np.max(abs_diff, axis=2).astype(np.uint8)
        # Threshold based on the highest single-channel difference
        # 0-255: How mathematically different must a pixel be to trigger the mask?
        # (Higher = ignores more subtle shading changes; Lower = grabs everything)
        #difference_threshold = 23 # good for maxRGB threshold_mode
        _, raw_mask = cv2.threshold(diff_max, difference_threshold, 255, cv2.THRESH_BINARY)
    elif threshold_mode == "LAB":
        # 2. Convert the smoothed images to LAB space
        ai_lab = cv2.cvtColor(ai_blur, cv2.COLOR_BGR2LAB).astype(np.float32)
        warp_lab = cv2.cvtColor(warp_blur, cv2.COLOR_BGR2LAB).astype(np.float32)
        # 3. Calculate LAB Distance
        diff_lab = ai_lab - warp_lab
        color_distance = np.sqrt(
            (diff_lab[:, :, 0] * 0.8)**2 +   # L (Lightness/Texture)
            (diff_lab[:, :, 1] * 2.3)**2 +   # A (Red/Green) - Boosted
            (diff_lab[:, :, 2] * 1.1)**2     # B (Blue/Yellow) - Boosted
        )
        color_distance = np.clip(color_distance, 0, 255).astype(np.uint8)
        # 0-255: How mathematically different must a pixel be to trigger the mask?
        # (Higher = ignores more subtle shading changes; Lower = grabs everything)
        #difference_threshold = 23
        _, raw_mask = cv2.threshold(color_distance, difference_threshold, 255, cv2.THRESH_BINARY)
    elif threshold_mode == "hybrid":
        abs_diff = cv2.absdiff(ai_blur, warp_blur)    
        # 2. Convert the smoothed images to LAB space
        ai_lab = cv2.cvtColor(ai_blur, cv2.COLOR_BGR2LAB).astype(np.float32)
        warp_lab = cv2.cvtColor(warp_blur, cv2.COLOR_BGR2LAB).astype(np.float32)
        # 3. Calculate LAB Distance
        diff_lab = ai_lab - warp_lab
        color_distance = np.sqrt(
            (diff_lab[:, :, 0] * 1.5)**2 +   # L (Lightness/Texture)
            (diff_lab[:, :, 1] * 0.5)**2 +   # A (Red/Green)
            (abs_diff[:, :, 1].astype(np.float32) * 1.5) **2 + # Green
            (diff_lab[:, :, 2] * 0)**2     # B (Blue/Yellow)
        )
        color_distance = np.clip(color_distance, 0, 255).astype(np.uint8)
        # 0-255: How mathematically different must a pixel be to trigger the mask?
        # (Higher = ignores more subtle shading changes; Lower = grabs everything)
        #difference_threshold = 23
        _, raw_mask = cv2.threshold(color_distance, difference_threshold, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(raw_mask)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_CLUSTER_AREA:
            cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    return clean_mask

def apply_6axis_bw_filter(image, red_pct=300, yellow_pct=100, green_pct=-200, 
                          cyan_pct=-200, blue_pct=-200, magenta_pct=-200):
    """
    Replicates the Affinity Photo 2 Black & White Adjustment Layer.
    Converts percentages to decimal weights and applies them using 
    a 6-axis piecewise linear color model.
    """
    # Convert percentages to multipliers (e.g., 300% -> 3.0)
    rw = red_pct / 100.0
    yw = yellow_pct / 100.0
    gw = green_pct / 100.0
    cw = cyan_pct / 100.0
    bw = blue_pct / 100.0
    mw = magenta_pct / 100.0

    # Convert to float32 in the [0, 1] range for precise math
    img_f = image.astype(np.float32) / 255.0
    
    # Split channels (Assuming OpenCV's default BGR format)
    b, g, r = cv2.split(img_f)
    
    # Calculate max and min combinations for hue isolation
    max_rg = np.maximum(r, g)
    max_gb = np.maximum(g, b)
    max_rb = np.maximum(r, b)
    
    min_rg = np.minimum(r, g)
    min_gb = np.minimum(g, b)
    min_rb = np.minimum(r, b)
    
    # The neutral base (luminance shared by all three channels)
    c_min = np.minimum(min_rg, b)
    
    # Isolate the 6 specific color ranges mathematically
    val_red     = np.maximum(0, r - max_gb)
    val_green   = np.maximum(0, g - max_rb)
    val_blue    = np.maximum(0, b - max_rg)
    
    val_yellow  = np.maximum(0, min_rg - b)
    val_cyan    = np.maximum(0, min_gb - r)
    val_magenta = np.maximum(0, min_rb - g)
    
    # Composite the grayscale image by multiplying the isolated hues by your weights
    gray_f = c_min + \
             (val_red * rw) + \
             (val_yellow * yw) + \
             (val_green * gw) + \
             (val_cyan * cw) + \
             (val_blue * bw) + \
             (val_magenta * mw)
             
    # Clip to valid range [0, 1] to handle the extreme 300% / -200% values
    gray_f = np.clip(gray_f, 0.0, 1.0)
    
    # Convert back to an 8-bit image
    gray_8u = (gray_f * 255.0).astype(np.uint8)
    
    return gray_8u

def get_face_landmarks(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5
    )
    # MediaPipe requires RGB images
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    face_mesh.close()
    return results.multi_face_landmarks
    
def get_mask(image, difference_mask, multi_face_landmarks, outer_indices):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for face_landmarks in multi_face_landmarks:
        feature_points = []
        for idx in outer_indices:
            pt = face_landmarks.landmark[idx]
            x = int(pt.x * w)
            y = int(pt.y * h)
            feature_points.append([x, y])            
            
        # Convert to a numpy array for OpenCV
        feature_points = np.array(feature_points, dtype=np.int32)        
        
        # Draw a filled white polygon using the lip boundary
        cv2.fillPoly(mask, [feature_points], 255)    
        
    # Check for overlapping white pixels between the new mask and the difference mask
    intersection = cv2.bitwise_and(mask, difference_mask)
    has_overlap = cv2.countNonZero(intersection) > 0
    
    return mask, intersection, has_overlap

def extract_mouth_patch_mediapipe(source_image, face_landmarks, crop_w=832, crop_h=768, oval_w=800, oval_h=750):
    """
    Uses MediaPipe landmarks to define a strict geometric oval around the mouth.
    Returns a full-resolution mask isolating only the difference pixels inside the oval.
    """
    img_h, img_w = source_image.shape[:2]

    # ==========================================
    # 1. EXTRACT ANATOMICAL ANCHORS (FIXED)
    # ==========================================
    # Initialize with extremes so the normalized coordinates (0.0 to 1.0) overwrite them safely
    pt_top = float('inf')
    pt_left = float('inf')
    pt_right = float('-inf')

    for face in face_landmarks:
        pt_top = min(face.landmark[0].y, pt_top)
        pt_left = min(face.landmark[61].x, pt_left)
        pt_right = max(face.landmark[291].x, pt_right)

    # Convert normalized coordinates to absolute pixels
    top_y = int(pt_top * img_h)
    left_x = int(pt_left * img_w)
    right_x = int(pt_right * img_w)

    # ==========================================
    # 2. CALCULATE REQUESTED BOUNDARIES
    # ==========================================
    # Y-Axis: Top lip minus 25px, stretching down 768px
    y1 = top_y - 25
    y2 = y1 + crop_h
    
    # X-Axis: Center point between the left and right corners, spanning 832px
    center_x = (left_x + right_x) // 2
    x1 = center_x - (crop_w // 2)
    x2 = x1 + crop_w

    # ==========================================
    # 3. BOUNDARY CLAMPING
    # ==========================================
    if x1 < 0:
        x2 -= x1  
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > img_w:
        x1 -= (x2 - img_w)
        x2 = img_w
    if y2 > img_h:
        y1 -= (y2 - img_h)
        y2 = img_h

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Calculate the true center of the resulting box (in case it was clamped)
    box_center_x = (x1 + x2) // 2
    box_center_y = (y1 + y2) // 2

    # ==========================================
    # 4. CREATE THE FULL-RESOLUTION OVAL MASK
    # ==========================================
    # Create a blank 4096x4096 black canvas
    full_oval_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    # Draw the white ellipse at the calculated center
    buffer = 100
    axes = (oval_w // 2 + buffer, oval_h // 2 + buffer)
    cv2.ellipse(full_oval_mask, (box_center_x, box_center_y), axes, 0, 0, 360, 255, -1)
    
    # Blur the entire canvas to feather the edge
    full_oval_mask = cv2.GaussianBlur(full_oval_mask, (31, 31), 0)

    # ==========================================
    # 5. MULTIPLY AGAINST SOURCE DIFFERENCE MASK
    # ==========================================
    # This keeps the white pixels from the source ONLY where the oval is white, 
    # feathers the edges, and turns everything outside the oval to black.
    if len(source_image.shape) == 3:
        oval_mask_3c = cv2.cvtColor(full_oval_mask, cv2.COLOR_GRAY2BGR)
        final_full_mask = (source_image.astype(float) * (oval_mask_3c.astype(float) / 255.0)).astype(np.uint8)
    else:
        final_full_mask = (source_image.astype(float) * (full_oval_mask.astype(float) / 255.0)).astype(np.uint8)

    # Return the full 4096x4096 masked image, plus the coordinates in case you need them later
    return final_full_mask, (x1, y1, x2, y2)

def generate_vfx_plates_auto(is_video, source_img, file_label, st_map_path, M_o2c_512, low_res_warped, upright_4k, output_dir, face_upsampler=None):
    print("\n--- INITIATING VFX PLATE GENERATION ---")    
    # Sub-pixel nudge in case the AI's center pivot is slightly off-center
    NUDGE_X_PX = 0.0
    NUDGE_Y_PX = 0.0
    CANVAS_SIZE = 4096

    h_src, w_src = source_img.shape[:2]    

    os.makedirs(output_dir, exist_ok=True)
    print(f"VFX Plates will be saved to: {output_dir}")

    # ==========================================
    # STEP 1: CANONICAL SPACE CONVERSION
    # ==========================================
    image_interpolation_function = cv2.INTER_LANCZOS4
    map_interpolation_function = cv2.INTER_LANCZOS4 #cv2.INTER_CUBIC was giving wobbly lines which was more noticed than the micro-jitter of LANCZOS
    ai_canonical_512 = cv2.warpAffine(low_res_warped, M_o2c_512, (512, 512), flags=image_interpolation_function)
    #ai_highres = cv2.resize(ai_canonical_512, (CANVAS_SIZE, CANVAS_SIZE), interpolation=image_interpolation_function)

    if face_upsampler is not None:
        print("\n   -> Running RealESRGAN 4x hallucination on AI Base...")
        # Push 512x512 to 2048x2048 using the neural network
        ai_2048, _ = face_upsampler.enhance(ai_canonical_512, outscale=4)
        # Stretch the remaining distance to 4096 using Lanczos
        ai_highres = cv2.resize(ai_2048, (CANVAS_SIZE, CANVAS_SIZE), interpolation=image_interpolation_function)
    else:
        ai_highres = cv2.resize(ai_canonical_512, (CANVAS_SIZE, CANVAS_SIZE), interpolation=image_interpolation_function)
    
    MAX_DIM = 1280.0
    s_up = max(h_src, w_src) / MAX_DIM if max(h_src, w_src) > MAX_DIM else 1.0
    s_canvas = CANVAS_SIZE / 512.0 

    M_c2o_highres = np.zeros((2, 3), dtype=np.float64)
    M_c2o_highres[:, 0:2] = M_c2o_512[:, 0:2] * (s_up / s_canvas)
    M_c2o_highres[:, 2] = M_c2o_512[:, 2] * s_up
    M_o2c_highres = cv2.invertAffineTransform(M_c2o_highres)
    
    if upright_4k is None:
        upright_4k = cv2.warpAffine(source_img, M_o2c_highres, (CANVAS_SIZE, CANVAS_SIZE), flags=image_interpolation_function)

    # ==========================================
    # STEP 2: AUTO-UNIFORM MATHEMATICAL EXPANSION
    # ==========================================
    use_st_map = True
    if use_st_map:
        print("\nCalculating Uniform Geometric Scale...")
        st_map_64 = cv2.imread(st_map_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if st_map_64 is None: 
            raise FileNotFoundError(f"CRITICAL: Missing image at '{st_map_path}'.")
        
        # 1. Extract X (Red), Y (Green), and Z (Blue/Depth) channels
        x_norm_4k = cv2.resize(st_map_64[:, :, 2], (CANVAS_SIZE, CANVAS_SIZE), interpolation=map_interpolation_function)
        y_norm_4k = cv2.resize(st_map_64[:, :, 1], (CANVAS_SIZE, CANVAS_SIZE), interpolation=map_interpolation_function)

        # 2. Convert absolute map values to raw pixel coordinates
        map_x_raw = (x_norm_4k * (CANVAS_SIZE - 1)).astype(np.float32)
        map_y_raw = (y_norm_4k * (CANVAS_SIZE - 1)).astype(np.float32)

        # ==========================================
        # --- Z-DEPTH DYNAMIC SCALING MATH ---
        # ==========================================
        # y_base, x_base = np.mgrid[0:CANVAS_SIZE, 0:CANVAS_SIZE].astype(np.float32)

        # # Extract the local displacement vectors
        # dx = map_x_raw - x_base
        # dy = map_y_raw - y_base

        # # Establish baseline depth (Pivot plane)
        # z_norm_4k = cv2.resize(st_map_64[:, :, 0], (CANVAS_SIZE, CANVAS_SIZE), interpolation=map_interpolation_function)
        # baseline_z = np.median(z_norm_4k)
        # depth_scale = baseline_z / (z_norm_4k + 1e-6)

        # # --- NEW: SPATIAL ATTENUATION MASK ---
        # # Create a soft radial gradient anchored to the center of the canvas.
        # center_x, center_y = CANVAS_SIZE // 2, CANVAS_SIZE // 2
        
        # # Adjust this radius! 1200px covers the nose/eyes nicely on a 4K canvas
        # # but fades out before hitting the chin or shirt collar.
        # radius_sq = 400 ** 2 
        # dist_sq = (x_base - center_x)**2 + (y_base - center_y)**2
        
        # # Generates a smooth falloff from 1.0 (center) to 0.0 (edges)
        # spatial_mask = np.exp(-dist_sq / radius_sq)

        # # Blend the scale factor using the mask.
        # # If mask is 0.0 (shirt), blended_scale becomes exactly 1.0 (standard flat warp).
        # # If mask is 1.0 (nose), blended_scale becomes the full depth_scale.
        # blended_scale = 1.0 + ((depth_scale - 1.0) * spatial_mask)

        # # Apply the feathered scale to the displacement vectors
        # map_x_raw = x_base + (dx * blended_scale)
        # map_y_raw = y_base + (dy * blended_scale)
        # ==========================================
                
        # 1. Read the exact out-of-bounds smear on all four edges
        c = CANVAS_SIZE // 2
        missing_left = -map_x_raw[c, 0]
        missing_right = map_x_raw[c, -1] - (CANVAS_SIZE - 1)
        missing_top = -map_y_raw[0, c]
        missing_bottom = map_y_raw[-1, c] - (CANVAS_SIZE - 1)
        
        # 2. Find the most severe shrinkage edge to set the uniform bounding box
        max_missing_edge = max(missing_left, missing_right, missing_top, missing_bottom)
        TOTAL_EXPANSION_PX = max_missing_edge * 2.0
        SCALE_FACTOR = (CANVAS_SIZE - TOTAL_EXPANSION_PX) / CANVAS_SIZE
        
        print(f"   -> Max Smear Detected: {max_missing_edge:.2f}px")
        print(f"   -> Total Bounding Box Expansion: {TOTAL_EXPANSION_PX:.2f}px")
        print(f"   -> Uniform Scale Multiplier: {SCALE_FACTOR:.6f}")

        # 3. Apply the Uniform Scale mathematically to the coordinates
        center_x = (CANVAS_SIZE / 2.0) - NUDGE_X_PX
        center_y = (CANVAS_SIZE / 2.0) - NUDGE_Y_PX

        map_x_calibrated = (map_x_raw - center_x) * SCALE_FACTOR + center_x
        map_y_calibrated = (map_y_raw - center_y) * SCALE_FACTOR + center_y

        warped_calibrated = cv2.remap(upright_4k, map_x_calibrated, map_y_calibrated, interpolation=image_interpolation_function, borderMode=cv2.BORDER_REPLICATE)

        # ==========================================
        # STEP 3: PROCEDURAL DIFFERENCE MASKING
        # ==========================================
        print("\nExtracting Difference Mask...")    
        cv2.imwrite(os.path.join(output_dir, f"02-2_upright_4k-{file_label.split('-')[0]}.png"), upright_4k)
        cv2.imwrite(os.path.join(output_dir, f"02-3_lp_warped-{file_label}.png"), warped_calibrated)

        # STEP 3.5: ST-MAP REFINEMENT & FINAL WARP
        # Pass the calibrated maps and the mask to generate the new coordinates
        # #compare_test = generate_difference_mask(ai_highres, warped_calibrated, 0,0)
        # compare_test = generate_difference_mask(ai_highres, warped_calibrated,11)
        # cv2.imwrite(os.path.join(output_dir, f"00-{file_label}-test_0.png"), compare_test)
        verbose = True
        if verbose:
            compare_test = generate_difference_mask(ai_highres, upright_4k,23) # was 11 for guidance, 23 for compare
            cv2.imwrite(os.path.join(output_dir, f"00-{file_label}-test_0.png"), compare_test) # before any refinement
        last_error = None
        for i in range(1,5):  # iterations of refinement (usually 3 is enough)
            compare_test = generate_difference_mask(ai_highres, warped_calibrated,23) # was 11 for guidance, 23 for compare
            if verbose:
                cv2.imwrite(os.path.join(output_dir, f"00-{file_label}-test_{i}.png"), compare_test)
            error = cv2.countNonZero(compare_test)
            if last_error is not None and error + 100 >= last_error:
                print(f"   -> Iteration {i}: No major improvement in error ({error} ~>= {last_error}). Stopping refinement.")
                break
            last_error = error

            #guidance = generate_difference_mask(ai_highres, warped_calibrated, 43,43,11)#43,11, 3)
            #guidance = modify_mask(compare_test, 43,35,43)
            guidance = modify_mask(compare_test, 93,35,93)
            #guidance = modify_mask(generate_difference_mask(ai_highres, warped_calibrated,11), 43,35,43)
            new_map_x, new_map_y = refine_st_map2(
                ai_highres, warped_calibrated, 
                map_x_calibrated, map_y_calibrated, 
                guidance)
            # Re-run the remap from the untouched, upright 4K source using the updated maps
            warp2_4k = cv2.remap(
                upright_4k, new_map_x, new_map_y, 
                interpolation=image_interpolation_function, 
                borderMode=cv2.BORDER_REPLICATE)
            map_x_calibrated, map_y_calibrated = new_map_x, new_map_y
            warped_calibrated = warp2_4k
            # #compare_test = generate_difference_mask(ai_highres, warp2_4k, 0,0)
            # #compare_test = guidance = modify_mask(generate_difference_mask(ai_highres, warped_calibrated,11), 43,30,43)
            # compare_test = generate_difference_mask(ai_highres, warp2_4k,11)
            # cv2.imwrite(os.path.join(output_dir, f"00-{file_label}-test_{i}.png"), compare_test)
            # error = cv2.countNonZero(compare_test)
            # if error >= last_error:
            #     print(f"   -> Iteration {i}: No improvement in error ({error} >= {last_error}). Stopping refinement.")
            #     break
            # map_x_calibrated, map_y_calibrated = new_map_x, new_map_y
            # warped_calibrated = warp2_4k
            # last_error = error
    else:
        warp2_4k = upright_4k
        
    #moving, pinned = get_warp_groups()
    
    # target_indices = []
    # # Loop through each dictionary in the groups list
    # for group in groups:
    #     # Append the list of indices from this group to our master list
    #     target_indices.extend(group['indices'])
    # # Remove any duplicates to ensure clean mathematical processing
    # target_indices = list(set(target_indices))
    # run_alignment_diagnostic(warp2_4k, ai_highres, target_indices)

    # warp2_4k = apply_unified_landmark_warp(warp2_4k, ai_highres, moving, pinned)
    # cv2.imwrite(os.path.join(output_dir, f"03-5lm_warped-{file_label}.png"), warp2_4k)

    #warped_calibrated_lf_matched = frequency_matched_transfer(warped_calibrated,ai_highres)
    #b, g, r = cv2.split(warped_calibrated_lf_matched)
    #punched_4k_rgba = cv2.merge((b, g, r, alpha_channel))
    # align to morphs
    #aligned_ai_highres = conform_base_to_plate(ai_highres, warped_calibrated, final_alpha)

    warp2_calibrated_lf_matched = frequency_matched_transfer(warp2_4k,ai_highres)
    #final_alpha = generate_difference_mask(ai_highres, warp2_calibrated_lf_matched, 0, 0,10)
    #cv2.imwrite(os.path.join(output_dir, f"03_plate_Alpha_Mask1-{file_label}.png"), final_alpha)
    #inpaint_mask = generate_difference_mask(ai_highres, warp2_calibrated_lf_matched, 43,43,24,"maxRGB")
    #inpaint_mask = modify_mask(generate_difference_mask(ai_highres, warp2_calibrated_lf_matched,24,"hybrid"), 43, 0, 43) # before RealGan4x
    inpaint_mask = modify_mask(generate_difference_mask(ai_highres, warp2_calibrated_lf_matched,10,"hybrid"), 43, 0, 43) # after RealGan4x
    cv2.imwrite(os.path.join(output_dir, f"03_plate_Alpha_Mask1b-{file_label}.png"), inpaint_mask)
        
    #b, g, r = cv2.split(warp2_calibrated_lf_matched)
    #final_alpha = generate_difference_mask(ai_highres, warp2_4k, 99,51)
    # Subject's Right Eye (Viewer's Left)
    right_socket_indices = [
        46, 53, 52, 65, 55,      # Top ridge
        193, 122,                # Inner drop
        115,                     # hugs the inner eye corner
        121, 120, 119, 118, 117, 111, # Bottom ridge
        130, 226                 # Outer drop
    ]
    left_socket_indices = [
        276, 283, 282, 295, 285, # Top ridge
        417, 351,                # Inner drop
        344,                     # hugs the inner eye corner
        350, 349, 348, 347, 346, 340, # Bottom ridge
        359, 300                 # Outer drop
    ]
    face_landmarks_driver = get_face_landmarks(ai_highres)
    face_landmarks_source = get_face_landmarks(warp2_calibrated_lf_matched)
    face_landmarks = (face_landmarks_driver or []) + (face_landmarks_source or [])
    binary_eyel_mask, minimal_eyel_mask, need_eyel = get_mask(ai_highres, inpaint_mask, face_landmarks, left_socket_indices)
    binary_eyer_mask, minimal_eyer_mask, need_eyer = get_mask(ai_highres, inpaint_mask, face_landmarks, right_socket_indices)
    # outer_lip_indices = [
    #     61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 
    #     291, 409, 270, 269, 267, 0, 37, 39, 40
    #     ]
    # binary_mouth_mask, minimal_mouth_mask, need_mouth = get_mask(ai_highres, inpaint_mask, face_landmarks, outer_lip_indices)
    # #run_alignment_diagnostic(warp2_calibrated_lf_matched, ai_highres, outer_lip_indices)
    h, w = ai_highres.shape[:2]
    circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    mask_blur = (25,55)
    final_alpha = np.zeros((h, w), dtype=np.uint8)
    need_mouth = True
    if need_mouth or is_video:
        #expanded_mask = cv2.dilate(minimal_mouth_mask, circular_kernel, iterations=1)
        #blur_mask = cv2.GaussianBlur(expanded_mask, mask_blur, 0)
        #final_alpha = blur_mask
        if face_landmarks_driver and len(face_landmarks_driver) > 0:
            # Pass the first detected face object
            final_alpha, coords = extract_mouth_patch_mediapipe(
                inpaint_mask, 
                face_landmarks_driver, 
                crop_w=832, crop_h=768
            )
            print(f"Extracted mouth patch at coordinates: {coords}")
        else:
            print("Warning: No face detected in AI base plate.")        
    if need_eyel or need_eyer or is_video:
        expanded_mask = cv2.dilate(minimal_eyel_mask, circular_kernel, iterations=1)
        blur_mask = cv2.GaussianBlur(expanded_mask, mask_blur, 0)
        final_alpha = cv2.bitwise_or(final_alpha, blur_mask)
        expanded_mask = cv2.dilate(minimal_eyer_mask, circular_kernel, iterations=1)
        blur_mask = cv2.GaussianBlur(expanded_mask, mask_blur, 0)
        final_alpha = cv2.bitwise_or(final_alpha, blur_mask)

    #alpha_channel = cv2.bitwise_not(final_alpha) 
    #punched2_4k_rgba = cv2.merge((b, g, r, alpha_channel))
    # align to morphs
    #aligned_ai_highres2 = conform_base_to_plate(ai_highres, warp2_4k, final_alpha)
    """
    mouth_blend = apply_6axis_bw_filter(warp2_4k, red_pct=300, yellow_pct=100, green_pct=-200, 
                          cyan_pct=-200, blue_pct=-200, magenta_pct=-200)
    mouth_base = apply_6axis_bw_filter(ai_highres, red_pct=100, yellow_pct=300, green_pct=-200, 
                          cyan_pct=-200, blue_pct=-200, magenta_pct=-200)
    difference_result = cv2.absdiff(mouth_base, mouth_blend)
    thresh_val = int((50 / 100.0) * 255)
    # 2. Convert to Grayscale
    # A difference blend on BGR images yields BGR differences. 
    # Grayscale merges them into a single luminosity map for clean masking.
    difference_result = cv2.cvtColor(difference_result, cv2.COLOR_BGR2GRAY)
    # 3. Apply the Binary Threshold
    # Any pixel brighter than thresh_val becomes 255 (White/Occluded)
    # Any pixel darker becomes 0 (Black/Transparent)
    _, binary_mask = cv2.threshold(difference_result, thresh_val, 255, cv2.THRESH_BINARY)
    circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 200))
    expanded_mask = cv2.dilate(binary_mask, circular_kernel, iterations=1)
    blur_radius = 51
    final_alpha = cv2.GaussianBlur(expanded_mask, (blur_radius, blur_radius), 0)
    """
    # ==========================================
    # STEP 4: EXPORT ASSETS
    # ==========================================
    print("Exporting discrete plates...")
    cv2.imwrite(os.path.join(output_dir, f"01_plate_LP-{file_label}.png"),ai_canonical_512)
    cv2.imwrite(os.path.join(output_dir, f"01_plate_AI_Base-{file_label}.png"), ai_highres)
    cv2.imwrite(os.path.join(output_dir, f"02_plate_4K_Warp_RGB-{file_label}.png"), warp2_4k)
    if need_mouth or need_eyel or need_eyer or is_video:
        cv2.imwrite(os.path.join(output_dir, f"03_plate_Alpha_Mask-{file_label}.png"), final_alpha)
    
    #cv2.imwrite(os.path.join(output_dir, f"04_plate_4K_Warp_RGBA-{file_label}.png"), punched2_4k_rgba)
    #cv2.imwrite(os.path.join(output_dir, f"04_plate_4K_Warp_RGBA-{file_label}.tiff"), punched2_4k_rgba)
    cv2.imwrite(os.path.join(output_dir, f"04_plate_4K_Warp_RGB-{file_label}.tiff"), warp2_calibrated_lf_matched)
    #b, g, r = cv2.split(warped_calibrated)
    #punched_4k_rgba = cv2.merge((b, g, r, alpha_channel))
    #cv2.imwrite(os.path.join(output_dir, f"04_plate_4K_WarpA_RGBA-{file_label}.png"), punched_4k_rgba)
    #cv2.imwrite(os.path.join(output_dir, f"04_plate_4K_WarpA_RGBA-{file_label}.tiff"), punched_4k_rgba)

    print("\n--- VFX PLATE GENERATION COMPLETE ---")
    return need_mouth, need_eyel or need_eyer, upright_4k

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, help="The source argument value")
    parser.add_argument('-d', type=str, help="The driver argument value")
    args,unknown_args = parser.parse_known_args()
    if not args.s:
        print("No -s source file argument was provided.")
        exit(1)
    source_file_path = args.s
    if not args.d:
        print("No -d driver file argument was provided.")
        exit(1)
    driver_file_path = args.d
    source_name = Path(source_file_path).stem
    driving_name = Path(driver_file_path).stem
    file_label = f"{source_name}-{driving_name}"

    # load what will be shared across all frames   
    OUTPUT_DIR = "output/tmp"
    M_o2c_512 = None
    source_img = None
    need_mouth = False
    need_eyes = False

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer
    import torch

    print("Loading RealESRGAN 4x model into VRAM...")
    rrdb_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    face_upsampler = RealESRGANer(
        scale=4,
        model_path='../CodeFormer/weights/realesrgan/RealESRGAN_x4plus.pth', # Adjust path if needed
        model=rrdb_model,
        tile=0,#400,
        tile_pad=0,#10,
        pre_pad=0,
        half=torch.cuda.is_available()
    )
    
    source_video = cv2.VideoCapture(source_file_path) if is_video(source_file_path) else None
    if not source_video:
        source_img = cv2.imread(source_file_path)
        if source_img is None: raise FileNotFoundError(f"CRITICAL: Missing Source 4K at '{source_file_path}'")    
        M_c2o_loaded = np.loadtxt(f"{OUTPUT_DIR}/metadata/M_c2o-{source_name}-{driving_name}.txt")
        if len(M_c2o_loaded.shape) == 1: M_c2o_loaded = M_c2o_loaded.reshape(-1, 3)
        M_c2o_512 = M_c2o_loaded[:2, :].copy()
        M_o2c_512 = cv2.invertAffineTransform(M_c2o_512)
    if is_video(driver_file_path):
        capd = cv2.VideoCapture(f"animations/{source_name}--{driving_name}.mp4")
        i = 0
        upright_4k = None
        while capd.isOpened():
            ret, low_res_warped = capd.read()
            if not ret:
                break # Reached the end of the video 
            if source_video:
                ret, source_img = source_video.read()
                if not ret:
                    break # Reached the end of the video                 
                M_c2o_loaded = np.loadtxt(f"{OUTPUT_DIR}/metadata/M_c2o-{source_name}-{driving_name}-{i:04d}.txt")
                if len(M_c2o_loaded.shape) == 1: M_c2o_loaded = M_c2o_loaded.reshape(-1, 3)
                M_c2o_512 = M_c2o_loaded[:2, :].copy()
                M_o2c_512 = cv2.invertAffineTransform(M_c2o_512)
            print(f"\nProcessing Frame {i:04d}...")
            ST_MAP = f"{OUTPUT_DIR}/metadata/st_map_{file_label}-{i:04d}.exr"
            frame_label = f"{file_label}-{i:04d}"         
            need_mouth_frame, need_eyes_frame, upright_4k = generate_vfx_plates_auto(True, source_img, frame_label, ST_MAP, M_o2c_512, low_res_warped, upright_4k, OUTPUT_DIR, face_upsampler)
            if source_video:
                upright_4k = None
            need_mouth = need_mouth or need_mouth_frame
            need_eyes = need_eyes or need_eyes_frame
            #if not need_mouth:
                # delete mouth masks      
            #if not need_eyes:
                # delete eye masks      
            i += 1
    else:
        ST_MAP = f"{OUTPUT_DIR}/metadata/st_map_{file_label}-0000.exr"
        lp_output = f"animations/{source_name}--{driving_name}.jpg"
        low_res_warped = cv2.imread(lp_output)
        if low_res_warped is None: 
            raise FileNotFoundError(f"CRITICAL: Missing AI output image at '{lp_output}'.")
        generate_vfx_plates_auto(False, source_img, file_label, ST_MAP, M_o2c_512, low_res_warped, None, OUTPUT_DIR, face_upsampler)
