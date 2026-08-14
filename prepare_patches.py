import cv2
import json
import numpy as np
import mediapipe as mp
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

def apply_unsharp_mask(image_input, radius=20, factor=0.5):
    """
    Applies an unsharp mask filter replicating Affinity Photo parameters.
    
    Parameters:
    -----------
    image_input : PIL.Image or str or np.ndarray
        The input image patch to sharpen.
    radius : float
        The blurring radius (maps directly to sigma in OpenCV).
    factor : float
        The sharpening intensity factor (Affinity Factor 0.5).
        
    Returns:
    --------
    PIL.Image
        The sharpened image patch.
    """
    # 1. Handle pipeline input types gracefully
    if isinstance(image_input, str):
        cv_img = cv2.imread(image_input)
    elif isinstance(image_input, Image.Image):
        cv_img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        cv_img = image_input.copy()
    else:
        raise ValueError("Unsupported image input type.")

    # Convert to float32 to prevent arithmetic underflow/overflow clipping during subtraction
    img_float = cv_img.astype(np.float32)

    # 2. Generate the low-frequency mask layer
    # For a radius of 20, we pass it as the standard deviation (sigma). 
    # Setting ksize to (0,0) forces OpenCV to auto-calculate the perfect window matrix size.
    blurred = cv2.GaussianBlur(img_float, (0, 0), sigmaX=radius, sigmaY=radius)

    # 3. Apply the Unsharp Mask formula: 
    # Sharpened = Original + Factor * (Original - Blurred)
    # Re-arranged for cv2.addWeighted optimization: (1 + Factor) * Original - Factor * Blurred
    sharpened = cv2.addWeighted(img_float, 1.0 + factor, blurred, -factor, 0)

    # 4. Clip boundaries to safe 8-bit space and convert back to uint8
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # Return to PIL Image to match your pipeline's synthesis outputs
    if isinstance(image_input, Image.Image):
        return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    
    return sharpened

def match_sizes(base_path, rgba_path):
    bg = cv2.imread(base_path)
    fg_rgba = cv2.imread(rgba_path, cv2.IMREAD_UNCHANGED)

    if bg is None or fg_rgba is None:
        raise FileNotFoundError("Could not find the input plates.")

    fg_rgba = cv2.resize(fg_rgba, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # Isolate the RGB and Alpha channels properly
    fg_rgb = fg_rgba[:, :, :3]
    alpha_channel = fg_rgba[:, :, 3] if len(fg_rgba.shape) > 3 else None
    
    return bg, fg_rgb, alpha_channel

def composite_alpha_over_base(bg, fg_rgb, alpha):
    alpha = alpha / 255.0
    # Expands a 2D mask (4096, 4096) into a 3D mask (4096, 4096, 1)
    # This allows NumPy to safely broadcast the single alpha value across all 3 RGB channels
    alpha = np.expand_dims(alpha, axis=-1) 
    
    return (bg * (1 - alpha) + fg_rgb * alpha).astype(np.uint8)

def isolate_face_zone(mask_image, safe_zone):
    """
    Deletes all boundary tears and ear/neck distortions by blacking out 
    everything outside a user-defined central 'Safe Zone'.
    """
    x_min, y_min, x_max, y_max = safe_zone
    clean_mask = np.zeros_like(mask_image) # Create a pure black canvas
    
    # Copy ONLY the pixels that fall inside the safe zone
    clean_mask[y_min:y_max, x_min:x_max] = mask_image[y_min:y_max, x_min:x_max]
    
    return clean_mask

def extract_patch(image, center_x, center_y, patch_width=768, patch_height=768):
    """Extracts a bounding box with independent width and height controls."""
    half_w = patch_width // 2
    half_h = patch_height // 2
    height, width = image.shape[:2]
    
    y1 = max(0, center_y - half_h)
    y2 = min(height, center_y + half_h)
    x1 = max(0, center_x - half_w)
    x2 = min(width, center_x + half_w)
    
    patch = image[y1:y2, x1:x2]
    
    # Pad with black if the crop hits the edge of the 4K canvas
    if patch.shape[0] != patch_height or patch.shape[1] != patch_width:
        patch = cv2.copyMakeBorder(
            patch, 
            max(0, half_h - center_y), max(0, (center_y + half_h) - height),
            max(0, half_w - center_x), max(0, (center_x + half_w) - width),
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
    return patch

mp_face_mesh = mp.solutions.face_mesh

def find_dynamic_centers_mediapipe(image_bgr):
    """Uses MediaPipe to mathematically hunt down facial features."""
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5
    ) as face_mesh:
        
        # MediaPipe requires RGB color space
        results = face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            print("WARNING: MediaPipe could not detect a face in this frame.")
            return None, None
            
        landmarks = results.multi_face_landmarks[0].landmark
        img_h, img_w = image_bgr.shape[:2]
        
        # The exact topological indices for both eyes AND eyebrows
        eye_brow_indices = [
            # Left Eye & Brow
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
            # Right Eye & Brow
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
            336, 296, 334, 293, 300, 276, 283, 282, 295, 285
        ]
        
        # The exact topological indices for the mouth
        mouth_indices = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            375, 291, 409, 270, 269, 267, 0, 37, 39, 40,
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
        ]
        
        # Calculate the true bounding box center for the Eyes/Brows
        eye_xs = [landmarks[i].x * img_w for i in eye_brow_indices]
        eye_ys = [landmarks[i].y * img_h for i in eye_brow_indices]
        eyes_center = (int((min(eye_xs) + max(eye_xs)) / 2), int((min(eye_ys) + max(eye_ys)) / 2))

        # Calculate the true bounding box center for the Mouth
        mouth_xs = [landmarks[i].x * img_w for i in mouth_indices]
        mouth_ys = [landmarks[i].y * img_h for i in mouth_indices]
        mouth_center = (int((min(mouth_xs) + max(mouth_xs)) / 2), int((min(mouth_ys) + max(mouth_ys)) / 2))
        
        return eyes_center, mouth_center
    

def generate_plates(plate_01, plate_03, plate_04, out_eyes_img, out_eyes_mask=None,out_mouth_img=None,out_mouth_mask=None):
    print("Generating Composite Plate...")
    mask = cv2.imread(plate_03, cv2.IMREAD_GRAYSCALE) if plate_03 else None
    #FACE_SAFE_ZONE = [1200, 800, 2800, 3400] 
    # mask = isolate_face_zone(mask, FACE_SAFE_ZONE)

    # Generate the RGB image FIRST so MediaPipe can see the face
    print("Executing MediaPipe Dynamic Feature Tracking...")
    patch_metadata = {
        "mouth": None,
        "eyes": None
    }
    # Feed the RGB image into the new tracker
    bg,composite_image,alpha_channel = match_sizes(plate_01, plate_04) if plate_03 else None, cv2.imread(plate_01, cv2.IMREAD_UNCHANGED), None
    full_composite_image = composite_alpha_over_base(bg,composite_image,alpha_channel) if alpha_channel else composite_image
    eyes_coords, mouth_coords = find_dynamic_centers_mediapipe(composite_image)
    
    if mouth_coords and out_mouth_img:
        print(f" -> Mouth detected at {mouth_coords}. Extracting patch...")
        mouth_mask_patch = extract_patch(mask, mouth_coords[0], mouth_coords[1], patch_width=832) if mask is not None else None
        is_empty = mask is not None and cv2.countNonZero(mouth_mask_patch) == 0
        if not is_empty:
            mouth_img_patch = extract_patch(full_composite_image, mouth_coords[0], mouth_coords[1], patch_width=832)
            mouth_h, mouth_w = mouth_img_patch.shape[:2]
            print(f"    > Mouth Patch Data | Center (X,Y): {mouth_coords} | Final Resolution: {mouth_w}x{mouth_h}. Writing to {out_mouth_img}")
            # We cast to int() because numpy data types can sometimes crash the JSON encoder
            patch_metadata["mouth"] = {
                "center_x": int(mouth_coords[0]),
                "center_y": int(mouth_coords[1]),
                "width": int(mouth_w),
                "height": int(mouth_h)
            }
            # sharpen the image before writing so IP-Adapetor will help with high resolution synthesis
            cv2.imwrite(out_mouth_img, apply_unsharp_mask(mouth_img_patch))
            if mask is not None and out_mouth_mask:
                cv2.imwrite(out_mouth_mask, mouth_mask_patch)
    else:
        print(" -> No mouth occlusion.")

    if eyes_coords:
        print(f" -> Eyes detected at {eyes_coords}. Extracting panoramic patch...")
        eyes_mask_patch = extract_patch(mask, eyes_coords[0], eyes_coords[1], patch_width=1280, patch_height=512) if mask is not None else None
        is_empty = mask is not None and cv2.countNonZero(eyes_mask_patch) == 0
        if not is_empty:
            eyes_img_patch = extract_patch(composite_image, eyes_coords[0], eyes_coords[1], patch_width=1280, patch_height=512)
            eyes_h, eyes_w = eyes_img_patch.shape[:2]
            print(f"    > Eyes Patch Data  | Center (X,Y): {eyes_coords} | Final Resolution: {eyes_w}x{eyes_h}. Writing to {out_eyes_img}")
            patch_metadata["eyes"] = {
                "center_x": int(eyes_coords[0]),
                "center_y": int(eyes_coords[1]),
                "width": int(eyes_w),
                "height": int(eyes_h)
            }
            cv2.imwrite(out_eyes_img, eyes_img_patch)
            if mask is not None and out_eyes_mask:
                cv2.imwrite(out_eyes_mask, eyes_mask_patch)
    else:
        print(" -> No eye occlusions detected in this frame.")

    return patch_metadata

def process_files(driver_file_path,source_name,driving_name,file_label):
    if driver_file_path.endswith(".mp4"):
        cap = cv2.VideoCapture(f"animations/{source_name}--{driving_name}.mp4")
        i = 0
        while cap.isOpened():
            ret, _ = cap.read()
            if not ret:
                break # Reached the end of the video                
            print(f"\nProcessing Frame {i:04d}...")
            frame_label = f"{file_label}-{i:04d}"            
            out_mouth_img  = f"output/tmp/04_plate_4K_Warp_mouth-{frame_label}.png"
            out_mouth_mask = f"output/tmp/03_plate_Alpha_Mask_mouth-{frame_label}.png"
            out_eyes_img   = f"output/tmp/04_plate_4K_Warp_eyes-{frame_label}.png"
            out_eyes_mask  = f"output/tmp/03_plate_Alpha_Mask_eyes-{frame_label}.png"
            patch_metadata = generate_plates(f"output/tmp/01_plate_AI_Base-{frame_label}.png",
                            f"output/tmp/03_plate_Alpha_Mask-{frame_label}.png",
                            f"output/tmp/04_plate_4K_Warp_RGB-{frame_label}.tiff",
                            out_eyes_img, out_eyes_mask,out_mouth_img,out_mouth_mask)
            if patch_metadata:
                json_output_path = f"output/tmp/metadata/patch_metadata-{frame_label}.json"
                with open(json_output_path, "w") as json_file:
                    json.dump(patch_metadata, json_file, indent=4)
                print(f"\nPatches generated. Metadata successfully written to {json_output_path}")
            i += 1
    else:
        out_mouth_img  = f"output/tmp/04_plate_4K_Warp_mouth-{file_label}.png"
        out_mouth_mask = f"output/tmp/03_plate_Alpha_Mask_mouth-{file_label}.png"
        out_eyes_img   = f"output/tmp/04_plate_4K_Warp_eyes-{file_label}.png"
        out_eyes_mask  = f"output/tmp/03_plate_Alpha_Mask_eyes-{file_label}.png"
        patch_metadata = generate_plates(f"output/tmp/01_plate_AI_Base-{file_label}.png",
                        f"output/tmp/03_plate_Alpha_Mask-{file_label}.png",
                        f"output/tmp/04_plate_4K_Warp_RGB-{file_label}.tiff",
                        out_eyes_img, out_eyes_mask,out_mouth_img,out_mouth_mask)
        if patch_metadata:
            json_output_path = f"output/tmp/metadata/patch_metadata-{file_label}.json"
            with open(json_output_path, "w") as json_file:
                json.dump(patch_metadata, json_file, indent=4)        
            print(f"\nPatches generated. Metadata successfully written to {json_output_path}")



def normalize_eyes_crop(image_bgr):
    """
    Uses MediaPipe to find the eyes, levels them horizontally, and scales 
    the image so the outermost corners of the eyes are exactly 1280 pixels apart.
    """
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5
    ) as face_mesh:
        
        # MediaPipe requires RGB color space
        results = face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            print("WARNING: MediaPipe could not detect a face in this frame.")
            return image_bgr
            
        landmarks = results.multi_face_landmarks[0].landmark
        img_h, img_w = image_bgr.shape[:2]
        
        # MediaPipe indices for the eyes (without the brows to get true eye centers)
        # Viewer's Left (Subject's Right)
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        # Viewer's Right (Subject's Left)
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
            336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
        
        # The exact topological outermost corners
        outer_left_idx = 33   # Viewer's left
        outer_right_idx = 263 # Viewer's right
        
        outer_left_x = landmarks[outer_left_idx].x * img_w
        outer_left_y = landmarks[outer_left_idx].y * img_h
        outer_right_x = landmarks[outer_right_idx].x * img_w
        outer_right_y = landmarks[outer_right_idx].y * img_h
        
        # 1. Calculate the exact centers of each eye
        left_eye_xs = [landmarks[i].x * img_w for i in left_eye_indices]
        left_eye_ys = [landmarks[i].y * img_h for i in left_eye_indices]
        left_eye_center = (sum(left_eye_xs) / len(left_eye_xs), sum(left_eye_ys) / len(left_eye_ys))
        
        right_eye_xs = [landmarks[i].x * img_w for i in right_eye_indices]
        right_eye_ys = [landmarks[i].y * img_h for i in right_eye_indices]
        right_eye_center = (sum(right_eye_xs) / len(right_eye_xs), sum(right_eye_ys) / len(right_eye_ys))
        
        # 2. Calculate the rotation angle to level the eyes
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Rotation center (midpoint between the two eye centers)
        eyes_midpoint = (
            (left_eye_center[0] + right_eye_center[0]) / 2,
            (left_eye_center[1] + right_eye_center[1]) / 2
        )
        
        # 3. Rotate the image
        # Using BORDER_REPLICATE prevents black triangles from appearing at the edges during slight rotations
        M = cv2.getRotationMatrix2D(eyes_midpoint, angle, 1.0)
        rotated_image = cv2.warpAffine(image_bgr, M, (img_w, img_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # 4. Calculate the distance between the outermost ends
        # (Euclidean distance is invariant to rotation, so we calculate it using the original coordinates)
        eye_distance = np.sqrt((outer_right_x - outer_left_x)**2 + (outer_right_y - outer_left_y)**2)
        
        # 5. Scale the image
        # Standardizing the distance to exactly 1280 pixels fulfills the logic:
        # (< 1300 upscales to 1280, > 1280 downscales to 1280)
        target_distance = 1100.0
        scale_factor = target_distance / eye_distance
        
        print(f"Normalizing eyes crop: Original distance = {eye_distance:.2f}, Scale factor = {scale_factor:.4f}")
        
        new_w = int(img_w * scale_factor)
        new_h = int(img_h * scale_factor)
        
        normalized_image = cv2.resize(rotated_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        return normalized_image
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, help="The source argument value")
    parser.add_argument('-d', type=str, help="The driver argument value")
    parser.add_argument('-training', type=int, help="Prepare patches for model/LoRA training. Value is resolution (512 or 1024 for SDXL)")
    args,unknown_args = parser.parse_known_args()
    if not args.s:
        print("No -s source file argument was provided.")
        exit(1)
    source_file_path = args.s
    source_name = Path(source_file_path).stem

    if args.training:
        source_file = cv2.imread(source_file_path)  # Just to check if the file exists
        oriented_face = normalize_eyes_crop(source_file)  # Normalize the eyes crop before extracting patches
        wiggle_room = 25 # training loop will randomly crop for augmentation
        eyes_coords, mouth_coords = find_dynamic_centers_mediapipe(oriented_face)
        if mouth_coords:
            w = max(832, args.training) + wiggle_room
            h = max(768, args.training) + wiggle_room
            mouth_img_patch = extract_patch(oriented_face, mouth_coords[0], mouth_coords[1], patch_width=w, patch_height=h)
            mouth_h, mouth_w = mouth_img_patch.shape[:2]
            print(f"    > Mouth Patch Data | Center (X,Y): {mouth_coords} | Final Resolution: {mouth_w}x{mouth_h}")
            out_name = f"output/training/mouth/{source_name}_mouth_target.png"
            cv2.imwrite(out_name, mouth_img_patch)
            print(f" -> Mouth patch saved to {out_name}")
            # blurred = cv2.resize(mouth_img_patch, (args.training//8, args.training//8))
            # blurred = cv2.resize(blurred, (w, h))
            # out_name = f"output/training/{args.training}/input/{source_name}_mouth_in.png"
            # cv2.imwrite(out_name, blurred)
        else:
            print(" -> No mouth detected.")
        if eyes_coords:
            eyes_img_patch = extract_patch(oriented_face, eyes_coords[0], eyes_coords[1], patch_width=1280+wiggle_room, patch_height=max(512, args.training)+wiggle_room)
            eyes_h, eyes_w = eyes_img_patch.shape[:2]
            print(f"    > Eyes Patch Data | Center (X,Y): {eyes_coords} | Final Resolution: {eyes_w}x{eyes_h}")
            out_name = f"output/training/eyes/{source_name}_eyes_target.png"
            print(out_name)
            print(f" -> Eyes patch saved to {out_name}")
            # blurred = cv2.resize(mouth_img_patch, (args.training//8, args.training//8))
            # blurred = cv2.resize(blurred, (w, h))
            # out_name = f"output/training/{args.training}/input/{source_name}_eyes_in.png"
            # cv2.imwrite(out_name, blurred)
        else:
            print(" -> No eyes detected.")
    else:
        if not args.d:
            print("No -d driver file argument was provided.")
            exit(1)
        driver_file_path = args.d
        driving_name = Path(driver_file_path).stem
        file_label = f"{source_name}-{driving_name}"
        process_files(driver_file_path,source_name,driving_name,file_label)
