import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
from pathlib import Path

def extract_semantic_square(image, landmarks, feature_indices, output_size=512, padding_factor=1.2):
    """
    Finds the exact boundaries of a facial feature, cuts a dynamic square,
    and upscales it to the target resolution using Lanczos4 interpolation.
    """
    img_h, img_w = image.shape[:2]

    # Extract pixel coordinates for all boundary indices
    points = []
    for idx in feature_indices:
        pt = landmarks[idx]
        points.append([int(pt.x * img_w), int(pt.y * img_h)])
    points = np.array(points)

    # Find the absolute bounding box of the feature
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    # Calculate the center and dynamically size the square
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2

    width = max_x - min_x
    height = max_y - min_y

    # Base size is the longest edge to ensure the whole feature fits
    base_size = max(width, height)

    # Add a slight padding so the crop doesn't sit exactly on the eyelid/lip edge
    extraction_size = int(base_size * padding_factor)

    # Ensure the dimension is an even number for clean array slicing
    if extraction_size % 2 != 0:
        extraction_size += 1

    print(f" -> Feature boundary detected. Original tight crop resolution: {extraction_size}x{extraction_size}")

    half_size = extraction_size // 2

    # Calculate crop coordinates bounded by the image dimensions
    y1 = max(0, int(center_y) - half_size)
    y2 = min(img_h, int(center_y) + half_size)
    x1 = max(0, int(center_x) - half_size)
    x2 = min(img_w, int(center_x) + half_size)

    patch = image[y1:y2, x1:x2]

    # Pad with black if the bounding box hit the physical edge of the source image
    if patch.shape[0] != extraction_size or patch.shape[1] != extraction_size:
        patch = cv2.copyMakeBorder(
            patch,
            max(0, half_size - int(center_y)), max(0, (int(center_y) + half_size) - img_h),
            max(0, half_size - int(center_x)), max(0, (int(center_x) + half_size) - img_w),
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

    # scale the tight crop to the IP-Adapter requirement using high-frequency preservation
    print(f" -> scaling {extraction_size}x{extraction_size} patch to {output_size}x{output_size}")
    ip_adapter_ready_patch = cv2.resize(patch, (output_size, output_size), interpolation=cv2.INTER_CUBIC)

    return ip_adapter_ready_patch


def get_face_landmarks(image):
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            print("ERROR: No face detected in the master frame.")
            return None

        print("Face detected. Calculating topological coordinates...")
        return results.multi_face_landmarks[0].landmark


def get_image_and_landmarks(source_frame_path):
    os.makedirs("output/tmp", exist_ok=True)
    print(f"Loading Master Frame: {source_frame_path}")
    image = cv2.imread(source_frame_path)
    if image is None:
        cap = cv2.VideoCapture(source_frame_path)
        if cap.isOpened():
            ret, image = cap.read()
            cap.release()
            if not ret:
                raise FileNotFoundError("Could not find the source frame.")

    landmarks = get_face_landmarks(image)
    return image, landmarks


if __name__ == "__main__":
    output_dir_tmp = "output/tmp"
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, help="The source argument value")
    args, unknown_args = parser.parse_known_args()

    if not args.s:
        print("No -s source file argument was provided.")
        exit(1)

    source_file_path = args.s
    source_label = Path(source_file_path).stem

    image, landmarks = get_image_and_landmarks(source_file_path)

    if landmarks:
        out_file_name = f"{output_dir_tmp}/eye_reference-{source_label}.png"
        print(f"Processing Right Eye Anchor {out_file_name}")
        # Array of perimeter indices for the Right Eye (Viewer's Left)
        right_eye_boundary_indices = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        eye_patch = extract_semantic_square(image, landmarks, right_eye_boundary_indices, output_size=512)
        cv2.imwrite(out_file_name, eye_patch)

        print(f"Processing Mouth Anchor {out_file_name}")
        # Array of perimeter indices for the Outer Lips
        mouth_boundary_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40]
        mouth_patch = extract_semantic_square(image, landmarks, mouth_boundary_indices, output_size=512)
        out_file_name = f"{output_dir_tmp}/mouth_reference-{source_label}.png"
        cv2.imwrite(out_file_name, mouth_patch)

        print("References generated successfully.")
