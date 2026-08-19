import warnings
warnings.filterwarnings("ignore")
import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import os

def combine_images_side_by_side(image_list, target_h=None):
    """
    Takes a list of images (assumed to be the same size) and concatenates them horizontally.
    Automatically handles channel mismatches (e.g., sticking a 1-channel mask next to a 3-channel RGB image).
    """
    if not image_list:
        raise ValueError("The image list is empty.")

    if len(image_list) == 1:
        return image_list[0]

    processed_images = []
    max_channels = 1
    target_w = None
    MAX_SIZE = True

    # Load images, determine target height, and find the maximum channel depth
    if MAX_SIZE and target_h is None:
        for img in image_list:
            # Handle different input types cleanly
            if isinstance(img, str):
                cv_img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            elif isinstance(img, Image.Image):
                if img.mode == 'RGBA':
                    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
                else:
                    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            elif isinstance(img, np.ndarray):
                cv_img = img.copy()
            else:
                raise ValueError("Unsupported image type in list.")

            # Set the target height based on the first image to prevent NumPy crashes
            if cv_img is None:
                continue
            if target_h is None or target_h < cv_img.shape[0]:
                target_h = cv_img.shape[0]
                target_w = cv_img.shape[1]
    for img in image_list:
        # Handle different input types cleanly
        if isinstance(img, str):
            cv_img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        elif isinstance(img, Image.Image):
            if img.mode == 'RGBA':
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
            else:
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        elif isinstance(img, np.ndarray):
            cv_img = img.copy()
        else:
            raise ValueError("Unsupported image type in list.")

        # Set the target height based on the first image to prevent NumPy crashes
        if cv_img is None:
            continue
        elif target_h is None:
            target_h = cv_img.shape[0]
            target_w = cv_img.shape[1]
        elif cv_img.shape[0] != target_h:
            if target_w is None:
                target_w = cv_img.shape[1] * target_h // cv_img.shape[0]
            cv_img = cv2.resize(cv_img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(img.replace(".jpg","up.jpg"),cv_img)

        channels = cv_img.shape[2] if len(cv_img.shape) == 3 else 1
        if channels > max_channels:
            max_channels = channels

        processed_images.append(cv_img)

    # Second Pass: Normalize all images to the maximum channel count
    normalized_images = []
    for cv_img in processed_images:
        channels = cv_img.shape[2] if len(cv_img.shape) == 3 else 1

        if channels == max_channels:
            normalized_images.append(cv_img)
        elif max_channels == 4: # Target is BGRA
            if channels == 3:
                normalized_images.append(cv2.cvtColor(cv_img, cv2.COLOR_BGR2BGRA))
            elif channels == 1:
                bgr = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
                normalized_images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA))
        elif max_channels == 3: # Target is BGR
            if channels == 1:
                normalized_images.append(cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR))

    # Stitch them all together horizontally
    combined_image = np.hstack(normalized_images)

    # If the input was a PIL image, return a PIL image
    if isinstance(image_list[0], Image.Image):
        if max_channels == 4:
            return Image.fromarray(cv2.cvtColor(combined_image, cv2.COLOR_BGRA2RGBA))
        else:
            return Image.fromarray(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))

    return combined_image, len(processed_images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, help="The source argument value")
    parser.add_argument('-d', type=str, help="The driver argument value")
    parser.add_argument('-o', type=bool, help="Overwrite some pre-existing assembled images")
    args, unknown_args = parser.parse_known_args()

    overwrite = args.o
    if not args.s:
        print("No -s source file argument was provided.")
        exit(1)
    source_file_path = args.s
    if not args.d:
        print("No -d driver file argument was provided.")
        exit(1)
    driver_file_path = args.d

    if not os.path.exists(source_file_path):
        print(f"Source file {source_file_path} does not exist.")
        exit(1)
    source_name = Path(source_file_path).stem
    driving_name = Path(driver_file_path).stem
    finals = [f"animations/{source_name}--{driving_name}.jpg",
         f"animations/{source_name}--{driving_name}-hd.png",
         #f"animations/{source_name}--{driving_name}-nip.png",
         ]
    output_file = f"animations/{source_name}--{driving_name}-combined.png"
    if not overwrite or not Path(output_file).exists():
        combined_image, processed_image_count = combine_images_side_by_side (finals)
        if processed_image_count > 1:
            cv2.imwrite(output_file, combined_image)

    output_dir_tmp = "output/tmp"
    file_label = f"{source_name}-{driving_name}"

    output_file = f"{output_dir_tmp}/compare/{file_label}-comparison.png"
    if not overwrite or not Path(output_file).exists():
        cropped = [f"{output_dir_tmp}/01_plate_AI_base-{file_label}.png",
            f"{output_dir_tmp}/03-4_refine_warped-{file_label}.png",
            f"{output_dir_tmp}/06_assembled{file_label}.png"]
        combined_image, processed_image_count = combine_images_side_by_side (finals)
        if processed_image_count > 1:
            cv2.imwrite(output_file, combined_image)
