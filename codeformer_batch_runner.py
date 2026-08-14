import os
import glob
import subprocess
import argparse

def run_codeformer_batch(codeformer_dir, input_dir="output/tmp", w=0.5):
    """
    Runs CodeFormer inference on the three required baselines.
    Assumes CodeFormer is installed locally in `codeformer_dir`.
    """
    print(f"\n{'='*50}\nStarting CodeFormer Batch Processing\n{'='*50}")
    
    # 1. Define the input prefixes we want to process
    # The dictionary maps the input prefix to the desired output suffix
    targets = {
        "01_plate_LP-": "CodeFormer_Base",
        "01_plate_AI_Base-": "CodeFormer_ESRGAN",
        "06_assembled": "CodeFormer_FinalHD" 
    }

    inference_script = os.path.join(codeformer_dir, "inference_codeformer.py")
    if not os.path.exists(inference_script):
        print(f"[ERROR] Could not find inference_codeformer.py in {codeformer_dir}")
        print("Please provide the correct path to your CodeFormer installation directory.")
        return

    # Create a temporary directory inside output/tmp to hold the raw CodeFormer outputs
    # CodeFormer creates its own subfolders (like 'results/whole_imgs_0.5/restored_imgs')
    temp_cf_out = os.path.join(input_dir, "cf_raw_output")
    os.makedirs(temp_cf_out, exist_ok=True)

    for prefix, suffix_label in targets.items():
        print(f"\n--- Processing Branch: {suffix_label} ---")
        
        # Find all files matching the prefix
        search_pattern = os.path.join(input_dir, f"{prefix}*.png")
        files_to_process = glob.glob(search_pattern)
        
        if not files_to_process:
            print(f"  No files found matching {search_pattern}. Skipping.")
            continue

        print(f"  Found {len(files_to_process)} files to process.")
        
        # Process each file individually to control the exact output naming
        for file_path in files_to_process:
            filename = os.path.basename(file_path)
            
            # Extract the unique file label (e.g., "DanByNickGregan01-d9")
            file_label = filename.replace(prefix, "").replace(".png", "")
            
            # The final filename we want for our metrics script
            final_out_name = f"01_plate_{suffix_label}-{file_label}.png"
            final_out_path = os.path.join(input_dir, final_out_name)
            
            # Skip if already processed
            if os.path.exists(final_out_path):
                print(f"  Skipping {final_out_name} (Already exists)")
                continue
                
            print(f"  Running CodeFormer on {filename}...")
            
            # Construct the CodeFormer command
            # Convert paths to absolute because we are changing the cwd for the subprocess
            abs_file_path = os.path.abspath(file_path)
            abs_temp_cf_out = os.path.abspath(temp_cf_out)
            
            cmd = [
                "python", inference_script,
                "-w", str(w),
                "--input_path", abs_file_path,
                "--output_path", abs_temp_cf_out
            ]
            
            # Run the command (suppressing the massive output CodeFormer usually prints)
            try:
                subprocess.run(cmd, cwd=codeformer_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                print(f"  [ERROR] CodeFormer failed on {filename}:")
                print(e.stderr.decode('utf-8'))
                continue
                
            # CodeFormer saves the output in a nested structure inside our temp folder.
            # It usually looks like: cf_raw_output/restored_imgs/filename.png
            # Or: cf_raw_output/whole_imgs_0.5/filename.png (depends on the exact CodeFormer version/args)
            
            # We need to find the output image and move/rename it.
            # Search recursively in the temp folder for the filename
            cf_generated_files = glob.glob(os.path.join(temp_cf_out, "**", filename), recursive=True)
            
            if cf_generated_files:
                # Move and rename the file to our target location
                os.rename(cf_generated_files[0], final_out_path)
                print(f"  -> Saved as {final_out_name}")
            else:
                print(f"  [WARNING] Could not locate the output file for {filename} in {temp_cf_out}")

    print("\nCleaning up temporary folders...")
    # Clean up the CodeFormer nested folders (if empty)
    try:
        import shutil
        shutil.rmtree(temp_cf_out)
    except Exception as e:
        print(f"Could not remove temporary folder {temp_cf_out}: {e}")

    print(f"\n{'='*50}\nCodeFormer Batch Processing Complete!\n{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CodeFormer on Baseline Images")
    parser.add_argument("--cf_dir", type=str, required=True, help="Absolute path to your local CodeFormer repository")
    parser.add_argument("--dir", type=str, default="output/tmp", help="Base directory containing the files to process")
    parser.add_argument("-w", type=float, default=0.5, help="Fidelity weight (0.0 to 1.0). Default is 0.5")
    
    args = parser.parse_args()
    
    run_codeformer_batch(codeformer_dir=args.cf_dir, input_dir=args.dir, w=args.w)