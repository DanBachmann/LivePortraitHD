import os
import glob
import json
#import urllib.request
#import traceback
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms, models
import matplotlib.pyplot as plt
import mediapipe as mp
import lpips

# Try to import InsightFace for CSIM. If it fails, we gracefully skip it.
try:
    from insightface.app import FaceAnalysis
    from numpy.linalg import norm
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[WARNING] 'insightface' is not installed. CSIM (Identity Preservation) will be skipped.")
    print("          To enable, run: pip install insightface onnxruntime")

class TextureLossEvaluator:
    """Calculates Neural Style/Texture Loss using a VGG16 Gram Matrix."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Initializing VGG16 for Texture Analysis...")
        vgg = models.vgg16(pretrained=True).features.to(self.device).eval()
        self.feature_layers = {'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3', '22': 'relu4_3'}
        self.model = vgg
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_features(self, image_tensor):
        features = {}
        x = image_tensor
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.feature_layers:
                features[self.feature_layers[name]] = x
        return features

    def gram_matrix(self, tensor):
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram / (d * h * w)

    def compute_loss(self, source_img_bgr, target_img_bgr):
        source_rgb = cv2.resize(cv2.cvtColor(source_img_bgr, cv2.COLOR_BGR2RGB), (256, 256))
        target_rgb = cv2.resize(cv2.cvtColor(target_img_bgr, cv2.COLOR_BGR2RGB), (256, 256))
        
        source_t = self.transform(source_rgb).unsqueeze(0).to(self.device)
        target_t = self.transform(target_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            source_features = self.get_features(source_t)
            target_features = self.get_features(target_t)
            
            style_loss = 0
            for layer in source_features:
                source_gram = self.gram_matrix(source_features[layer])
                target_gram = self.gram_matrix(target_features[layer])
                layer_loss = torch.mean((source_gram - target_gram) ** 2)
                style_loss += layer_loss.item()
                
        return style_loss

class MetricsSuite:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.texture_eval = TextureLossEvaluator()

        print("Initializing MediaPipe FaceMesh...")
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

        self.face_app = None
        if INSIGHTFACE_AVAILABLE:
            print("Initializing ArcFace (InsightFace) for CSIM...")
            self.face_app = FaceAnalysis(name='buffalo_l')
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        print("Initializing LPIPS (with spatial=True)...")
        # spatial=True is required here to return a 2D map instead of a single averaged scalar
        self.lpips_vgg = lpips.LPIPS(net='vgg', spatial=True).to(self.device)
        self.lpips_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def evaluate_texture(self, source_img, target_img):
        return self.texture_eval.compute_loss(source_img, target_img)

    def evaluate_lmd(self, base_img, target_img):
        res_base = self.mp_face_mesh.process(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))
        res_target = self.mp_face_mesh.process(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))

        if not res_base.multi_face_landmarks or not res_target.multi_face_landmarks:
            return None
        
        lm_base = np.array([[lm.x, lm.y, lm.z] for lm in res_base.multi_face_landmarks[0].landmark])
        lm_target = np.array([[lm.x, lm.y, lm.z] for lm in res_target.multi_face_landmarks[0].landmark])
        
        return np.mean(np.linalg.norm(lm_base - lm_target, axis=1))

    def evaluate_csim(self, source_img, target_img):
        if not self.face_app:
            return None
            
        faces_source = self.face_app.get(source_img)
        faces_target = self.face_app.get(target_img)
        
        if not faces_source or not faces_target:
            return None
            
        emb1 = faces_source[0].embedding
        emb2 = faces_target[0].embedding
        csim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
        return float(csim)

    def evaluate_radial_power_spectrum(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (512, 512)).astype(np.float64)
        
        h, w = gray.shape
        window_y = np.hanning(h)
        window_x = np.hanning(w)
        window_2d = np.outer(window_y, window_x)
        
        gray_windowed = gray * window_2d
        
        f = np.fft.fft2(gray_windowed)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift) ** 2
        
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[0:h, 0:w]
        r = np.hypot(x - cx, y - cy).astype(int)
        
        tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())
        
        nr[nr == 0] = 1 
        radialprofile = tbin / nr
        
        kernel_size = 3
        kernel = np.ones(kernel_size) / kernel_size
        radialprofile = np.convolve(radialprofile, kernel, mode='same')
        
        return np.log10(radialprofile + 1e-8)

    def evaluate_masked_lpips(self, uncalibrated_img, target_img, mask_img, evaluate_skin=True):
        """
        Calculates LPIPS on a specific region defined by the mask.
        Images are downscaled to 512x512 max to prevent CUDA OOM on 8GB GPUs.
        """
        if uncalibrated_img is None or mask_img is None:
            return None

        def _center_crop_to_match(img, target_h, target_w):
            h, w = img.shape[:2]
            if h == target_h and w == target_w:
                return img
            start_y = max(0, (h - target_h) // 2)
            start_x = max(0, (w - target_w) // 2)
            return img[start_y:start_y+target_h, start_x:start_x+target_w]

        target_h, target_w = target_img.shape[:2]
        
        # Force the uncalibrated image and mask to be the same cropped dimensions
        uncalibrated_img = _center_crop_to_match(uncalibrated_img, target_h, target_w)
        mask_img = _center_crop_to_match(mask_img, target_h, target_w)

        # OOM FIX: Downscale large images to 512x512 before passing to VGG
        MAX_DIM = 512
        if target_h > MAX_DIM or target_w > MAX_DIM:
            scale = MAX_DIM / max(target_h, target_w)
            new_w = int(target_w * scale)
            new_h = int(target_h * scale)
            
            uncalibrated_img = cv2.resize(uncalibrated_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            target_img = cv2.resize(target_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask_img = cv2.resize(mask_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        uncal_rgb = cv2.cvtColor(uncalibrated_img, cv2.COLOR_BGR2RGB)
        target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        if len(mask_img.shape) == 3:
            mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask_img

        if evaluate_skin:
            # 1.0 for Skin (black in original mask)
            eval_mask = 1.0 - (mask_gray.astype(np.float32) / 255.0)
        else:
            # 1.0 for Occlusions/Mouth (white in original mask)
            eval_mask = mask_gray.astype(np.float32) / 255.0

        uncal_t = self.lpips_transform(uncal_rgb).unsqueeze(0).to(self.device)
        target_t = self.lpips_transform(target_rgb).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(eval_mask).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist_map = self.lpips_vgg(uncal_t, target_t, normalize=True)
            
            mask_resized = torch.nn.functional.interpolate(
                mask_t, size=dist_map.shape[-2:], mode='bilinear', align_corners=False
            )

            masked_dist = dist_map * mask_resized
            mask_sum = mask_resized.sum()
            
            if mask_sum > 0:
                final_score = masked_dist.sum() / mask_sum
            else:
                if not evaluate_skin:
                    return None 
                final_score = dist_map.mean()

        # OOM FIX: Explicitly free VRAM
        del uncal_t, target_t, mask_t, dist_map, mask_resized, masked_dist
        torch.cuda.empty_cache()

        return final_score.item()

def generate_report_plots(file_label, results, rps_data, output_dir):
    methods = [m for m in results.keys() if m != "Source HR"]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Quality Metrics Report: {file_label}", fontsize=20, fontweight='bold')
    
    COLORS = {
        "Source HR": "#666666",
        "Baseline LP": "#ffcc99",
        "CodeFormer_Base": "#ff9933",
        "Real-ESRGAN": "#99ccff",
        "CodeFormer_ESRGAN": "#3399ff",
        "Final HD": "#c2c2f0",
        "CodeFormer_FinalHD": "#8a8ae6"
    }

    # 1. LPIPS Skin vs Occlusions (Grouped Chart)
    ax = axes[0, 0]
    # Filter labels to only include methods that successfully computed LPIPS
    labels = [m for m in methods if results.get(m, {}).get("LPIPS_Skin") is not None]
    
    if labels:
        skin_vals = [results[m]["LPIPS_Skin"] for m in labels]
        occ_vals = [results[m]["LPIPS_Occlusion"] for m in labels]
        
        # Re-calculate 'x' positions based strictly on the number of valid labels
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, skin_vals, width, label='Masked (Skin Only)', color=[COLORS.get(l, '#cccccc') for l in labels])
        ax.bar(x + width/2, occ_vals, width, label='Unmasked (Occlusions)', color=[COLORS.get(l, '#cccccc') for l in labels], alpha=0.6, hatch='///', edgecolor='black')
        
        ax.set_title("LPIPS: Skin vs Occlusions (Lower = Better Spatial Alignment)")
        ax.set_ylabel("LPIPS Distance")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "LPIPS N/A (Missing Mask or Data)", ha='center', va='center')
        ax.set_title("LPIPS: Skin vs Occlusions")
    
    # 2. Neural Texture Loss
    ax = axes[0, 1]
    vals = [results[m]["Texture_Loss_Vs_Source"] for m in methods if results[m].get("Texture_Loss_Vs_Source") is not None]
    labels_tex = [m for m in methods if results[m].get("Texture_Loss_Vs_Source") is not None]
    if vals:
        ax.bar(labels_tex, vals, color=[COLORS.get(l, '#cccccc') for l in labels_tex])
        ax.set_title("Texture Loss vs Source (Lower = Better Texture Match)")
        ax.set_ylabel("Gram Matrix MSE")
        ax.tick_params(axis='x', rotation=45)

    # 3. CSIM
    ax = axes[1, 0]
    vals = [results[m]["CSIM_Vs_Source"] for m in methods if results[m].get("CSIM_Vs_Source") is not None]
    labels_csim = [m for m in methods if results[m].get("CSIM_Vs_Source") is not None]
    if vals:
        ax.bar(labels_csim, vals, color=[COLORS.get(l, '#cccccc') for l in labels_csim])
        ax.set_title("ArcFace Identity Similarity (CSIM) (Higher = Better Likeness)")
        ax.set_ylabel("Cosine Similarity")
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis='x', rotation=45)

    # 4. Radial Power Spectrum
    ax = axes[1, 1]
    for method, profile in rps_data.items():
        ax.plot(profile[:250], label=method, linewidth=2 if method == "Final HD" else 1.5, alpha=0.8, color=COLORS.get(method, '#333333'))
    
    ax.set_title("Radial Power Spectrum (Higher on right = Sharper Micro-Details)")
    ax.set_xlabel("Spatial Frequency")
    ax.set_ylabel("Log Power")
    ax.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plot_path = os.path.join(output_dir, f"{file_label}_metrics_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

def run_evaluation(base_dir="output/tmp", out_dir="output/metrics"):
    os.makedirs(out_dir, exist_ok=True)
    suite = MetricsSuite()
    
    search_pattern = os.path.join(base_dir, "06_assembled*.png")
    final_files = glob.glob(search_pattern)
    
    if not final_files:
        print(f"No assembled files found matching {search_pattern}")
        return

    for final_path in final_files:
        filename = os.path.basename(final_path)
        file_label = filename.replace("06_assembled", "").replace(".png", "")
        source_file_label = file_label.split('-')[0]
        
        print(f"\n{'='*50}\nEvaluating Set: {file_label}\n{'='*50}")
        
        paths = {
            "Source HR": os.path.join(base_dir, f"02-2_upright_4k-{source_file_label}.png"),
            "Baseline LP": os.path.join(base_dir, f"01_plate_LP-{file_label}.png"),
            "CodeFormer_Base": os.path.join(base_dir, f"01_plate_CodeFormer_Base-{file_label}.png"),
            "Real-ESRGAN": os.path.join(base_dir, f"01_plate_AI_Base-{file_label}.png"),
            "CodeFormer_ESRGAN": os.path.join(base_dir, f"01_plate_CodeFormer_ESRGAN-{file_label}.png"),
            "Final HD": final_path,
            "CodeFormer_FinalHD": os.path.join(base_dir, f"01_plate_CodeFormer_FinalHD-{file_label}.png")
        }

        uncalibrated_path = os.path.join(base_dir, f"02_plate_4K_Warp_RGB-{file_label}.png")
        mask_path = os.path.join(base_dir, f"03_plate_Alpha_Mask-{file_label}.png")
        
        images = {}
        for method, path in paths.items():
            if os.path.exists(path):
                img = cv2.imread(path)
                # CRITICAL OpenCV FIX: Check if image object is valid and non-empty
                if img is not None and img.size > 0:
                    images[method] = img
                else:
                    print(f"[WARNING] Skipping {path} (Failed to load or image is corrupted)")
            else:
                pass 

        uncalibrated_img = cv2.imread(uncalibrated_path) if os.path.exists(uncalibrated_path) else None
        if uncalibrated_img is not None and uncalibrated_img.size == 0:
            uncalibrated_img = None

        mask_img = cv2.imread(mask_path) if mask_path and os.path.exists(mask_path) else None
        if mask_img is not None and mask_img.size == 0:
            mask_img = None

        if "Source HR" not in images or "Baseline LP" not in images:
            print(f"[WARNING] Skipping {file_label} due to missing Source or Baseline.")
            continue
            
        source_img = images["Source HR"]
        baseline_img = images["Baseline LP"]
        
        results = {}
        rps_data = {}
        
        rps_data["Source HR"] = suite.evaluate_radial_power_spectrum(source_img)
        results["Source HR"] = {"RPS": rps_data["Source HR"].tolist()}
        
        for method, img in images.items():
            if method == "Source HR":
                continue
                
            print(f"  Calculating metrics for: {method}...")
            res = {}
            
            # 1. Evaluate LPIPS for ALL methods against the uncalibrated plate
            if uncalibrated_img is not None and mask_img is not None:
                res["LPIPS_Skin"] = suite.evaluate_masked_lpips(uncalibrated_img, img, mask_img, evaluate_skin=True)
                res["LPIPS_Occlusion"] = suite.evaluate_masked_lpips(uncalibrated_img, img, mask_img, evaluate_skin=False)
            
            # 2. Texture Loss
            res["Texture_Loss_Vs_Source"] = suite.evaluate_texture(source_img, img)
            
            # 3. LMD
            res["LMD_Vs_Baseline"] = suite.evaluate_lmd(baseline_img, img)
            
            # 4. CSIM
            res["CSIM_Vs_Source"] = suite.evaluate_csim(source_img, img)
            
            # 5. RPS
            rps_arr = suite.evaluate_radial_power_spectrum(img)
            rps_data[method] = rps_arr
            res["RPS"] = rps_arr.tolist()
            
            results[method] = res
            torch.cuda.empty_cache()

        json_path = os.path.join(out_dir, f"{file_label}_metrics.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f" -> Saved Data: {json_path}")
        
        generate_report_plots(file_label, results, rps_data, out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive High-Res Metrics Suite")
    parser.add_argument("--dir", type=str, default="output/tmp", help="Base directory containing the files")
    parser.add_argument("--out", type=str, default="output/metrics", help="Directory to save the reports")
    args = parser.parse_args()
    
    run_evaluation(base_dir=args.dir, out_dir=args.out)
    print("\nMetrics evaluation complete.")