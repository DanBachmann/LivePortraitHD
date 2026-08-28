import os
import glob
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from math import pi

METRICS_DIR = "output/metrics"
OUTPUT_DIR = os.path.join(METRICS_DIR, "summary_report")

# Expected methods in the JSON files
METHODS = [
    "Baseline LP",
    "CodeFormer_Base",
    "Real-ESRGAN",
    "CodeFormer_ESRGAN",
    "Final HD",
    "CodeFormer_FinalHD"
]

# Colors matching previous reports, with CodeFormer pairs
COLORS = {
    "Source HR": "#666666",
    "Baseline LP": "#ffcc99",
    "CodeFormer_Base": "#ff9933",
    "Real-ESRGAN": "#99ccff",
    "CodeFormer_ESRGAN": "#3399ff",
    "Final HD": "#c2c2f0",
    "CodeFormer_FinalHD": "#8a8ae6"
}

# Define which metrics are "Lower is Better" vs "Higher is Better"
METRIC_RULES = {
    "Texture_Loss_Vs_Source": {
        "name": "Texture Loss (Gram Matrix)",
        "lower_is_better": True,
        "radar_range": (0.0, 0.0001)
    },
    "LMD_Vs_Baseline": {
        "name": "Landmark Distance (Expression)",
        "lower_is_better": True,
        "radar_range": (0.0, 0.015),
        "skip_baseline": True
    },
    "CSIM_Vs_Source": {
        "name": "CSIM (Identity Similarity)",
        "lower_is_better": False,
        "radar_range": (0.5, 1.0)
    },
    "LPIPS_Skin": {
        "name": "Masked LPIPS (Skin Alignment)",
        "lower_is_better": True,
        "radar_range": (0.0, 0.4)
    },
    "LPIPS_Occlusion": {
        "name": "Unmasked LPIPS (Occlusions)",
        "lower_is_better": True,
        "radar_range": (0.0, 0.4)
    }
}

def load_all_json_data(metrics_dir):
    search_pattern = os.path.join(metrics_dir, "*_metrics.json")
    json_files = glob.glob(search_pattern)

    if not json_files:
        print(f"No JSON files found in {metrics_dir}")
        return None

    print(f"Found {len(json_files)} metric files. Aggregating data...")

    raw_data = {metric: {method: [] for method in METHODS} for metric in METRIC_RULES.keys()}
    frame_data = {}
    rps_data = {method: [] for method in METHODS + ["Source HR"]}

    for filepath in json_files:
        filename = os.path.basename(filepath)
        file_label = filename.replace("_metrics.json", "")

        with open(filepath, 'r') as f:
            data = json.load(f)

        frame_data[file_label] = data

        for method in METHODS + ["Source HR"]:
            if method in data:
                if method in METHODS:
                    for metric in METRIC_RULES.keys():
                        if metric in data[method] and data[method][metric] is not None:
                            # Skip baseline for LMD
                            if metric == "LMD_Vs_Baseline" and method == "Baseline LP":
                                continue
                            raw_data[metric][method].append(data[method][metric])

                # Collect RPS array if present
                if "RPS" in data[method] and data[method]["RPS"] is not None:
                    rps_data[method].append(data[method]["RPS"])

    return raw_data, frame_data, rps_data

def compute_summary_statistics(raw_data):
    stats = {metric: {} for metric in METRIC_RULES.keys()}

    for metric in METRIC_RULES.keys():
        for method in METHODS:
            if not raw_data[metric][method]:
                continue
            values = raw_data[metric][method]
            stats[metric][method] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std_dev": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values)
            }
    return stats

def export_csv_and_json(stats, frame_data, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Export JSON Summary
    json_path = os.path.join(out_dir, "summary_statistics.json")
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Saved: {json_path}")

    # 2. Export Full CSV
    csv_path = os.path.join(out_dir, "all_frames_raw_data.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        headers = ["Frame_Label"]
        for method in METHODS:
            for metric in METRIC_RULES.keys():
                headers.append(f"{method}_{metric}")
        writer.writerow(headers)

        for frame_label, data in frame_data.items():
            row = [frame_label]
            for method in METHODS:
                for metric in METRIC_RULES.keys():
                    val = data.get(method, {}).get(metric, "N/A")
                    row.append(val)
            writer.writerow(row)
    print(f"Saved: {csv_path}")

def plot_box_and_whisker(raw_data, out_dir):
    # Separate LPIPS from standard plotting
    standard_metrics = [m for m in METRIC_RULES.keys() if "LPIPS" not in m]
    num_metrics = len(standard_metrics) + 1 # +1 for the combined LPIPS chart
    cols = 2
    rows = (num_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
    fig.suptitle("Metric Distributions Across All Frames (Box & Whisker)", fontsize=20, fontweight='bold', y=0.98)
    axes = np.atleast_1d(axes).flatten()

    # 1. Plot standard metrics
    for i, metric in enumerate(standard_metrics):
        rules = METRIC_RULES[metric]
        ax = axes[i]
        plot_data, labels, colors = [], [], []

        for method in METHODS:
            if method in raw_data[metric] and raw_data[metric][method]:
                plot_data.append(raw_data[metric][method])
                labels.append(method)
                colors.append(COLORS.get(method, "#333"))

        if not plot_data:
            continue

        bplot = ax.boxplot(plot_data, labels=labels, patch_artist=True, notch=True)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_title(f"{rules['name']}\n({'Lower' if rules['lower_is_better'] else 'Higher'} is Better)")
        ax.set_ylabel("Score")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        if metric == "CSIM_Vs_Source": ax.set_ylim(0.5, 1.0)

    # 2. Plot combined LPIPS
    ax = axes[len(standard_metrics)]
    plot_data, labels, colors = [], [], []
    for method in METHODS:
        if raw_data["LPIPS_Skin"].get(method):
            plot_data.append(raw_data["LPIPS_Skin"][method])
            labels.append(f"{method}\n(Skin)")
            colors.append(COLORS.get(method, "#333"))

            plot_data.append(raw_data["LPIPS_Occlusion"][method])
            labels.append(f"{method}\n(Occ)")
            colors.append(COLORS.get(method, "#333"))

    if plot_data:
        bplot = ax.boxplot(plot_data, labels=labels, patch_artist=True, notch=True)
        for i, (patch, color) in enumerate(zip(bplot['boxes'], colors)):
            patch.set_facecolor(color)
            if i % 2 == 1: # Unmasked (every second item)
                patch.set_hatch('///')
                patch.set_alpha(0.5)
            else:
                patch.set_alpha(0.8)

        ax.set_title("LPIPS: Masked Skin vs Unmasked Occlusions\n(Lower is Better Spatial Alignment)")
        ax.set_ylabel("LPIPS Distance")
        ax.tick_params(axis='x', rotation=90)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # 3. Clean up empty axes
    for j in range(num_metrics, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "plot_01_boxplots.png"), dpi=150)
    plt.close()

def plot_bar_charts_with_error(stats, out_dir):
    # Separate LPIPS from standard plotting
    standard_metrics = [m for m in METRIC_RULES.keys() if "LPIPS" not in m]
    num_metrics = len(standard_metrics) + 1 # +1 for combined LPIPS
    cols = 2
    rows = (num_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
    fig.suptitle("Average Performance with Variance (Mean ± StdDev)", fontsize=20, fontweight='bold', y=0.98)
    axes = np.atleast_1d(axes).flatten()

    # 1. Plot standard metrics
    for i, metric in enumerate(standard_metrics):
        rules = METRIC_RULES[metric]
        ax = axes[i]
        means, stds, labels, colors = [], [], [], []

        for method in METHODS:
            if method in stats[metric] and stats[metric][method]:
                means.append(stats[metric][method]["mean"])
                stds.append(stats[metric][method]["std_dev"])
                labels.append(method)
                colors.append(COLORS.get(method, "#333"))

        if not means: continue

        x_pos = np.arange(len(labels))
        ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.8, color=colors, capsize=10, ecolor='black')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title(f"{rules['name']}\n({'Lower' if rules['lower_is_better'] else 'Higher'} is Better)")
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        if metric == "CSIM_Vs_Source": ax.set_ylim(0.5, 1.0)

    # 2. Plot combined Grouped Bar Chart for LPIPS
    ax = axes[len(standard_metrics)]
    valid_methods = [m for m in METHODS if stats["LPIPS_Skin"].get(m)]

    if valid_methods:
        skin_means = [stats["LPIPS_Skin"][m]["mean"] for m in valid_methods]
        skin_stds = [stats["LPIPS_Skin"][m]["std_dev"] for m in valid_methods]
        occ_means = [stats["LPIPS_Occlusion"][m]["mean"] for m in valid_methods]
        occ_stds = [stats["LPIPS_Occlusion"][m]["std_dev"] for m in valid_methods]
        method_colors = [COLORS.get(m, "#333") for m in valid_methods]

        x_pos = np.arange(len(valid_methods))
        width = 0.35

        ax.bar(x_pos - width/2, skin_means, width, yerr=skin_stds, label='Masked (Skin Only)', color=method_colors, alpha=0.8, capsize=5, ecolor='black')
        ax.bar(x_pos + width/2, occ_means, width, yerr=occ_stds, label='Unmasked (Full Image)', color=method_colors, alpha=0.5, hatch='///', capsize=5, ecolor='black')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(valid_methods, rotation=45, ha='right')
        ax.set_title("LPIPS: Masked Skin vs Unmasked Occlusions\n(Lower is Better Spatial Alignment)")
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.legend(fontsize=9)

    # 3. Clean up empty axes
    for j in range(num_metrics, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "plot_02_barcharts.png"), dpi=150)
    plt.close()

def plot_normalized_radar_chart(stats, out_dir):
    # Uses ALL 5 metrics
    categories = list(METRIC_RULES.keys())
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.suptitle("Overall Method Assessment (Normalized Radar Chart)\nFurther Outside = Better Relative Performance", fontsize=16, fontweight='bold', y=1.05)

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    display_names = [METRIC_RULES[c]["name"].replace(" ", "\n") for c in categories]
    plt.xticks(angles[:-1], display_names, color='black', size=12)

    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["", "", "", ""], color="grey", size=10)
    plt.ylim(0, 1)

    for method in METHODS:
        values = []
        for cat in categories:
            if method not in stats[cat] or not stats[cat][method]:
                values.append(0.1)
                continue

            val = stats[cat][method]["mean"]
            range_min, range_max = METRIC_RULES[cat]["radar_range"]

            norm_val = (val - range_min) / (range_max - range_min)
            norm_val = max(0.0, min(1.0, norm_val))

            if METRIC_RULES[cat]["lower_is_better"]:
                norm_val = 1.0 - norm_val

            visual_val = 0.1 + (0.9 * norm_val)
            values.append(visual_val)

        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=method, color=COLORS.get(method, "#333"))
        ax.fill(angles, values, color=COLORS.get(method, "#333"), alpha=0.25)

    plt.legend(loc='center left', bbox_to_anchor=(1.3, 0.5), title="Methods", fontsize=10, title_fontsize=12)
    plt.savefig(os.path.join(out_dir, "plot_03_radar_chart.png"), dpi=150, bbox_inches='tight')
    plt.close()

def plot_tradeoff_scatter(stats, out_dir):
    """Plots Texture Loss vs Masked LPIPS to show the generative trade-off."""
    if "LPIPS_Skin" not in stats or "Texture_Loss_Vs_Source" not in stats:
        return

    plt.figure(figsize=(10, 7))
    plt.title("Generative Trade-off: Texture Match vs Spatial Alignment\n(Bottom Left is Best)", fontsize=16, fontweight='bold')

    for method in METHODS:
        if method in stats["LPIPS_Skin"] and method in stats["Texture_Loss_Vs_Source"]:
            if stats["LPIPS_Skin"][method] and stats["Texture_Loss_Vs_Source"][method]:
                x_val = stats["LPIPS_Skin"][method]["mean"]
                y_val = stats["Texture_Loss_Vs_Source"][method]["mean"]

                plt.scatter(x_val, y_val, color=COLORS.get(method, "#333"), s=300, edgecolor='black', zorder=5, label=method)

    plt.xlabel("Masked LPIPS (Skin Alignment)", fontsize=12)
    plt.ylabel("Texture Loss vs Source (Gram Matrix)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add an ideal zone annotation
    plt.axvspan(0, 0.1, color='green', alpha=0.05, zorder=1)
    plt.axhspan(0, 0.00002, color='green', alpha=0.05, zorder=1)
    plt.text(0.01, 0.000005, "Ideal Zone", color='green', fontsize=12, fontweight='bold', alpha=0.5)

    # Clean, color-coded legend placed safely outside the chart bounds
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title="Methods", fontsize=10, title_fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_04_tradeoff_scatter.png"), dpi=150, bbox_inches='tight')
    plt.close()

def plot_average_rps(rps_data, out_dir):
    plt.figure(figsize=(10, 6))
    plt.title("Average Radial Power Spectrum Across All Frames\n(Higher on right = Sharper Micro-Details)", fontsize=16, fontweight='bold')

    has_data = False
    for method in ["Source HR"] + METHODS:
        if method in rps_data and rps_data[method]:
            has_data = True
            arr = np.array(rps_data[method])
            mean_rps = np.mean(arr, axis=0)

            is_hd = (method == "Final HD")
            is_src = (method == "Source HR")

            plt.plot(
                mean_rps[:250],
                label=method,
                color=COLORS.get(method, "#333"),
                linewidth=2.5 if (is_hd or is_src) else 1.5,
                alpha=1.0 if is_src else 0.8,
                linestyle='dashed' if is_src else 'solid'
            )

    if not has_data:
        print("Skipping RPS Chart (No data found).")
        plt.close()
        return

    plt.xlabel("Spatial Frequency")
    plt.ylabel("Average Log Power")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_05_average_rps.png"), dpi=150)
    plt.close()

def run():
    print(f"\n{'='*50}\nStarting Summary Report Generation\n{'='*50}")

    res = load_all_json_data(METRICS_DIR)
    if not res:
        return

    raw_data, frame_data, rps_data = res
    stats = compute_summary_statistics(raw_data)

    export_csv_and_json(stats, frame_data, OUTPUT_DIR)
    plot_box_and_whisker(raw_data, OUTPUT_DIR)
    plot_bar_charts_with_error(stats, OUTPUT_DIR)
    plot_normalized_radar_chart(stats, OUTPUT_DIR)
    plot_tradeoff_scatter(stats, OUTPUT_DIR)
    plot_average_rps(rps_data, OUTPUT_DIR)

    print("\nSummary Report Generation Complete!")
    print(f"All reports have been saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run()
