import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import ast

DA_NAMES = [
    "Pad & RandomCrop", "HorizontalFlip", "VerticalFlip", "Rotate",
    "Affine (Translate)", "Affine (Shear)", "Perspective", "ElasticTransform",
    "ChannelShuffle", "ToGray", "GaussianBlur", "GaussNoise", "InvertImg",
    "Posterize", "Solarize", "Sharpen (Kernel)", "Sharpen (Gaussian)",
    "Equalize", "ImageCompression", "RandomGamma", "MedianBlur", "MotionBlur",
    "CLAHE", "RandomBrightnessContrast", "PlasmaBrightnessContrast",
    "CoarseDropout", "Blur", "HueSaturationValue", "ColorJitter",
    "RandomResizedCrop", "AutoContrast", "Erasing", "RGBShift",
    "PlanckianJitter", "ChannelDropout", "Illumination (Linear)",
    "Illumination (Corner)", "Illumination (Gaussian)", "PlasmaShadow",
    "RandomRain", "SaltAndPepper", "RandomSnow", "OpticalDistortion",
    "ThinPlateSpline"
]

def parse_genotype(genotype_str):
    try:
        data = ast.literal_eval(genotype_str)
        if isinstance(data, list) and len(data) == 2 and isinstance(data[0], list) and isinstance(data[1], list):
            if len(data[0]) == 0:
                return data[1]
            return data
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []

def analyze_da_distribution(folder_path, experiment_name="DA Distribution"):
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not all_files:
        print(f"ERROR: No CSV files found in: {folder_path}")
        return

    da_counts = {i: 0 for i in range(len(DA_NAMES))}
    total_augmentations_count = 0
    total_seeds = 0

    for filename in all_files:
        try:
            df = pd.read_csv(filename, sep=';', on_bad_lines='skip')
            if 'best_fitness' not in df.columns or 'best_individual' not in df.columns:
                continue

            df['best_fitness'] = pd.to_numeric(df['best_fitness'], errors='coerce')
            df = df.dropna(subset=['best_fitness'])

            if df.empty:
                continue

            best_row = df.loc[df['best_fitness'].idxmax()]
            genotype_str = best_row['best_individual']
            augs_list = parse_genotype(genotype_str)

            if augs_list:
                total_seeds += 1
                for item in augs_list:
                    if isinstance(item, list) and len(item) > 0:
                        aug_id = item[0]
                        if isinstance(aug_id, int) and 0 <= aug_id < len(DA_NAMES):
                            da_counts[aug_id] += 1
                            total_augmentations_count += 1

        except Exception as e:
            print(f"Warning reading {os.path.basename(filename)}: {e}")

    active_augs = {DA_NAMES[k]: v for k, v in da_counts.items() if v > 0}

    if not active_augs:
        print("No augmentations found.")
        return

    sorted_augs = sorted(active_augs.items(), key=lambda item: item[1], reverse=True)
    
  
    print(f"\n--- TOP 5 PARA LATEX ({experiment_name} | {total_seeds} Seeds) ---")
    for i, (name, count) in enumerate(sorted_augs[:5]):
        percentage = (count / total_augmentations_count) * 100
        print(f"{i+1} & {name:25s} & {percentage:.1f}\\%")
    print("--------------------------------------------------------\n")

    labels = [k for k, v in sorted_augs]
    values = [v for k, v in sorted_augs]
    plt.figure(figsize=(12, 8))
    bars = plt.bar(labels, values, color='#4c72b0', edgecolor='black', alpha=0.9)
    plt.xlabel('Data Augmentation Strategy', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency (Count)', fontsize=12, fontweight='bold')
    plt.title(f'Most Selected Augmentations - {experiment_name}\n(Across {total_seeds} seeds)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    max_y = max(values)
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_augmentations_count) * 100
        plt.text(bar.get_x() + bar.get_width()/2, height + (max_y * 0.01), 
                 f"{percentage:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.ylim(0, max_y * 1.15) 
    plt.tight_layout()
    output_file = os.path.join(folder_path, f'DA_Distribution_{experiment_name}.png')
    plt.savefig(output_file, dpi=300)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        name = sys.argv[2] if len(sys.argv) > 2 else "MedMNIST"
        analyze_da_distribution(folder, name)
    else:
        print("Usage: python script.py <csv_folder> [Experiment_Name]")