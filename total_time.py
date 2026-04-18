import os
import glob
import pandas as pd
import numpy as np

def calculate_time_statistics(results_folder, dataset, model):

    pattern = os.path.join(results_folder, f"ONT_DAOP_{dataset}_{model}_*.csv")
    files = glob.glob(pattern)
    
    seed_times = []

    if not files:
        print(f"No files found in '{results_folder}'")
        return

    for f in files:
        try:
            df = pd.read_csv(f, sep=';')
            
            if 'total_time' in df.columns:
                time_values = pd.to_numeric(df['total_time'], errors='coerce').dropna()
                
                seed_time = time_values.sum()
                seed_times.append(seed_time)
            else:
                print(f"Column 'total_time' not found in {os.path.basename(f)}")
        
        except Exception as e:
            print(f"Error reading {os.path.basename(f)}: {e}")

    if not seed_times:
        print("No time data found to process.")
        return

    total_accumulated_time = sum(seed_times)
    mean_time = np.mean(seed_times)
    std_time = np.std(seed_times)

    def format_hms(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}h {m}m {s}s"

    print(f"\nCOMPUTATIONAL COST ANALYSIS: {dataset} - {model}")
    print(f"--------------------------------------------------")
    print(f"Processed seeds: {len(seed_times)}")
    print(f"Total Time (sum of all seeds): {format_hms(total_accumulated_time)} ({total_accumulated_time:.2f}s)")
    print(f"Average per seed: {format_hms(mean_time)} ({mean_time:.2f}s)")
    print(f"Standard deviation (STD): {std_time:.2f}s")
    print(f"--------------------------------------------------")
    
    print(f"For Excel (Average ± STD): {mean_time:.2f} ± {std_time:.2f}")
    print(f"In hours (Average ± STD): {mean_time/3600:.4f} ± {std_time/3600:.4f}")


calculate_time_statistics("csv_path", "dermamnist", "ResNet18")