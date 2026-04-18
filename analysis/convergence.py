import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import re
import seaborn as sns
import ast

plt.style.use('seaborn-v0_8-paper') 
sns.set_context("paper", font_scale=1.5)

def load_and_process(file_path):
    df = pd.read_csv(file_path, sep=';')
    
    def get_pop_fitness(pop_str):
        try:
         
            pop_list = ast.literal_eval(pop_str)
            return [ind[1] for ind in pop_list if ind[1] is not None]
        except:
            return []

    df['pop_fits'] = df['population'].apply(get_pop_fitness)
    
    df['pop_max'] = df['pop_fits'].apply(lambda x: max(x) if x else np.nan)
    df['pop_min'] = df['pop_fits'].apply(lambda x: min(x) if x else np.nan)
    
    return df

def plot_performance_evolution(df, save_name='performance_evolution_restored.png'):
     
    epoch_df = df.groupby('epoch').agg({
        'val_acc': 'last',
        'best_fitness': 'max',
        'phase': 'first'
    }).reset_index()

    epoch_df.loc[epoch_df['best_fitness'] == 0, 'best_fitness'] = np.nan

    plt.figure(figsize=(10, 6))

    plt.plot(epoch_df['epoch'], epoch_df['val_acc'], label='Master Model Accuracy', 
            color='#1f77b4', linewidth=2.5, marker='o', markersize=4)

    plt.plot(epoch_df['epoch'], epoch_df['best_fitness'], label='Best EA Fitness', 
            color='#ff7f0e', linewidth=2, linestyle='--', marker='x', markersize=5)

    warmup_end = df[df['phase'] == 'Warmup']['epoch'].max()
    if pd.notna(warmup_end):
        plt.axvline(x=warmup_end, color='red', linestyle=':', linewidth=2, label='End of Warmup')

    # Ajustes de estilo
    plt.title('Evolutionary Policy Discovery vs. Master Model Convergence', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy / Fitness', fontsize=12)
    plt.legend(frameon=True, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig('performance_evolution_clean.png', dpi=300)
    plt.show()

def plot_fitness_distribution(df, save_name='fitness_distribution_restored.png'):
    ea_phases = df[df['phase'] == 'EA']
    epochs = np.linspace(ea_phases['epoch'].min(), ea_phases['epoch'].max(), 6, dtype=int)
    
    data_to_plot = []
    labels = []
    
    for e in epochs:
        fits = ea_phases[ea_phases['epoch'] == e].iloc[-1]['pop_fits']
        if fits:
            data_to_plot.append(fits)
            labels.append(f'Epoch {e}')

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data_to_plot, palette="Set2")
    plt.xticks(range(len(labels)), labels)
    plt.title('Population Fitness Distribution (Online Evolution Phase)', fontsize=14)
    plt.ylabel('Validation Accuracy (Fitness)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    
def loss_convergence_analysis(df, save_name='loss_convergence.png'):
    epoch_df = df.groupby('epoch').agg({
        'train_loss': 'last',
        'phase': 'first'
    }).reset_index()

    plt.figure(figsize=(10, 5))

    # Plot da Train Loss
    plt.plot(epoch_df['epoch'], epoch_df['train_loss'], color='red', linewidth=2, label='Training Loss')

    # Marcar o fim do Warmup
    warmup_end = df[df['phase'] == 'Warmup']['epoch'].max()
    if pd.notna(warmup_end):
        plt.axvline(x=warmup_end, color='black', linestyle='--', alpha=0.5)

    plt.title('Training Loss Convergence during Interleaved Policy Optimisation', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig('training_loss_convergence.png', dpi=300)
    plt.show()
    


def plot_macro_micro_side_by_side(path_pattern, save_name='combined_dynamics.png'):
    files = glob.glob(path_pattern)
    if not files:
        print(f"Erro: Nenhum ficheiro encontrado para: {path_pattern}")
        return

    macro_acc = []
    micro_fit = []
    
    for f in files:
        df = pd.read_csv(f, sep=';')
        
        epoch_df = df.groupby('epoch')['val_acc'].last().reset_index()
        macro_acc.append(epoch_df['val_acc'].values)
        
        ea_df = df[df['best_fitness'] > 0].copy()
        ea_df = ea_df.dropna(subset=['best_fitness'])
        micro_fit.append(ea_df['best_fitness'].values)

    min_len_macro = min(len(m) for m in macro_acc)
    macro_acc = np.array([m[:min_len_macro] for m in macro_acc])
    
    min_len_micro = min(len(m) for m in micro_fit)
    micro_fit = np.array([m[:min_len_micro] for m in micro_fit])
    
    epochs = np.arange(1, min_len_macro + 1)
    macro_mean = np.mean(macro_acc, axis=0)
    macro_std = np.std(macro_acc, axis=0)
    macro_max = np.max(macro_acc, axis=0)
    
    steps = np.arange(1, min_len_micro + 1)
    micro_mean = np.mean(micro_fit, axis=0)
    micro_max = np.max(micro_fit, axis=0)

 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    ax1.fill_between(epochs, macro_mean - macro_std, macro_mean + macro_std, 
                     color='#1f77b4', alpha=0.2, label='Std Dev (10 Seeds)')
    ax1.plot(epochs, macro_mean, color='#1f77b4', linewidth=2.5, label='Mean Val Acc (10 Seeds)')
    ax1.plot(epochs, macro_max, color='#ff7f0e', linewidth=2, linestyle='--', label='Max Val Acc (Best Seed)')
    
    df_ref = pd.read_csv(files[0], sep=';')
    if 'phase' in df_ref.columns:
        warmup_end = df_ref[df_ref['phase'] == 'Warmup']['epoch'].max()
        if pd.notna(warmup_end) and warmup_end < min_len_macro:
            ax1.axvline(x=warmup_end, color='red', linestyle=':', linewidth=2, label='End of Warmup')
            
    ax1.set_title('(a) Master Model Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Accuracy', fontsize=12)
    ax1.set_xlim(1, min_len_macro)
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.5)

 
    ax2.plot(steps, micro_mean, color='#1f77b4', linewidth=2, label='Mean Fitness (10 Seeds)')
    ax2.plot(steps, micro_max, color='#ff7f0e', linewidth=2, linestyle='--', label='Max Fitness (Best Seed)')
    
    for x_line in range(10, min_len_micro + 1, 10):
        ax2.axvline(x=x_line, color='gray', linestyle='--', alpha=0.3)

    ax2.set_title('(b) Evolutionary Search Progression', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Cumulative Generations ', fontsize=12)
    ax2.set_ylabel('Fitness Score', fontsize=12)
    ax2.set_xlim(1, min_len_micro)
    ax2.legend(loc='lower right')
    
    ax2.grid(axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()



file_path = 'csv_path'
data = load_and_process(file_path)

#plot_performance_evolution(data)
plot_fitness_distribution(data)
#loss_convergence_analysis(data)

#plot_macro_micro_side_by_side('csv_path)

