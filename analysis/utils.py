
import os
import torch
import time
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score, roc_curve, auc



def write_stats(config, epoch, sub_gen, phase, population, best_individual, history, epoch_duration=0.0):
    def clean_val(val):
        if isinstance(val, (list, np.ndarray)):
            val = val[-1] if len(val) > 0 else 0.0
        
        try:
            return float(val)
        except:
            return 0.0

    if population and len(population) > 0:
        fitness_values = [clean_val(ind[1]) for ind in population if ind[1] is not None]
        avg_fitness = np.mean(fitness_values) if fitness_values else 0.0
        std_fitness = np.std(fitness_values) if fitness_values else 0.0
        total_time = sum([clean_val(ind[2]) for ind in population if ind[2] is not None])
        if total_time == 0.0:
            print(f"Epoch duration: {epoch_duration}")
            total_time = epoch_duration
        
        population_to_save = [[ind[0], clean_val(ind[1])] for ind in population]
    else:
        print(f"Epoch duration: {epoch_duration}")
        avg_fitness, std_fitness, total_time, population_to_save = 0.0, 0.0, 0.0, []

    if best_individual and isinstance(best_individual, list) and len(best_individual) >= 2:
        best_fitness_val = clean_val(best_individual[1])
        best_genotype = best_individual[0]
    elif best_individual and isinstance(best_individual, list) and len(best_individual) == 1:
        best_genotype = best_individual[0]
        best_fitness_val = clean_val(history.get('val_wf1', 0.0)) 
    else:
        best_genotype = "None"
        best_fitness_val = 0.0

    v_acc  = clean_val(history.get('val_acc', 0.0))
    v_auc  = clean_val(history.get('val_auc', 0.0))
    v_bal  = clean_val(history.get('val_bal_acc', 0.0))
    v_mcc  = clean_val(history.get('val_matthews', 0.0))
    v_spec = clean_val(history.get('val_specificity', 0.0))
    
    t_loss = clean_val(history.get('sl_hist_loss', 0.0))
    t_acc  = clean_val(history.get('sl_hist_acc', 0.0))

    v_cm = history.get('val_confusion_matrix', None)
    if v_cm is not None:
        v_cm_str = np.array2string(np.array(v_cm), separator=',').replace('\n', '')
    else:
        v_cm_str = "None"

    if not os.path.exists(config['output_csv_folder']):
        os.makedirs(config['output_csv_folder'])
        
    file_path = os.path.join(config['output_csv_folder'], f'{config["experiment_name"]}_{config["seed"]}.csv')
    file_path_backup = os.path.join(config['output_csv_folder'], f'{config["experiment_name"]}_{config["seed"]}_backup.csv')

    header = 'epoch;sub_gen;phase;train_loss;train_acc;avg_fitness;std_fitness;best_fitness;val_acc;val_auc;val_bal_acc;val_matthews;val_specificity;val_confusion_matrix;total_time;best_individual;population\n'
    
    data_line = f'{epoch};{sub_gen};{phase};{t_loss:.6f};{t_acc:.6f};{avg_fitness:.6f};{std_fitness:.6f};{best_fitness_val:.6f};{v_acc:.6f};{v_auc:.6f};{v_bal:.6f};{v_mcc:.6f};{v_spec:.6f};{v_cm_str};{total_time:.2f};{best_genotype};{population_to_save}\n'

    try:
        write_header = not os.path.exists(file_path) or os.stat(file_path).st_size == 0
        with open(file_path, 'a') as stats_file:
            if write_header:
                stats_file.write(header)
            stats_file.write(data_line)
            
    except Exception as e:
        print(f"Error writing stats to file {file_path}: {e}")
        try:
            write_header_bak = not os.path.exists(file_path_backup) or os.stat(file_path_backup).st_size == 0
            with open(file_path_backup, 'a') as stats_file:
                if write_header_bak:
                    stats_file.write(header)
                stats_file.write(data_line)
                print(f"Successfully saved to backup: {file_path_backup}")
                
        except Exception as e:
            print(f"Error writing stats to backup file {file_path_backup}: {e}")

def write_epoch_stats(config, epoch, population, best_individual):
    fitness_values = [ind[1] for ind in population if ind[1] is not None]
    avg_fitness = np.mean(fitness_values) if fitness_values else 0.0
    std_fitness = np.std(fitness_values) if fitness_values else 0.0
    
    best_fitness_val = best_individual[1]
    best_genotype = best_individual[0] 

    history = best_individual[3]
    v_acc  = history.get('val_acc', 0.0)
    v_auc  = history.get('val_auc', 0.0)
    v_bal  = history.get('val_bal_acc', 0.0)
    v_mcc  = history.get('val_matthews', 0.0)
    v_spec = history.get('val_specificity', 0.0)
    v_cm   = history.get('val_confusion_matrix', None)
    
    t_loss = history.get('sl_hist_loss', [0.0])[-1]
    t_acc  = history.get('sl_hist_acc', [0.0])[-1]
    
    total_time = sum([individual[2] for individual in population if individual[2] is not None])
    population_to_save = [[ind[0], ind[1]] for ind in population]
    
    if v_cm is not None:
        v_cm_str = np.array2string(v_cm, separator=',').replace('\n', '')
    else:
        v_cm_str = "None"

    if not os.path.exists(config['output_csv_folder']):
        os.makedirs(config['output_csv_folder'])

    file_path = os.path.join(config['output_csv_folder'], f'{config["experiment_name"]}_{config["seed"]}.csv')
    file_path_backup = os.path.join(config['output_csv_folder'], f'{config["experiment_name"]}_{config["seed"]}_backup.csv')

    header = 'epoch;train_loss;train_acc;avg_fitness;std_fitness;best_fitness;val_acc;val_auc;val_bal_acc;val_matthews;val_specificity;val_confusion_matrix;total_time;best_individual;population\n'
    
    data_line = f'{epoch};{t_loss:.6f};{t_acc:.6f};{avg_fitness:.6f};{std_fitness:.6f};{best_fitness_val:.6f};{v_acc:.6f};{v_auc:.6f};{v_bal:.6f};{v_mcc:.6f};{v_spec:.6f};{v_cm_str};{total_time:.2f};{best_genotype};{population_to_save}\n'

    try:
        write_header = not os.path.exists(file_path) or os.stat(file_path).st_size == 0
        with open(file_path, 'a') as stats_file:
            if write_header:
                stats_file.write(header)
            stats_file.write(data_line)
            
    except Exception as e:
        print(f"Error writing stats to file {file_path}: {e}")
        try:
            write_header_bak = not os.path.exists(file_path_backup) or os.stat(file_path_backup).st_size == 0
            with open(file_path_backup, 'a') as stats_file:
                if write_header_bak:
                    stats_file.write(header)
                stats_file.write(data_line)
                print(f"Successfully saved to backup: {file_path_backup}")
                
        except Exception as e:
            print(f"Error writing stats to backup file {file_path_backup}: {e}")
    
def write_gen_stats(config, gen, population, best_individual):
    avg_fitness = np.mean([individual[1] for individual in population])
    std_fitness = np.std([individual[1] for individual in population])
    best_fitness_val = best_individual[1]
    best_individual_genotype = best_individual[0] 

    history = best_individual[3]
    v_acc  = history.get('val_acc', 0.0)
    v_auc  = history.get('val_auc', 0.0)
    v_bal  = history.get('val_bal_acc', 0.0)
    v_mcc  = history.get('val_matthews', 0.0)
    v_spec = history.get('val_specificity', 0.0)
    v_cm  = history.get('val_confusion_matrix', None)
    
    total_time = sum([individual[2] for individual in population if individual[2] is not None])

    if not os.path.exists(config['output_csv_folder']):
        os.makedirs(config['output_csv_folder'])

    file_path = os.path.join(config['output_csv_folder'], f'{config["experiment_name"]}_{config["seed"]}.csv')
    file_path_backup = os.path.join(config['output_csv_folder'], f'{config["experiment_name"]}_{config["seed"]}_backup.csv')

   # Save the population with only genotype and fitness to avoid issues with exceeding CSV limits
    population_to_save = [[ind[0], ind[1]] for ind in population]
    
    if v_cm is not None:
        v_cm_str = np.array2string(v_cm, separator=',').replace('\n', '')
    else:
        v_cm_str = "None"

    header = 'generation;avg_fitness_val;std_fitness_val;best_fitness;val_acc;val_auc;val_bal_acc;val_matthews;val_specificity;val_confusion_matrix;total_time;best_individual;population\n'
    
    try:
        write_header = not os.path.exists(file_path) or os.stat(file_path).st_size == 0
        with open(file_path, 'a') as stats_file:
            if write_header:
                stats_file.write(header)
            stats_file.write(f'{gen};{avg_fitness};{std_fitness};{best_fitness_val};{v_acc};{v_auc};{v_bal};{v_mcc};{v_spec};{v_cm_str};{total_time};{best_individual_genotype};{population_to_save}\n')
    except Exception as e:
        print(f"Error writing stats to file {file_path}: {e}")
        try:
            write_header_bak = not os.path.exists(file_path_backup) or os.stat(file_path_backup).st_size == 0
            with open(file_path_backup, 'a') as stats_file:
                if write_header_bak:
                    stats_file.write(header)
                stats_file.write(f'{gen};{avg_fitness};{std_fitness};{best_fitness_val};{v_acc};{v_auc};{v_bal};{v_mcc};{v_spec};{v_cm_str};{total_time};{best_individual_genotype};{population_to_save}\n')
        except Exception as e:
            print(f"Error writing stats to backup file {file_path_backup}: {e}")
            print(f'{gen};{avg_fitness};{std_fitness};{best_fitness_val};{v_acc};{v_auc};{v_bal};{v_mcc};{v_spec};{v_cm_str};{total_time};{best_individual_genotype};{population_to_save}')
    

# Save the configuration into a config.json file
def save_config_file (config_file_path,optimizer, learning_rate, batch_size, epochs):
    config = {
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs
    }
    with open(config_file_path, 'w') as config_file:
        json.dump(config, config_file,indent=1)
    print(f"Config saved to {config_file_path}")

# Save the training results into a history.json file
def save_train_metrics(history_file_path, train_losses, val_losses, train_accuracies, val_accuracies, best_epoch, elapsed_time):
    history = {
        "Train Losses": train_losses,
        "Validation Losses": val_losses,
        "Train Accuracies": train_accuracies,
        "Validation Accuracies": val_accuracies,
        "Best model saved after epoch": best_epoch,
        "Time": elapsed_time,
    }
    with open(history_file_path, 'w') as history_file:
        json.dump(history, history_file,indent=1)
    print(f"History saved to {history_file_path}")

# Save the continued training results into a history.json file
def save_continued_train_metrics(history_file_path, train_losses, train_accuracies, elapsed_time):
    history = {
        "Train Losses": train_losses,
        "Train Accuracies": train_accuracies,
        "Time": elapsed_time,
    }
    with open(history_file_path, 'w') as history_file:
        json.dump(history, history_file,indent=1)
    print(f"History saved to {history_file_path}")

# Save the model test results into a metrics.json file
def save_test_metrics(metrics_file_path, balanced_accuracy, acc, precision, recall, f1, roc_auc, matthews, specificity,tnr):
    metrics = {
        "Balanced Accuracy": balanced_accuracy,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "roc auc": roc_auc,
        "Matthews corrcoef": matthews,
        "Specificity": specificity,
        "NPV": tnr
    }
    with open(metrics_file_path, 'w') as metrics_file:
        json.dump(metrics, metrics_file,indent=1)
    print(f"Metrics saved to {metrics_file_path}")

# Save the true class and predicted class into a pred_probs.csv file
def save_predictions(predictions_file_path, y_true, y_pred, prob_scores, processo=None):
   
    prob_scores = np.array(prob_scores)

    data = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba_0": prob_scores[:, 0],
        "y_proba_1": prob_scores[:, 1],
    }

    if processo is not None:
        data = {"Processo": processo, **data}

    df = pd.DataFrame(data)
    df.to_csv(predictions_file_path, index=False)
    print(f"Predictions saved to {predictions_file_path}")


#Plot train and validation loss and accuracy vs epochs
def plot_TrainVal_LossAcc(plots_file_path,train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(15, 7))

    #loss vs epochs
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()

    #accuracy vs epochs
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()
    plt.savefig(plots_file_path)
    print(f"Plots saved to {plots_file_path}")

#Plot train loss and accuracy vs epochs
def plot_Train_LossAcc(plots_file_path, train_losses, train_accuracies):
    plt.figure(figsize=(12, 5))

    # Loss vs epochs
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.legend()

    # Accuracy vs epochs
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plots_file_path)
    print(f"Plots saved to {plots_file_path}")

# Plot confusion matrix
def plot_confusionMatrix(cm_file_path, conf_matrix, class_names, title_fig):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title_fig)
    plt.tight_layout()
    plt.savefig(cm_file_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix plot saved to {cm_file_path}")

# Plot Roc curve
def plot_RocCurve(roc_file_path, targets, prob_scores):
    fpr, tpr, thresholds = roc_curve(targets, np.array(prob_scores)[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Model Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_file_path)
    plt.close()
    print(f"ROC curve saved to {roc_file_path}")
    
def plot_RocCurve_both(roc_file_path, targets, prob_scores, class_names):
    plt.figure(figsize=(8, 6))
    n_classes = len(class_names)
    
    if n_classes == 2:
        scores = prob_scores[:, 1] if prob_scores.ndim > 1 else prob_scores
        fpr, tpr, _ = roc_curve(targets, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    
    else:
        y_test_bin = label_binarize(targets, classes=range(n_classes))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], prob_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right", fontsize='small', ncol=2 if n_classes > 5 else 1)
    plt.tight_layout()
    plt.savefig(roc_file_path, dpi=300)
    plt.close()
    print(f"ROC curve saved: {roc_file_path}")
