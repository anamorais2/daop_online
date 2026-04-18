import optuna
import EA as EA
import configs.config as config_base
import analysis.utils as utils 
from main import reset_config 
import os
import copy
import torch
import time

os.makedirs("optuna_results_population", exist_ok=True)

def objective(trial):
    config = copy.deepcopy(config_base.config)
    
 
    config['warmup_epochs'] = trial.suggest_int('warmup_epochs', 10, 50, step=5)
    config['gens_per_epoch'] = trial.suggest_int('gens_per_epoch', 1, 15)
    config['epochs'] = trial.suggest_int('ea_internal_epochs', 1, 30)
    config['population_size'] = trial.suggest_int("population_size", 4, 10)
    config['max_chromosomes'] = trial.suggest_int("max_chromosomes", 3, 8)
    
    config['experiment_name'] = f"OPT_POPULATION_Trial_{trial.number}_W{config['warmup_epochs']}_G{config['gens_per_epoch']}"
    config['seed'] = 0 
    
    reset_config(config)
    
    dataset_vars = config['load_dataset_func'](config)
    config['dataset_vars'] = dataset_vars
    master_model = config['model'](num_classes=config['num_classes'])
    
    population = [EA.create_individual(config) for _ in range(config['population_size'])]
    best_val_acc_in_trial = 0.0

    for epoch in range(config['start_epochs'], config['stop_epochs'] + 1):
        
        if epoch > config['warmup_epochs']:
            best_ind, population = EA.ea_step(config, epoch, population, master_model)
            current_DA = best_ind[0]
        else:
            current_DA = []

        original_epochs = config['epochs']
        config['epochs'] = 1 
        val_acc, history = config['individual_evaluation_func'](
            current_DA, config, dataset_vars=dataset_vars, current_model=master_model
        )
        config['epochs'] = original_epochs
        
        phase = "Warmup" if epoch <= config['warmup_epochs'] else "Training"
        utils.write_stats(config, epoch, 0, phase, population, current_DA, history)

        if val_acc > best_val_acc_in_trial:
            best_val_acc_in_trial = val_acc
            model_path = f"optuna_results_population/model_trial_{trial.number}.pth"
            torch.save(master_model.model.state_dict(), model_path)
            trial.set_user_attr("model_path", model_path)

        trial.report(val_acc, epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch}")
            raise optuna.exceptions.TrialPruned()

    return best_val_acc_in_trial

if __name__ == "__main__":

    storage_db = "sqlite:///hpo_medmnist_study_population.db"
    
    study = optuna.create_study(
        study_name="MedMNIST_EA_Optimization_Population",
        direction='maximize',
        storage=storage_db,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,  
            n_warmup_steps=10,   
            interval_steps=5    
        )
    )

    timeout_hours = 144
    print(f"Starting HPO for {timeout_hours} hours. Parallel workers allowed.")
    
    study.optimize(objective, timeout=timeout_hours * 3600)

    df_results = study.trials_dataframe()
    df_results.to_csv("optuna_hpo_final_summary_population.csv", index=False)
    
    print("\n--- OPTIMIZATION FINISHED ---")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Val Accuracy: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")