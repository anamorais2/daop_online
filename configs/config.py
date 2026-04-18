import json
import os
import torch
import DA.data_augmentation_albumentations as data_augmentation_albumentations
import state_manager_torch
import mutations
import chromosomes
import train_with_DA
import models.resnet as resnet_models 
import sl_evaluation_medmnist 
import data_processing.data_medmnist as data_medmnist

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

DATA_FLAG = 'organcmnist'  # Change this to switch datasets
MODEL_FLAG = 'ResNet18'  


config = {}

config['num_classes'] = 11

if MODEL_FLAG == 'ResNet18':
        config['model'] = resnet_models.TrainResNet18
elif MODEL_FLAG == 'ResNet50':
        config['model'] = resnet_models.TrainResNet50

config['online_training'] = True
config['population'] = True
config['optimizer'] = True

if config['online_training']:
        if config['population']:
                if config['optimizer']:
                        config['base_experiment_name'] = f"ONT_DAOP_{DATA_FLAG}_{MODEL_FLAG}_Optimizer_Population_Baseline"
                else:
                        config['base_experiment_name'] = f"ONT_DAOP_{DATA_FLAG}_{MODEL_FLAG}_Population"
        else:
                if config['optimizer']:
                        config['base_experiment_name'] = f"ONT_DAOP_{DATA_FLAG}_{MODEL_FLAG}_Optimizer"
                else:
                        config['base_experiment_name'] = f"ONT_DAOP_{DATA_FLAG}_{MODEL_FLAG}"
else:
        config['base_experiment_name'] = f"DAOP_{DATA_FLAG}_{MODEL_FLAG}"


# experiment configs
config['experiment_name'] = config['base_experiment_name']
config['output_csv_folder'] = "Results" + "_" + config['base_experiment_name'] 
config['seeds'] = range(5)
config['seed'] = config['seeds'][0]
config['state_folder'] = "VAL_states"
config['state_file'] = None
config['load_state'] = state_manager_torch.load_state
config['save_state'] = state_manager_torch.save_state
config['every_gen_state_reset'] = None

# dataset configs
config['dataset'] = DATA_FLAG
config['dim'] = (28, 28, 3)


#Config for breastmnist
config['load_pretrained'] = False
config['load_dataset_func'] = data_medmnist.load_medmnist_datasets
config['data_loader_func'] = data_medmnist.create_medmnist_loaders

config['cache_folder'] = f"cache_{DATA_FLAG}_torch"
config['delete_cache'] = False

config['num_cols'] = []
config['categorical_cols'] = []
config['image_cols'] = ['images'] 
config['y_col'] = "label"
config['num_numerical'] = 0
config['num_categories'] = []

# data loading functions
config['dataset_vars'] = None 


# augmentation configs
config['individual_evaluation_func'] = train_with_DA.train_and_evaluate_EML
config['augment_dataset'] = True
config['min_da_prob'] = 0.1
config['max_da_prob'] = 0.9
config['da_funcs'] = data_augmentation_albumentations.da_funcs_probs(config['min_da_prob'], config['max_da_prob'], config['dim'][:2])
config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {config['device']}")
# config['shuffle_dataset'] = False     # shuffle with sampler
config['evolution_type'] = "simultaneous"   # "same"/"simultaneous"

# model training configs
config['lr'] = 3.7e-4
config['weight_decay'] = 1e-3
config['framework'] = 'torch'
config['evaluate_on_test'] = False
config['epochs'] = 13
config['start_epochs'] = 1
config['stop_epochs'] = 100
config['final_eval_epochs'] = 0

config['batch_size'] = 128
config['num_workers'] = 4
config['model_evaluate_func'] = sl_evaluation_medmnist.evaluate_sl
config['check_memory_leaks'] = False
config['shuffle_dataset'] = True

# evolutionary algorithm configs
config['n_pr'] = 4
config['start_parent'] = None
config['start_population'] = None
config['recovered_population'] = None
config['best_n'] = 5
config['best_individuals'] = None
config['extended_isolated_run'] = False
config['current_run_generations'] = 0
config['max_generations_per_run'] = None
config['start_gen'] = 1
config['stop_gen'] = 10
config['population_size'] = 10
config['max_chromosomes'] = 3
config['recalculate_best'] = True
config['create_da_func'] = chromosomes.random_da_func(len(config['da_funcs']))
config['create_pr'] = chromosomes.random_pr
config['create_chromosome'] = chromosomes.create_chromosome_2_levels(config['create_da_func'], config['create_pr'], config['n_pr'])
config['da_func_mutation'] = chromosomes.random_da_func(len(config['da_funcs']))
config['pr_mutation'] = chromosomes.random_pr_gaussian(0.1)
config['mutation'] = mutations.mutate_remove_change_add_seq(0.66, 0.33, 0.66)
config['evolution_mods'] = {}


#EML
config['warmup_epochs'] = 100
config['gens_per_epoch'] = 10
config['current_DA'] = []  # No augmentations during warmup