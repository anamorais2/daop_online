import os
import sys
import pandas as pd
import ast
import configs.config as config_base
import torch
import EA as EA


def completed_run(config):
    file_path = os.path.join(config['output_csv_folder'], f'{config["dataset"]}_{config["experiment_name"]}_{config["seed"]}.csv')

  
    if not os.path.exists(file_path):
        return False


    df = pd.read_csv(file_path, sep=';')
    max_epoch = df['epoch'].iloc[-1]
    if max_epoch >= config['epochs']:
        print(f"Seed {config['seed']} already completed {max_epoch} epochs.")
        del df
        return True
    

    # Get the last generation and parent
    config['start_epoch'] = max_epoch + 1
    start_parent_str = df['best_individual'].iloc[-1]

    try:
        config['start_parent'] = ast.literal_eval(start_parent_str)
        print(f"Recovered parent: {config['start_parent']}")
    except:
        print(f"Failed to parse parent {start_parent_str}")
        config['start_parent'] = None
        
    last_pop_str = df['population'].iloc[-1]
    try:
        recovered_pop = ast.literal_eval(last_pop_str)
        config['recovered_population'] = []
        for ind_data in recovered_pop:
            config['recovered_population'].append([ind_data[0], None, None, None])
            
        print(f"Recovered full population of {len(config['recovered_population'])} individuals (Fitness reset for fresh evaluation).")
    except Exception as e:
        print(f"Could not recover full population, will evolve from parent. Error: {e}")
        config['recovered_population'] = None

    # Get the best individuals
    df_top_n = df.nlargest(config['best_n'], 'best_fitness')
    for individual, fitness in zip(df_top_n['best_individual'], df_top_n['best_fitness']):
        try:
            config['best_individuals'].append([ast.literal_eval(individual), fitness, None, None, None])
            print(f"Recovered top 5 individual: {config['best_individuals'][-1]}")
        except:
            print(f"Failed to parse top 5 individual {individual}")

    config['best_individuals'].sort(key=lambda x: x[1], reverse=True)


    # Check if the state file exists
    state_file = f'state_{config["dataset"]}_{config["experiment_name"]}_{config["seed"]}.pickle'
    if os.path.exists(os.path.join(config['state_folder'], state_file)):
        config['state_file'] = state_file

    del df
    return False


def reset_config(config):
    config['start_epoch'] = 1
    config['start_parent'] = None
    config['best_individuals'] = []
    config['state_file'] = None
    config['epochs'] = config['epochs']
    #config['pretext_epochs'] = config['base_pretext_epochs']
    #config['downstream_epochs'] = config['base_downstream_epochs']
    #config['pretrained_pretext_model'] = config['base_pretrained_pretext_model']
    
    


skip_runs = [
    # seed
    # 0,
    # 1,
]

# skip_until_mutation = 1
skip_until_run = None
    
if __name__ == "__main__":
    config = config_base.config

    try:
        if len(sys.argv) > 1:
            config['seeds'] = [int(i) for i in sys.argv[1].split(",")]
    except:
        print("Failed to parse seeds from command line arguments")

    for seed in config['seeds']:
        config['seed'] = seed
        reset_config(config)


        if completed_run(config) or seed in skip_runs or (skip_until_run is not None and seed != skip_until_run):
            print(f"Skipping seed {seed}")
            continue
        
        print(f"Running seed {seed}")
        
        print("Loading dataset...")
        dataset_vars = config['load_dataset_func'](config)
        config['dataset_vars'] = dataset_vars
        master_model = config['model'](num_classes=config['num_classes'])
        
        if config.get('recovered_population') is not None:
            print("Resuming with exact population from CSV...")
            population = config['recovered_population']
            
        elif config.get('start_parent') is not None:
            print("Initializing population from recovered parent (new mutations)...")
            population = [[config['start_parent'], None, None, None]]
            for _ in range(config['population_size'] - 1):
                mutated = config['mutation'](config['start_parent'], config)
                population.append([mutated, None, None, None])
        else:
            population = [EA.create_individual(config) for _ in range(config['population_size'])]
        
        
        best_val_acc_global = 0.0
        best_model_path = os.path.join(config['output_csv_folder'], f"best_model_seed_{seed}.pth")
        
        for epoch in range(config['start_epochs'], config['stop_epochs'] + 1):
            
            print(f"\n==== Epoch {epoch}/{config['stop_epochs']} ====")
            
            #Warmup phase
            if epoch > config['warmup_epochs']:
                best_ind, population = EA.ea_step(config, epoch, population, master_model)
                current_DA = best_ind[0]
            else:
                print(f"Warmup phase - training master model without DAOP augmentations - Epoch {epoch}/{config['warmup_epochs']}")
                current_DA = None
                
            print(f"Training model with DA: {current_DA}")
                        
            # Forçamos 1 época de treino para atualização permanente dos pesos
            original_epochs = config['epochs']
            config['epochs'] = 1 
            
            val_acc, _ = config['individual_evaluation_func'](
                current_DA, config, dataset_vars=dataset_vars, current_model=master_model
            )
            
            config['epochs'] = original_epochs
            
            if val_acc > best_val_acc_global:
                best_val_acc_global = val_acc
                print(f" >>> NEW BEST MODEL FOUND! Epoch {epoch} - Val Acc: {val_acc:.4f}. Saving...")
                torch.save(master_model.model.state_dict(), best_model_path)
            
        print(f"\n==== Final Test Set Evaluation - Seed {seed} ====")
        
        if os.path.exists(best_model_path):
            print(f"Loading best recorded model from {best_model_path} for prospective evaluation...")
            master_model.model.load_state_dict(torch.load(best_model_path))
            
        config['evaluate_on_test'] = True
        config["epochs"] = config['final_eval_epochs']
        config['individual_evaluation_func'](
                current_DA, config, dataset_vars=dataset_vars, current_model=master_model
            )
            
            
        #best_ind = EA.ea(config)
            
        #------------------------ ONLY TEST EVALUATION OF THE BEST INDIVIDUAL AFTER FINDING THE BEST MODEL ------------------------
        # Dont forget to comment the line 105 and the lines 118 to 122, and uncomment the lines 110 to 115. 
            
        #import data_processing.data_processing_multimodal as data_processing
            
        #_, _, test_loader = data_processing.create_data_loaders_multimodal(
        #    config, *dataset_vars, transform=None)
            
        #from sl_evaluation_multimodal import evaluate_only_test_set
        #evaluate_only_test_set(test_loader, config)
            
        #print(f"Best individual for seed {seed}: {best_ind}")
        #print("Execution of the best individual on the test set:")
        #config['evaluate_on_test'] = True
        #config["epochs"] = config['final_eval_epochs']
        #da, hist = config['individual_evaluation_func'](best_ind[0], config)
            
        print(f"Completed seed {seed}")
        

    if config['delete_cache']:
        import shutil
        shutil.rmtree(config['cache_folder'])
