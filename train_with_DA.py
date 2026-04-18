def train_and_evaluate_individual(individual, config, dataset_vars=None):
    
    if dataset_vars is None:
        dataset_vars = config['load_dataset_func'](config)

    loaders = config['data_loader_func'](config=config,train_data=dataset_vars[0], val_data=dataset_vars[1], test_data=dataset_vars[2],tab_preproc=dataset_vars[3], transform=individual)

    fitness, hist = config['model_evaluate_func'](*loaders, config)

    print("Fitness (WF1):", fitness)

    return fitness, hist

def train_and_evaluate_individual_incremental(individual, config, dataset_vars=None):
    
    if dataset_vars is None:
        dataset_vars = config['load_dataset_func'](config)

    loaders = config['data_loader_func'](config=config,train_data=dataset_vars[0], sup_data= None, val_data=dataset_vars[1], test_data=dataset_vars[2],tab_preproc=dataset_vars[3], transform=individual)

    fitness, hist = config['model_evaluate_func'](*loaders, config)

    print("Fitness (WF1):", fitness)

    return fitness, hist

def train_and_evaluate_EML(individual, config, dataset_vars=None, current_model=None):
    
    if dataset_vars is None:
        dataset_vars = config['load_dataset_func'](config)

    loaders = config['data_loader_func'](config=config,train_data=dataset_vars[0], val_data=dataset_vars[1], test_data=dataset_vars[2],tab_preproc=dataset_vars[3], transform=individual)

    if current_model is not None:
        fitness, hist = config['model_evaluate_func'](*loaders, config, model=current_model)
    else:
        fitness, hist = config['model_evaluate_func'](*loaders, config)

    print("Fitness (ACC):", fitness)

    return fitness, hist
