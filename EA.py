import copy
import os
import sys
import time
import numpy as np
import random
import gc
from pympler.tracker import SummaryTracker
import os
import numpy as np
import analysis.utils as utils


def create_individual(config):
    genotype = [config['create_chromosome']() for _ in range(random.randint(1, config['max_chromosomes']))]
    
    return [
        genotype, # index 0: Genótipo
        None,     # index 1: Fitness (Val WF1)
        None,     # index 3: Tempo de treino
        None,     # index 4: Histórico 
    ]

def ea(config):

    if config['check_memory_leaks']:
        tracker = SummaryTracker()
        tracker.print_diff()

    if config['load_state']:
        config['load_state'](config)

    if config['start_parent'] is not None:
        best_gen_individual = [config['start_parent'], None, None, None]
        print(f"Starting from parent: {best_gen_individual[0]}")
        config['start_parent'] = None
    else:
        best_gen_individual = create_individual(config)

    for gen, evolution_mod in config['evolution_mods'].items():
        if gen >= config['start_gen']:
            break

        evolution_mod(config, past_gen=True)

    # main evolution loop
    for gen in range(config['start_gen'], config['stop_gen']+1):
        if config['max_generations_per_run'] is not None:
            config['current_run_generations'] += 1
            if config['current_run_generations'] > config['max_generations_per_run']:
                print("Max generations reached")
                sys.exit()

        print(f"\n--- Generation {gen} ---")
        #config['generation'] = gen

        if config['every_gen_state_reset']:
            config['every_gen_state_reset'](config)

        if config['evolution_mods'].get(gen) is not None:
            print(f"Running evolution mod for generation {gen}")
            config['evolution_mods'][gen](config)

        # new generation population
        if config['start_population'] is None:
            population = [copy.deepcopy(best_gen_individual)]
            for _ in range(config['population_size']-1):
                mutated_genotype = config['mutation'](best_gen_individual[0], config)

                offspring = [
                    mutated_genotype,    # mutated chromosomes
                    None,  # reset fitness 
                    None,  # reset training time
                    None,  # reset training history
                ]
                
                population.append(offspring)

            print("Parent:", population[0][:-1])
        else:
            population = config['start_population']
            config['start_population'] = None

        for individual in population:
            print(individual[0])

        # get fitness of each individual
        for i in range(config['population_size']):
            individual = population[i]
            print(f"\nTraining individual {i+1} for {config['epochs']} epochs: {individual[0]}")
            if individual[1] is None or config['recalculate_best']:

                start_time = time.perf_counter()
                fitness, history = config['individual_evaluation_func'](individual[0], config, dataset_vars=config.get('dataset_vars'))
                end_time = time.perf_counter()
                
                individual[1] = fitness
                individual[2] = end_time - start_time
                individual[3] = history
                
                # individual[1], individual[2], individual[4] = [i, i, [i]]
                print("Epochs: ", config['epochs'])

                gc.collect()

                print(f"Training time: {end_time - start_time:.2f} seconds")


        # best individual
        best_gen_individual = copy.deepcopy(max(population, key=lambda x: x[1]))
            
        utils.write_gen_stats(config, gen, population, best_gen_individual)

        if config['save_state']:
            config['save_state'](config)

        # print("Best individual:", config['best_individual'])

        # check if new top n was found
        if len(config['best_individuals']) < config['best_n'] or best_gen_individual[1] > config['best_individuals'][-1][1]:
            if len(config['best_individuals']) >= config['best_n']:
                del config['best_individuals'][-1]
            config['best_individuals'].append(best_gen_individual)
            config['best_individuals'].sort(key=lambda x: x[1], reverse=True)
            print(f"New Top {config['best_n']} individual:", best_gen_individual[:-1])

            # list top n individuals
            print(f"Updated Top {config['best_n']} individuals:")
            for i, individual in enumerate(config['best_individuals']):
                print(f"\tTop {i+1}/{config['best_n']} individual: {individual[:-1]}")



        for i in range(config['population_size']-1, -1, -1):
            if population[i][0] != best_gen_individual[0]: 
                for j in range(len(population[i])-1, -1, -1):
                    del population[i][j]
                del population[i]

        # print("Population size", len(population))

        del population

        if config['check_memory_leaks']:
            tracker.print_diff()


    return best_gen_individual


def ea_step(config, epoch, population, current_model):
    print(f"\n--- [EML Online] Evolution Step for Epoch {epoch} ---")
    
    gens_per_epoch = config.get('gens_per_epoch', 1)
    
    for gen in range(gens_per_epoch):
        if gens_per_epoch > 1:
            print(f"\n>> Sub-Generation {gen+1}/{gens_per_epoch} (Epoch {epoch})")
        
        for i in range(len(population)):
            ind = population[i]
            
            if ind[1] is None or config.get('recalculate_best', False):
                
                print(f"Evaluating strategy {i+1}/{len(population)}: {ind[0]}")
                
                original_weights = copy.deepcopy(current_model.model.state_dict())
                real_optimizer_state = getattr(current_model, 'current_optimizer', None)
                current_model.current_optimizer = None 
                
                start_time = time.perf_counter()
                
                fitness, history = config['individual_evaluation_func'](
                    ind[0], 
                    config, 
                    dataset_vars=config.get('dataset_vars'), 
                    current_model=current_model
                )
                
                ind[1] = fitness
                ind[2] = time.perf_counter() - start_time
                ind[3] = history
                
                current_model.model.load_state_dict(original_weights)
                current_model.current_optimizer = real_optimizer_state
                gc.collect()

        best_ind = copy.deepcopy(max(population, key=lambda x: x[1]))
        
        utils.write_stats(config, epoch, gen + 1, "EA", population, best_ind, best_ind[3])
    
        new_population = [best_ind] 
        for _ in range(config['population_size'] - 1):
            mutated_genotype = config['mutation'](best_ind[0], config)
            new_population.append([mutated_genotype, None, None, None])
        
        population = new_population

   
    #utils.write_epoch_stats(config, epoch, population, best_ind)

    if len(config['best_individuals']) < config['best_n'] or best_ind[1] > config['best_individuals'][-1][1]:
        
        if len(config['best_individuals']) >= config['best_n']:
            del config['best_individuals'][-1]
            
        config['best_individuals'].append(best_ind)
        
        config['best_individuals'].sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n New Top {config['best_n']} individual found in Epoch {epoch}! ***")
        print(f"New entry: Fitness {best_ind[1]:.4f} | Strategy: {best_ind[0]}")

        print(f"\n--- Updated Top {len(config['best_individuals'])} Global Individuals ---")
        for i, individual in enumerate(config['best_individuals']):
            print(f"\tRank {i+1}: {individual[:-1]}")
        print("---------------------------------------------------\n")

    return best_ind, population

def ea_step_optimizer(config, epoch, population, current_model):
    print(f"\n--- [EML Online] Evolution Step for Epoch {epoch} ---")
    
    gens_per_epoch = config.get('gens_per_epoch', 1)
    
    for gen in range(gens_per_epoch):
        if gens_per_epoch > 1:
            print(f"\n>> Sub-Generation {gen+1}/{gens_per_epoch} (Epoch {epoch})")
        
        for i in range(len(population)):
            ind = population[i]
            
            if ind[1] is None or config.get('recalculate_best', False):
                
                print(f"Evaluating strategy {i+1}/{len(population)}: {ind[0]}")
                
                original_weights = copy.deepcopy(current_model.model.state_dict())
                original_opt_state = None
                if hasattr(current_model, 'current_optimizer') and current_model.current_optimizer is not None:
                    original_opt_state = copy.deepcopy(current_model.current_optimizer.state_dict())
                
                
                start_time = time.perf_counter()
                
                fitness, history = config['individual_evaluation_func'](
                    ind[0], 
                    config, 
                    dataset_vars=config.get('dataset_vars'), 
                    current_model=current_model
                )
                
                ind[1] = fitness
                ind[2] = time.perf_counter() - start_time
                ind[3] = history
                
                current_model.model.load_state_dict(original_weights)
                if original_opt_state is not None:
                    current_model.current_optimizer.load_state_dict(original_opt_state)
                    
                gc.collect()

        best_ind = copy.deepcopy(max(population, key=lambda x: x[1]))
        
        utils.write_stats(config, epoch, gen + 1, "EA", population, best_ind, best_ind[3])
    
        new_population = [best_ind] 
        for _ in range(config['population_size'] - 1):
            mutated_genotype = config['mutation'](best_ind[0], config)
            new_population.append([mutated_genotype, None, None, None])
        
        population = new_population

   
    #utils.write_epoch_stats(config, epoch, population, best_ind)

    if len(config['best_individuals']) < config['best_n'] or best_ind[1] > config['best_individuals'][-1][1]:
        
        if len(config['best_individuals']) >= config['best_n']:
            del config['best_individuals'][-1]
            
        config['best_individuals'].append(best_ind)
        
        config['best_individuals'].sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n New Top {config['best_n']} individual found in Epoch {epoch}! ***")
        print(f"New entry: Fitness {best_ind[1]:.4f} | Strategy: {best_ind[0]}")

        print(f"\n--- Updated Top {len(config['best_individuals'])} Global Individuals ---")
        for i, individual in enumerate(config['best_individuals']):
            print(f"\tRank {i+1}: {individual[:-1]}")
        print("---------------------------------------------------\n")

    return best_ind, population



