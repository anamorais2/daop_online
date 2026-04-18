# DAOP Online Project Documentation

DAOP Online is a framework designed to optimise data augmentation policies while a model is training. This repository contains the code to run evolutionary searches on medical imaging datasets from the MedMNIST benchmark. The system uses a ResNet-18 architecture and evolves policies specifically for each dataset to improve classification performance.

## Installation
To set up the environment, you should first clone the repository and install the necessary dependencies. It is recommended to use a virtual environment to avoid conflicts.

1. Clone the repository to your local machine.
2. Install the requirements using the provided file: pip install -r requirements.txt.
3. Ensure you have a working installation of PyTorch, torchvision, and Albumentations.

## Configuration and Setup
Instead of using complex command-line arguments, the experiments are configured by editing the variables in the configuration file or at the top of the main script. This makes it easier to manage fixed settings like the dataset or model architecture.

To change the experiment settings, simply adjust these flags in the code:
- DATA_FLAG: Choose between 'breastmnist', 'dermamnist', 'pneumoniamnist', or 'organcmnist'.
- RESNET_FLAG: Choose between 'resnet18' or 'resnet50'.
- NUM_EPOCHS: Total training epochs (e.g. 100).
- WARMUP_EPOCHS: Number of epochs before the evolution starts (e.g. 45).

## Running the Evolutionary Search
Once you have configured your settings in the file, you can start the process. If you need to run multiple independent trials, you can simply pass the seed value:

python main_online_daop.py 1

The process involves:
1. Warmup Phase: The model trains for a fixed number of epochs with standard augmentations to stabilise weights.
2. Evolutionary Phase: The (1+9) strategy begins. In each epoch, nine new candidate policies are evaluated against the current best.
3. Snapshot and Restore: The system saves the master model weights before testing candidates and restores them afterwards. This prevents the search process from negatively affecting the main model training.

## Running Meta-Optimisation with Optuna
We also use Optuna to find the best structural hyperparameters for the framework itself. This helps determine the ideal warmup period and the number of inner epochs for the evolutionary search.

To run an Optuna study:
python optuna_study.py 

## Troubleshooting and CUDA
If you encounter errors during execution, please check the following common issues:

CUDA Out of Memory:
The evolutionary search evaluates multiple candidates, which can be demanding for the GPU. If you run out of memory, try reducing the batch_size in your configuration.

Verify GPU Availability:
You can check if your system recognises the GPU by running:
python -c "import torch; print(torch.cuda.is_available())"

Check GPU Usage:
Use the command 'nvidia-smi' in your terminal to monitor memory consumption and ensure other processes are not using the same hardware.

## Results and Logs
All experimental data is saved in the Results folder. The system generates CSV files with validation accuracy, the best policies found, and population fitness distribution.