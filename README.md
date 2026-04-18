# DAOP Online

## Project Description

DAOP Online is an innovative framework developed to optimise data augmentation policies in real time during the training of machine learning models. This project uses evolutionary algorithms to explore and identify the best combinations of data augmentation transformations, specifically tailored to medical datasets from the MedMNIST benchmark. The system supports ResNet architectures (18 and 50) and evolves custom policies for each dataset to improve classification performance.

The project is based on research into automatic optimisation of data processing pipelines, allowing models to adapt dynamically to the characteristics of the data during training.

## Key Features

- Online Optimisation: Adjusts data augmentation policies while the model is training, without the need for separate pre-training.
- Evolutionary Algorithms: Implements methods such as (1+9) to evolve populations of augmentation policies.
- Multiple Dataset Support: Compatible with MedMNIST datasets such as `breastmnist`, `dermamnist`, `pneumoniamnist`, `organcmnist`, and others.
- Flexible Models: Works with ResNet-18 and ResNet-50.
- Evaluation and Logging: Generates CSV logs with validation metrics, the best policies discovered, and population fitness statistics.
- Hyperparameter Optimisation: Uses Optuna to tune structural parameters like warmup duration and inner training epochs.
- State Management: Saves and restores model states to continue interrupted experiments.

## Project Structure

- `DA/`: Data augmentation modules using Albumentations.
- `analysis/`: Tools for analysing results.
- `configs/`: Configuration files (e.g. `config.py`).
- `data_processing/`: Data processing for MedMNIST.
- `models/`: ResNet model implementations.
- `chromosomes.py`: Representation of augmentation policies as chromosomes.
- `mutations.py`: Mutation operators for evolutionary search.
- `EA.py`: Main evolutionary algorithm.
- `main.py`, `main_best.py`, etc.: Main execution scripts.
- `requirements.txt`: Project dependencies.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd daop_online
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Check PyTorch and CUDA**:
   - Ensure PyTorch is installed with CUDA support if available.
   - Test with: `python -c "import torch; print(torch.cuda.is_available())"`

## Configuration

Configurations are managed in the `configs/config.py` file. Main variables to adjust:

- `DATA_FLAG`: Select the dataset (e.g. `'organcmnist'`, `'breastmnist'`, etc.).
- `MODEL_FLAG`: Select the model (`'ResNet18'` or `'ResNet50'`).
- Other settings: number of epochs, population size, optimiser options, etc.

For multiple experiments, edit the flags directly in the code or pass arguments through the command line if supported.

## Usage

### Basic Execution

To start an experiment with evolutionary optimisation:

```bash
python main.py [seed]
```

- Replace `[seed]` with an integer for reproducibility, e.g. `1`.

The workflow includes:
1. **Warmup Phase**: Initial training with standard augmentations to stabilise the model weights.
2. **Evolutionary Phase**: Evaluate candidate policies and evolve the population.
3. **Snapshot and Restore**: Save and restore model state to avoid interference from candidate evaluation.

### Hyperparameter Optimisation with Optuna

To tune structural parameters such as warmup length and inner epoch count:

```bash
python main_optuna.py
```

### Other Scripts

- `main_best.py`: Run with the best known configuration.
- `main_optimizer.py`: Focus on optimisation experiments.
- `tests_stats.py`: Statistical analysis of experimental results.
- `sl_evaluation_medmnist.py`: Supervised learning evaluation.

## Results and Logs

- Results are saved in the `Results_[experiment_name]/` folder.
- Generated CSV files typically include:
  - Validation accuracy by epoch.
  - Best augmentation policies found.
  - Population fitness distribution.
- Use the `analysis/` folder to visualise and review experiment outputs.

## Troubleshooting

- **CUDA Out of Memory**: Reduce the `batch_size` in the configuration or use a GPU with more memory.
- **GPU Verification**: Run `nvidia-smi` to monitor GPU usage.
- **Interrupted Runs**: The system can resume interrupted experiments using saved CSV logs and state management.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a branch for your feature: `git checkout -b feature/new-feature`.
3. Commit your changes: `git commit -am 'Adds new feature'`.
4. Push to the branch: `git push origin feature/new-feature`.
5. Open a Pull Request.


## References

- Based on research into AutoAugment and evolutionary optimisation for data augmentation.
- Dataset benchmark: [MedMNIST](https://medmnist.com/).
- Libraries: PyTorch, Albumentations, Optuna.

For questions or support, open an issue in the repository.