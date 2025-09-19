# Neural Network Hyperparameter Tuning on MNIST
A comprehensive hyperparameter tuning framework for neural networks, featuring early stopping, learning rate scheduling, and statistical validation across multiple random seeds.

## Project Overview
This project demonstrates systematic hyperparameter optimization for neural networks using MNIST. It goes beyond simple grid search by implementing:

- Statistical validation with 5 random seeds
- Early stopping based on train-test gap and improvement plateaus
- Cosine annealing learning rate scheduling
- Comprehensive logging of all experiments

This project begins with a simple 3 layer neural network and aims to establish a foundation first with 
learning rate, batch size and epoch hyperparameter tuning. 

Even at this early stage, statistical validation with 5 random seeds is used for analysis and the use of a custom stopping method is put in place based on the test train gap and plateaus.

Experimentally at this stage custom learning rate scehduling with PD feedback and cosine annealing with warm starts were looked at with the aim to unlock more performace even at this early stage. 

From this baseline the architecture of the neural network can then be investigated with additional layers
tunnels, different activation functions and optimizers.

Following this the neural netowrk will then go into optimsiation refinement introducing momentum and then
looking at the connection with batch size.

After this regularisation through dropout will be introduced which can unlock more complex archtectures
which naturally overfit.

## Key Features
- Multi-seed validation: Tests each configuration with 5 different random seeds for statistical robustness
- Smart early stopping: Monitors train-test gap and improvement rate
- Learning rate scheduling: Uses CosineAnnealingWarmRestarts for adaptive learning
- Comprehensive tracking: Saves both epoch-level and summary statistics to results\ directory
- Work in progress: Currently refactoring into modular components for better usability

## Experiments Covered
Experiments Completed:
- Learning rates: [0.001, 0.01, 0.05, 0.1, 0.05] vs Batch Size: [32, 64, 96]
- Early stopping thresholds for plateau detection [0.1% and 0.05%]
- Effect of early stopping on the accuracy performance and test-train gap
- Learning Rate attenuation methods both custom and cosine annealing and the effect on accuracy and test-train gap
- Architecture Optimzation = hiden layers(1-4), layer width(finnel v constant)

Experiments Coming Up:
- Architecture Optimisation = activations (LeakyReLU), optimizers (adam)
- Momentum vs batch rate tuning for refinement
- Dropout for unlocking complex architectures and models which tend to overfit

## Future Improvements
Code Structure:
- Extend refactoring to architecture search - optimizer, activation function
- Add visualisation of the results = bar charts, line graphs and heatmaps 

## Change Log

### Refactoring Progress (16th September 2025)
- Separated configuration into `config_test.py` for better modularity
- Dynamic experiment ID generation based on search parameters
- Flexible hyperparameter combinations using `itertools`
- Improved dataframe structure for any parameter combination

### What Changed
- Configuration and experiment setup now separate from training logic
- Can easily add/remove hyperparameters in config file
- Automatic handling of fixed vs search parameters
- More maintainable and extensible codebase

### Next Steps
- Extract model definition to separate file
- Create experiment comparison and visualization module for results

### Refactoring Progress (19th September 2025)
- Neural Network architecture definition has been moved to the config file
- Updated summary_df format to be more user friendly with exp_id only for columns
- Added momentum and architecture to the search combinations

### What Changed
- Mutiple architectures with different hidden layer design can now be set in the config file
- Momentum has been added to the combinations loop and can now be cycled over
- Summary_df has been reformatted to have exp_id for columns and averages, stds etc as rows for better userbility
- Network class within hyperparameter_tuning.py is updated to automatically create the netowrk based on the config file
- config_test.py has been renamed to config.py removing additional step used before for troubleshooting

### Next Steps
- Create experiment comparison and visualization module for results

## Usage 
```bash
python hyperparameter_tuning.py ```

