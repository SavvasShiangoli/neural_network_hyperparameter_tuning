# config.py - Configuration file for hyperparameter tuning

# Random Seeds for Cross Validation
SEEDS = [42, 123, 456, 789, 999]

# Hyperparameters to test
HYPERPARAMETERS = {
    'learning_rates': [0.001, 0.01, 0.05, 0.1, 0.5],  # strength of backpropagation
    'batch_sizes': [32, 64, 96],  # Split of each epoch into mini batches
    'num_epochs': 42,  # Number of times the data is used in training
    'momentum': 0,  # Momentum uses previous gradients as additional terms in calculating step size
    'dropout': 0  # probability a neuron will be zeroed
}

# Model Architecture Parameters
MODEL_PARAMS = {
    'input_size': 784,  # MNIST 28x28 images
    'hidden_size': 50,
    'output_size': 10,  # 10 digit classes
    'dropout_rate': HYPERPARAMETERS['dropout']
}

# Training Parameters
TRAINING_PARAMS = {
    'momentum': HYPERPARAMETERS['momentum'],
    'device': 'auto',  # 'auto', 'cuda', or 'cpu'
}

# Learning Rate Scheduler Parameters
SCHEDULER_PARAMS = {
    'use_scheduler': True,
    'scheduler_type': 'cosine_annealing_warm_restarts',
    'T_0': 6,
    'T_mult': 2,
    'eta_min_ratio': 0.1  # Minimum LR as ratio of initial LR (10%)
}

# Early Stopping Conditions
EARLY_STOPPING = {
    'train_test_gap_threshold': 0.5,  # greater than this = too much overfitting
    'min_train_gradient': 0.05,  # less than this for gradient check
    'min_train_gradient_epochs': 5,  # Number of epochs to allow before detecting plateau
    'enable_early_stopping': True
}

# Data Parameters
DATA_PARAMS = {
    'dataset': 'MNIST',
    'data_root': 'data',
    'shuffle_train': True,
    'shuffle_test': True,
    'download': True
}

# Output Parameters
OUTPUT_PARAMS = {
    'results_dir': 'results',
    'checkpoint_frequency': 5,  # Save every N epochs
    'summary_filename': 'summary_bs32-96_lr0.001-0.5_hyper_cosineanneal.csv',
    'epoch_filename': 'epochdata_bs32-96_lr0.001-0.5_hyper_cosineanneal.csv'
}

# Advanced Learning Rate Control (your custom implementation)
ADAPTIVE_LR_PARAMS = {
    'enable_adaptive_lr': False,  # Set to True to enable your custom LR adaptation
    'gradient_factor_enabled': True,
    'proportional_factor_enabled': True
}

# Validation Parameters
VALIDATION_PARAMS = {
    'moving_average_window': 5,  # Number of epochs for moving average calculation
    'validation_start_epoch': 6  # When to start validation-based early stopping
}