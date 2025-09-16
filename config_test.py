import itertools
import pandas as pd

# For cross validation and statistical analysis for the model
SEEDS = [42, 123, 456, 789, 999]

# Paramaters which effect the learning of the model
HYPERPARAMETERS = {
    'learning_rate': [1, 0.01, 0.05, 0.1, 0.5],
    'batch_size': [32, 64, 96],
    'num_epochs':42,
    'momentum':0,
    'dropout':0 
}

# Paramaters which dictate the early stopping of fitting
EARLY_STOPPING = {
    'train_test_gap_threshold': 0.5,
    'min_train_gradient': 0.05,
    'min_train_gradient_epochs': 5
}

# Based on the hyperparamaters creates a list of dictionaries of every combination that will be tested
# Combination starts with the fixed hyperparameters and ends with the search parameters
def get_search_combinations():
    """
    Generate all combinations of hyperparameters to search over.
    Returns list of dictionaries, each representing one experiment configuration.
    """
    # Separate search parameters (lists) from fixed parameters (single values)
    search_params = {}
    fixed_params = {}
    
    for param_name, param_value in HYPERPARAMETERS.items():
        if isinstance(param_value, list) and len(param_value) > 1:
            search_params[param_name] = param_value
        else:
            # Handle both single values and single-item lists
            fixed_params[param_name] = param_value[0] if isinstance(param_value, list) else param_value
    
    if not search_params:
        # If no search dimensions, return single combination with fixed params
        return [fixed_params.copy()]
   
    # Get all parameter names and their values for search
    param_names = list(search_params.keys())
    param_values = list(search_params.values())
   
    # Generate all combinations
    combinations = []
    for combo in itertools.product(*param_values):
        # Create config dict for this combination
        config = fixed_params.copy()  # Start with fixed params
       
        # Add the search parameters for this combination
        for param_name, param_value in zip(param_names, combo):
            config[param_name] = param_value
       
        combinations.append(config)
   
    return combinations


# Generates the summary and epoch dataframes
def create_summary_dataframes():
    combinations = get_search_combinations()
    
    # Setup empty lists for the dataframe column headings
    summary_columns = []
    epoch_columns = []
    exp_ids = []

    num_epochs = HYPERPARAMETERS['num_epochs']

    # Create the experiment ID which will be used for the heading of the columns for each combination
    for combination in combinations:
        exp_id_parts = []
        # Pulls out only the paramaters which are being searched over to include in the experiment identifier
        for param_name, param_value in HYPERPARAMETERS.items():
            if isinstance(param_value, list) and len(param_value) > 1:
                value = combination[param_name]
                exp_id_parts.append(f"{param_name}{value}")

        # join the parts into a single string
        exp_id = "_".join(exp_id_parts)
        exp_ids.append(exp_id)
    
        # Extends the columns using the experiment ID and the following extensions
        summary_columns.extend([
                f'{exp_id}_train_acc_mean', f'{exp_id}_train_acc_std',
                f'{exp_id}_train_loss_mean', f'{exp_id}_train_loss_std',
                f'{exp_id}_test_acc_mean', f'{exp_id}_test_acc_std',
                f'{exp_id}_test_loss_mean', f'{exp_id}_test_loss_std'
            ])
        
        for seed in SEEDS:
            epoch_columns.extend([
                f'{exp_id}_seed{seed}_train_acc', 
                f'{exp_id}_seed{seed}_train_loss',
                f'{exp_id}_seed{seed}_test_acc', 
                f'{exp_id}_seed{seed}_test_loss'
            ])
    
    # Create the dataframes from the columns and epochs
    summary_df = pd.DataFrame(columns=summary_columns)
    epoch_df = pd.DataFrame(index=range(1, num_epochs+1), columns=epoch_columns)
    epoch_df.index.name = 'epoch'

    return summary_df, epoch_df, exp_ids