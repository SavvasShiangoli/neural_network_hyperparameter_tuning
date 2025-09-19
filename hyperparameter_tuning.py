import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import config_test as config

class NN(nn.Module):  # NN inherits from PyTorch's nn.Module
    def __init__(self, dropout_rate):
        super(NN, self).__init__() 

        self.flatten = nn.Flatten() # Explicitly flattens the layer
        self.fc1 = nn.Linear(784, 48)    # Automatically creates self.fc1.weights, self.fc1.biases
        self.fc2 = nn.Linear(48, 35)   # Automatically creates self.fc2.weights, self.fc2.biases
        self.fc3 = nn.Linear(35, 10)   # Automatically creates self.fc2.weights, self.fc2.biases

        # Single Dropout Layer that can be reused
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.flatten(x) # flattens the data
        x = F.relu(self.fc1(x)) # applies ReLU at the hidden layer
        x = self.dropout(x) # applies dropout after the hidden layer
        x = F.relu(self.fc2(x)) # applies ReLU at the hidden layer
        x = self.dropout(x) # applies dropout after the hidden layer
        x = self.fc3(x)

        return x
    
    def check_accuracy(self, loader, criterion):
        correct = 0
        total = 0
        total_loss = 0
        num_batches = 0
        
        self.eval() 
        with torch.no_grad():
            for data, targets in loader:
                data = data.to(device)
                targets = targets.to(device)
                
                scores = self(data)
                loss = criterion(scores, targets)

                total_loss += loss.item()
                num_batches += 1

                _, predictions = scores.max(1)
                total += targets.size(0)     
                correct += (predictions == targets).sum() 
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / num_batches

        self.train() 

        return accuracy.item(), avg_loss

# Set the seeds for cross validation
seeds = config.SEEDS

# Create results directory
Path("results").mkdir(exist_ok=True)

# CPU only
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    print(f"🌱 Seed set to {seed}")


# Set the device to run the model on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Global Hyperparameters
momentum = config.HYPERPARAMETERS['momentum'] # Momentum uses previous gradients as additional terms in calculating step size
dropout = config.HYPERPARAMETERS['dropout'] # probability a neuron will be zeroed

# Early Stop Parameters
train_test_gap_threshold = config.EARLY_STOPPING['train_test_gap_threshold'] # greater than this = too much overfitting
min_train_gradient = config.EARLY_STOPPING['min_train_gradient'] # less than this for 
min_train_gradient_epochs = config.EARLY_STOPPING['min_train_gradient_epochs'] # Number of epochs to allow breakout before detecting plateu

# Dataframes and experiment ids for storing and saving the data
summary_df, epoch_df, exp_ids = config.create_summary_dataframes()


def train_with_seed(learning_rate, batch_size, momentum, num_epochs, seed, epoch_df, exp_id):
    # Set the seed before the model is initialised
    set_seed(seed)

    # Early finishing counter
    consecutive_low_improvement = 0

    # Load Datasets for training and testing
    train_data = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True) 
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # Initialise the network
    model = NN(dropout_rate=dropout).to(device) # Initialise the network to the device

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Optimizer updates model parameters, uses stochastic gradient descent. Learning rate and momentum affect parameter updates
    optimizer = optim.SGD(model.parameters(), 
                        lr=learning_rate, 
                        momentum=momentum) 
    
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 
    #                                             T_0=6, 
    #                                             T_mult=2, 
    #                                             eta_min=learning_rate * 0.1  # 10% minimum
    #                                             )
    
    # TRAINING LOOP
    for epoch in range(num_epochs):

        print(f"Epoch {epoch+1}/{num_epochs}")

        #### LEARNING PARAMETER UDPATES WITH PROPORTIONAL AND GRADIENT CONTROL ####

        # If epochs = 6 create first moving average data value from epochs 1-5 - won't be used until epoch 7
        if epoch == 5:
            train_moving_av_n2 = epoch_df.loc[epoch-5:epoch, f'{exp_id}_seed{seed}_train_acc'].mean()
            test_moving_av_n2 = epoch_df.loc[epoch-5:epoch, f'{exp_id}_seed{seed}_test_acc'].mean()

        # If epochs > 7 start to utlise the proportional and gradient based attenuation of the learning rate
        if epoch >= 6:

            # Move the previous n2 data to n1
            train_moving_av_n1 = train_moving_av_n2
            test_moving_av_n1 = test_moving_av_n2

            # Calculate the new 5 epoch moving average
            train_moving_av_n2 = epoch_df.loc[epoch-5:epoch, f'{exp_id}_seed{seed}_train_acc'].mean()
            test_moving_av_n2 = epoch_df.loc[epoch-5:epoch, f'{exp_id}_seed{seed}_test_acc'].mean()

            # Break Case 1 - Check if the train_test_gap is greater than the threshold
            train_test_gap = train_moving_av_n2 - test_moving_av_n2
            if train_test_gap >= train_test_gap_threshold:
                print(f'Test Train Gap Threshold {train_test_gap_threshold} exceeded')
                break

            # Calculate the gradients
            train_grad = train_moving_av_n2 - train_moving_av_n1
            test_grad = test_moving_av_n2 - test_moving_av_n1

            # Break Case 2 - test accuracy improvement < 0.1 for consecutive epochs
            if test_grad < min_train_gradient and epoch >= 6:  # After sufficient epochs
                consecutive_low_improvement += 1
                if consecutive_low_improvement >= min_train_gradient_epochs:
                    print(f'Early stopping: Test accuracy improvement < {min_train_gradient}% for 5 consecutive epochs')
                    break
            else:
                consecutive_low_improvement = 0
            
            # Calculate the gradient factor checking with protection for zero train_grad
            if train_grad != 0:
                gradient_factor = test_grad/train_grad
            else:
                gradient_factor = 1.0

            # Calculate the proportional factor checking if train accuracy > test accuracy
            if train_moving_av_n2 > test_moving_av_n2:
                x = train_test_gap_threshold - train_test_gap
                train_test_gap_factor = (0.9 / train_test_gap_threshold) * x + 0.1
            else:
                train_test_gap_factor = 1
            
            # Calculate the new learning rate and udpate the optimizer
            new_learning_rate = learning_rate # * gradient_factor * train_test_gap_factor

            if new_learning_rate != learning_rate:
                print(f'Learning rate updated: {learning_rate:.6f} → {new_learning_rate:.6f}')

            for param_group in optimizer.param_groups:
                param_group['lr'] = new_learning_rate

        model.train() # ensures the model is in training mode

        for batch_idx, (data, targets) in enumerate(train_loader): # data = images, target is the correct identification, want to get the batch index
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Forward
            scores = model(data)
            loss = criterion(scores, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient Descent
            optimizer.step()

        # # Update the learning rate for the next epoch using the scheduler
        # scheduler.step()

        # # Extract and print out the new learning rate
        # new_lr = optimizer.param_groups[0]['lr']
        # print(f"Updated learning rate for next epoch: {new_lr:.6f}")
        
        # Check accuracy after each epoch
        train_acc, train_loss = model.check_accuracy(train_loader, criterion)
        test_acc, test_loss = model.check_accuracy(test_loader, criterion)

        # Save epoch data to the datframe
        epoch_df.loc[epoch+1, f'{exp_id}_seed{seed}_train_acc'] = train_acc
        epoch_df.loc[epoch+1, f'{exp_id}_seed{seed}_train_loss'] = train_loss
        epoch_df.loc[epoch+1, f'{exp_id}_seed{seed}_test_acc'] = test_acc
        epoch_df.loc[epoch+1, f'{exp_id}_seed{seed}_test_loss'] = test_loss

        # Save every 5 epochs = saftey encase the model get's interrupted
        if (epoch + 1) % 5 == 0:
            epoch_df.to_csv("results/epochdata_arch_test1.csv")
            print(f"Checkpoint saved at epoch {epoch + 1}")

        print(f"Train Accuracy: {train_acc:.2f}%, Train Loss: {train_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%, Test Loss: {test_loss:.4f}")

    
    # Return final accuracies after ALL epochs complete
    final_train_acc, final_train_loss = model.check_accuracy(train_loader, criterion)
    final_test_acc, final_test_loss = model.check_accuracy(test_loader, criterion)

    print(f"Final Train Accuracy: {final_train_acc:.2f}%, Final Train Loss: {final_train_loss:.4f}")
    print(f"Final Test Accuracy: {final_test_acc:.2f}%, Final Test Loss: {final_test_loss:.4f}")
    
    return (final_train_acc, final_train_loss), (final_test_acc, final_test_loss)


def evaluate_hyperparameters(learning_rate, batch_size, momentum, num_epochs, seeds, epoch_df, exp_id):
    train_results = []
    test_results = []
    for seed in seeds:
        (train_acc, train_loss), (test_acc, test_loss) = train_with_seed(learning_rate, batch_size, momentum, num_epochs, seed, epoch_df, exp_id)
        train_results.append((train_acc, train_loss))
        test_results.append((test_acc, test_loss))

    # Separate accuracies and losses
    train_accs = [result[0] for result in train_results]
    train_losses = [result[1] for result in train_results]
    test_accs = [result[0] for result in test_results]
    test_losses = [result[1] for result in test_results]
    
    # Return mean and standard deviation for both train and test, accuracy and loss
    return {
        'mean_train_accuracy': np.mean(train_accs),
        'std_train_accuracy': np.std(train_accs),
        'mean_train_loss': np.mean(train_losses),
        'std_train_loss': np.std(train_losses),
        'mean_test_accuracy': np.mean(test_accs),
        'std_test_accuracy': np.std(test_accs),
        'mean_test_loss': np.mean(test_losses),
        'std_test_loss': np.std(test_losses),
        'all_results': {'train': train_results, 'test': test_results}
    }

# Main hyperparameter search
print("Starting hyperparameter search...")
print("=" * 50)

combinations = config.get_search_combinations()

for combination, exp_id in zip(combinations, exp_ids):
    learning_rate = combination['learning_rate']
    batch_size = combination['batch_size']
    num_epochs = combination['num_epochs']
    momentum = combination['momentum']
    architecture = config.ARCHITECTURES[combination['architecture']]

    print(f"\nTesting Architecture: {combination['architecture']}, Learning Rate: {learning_rate}, Momentum: {momentum}, Batch Size: {batch_size}")

    metrics = evaluate_hyperparameters(learning_rate, batch_size, momentum, num_epochs, seeds, epoch_df, exp_id)

    # Adding the data to the summary dataframe 
    summary_df.loc['train_acc_mean', f'{exp_id}'] = metrics['mean_train_accuracy']
    summary_df.loc['train_acc_std', f'{exp_id}'] = metrics['std_train_accuracy']
    summary_df.loc['train_loss_mean', f'{exp_id}'] = metrics['mean_train_loss']
    summary_df.loc['train_loss_std', f'{exp_id}'] = metrics['std_train_loss']
    summary_df.loc['test_acc_mean', f'{exp_id}'] = metrics['mean_test_accuracy']
    summary_df.loc['test_acc_std', f'{exp_id}'] = metrics['std_test_accuracy']
    summary_df.loc['test_loss_mean', f'{exp_id}'] = metrics['mean_test_loss']
    summary_df.loc['test_loss_std', f'{exp_id}'] = metrics['std_test_loss']

    # Save summary data to a seperate file
    summary_df.to_csv("results/summary_arch_test1.csv")

    print(f"Train: {metrics['mean_train_accuracy']:.2f}% ± {metrics['std_train_accuracy']:.2f}%, Loss: {metrics['mean_train_loss']:.4f} ± {metrics['std_train_loss']:.4f}")
    print(f"Test:  {metrics['mean_test_accuracy']:.2f}% ± {metrics['std_test_accuracy']:.2f}%, Loss: {metrics['mean_test_loss']:.4f} ± {metrics['std_test_loss']:.4f}")
    print("-" * 40)

