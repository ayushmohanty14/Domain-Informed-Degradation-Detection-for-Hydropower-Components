import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence  # Use pad_sequence from torch.nn.utils.rnn
import matplotlib.pyplot as plt
########################################################################################################################
### DATA PREPROCESSING
# Load the data from CSV files
data_power_train = pd.read_csv('Data/Detection2/power_train.csv')
data_vibration_train = pd.read_csv('Data/Detection2/vibration_train.csv')
data_power_test = pd.read_csv('Data/Detection2/power_test.csv')
data_vibration_test = pd.read_csv('Data/Detection2/vibration_test.csv')

data_power_test = pd.read_csv('Data/Detection2/power_test.csv')
data_vibration_test = pd.read_csv('Data/Detection2/vibration_test.csv')

# Get the column names and remove the timestamp column
power_train_column_names = data_power_train.columns
power_test_column_names = data_power_test.columns

# Rearrange the columns in data_power_train and data_vibration_train
data_power_train = data_power_train[power_train_column_names]
data_vibration_train = data_vibration_train[power_train_column_names]

# Rearrange the columns in data_power_degraded and data_vibration_degraded
data_power_test = data_power_test[power_test_column_names]
data_vibration_test = data_vibration_test[power_test_column_names]

# Assuming the data in CSV starts from 2nd row (index 1) and timestamp is in the first column, extract the data values
power_train = data_power_train.iloc[1:, 1:].values.astype(np.float32)
vibration_train = data_vibration_train.iloc[1:, 1:].values.astype(np.float32)
power_test = data_power_test.iloc[1:, 1:].values.astype(np.float32)
vibration_test = data_vibration_test.iloc[1:, 1:].values.astype(np.float32)

# Define the sequence length for LSTM
sequence_length = 91

# Function to trim first and last few rows from the data (depending on sequence_length)
def create_sequences(data):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length, :]
        sequences.append(sequence)
    return sequences

power_train_sequences = create_sequences(power_train)
vibration_train_sequences = create_sequences(vibration_train)
power_test_sequences = create_sequences(power_test)
vibration_test_sequences = create_sequences(vibration_test)

power_train_tensor = torch.tensor(power_train_sequences, dtype=torch.float32)
vibration_train_tensor = torch.tensor(vibration_train_sequences, dtype=torch.float32)
power_test_tensor = torch.tensor(power_test_sequences, dtype=torch.float32)
vibration_test_tensor = torch.tensor(vibration_test_sequences, dtype=torch.float32)
########################################################################################################################
### LSTM MODEL

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

power_train_cols = power_train_tensor.shape[-1]
power_test_cols = power_test_tensor.shape[-1]
input_size = 2   # Power and vibration are the input features
hidden_size = 32
num_layers = 3
output_size = 30  # Predicting the next time-step of vibration

# Initializing the LSTM model
model = LSTM(input_size, hidden_size, num_layers, output_size)
########################################################################################################################
## TRAINING

# Training configuration
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

# train_losses = []
#
# # Training loop
# for epoch in range(num_epochs):
#     print('Epoch: {}'.format(epoch + 1))
#     epoch_loss = 0.0
#     for col in range(power_train_cols):
#         train_dataset = TensorDataset(power_train_tensor[:, :, col].unsqueeze(2), vibration_train_tensor[:, :, col].unsqueeze(2))
#         train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
#         for batch_power, batch_vibration in train_loader:
#             optimizer.zero_grad()
#             combined_data = torch.cat((batch_power[:, :(sequence_length-31), :], batch_vibration[:, :(sequence_length-31), :]), dim=2)
#             outputs = model(combined_data)
#             loss = criterion(outputs, batch_vibration[:, -30:, 0])
#             loss.backward()
#             optimizer.step()
#         epoch_loss += loss.item()
#     train_losses.append(epoch_loss / (power_train_cols*len(train_loader)))
#
# # Plot training error
# plt.figure()
# plt.plot(train_losses)
# plt.xlabel('Epoch')
# plt.ylabel('MSE Loss')
# plt.title('Training Error')
# plt.grid()
# plt.show()
#
# # Save the trained model
# torch.save(model.state_dict(), 'trained_model.pth')
########################################################################################################################
## TESTING

# Load the trained state dictionary
model.load_state_dict(torch.load('trained_model.pth'))

# Set the model to evaluation mode (important for Dropout and BatchNorm layers)
model.eval()

# Define the loss function
criterion = nn.MSELoss()

power_test_sequences = create_sequences(power_test)
vibration_test_sequences = create_sequences(vibration_test)

power_test_tensor = torch.tensor(power_test_sequences, dtype=torch.float32)
vibration_test_tensor = torch.tensor(vibration_test_sequences, dtype=torch.float32)

# Testing loop
test_losses = []

for col in range(power_test_cols):
    test_dataset = TensorDataset(power_test_tensor[:, :, col].unsqueeze(2), vibration_test_tensor[:, :, col].unsqueeze(2))
    test_loader = DataLoader(test_dataset, batch_size=15, shuffle=False)

    column_test_losses = []
    with torch.no_grad():
        for batch_power, batch_vibration in test_loader:
            combined_data = torch.cat((batch_power[:, :(sequence_length-31), :], batch_vibration[:, :(sequence_length-31), :]), dim=2)
            outputs = model(combined_data)
            loss = criterion(outputs, batch_vibration[:, -30:, 0])
            column_test_losses.append(loss.item())
    test_losses.append(column_test_losses)

# Convert test_errors to a numpy array for easier analysis and visualization
test_losses = np.array(test_losses)
print("Test Loss Shape:", test_losses.shape)

# Plot testing error
for col in range(power_test_cols):
    plt.figure()
    plt.plot(test_losses[col], linestyle='dotted', marker='o', color='green')
    plt.xlabel('Operational Time (Days)')
    plt.ylabel('MSE Loss')
    plt.title('Testing Error for {}'.format(power_test_column_names[col+1]))
    plt.ylim(0, 50)
    plt.grid()
    plt.show()
########################################################################################################################
# ## TESTING ON TRAINING DATASET (to check for outliers)
# # Load the trained state dictionary
# model.load_state_dict(torch.load('trained_model.pth'))
#
# # Set the model to evaluation mode (important for Dropout and BatchNorm layers)
# model.eval()
#
# # Define the loss function
# criterion = nn.MSELoss()
#
# power_test_sequences = create_sequences(power_train)
# vibration_test_sequences = create_sequences(vibration_train)
#
# power_test_tensor = torch.tensor(power_test_sequences, dtype=torch.float32)
# vibration_test_tensor = torch.tensor(vibration_test_sequences, dtype=torch.float32)
#
# # Testing loop
# test_losses = []
#
# for col in range(power_train_cols):
#     test_dataset = TensorDataset(power_train_tensor[:, :, col].unsqueeze(2), vibration_train_tensor[:, :, col].unsqueeze(2))
#     test_loader = DataLoader(test_dataset, batch_size=15, shuffle=False)
#
#     column_test_losses = []
#     with torch.no_grad():
#         for batch_power, batch_vibration in test_loader:
#             combined_data = torch.cat((batch_power[:, :(sequence_length-1), :], batch_vibration[:, :(sequence_length-1), :]), dim=2)
#             outputs = model(combined_data)
#             loss = criterion(outputs, batch_vibration[:, -1, :])
#             column_test_losses.append(loss.item())
#     test_losses.append(column_test_losses)
#
# # Convert test_errors to a numpy array for easier analysis and visualization
# test_losses = np.array(test_losses)
# print("Test Loss Shape:", test_losses.shape)
#
# # Plot testing error
# for col in range(power_train_cols):
#     plt.figure()
#     plt.plot(test_losses[col], linestyle='dotted', marker='o')
#     plt.xlabel('Batched Time Step')
#     plt.ylabel('MSE Loss')
#     plt.title('Testing Error for {}'.format(power_train_column_names[col+1]))
#     plt.ylim(0, 100)
#     plt.grid()
#     plt.show()