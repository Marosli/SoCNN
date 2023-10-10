import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset  
import scipy.signal

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("CUDA GPU is available!")
else:
    print("CUDA GPU is not available!")

# Data extraction
path='ATHLON/batteryCombined.log'
# path='ATHLON/battery.log.1'

levels = []
voltages = []
currents = []

with open(path, 'r') as file:
    for line in file:
        words = line.split()
        if len(words) >= 9 and words[2] == "INFO" and words[3] == "Level:" and words[5] == "Voltage:" and words[7] == "Current:":
            level = float(words[4])
            voltage = float(words[6])
            current = float(words[8])

            # Filter out voltage values lower than 24
            if voltage >= 24 and level>=20:
                levels.append(level)
                voltages.append(voltage)
                currents.append(current)

def create_mini_batches(input_seq, output_seq, batch_size):
    mini_batches = []
    data_size = len(input_seq)
    for i in range(0, data_size, batch_size):
        inputs = input_seq[i:min(i + batch_size, data_size)]
        targets = output_seq[i:min(i + batch_size, data_size)]
        mini_batches.append((inputs, targets))
    return mini_batches



# Data preparation
data = list(zip(voltages, currents, levels))
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

voltages_train, currents_train, levels_train = zip(*train_data)
voltages_test, currents_test, levels_test = zip(*test_data)


# plt.plot(levels, label='True Values')
# plt.plot(levels_train, label='True Values')
# # Plot predicted values
# plt.rcParams['figure.figsize'] = (16, 4)
# # Add labels, title, and legend
# plt.xlabel('Time Steps')
# plt.ylabel('Output')
# plt.title('True vs. Predicted Values')
# plt.legend()
# plt.savefig('RecentPlots/plot1.png')

scaler = MinMaxScaler()
train_input_data = scaler.fit_transform(list(zip(voltages_train, currents_train)))
test_input_data = scaler.transform(list(zip(voltages_test, currents_test)))

output_scaler = MinMaxScaler()
train_output_data = output_scaler.fit_transform(np.array(levels_train)[:, None])
test_output_data = output_scaler.transform(np.array(levels_test)[:, None])

def create_sequences(input_data, output_data, sequence_length):
    input_sequences = []
    output_sequences = []
    for i in range(len(input_data) - sequence_length):
        input_sequences.append(input_data[i:i+sequence_length])
        output_sequences.append(output_data[i+sequence_length])
    return torch.tensor(np.array(input_sequences)), torch.tensor(np.array(output_sequences))


sequence_length = 32
train_input_seq, train_output_seq = create_sequences(train_input_data, train_output_data, sequence_length)
test_input_seq, test_output_seq = create_sequences(test_input_data, test_output_data, sequence_length)

print(f"Size of test_output_seq: {test_output_seq.shape}")


batch_size = 64
train_mini_batches = create_mini_batches(train_input_seq, train_output_seq, batch_size)
test_mini_batches = create_mini_batches(test_input_seq, test_output_seq, batch_size)

print(f"Size of test_mini_batches: {len(test_mini_batches)}")


# Define the LSTM model
class SoCPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SoCPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
                
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Model parameters
input_size = 2
hidden_size = 128
num_layers = 1
output_size = 1
learning_rate = 0.005
num_epochs = 20


# Create the model
model = SoCPredictor(input_size, hidden_size, num_layers, output_size).to(device)

# Load model
# checkpoint_state_dict = torch.load(checkpoint_file_path)
# model.load_state_dict(checkpoint_state_dict)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)


lossData=[]
# plt.ion()
# Train the model
all_train_predictions = []
all_test_predictions = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for inputs, targets in train_mini_batches:
        # Move mini-batches to the device
        inputs, targets = inputs.float().to(device), targets.float().to(device)

        optimizer.zero_grad()
        train_predictions = model(inputs)  

        if epoch == num_epochs - 1:
            all_train_predictions.extend(train_predictions.detach().cpu().numpy())

        loss = criterion(train_predictions, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    # scheduler.step(epoch_loss)
    print(loss)
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        lossData.append(epoch_loss)
        checkpoint_path = f"model_checkpoint_epoch_{epoch+1}.pth"
        full_checkpoint_path = os.path.join(os.getcwd(), checkpoint_path)
        torch.save(model.state_dict(), full_checkpoint_path)


# Evaluate the model
model.eval()
all_test_predictions = []
with torch.no_grad():
    for inputs, targets in test_mini_batches:
        # Move mini-batches to the device
        inputs, targets = inputs.float().to(device), targets.float().to(device)

        test_predictions = model(inputs)
        all_test_predictions.extend(test_predictions.detach().cpu().numpy())

from sklearn.metrics import mean_squared_error

# Rescale the predicted and true values to their original range
true_values = output_scaler.inverse_transform(np.array(test_output_seq))
predicted_values = output_scaler.inverse_transform(np.array(all_test_predictions))

# Calculate MSE
mse = mean_squared_error(true_values, predicted_values)
print(f"Test MSE: {mse}")

# Create a plot
plt.figure(figsize=(16, 4))
plt.plot(true_values, label='True Values', color='blue')
plt.plot(predicted_values, label='Predicted Values', color='red')

# Add labels, title, and legend
plt.xlabel('Time Steps')
plt.ylabel('Output')
plt.title('True vs. Predicted Values on Test Data')
plt.legend()

plt.savefig('RecentPlots/plot7.png')


# # Convert the train predictions to their original scale
# all_train_predictions_unscaled = output_scaler.inverse_transform(all_train_predictions)
# all_test_predictions_unscaled = output_scaler.inverse_transform(all_test_predictions)



# plt.figure()
# plt.plot(levels_test[sequence_length:], label='True Test Values')
# plt.plot(all_test_predictions_unscaled, label='Test Predicted Values')
# plt.xlabel('Time Steps')
# plt.ylabel('Output')
# plt.title('Test True vs. Predicted Values')
# plt.legend()
# plt.savefig('RecentPlots/plot7.png')
