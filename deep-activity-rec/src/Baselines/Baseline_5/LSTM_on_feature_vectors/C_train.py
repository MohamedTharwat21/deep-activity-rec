import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

"""
import the custom train , val and test datasets
"""
from B_custom_train_val_test_datasets import train_data , val_data , test_data




# LSTM Model
# class PlayerLSTM(nn.Module):
#     def __init__(self, input_size=2048, hidden_size=128, num_layers=2, num_classes=10):
#         super(PlayerLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)
    
#     def forward(self, x):
#            PyTorch's LSTM default behavior,
#            which implicitly initializes the hidden state and cell state to zero if not provided
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])  # Only take the output from the last time step
#         return out

    
    

class PlayerLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, num_layers=2, num_classes=9):
        super(PlayerLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer: input_size is 2048 (feature vector size), hidden_size is LSTM hidden state size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer to map the LSTM output to class scores
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x is of shape (batch_size, seq_length, input_size), e.g., (batch_size, 9, 2048)
        
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_size)
        
        # Take the output from the last time step (for classification)
        out = out[:, -1, :]  # (batch_size, hidden_size)
        
        # Pass through fully connected layer
        out = self.fc(out)  # (batch_size, num_classes)
        
        return out




def calculate_accuracy(predictions, labels):
    _, predicted_labels = torch.max(predictions, 1)
    correct = (predicted_labels == labels).sum().item()
    return correct / len(labels)





def train_lstm(train_dataset, val_dataset, test_dataset, batch_size, epochs, lr=0.001):
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Instantiate the model, loss function, and optimizer
    model = PlayerLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for features, labels in train_loader:
            # Move data to device
            features, labels = features.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)  # Shape: (batch_size, num_classes)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss and accuracy
            running_loss += loss.item()
            correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
            total_predictions += labels.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_predictions / total_predictions
        
        # Validation phase
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy*100:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy*100:.2f}%')

    print('Training complete')

    # Testing phase
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%')
    
    return model





def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
            total_predictions += labels.size(0)
    
    average_loss = running_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions
    return average_loss, accuracy





if __name__ == "__main__" :
    # Train the LSTM model with training and validation, and evaluate on the test set
    trained_model = train_lstm(train_data, 
                               val_data,
                               test_data , 
                               batch_size = 32 , 
                               epochs = 10)
    
    # save the model
    torch.save(trained_model.state_dict() , 'B5_v1_LSTM.pth')