
"""Baseline B6
    ● Same B3 steps A and B
        ○ For B, you will extract representations for each clip of 9 frames
    ● For C
        ○ Do LSTM on sequences from step B
    ● This is a model where LSTM is applied on the image level only"""


from A_prepare_data import get_train_val_test , get_annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np 
 
 

# Step 1: Define a function to encode labels
def encode_labels(label_dict):
    unique_labels = set()
    for class_id, files in label_dict.items():
        for file_id, label in files.items():
            unique_labels.add(label)
    
    label_to_index = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    return label_to_index


# Step 2: Create the Custom Dataset
class CustomFeatureDataset(Dataset):
    def __init__(self, file_paths, label_dict):
        self.file_paths = file_paths
        self.label_dict = label_dict
        self.label_to_index = encode_labels(label_dict)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load the feature vector from .npy file
        feature_vector = np.load(self.file_paths[idx])
        label = self.get_label(self.file_paths[idx])
        
        return torch.tensor(feature_vector, dtype=torch.float32), label

    def get_label(self, file_path):
        # Extract class_id and file_id from file_path
        parts = file_path.split('/')
        class_id = parts[-2]  # Assuming the directory name is the class_id
        file_id = parts[-1].split('.')[0]  # Extract the file id without extension
        
        # Return the encoded label
        return self.label_to_index[self.label_dict[class_id][file_id]]



# LSTM Model
class PlayerLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, num_layers=2, num_classes=8):
        super(PlayerLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer: input_size is 2048 (feature vector size), hidden_size is LSTM hidden state size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  #(batch_size, 9, 2048)
        
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
        # -1 means to take the last hidden state onlyyyy 
        # B7 will take all the out
        
        out = out[:, -1, :]  # (batch_size, hidden_size)
        
        # Pass through fully connected layer
        out = self.fc(out)  # (batch_size, num_classes)
        
        return out



def calculate_accuracy(predictions, labels):
    _, predicted_labels = torch.max(predictions, 1)
    correct = (predicted_labels == labels).sum().item()
    return correct / len(labels)


def train_lstm_scene_level(train_dataset, val_dataset, test_dataset, batch_size , epochs, lr=0.001):
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
    
    train_dict_scene_level_labels ,\
            val_dict_scene_level_labels ,\
                test_dict_scene_level_labels = get_annotations()

    train_dataset , val_dataset , test_dataset = get_train_val_test()


    # Create the dataset
    train_data = CustomFeatureDataset(train_dataset, train_dict_scene_level_labels)
    val_data = CustomFeatureDataset(val_dataset, val_dict_scene_level_labels)
    test_data = CustomFeatureDataset(test_dataset , test_dict_scene_level_labels)


    trained_model_on_scene = train_lstm_scene_level(train_data ,
                                                    val_data ,
                                                    test_data ,
                                                    batch_size = 32 ,
                                                    epochs = 60)