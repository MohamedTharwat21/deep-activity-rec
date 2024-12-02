"""LSTM-2 on the scene level"""
"""Splitting the dataset"""

import numpy as np
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


"""scene_level_annotations"""
from Baseline_6.A_prepare_data import get_annotations

root = '/kaggle/input/lstm9-2048-b7-feature-vectors-472598a/kaggle/working/features/crop-level/Lstm_B7_9_2048'
# Assuming you have defined your dataset classes and DataLoader as `train_loader`, `val_loader`
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_ids = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31",
                "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]
val_ids = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]
test_ids = ["4", "5", "9", "11", "14", "20", "21", "25", "29", "34", "35", "37", "43", "44", "45", "47"]


"""train , val and test dataset (features of the first LSTM-1)"""

def get_train_val_test_scene_lvl():

    train_dataset , val_dataset , test_dataset = [] ,[] ,[]
    for vid_id in range(55) :
        if str(vid_id) in train_ids :
            path = os.path.join(root , str(vid_id))
            for seq in os.listdir(path) :
                train_dataset.append(os.path.join(root , str(vid_id) , seq))

        elif str(vid_id) in val_ids :
            path = os.path.join(root , str(vid_id))
            for seq in os.listdir(path) :
                val_dataset.append(os.path.join(root , str(vid_id) , seq))
                
        else : 
            path = os.path.join(root , str(vid_id))
            for seq in os.listdir(path) :
                test_dataset.append(os.path.join(root , str(vid_id) , seq))

    return train_dataset , val_dataset  , test_dataset




"""Custom_dataset"""
class PlayerSequenceDataset(Dataset):
    def __init__(self, feature_files, label_dict):
        self.features = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        all_labels = []
        
        # Collect all labels for encoding
        for file_path in feature_files:
            parent_dir = file_path.split('/')[-2]  # Get the parent directory (e.g., '0')
            file_id = file_path.split('/')[-1].replace('.npy', '')  # Get the file name (e.g., '12345')
            label = label_dict[parent_dir][file_id]  # Get the label for this instance
            all_labels.append(label)
        
        # Fit the label encoder on the full set of labels
        self.label_encoder.fit(all_labels)
        
        # Now process features and encode labels
        for file_path in feature_files:
            parent_dir = file_path.split('/')[-2]
            file_id = file_path.split('/')[-1].replace('.npy', '')
            
            # Load features
            features = np.load(file_path)  # Shape: (9, 2048)
            
            # Get corresponding label
            label = label_dict[parent_dir][file_id]
            encoded_label = self.label_encoder.transform([label])[0]  # Encode the label

            # Store features and label
            self.features.append(features)  # Each file is (9, 2048)
            self.labels.append(encoded_label)  # Corresponding encoded label

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Return individual sequence and encoded label
        feature_seq = torch.tensor(self.features[idx], dtype=torch.float32)  # Shape: (9, 2048)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Encoded label
        
        return feature_seq, label

    


"""LSTM Model"""

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=2048, hidden_size=128, num_layers=2, num_classes=8):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        h0 = torch.zeros(2, x.size(0), 128).to(x.device)  # Initialize hidden state
        c0 = torch.zeros(2, x.size(0), 128).to(x.device)  # Initialize cell state
        
        out,_ = self.lstm(x, (h0, c0))  # LSTM output
        out = out[:, -1, :]  # Take the output of the last LSTM cell (many-to-one)
        out = self.fc(out)  # Fully connected layer
        return out

    
    
    
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
     
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Accuracy calculation
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate training accuracy for the epoch
        train_accuracy = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)

        # Validation step
        val_loss, val_accuracy = evaluate_model(model, val_loader)
        
        # Print loss and accuracy for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    
    print('Finished Training!')


    
    
def evaluate_model(model, val_loader):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return val_loss, accuracy
            
            


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')




def inference(model, example_features):
    model.eval()
    example_features = torch.tensor(example_features, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        outputs = model(example_features)
        _, predicted = torch.max(outputs, 1)
        print(f'Predicted class: {predicted.item()}')
    



if __name__ == "__main__" :

    """"train dataset"""
    train_dataset , val_dataset  , test_dataset = get_train_val_test_scene_lvl()

    """scene_level_annotations"""
    train_dict_scene_level_labels , val_dict_scene_level_labels ,test_dict_scene_level_labels = get_annotations()

    """Custom Dataset"""
    train_data = PlayerSequenceDataset(train_dataset, train_dict_scene_level_labels)
    val_data = PlayerSequenceDataset(val_dataset, val_dict_scene_level_labels)
    test_data = PlayerSequenceDataset(test_dataset, test_dict_scene_level_labels)

    """constructing the dataloader"""
    train_loader = DataLoader(train_data , batch_size = 32 , shuffle = True)
    val_loader = DataLoader(val_data , batch_size = 32 , shuffle = False)
    test_loader = DataLoader(test_data , batch_size = 32 , shuffle = False)

    """Initialize model"""
    model = LSTMClassifier(input_size=2048, hidden_size=128, num_layers=2, num_classes=8).to(device)

    """Train the model"""
    train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001)

    """Test the model"""
    test_model(model, test_loader)

    
    """Example usage"""
    example_features = np.load('/kaggle/input/lstm9-2048-b7-feature-vectors-472598a/kaggle/working/features/crop-level/Lstm_B7_9_2048/0/13286.npy')  # Load a single example
    inference(model, example_features)