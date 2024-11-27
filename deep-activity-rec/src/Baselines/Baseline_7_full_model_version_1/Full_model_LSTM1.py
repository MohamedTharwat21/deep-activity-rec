"""
This script is for loading the data and train lstm-1 on the person level (crop level)
and extract features using the trained lstm-1 after finishing training 
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import math
"""person_level_annotations"""
from Baseline_5.LSTM_on_feature_vectors.A_prepare_data_splits_and_annotations import get_person_lvl_annot
"""scene_level_annotations"""
from Baseline_6.A_prepare_data import get_annotations



videos_annot = 'write_your_path_here'
train_ids = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31",
                "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]

val_ids = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]
test_ids = ["4", "5", "9", "11", "14", "20", "21", "25", "29", "34", "35", "37", "43", "44", "45", "47"]




"""Features_for_LSTM1"""
def get_trainValTest_dataset(root) :

    # the features extracted using B3-classifier 
    # we already extracted it before , so we will use it here 

    # root = '/kaggle/input/resnet-feature-vectors-9-12-472598a/kaggle/working/features/image-level/resnet'
    train_dataset ,val_dataset ,test_dataset = [] , [] , []  

    for vid_id in range(55) : 
        vid_id = str(vid_id)
        path = os.path.join(root , str(vid_id))
        for seq in os.listdir(path) :

            if vid_id in train_ids : 
                train_dataset.append(os.path.join(root , str(vid_id) , seq))

            elif vid_id in val_ids :
                val_dataset.append(os.path.join(root , str(vid_id) , seq))

            else :
                test_dataset.append(os.path.join(root , str(vid_id) , seq))
    return train_dataset , val_dataset , test_dataset




"""Custom_person_level_dataset"""
label_encoding = {
    'standing': 0,
    'setting': 1,
    'digging': 2,
    'spiking': 3,
    'blocking': 4,
    'moving': 5,
    'jumping': 6,
    'falling': 7,
    'waiting': 8
}




class PlayerSequenceDataset(Dataset):
    def __init__(self, feature_files, label_dict, label_encoding):
        self.features = []
        self.labels = []
        self.label_encoding = label_encoding
        
        # Load all player sequences from all feature files
        for file_path in feature_files:
            # Extract parent directory and file_id
            parent_dir = file_path.split('/')[-2]  # Get the parent directory (e.g., '0')
            file_id = file_path.split('/')[-1].replace('.npy', '')  # Get the file name (e.g., '55594')

            # Load features
            features = np.load(file_path)  # Shape: (9, num_players, 2048)
            num_players = features.shape[1]
            
            # Get corresponding labels from the nested dictionary (parent_dir -> file_id -> labels)
            labels = label_dict[parent_dir][file_id]  # list of labels for the players
            encoded_labels = [self.label_encoding[label] for label in labels]

            # Store each player as an individual instance
            for player_idx in range(num_players):
                self.features.append(features[:, player_idx, :])  # Shape: (9, 2048)
                self.labels.append(encoded_labels[player_idx])    # Single label for the player
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Return individual player's sequence and label
        player_features = torch.tensor(self.features[idx], dtype=torch.float32)  # Shape: (9, 2048)
        player_label = torch.tensor(self.labels[idx], dtype=torch.long)  # Single integer label
        
        return player_features, player_label




"""
LSTM-1 training 
"""
class PlayerLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=2048, num_layers=2, num_classes=9):
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
        
        # we want to return the 9 timesteps (in inference)
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





"""Feature Extraction using lstm-1"""

# Create an instance of the model class first (same as the one you used for training)
class PlayerLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=2048, num_layers=2):  # Updated to 2 LSTM layers
        super(PlayerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 9)  # Map to 9 time steps instead of 2048
    
    def forward(self, x):
        # x shape: (batch_size, 9, 2048)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, 9, hidden_size)
        output = self.fc(lstm_out)  # Shape: (batch_size, 9, 9)
        # return output
        # why not easily to return lstm_out to have (9 * 2048) ??
        return lstm_out



    
def extract_sequence_representation(file_path, model):
    """
    Extract a sequence-level representation from a file, considering variable player count.
    This uses the entire sequence of hidden states for each player (not just the last one).
    
    Args:
        file_path (str): Path to the .npy file.
        model (torch.nn.Module): The trained LSTM model.
    
    Returns:
        torch.Tensor: A sequence-level representation after max-pooling over all player representations.
    """
    # Load the .npy file, which contains features in shape (9, num_players, 2048)
    features = np.load(file_path)  # Shape: (9, num_players, 2048)
    num_players = features.shape[1]
    
    # Store the representations for all players
    player_representations = []
    
    # Process each player's sequence independently
    for player_idx in range(num_players):
        # Extract the player's sequence: Shape (9, 2048)
        player_seq = torch.tensor(features[:, player_idx, :]).unsqueeze(0)  # Shape: (1, 9, 2048)
        
        with torch.no_grad():
            # Pass through the trained LSTM model and get output for all 9 time steps
            player_output = model(player_seq)  # Shape: (1, 9, 2048), where 9 is the number of time steps
            player_representations.append(player_output.squeeze(0))  # Remove batch dimension, Shape: (9, 2048)
    
    # Stack player representations into a single tensor: Shape (num_players, 9, 2048)
    player_representations = torch.stack(player_representations)  # Shape: (num_players, 9, 2048)
    
    # Apply max-pooling across the player representations for each time step
    max_pooled_representation = torch.max(player_representations, dim=0)[0]  # Shape: (9, 2048)
    
    return max_pooled_representation




# Let's save the representations 
def process_and_save_files(file_paths, base_output_dir, model):
    """
    Processes a list of file paths, extracts features, and saves them using np.save.
    
    Args:
        file_paths (list of str): List of input .npy file paths.
        base_output_dir (str): Base directory to save the output features.
        model (torch.nn.Module): The trained LSTM model.
    """
    for file_path in file_paths:
        
        # Extract directory structure from file_path (e.g., '1/43825.npy')
        rel_path = os.path.relpath(file_path, '/kaggle/input/resnet-feature-vectors-9-12-472598a/kaggle/working/features/image-level/resnet')
        output_path = os.path.join(base_output_dir, rel_path)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Extract the max-pooled representation from the input file
        max_pooled_representation = extract_sequence_representation(file_path, model)
        # print(max_pooled_representation.shape)
        
        # Save the max-pooled representation as a NumPy array
        np.save(output_path, max_pooled_representation.numpy())
        print(f"Saved max-pooled representation to {output_path}")




        
    
if __name__ == "__main__" :

    """scene_level_annotations"""
    train_dict_scene_level_labels , val_dict_scene_level_labels ,test_dict_scene_level_labels = get_annotations()


    """person_level_annotations"""
    train_dict_person_level_labels , val_dict_person_level_labels ,test_dict_person_level_labels = get_person_lvl_annot()
    

    """getting the dataset from the extracted features"""
    root = '/kaggle/input/resnet-feature-vectors-9-12-472598a/kaggle/working/features/image-level/resnet'
    train_dataset , val_dataset , test_dataset = get_trainValTest_dataset(root=root)
     

    """Creating Custom dataset"""
    train_data = PlayerSequenceDataset(train_dataset, train_dict_person_level_labels , label_encoding)
    val_data = PlayerSequenceDataset(val_dataset, val_dict_person_level_labels , label_encoding)
    test_data = PlayerSequenceDataset(test_dataset, test_dict_person_level_labels , label_encoding)


    """Training Lstm-1 on the crop or person level"""
    # Train the LSTM model with training and validation, and evaluate on the test set
    trained_model = train_lstm(train_data, val_data, test_data , batch_size = 32, epochs = 10)
    

    """Save LSTM-1"""
    torch.save(trained_model.state_dict(), 'lstm1_b7_full_model_parameters.pth')


    """feature_extraction"""

    """ it will take much more time than B5 
        as , we here in B7 , we are returning for each player the whole hidden states
        of all the timesteps .

        your input is (12 * 9  * 2048) and output is (12 * 9  * 2048) 
        12 is just way of batching , then MaxPool the 12 players 
        to save at the end ,  (1 * 9 * 2048) , squeeze(0) will be (9 * 2048) in the final
        (file.npy).shape will be (9 * 2048) #representing the whole scene now , 
        after max pool we moved from person level to scene level . """
            

    """Load the trained model"""
    # the second class
    model = PlayerLSTM()
    model.load_state_dict(torch.load('lstm1_b7_full_model_parameters.pth'))
    model.eval()

    # Base output directory where the features will be saved
    base_output_dir = '/kaggle/working/features/crop-level/Lstm_B7_9_2048'

    train_file_paths = train_dataset
    val_file_paths = val_dataset
    test_file_paths = test_dataset


    """Process all files and save their features"""
    process_and_save_files(train_file_paths, base_output_dir, model)
    process_and_save_files(val_file_paths, base_output_dir, model)
    process_and_save_files(test_file_paths, base_output_dir, model)
