import torch
from torch.utils.data import Dataset
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

 
from A_prepare_data_splits_and_annotations  import train_dataset , val_dataset , test_dataset 
from A_prepare_data_splits_and_annotations  import train_dict_person_level_labels , val_dict_person_level_labels , test_dict_person_level_labels



# create custom dataset
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




# Custom train val test datasets

train_data = PlayerSequenceDataset(train_dataset, 
                                   train_dict_person_level_labels ,
                                 label_encoding)

val_data = PlayerSequenceDataset(val_dataset, 
                                 val_dict_person_level_labels , 
                                 label_encoding)

test_data = PlayerSequenceDataset(test_dataset,
                                   test_dict_person_level_labels , 
                                   label_encoding)




