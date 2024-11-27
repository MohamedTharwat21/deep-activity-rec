import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import os

# Create an instance of the model class first (same as the one you used for training)
class PlayerLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, num_layers=2, num_classes=9):
        super(PlayerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # Add the fully connected layer

    def forward(self, x):
        # Forward pass through LSTM
        output, (hn, cn) = self.lstm(x)
        
        # hn[-1] is the final hidden state from the last LSTM layer
        lstm_out = hn[-1]  # Shape: (batch_size, hidden_size)
        
        # Pass through the fully connected layer
        # output = self.fc(lstm_out)  # Shape: (batch_size, num_classes)
        return lstm_out

    


# Load the trained model
model = PlayerLSTM()
model.load_state_dict(torch.load('B5_v1_LSTM.pth'))
model.eval()





def extract_sequence_representation(file_path, model):
    """
    Extract a single sequence-level representation from a file, considering variable player count.
    
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
    
    # Process each player sequence independently
    for player_idx in range(num_players):
        # Extract the player's sequence: Shape (9, 2048)
        player_seq = torch.tensor(features[:, player_idx, :]).unsqueeze(0)  # Shape: (1, 9, 2048)
        
        with torch.no_grad():
            # Pass through the trained LSTM model
            player_output = model(player_seq)  # LSTM hidden state output for the player (1, hidden_size)
            player_representations.append(player_output)
    
    # Stack player representations into a single tensor
    player_representations = torch.stack(player_representations)  # Shape: (num_players, hidden_size)
    
    # Apply max-pooling across the player representations
    max_pooled_representation = torch.max(player_representations, dim=0)[0]  # Shape: (hidden_size,)
    
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
        rel_path = os.path.relpath(file_path,
                                    '/kaggle/input/b5-v1-resnet-feature-vectors-9-12-472598a/kaggle/working/features/image-level/resnet')
        output_path = os.path.join(base_output_dir, rel_path)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Extract the max-pooled representation from the input file
        max_pooled_representation = extract_sequence_representation(file_path, model)
        
        # Save the max-pooled representation as a NumPy array
        np.save(output_path, max_pooled_representation.numpy())
        # print(f"Saved max-pooled representation to {output_path}")

        




if __name__ == "__main__" :
    # Base output directory where the features will be saved
    base_output_dir = '/kaggle/working/features/crop-level/Lstm'
    

    # get the train, val , test data
    from LSTM_on_feature_vectors.A_prepare_data_splits_and_annotations import train_dataset , val_dataset , test_dataset
    
    train_file_paths = train_dataset
    val_file_paths = val_dataset
    test_file_paths = test_dataset


    # Process all files and save their features
    process_and_save_files(train_file_paths, base_output_dir, model)
    process_and_save_files(val_file_paths, base_output_dir, model)
    process_and_save_files(test_file_paths, base_output_dir, model)