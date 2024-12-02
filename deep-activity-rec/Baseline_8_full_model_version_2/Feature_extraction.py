"""Baseline B8 : this is the same as baseline-7 (full model v1 ) 
   but before maxpooling the players (9 * 12 * 2048) , i will seperate the 
   right team of the left team , hopefully to improve the accuarcy .

   ● The scene representation is not pool of all players
   ● X = Pool team 1  6 players
   ● Y = Pool team 2  6 players
   ● Let scene representation concatenation of X and Y """

import os
import numpy as np
import cv2
import torch
from PIL import __version__ as PILLOW_VERSION
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
# from volleyball_annot_loader import load_tracking_annot




# load the data to separate the team into team 1 and team 2 , to pool each team separately
""" i will use the pickle dataset i saved in the below python file
    deep-activity-rec\src\Baselines\Baseline_5\Prepare_dataset_in_pickle_file\volleyball_annot_loader.py
   
   # 'i loaded it using the below code on KAGGLE cloud'
   import kagglehub
   root = kagglehub.dataset_download('mohammedtharwat339/volleybal-annotations-for-all-the-dataset-77a',
   path='annot_all.pkl')
   
   # don't forget the class in boxinf.py  which is in src\Baselines\Baseline_5\Prepare_dataset_in_pickle_file
   # to call 
   
   # this is the dataset

   import pickle
   with open(root, 'rb') as file:
      videos_annot = pickle.load(file)

   """




# i saved it in kaggle disk (kaggle datasets)
videos_annot = 'path_to_your_dataset'
# Check if a GPU is available and if not, use a CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""Sort the players before separation
   Iterate through the dataset
   here we will overwrite on the dataset "videos_annot" """
def sort_team():
   def get_box_coordinates(crop) : 
      x, y, _, _ = crop.box
      return (x, y)
    
   for video_id in videos_annot:
      for frame_id in videos_annot[video_id]:
         # Get the list of BoxInfo objects in the current frame
         boxes = videos_annot[video_id][frame_id]['frame_boxes_dct']
   
         # Sort the boxes based on the (x, y) coordinates
         # 13281 ---> [12 players objects]
         for frame_number, crops in boxes.items():
               boxes[frame_number] = sorted(crops, key=get_box_coordinates)
               # boxes[frame_number] = sorted(crops, key=lambda crop: (crop.box[0], crop.box[1]))


     

"""Feature Extraction using Pretrained classifier of B-3 
   this done for the crops after sorting , to use it in the separation version"""

def extract_features_separation_ver() :
    from Baseline_5.Prepare_dataset_in_pickle_file.extract_features import \
                                 check , prepare_model , extract_features
    

    check()
    
   # image_level: extract features for the whole image or just a crop
    image_level = False

    """it's supposed to use the trained model which in prepare_model """
    model, preprocess = prepare_model(image_level)

   # this is which i supposed to use last time to extract features
    """ path_to_your_finetuned_model = kagglehub.dataset_download('mohamedtharwat123/unique-model-classifier-472596a'
                                                          , path='model_with_7_epochs.pth')
        model = torch.load(f'{path_to_your_finetuned_model}')"""
    

    videos_root = '/kaggle/input/volleyball/volleyball_/videos' # after separation
    output_root = '/kaggle/working/features/crop-level/resnet/separation_version' # to save in 
    
    
    for video_id in videos_annot:  
        video_dir_path = os.path.join(videos_root, video_id)
        if not os.path.isdir(video_dir_path):
            continue

        for frame_id in videos_annot[video_id]:
            # Get the list of BoxInfo objects in the current frame
            boxes = videos_annot[video_id][frame_id]['frame_boxes_dct']
            
            
            clip_dir_path = os.path.join(video_dir_path, frame_id)

            if not os.path.isdir(clip_dir_path):
                continue
 
            # Sort the boxes based on the (x, y) coordinates
            # 13281 ---> [12 players objects]
            # for frame_number, crops in boxes.items():
            # boxes[frame_number] = sorted(crops, key=get_box_coordinates)

                
            output_file = os.path.join(output_root, video_id)
            if not os.path.exists(output_file):
                os.makedirs(output_file)

            output_file = os.path.join(output_file, f'{frame_id}.npy')

            ##all it will need 'frame_boxes_dic' which are 9 sequences
            extract_features(clip_dir_path, boxes , output_file, model, preprocess, image_level = image_level)



"""Feature Extraction using lstm-1 (trained on crops in B7)"""

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
    



""" To achieve your requirement, we can split the players into two groups: 
    the first group (e.g., first 6 players) and the second group (remaining players).
    We will perform max-pooling for each group separately and concatenate the results
    to form the final representation of shape (9, 4096). """
def extract_sequence_representation(file_path, model):
    """
    Extract a sequence-level representation from a file, considering variable player count.
    This uses the entire sequence of hidden states for each player (not just the last one).
    Players are divided into two groups for separate max-pooling, followed by concatenation.
    
    Args:
        file_path (str): Path to the .npy file.
        model (torch.nn.Module): The trained LSTM model.
    
    Returns:
        torch.Tensor: A sequence-level representation of shape (9, 4096), after max-pooling 
                      and concatenation of the two groups' representations.
    """


    # Load the .npy file, which contains features in shape (9, num_players, 2048)
    features = np.load(file_path)  # Shape: (9, num_players, 2048)
    num_players = features.shape[1]

    # Store the representations for all players ) (12 players)
    player_representations = []

    # Process each player's sequence independently
    for player_idx in range(num_players):
        # Extract the player's sequence: Shape (9, 2048)
        player_seq = torch.tensor(features[:, player_idx, :]).unsqueeze(0)  # Shape: (1, 9, 2048)

        with torch.no_grad():
            # Pass through the trained LSTM model and get output for all 9 time steps
            player_output = model(player_seq)  # Shape: (1, 9, 2048)
            player_representations.append(player_output.squeeze(0))  # Shape: (9, 2048)

    # Stack player representations into a single tensor: Shape (num_players, 9, 2048)
    player_representations = torch.stack(player_representations)  # Shape: (num_players, 9, 2048)

    # Split players into two groups
    split_index = min(6, num_players)  # Ensure we don't exceed the number of players
    group1 = player_representations[:split_index]  # First group (up to 6 players)
    group2 = player_representations[split_index:]  # Remaining players

    # Apply max-pooling across the player representations for each group
    group1_max_pooled = torch.max(group1, dim=0)[0]  # Shape: (9, 2048)
    
    """If there are fewer than 6 players in group2, 
       it falls back to a tensor of zeros with the same shape as group1_max_pooled."""
    group2_max_pooled = torch.max(group2, dim=0)[0] if group2.size(0) > 0 else torch.zeros_like(group1_max_pooled)

    # Concatenate the max-pooled results: Shape (9, 4096)
    final_representation = torch.cat([group1_max_pooled, group2_max_pooled], dim=1)

    return final_representation




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
        rel_path = os.path.relpath(file_path, 'separated_resnet')
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

   from Baseline_7_full_model_version_1.Full_model_LSTM1 import get_trainValTest_dataset

   """Do sorting over the team , overwriting the dataset """
   sort_team()

   """Extracting features using the separated data"""
   extract_features_separation_ver()

   """features - sorted version"""
   root = '/kaggle/input/separatedteam-resnet-feature-vectors-9-12-472598a/kaggle/working/features/crop-level/resnet/separation_version'
   train_dataset , val_dataset , test_dataset = get_trainValTest_dataset(root=root)

   """Load the trained model"""
   # the second class
   model = PlayerLSTM()
   model.load_state_dict(torch.load('lstm1_b7_full_model_parameters.pth'))
   model.eval()

   # Base output directory where the features will be saved
   base_output_dir = '/kaggle/working/features/crop-level/Lstm_B8_9_2048'

   train_file_paths = train_dataset
   val_file_paths = val_dataset
   test_file_paths = test_dataset

   """Process all files and save their features"""
   """it took me around ~ 4 hours to extract features """
   process_and_save_files(train_file_paths, base_output_dir, model)
   process_and_save_files(val_file_paths, base_output_dir, model)
   process_and_save_files(test_file_paths, base_output_dir, model)