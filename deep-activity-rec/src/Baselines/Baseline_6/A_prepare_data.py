""" Here will MaxPool all the players on the frame before lstm 
    B6 maxpool then lstm (scene level) 9 * 12 * 2048 --> 9 * 2048
    B5 lstm (person level) then maxpool 
    WE are in the scene level , use the scene labels """

import os
import numpy as np
from Baseline_5.LSTM_on_feature_vectors.A_prepare_data_splits_and_annotations \
             import train_dataset , val_dataset , test_dataset 


def process_and_save_files(file_paths, base_output_dir):
    
    for file_path in file_paths:
        
        # Extract directory structure from file_path (e.g., '1/43825.npy')
        # rel_path = os.path.relpath(file_path, '/kaggle/input/resnet-feature-vectors-9-12-472598a/kaggle/working/features/image-level/resnet')
        rel_path = os.path.relpath(file_path, '/kaggle/input/b5-v1-resnet-feature-vectors-9-12-472598a/kaggle/working/features/image-level/resnet')

        output_path = os.path.join(base_output_dir, rel_path)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        file = np.load(file_path)
        # Extract the max-pooled representation from the input file
        max_pooled_representation = np.max(file, axis=1)  # Shape: (9, 2048)
        
    
        # if you used pytorch  
        # player_repre = torch.tensor(file)
        # pooled_tensor = torch.max(player_repre, dim=1).values # Shape: (9, 2048)
        # np.save(output_path, max_pooled_representation.numpy())
        
        
        # Save the max-pooled representation as a NumPy array
        np.save(output_path, max_pooled_representation)
        print(f"Saved max-pooled representation to {output_path}")
        


train_ids = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31",
                "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]

val_ids = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]
test_ids = ["4", "5", "9", "11", "14", "20", "21", "25", "29", "34", "35", "37", "43", "44", "45", "47"]



"""Load .npy files that have Max Pooled feature representations .
splitting the dataset"""

def get_train_val_test() : 


    root = '/kaggle/working/features/crop-level/MaxPooled_players'
    train_dataset = []
    val_dataset = []
    test_dataset = []

    for vid_id in train_ids : 
        path = os.path.join(root , str(vid_id))
        for seq in os.listdir(path) :
            train_dataset.append(os.path.join(root , str(vid_id) , seq))
    for vid_id in val_ids : 
        path = os.path.join(root , str(vid_id))
        for seq in os.listdir(path) :
            val_dataset.append(os.path.join(root , str(vid_id) , seq))
    for vid_id in test_ids : 
        path = os.path.join(root , str(vid_id))
        for seq in os.listdir(path) :
            test_dataset.append(os.path.join(root , str(vid_id) , seq))


    return train_dataset , val_dataset , test_dataset



def get_annotations() :
    train_dict_scene_level_labels = {}
    val_dict_scene_level_labels = {}
    test_dict_scene_level_labels = {}

    videos_annot = 'write_your_path_here'
    # loop on 154 instances
    for vid_id in train_ids :
        dict_helper = {}
        for clip in videos_annot[vid_id].keys():
            label_of_the_clip = videos_annot[vid_id][str(clip)]['category']
            dict_helper[clip] = label_of_the_clip
            train_dict_scene_level_labels[vid_id] =  dict_helper 
            
    # loop on 154 instances
    for vid_id in val_ids :
        dict_helper = {}
        for clip in videos_annot[vid_id].keys():
            label_of_the_clip = videos_annot[vid_id][str(clip)]['category']
            dict_helper[clip] = label_of_the_clip
            val_dict_scene_level_labels[vid_id] =  dict_helper 
            
    # loop on 154 instances
    for vid_id in test_ids :
        dict_helper = {}
        for clip in videos_annot[vid_id].keys():
            label_of_the_clip = videos_annot[vid_id][str(clip)]['category']
            dict_helper[clip] = label_of_the_clip
            test_dict_scene_level_labels[vid_id] =  dict_helper 


    return train_dict_scene_level_labels , val_dict_scene_level_labels , test_dict_scene_level_labels



if __name__ == "__main__" :

    # Base output directory where the features will be saved
    base_output_dir = '/kaggle/working/features/crop-level/MaxPooled_players'
    # train_dataset , val_dataset , test_dataset = get_train_val_test(train_ids , val_ids , test_ids )
    
    
    # Process all files and save their features
    process_and_save_files(train_dataset, base_output_dir)
    process_and_save_files(val_dataset, base_output_dir)
    process_and_save_files(test_dataset, base_output_dir)
