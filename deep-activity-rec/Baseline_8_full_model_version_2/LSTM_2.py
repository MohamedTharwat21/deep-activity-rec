"""LSTM-2 on the scene level"""

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

"""LSTM-2 Model"""
from Baseline_7_full_model_version_1.Full_model_LSTM2 import PlayerSequenceDataset , \
      LSTMClassifier ,\
      train_model , \
      evaluate_model , test_model , get_train_val_test_scene_lvl
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""these are my configs for B8 second LSTM level"""
cfg = { 
      
  "input_size"  :  4096,
  "hidden_size" :  512, 
  "num_layers"  :  2, 
  "num_classes" :  8,
  "num_epochs"  : 10, 
  "learning_rate": 0.001 ,
  "batch_size" :  32 ,
  "shuffle"   :  True
}



if __name__ == "__main__" :
    """"train dataset"""
    train_dataset , val_dataset  , test_dataset = get_train_val_test_scene_lvl()

    """scene_level_annotations"""
    train_dict_scene_level_labels ,\
         val_dict_scene_level_labels ,test_dict_scene_level_labels = get_annotations()

    """Custom Dataset"""
    train_data = PlayerSequenceDataset(train_dataset, train_dict_scene_level_labels)
    val_data = PlayerSequenceDataset(val_dataset, val_dict_scene_level_labels)
    test_data = PlayerSequenceDataset(test_dataset, test_dict_scene_level_labels)

    """constructing the dataloader"""
    train_loader = DataLoader(train_data , batch_size = 32 , shuffle = True)
    val_loader = DataLoader(val_data , batch_size = 32 , shuffle = False)
    test_loader = DataLoader(test_data , batch_size = 32 , shuffle = False)

    """Initialize model"""
    model = LSTMClassifier(cfg["input_size"], cfg["hidden_size"], cfg["num_layers"]
                           , cfg["num_classes"]).to(device)

    """Train the model"""
    train_model(model, train_loader, val_loader, cfg['num_epochs'], cfg["learning_rate"])

    """Test the model"""
    test_model(model, test_loader)


    """ Finished Training!
    Test Accuracy: 82.42% """