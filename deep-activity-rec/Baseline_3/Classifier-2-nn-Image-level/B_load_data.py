import os
import pickle

"""load the dataset from the pickle files ."""

root = r'/kaggle/input/b3-dataset-12-players-crops-annot-4732598a'



def load_pickle_files(root) :
    # Load the list from the pickle file

    # Load the crops
    with open(os.path.join(root , 'B3_train_data_12_players.pkl'), 'rb') as f:
        train_data = pickle.load(f)
        
    with open(os.path.join(root , 'B3_val_data_12_players.pkl'), 'rb') as f:
        validation_data = pickle.load(f)

    with open(os.path.join(root , 'B3_test_data_12_players.pkl'), 'rb') as f:
        test_data = pickle.load(f)


    return  train_data , validation_data , test_data



train_data , validation_data , test_data = load_pickle_files(root)