import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


from C_Feature_extractor import train_features , val_features , test_features


class CustomFeatureDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (list of tuples): Each tuple is (first, second)
                                   where 'first' is a list of 9 tensors, 
                                   each of size [2048], and 'second' is the label/class.
        """
        
        self.data = data
        # Extract labels and encode them
        labels = [label for _ , label in data]
        
        # this is a magic
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        first = self.data[idx][0]  # list of 9 tensors
        second = self.encoded_labels[idx]  # encoded label
        
        # Convert the list of tensors into a single tensor
        first = torch.stack(first)
        
        return first, torch.tensor(second,
                                   dtype= torch.long)

    def get_classes(self):
        return self.label_encoder.classes_

    
    
def create_dataloader(data,
                      batch_size=32,
                      shuffle=True,
                      num_workers=0):
    
    dataset = CustomFeatureDataset(data)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader




if __name__ == "__main__" :

    # train_features is a list of tuples ([9 tensors] , label) 
    train_loader = create_dataloader(train_features)
    val_loader = create_dataloader(val_features)
    test_loader = create_dataloader(test_features)