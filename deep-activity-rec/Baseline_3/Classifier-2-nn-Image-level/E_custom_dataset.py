import torch
from torch.utils.data import Dataset, DataLoader
from C_prepare_data_for_feature_extraction import train_annotation , val_annotation  ,test_annotation
from D_feature_extractor import train_max_pooled_features_list , val_max_pooled_features_list , test_max_pooled_features_list

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Assuming you already have lists: 
# train_features, val_features, test_features
# train_labels, val_labels, test_labels (for 8 classes)

# Create datasets
train_dataset = CustomDataset(train_max_pooled_features_list, train_annotation)
val_dataset = CustomDataset(val_max_pooled_features_list, val_annotation)
test_dataset = CustomDataset(test_max_pooled_features_list, test_annotation)



# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)