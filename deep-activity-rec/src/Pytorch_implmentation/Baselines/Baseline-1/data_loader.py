import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class VolleyballDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        
        """
        Args:
            data_dict (dict): Dictionary with image paths as keys and labels as values.
                              Keys should be paths to the directories containing the frames, 
                              and the filename should be the specific frame name.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.data_dict = data_dict
        self.transform = transform
        
        # Append '.jpg' to the full image path, including the frame filename
        # os.path.join(path, os.path.basename(path)) + '.jpg':
        # path is the full path to the directory containing the frames.
        # os.path.basename(path) extracts the last part of the path, which is 
        # the frame directory name (e.g., 10535).
        # os.path.join(path, os.path.basename(path)) + '.jpg' constructs the 
        # full path to the specific image file by joining the directory path with 
        # the frame directory name and appending .jpg.
        
        
        self.image_paths = [os.path.join(path, os.path.basename(path)) + '.jpg' for path in data_dict.keys()]
        self.labels = list(data_dict.values())
        
        # Create a mapping from class labels to indices
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(set(self.labels)))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load the image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Convert label to a numeric value
        label_idx = self.class_to_idx[label]
        
        return image, label_idx

import torchvision.transforms as transforms


def Transform() : 
        transform = {'train' : transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),      # Randomly flip images horizontally with a probability of 0.5
        transforms.RandomRotation(degrees=15),       # Randomly rotate images by +/- 15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change the brightness, contrast, saturation, and hue
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Randomly crop and resize to 224x224
        transforms.ToTensor(),                       # Convert PIL images to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ]) ,
                'val' : transforms.Compose([
                    transforms.Resize((224 , 224)),
        transforms.ToTensor(),                       # Convert PIL images to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])
                ,
                'test' : transforms.Compose([
                    transforms.Resize((224 , 224)),
        transforms.ToTensor(),                       # Convert PIL images to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])
    }


transform = Transform()


# Create an instance of the dataset
custom_train_data = VolleyballDataset(train_dataset, transform=transform['train'])
# print(volleyball_dataset.__len__())
# Create a DataLoader
train_loader = DataLoader(custom_train_data, batch_size=32, shuffle= True ) #, num_workers=4)


custom_val_data = VolleyballDataset(val_dataset, transform=transform['val'])
# Create a DataLoader
val_loader = DataLoader(val_data, batch_size=32, shuffle= True)


custom_test_data = VolleyballDataset(test_dataset, transform=transform['test'])
# Create a DataLoader
test_loader = DataLoader(test_data, batch_size=32, shuffle= True)


# Do some check
def check(train_loader) :
    # Example of iterating through the data loader
    for images, labels in train_loader:
        # this prints shapes of the batch 
        print(images.size())  # (batch_size, 3, 224, 224)
        print(labels)         # Labels as tensor indices
        break

check(train_loader)

"""
torch.Size([32, 3, 224, 224])
tensor([0, 6, 4, 6, 0, 2, 0, 4, 6, 6, 2, 0, 5, 2, 4, 4, 1, 7, 3, 1, 0, 5, 2, 5,
        6, 0, 2, 4, 5, 0, 0, 6])
"""



