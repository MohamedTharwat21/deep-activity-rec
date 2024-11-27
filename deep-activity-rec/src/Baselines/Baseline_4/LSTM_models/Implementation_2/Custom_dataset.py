"""

if You are running on a limited machine hardware (GPUs) 
set the batch size = 8 , to avoid Cuda out of memory error .

"""

from A_Get_data import train_data
from A_Get_data import validation_data 
from A_Get_data import test_data


import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder


class SequenceImageDataset(Dataset):
    def __init__(self, 
                 image_paths,
                 labels, 
                 transform=None):
        

        """
        Args:
            image_paths (list): A list where each element is a list of 9 image file paths.
            labels (list): A list of labels, one for each sequence of 9 images.
            transform (callable, optional): A function/transform to apply to each image.
        """


        self.image_paths = image_paths  # List of lists, each sublist contains 9 image paths
        self.labels = labels  # List of labels, one per sequence
        self.transform = transform
        
                # Extract labels and encode them
        labels = [label for label in self.labels]
        
        # this is a magic
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
    

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the 9 images for the given index
        image_sequence = []
        for img_path in self.image_paths[idx]:
            img = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB format
            if self.transform:
                img = self.transform(img)
            image_sequence.append(img)

        # Stack the images along a new dimension to form a tensor of shape (9, C, H, W)
        image_sequence = torch.stack(image_sequence)  # (sequence_length, channels, height, width)

        # Get the corresponding label
        label = self.encoded_labels[idx]

        return image_sequence, label
    
    def get_classes(self):
        return self.label_encoder.classes_


# Define a transformation (optional, e.g., resizing and normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])



# DataLoader to fetch batches
from torch.utils.data import DataLoader

def create_dataloader(data,
                    batch_size= 8 , 
                    shuffle=True,
                    num_workers=0):
    
    image_paths , labels = zip(*data)
    image_paths , labels = list(image_paths) , list(labels)
    # Create the dataset
    dataset = SequenceImageDataset(image_paths=image_paths,
                                  labels=labels,
                                  transform=transform)
    
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader





if __name__ == "__main" :

    # train_features is a list of tuples ([9 tensors] , label) 
    train_loader_reslstm = create_dataloader(train_data)
    val_loader_reslstm = create_dataloader(validation_data)
    test_loader_reslstm = create_dataloader(test_data)