
from A_Get_data_for_A_step import validation_crops , validation_annotations
from A_Get_data_for_A_step import test_crops , test_annotations
from B_upsample_undersample import train_images , train_labels




import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms




# Create a mapping from class labels to indices
class_to_idx = {cls: idx for idx, cls in enumerate(sorted(set(validation_annotations)))}
print(class_to_idx)



class VolleyballDataset(Dataset):
    def __init__(self, crops , annotations , transform=None):
        """
        Args:
            data_dict (dict): Dictionary with image paths as keys and labels as values.
                              Keys should be paths to the directories containing the frames, 
                              and the filename should be the specific frame name.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.crops = crops
        self.annotations = annotations
        self.transform = transform # Ensure this is a transform pipeline, not a dictionary
        
        # Append '.jpg' to the full image path, including the frame filename
        # os.path.join(path, os.path.basename(path)) + '.jpg':
        # path is the full path to the directory containing the frames.
        # os.path.basename(path) extracts the last part of the path, which is 
        # the frame directory name (e.g., 10535).
        # os.path.join(path, os.path.basename(path)) + '.jpg' constructs the 
        # full path to the specific image file by joining the directory path with 
        # the frame directory name and appending .jpg.
        
        
        self.images = crops
        self.labels = annotations
        # Create a mapping from class labels to indices
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(set(self.labels)))}

    
    def __len__(self):
        return len(self.images)

    
    def __getitem__(self, idx):
        image = self.images[idx]
        annotation = self.annotations[idx]

        if self.transform:
              image = self.transform(image)
            
        # Convert label to a numeric value
        label_idx = self.class_to_idx[annotation]

        # Here, annotation processing can be done if needed
        return image, label_idx
    

data_transforms = {
    'train': transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}





custom_train_dataset = VolleyballDataset(train_images ,
                                          train_labels,
                                            transform= None )

custom_val_dataset = VolleyballDataset(validation_crops , 
                                       validation_annotations,
                                         transform=data_transforms['val'])

custom_test_dataset = VolleyballDataset(test_crops , 
                                        test_annotations , 
                                        transform=data_transforms['test'])


batch_size = 32

train_loader = DataLoader(custom_train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

val_loader = DataLoader(custom_val_dataset, 
                        batch_size=batch_size, 
                        shuffle=False)

test_loader = DataLoader(custom_test_dataset,
                          batch_size=batch_size, 
                          shuffle=False)