from B_load_data import train_data , validation_data , test_data

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms




# splitting crops from annotations
train_crops , train_annotation = zip(*train_data)
train_crops , train_annotation = list(train_crops) , list(train_annotation)

val_crops , val_annotation = zip(*validation_data)
val_crops , val_annotation = list(val_crops) , list(val_annotation)

test_crops , test_annotation =  zip(*test_data)
test_crops , test_annotation = list(test_crops) , list(test_annotation)




# pre processing the data
# here we overwrite train_crops and train_annotations


# Create a mapping from class labels to indices
class_to_idx = {cls: idx for idx, cls in enumerate(sorted(set(train_annotation)))}


# Define the transform function
def transform(image):
    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transformation(image)



# i , j = range(len(train_crops)) , range(len(train_crops)) 
for item in range(len(train_crops)):
    # Apply the transformations to each image
    train_crops[item] = [transform(image) for image in train_crops[item]]
    # Stack the images into a single tensor
    train_crops[item] = torch.stack(train_crops[item])  # Shape: (N, C, H, W)
    train_annotation[item] = class_to_idx[ train_annotation[item]]   



for item in range(len(val_crops)):
    # Apply the transformations to each image
    val_crops[item] = [transform(image) for image in val_crops[item]]
    # Stack the images into a single tensor
    val_crops[item] = torch.stack(val_crops[item])  # Shape: (N, C, H, W)
    val_annotation[item] = class_to_idx[ val_annotation[item]]   



for item in range(len(test_crops)):
    # Apply the transformations to each image
    test_crops[item] = [transform(image) for image in test_crops[item]]
    # Stack the images into a single tensor
    test_crops[item] = torch.stack(test_crops[item])  # Shape: (N, C, H, W)
    test_annotation[item] = class_to_idx[ test_annotation[item]]   







# class VolleyballDataset(Dataset):
#     def __init__(self, image_lists, labels, transform=None):
#         self.image_lists = image_lists  # A list of lists of PIL images
#         self.labels = labels  # A list of labels corresponding to each list of images
#         self.transform = transform    
#         # Create a mapping from class labels to indices
#         self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(set(self.labels)))}

        
#     def __len__(self):
#         return len(self.image_lists)

    
#     def __getitem__(self, idx):
#         images = self.image_lists[idx]
#         label = self.labels[idx]
#         # Apply transforms to each image in the list
#         # this is just the modification
#         if self.transform:
#             images = [self.transform(image) for image in images]      
#         # Optionally stack or concatenate the images if necessary
#         images = torch.stack(images)  # Stack into a single tensor of shape (N, C, H, W)
#         # Convert label to a numeric value
#         label_idx = self.class_to_idx[label]
#         return images, label_idx



# data_transforms = {
#     'train': transforms.Compose([ 
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
    
    
#     'val': transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'test': transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }