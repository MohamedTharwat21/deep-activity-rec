import os
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as transforms


def load_annotations(video_dir):
    """Load annotations from the annotations.txt file."""
    annotations_path = os.path.join(video_dir, 'annotations.txt')
    annotations = {}
    with open(annotations_path, 'r') as file:
        for line in file :
            #modifications
            parts = line.strip().split()
            frame_id, annotation = parts[0] , parts[1]
            annotations[int(frame_id[:-4])] = annotation
    return annotations


def map_frame_to_annotation(dataset_dir, video_ids):
    # the middle frame has the same name as the clip which contains it
    """Map frames in a video directory to their annotations."""
    frame_to_annotation_map = {}
    #  video_id = 0  
    for video_id in video_ids:
        #volleyball_      str(0)
        # concatenate the video_id to the the path
        video_dir = os.path.join(dataset_dir, str(video_id))
        # annotations with vid 0
        # this function to get the [annotations.txt] from the videp
        # dir , and we take what we want to train on
        annotations = load_annotations(video_dir)
        for frame_dir in os.listdir(video_dir):
            frame_path = os.path.join(video_dir, frame_dir)
            if os.path.isdir(frame_path):
                frame_id = int(frame_dir)  # Convert frame directory name to an integer ID
                # this is take the target frame only from the whole frames
                if frame_id in annotations:
                    frame_to_annotation_map[frame_path] = annotations[frame_id]
    return frame_to_annotation_map


def save_pickle_files(train_mapping , validation_mapping , test_mapping ) :
    # Save the dictionary as a pickle file
    with open('train_mapping.pkl', 'wb') as f:
        pickle.dump(train_mapping, f)
    with open('val_mapping.pkl', 'wb') as f:
        pickle.dump(validation_mapping, f)
    with open('test_mapping.pkl', 'wb') as f:
        pickle.dump(test_mapping, f)
    
    
def load_pickle_files() :
    """ here we load from the disk after we have saved them
        i saved them on kaggle datasets so u have to save them first 
        then load them using this fun
        Load the dictionary from the pickle file"""

    with open('train_mapping.pkl', 'rb') as f:
        train_mapping = pickle.load(f) 
    with open('val_mapping.pkl', 'rb') as f:
        val_mapping = pickle.load(f)
    with open('test_mapping.pkl', 'rb') as f:
        test_mapping = pickle.load(f)

    return train_mapping , val_mapping , test_mapping


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


def Transform() : 
    """Define the transforms for train, validation, and test sets"""
    transform = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),      # Randomly flip images horizontally with a probability of 0.5
            transforms.RandomRotation(degrees=15),       # Randomly rotate images by +/- 15 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change brightness, contrast, saturation, and hue
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Randomly crop and resize to 224x224
            transforms.ToTensor(),                       # Convert PIL images to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
        ])
    }


def get_data_loader():

    transform = Transform()
    def loader(data , shuffle = False) :
        data_loader = DataLoader(data, batch_size=32,
                                 shuffle= shuffle) 
                                 #, num_workers=4)
        return data_loader
    

    # Do some check
    def check(train_loader) :
        # Example of iterating through the data loader
        for images, labels in train_loader:
            # this prints shapes of the batch 
            print(images.size())  # (batch_size, 3, 224, 224)
            print(labels)         # Labels as tensor indices
            break
        

    train_dataset , val_dataset , test_dataset = load_pickle_files()
    custom_train_data = VolleyballDataset(train_dataset,
                                         transform=transform['train'])
    custom_val_data = VolleyballDataset(val_dataset,
                                         transform=transform['val'])
    custom_test_data = VolleyballDataset(test_dataset, 
                                         transform=transform['test'])
    
    train_loader =  loader(custom_train_data, shuffle= True)
    val_loader =  loader(custom_val_data)
    test_loader =  loader(custom_test_data)
   
    # check(train_loader)
    """
    torch.Size([32, 3, 224, 224])
    tensor([0, 6, 4, 6, 0, 2, 0, 4, 6, 6, 2, 0, 5, 2, 4, 4, 1, 7, 3, 1, 0, 5, 2, 5,
            6, 0, 2, 4, 5, 0, 0, 6])
    """
    
    return custom_train_data , custom_val_data , custom_test_data ,\
    train_loader , val_loader , test_loader 



if __name__ == "__main__" :
    """ you have to go into the videos dir to find the clips 
        this specified path as i uploaded the dataset to kaggle
        then i upload it to the input dir to read it """
     
     
    dataset_dir = '/kaggle/input/volleyball/volleyball_/videos'

    train_videos = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
    validation_videos = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
    test_videos = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
    
    # Get the mapping for training videos
    train_mapping = map_frame_to_annotation(dataset_dir, train_videos)  #2152
    validation_mapping = map_frame_to_annotation(dataset_dir, validation_videos)  
    test_mapping = map_frame_to_annotation(dataset_dir, test_videos)  
    print('end..')

    save_pickle_files(train_mapping , validation_mapping , test_mapping)
