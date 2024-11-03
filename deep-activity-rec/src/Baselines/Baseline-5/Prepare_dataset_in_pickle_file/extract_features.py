
"""
    After running this file you will have feature representation of the crops 
    saved in .npy files
    like that  : 
        file = np.load('/kaggle/working/features/image-level/resnet/0/13286.npy')
        print(file.shape)
            (9, 12, 2048)  9 frames - 12 players or crops in each frame represented by
            feature vector of length 2048 . (Resnet50 final fc layer which i trained in B3)


            these features files will be used in {B5} , {B6} , {B7 full model} , {B8 full model v2}
"""



"""

    future note : in this file which is for "feature extraction"
    i could use the annotation pickle file which i saved in "volleyball_annot_loader.py" ,
    but i called everything again.

"""



import os
import numpy as np
import cv2
import torch
from PIL import __version__ as PILLOW_VERSION
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from volleyball_annot_loader import load_tracking_annot


dataset_root = '/home/moustafa/0hdd/research/sfu/volleyball-datasets'


def check():
    print('torch: version', torch.__version__)
    # Check for availability of CUDA (GPU)
    if torch.cuda.is_available():
        print("CUDA is available.")
        # Get the number of GPU devices
        num_devices = torch.cuda.device_count()
        print(f"Number of GPU devices: {num_devices}")

        # Print details for each CUDA device
        for i in range(num_devices):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU.")

    # Get the name of the current device
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else "CPU"
    print(f"Current device: {current_device}")




def prepare_model(image_level = False):
    if image_level:
        # image has a lot of space around objects. Let's crop around
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        # already croped box. Don't crop more
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Check if a GPU is available and if not, use a CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     # resnet50 alexnet
    #     model = models.resnet50(pretrained=True)  # You can also use 'mobilenet_v3_large'
    
    
    #     model.eval()  # Set the model to evaluation mode if you are using it for inference


    #the old model
    # path_to_your_finetuned_model = r'/kaggle/input/unique-model-classifier-472596a/model_with_7_epochs.pth'

    
    path_to_your_finetuned_model = r'/kaggle/input/b3-first-trained-model-v1-4732598a/b3_trained_model_v1.pth'
    
    # # Load the entire model (adjust the path to your model)
    # model = torch.load(f'{path_to_your_finetuned_model}' ,  weights_only=False)

     # Step 1: Load the fine-tuned ResNet50 model
    model = models.resnet50(num_classes=9)  # ResNet50 with 9 output classes
    model.load_state_dict(torch.load(f"{path_to_your_finetuned_model}"))
    # model.to(device)
    # model.eval()  # Set model to evaluation mode Set the model to evaluation mode



    # Modify the model to remove the final classification layer
    class FeatureExtractor(nn.Module):
        def __init__(self, original_model):
            super(FeatureExtractor, self).__init__()
            # Copy all layers except the last fully connected layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)  # Flatten the output to a 2048-length vector
            return x

    # Create the feature extractor
    feature_extractor = FeatureExtractor(model)
      
    #     # Remove the classification head (i.e., the fully connected layers)
    #     model = nn.Sequential(*(list(model.children())[:-1]))
    
    #     # Send the model to the device (CPU or GPU)
    #     model.to(device)
        
    feature_extractor.to(device)

    #     # Set the model to evaluation mode
    #     model.eval()
    feature_extractor.eval()

    return feature_extractor, preprocess






def extract_features(clip_dir_path,
                     annot_file,
                     output_file,
                     model, preprocess,
                     image_level=False):
   
    # Load annotations (frame_boxes_dct holds 9 frames, each frame has info about 12 players)
    frame_boxes_dct = load_tracking_annot(annot_file)

    # List to accumulate all frames' features
    all_frames_features = []

    with torch.no_grad():
        # Iterate over the 9 frames
        for frame_id, boxes_info in frame_boxes_dct.items():
            try:
                # Get the image path
                img_path = os.path.join(clip_dir_path, f'{frame_id}.jpg')
                image = Image.open(img_path).convert('RGB')
                
                if image_level:
                    # Image-level feature extraction
                    preprocessed_image = preprocess(image).unsqueeze(0)
                    dnn_repr = model(preprocessed_image).to(device)
                    dnn_repr = dnn_repr.view(1, -1)
                else:
                    # Player-level feature extraction (12 players per frame)
                    preprocessed_images = []  # To store preprocessed crops of the 12 players
                    for box_info in boxes_info:
                        # Extract each player's crop and preprocess
                        x1, y1, x2, y2 = box_info.box
                        cropped_image = image.crop((x1, y1, x2, y2))
                        preprocessed_images.append(preprocess(cropped_image).unsqueeze(0))
                    
                    # Batch process the 12 cropped player images
                    preprocessed_images = torch.cat(preprocessed_images).to(device)
                    dnn_repr = model(preprocessed_images).to(device)  # Extract features
                    dnn_repr = dnn_repr.view(len(preprocessed_images), -1).to('cpu')  # 12 x 2048

                # Append features of the current frame to the list
                all_frames_features.append(dnn_repr.numpy())

                print(f'\t {frame_id} features processed.')

            except Exception as e:
                print(f"Error processing frame {frame_id}: {e}")

    # Save all frame features at once after processing
    np.save(output_file, np.array(all_frames_features))  # Shape: (9 frames, 12 players, 2048 features)
    print(f'All features saved to {output_file}')





def temp():
    categories_dct = {
        'l-pass': 0,
        'r-pass': 1,
        'l-spike': 2,
        'r_spike': 3,
        'l_set': 4,
        'r_set': 5,
        'l_winpoint': 6,
        'r_winpoint': 7
    }

    train_ids = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31",
                 "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]


    val_ids = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]






if __name__ == '__main__':
    check()
    
    # image_level: extract features for the whole image or just a crop
    image_level = False
    model, preprocess = prepare_model(image_level)
    
    #dataset_root = '/teamspace/studios/this_studio'
    videos_root = f'/kaggle/input/volleyball/volleyball_/videos'
    annot_root = f'/kaggle/input/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation'
    
    # to save in 
    output_root = f'/kaggle/working/features/image-level/resnet'

    videos_dirs = os.listdir(videos_root)
    videos_dirs.sort()

    # Iterate on each video and for each video iterate on each clip
    for idx, video_dir in enumerate(videos_dirs):
    
        video_dir_path = os.path.join(videos_root, video_dir)

        if not os.path.isdir(video_dir_path):
            continue

        print(f'{idx}/{len(videos_dirs)} - Processing Dir {video_dir_path}')

        clips_dir = os.listdir(video_dir_path)
        clips_dir.sort()

        # looping on 158 clip or training example
        for clip_dir in clips_dir:
            clip_dir_path = os.path.join(video_dir_path, clip_dir)

            if not os.path.isdir(clip_dir_path):
                continue

            # print(f'\t{clip_dir_path}')


            # which we already saved in a pickle file
            annot_file = os.path.join(annot_root, video_dir, clip_dir, f'{clip_dir}.txt')
            output_file = os.path.join(output_root, video_dir)


            if not os.path.exists(output_file):
                os.makedirs(output_file)

            output_file = os.path.join(output_file, f'{clip_dir}.npy')
            extract_features(clip_dir_path, annot_file, output_file, model, preprocess, image_level = image_level)