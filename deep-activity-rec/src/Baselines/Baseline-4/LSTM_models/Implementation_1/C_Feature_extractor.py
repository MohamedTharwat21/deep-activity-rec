
from A_Get_data import train_data , validation_data , test_data

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Define the device to run the model on (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# this is the structure of the model we have trained

# Load the fine-tuned ResNet50 model
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
# Modify the final fully connected layer to match the number of classes (8 in this case)
model.fc = nn.Linear(num_features, 8)


# Load the model weights (state_dict)

"""
this was the old trained model
model_path = "/kaggle/input/baseline1-model-classifier-4732598a/baseline1.pth
model = torch.load(model_path, map_location=device)

"""

# Step 1: Load the fine-tuned ResNet50 model
model = models.resnet50(num_classes=8)  # ResNet50 with 8 output classes
model.load_state_dict(torch.load("/kaggle/input/b1-trained-model-v1-4732598a/b1_trained_model_v1.pth"))
model.to(device)
model.eval()  # Set model to evaluation mode Set the model to evaluation mode

# Separate the feature extractor and classifier
feature_extractor = nn.Sequential(*list(model.children())[:-1])

# this is a good part
classifier = model.fc

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Function to load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)


# Function to extract the 2048-dimensional feature vector
def extract_features(image):
    image = load_image(image)
    with torch.no_grad():  # No need to track gradients during inference
        features = feature_extractor(image)
        features = features.view(features.size(0), -1)  # Flatten to a 2048-dimensional vector
    return features 


# Function to predict the class using the classifier
# def predict_class(features):
#     with torch.no_grad():
#         output = classifier(features)
#         _, predicted_class = torch.max(output, 1)  Get the index of the max log-probability

#     return predicted_class.item()



# 9 frames with one label
# 9 feature vectors with one label
# make list of tuples to keep the label



def Extract_features_of_train_data(train_data) :

    train_features= [] 
    # train_labels = []
    cont = 0
    for middle_9_frames in train_data:
        cont = cont + 1
        nine_vec , labels = [] , []
        #[0] to get the frames not the labels
        for frame in middle_9_frames[0] :
            
            feature_vector = extract_features(frame)
            # predicted_class = predict_class(feature_vector)   
            nine_vec.append(feature_vector)
            # labels.append(predicted_class)
        
        if (cont % 10 == 0) : 
            print(f'{cont} Sequences have finished ..')
        
        label = middle_9_frames[1]
        train_features.append( (nine_vec , label) )
        #   train_labels.append(labels)

    return train_features





def Extract_features_of_validation_data(validation_data) :

    val_features , val_labels = [] , []
    cont = 0
    for middle_9_frames in validation_data :
        cont = cont + 1
        nine_vec , labels = [] , []
        for frame in middle_9_frames[0] :
            
            feature_vector = extract_features(frame)
            #predicted_class = predict_class(feature_vector)   
            nine_vec.append(feature_vector)
            # labels.append(predicted_class)
        
        if (cont % 10 == 0) : 
            print(f'{cont} Sequences have finished ..')
        
        label = middle_9_frames[1]
        val_features.append( (nine_vec , label) )
    
        #val_labels.append(labels)

    return val_features




def Extract_features_of_test_data(test_data) :

    test_features , test_labels = [] , []
    cont = 0
    for middle_9_frames in test_data :
        cont = cont + 1
        nine_vec , labels = [] , []
        for frame in middle_9_frames[0] :
            
            feature_vector = extract_features(frame)
            #predicted_class = predict_class(feature_vector)   
            nine_vec.append(feature_vector)
            # labels.append(predicted_class)
        
        if (cont % 10 == 0) : 
            print(f'{cont} Sequences have finished ..')
            
        label = middle_9_frames[1]
        test_features.append( (nine_vec , label) )
        #     test_features.append(nine_vec)
        #     test_labels.append(labels)

        return test_features
    

if __name__ == "__main__" :
    
    train_features = Extract_features_of_train_data(train_data) 
    val_features = Extract_features_of_validation_data(validation_data) 
    test_features = Extract_features_of_test_data(test_data) 
