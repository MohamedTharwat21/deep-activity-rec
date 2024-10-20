import torch
import torch.nn as nn
import torchvision.models as models



# Step 1: Load the fine-tuned ResNet50 model
model = models.resnet50(num_classes=9)  # ResNet50 with 9 output classes
model.load_state_dict(torch.load("/kaggle/input/b3-first-trained-model-v1-4732598a/b3_trained_model_v1.pth"))
model.eval()  # Set model to evaluation mode



# Remove the final classification layer to get features before FC
model = nn.Sequential(*list(model.children())[:-1])  # Remove the final FC layer



# Assuming train_crops is a list of tensors, each tensor of shape [10, 3, 224, 224]
train_max_pooled_features_list = []  # To store the max pooled features for each crop



# Step 2: Loop over the list of tensors (train_crops)
for crop_tensor in train_crops:
    with torch.no_grad():  # Disable gradient calculation for inference
        features = model(crop_tensor)  # Output shape: [10, 2048, 1, 1]



    # Step 3: Max pool across the player dimension
    num_players = crop_tensor.shape[0]
    features = features.view(num_players, 2048)  # Reshape to [10, 2048]
    max_pooled_features = torch.max(features, dim=0)[0]  # Max pool across dimension 0 (10 players)

    # Step 4: Save the max pooled feature (1 * 2048)
    train_max_pooled_features_list.append(max_pooled_features)  # Save each max pooled feature



# Now max_pooled_features_list contains the max pooled features for each tensor in train_crops



# Extract features for val and test 
# Assuming train_crops is a list of tensors, each tensor of shape [10, 3, 224, 224]

val_max_pooled_features_list = []  # To store the max pooled features for each crop

# Step 2: Loop over the list of tensors (train_crops)
for crop_tensor in val_crops:
    with torch.no_grad():  # Disable gradient calculation for inference
        features = model(crop_tensor)  # Output shape: [10, 2048, 1, 1]

    # Step 3: Max pool across the player dimension
    num_players = crop_tensor.shape[0]
    features = features.view(num_players, 2048)  # Reshape to [10, 2048]
    max_pooled_features = torch.max(features, dim=0)[0]  # Max pool across dimension 0 (10 players)

    # Step 4: Save the max pooled feature (1 * 2048)
    val_max_pooled_features_list.append(max_pooled_features)  # Save each max pooled feature

# Now max_pooled_features_list contains the max pooled features for each tensor in val_crops




# Extract features for val and test 
# Assuming train_crops is a list of tensors, each tensor of shape [10, 3, 224, 224]

test_max_pooled_features_list = []  # To store the max pooled features for each crop

# Step 2: Loop over the list of tensors (train_crops)
for crop_tensor in test_crops:
    with torch.no_grad():  # Disable gradient calculation for inference
        features = model(crop_tensor)  # Output shape: [10, 2048, 1, 1]

    # Step 3: Max pool across the player dimension
    num_players = crop_tensor.shape[0]
    features = features.view(num_players, 2048)  # Reshape to [10, 2048]
    max_pooled_features = torch.max(features, dim=0)[0]  # Max pool across dimension 0 (10 players)

    # Step 4: Save the max pooled feature (1 * 2048)
    test_max_pooled_features_list.append(max_pooled_features)  # Save each max pooled feature

# Now max_pooled_features_list contains the max pooled features for each tensor in val_crops