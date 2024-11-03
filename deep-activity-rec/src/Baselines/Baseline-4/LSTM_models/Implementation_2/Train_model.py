"""
-- this is will take much more time than Implementation 1
-- In order to avoid Cuda out of memory error (you should reduce your batch_size = 8 for example)


    # OutOfMemoryError: CUDA out of memory.
    # Tried to allocate 50.00 MiB. GPU 0 has a total capacity of 14.74 GiB
    # of which 2.12 MiB is free. Process 44019 has 14.74 GiB memory in use.
    # Of the allocated memory 14.31 GiB is allocated by PyTorch,
    # and 320.39 MiB is reserved by PyTorch but unallocated. 
    # If reserved but unallocated memory is large try setting 
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid 
    # fragmentation. See documentation for Memory Management
"""

from Custom_dataset import train_loader_reslstm
from Custom_dataset import val_loader_reslstm
from Custom_dataset import test_loader_reslstm



import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import models, transforms
from PIL import Image

# Define the device to run the model on (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)


class ResNetLSTM(nn.Module):
    def __init__(self, pretrained_model, num_classes=8, hidden_size=512, num_layers=1):
        super(ResNetLSTM, self).__init__()

        # Use the loaded ResNet model passed during initialization
        self.feature_extractor = nn.Sequential(*list(pretrained_model.children())[:-1])

        # LSTM layer
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()  # x shape: (batch_size, sequence_length, channels, height, width)
        
        # Extract features from each image using ResNet
        features = []
        for i in range(seq_len):
            img = x[:, i, :, :, :]  # Get the i-th image in the sequence
            feat = self.feature_extractor(img)  # Extract ResNet features
            feat = feat.view(batch_size, -1)  # Flatten the feature map
            features.append(feat)
        
        # Stack the features to form a sequence
        features = torch.stack(features, dim=1)  # Shape: (batch_size, sequence_length, 2048)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(features)  # lstm_out shape: (batch_size, sequence_length, hidden_size)
        
        # Take the output from the last LSTM time step
        lstm_last_out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Pass through the fully connected layer to get class scores
        out = self.fc(lstm_last_out)  # Shape: (batch_size, num_classes)
        
        return out



model = models.resnet50(num_classes=8)  # ResNet50 with 8 output classes
model.load_state_dict(torch.load("/kaggle/input/b1-trained-model-v1-4732598a/b1_trained_model_v1.pth"))
# Create an instance of the ResNetLSTM model
model = ResNetLSTM(model).to(device)


# (hidden size and num_layers) must be modified to get a good training
# Hyperparameters
# input_size = 2048      # Size of each feature vector
# hidden_size = 512      # Number of features in hidden state
# num_layers = 2         # Number of stacked LSTM layers
# num_classes = 8        # Number of output classes (assuming 8 classes)
# sequence_length = 9    # Length of the input sequence

# Create model
# model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
# model.to(device)



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)




def train_lstm_classifier(train_loader, val_loader, test_loader, num_epochs):
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        print('')
        print(f'epoch : {epoch} ')
        print('')
        for batch_idx, (data, target) in enumerate(train_loader):
            # data: shape (batch_size, 1, sequence_length, input_size)
            # target: shape (batch_size)
            
            # data = data.squeeze(2)  # Remove the extra dimension
            data, target = data.to(device), target.to(device)

            # Forward pass
            outputs = model(data)  # outputs: (batch_size, num_classes)
            
            # Calculate loss
            loss = criterion(outputs, target)
            train_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)  # predicted: (batch_size)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                # data = data.squeeze(2)  # Remove the extra dimension
                data, target = data.to(device), target.to(device)
                
                outputs = model(data)  # outputs: (batch_size, num_classes)
                
                # Calculate loss
                loss = criterion(outputs, target)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)  # predicted: (batch_size)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct_val / total_val

        # Print training and validation results for the current epoch
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
    
    # Testing phase
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            #   data = data.squeeze(2)  # Remove the extra dimension
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)  # outputs: (batch_size, num_classes)
            
            # Calculate loss
            loss = criterion(outputs, target)
            test_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)  # predicted: (batch_size)
            total_test += target.size(0)
            correct_test += (predicted == target).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')





b4_trained_lstm_imp2_v1 = train_lstm_classifier(train_loader_reslstm,
                                                val_loader_reslstm,
                                                test_loader_reslstm,
                                                num_epochs = 10)