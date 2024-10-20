import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader

# import custom datasets and DataLoaders
from Torch_Data_Loader import custom_train_data , custom_test_data , custom_val_data 
from Torch_Data_Loader import train_loader , val_loader , test_loader 

def finetuned_resnet50() :

    # Load the pre-trained ResNet-50 model
    model = resnet50(pretrained=True)
    
    # # Modify the final fully connected layer for 9 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8)
    
    # Move the model to the GPU if available
    model = model.to(device)

    return model



def train_model(model , custom_train_data , custom_val_data , train_loader , val_loader , test_loader) :

    # Example learning rate scheduler
    from torch.optim.lr_scheduler import StepLR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Weight Decay (L2 Regularization): Add weight decay to your optimizer. 
    # to prevent overfitting
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    num_epochs = 15
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
    
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    
        epoch_loss = running_loss / len(custom_train_data)
        epoch_acc = running_corrects.double() / len(custom_train_data)
    
        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
    
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
    
        epoch_loss = running_loss / len(custom_val_data)
        epoch_acc = running_corrects.double() / len(custom_val_data)
    
        print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        scheduler.step()
        
    
    print('Training Ended Successfully.')
    
    return model 

b1_trained_model_v1 = train_model(finetuned_resnet50(),
                                  custom_train_data,
                                  custom_val_data, 
                                  train_loader,
                                  val_loader,
                                    test_loader) 





def test_model(b1_trained_model_v1, custom_test_data, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = b1_trained_model_v1.to(device)
    model.eval()
    running_corrects = 0
    with torch.no_grad():  # No arguments are needed here
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / len(custom_test_data)
    print(f'Test Accuracy: {test_acc * 100:.4f} %')

# Calling the function
test_model(b1_trained_model_v1, custom_test_data, test_loader)
