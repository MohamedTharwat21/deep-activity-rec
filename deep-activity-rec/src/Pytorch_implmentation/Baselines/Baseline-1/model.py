import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader


def finetuned_resnet50() :

    
    # Load the pre-trained ResNet-50 model
    model = resnet50(pretrained=True)
    
    # # Modify the final fully connected layer for 9 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8)
    
    # Move the model to the GPU if available
    model = model.to(device)

    return model



def train_model(model) :

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
    
    num_epochs = 10
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
    
        epoch_loss = running_loss / len(train_data)
        epoch_acc = running_corrects.double() / len(train_data)
    
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
    
        epoch_loss = running_loss / len(val_data)
        epoch_acc = running_corrects.double() / len(val_data)
    
        print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        scheduler.step()
        
    
    print('Training Ended Successfully.')




def test_model() :
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    running_corrects = 0
    with torch.no_grad(model = b1_trained_model_v1 , test_loader):
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    
    test_acc = running_corrects.double() / len(test_data)
    print(f'Test Accuracy: {test_acc*100:.4f} %')

test_model(model = b1_trained_model_v1 , test_loader)






