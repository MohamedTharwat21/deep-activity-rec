import torch.nn as nn
import torch.optim as optim


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2048, 512)  # First fully connected layer
        self.dropout1 = nn.Dropout(0.1)  # Dropout layer with 50% dropout
        self.relu = nn.ReLU()            # ReLU activation
        self.fc2 = nn.Linear(512, 128)   # Second fully connected layer
        self.dropout2 = nn.Dropout(0)  # Dropout layer with 50% dropout
        self.fc3 = nn.Linear(128, 8)     # Final layer (output for 8 classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)  # Apply dropout after first fully connected layer
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)  # Apply dropout after second fully connected layer
        x = self.fc3(x)
        return x


# Instantiate the model
model = SimpleNN()


# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)



## Training
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training function with printing of train loss, train accuracy, val loss, and val accuracy
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs= 200):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Training loop
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Zero out gradients
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()  # Accumulate training loss
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_accuracy = 100 * correct_train / total_train  # Calculate training accuracy

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for val_features, val_labels in val_loader:
                val_features, val_labels = val_features.to(device), val_labels.to(device)
                
                outputs = model(val_features)
                loss = criterion(outputs, val_labels)
                
                val_loss += loss.item()  # Accumulate validation loss
                
                _, predicted = torch.max(outputs, 1)
                total_val += val_labels.size(0)
                correct_val += (predicted == val_labels).sum().item()
        
        val_accuracy = 100 * correct_val / total_val  # Calculate validation accuracy

        # Print the losses and accuracies
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {running_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer)




## testing


# Test function
def test_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad():
        for test_features, test_labels in test_loader:
            test_features, test_labels = test_features.to(device), test_labels.to(device)
            
            outputs = model(test_features)
            _, predicted = torch.max(outputs, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


# Evaluate on the test set
test_model(model, test_loader)



if __name__ == "__main__" :
    pass