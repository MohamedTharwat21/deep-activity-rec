"""

build an LSTM classifier that takes a sequence of 9 feature
vectors [tensors] (each of length 2048) and a one Label.

"""

from D_custom_dataset import train_loader , val_loader ,  test_loader
import torch
import torch.nn as nn


# Define the device to run the model on (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMClassifier(nn.Module):
    def __init__(self, input_size,
                  hidden_size, 
                  num_layers, 
                  num_classes):
        
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
                           # 2048  to  512
                           # 512   to  8  cls
        self.lstm = nn.LSTM(input_size,
                             hidden_size,
                            num_layers,
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size,
                            num_classes) #it will be applied only on the last frame 
        # of the 9 frames
        
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) #lstm specific

        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        classification = self.fc(out[:, -1, :])  # out: tensor of shape (batch_size, seq_length, num_classes)
        return classification
    




# (hidden size and num_layers) must be modified to get a good training

# Hyperparameters
input_size = 2048      # Size of each feature vector
hidden_size = 512      # Number of features in hidden state
num_layers = 2         # Number of stacked LSTM layers
num_classes = 8        # Number of output classes (assuming 8 classes)
sequence_length = 9    # Length of the input sequence

# Create model
model = LSTMClassifier(input_size,
                        hidden_size,
                          num_layers,
                            num_classes)



model.to(device)



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_lstm_classifier(train_loader,
                           val_loader,
                          test_loader,
                         num_epochs):
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # data: shape (batch_size, 1, sequence_length, input_size)
            # target: shape (batch_size)
            
            data = data.squeeze(2)  # Remove the extra dimension
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
                data = data.squeeze(2)  # Remove the extra dimension
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
            data = data.squeeze(2)  # Remove the extra dimension
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




# Example inference
def predict_sequence(feature_vectors):
    model.eval()
    with torch.no_grad():
        feature_vectors = torch.tensor(feature_vectors).unsqueeze(0).to(device)  # Add batch dimension
        outputs = model(feature_vectors)
        _, predicted_labels = torch.max(outputs, 1)
        return predicted_labels.squeeze(0).cpu().numpy()



if __name__ == "__main__" :

    b4_trained_lstm_imp1 = train_lstm_classifier(train_loader,
                                                val_loader,
                                                test_loader,
                                                num_epochs = 20)
    


    # save the model 
    torch.save(b4_trained_lstm_imp1 , 'b4_trained_lstm_imp1_v1.pth')