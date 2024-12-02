from B_prepare_annotations import get_train_val_test_Scene_annot , split_train_val_test
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np



# Define a function to encode labels
def encode_labels(label_dict):
    unique_labels = set()
    for class_id, files in label_dict.items():
        for file_id, label in files.items():
            unique_labels.add(label)
    
    label_to_index = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    return label_to_index




# Create the Custom Dataset
class CustomFeatureDataset(Dataset):
    def __init__(self, file_paths, label_dict):
        self.file_paths = file_paths
        self.label_dict = label_dict
        self.label_to_index = encode_labels(label_dict)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load the feature vector from .npy file
        feature_vector = np.load(self.file_paths[idx])
        label = self.get_label(self.file_paths[idx])
        
        return torch.tensor(feature_vector.squeeze(0), dtype=torch.float32), label

    def get_label(self, file_path):
        # Extract class_id and file_id from file_path
        parts = file_path.split('/')
        class_id = parts[-2]  # Assuming the directory name is the class_id
        file_id = parts[-1].split('.')[0]  # Extract the file id without extension
        
        # Return the encoded label
        return self.label_to_index[self.label_dict[class_id][file_id]]






# Define the Neural Network with Dropout
class SimpleNN(nn.Module):
    def __init__(self, input_size=512, num_classes=8, dropout_rate=0.5):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x





# Training and Validation with Early Stopping
def train_model(train_dataset, val_dataset, test_dataset, num_epochs , batch_size=32, learning_rate=0.001, patience=5):
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_accuracy = 100 * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        print('-' * 100)


    # Testing loop
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total
    print(' ')
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')




if __name__ == "__main__" :

    train_dataset_NN , val_dataset_NN , test_dataset_NN = split_train_val_test
    train_dict_scene_level_labels , val_dict_scene_level_labels , test_dict_scene_level_labels = get_train_val_test_Scene_annot


    # Create the dataset
    train_dataseet_nn = CustomFeatureDataset(train_dataset_NN, train_dict_scene_level_labels)
    val_dataseet_nn = CustomFeatureDataset(val_dataset_NN, val_dict_scene_level_labels)
    test_dataseet_nn = CustomFeatureDataset(test_dataset_NN, test_dict_scene_level_labels)


    trained_NN_model = train_model(train_dataseet_nn,
                                      val_dataseet_nn,
                                      test_dataseet_nn , 
                                      num_epochs = 70)
    


    # save the model (Nerual Network)
    torch.save(trained_NN_model , 'trained_NN_model_b5.pth')