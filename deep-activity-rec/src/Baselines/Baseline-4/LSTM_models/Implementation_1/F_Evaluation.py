"""
Obtain predictions and true labels: After finishing the training, 
you need to run your model on the test or validation set and collect
predictions and corresponding true labels.
"""


import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
# get the train_features to get the classes from it
from C_Feature_extractor import train_features
# get test_loader to evaluate on it
from D_custom_dataset import test_loader
from E_Train_LSTM_on_Features import b4_trained_lstm_imp1





# Assuming 'train_features' is your dataset containing features and labels
def Get_classes() :
    # Extract labels
    labels = [label for _, label in train_features]

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Fit the label encoder and transform the labels
    encoded_labels = label_encoder.fit_transform(labels)

    # Get the unique labels and their corresponding encoded values
    unique_labels = list(set(labels))
    encoded_unique_labels = label_encoder.transform(unique_labels)

    # Create the mapping between class names and their encoded labels
    label_mapping = dict(zip(unique_labels, encoded_unique_labels))

    # Sort the mapping based on the encoding number
    sorted_label_mapping = dict(sorted(label_mapping.items(),
                                        key=lambda item: item[1]))
    



# Assuming you have a DataLoader for your validation or test set
def evaluate_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming 'b4_trained_lstm_imp1' 
# is your trained ResNet50 model 
# and 'test_loader' is your DataLoader

preds, true_labels = evaluate_model(b4_trained_lstm_imp1, test_loader, device)


# Print classification report
print("Classification Report:")
print(classification_report(true_labels,
                            preds, 
                            target_names=['l-pass', 
                                          'l-spike', 
                                          'l_set', 
                                          'l_winpoint',
                                          'r-pass', 
                                          'r_set',
                                          'r_spike',
                                          'r_winpoint']))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(true_labels, preds))




accuracy = accuracy_score(true_labels, preds)
f1 = f1_score(true_labels,
             preds,
             average='weighted')  # Can be 'macro', 'micro', or 'weighted'

precision = precision_score(true_labels, preds, average='weighted')
recall = recall_score(true_labels, preds, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")


 