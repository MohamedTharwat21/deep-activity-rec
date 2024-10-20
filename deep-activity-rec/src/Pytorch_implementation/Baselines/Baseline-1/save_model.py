# Attention !

# this part [if you are doing the training 
# on online cloud based service like colab or kaggle 
# and you do not like to save your work in your disk 
# if you are kaggler you can use kaggle datasets 
# as your own disk]


from model import b1_trained_model_v1

def save_model_on_kaggle_datasets() : 
    import json
    !mkdir -p ~/.kaggle
    !cp /kaggle/input/jsssooonfile/kaggle.json  ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
    !kaggle datasets init -p /kaggle/working/
    
    # Path to the metadata file
    metadata_path = '/kaggle/working/dataset-metadata.json'
    
    # Load existing metadata
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)
    
    # Update the metadata fields with a unique title and ID
    metadata['id'] = 'mohammedtharwat339/b1-trained-model-v1-4732598a'  # Ensure the ID is unique
    metadata['title'] = 'b1_trained_model_v1_4732598a'  # Ensure the title is unique
    
    # Save the updated metadata
    with open(metadata_path, 'w') as file:
        json.dump(metadata, file, indent=4)
    
    print("Metadata updated successfully!")
    
    # Upload the dataset
    !kaggle datasets create -p /kaggle/working/


if __name__ == '__main__' :
    import torch
    # Assuming `model` is your trained PyTorch model

    # it will be saved on kaggle current working directory /kaggle/working/
    # so i will save in it , then i will move to dataset dir

    torch.save(b1_trained_model_v1.state_dict(), 'b1_trained_model_v1.pth')
    save_model_on_kaggle_datasets()