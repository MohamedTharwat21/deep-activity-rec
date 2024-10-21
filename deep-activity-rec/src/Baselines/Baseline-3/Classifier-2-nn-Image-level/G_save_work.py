
# Attention !

"""Go to the notebook version to run Linux Kernel Commands """

"""this part [if you are doing the training 
on online cloud based service like colab or kaggle 
and you do not like to save your work in your disk 
if you are kaggler you can use kaggle datasets 
as your own disk]"""





# move them
# You have to move the saved pickle files to easily load it again fast 
def save_data_on_kaggle_datasets() : 
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
    metadata['id'] = 'mohammedtharwat339/b3-dataset-12-players-crops-annot-4732598a'  # Ensure the ID is unique
    metadata['title'] = 'b3-12-players-dataset-crops-annot-4732598a'  # Ensure the title is unique
    
    # Save the updated metadata
    with open(metadata_path, 'w') as file:
        json.dump(metadata, file, indent=4)
    
    print("Metadata updated successfully!")
    
    # Upload the dataset
    !kaggle datasets create -p /kaggle/working/



if __name__ == "__main__" :
    save_data_on_kaggle_datasets()