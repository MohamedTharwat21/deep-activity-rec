

"""
this is scripts is mainly solves the problem of imbalanced classes in the training data
so , it will solve the problem on the train data 
in the test and val sets we want to be the same distribution as the actual data
to be representative 
"""
# Get the data splits
 
from A_Get_data_for_A_step import train_crops , train_annotations 
from A_Get_data_for_A_step import validation_crops , validation_annotations
from A_Get_data_for_A_step import test_crops , test_annotations

import random
from collections import Counter
from collections import defaultdict
from torchvision import transforms



def return_stats(train_annotations) :


    # this is a good code for calculating the counts
    """    

    class_counts = Counter([label for _, label in train_dataset_filtered])
    print(class_counts)

    class_counts = Counter([label for _, label in val_dataset_filtered])
    print(class_counts)

    class_counts = Counter([label for _, label in test_dataset_filtered])
    print(class_counts)


    """


    # Initialize the dictionary with all action classes set to 0
    action_counts = {
        'standing': 0,
        'waiting': 0,
        'digging': 0, 
        'setting': 0,
        'spiking': 0, 
        'blocking': 0, 
        'moving': 0, 
        'jumping': 0, 
        'falling': 0
    }

    # Increment the count for each action in the train_mapping
    for val in train_annotations:
        
        # is a way of search
        if val in action_counts:
            action_counts[val] += 1

    # Print the counts for each action class
    for action, count in action_counts.items():
        print(f"{action.capitalize()}: {count}")




# 1: reduce the number of the highest class
# 2: upsample the lowest classes

# Need to be modified as the data becomes lists not adict 

# this will be called in Modify_splitting()
def filter_class_up_down(crops ,
                annotations ,
                target_class_to_undersample,
                target_class_to_upsample,
                max_samples):
    """
    Reduces the number of examples for the target_class to max_samples in the given dataset.
    
    Args:
    - dataset (VolleyballDataset): The dataset to filter.
    - target_class (int): The class label that needs to be reduced.
    - max_samples (int): The maximum number of examples to retain for this class.
    
    Returns:
    - A filtered dataset.
    """

    # class_counts = Counter([label for _, label in dataset])
    # Separate out the data for the target class and other classes
    
 
    target_class_down = [ (image, label) for image, label in zip(crops , annotations ) if label == target_class_to_undersample ]
    target_class_up = [ (image, label) for image, label in zip(crops , annotations ) if label == target_class_to_upsample ]
    
    other_class_data = [(image, label) for image, label in zip(crops , annotations ) 
                        if ((label != target_class_to_undersample) and (label != target_class_to_upsample))]
    
    #target_class_to_undersample
    # Randomly sample from the target class data
    random.shuffle(target_class_down)
    reduced_target_class_data = target_class_down[:max_samples]

    target_class_up = target_class_up * 6
    random.shuffle(target_class_up)
    
    
    # Combine back the reduced target class data with other classes
    filtered_data = reduced_target_class_data + other_class_data + target_class_up
    random.shuffle(filtered_data)  # Shuffle the dataset to mix examples from different classes
    
    return filtered_data





# this is will be called in Train_Data_Augmentation()
def Modify_splitting(train_annotations ,
                     train_crops ) :
    # target_class = 0  # Replace with the label of the class you want to reduce
    target_class_to_undersample = 'standing'  
    target_class_to_upsample = 'jumping'
    # max_samples = 100  # Replace with the desired number of samples for that class



    # list of tuples
    # call the filter_class_up_down ()
    train_dataset_filtered = filter_class_up_down(  train_crops ,
                                            train_annotations  ,
                                            target_class_to_undersample ,
                                            target_class_to_upsample,
                                            12000)


    # let's sperate this list of tuples to 2 lists 
    train_images , train_labels = zip(*train_dataset_filtered)


    # override to convert to lists
    train_images , train_labels  = list(train_images) , list(train_labels)



    return train_images , train_labels

    # just name change
    # use it as it is (to be representative about the actual Distribution)
    # val_images , val_labels =  validation_crops , validation_annotations
    # test_images ,test_labels =  test_crops , test_annotations
    # return  val_images , val_labels , test_images , test_labels





def Train_Data_Augmentation(train_images ,
                      train_labels
                      ) :
    


    # Initialize dictionaries to hold crops for each class
    class_crops = defaultdict(list)

    # Separate crops based on their class
    # like frequency array 
    for crop, annotation in zip(train_images, 
                                train_labels):
        class_crops[annotation].append(crop)

        
    # Define augmentation pipeline
    augmentation_pipeline = {
        'standing' : transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2,
                                    contrast=0.2,
                                    saturation=0.2,
                                    hue=0.2) ,

            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            #i transfromed the img here to tensor 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]) ]),
                
        'other_classes': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2,
                                    contrast=0.2,
                                    saturation=0.2, 
                                    hue=0.2) ,
                                    
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            #i transfromed the img here to tensor 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
        }
    


        # Function to perform augmentation (customize this with your augmentation needs)
    def augment_images(images, augmentation_pipeline):
        augmented_images = []
        for image in images:
            augmented_image = augmentation_pipeline(image)  # Apply pipeline directly to the image
            augmented_images.append(augmented_image)
        return augmented_images
    



    augmented_class_crops = {}
    # augmented_class_crops['standing'] = class_crops['standing']
    for class_name, crops in class_crops.items():
        if class_name == 'standing' : 
            augmented_class_crops[class_name] = augment_images(crops,
                                                                augmentation_pipeline['standing'])
        else : 
            augmented_class_crops[class_name] = augment_images(crops,
                                                                augmentation_pipeline['other_classes'])

            
    # augmented_class_crops['jumping'] = augment_images(class_crops['jumping'], augmentation_pipeline)
    # augmented_class_crops will now contain augmented images for each class


    return augmented_class_crops



    """

    # print(len(class_crops['standing']))
    # print(len(class_crops['jumping']))

    for cls ,_ in augmented_class_crops.items():
        print(len(augmented_class_crops[cls]))


    1127
    2353
    542
    10000
    906
    1518
    1050
    538
    610

    """




def Prepare_train_data_splits(augmented_class_crops) :

    # Assuming your dictionary is called `class_dict`
    images = []
    labels = []

    # Step 1: Convert the dictionary into two lists
    for label, img_list in augmented_class_crops.items():
        for img in img_list:
            images.append(img)
            labels.append(label)

            
    # to save the pairing
    # Step 2: Combine images and labels into pairs
    combined = list(zip(images, labels))

    # Step 3: Shuffle the combined list
    random.shuffle(combined)

    # Step 4: Unzip the shuffled list back into images and labels
    images, labels = zip(*combined)

    # Convert back to list (since zip returns tuples)
    train_images = list(images)
    train_labels = list(labels)

    # Now `images` and `labels` are shuffled

    return train_images , train_labels


def Get_train_val_test_data()  :
    pass




if __name__ == "__main__" :

    return_stats(train_annotations = train_annotations)
    augmented_class_crops = Train_Data_Augmentation(train_crops , train_annotations)
    train_images , train_labels = Prepare_train_data_splits(augmented_class_crops)