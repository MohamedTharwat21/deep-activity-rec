# here is the first classifier (which is step A of b3 finetuned person classification)

import os
from PIL import Image
import pickle

dataset_dir = "/kaggle/input/volleyball/volleyball_/videos"

def process_annotations(Data) :
    
    crops , annotations = [] , []
    for video_id in Data : 
        video_dir = os.path.join(dataset_dir, str(video_id))
        annotations_path = os.path.join(video_dir, "annotations.txt")

        with open(annotations_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            elements = line.strip().split()
            frame_id = elements[0].replace('.jpg', '')
            player_annotations = elements[2:]

            # Full path to the frame directory and image file
            frame_dir = os.path.join(video_dir, frame_id)
            image_path = os.path.join(frame_dir, f"{frame_id}.jpg")
 
            
            # Process each player annotation (5 elements: action, x, y, w, h)
            for i in range(0, len(player_annotations), 5):

                x, y, w, h = map(int, player_annotations[i:i+4])
                bbox = (x, y, x + w, y + h)
                #         print(bbox)  
                action = player_annotations[i+4]
                #         print(action) #falling
                with Image.open(image_path) as img:
                    cropped_img = img.crop(bbox)
                    #print(cropped_img  , action)
                    #crop_to_annotation[cropped_img] = action
                    crops.append(cropped_img)
                    annotations.append(action)

                    
    return crops , annotations




def save_pickle_files(train_crops ,
                      train_annotations , 
                      validation_crops ,
                      validation_annotations ,
                      test_crops ,
                      test_annotaitons) :

    import pickle
    # Save the lists as a pickle file
    # save the crops
    
    with open('train_crops.pkl', 'wb') as f:
        pickle.dump(train_crops, f)
    with open('validation_crops.pkl', 'wb') as f:
        pickle.dump(validation_crops, f)
    with open('test_crops.pkl', 'wb') as f:
        pickle.dump(test_crops, f)


    
    # save the annotations
    with open('train_annotations.pkl', 'wb') as f:
        pickle.dump(train_annotations, f)
    with open('validation_annotations.pkl', 'wb') as f:
        pickle.dump(validation_annotations, f)
    with open('test_annotations.pkl', 'wb') as f:
        pickle.dump(test_annotations, f)


    


def load_pickle_files() :
    # Load the list from the pickle file

    # Load the crops
    with open('train_crops.pkl', 'rb') as f:
        train_crops = pickle.load(f)
        
    with open('validation_crops.pkl', 'rb') as f:
        validation_crops = pickle.load(f)

    with open('test_crops.pkl', 'rb') as f:
        test_crops = pickle.load(f)


    # load the annotations
    with open('train_annotations.pkl', 'rb') as f:
        train_annotations = pickle.load(f)
        
    with open('validation_annotations.pkl', 'rb') as f:
        validation_annotations = pickle.load(f)

    with open('test_annotations.pkl', 'rb') as f:
        test_annotations = pickle.load(f)

    return  train_crops ,train_annotations ,  validation_crops ,validation_annotations ,test_crops ,test_annotations




if __name__ == '__main__' : 



    train_videos = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
    validation_videos = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
    test_videos = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
    

    
    train_crops , train_annotations = process_annotations(train_videos)
    validation_crops , validation_annotations = process_annotations(validation_videos)
    test_crops , test_annotations = process_annotations(test_videos)
    # print(annotations)



    print("Cropping and annotation saving completed.")

  


    # Save 
    save_pickle_files(train_crops ,
                        train_annotations , 
                        validation_crops ,
                        validation_annotations ,
                        test_crops ,
                        test_annotations)
    


    # this all you need , don't create a pickle each time , just laod then call
    root = '/kaggle/input/b3-dataset-crops-and-annotations-4732598a/'
    # load 
    train_crops ,
    train_annotations ,
    validation_crops ,
    validation_annotations ,
    test_crops ,test_annotations = load_pickle_files(root)

    # /kaggle/input/b3-dataset-crops-and-annotations-4732598a/test_annotations.pkl