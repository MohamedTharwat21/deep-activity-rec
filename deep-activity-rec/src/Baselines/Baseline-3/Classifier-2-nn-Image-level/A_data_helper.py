
# we ended that we have trained classifier (first classifier) [b3_trained_model_v1.pth]
# and pickle files saved to next classifier (second classifier)


"""""
-- i write this script to return each 12 players in a frame with each other 
-- the frist script was not interested to combine the players of the frame
-- just have crops and shuffle them form the whole dataset 

"""""


import os
from PIL import Image




dataset_dir = "/kaggle/input/volleyball/volleyball_/videos"



def process_annotations(Data) :
    
    #crops , annotations = [] , []
    #persons , annotation = [] , []
    
    data = []
    for video_id in Data : 
       
        video_dir = os.path.join(dataset_dir, str(video_id))
        annotations_path = os.path.join(video_dir, "annotations.txt")

        with open(annotations_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            crops , annotations = [] , []
            elements = line.strip().split()
            frame_id = elements[0].replace('.jpg', '')
            
            #frame annotation
            #this is for the second classifier
            
            frame_annotation = elements[1]  
            player_annotations = elements[2:]

            # Full path to the frame directory and image file
            frame_dir = os.path.join(video_dir, frame_id)
            image_path = os.path.join(frame_dir, f"{frame_id}.jpg")
 
            
            # Process each player annotation (5 elements: action, x, y, w, h)
            for i in range(0, len(player_annotations), 5):

                x, y, w, h = map(int, player_annotations[i:i+4])
                bbox = (x, y, x + w, y + h)
                # print(bbox)  

                action = player_annotations[i+4]
                # print(action) #falling


                with Image.open(image_path) as img:
                    cropped_img = img.crop(bbox)
                    #print(cropped_img  , action)
                    #crop_to_annotation[cropped_img] = action
                    crops.append(cropped_img)
                    #annotations.append(action)
            
            data.append((crops , frame_annotation  ))

#             crops.append(frame_annotation)
#             data_dict[frame_id] = crops #.append(frame_annotation)
#             persons.extend(crops)
#             annotation.append(frame_annotation)
    
    
    return data

# let's save them in pickle 

def save_pickle_files(train_data ,
                     validation_data ,
                     test_data) :

    import pickle
    # Save the lists as a pickle file
    # save the crops
    
    with open('B3_train_data_12_players.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open('B3_val_data_12_players.pkl', 'wb') as f:
        pickle.dump(validation_data, f)
    with open('B3_test_data_12_players.pkl', 'wb') as f:
        pickle.dump(test_data, f)




if __name__ == "__main__" :
    

    train_videos = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
    validation_videos = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
    test_videos = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]


 

    # data_dict = process_annotations(train_videos)

    train_data = process_annotations(train_videos)
    validation_data = process_annotations(validation_videos)
    test_data = process_annotations(test_videos)
    print("Cropping and annotation saving completed.")



    save_pickle_files(train_data , validation_data , test_data) 