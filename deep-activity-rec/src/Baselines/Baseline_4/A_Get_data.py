"""

use the classifier b1-tuned on the image level to extract repre per clip
use 9 frames per clip
now you have a sequence of 9 frames -- apply classifier -- then you have 9 feat. vec
apply lstm on these 9 feat vec 

"""



# python code to get the middle 9 frames to use them to extract features of them
# using finetuned resnet50 from baseline-1 , and get their annotations too


import os
import pickle 



def get_middle_9_frames(frame_ids):
    middle_index = len(frame_ids) // 2
    return frame_ids[middle_index-4:middle_index+5], frame_ids[middle_index]



def process_video(video_id, base_path):
    video_path = os.path.join(base_path, str(video_id))
    annotations_file = os.path.join(video_path, 'annotations.txt')
    
    if not os.path.exists(annotations_file):
        return []

    # Read annotations file
    with open(annotations_file, 'r') as f:
        annotations = {line.split()[0]: line.split()[1] for line in f.readlines()}

    video_data = []

    # Process each frame directory inside the video
    for frame_dir in os.listdir(video_path):
        frame_dir_path = os.path.join(video_path, frame_dir)
        
        if not os.path.isdir(frame_dir_path):
            continue

        # Get all frame IDs
        frame_ids = sorted([int(img.split('.')[0]) for img in os.listdir(frame_dir_path)])
        
        if len(frame_ids) < 41:
            continue  # skip if there are not enough frames

        # Get the middle 9 frames
        middle_9_frames, middle_frame = get_middle_9_frames(frame_ids)
        

        # Get the annotation for the middle frame
        label = annotations.get(str(middle_frame)+'.jpg')
        
        if label:
            # add the full path
            i = 0
            for frame in middle_9_frames:
                middle_9_frames[i] = os.path.join(frame_dir_path , str(frame) + '.jpg' ) 
                i = i + 1
            video_data.append( (middle_9_frames, label) )

    return video_data

def process_dataset(video_ids, base_path):
    dataset_data = []

    for video_id in video_ids:
        video_data = process_video(video_id, base_path)
        dataset_data.extend(video_data)
        print(f"{video_id} has completed .." )

    return dataset_data



def save_files_trainvaltest(train_data,
                           validation_data,
                           test_data):
    with open('B4_train_annot_data_image_level.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open('B4_val_annot_data_image_level.pkl', 'wb') as f:
        pickle.dump(validation_data, f)
    with open('B4_test_annot_data_image_level.pkl', 'wb') as f:
        pickle.dump(test_data, f)




if __name__ == "__main__" :

    base_path = '/kaggle/input/volleyball/volleyball_/videos'

    train_video_ids = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
    validation_video_ids = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
    test_video_ids = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]



    train_data = process_dataset(train_video_ids, base_path)
    validation_data = process_dataset(validation_video_ids, base_path)
    test_data = process_dataset(test_video_ids, base_path)



    # saving pickle version of the files
    save_files_trainvaltest(train_data ,
                            validation_data ,
                            test_data)



