import os
import pickle

def load_annotations(video_dir):
    """Load annotations from the annotations.txt file."""
    annotations_path = os.path.join(video_dir, 'annotations.txt')
    annotations = {}

    with open(annotations_path, 'r') as file:
        for line in file:

            #modifications
            parts = line.strip().split()
            frame_id, annotation = parts[0] , parts[1]

            annotations[int(frame_id[:-4])] = annotation

    return annotations


def map_frame_to_annotation(dataset_dir, video_ids):

    # the middle frame has the same name as the clip which contains it
    """Map frames in a video directory to their annotations."""
    frame_to_annotation_map = {}

    #  video_id = 0  
    for video_id in video_ids:
                                #volleyball_      str(0)
                                # concatenate the video_id to the the path

        video_dir = os.path.join(dataset_dir, str(video_id))

                                #  annotations with vid 0
                                # this function to get the [annotations.txt] from the videp
                                # dir , and we take what we want to train on
        annotations = load_annotations(video_dir)

        for frame_dir in os.listdir(video_dir):
            frame_path = os.path.join(video_dir, frame_dir)
            if os.path.isdir(frame_path):
                frame_id = int(frame_dir)  # Convert frame directory name to an integer ID

                 

                # this is take the target frame only from the whole frames
                if frame_id in annotations:
                    frame_to_annotation_map[frame_path] = annotations[frame_id]

    return frame_to_annotation_map


def save_pickle_files(train_mapping , validation_mapping , test_mapping ) :
    # Save the dictionary as a pickle file
    with open('train_mapping.pkl', 'wb') as f:
        pickle.dump(train_mapping, f)
    with open('val_mapping.pkl', 'wb') as f:
        pickle.dump(validation_mapping, f)
    with open('test_mapping.pkl', 'wb') as f:
        pickle.dump(test_mapping, f)
    
    
def load_pickle_files() :

    # here we load from the disk after we have saved them
    # i saved them on kaggle datasets so u have to save them first 
    # then load them using this fun

    # Load the dictionary from the pickle file
    with open('train_mapping.pkl', 'rb') as f:
        train_mapping = pickle.load(f)
        
    with open('val_mapping.pkl', 'rb') as f:
        val_mapping = pickle.load(f)

    with open('test_mapping.pkl', 'rb') as f:
        test_mapping = pickle.load(f)

    return train_mapping , val_mapping , test_mapping



if __name__ == "__main__" :
    # Example usage
    # you have to go into the videos dir to find the clips 
    # this specified path as i uploaded the dataset to kaggle
    # then i upload it to the input dir to read it 
    # 
     
    dataset_dir = '/kaggle/input/volleyball/volleyball_/videos'

    
    train_videos = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
    validation_videos = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
    test_videos = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

    
    # Get the mapping for training videos
    train_mapping = map_frame_to_annotation(dataset_dir, train_videos)  #2152
    validation_mapping = map_frame_to_annotation(dataset_dir, validation_videos)  
    test_mapping = map_frame_to_annotation(dataset_dir, test_videos)  
    print('end..')

    save_pickle_files(train_mapping , validation_mapping , test_mapping)
    
    # train_dataset , val_dataset  , test_dataset = load_pickle_files()