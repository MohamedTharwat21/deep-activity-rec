"""

we will call the dataset saved using pickle file
here then construct it to be ready for LSTM

"""
import os

videos_annot = 'Path_to_your_pickle_file'


def split_tra_val_test( train_ids , val_ids , test_ids ) :

    """  
    this function to get the representation in a list 
    (and to split them to train val test sets)
    """
    
    root = '/kaggle/input/b5-v1-resnet-feature-vectors-9-12-472598a/kaggle/working/features/image-level/resnet'

    train_dataset = []
    val_dataset = []
    test_dataset = []

    for vid_id in train_ids : 
        path = os.path.join(root , str(vid_id))
        for seq in os.listdir(path) :
            train_dataset.append(os.path.join(root , str(vid_id) , seq))
            
            
            
    for vid_id in val_ids : 
        path = os.path.join(root , str(vid_id))
        for seq in os.listdir(path) :
            val_dataset.append(os.path.join(root , str(vid_id) , seq))
            
            
            
    for vid_id in test_ids : 
        path = os.path.join(root , str(vid_id))
        for seq in os.listdir(path) :
            test_dataset.append(os.path.join(root , str(vid_id) , seq))



    
    return train_dataset , val_dataset , test_dataset
            


def get_train_annotations(train_ids) :

    train_dict_person_level_labels = {}
    # loop on 154 instances
    for vid_id in train_ids :
        dict_helper = {}
        for clip in videos_annot[vid_id].keys():
            seq_to_take_annot = videos_annot[vid_id][str(clip)]['frame_boxes_dct'][int(clip)]
            labels = []
                    
            for player in seq_to_take_annot :
                labels.append(player.category)

            dict_helper[clip] = labels
            train_dict_person_level_labels[vid_id] =  dict_helper 


    return train_dict_person_level_labels
        
            

def get_val_annotations(val_ids) :


    val_dict_person_level_labels = {}
    # loop on 154 instances
    for vid_id in val_ids :

        dict_helper = {}
        for clip in videos_annot[vid_id].keys():
            seq_to_take_annot = videos_annot[vid_id][str(clip)]['frame_boxes_dct'][int(clip)]
            labels = []

            for player in seq_to_take_annot :
                labels.append(player.category)

            dict_helper[clip] = labels
            val_dict_person_level_labels[vid_id] =  dict_helper 


    return val_dict_person_level_labels



def get_test_annotations(test_ids) :

    test_dict_person_level_labels = {}

    # loop on 154 instances
    for vid_id in test_ids :

        dict_helper = {}
        for clip in videos_annot[vid_id].keys():
            seq_to_take_annot = videos_annot[vid_id][str(clip)]['frame_boxes_dct'][int(clip)]
            labels = []
            
            for player in seq_to_take_annot :
                labels.append(player.category)
                
            dict_helper[clip] = labels
            test_dict_person_level_labels[vid_id] =  dict_helper     


    return test_dict_person_level_labels





if __name__ == "__main__" :


    train_ids = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31",
                 "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]
    
    val_ids = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]
    test_ids = ["4", "5", "9", "11", "14", "20", "21", "25", "29", "34", "35", "37", "43", "44", "45", "47"]



    # annotations (Output)
    train_dict_person_level_labels = get_train_annotations(train_ids)
    val_dict_person_level_labels = get_val_annotations(val_ids)
    test_dict_person_level_labels = get_test_annotations(test_ids)


    # features (crops) (Input)
    train_dataset , val_dataset , test_dataset = split_tra_val_test(train_ids ,
                                                                    val_ids ,
                                                                    test_ids)