import os

"""
prepare annotations on the image level (8 classes) 
for NN training 

"""

videos_annot = "Path_to_your_pickle_file"




train_ids = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31",
                 "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]
val_ids = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]
test_ids = ["4", "5", "9", "11", "14", "20", "21", "25", "29", "34", "35", "37", "43", "44", "45", "47"]



def get_train_val_test_Scene_annot() :

    train_dict_scene_level_labels = {}
    val_dict_scene_level_labels = {}
    test_dict_scene_level_labels = {}


    # loop on 154 instances
    for vid_id in train_ids :
        dict_helper = {}
        for clip in videos_annot[vid_id].keys():
            label_of_the_clip = videos_annot[vid_id][str(clip)]['category']
            dict_helper[clip] = label_of_the_clip
            train_dict_scene_level_labels[vid_id] =  dict_helper 
            
    # loop on 154 instances
    for vid_id in val_ids :
        dict_helper = {}
        for clip in videos_annot[vid_id].keys():
            label_of_the_clip = videos_annot[vid_id][str(clip)]['category']
            dict_helper[clip] = label_of_the_clip
            val_dict_scene_level_labels[vid_id] =  dict_helper 
            
    # loop on 154 instances
    for vid_id in test_ids :
        dict_helper = {}
        for clip in videos_annot[vid_id].keys():
            label_of_the_clip = videos_annot[vid_id][str(clip)]['category']
            dict_helper[clip] = label_of_the_clip
            test_dict_scene_level_labels[vid_id] =  dict_helper 



    
    return train_dict_scene_level_labels , val_dict_scene_level_labels , test_dict_scene_level_labels




def split_train_val_test() :
    root = '/kaggle/working/features/crop-level/Lstm'

    train_dataset_NN = []
    val_dataset_NN = []
    test_dataset_NN = []

    for vid_id in train_ids : 
        path = os.path.join(root , str(vid_id))
        for seq in os.listdir(path) :
            train_dataset_NN.append(os.path.join(root , str(vid_id) , seq))
            
            
            
    for vid_id in val_ids : 
        path = os.path.join(root , str(vid_id))
        for seq in os.listdir(path) :
            val_dataset_NN.append(os.path.join(root , str(vid_id) , seq))
            
            
            
    for vid_id in test_ids : 
        path = os.path.join(root , str(vid_id))
        for seq in os.listdir(path) :
            test_dataset_NN.append(os.path.join(root , str(vid_id) , seq))



    return train_dataset_NN , val_dataset_NN , test_dataset_NN