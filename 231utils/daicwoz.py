import numpy as np
import pandas as pd
import csv
import torch
import os

LABELS_TRAIN = "/home/ubuntu/data/daicwoz/daicwoz_manifests/train.csv"
LABELS_VAL = "/home/ubuntu/data/daicwoz/daicwoz_manifests/val.csv"

ACTION_NAMES = {0: "minimal", 1: "mildLow", 2: "modMedium", 3: "severeHigh"}

def make_label_dicts():
    train_labels, val_labels = {}, {}
    
    train_df = pd.read_csv(LABELS_TRAIN)
    val_df = pd.read_csv(LABELS_VAL)

    # Go through the train and val csvs and make a dict
    for index, row in train_df.iterrows():
        participant_id = int(row["participant_id"])
        phq_bucket = int(row["PHQ_bucket"])
        train_labels[participant_id] = phq_bucket

    for index, row in val_df.iterrows():
        participant_id = int(row["participant_id"])
        phq_bucket = int(row["PHQ_bucket"])
        val_labels[participant_id] = phq_bucket

    return train_labels, val_labels

def kpt_name_formatter(x):
    num = str(x).zfill(5)
    return 'image_' + num + '.pt'

def read_all_keypoints(src_root_dir, dst_root_dir, train_labels, val_labels, frames_to_sample):
    # Go through all sessions from 300 to 492 inclusive
    for session in range(300, 493):
        if session in [342,394,398,460]:
            continue
        session_path = src_root_dir + "/" + str(session) + "_CLNF_features3D.txt"
        csv_df = pd.read_csv(session_path)
        skip = 30 #len(csv_df.index) // frames_to_sample
        ctr = 0
        for index, row in csv_df.iterrows():
            # Skip if it is not needed
            if index % skip != 0: # or ctr >= frames_to_sample:
                continue

            # Otherise: Keep this frame and write to dst
            # Get the class from the correct dict
            classname = "no class"
            if session in train_labels:
                classname = ACTION_NAMES[train_labels[session]]
            elif session in val_labels:
                classname = ACTION_NAMES[val_labels[session]]
            else:
                print("Session: " + str(session) + " not found in train or val csv")
                continue

            # Make class folder if it does not exist yet
            if not os.path.exists(dst_root_dir + "/" + classname):
                os.makedirs(dst_root_dir + "/" + classname)

            # Make video name
            videoname = str(session)

            # Make video folder if it does not exist yet
            if not os.path.exists(dst_root_dir + "/" + classname + "/" + videoname):
                os.makedirs(dst_root_dir + "/" + classname + "/" + videoname)

            # Format image name
            kptfilename = kpt_name_formatter(ctr)

            # Make dst path
            dst_path = dst_root_dir + "/" + classname + "/" + videoname + "/" + kptfilename

            # Read the keypoints into a torch tensor
            if (row[" X"+str(0)]) == " -1.#IND":
            #if (row[" x"+str(0)]) == " -1.#IND":
                print("Something is wrong with session " + str(session))
                continue
            keypts = np.array([[float(row[" X"+str(i)]), float(row[" Y"+str(i)]), float(row[" Z"+str(i)])] for i in range(68)])
            #keypts = np.array([[float(row[" x"+str(i)]), float(row[" y"+str(i)])] for i in range(68)])
            keypts = torch.from_numpy(keypts)

            # Write the torch tensor to the dst path
            torch.save(keypts, dst_path)
            ctr += 1

        print("Done with video: " + str(session) + ", wrote " + str(ctr) + " files")
    print("Done with all videos in daicwoz dataset")

def makeTrainTestList(train_labels, val_labels):
    with open("/home/ubuntu/data/daicwoz_1fps/daicwoz_multiclass_labels/trainlist01.txt", 'a+') as trainFile:
        for key in train_labels:
            # Write to a file
            value = train_labels[key]
            classname = ACTION_NAMES[value]
            trainFile.write(classname + "/" + str(key) + " " + str(value) + "\n")

    with open("/home/ubuntu/data/daicwoz_1fps/daicwoz_multiclass_labels/vallist01.txt", 'a+') as valFile:
        for key in val_labels:
            # Write to a file
            value = val_labels[key]
            classname = ACTION_NAMES[value]
            valFile.write(classname + "/" + str(key) + " " + str(value) + "\n")

    with open("/home/ubuntu/data/daicwoz_1fps/daicwoz_binary_labels/trainlist01.txt", 'a+') as trainFile:
        for key in train_labels:
            # Write to a file
            value = train_labels[key]
            classname = ACTION_NAMES[value]
            if classname != "minimal":
                classname = "notable"
                value = 1
            trainFile.write(classname + "/" + str(key) + " " + str(value) + "\n")

    with open("/home/ubuntu/data/daicwoz_1fps/daicwoz_binary_labels/vallist01.txt", 'a+') as valFile:
        for key in val_labels:
            # Write to a file
            value = val_labels[key]
            classname = ACTION_NAMES[value]
            if classname != "minimal":
                classname = "notable"
                value = 1
            valFile.write(classname + "/" + str(key) + " " + str(value) + "\n")

def main():
    src_root_dir = "/home/ubuntu/data/daicwoz/daicwoz_3Dfeats"
    dst_root_dir = "/home/ubuntu/data/daicwoz_1fps/daicwoz_multiclass_keypts_3d"
    if not os.path.exists(dst_root_dir):
        os.makedirs(dst_root_dir)
    train_labels, val_labels = make_label_dicts()
    print("Made label dicts")
    read_all_keypoints(src_root_dir, dst_root_dir, train_labels, val_labels, 353)
    #makeTrainTestList(train_labels, val_labels)
    print("Done")

main()
