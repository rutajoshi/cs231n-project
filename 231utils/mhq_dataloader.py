import os
import shutil
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
import csv

DATADIRS = []
#DATADIRS.append("/share/pi/schul/schul-behavioral/data/ai-behavioral-health-mp4/")
DATADIRS.append("/home/ubuntu/data/video")

# MAKE THIS DIR before you run the script
TARGET_DIR = '/home/ubuntu/data/processed_video/phq9_data' #"/Users/ruta/stanford/pac/mentalhealth/mhq_local_targets"
LABEL_DIR = '/home/ubuntu/data/processed_video/phq9_mse_labels' #"/Users/ruta/stanford/pac/mentalhealth/mhq_local_labels"

ACTIONS_TO_KEEP = range(4)
# 0-4 = minimal risk
# 5-9 = mild/low risk
# 10-14 = moderate/medium risk
# 15+ = severe/high risk
ACTION_NAMES = ["minimal", "mildLow", "modMedium", "severeHigh"]

ALL_LABELS = '/home/ubuntu/ketan/questionnaire_data_clean.csv'
LABELS_CSV = '/home/ubuntu/data/processed_video/split_csvs' #"/Users/ruta/stanford/pac/mentalhealth/split_csvs"

# Copy files to target directory into the right directory structure given the class
def organize_files_by_class(dirname):
    # Read the labels csv into a df so that you can read the files in numerical order
    csv_df = pd.read_csv(ALL_LABELS)

    # For each line in the csv, skipping the header, find the corresponding video file
    for index, row in csv_df.iterrows():
        patient_id = int(row['participant_id'])
        bucket = int(row['PHQ9_bucket'])
        #bucket = int(row['GAD7_bucket'])
        
        filename = "zoom_" + str(patient_id) + ".mp4"
        if patient_id < 34:
            filename = "inperson_" + str(patient_id) + ".mkv"
        filepath = dirname + "/" + filename
        assert(os.path.isfile(filepath))
        
        # Move the file to TARGET_DIR/class_dir
        class_name = ACTION_NAMES[bucket]
        class_dir = TARGET_DIR + "/" + class_name + "/"
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        filebase, ext = filename.split(".")
        new_filename = class_dir+filebase+"_"+str(bucket)+"."+ext
        shutil.copy(filepath, new_filename)
        print("Copied file: " + filename)

# Make classInd files
def make_class_index(targetdir):
    # Make classInd.txt
    df = pd.DataFrame({
        "Numbers": list(ACTIONS_TO_KEEP),
        "ActionNames": ACTION_NAMES
    })
    df.to_csv(LABEL_DIR+"/"+"classInd.txt", header=None, index=None, sep=' ', mode='a')

# Make X and Y given the directory structure includes question splits
def make_XY_questions(targetdir):
    X, y = [], []
    for classname in os.listdir(targetdir):
        class_dir = targetdir + "/" + classname + "/"
        for foldername in os.listdir(class_dir):
            zoom, patientID, actionID = foldername.split("_")
            actionID, ext = actionID.split(".")
            action_num = int(actionID)

            video_dir = class_dir + "/" + foldername + "/"
            for filename in os.listdir(video_dir):
                y.append(action_num)
                X.append(classname + "/" + foldername + "/" + filename)
    return X, y

def make_txtfile(csvfilename, txtfilename):
    with open(csvfilename, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        X_section, y_section = [], []
        for row in csv_reader:
            if (line_count != 0):
                #video_num, class_num = int(row[0]), int(row[-1]) #PHQ9
                video_num, class_num = int(row[0]), int(row[-2]) #GAD7
                if (video_num == 2): # No audio to annotate
                    line_count += 1
                    continue

                classname = ACTION_NAMES[class_num] 
                filename = "zoom_" + str(video_num) + ".mp4"
                if (video_num < 34):
                    filename = "inperson_" + str(video_num) + ".mkv"
                filepath = classname + "/" + filename
                
                X_section.append(filepath)
                y_section.append(class_num)
            line_count += 1
        data_df = pd.DataFrame({
            "filename": X_section,
            "label": y_section
        })
        data_df = data_df.sort_values(by=['label'])
        data_df.to_csv(LABEL_DIR+"/"+txtfilename, header=None, index=None, sep=' ', mode='a')


# Get train/val/test splits from the csvs
def get_splits():
    # Make trainlist
    train_csv = LABELS_CSV + "/" + "questionnaire_data_clean_train.csv" 
    # Make vallist
    val_csv = LABELS_CSV + "/" + "questionnaire_data_clean_val.csv"
    # Make testlist
    test_csv = LABELS_CSV + "/" + "questionnaire_data_clean_test.csv"

    # For each csv, read it line by line
    # For each line, you get a video number and a label number
    # Write a line to the dataframe for each line
    # When done, write to txt file
    make_txtfile(train_csv, "trainlist01.txt")
    make_txtfile(val_csv, "vallist01.txt")
    make_txtfile(test_csv, "testlist01.txt")


# Make X and Y for train test splitting
def make_XY(targetdir):
    X, y = [], []
    for classname in os.listdir(targetdir):
        class_dir = targetdir + "/" + classname + "/"
        for filename in os.listdir(class_dir):
            zoom, patientID, actionID = filename.split("_")
            actionID, ext = actionID.split(".")
            action_num = int(actionID)
            y.append(action_num)
            X.append(classname + "/" + filename)
    return X, y

# Make trainlist01.txt and testlist01.txt
def make_train_test_lists(X, y, test_size=0.2):
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

    # Make trainlist01
    traindf = pd.DataFrame({
        "filename": X_train,
        "label": y_train
    })
    traindf = traindf.sort_values(by=['label'])
    traindf.to_csv(LABEL_DIR+"/"+"trainlist01.txt", header=None, index=None, sep=' ', mode='a')

    # Make testlist01
    testdf = pd.DataFrame({
        "filename": X_test,
        "label": y_test
    })
    testdf = testdf.sort_values(by=['label'])
    testdf.to_csv(LABEL_DIR+"/"+"testlist01.txt", header=None, index=None, sep=' ', mode='a')


if __name__ == '__main__':

    for datadir in DATADIRS:
        organize_files_by_class(datadir)
        print("done organizing files by class for datadir: " + datadir)

    make_class_index(TARGET_DIR)
    print("made class index")

    get_splits()
    print("made train, val, and test lists")
    
    #X, y = make_XY(TARGET_DIR)
    #print("made X and y")
    
    #make_train_test_lists(X, y, test_size=0.2)
    #print("finished splitting")
