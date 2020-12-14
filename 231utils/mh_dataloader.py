import os
import shutil
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split

DATADIRS = []
DATADIRS.append("/share/pi/schul/schul-behavioral/data/ai-behavioral-health-mp4/")

# MAKE THIS DIR before you run the script
TARGET_DIR = "/share/pi/schul/schul-behavioral/data/3dr_datasets/mh_targets"
LABEL_DIR = "/share/pi/schul/schul-behavioral/data/3dr_datasets/mh_labels"

ACTIONS_TO_KEEP = range(4)
# 0-4 = minimal risk
# 5-9 = mild/low risk
# 10-14 = moderate/medium risk
# 15+ = severe/high risk
ACTION_NAMES = ["minimal", "mildLow", "modMedium", "severeHigh"]

LABELS_CSV = "/share/pi/schul/schul-behavioral/data/virtual_questionnaire_data_clean.csv"

# Copy files to target directory into the right directory structure given the class
def organize_files_by_class(dirname):
    # Read the labels csv into a df so that you can read the files in numerical order
    csv_df = pd.read_csv(LABELS_CSV)

    # For each line in the csv, skipping the header, find the corresponding video file
    for index, row in csv_df.iterrows():
        patient_id = int(row['participant_id'])
        bucket = int(row['PHQ9_bucket'])
        
        filename = "zoom_" + str(patient_id) + ".mp4"
        filepath = dirname + "/" + filename
        assert(os.path.isfile(filepath))
        
        # Move the file to TARGET_DIR/class_dir
        class_name = ACTION_NAMES[bucket]
        class_dir = TARGET_DIR + "/" + class_name + "/"
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        filebase, ext = filename.split(".")
        new_filename = class_dir+filebase+"_"+str(bucket)+"."+ext
        shutil.copy(dirname+filename, new_filename)
        print("Copied file: " + filename)

# Make classInd files
def make_class_index(targetdir):
    # Make classInd.txt
    df = pd.DataFrame({
        "Numbers": list(ACTIONS_TO_KEEP),
        "ActionNames": ACTION_NAMES
    })
    df.to_csv(LABEL_DIR+"/"+"classInd.txt", header=None, index=None, sep=' ', mode='a')


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
    
    X, y = make_XY(TARGET_DIR)
    print("made X and y")
    
    make_train_test_lists(X, y, test_size=0.2)
    print("finished splitting")
