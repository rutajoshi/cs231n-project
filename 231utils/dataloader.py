import os
import shutil
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split

DATADIRS = []
#DATADIRS.append("/home/ruta/dataset/nturgbd_rgb_s001/nturgbd_rgb/")
#DATADIRS.append("/home/ruta/dataset/nturgbd_rgb_s002/nturgbd_rgb/")
#DATADIRS.append("/home/ruta/dataset/nturgbd_rgb_s004/nturgbd_rgb/")
#DATADIRS.append("/home/ruta/dataset/nturgbd_rgb_s015/nturgbd_rgb/")
DATADIRS.append("/home/ruta/dataset/nturgbd_rgb_s005/nturgbd_rgb/")

#DATADIRS.append("/home/ruta/dataset/nturgbd_rgb_s016/nturgbd_rgb/")
#DATADIRS.append("/home/ruta/dataset/nturgbd_rgb_s017/nturgbd_rgb/")

# MAKE THIS DIR before you run the script
TARGET_DIR = "/home/ruta/teeny_data/nturgb/"
LABEL_DIR = "/home/ruta/teeny_ntuTrainTestlist/"

ACTIONS_TO_KEEP = range(41, 50)
ACTION_NAMES = ["sneezeCough", "staggering", "fallingDown",
                "headache", "chestPain", "backPain",
                "neckPain", "nauseaVomiting", "fanSelf"]

ALL_ACTIONS_TO_KEEP = range(1, 61)
ALL_ACTIONS = ["drinkWater",
        "eatMeal",
        "brushTeeth",
        "brushHair",
        "drop",
        "pickUp",
        "throw",
        "sitDown",
        "standUp",
        "clapping",
        "reading",
        "writing",
        "tearUpPaper",
        "putOnJacket",
        "takeOffJacket",
        "putOnShoe",
        "takeOffShoe",
        "putOnGlasses",
        "takeOffGlasses",
        "putOnHat",
        "takeOffHat",
        "cheerUp",
        "handWaving",
        "kickingSomething",
        "reachIntoPocket",
        "hopping",
        "jumpUp",
        "phoneCall",
        "playWithPhone",
        "typeOnKeyboard",
        "pointToSomething",
        "takingSelfie",
        "checkTime",
        "rubTwoHands",
        "nodHead",
        "shakeHead",
        "wipeFace",
        "salute",
        "putPalmsTogether",
        "crossHandsInFront",
        "sneezeCough",
        "staggering",
        "fallingDown",
        "headache",
        "chestPain",
        "backPain",
        "neckPain",
        "nauseaVomiting",
        "fanSelf",
        "punchSlap",
        "kicking",
"pushing",
"patOnBack",
"pointFinger",
"hugging",
"givingObject",
"touchPocket",
"shakingHands",
"walkingTowards",
"walkingApart"]


# Loop through the files and don't delete any actions.
def no_filter_dir(dirname):
    for filename in os.listdir(dirname):
        left, _ = filename.split("_")
        action_num = int(left[-3:])
        # move the file to the TARGET_DIR/class_dir
        class_dir = TARGET_DIR+ALL_ACTIONS[action_num - 1] + "/"
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        shutil.copy(dirname+filename, class_dir+filename)

# Loop through the files and delete the ones that are not from actions 41-49
def filter_dir(dirname):
    for filename in os.listdir(dirname):
        left, _ = filename.split("_")
        action_num = int(left[-3:])
        if (action_num in ACTIONS_TO_KEEP):
            # move the file to the TARGET_DIR/class_dir
            class_dir = TARGET_DIR+ACTION_NAMES[action_num - 41] + "/"
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)
            shutil.copy(dirname+filename, class_dir+filename)

# Make classInd files
def make_class_index(targetdir):
    # Make classInd.txt
    df = pd.DataFrame({
        "Numbers": list(ACTIONS_TO_KEEP),
        "ActionNames": ACTION_NAMES
    })
    df.to_csv(LABEL_DIR+"classInd.txt", header=None, index=None, sep=' ', mode='a')

# Make classInd files for all actions
def make_class_index_all(targetdir):
    # Make classInd.txt
    df = pd.DataFrame({
        "Numbers": list(ALL_ACTIONS_TO_KEEP),
        "ActionNames": ALL_ACTIONS
    })
    df.to_csv(LABEL_DIR+"classInd.txt", header=None, index=None, sep=' ', mode='a')


# Make X and Y for train test splitting
def make_XY(targetdir):
    X, y = [], []
    for classname in os.listdir(targetdir):
        class_dir = targetdir + classname + "/"
        for filename in os.listdir(class_dir):
            left, _ = filename.split("_")
            action_num = int(left[-3:])
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
    traindf.to_csv(LABEL_DIR+"trainlist01.txt", header=None, index=None, sep=' ', mode='a')

    # Make testlist01
    testdf = pd.DataFrame({
        "filename": X_test,
        "label": y_test
    })
    testdf = testdf.sort_values(by=['label'])
    testdf.to_csv(LABEL_DIR+"testlist01.txt", header=None, index=None, sep=' ', mode='a')


if __name__ == '__main__':


    # Go through all the DATADIRS and filter them:
    #for datadir in DATADIRS:
    #    filter_dir(datadir)
    #    print("done filtering: " + datadir)

    # Go through all the DATADIRS and filter them:
    #for datadir in DATADIRS:
    #    no_filter_dir(datadir)
    #    print("done filtering: " + datadir)

    make_class_index(TARGET_DIR)
    print("made class index")
    
    #make_class_index_all(TARGET_DIR)
    #print("made class index for all classes")

    X, y = make_XY(TARGET_DIR)
    print("made X and y")
    make_train_test_lists(X, y, test_size=0.2)
    print("finished splitting")
