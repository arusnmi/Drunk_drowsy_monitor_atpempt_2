# Drunk and drowsy monitor 

This Steramlit app detects drunk and drowsy driving 

# Overview of the problem:

In india during 2022, there were more than 10,000 car crashes caused by drunk driving, and over 4000 were in the state of utharakand alone, while the goverment only caught around 3,700 drunk drivers. The current system for deteting drunk driving does not work effectly with only a 37% sucess rate. 


paper proving my point: [https://sansad.in/getFile/loksabhaquestions/annex/184/AU626_Fw1ML5.pdf?source=pqals](https://sansad.in/getFile/loksabhaquestions/annex/184/AU626_Fw1ML5.pdf?source=pqals)

# Data preprocessing


1. Removal of non wanted inages: there were cropped eye images that interfered with the data given 
2. Resizing the images: i resized the images to 227x227 to keep it uniform 
3. Greyscaling images: making the images greyscaled to make detections better as greyscale
4. Creating folders: created eyes, head tilt, and mouth folders to calcluate the tresholds. 


# Model: midiapipe
Mediapipe is a model that uses a mesh model for face detection, which is nesscary for EAR, MAR and head tilt ration. 


# Metirc and accruacy: 

this is directly from the JSON file in the reposetroy: 
{
  "EAR_threshold": 0.28671591397985224,
  "EAR_metrics": {
    "precision": 0.6666666666666666,
    "recall": 0.7142857142857143,
    "f1": 0.689655172413793,
    "TP": 10,
    "FP": 5,
    "FN": 4
  },
  "MAR_threshold": 0.0,
  "MAR_metrics": {
    "precision": 0.4897959183673469,
    "recall": 1.0,
    "f1": 0.6575342465753424,
    "TP": 24,
    "FP": 25,
    "FN": 0
  },
  "HTR_threshold": 0.0,
  "HTR_metrics": {
    "precision": 0.22448979591836735,
    "recall": 1.0,
    "f1": 0.36666666666666664,
    "TP": 11,
    "FP": 38,
    "FN": 0
  }
}

# Demo images
demo images are found in the folder called "Demo images"

# Problems

There are massive problems in my code where the memory opmization is not very good, so the cloud app does not work, for the future, i will try to memory optmize it. the code runs localy as myy coumputer has more ram than the free services of steramlt cloud
