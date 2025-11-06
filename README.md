# Drunk and drowsy monitor 
https://drunkdrowsymonitoratpempt2-cwsldbhukaqizpt8eghmjk.streamlit.app

This Steramlit app detects drunk and drowsy driving. It wouldbe targeted against all drivers. The system would use the live webcam that would be placed on every car, and if the indacators that their eyes close, or they are yawning, it would detect drowsyness, and if they are not looking at the road, they are distracted, I used head tilt as a metric. 

# Overview of the problem:

In india during 2022, there were more than 10,000 car crashes caused by drunk driving, and over 4000 were fatal crashes while the goverment only caught around over 28,000 drunk drivers(note, this number may be off as this is covering only pune, dehli, and benglerue). The current human system wher epolice officers would top a car and check the car with abrethlyzer is very inefficent and misses a lot of drunk drivers. 


Transcript from the lok shaba discussion in 2022 proving my point: [https://sansad.in/getFile/loksabhaquestions/annex/184/AU626_Fw1ML5.pdf?source=pqals](https://sansad.in/getFile/loksabhaquestions/annex/184/AU626_Fw1ML5.pdf?source=pqals) 

Figures for catching drunk drivers:[https://www.thehindu.com/news/cities/Delhi/over-12000-booked-for-drunken-driving-in-first-half-of-2024-27-increase-from-last-year-says-delhi-traffic-police/article68375250.ece#:~:text=According%20to%20a%20Hindu%20Bureau%20article%2C%20the,**Last%20year**%209%2C837%20violators%20booked%20for%20DUI](https://www.thehindu.com/news/cities/Delhi/over-12000-booked-for-drunken-driving-in-first-half-of-2024-27-increase-from-last-year-says-delhi-traffic-police/article68375250.ece#:~:text=According%20to%20a%20Hindu%20Bureau%20article%2C%20the,**Last%20year**%209%2C837%20violators%20booked%20for%20DUI)

[https://timesofindia.indiatimes.com/city/mumbai/increase-in-drunk-driving-cases-in-mumbai-in-2024-compared-to-2023/articleshow/111270382.cms#:~:text=3.8k%20drunk%20driving%20cases,traffic%20on%20roads%20is%20lower.](https://timesofindia.indiatimes.com/city/mumbai/increase-in-drunk-driving-cases-in-mumbai-in-2024-compared-to-2023/articleshow/111270382.cms#:~:text=3.8k%20drunk%20driving%20cases,traffic%20on%20roads%20is%20lower.)

[https://www.hindustantimes.com/cities/pune-news/traffic-police-take-action-against-5k-violators-under-drink-drive-campaign-in-2024-101735500854538.html](https://www.hindustantimes.com/cities/pune-news/traffic-police-take-action-against-5k-violators-under-drink-drive-campaign-in-2024-101735500854538.html)
# Data preprocessing


1. Removal of non wanted inages: there were cropped eye images that interfered with the data given 
2. Resizing the images: i resized the images to 227x227 to keep it uniform 
3. Greyscaling images: making the images greyscaled to make detections better as greyscale
4. Creating folders: created eyes, head tilt, and mouth folders to calcluate the tresholds.
5. Splitting thr data: using the 70, 15 and 15 rule for splitting the data, 70% training, 15% test and valadation


# Model: midiapipe
Mediapipe is a model that uses a mesh model for face detection, which is nesscary for EAR, MAR and head tilt ration. I used the in built Facemesh model that mediapipe uses, and to calcluate the EAR, MAR, and head tilt, Iwould take every training image, and calcluate the metrics, then i would save thise metrics in a csv file and then calcluate the avrage values to get each value.  


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

# Project demo images
 Demo images for showing the project working are found in the folder called "Demo images"


# Dataset:

[https://drive.google.com/drive/folders/1lpSmeOI-rY1MU-9nPRYwP7CWWVDckKbL?usp=sharing](https://drive.google.com/drive/folders/1lpSmeOI-rY1MU-9nPRYwP7CWWVDckKbL?usp=sharing)

# Plans for the future

If i get time later, i will try to optmize the code so that 
