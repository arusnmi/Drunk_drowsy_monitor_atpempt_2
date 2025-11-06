<img width="704" height="232" alt="image" src="https://github.com/user-attachments/assets/9f08533e-1676-445f-a6db-b30f67ef9397" /># Drunk and drowsy monitor 
https://drunkdrowsymonitoratpempt2-cwsldbhukaqizpt8eghmjk.streamlit.app

This Steramlit app detects drunk and drowsy driving. It wouldbe targeted against all drivers. The system would use the live webcam that would be placed on every car, and if the indacators that their eyes close, or they are yawning, it would detect drowsyness, and if they are not looking at the road, they are distracted, I used head tilt as a metric. 

# Main research findings:

In india during 2022, there were more than 10,000 car crashes caused by drunk driving, and over 4000 were fatal crashes while the goverment only caught around over 28,000 drunk drivers(note, this number may be off as this is covering only pune, dehli, and benglerue). The current human system wher epolice officers would top a car and check the car with abrethlyzer is very inefficent and misses a lot of drunk drivers. 

# Refrences
Transcript from the lok shaba discussion in 2022 proving my point: [https://sansad.in/getFile/loksabhaquestions/annex/184/AU626_Fw1ML5.pdf?source=pqals](https://sansad.in/getFile/loksabhaquestions/annex/184/AU626_Fw1ML5.pdf?source=pqals) 

Figures for catching drunk drivers:[https://www.thehindu.com/news/cities/Delhi/over-12000-booked-for-drunken-driving-in-first-half-of-2024-27-increase-from-last-year-says-delhi-traffic-police/article68375250.ece#:~:text=According%20to%20a%20Hindu%20Bureau%20article%2C%20the,**Last%20year**%209%2C837%20violators%20booked%20for%20DUI](https://www.thehindu.com/news/cities/Delhi/over-12000-booked-for-drunken-driving-in-first-half-of-2024-27-increase-from-last-year-says-delhi-traffic-police/article68375250.ece#:~:text=According%20to%20a%20Hindu%20Bureau%20article%2C%20the,**Last%20year**%209%2C837%20violators%20booked%20for%20DUI)

[https://timesofindia.indiatimes.com/city/mumbai/increase-in-drunk-driving-cases-in-mumbai-in-2024-compared-to-2023/articleshow/111270382.cms#:~:text=3.8k%20drunk%20driving%20cases,traffic%20on%20roads%20is%20lower.](https://timesofindia.indiatimes.com/city/mumbai/increase-in-drunk-driving-cases-in-mumbai-in-2024-compared-to-2023/articleshow/111270382.cms#:~:text=3.8k%20drunk%20driving%20cases,traffic%20on%20roads%20is%20lower.)

[https://www.hindustantimes.com/cities/pune-news/traffic-police-take-action-against-5k-violators-under-drink-drive-campaign-in-2024-101735500854538.html](https://www.hindustantimes.com/cities/pune-news/traffic-police-take-action-against-5k-violators-under-drink-drive-campaign-in-2024-101735500854538.html)
# Data preprocessing and prepration


1. Removal of non wanted inages: there were cropped eye images that interfered with the data given 
2. Resizing the images: i resized the images to 227x227 to keep it uniform 
3. Greyscaling images: making the images greyscaled to make detections better as greyscale
4. Creating folders: created eyes, head tilt, and mouth folders to calcluate the tresholds.
5. Splitting thr data: using the 70, 15 and 15 rule for splitting the data, 70% training, 15% test and valadation


# Model usage and training prameaters
Mediapipe is a model that uses a mesh model for face detection, which is nesscary for EAR, MAR and head tilt ration. I used the in built Facemesh model that mediapipe uses, and to calcluate the EAR, MAR, and head tilt, Iwould take every training image, and calcluate the metrics, then i would save thise metrics in a csv file and then calcluate the avrage values to get each value. using the Facemesh model i use the points placed for eyes to calcluate the EAR, and the points for th mouth to calcluate MAR. Then i used the points of the left cheek and right cheek to mesure head tilt. i then based on the avrages , i fine tuened the numbers to calcluate the final resualts given th the metrics. i would use the steramlit interfacce to then mesure the values for EAR and MAR to fine tune them futher. 

all the formulas are given in the folder "code calcluations"



# Metirc and accruacy: 

Based on the values in the JSON file given above, the metrics that are calcluated are 0.2 for the EAR, 0.6 for the MAR, and 0.5 for the Head tilt. the accruacy of these ratios are low, but they still work. 

# Project demo images
 Demo images for showing the project working are found in the folder called "Demo images"


# Dataset:

[https://drive.google.com/drive/folders/1lpSmeOI-rY1MU-9nPRYwP7CWWVDckKbL?usp=sharing](https://drive.google.com/drive/folders/1lpSmeOI-rY1MU-9nPRYwP7CWWVDckKbL?usp=sharing)

# Memory optmization

When i completed my code i found that it only worked on my laptop and not on the cloud because it was not memory efficant, so i did some emeory optmizition. my code not only calcluates every 2nd frame, so half of the frames are not going through which means that it is memory effeciant. i then removed fetures such as the upload a video function to solly use live webcam, i also used steramlit Webrtc to change the resoloution of the camera to allow for better memory optmization. 

# Plans for the future

If i get time later, i will try to optmize the code so that it runs smother, i had chalenges in making it work on steramlit cloud, as the app was not optmized then. i would like toi add an alarm function to the app as well. the camera was also grainy as i was using steramlit webrtc, i plan to change that as well.

