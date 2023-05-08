'''
pip install opencv-python
pip install matplotlib
pip install mediapipe
'''

import cv2 as cv
import mediapipe as mp #Google's open source framework for cv apps

#define mediapipe pose estimator object
mp_draw=mp.solutions.drawing_utils #module used to draw the skeleton on the object detected in the frame;
#includes functions for drawing different shapes and for adding text to images  

mp_pose=mp.solutions.pose #module that includes pre-trained depp learning model to estimate human poses from images/videos
#provides utilities for extracting features from poses such as position, orientation and velocity of different body parts

pose=mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) #function that creates an instance of a pose estimation model
#uses ML techniques to to identify and track a person's shoulders, wrists, elbows, etc.;
#min_detection_confidence sets threshold value for confidence in detection( high value implying that only more confident detections are accepted);
#min_tracking_confidence sets threshold value for confidence in tracking (high value implying that only more confident landmarks are considered);
#landmarks are keypoints recognised by an ML model corresponding to face, hand, etc.; represented by a set of coordinates (x,y)

stat=str()
ac=t_ac=0
ac_list=[] #to plot the accruacy graph

cap=cv.VideoCapture(0) #creates a video capture object 

#the dimensions of the output video are set using the following statements
cap.set(cv.CAP_PROP_FRAME_WIDTH,1920) 
cap.set(cv.CAP_PROP_FRAME_HEIGHT,1080)

while True:
    frame=cv.flip(cap.read()[1],1) #each frame is read from the video capture object and flipped horizontally to produce a mirrored image
    try:
        res=pose.process(cv.cvtColor(frame,cv.COLOR_BGR2RGB)) #function that takes an RGB frame and detect and estimate human pose within that frame;
        #returns a dictionary containing detected landmarks and the corresponding confidence scores;
        #converts BGR to RGB since mediapipe expects input in RGB format
        landmarks=res.pose_landmarks #returns a list of the detected landmarks detected in the given Pose object (3D points);
        #there are 33 landmarks numbered from 0 to 32 corresponding to different parts of the person's body
        if landmarks is not None:
            x_min=frame.shape[1] #stores the width of the frame
            y_min=frame.shape[0] #stores the height of the frame
            x_max=0
            y_max=0
            for landmark in landmarks.landmark: #iterating through all the landmark objects stored in landmarks object; 0 for nose, 1 for left eye and so on... 
                x,y=int(landmark.x*frame.shape[1]),int(landmark.y*frame.shape[0]) #calculate the pixel coordinates of a landmark point;
                #landmark.x and landmark.y represent normalized coordinates of the landmark with respect to the eidth and height of the frame (range between (0,0) a nd (1,1)) 
                if x<x_min:
                    x_min=x
                if x>x_max:
                    x_max=x
                if y<y_min:
                    y_min=y
                if y>y_max:
                    y_max=y
            cv.rectangle(frame,(x_min,y_min),(x_max,y_max),(255,0,0),2) #rectangle is drawn around the object which was detected before using the coordinates calculated above
            w,h=x_max-x_min,y_max-y_min #width and height of the rectangle is calculated suing coordinates calculated above
            a_ratio=w/h #aspect ratio of the object is calculated
            if a_ratio<=0.4: #if aspect ratio is less than or equal to 0.4, the person is standing
                stat='STANDING'
            elif a_ratio>2: #if aspect ratio is greater than 2, the person is lying down
                stat='LYING DOWN'
            else: #else, the person is sitting down
                stat='SITTING'
                
            '''
            #accuracy is calculated by comparing the coordinates of the landmarks of the hips, knees, and shoulders of reafch case i.e. standing up, sitting down and lying down
            if stat=='LYING DOWN':
                if landmarks.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y>landmarks.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y and landmarks.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y>landmarks.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y:
                    ac+=1
                elif landmarks.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y<landmarks.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y and landmarks.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y<landmarks.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y:
                    ac+=1
            elif stat=='SITTING':
                if landmarks.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y==landmarks.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y and landmarks.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y==landmarks.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y:
                    ac+=1
            elif stat=='STANDING':
                if landmarks.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x==landmarks.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x and landmarks.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x==landmarks.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x:
                    ac+=1    
            t_ac+=1
            ac_list.append(ac/t_ac)
            '''
            cv.putText(frame,stat,(x_min,y_min),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),2) #the posture of the person is siplayed alongside the rectangle drawn around the person
        mp_draw.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS) #the landmarks are plotted and drawn on teh person in the frame ;
        #res.pose_landmarks returns the coordinates of the landmarks detected; mp_pose.POSE_CONNECTIONS is a predefined list of landmark connections for the medipose Pose model
        cv.imshow('final',frame) #the frame is displayd to the user
    except:
        break
    if cv.waitKey(1) & 0xFF==ord('q'): #the frame is closed when the 'q' key is hit on teh keyboard
        break
cap.release()
cv.destroyAllWindows()

'''
plt.plot(ac_list)
plt.title('ACCURACY GRAPH')
plt.xlabel('FRAME')
plt.ylabel('ACCURACY')
plt.show()
'''
