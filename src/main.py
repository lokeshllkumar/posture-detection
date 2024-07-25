import cv2 as cv
import mediapipe as mp
import matplotlib.pyplot as plt

mp_draw = mp.solutions.drawing_utils 

mp_pose = mp.solutions.pose 
pose = mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) 

stat = str()
ac = t_ac = 0
ac_list = [] 
cap = cv.VideoCapture(0) 

cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920) 
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    frame = cv.flip(cap.read()[1], 1) 
    try:
        res = pose.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB)) 
        landmarks = res.pose_landmarks 

        if landmarks is not None:
            x_min = frame.shape[1]
            y_min = frame.shape[0] 
            x_max = 0
            y_max = 0
            for landmark in landmarks.landmark: 
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) 
                
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y
            cv.rectangle(frame, (x_min,y_min), (x_max,y_max), (255, 0, 0), 2) 
            w, h = x_max - x_min, y_max - y_min 
            a_ratio = w / h 
            if a_ratio <= 0.4:
                stat = 'STANDING'
            elif a_ratio>2:
                stat = 'LYING DOWN'
            else: 
                stat = 'SITTING'
            cv.putText(frame, stat, (x_min, y_min), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2) 

            try:
                if stat == 'LYING DOWN':
                    if int(landmarks.landmark[11].y * frame.shape[0]) > int(landmarks.landmark[12].y * frame.shape[0]) and int(landmarks.landmark[23].y * frame.shape[0]) > int(landmarks.landmark[24].y * frame.shape[0]):
                        ac += 1
                    elif int(landmarks.landmark[11].y * frame.shape[0]) < int(landmarks.landmark[12].y * frame.shape[0]) and int(landmarks.landmark[23].y * frame.shape[0]) < int(landmarks.landmark[24].y * frame.shape[0]):
                        ac += 1
                elif stat == 'SITTING':
                    if int(landmarks.landmark[23].y * frame.shape[0]) - 6 <= int(landmarks.landmark[25].y * frame.shape[0]) <= int(landmarks.landmark[23].y * frame.shape[0]) + 6 and int(landmarks.landmark[24].y * frame.shape[0]) - 6 <= int(landmarks.landmark[26].y * frame.shape[0]) <= int(landmarks.landmark[24].y * frame.shape[0]) + 6:
                        ac += 1
                elif stat == 'STANDING':
                    if int(landmarks.landmark[23].x * frame.shape[1]) - 6 <= int(landmarks.landmark[25].x * frame.shape[1]) <= int(landmarks.landmark[23].x * frame.shape[1]) + 6 and int(landmarks.landmark[24].x * frame.shape[1]) - 6 <= int(landmarks.landmark[26].x * frame.shape[1]) <= int(landmarks.landmark[24].x * frame.shape[1]) + 6:
                        ac += 1
                else:
                    pass
                t_ac += 1
                ac_list.append(ac / t_ac)
            except:
                pass
        mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS) 
        cv.imshow('final', frame)
    except:
        break
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

plt.plot(ac_list)
plt.title('ACCURACY GRAPH')
plt.xlabel('FRAME')
plt.ylabel('ACCURACY')
plt.show()
