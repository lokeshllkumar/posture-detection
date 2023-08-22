#to draw contours over objects in frame

import cv2

cap=cv2.VideoCapture(0)

while(True):
    frame=cv2.flip(cap.read()[1],1)
    thresh=cv2.threshold(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),150,255,cv2.THRESH_BINARY)[1]
    contours,hierarchy=cv2.findContours(image=thresh,mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image=frame,contours=contours,contourIdx=-1,color=(0,50,255),thickness=2,lineType=cv2.LINE_AA)
    cv2.imshow('bruh',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cap.destroyAllWindows()
