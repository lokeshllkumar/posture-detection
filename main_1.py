'''
pip install opencv-python
numpy is preinstalled 
pip install matplotlib
pip install -U scikit-learn
'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cap=cv.VideoCapture(0)
ct=0
bg=cv.createBackgroundSubtractorMOG2()
stat=None

loss=[]

t_label,p_label=[],[]

num_c,ac_list=0,[]

while True:
    frame=cv.flip(cap.read()[1],1)
    fgmask=bg.apply(cv.cvtColor(frame,cv.COLOR_BGR2GRAY))
    contours=cv.findContours(image=fgmask,mode=cv.RETR_TREE,method=cv.CHAIN_APPROX_NONE)[0]

    thresh=cv.threshold(cv.cvtColor(frame,cv.COLOR_BGR2GRAY),127,255,cv.THRESH_BINARY)[1]
    kernel=cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
    op=cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel)
    cl=cv.morphologyEx(op,cv.MORPH_CLOSE,kernel)
    sk=np.zeros(cl.shape,np.uint8)
    el=cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
    stat=True
    while stat:
        eroded=cv.erode(cl,el)
        temp=cv.dilate(eroded,el)
        temp=cv.subtract(cl,temp)
        sk=cv.bitwise_or(sk,temp)
        if cv.countNonZero(cl:=eroded.copy())==0:
            stat=False
    cv.imshow('skel',sk)
    if contours:
        ar=[]
        for contour in contours:
            ar.append(cv.contourArea(contour))
        M=cv.moments(contours[ar.index(max(ar,default=0))])
        x,y,w,h=cv.boundingRect(contours[ar.index(max(ar,default=0))])
        cv.drawContours(fgmask,[contours[ar.index(max(ar,default=0))]],0,(255,255,255),1,maxLevel=0)
        a_ratio=w/h
        color=(0,255,0)
        if 0.9<=a_ratio<=1.1:
            stat='STANDING'
            loss.append(0)
            t_label.append('STANDING')
            num_c+=1
        elif a_ratio>1.1:
            stat='LYING'
            color=(0,0,255)
            loss.append(1)
            t_label.append('LYING')
        else:
            stat='SITTING'
            loss.append(2)
            t_label.append('SITTING')
            num_c+=1
        p_label.append(stat)
    cv.rectangle(frame,(x,y),(x+w,y+h),color,2)
    cv.putText(frame,stat,(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),2)
    cv.imshow('bruh',frame)    
    if cv.waitKey(1) & 0xFF==ord('q'):
        break
    ct+=1
    ac_list.append(num_c/ct)
cap.release()
cv.destroyAllWindows()

plt.plot(loss)
plt.title('loss graph')
plt.xlabel('frame')
plt.ylabel('loss')
plt.show()

plt.plot(ac_list)
plt.title('accuracy graph')
plt.xlabel('frame')
plt.ylabel('accuracy')
plt.show()

c_mat=confusion_matrix(t_label,p_label,labels=['STANDING','LYING','SITTING'])
print("confusion matrix:")
print(c_mat)
