import cv2

cap=cv2.VideoCapture(0)
ct=0
bg=cv2.createBackgroundSubtractorMOG2()
stat=None

while(True):
    frame=cv2.flip(cap.read()[1],1)
    try:
        fgmask=bg.apply(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        #thresh=cv2.threshold(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),150,255,cv2.THRESH_BINARY)[1]
        contours=cv2.findContours(image=fgmask,mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_NONE)[0]
        if contours:
            ar=[]
            for contour in contours:
                ar.append(cv2.contourArea(contour))
            M=cv2.moments(contours[ar.index(max(ar,default=0))])
            x,y,w,h=cv2.boundingRect(contours[ar.index(max(ar,default=0))])
            cv2.drawContours(fgmask,[contours[ar.index(max(ar,default=0))]],0,(255,255,255),3,maxLevel=0)
            if h<w:
                ct+=1
            if ct>200:
                stat='FALLEN'
                cv2.putText(frame,stat,(x,y),cv2.FONT_HERSHEY_PLAIN,0.5,(255,255,255),2)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            if h>w:
                stat=''
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow('bruh',frame)    
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    except Exception as e:
        break
cap.release()
#cap.destroyAllWindows()
