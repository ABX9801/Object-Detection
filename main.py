import cv2
from darkflow.net.build import TFNet
import numpy as np

option = {
    'model':'cfg/yolo.cfg',
    'load' :'bin/yolov2.weights',
    'threshold': 0.35,
    'gpu': 1.0

}

tfnet = TFNet(option)

capture = cv2.VideoCapture('dataset/gta52.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/output1.mp4',fourcc,20.0,(640,480))

while (capture.isOpened()):
    ret,frame = capture.read()
    if ret==True:
        results = tfnet.return_predict(frame)
        for i in range(len(results)):
            tl = (results[i]['topleft']['x'],results[i]['topleft']['y'])
            br = (results[i]['bottomright']['x'],results[i]['bottomright']['y'])
            label = results[i]['label']
            frame = np.array(frame,np.uint8)
            frame= cv2.rectangle(frame,tl,br,(0,255,0),7)
            frame = cv2.putText(frame,label,tl,cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        cv2.imshow('frame',frame)
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
out.release()
cv2.destroyAllWindows()
