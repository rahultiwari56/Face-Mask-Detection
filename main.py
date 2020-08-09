import numpy as np
import cv2
from prediction import Inference

prediction = Inference()

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here

    # print(type(frame))
    prediction_res = prediction.predict(frame)
    
    print(prediction_res)
    font = cv2.FONT_HERSHEY_SIMPLEX
  
    cv2.putText(frame,prediction_res,(50,50), font, 1,(0,0,255),2,cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()