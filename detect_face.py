import cv2
import numpy as np
from PIL import Image
# from torch_mtcnn import detect_faces
from facenet_pytorch import MTCNN
import itertools 
from prediction import Inference

prediction = Inference()

mtcnn = MTCNN(margin=20,keep_all=True,device='cpu')

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    prediction_res = prediction.predict(frame)

    boxes,_ = mtcnn.detect(frame)
    print(boxes)

    try:
        for i in range(len(boxes)):
            if prediction_res=='With Mask':
                cv2.rectangle(frame,
                        (boxes[i][0], boxes[i][1]),
                        (boxes[i][2], boxes[i][3]),
                        (0,255,0),
                        2)
            else:
                cv2.rectangle(frame,
                    (boxes[i][0], boxes[i][1]),
                    (boxes[i][2], boxes[i][3]),
                    (255,0,0),
                    2)

            cv2.imshow('frame',frame)
            
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(frame,prediction_res,(50,50), font, 1,(0,0,255),2,cv2.LINE_AA)

    # Display the resulting frame
    except:
        cv2.imshow('frame',frame)

    # # Single image
    # mtcnn(frame, save_path='single_image.jpg');

    # # Batch
    # save_paths = [f'image_{i}.jpg' for i in range(len(frames))]
    # mtcnn(frames, save_path=save_paths);
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()