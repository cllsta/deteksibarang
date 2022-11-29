import cv2
thres = 0.45

cap = cv2.VideoCapture(0)
cap.set(3,1280)
###

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as x:
    classNames = x.read().rstrip('\n').split('\n')
    #####

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
    ####


net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(416,416)
net.setInputScale(1/255)

#####


while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIds,bbox)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            
            if (classNames[classId-1] == 'absensi'):
                cv2.rectangle(img,box,color=(0,0,0),thickness=2)
                cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
            else:
                cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow("Deteksi python",img)
    if cv2.waitKey(30) & 0xff == ord('q'):  
        break