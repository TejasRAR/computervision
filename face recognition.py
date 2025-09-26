import cv2
import numpy as np


net = cv2.dnn.readNetFromDarknet(
    'yolov3-face.cfg', 'yolov3-wider_16000.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_model.yml')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(
        frame, 1/255, (416, 416), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layernames = net.getLayerNames()
    outputlayers = [layernames[i[0] - 1]
                    for i in net.getUnconnectedOutLayers()]
    detections = net.forward(outputlayers)

    h, w = frame.shape[:2]
    boxes = []
    for det in detections:
        for i in range(len(det)):
            conf = det[i][4]
            if conf > 0.5:
                box = det[i][:4] * np.array([w, h, w, h])
                centerX, centerY, width, height = box.astype('int')
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])

    for (x, y, width, height) in boxes:

        face = frame[y:y+height, x:x+width]

        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, 'Detected Face', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
