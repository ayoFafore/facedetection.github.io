import cv2
import numpy as np
import imutils
from face_detection_cv import detect_face
from object_detection_cv import detect_object

def get_detection(frame, face_radio, object_radio, all_objects, object_name):
    """
    This function get parameters as 
    Frame: Input Image
    face_radio: Yes or NO, Enabling/Disabling Face Detection model
    object_radio: YES or NO, Enabling/Disabling Object Detection model
    all_objects: YES or NO, detecting all objects or a specific one
    object_name: Name of object in case selected one.


    Returns:
    OutPut Frame: Image with bounding box around the faces and objects
    faces: Face locations
    objects: Object locations and names
    """
    frame = imutils.resize(frame, width=800)
    (h, w) = frame.shape[:2]
    if object_radio == "Yes":
        idxs, boxes, confidences, classIDs, LABELS, COLORS = detect_object(frame) # Calling object detection model
    if face_radio == "Yes":
        detections = detect_face(frame) # Calling face detection model
    faces = dict()
    objects = dict()
    if face_radio == "Yes":
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e. probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
            
            # filter out weak detections
            if confidence > 0.5:
                # compute the (x,y) coordinates of the bounding box
                # for the face and extract the face ROI
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')
                
                # expand the bounding box a bit
                # (from experiment, the model works better this way)
                # and ensure that the bounding box does not fall outside of the frame
                startX = max(0, startX-20)
                startY = max(0, startY-20)
                endX = min(w, endX+20)
                endY = min(h, endY+20)
                faces[i] = (startX, startY, endX, endY)
                # Drawing bounding box around the faces
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 4)

    if object_radio == "Yes":
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[classIDs[i]]]
                if all_objects == "No": # Checking if all objects or single one
                    if LABELS[classIDs[i]] != object_name:
                        continue
                # Drawing bounding box around the objects
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                objects[text + str(i)] = (x, y, w, h)
                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, faces, objects