import cv2

# Loading face detection model
proto_path = 'face_detector_model/deploy.prototxt'
model_path = 'face_detector_model/res10_300x300_ssd_iter_140000.caffemodel'
detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)


def detect_face(frm):
    # grab the frame dimensions and convert it to a blob
    # blob is used to preprocess image to be easy to read for NN
    # basically, it does mean subtraction and scaling
    # (104.0, 177.0, 123.0) is the mean of image in FaceNet
    blob = cv2.dnn.blobFromImage(cv2.resize(frm, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # pass the blob through the network 
    # and obtain the detections and predictions
    detector_net.setInput(blob)
    detections = detector_net.forward()
    return detections
