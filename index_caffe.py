import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]

blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

net.setInput(blob)
detections = net.forward()

for i in np.arange(0, detections.shape[2]):
	confidence = detections[0, 0, i, 2]
	if confidence > 0.6:
		idx = int(detections[0, 0, i, 1])
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		label = '{:0.2f}% {}'.format(confidence*100, CLASSES[idx])
		print(label)
		cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)

		y = startY - 15 if startY - 15 > 15 else startY + 15

		cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 1)

cv2.imshow("Output", image)
cv2.waitKey(0)