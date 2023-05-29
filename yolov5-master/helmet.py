import cv2
import torch
import RPi.GPIO as GPIO

from models.experimental import attempt_load
from utils.general import non_max_suppression

weights_path = "/home/pi/webapps/jmbest.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = attempt_load(weights_path, device=device)


img_size = 640
conf_thres = 0.25
iou_thres = 0.45

pin = 1
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(pin, GPIO.OUT)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=None)

    head_detected = False

    for *xyxy, conf, cls in reversed(pred[0]):
        label = f"{model.names[int(cls)]} {conf:.2f}"
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        if model.names[int(cls)] == "head":
            head_detected = True

    if head_detected:
        GPIO.output(pin, GPIO.HIGH)

    else:
        GPIO.output(pin, GPIO.LOW)

    # cv2.imshow("YOLOv5", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
