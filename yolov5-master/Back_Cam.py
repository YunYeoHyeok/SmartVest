import cv2
import torch
import datetime
import time

# import RPi.GPIO as GPIO
from distance import distance
from Api_SendtoAWS import Send_to_AWS
from db_info import db_connect, db_info
from models.experimental import attempt_load
from utils.general import non_max_suppression
from S3 import upload_to_s3


db = db_connect(db_info)
cur = db.cursor()

# GPIO.setmode(GPIO.BCM)
# GPIO.setwarnings(False)
# GPIO.setup(16, GPIO.OUT)

# p = GPIO.PWM(16, 1)

Frq = [262]

weights_path = "C:/yyh/pythonOpenCV/Final_Project/yolov5-master/best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load(weights_path, device=device)


img_size = 640
conf_thres = 0.50  # 정확도
iou_thres = 0.45

fourcc = cv2.VideoWriter_fourcc(*"AVC1")
save_time = 80  # 초 단위
s_frame = 5  # 1초당 프레임 수 30x10 = 300

cap = cv2.VideoCapture(0)

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"C:/yyh/save/{current_time}.mp4"
out = cv2.VideoWriter(filename, fourcc, s_frame, (640, 480))

start_time = datetime.datetime.now()

bbox_list = []
vest = input("안전조끼 번호를 입력하세요 : ")
buz = False

camera_size = (320, 240)

while True:
    ret, frame = cap.read()

    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=None)

    if len(pred) > 0:
        for *xyxy, conf, cls in reversed(pred[0]):
            label = f"{model.names[int(cls)]} {conf:.2f}"
            x1, y1, x2, y2 = map(int, xyxy)
            # bbox 좌표값을 img_size 이미지 기준으로 변환
            x1 = int(x1 * frame.shape[1] / img_size)
            y1 = int(y1 * frame.shape[0] / img_size)
            x2 = int(x2 * frame.shape[1] / img_size)
            y2 = int(y2 * frame.shape[0] / img_size)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )

            bbox_list.append((x1, y1, x2, y2))

        for i in range(len(bbox_list)):
            bbox1 = bbox_list[i]
            dist = distance(bbox1, camera_size)

            # 거리가 가까워지면 "근접" 출력
            if dist < 300:  # 값을 한 번 바꿔보기
                # buz = True
                print("근접")
            # print(dist)
    if buz == True:
        # p.start(50)
        # print("buz")
        for i in range(0, 1):
            # p.ChangeFrequency(Frq[i])
            time.sleep(0.5)
            cur.execute(
                f"insert into Buzzer(Buz, BuzTime , BuzReason, vest) values('Buz ON', now(), 'Danger!' , ('{vest}'))"
            )
            db.commit()
            Send_to_AWS("Buzzer")
        buz = False

    elif buz == False:
        print("no buz")
        # p.stop()

    bbox_list = []

    out.write(frame)

    if (datetime.datetime.now() - start_time).seconds >= save_time:
        out.release()
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"C:/yyh/save/{current_time}.mp4"
        out = cv2.VideoWriter(filename, fourcc, s_frame, (640, 480))
        try:
            upload_to_s3()
        except Exception as e:
            print(e)
        start_time = datetime.datetime.now()

    cv2.imshow("YOLOv5", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
