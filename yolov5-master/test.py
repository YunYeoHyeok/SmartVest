import cv2
import torch
import datetime
import time
import numpy as np
import boto3
import os
from models.experimental import attempt_load
from utils.general import non_max_suppression


def s3_connection():
    try:
        s3 = boto3.client(
            service_name="s3",
            region_name="ap-northeast-2",
            aws_access_key_id="AKIA3YT2IIE7JKTAPYJX",
            aws_secret_access_key="uxpYPVCoD/lUaHsC8ErxxWIQz7CgA98By7nz7FBdV",
        )
    except Exception as e:
        print(e)
    else:
        print("s3 bucket connected!")
        return s3


def upload_to_s3():
    s3 = s3_connection()

    folder_path = "C:/yyh/save"

    # 최신 파일 찾기
    latest_file = max(
        (f.path for f in os.scandir(folder_path) if f.is_file()), key=os.path.getctime
    )

    for filename in os.listdir(folder_path):
        if filename == os.path.basename(latest_file):
            continue

        try:
            with open(os.path.join(folder_path, filename), "rb") as f:
                s3.upload_fileobj(
                    f,
                    "project-s3-data",
                    filename,
                )
        except Exception as e:
            print(e)
        else:
            os.remove(os.path.join(folder_path, filename))


def distance(bbox1, camera_size):
    x1, y1, x2, y2 = bbox1
    camera_x, camera_y = camera_size
    x_center1, y_center1 = (x1 + x2) / 2, (y1 + y2) / 2
    return np.sqrt((x_center1 - camera_x) ** 2 + (y_center1 - camera_y) ** 2)


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
            if dist < 50:  # 값을 한 번 바꿔보기
                # buz = True
                print("근접")
                print(dist)

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
