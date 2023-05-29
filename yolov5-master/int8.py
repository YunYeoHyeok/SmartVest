import torch
from models.experimental import attempt_load

# 커스텀 YOLOv5 모델을 로드합니다.
weights = "C:/yyh/pythonopencv/final_project/yolov5-master/best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load(weights, device=device)  # weights파일 로드
stride = max(int(model.stride.max()), 32)  # 모델에서 사용하는 가장 큰 stride 값을 확인
input_size = (416, 416)  # 입력 크기
image = torch.zeros((1, 3, *input_size)).to(device)  # 모델의 입력 크기와 일치하는 텐서 생성
_ = model(image)  # 모델을 실행하여 출력 크기를 조정
model = model.to(device).eval()  # 모델을 GPU로 올리고 평가 모드로 변경
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# 양자화합니다.
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Conv2d}, dtype=torch.qint8
)

# TorchScript 형식으로 변환합니다.
scripted_model = torch.jit.trace(
    quantized_model, torch.randn(1, 3, *input_size).to(device)
)

# 모델을 저장합니다.
scripted_model.save(
    "C:/yyh/pythonopencv/final_project/yolov5-master/quantized_custom_yolov5.pt"
)
