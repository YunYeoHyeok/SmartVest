import tensorflow as tf
import os
import cv2
import numpy as np
from PIL import Image


def read_n_preprocess(image):
    image = cv2.resize(image, (224, 224))  # MobileNetV2는 224 224이 대중적으로 사용
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR->RGB로 변환
    image = image.astype(np.float32) / 255.0  # 이미지 0~255범위의 픽셀값을 0~1로 정규화
    return image


# 경로 바꿔야 함! 코랩에서 돌렸음


def create_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False
        # MobileNetV2의 장점인 임베디드 기기에 사용하기에 최적화됨
    )

    for layer in base_model.layers:
        layer.trainable = False

    model = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            # MoblieNet2D에서 추출한 특징을 입력받아, 공간 차원을
            # 평균화 하여 3D텐서를 1D텐서로 변환
            tf.keras.layers.Dense(256, activation="relu"),
            # 256은 뉴런의 수를 의미하며 관행적으로 2의 제곱을 사용
            tf.keras.layers.BatchNormalization(),
            # 입력 데이터를 정규화하여 학습 안정성을 높임
            tf.keras.layers.Dropout(0.5),
            # 과적합(Overfitting)을 방지하기 위해 일부 뉴런을 비활성화
            tf.keras.layers.Dense(1, activation="sigmoid"),
            # Sigmoid 0과 1로 받아 최총적으로 이진 분류 결과를 출력
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


train_data_dir = "/content/gdrive/My Drive/Colab Notebooks/train_data"
val_data_dir = "/content/gdrive/My Drive/Colab Notebooks/val_data"

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=read_n_preprocess,
    horizontal_flip=True,  # 수평으로 이미지를 뒤집음
    vertical_flip=True,  # 무작위로 세로 방향으로 이미지를 뒤집음
    brightness_range=[0.7, 1.3],  # 이미지 밝기를 70%에서 130%이내에서 무작위 조정
    rotation_range=20,  # 30도 내에서 무작위로 이미지 회전
    zoom_range=0.1,  # 0.9~1.1배 사이로 확대, 축소
    width_shift_range=0.1,  # 전체넓이 x 0.1 범위내에서 무작위 수직이동
    height_shift_range=0.1,  # 수평이동
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=read_n_preprocess
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    classes=["nonperson", "person"],
    shuffle=True,
)
val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    classes=["nonperson", "person"],
    shuffle=True,
)

model = create_model()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
# 모델이 과적합을 방지하기위해 사용
final = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    callbacks=[early_stopping],
)
# Optimizer 모델이 가중치를 업데이트하는 최적화 알고리즘 lr은 학습률을 의미함
# Loss 이진교차엔트로피(binary crossentropy) 손실 함수를 사용, 예측과 실제 값
# 사이의 차이를 계산하여 모델이 얼마나 정확한 예측을 하는지 평가
# Metrics 모델 학습하는 동안 평가지표
# Epochs 학습량
model.save("/content/gdrive/My Drive/Colab Notebooks/face_detection_model1.h5")
