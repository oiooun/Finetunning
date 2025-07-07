import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
import math
import numpy as np

# 데이터 경로와 하이퍼파라미터 정의
TRAIN_DATA_DIR = './cats_and_dogs_small/train'
VALIDATION_DATA_DIR = './cats_and_dogs_small/validation'
TEST_DATA_DIR = './cats_and_dogs_small/test'

# 학습, 검증 데이터 샘플 수와 클래스 수, 이미지 크기, 배치 크기 설정
TRAIN_SAMPLES = 800 * 2
VALIDATION_SAMPLES = 400 * 2
NUM_CLASSES = 2
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 64

# 학습 데이터 증강 및 전처리 설정
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,  # ImageNet 사전 학습된 모델의 전처리 방식 적용
                                   rotation_range=20,  # 이미지 회전 범위 설정
                                   width_shift_range=0.2,  # 이미지 너비 이동 범위 설정
                                   height_shift_range=0.2,  # 이미지 높이 이동 범위 설정
                                   zoom_range=0.2,  # 이미지 확대 및 축소 범위 설정
                                   horizontal_flip=True,  # 수평 반전 허용
                                   vertical_flip=True)  # 수직 반전 허용

# 검증 데이터 전처리 설정
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# 학습 데이터 생성기 설정 (디렉토리에서 이미지 로드 및 증강)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=True,  # 학습 시 데이터를 섞어 무작위화
    seed=12345,  # 무작위 초기값 고정
    class_mode='categorical'  # 다중 클래스 분류용 라벨 형식 지정
)

# 검증 데이터 생성기 설정 (디렉토리에서 이미지 로드)
validation_generator = val_datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=False,  # 검증 시에는 데이터 순서 유지
    class_mode='categorical'  # 다중 클래스 분류용 라벨 형식 지정
)

# 사전 학습된 MobileNet 모델 불러오기 (전체 네트워크 확인을 위한 출력)
model = MobileNet()
model.summary()

# 사전 학습된 MobileNet 모델 불러오기 (마지막 출력 레이어 제외)
model1 = MobileNet(include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
model1.summary()


# MobileNet을 기반으로 커스텀 모델 생성 함수 정의
def model_maker():
    base_model = MobileNet(include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    # 마지막 20개 레이어를 제외한 나머지 레이어를 학습 불가능하도록 설정
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # 모델 구조 정의
    input1 = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    custom_model = base_model(input1)  # MobileNet의 출력을 가져옴
    custom_model = GlobalAveragePooling2D()(custom_model)  # 전역 평균 풀링을 사용하여 평탄화
    custom_model = Dense(64, activation='relu')(custom_model)  # 완전 연결 층 추가
    custom_model = Dropout(0.5)(custom_model)  # 과적합 방지를 위한 드롭아웃
    predictions = Dense(NUM_CLASSES, activation='softmax')(custom_model)  # 예측 레이어

    return Model(inputs=input1, outputs=predictions)


# 모델 인스턴스 생성 및 컴파일
model_final = model_maker()
model_final.compile(loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(1e-4),  # Adam 최적화 사용
                    metrics=['acc'])  # 정확도를 평가 지표로 설정

# 모델 학습
history = model_final.fit(
    train_generator,
    steps_per_epoch=TRAIN_SAMPLES // BATCH_SIZE,  # epoch당 스텝 수
    epochs=40,  # 학습 epoch 수
    validation_data=validation_generator,
    validation_steps=VALIDATION_SAMPLES // BATCH_SIZE  # 검증 데이터의 스텝 수
)

# 학습 과정 시각화 (손실 값 변화)
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])  # 학습 데이터 손실 값
plt.plot(history.history['val_loss'])  # 검증 데이터 손실 값
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# 테스트 데이터 전처리 및 생성기 설정
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_genarator = test_datagen.flow_from_directory(
    TEST_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical'
)

# 모델 테스트 평가
model_final.evaluate(test_genarator, steps=800 // BATCH_SIZE, verbose=1)

# 모델 저장
model_final.save('mobilenet_model.keras')
