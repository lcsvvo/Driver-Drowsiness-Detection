# 요약
# 이미지 1장 -> 얼굴 찾기 -> 하품 판정, 눈 영역 잘라내기 -> 눈 감김 판정 -> "하품"/"눈 감음" 출력
# 특징
# 눈 모델(Eye Model)과 하품 모델(Yawn Model)을 따로 사용

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle


class Drowsiness_Detector: # 졸음 감지를 담당하는 기능들을 하나로 묶어놓은 상자
    def __init__(self, list_models):
        """
        :param list_models: a list containing the path of two models

        """
        self.detector = MTCNN() # 얼굴 검출기, MTCHH: 얼굴의 위치 좌표를 알려줌(x, y, width, height)
        self.models = self.get_models_ready() # 1. 학습된 눈, 하품 모델 불러옴
        self.list_models = list_models

    def get_models_ready(self): # 2. 이미 학습된 CNN 불러옴
        eye_model = tf.keras.models.load_model(self.list_models[0]) # 눈 모델
        yawn_model = tf.keras.models.load_model(self.list_models[1]) # 하품 모델
        return eye_model, yawn_model

    @staticmethod
    def sharpen(img): # 3. 이미지 선명하게 만듦, 눈 crop -> resize -> sharpen -> CNN
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
        return img

    @staticmethod
    def return_boxes(result_list): # 얼굴에서 눈 위치 추정
        print(result_list)
        for result in result_list:
            x, y, width, height = result['box'] # 1. 얼굴의 위치와 크기 가져옴
            height = height / 2
            # 2. 왼쪽, 오른쪽 눈 영역을 사각형으로 만듦
            eye1 = Rectangle((x, y), width / 2, height, fill=False, color='red')
            eye2 = Rectangle(((x + (width / 2)), y), width / 2, height, fill=False, color='red')
            return eye1, eye2

    def extract_eye(self, image, img, image_matrix_given=True): # 얼굴 이미지에서 눈만 잘라냄
        # 입력 이미지 -> MTCNN으로 얼굴 검출  -> 256x256 resize -> sharpen -> grayscale
        if not image_matrix_given:
            image = pyplot.imread(img)
        img = self.return_boxes(self.detector.detect_faces(image))
        if img is None:
            return False
        left = img[0]
        x = int(left.xy[0])
        y = int(left.xy[1])
        xw = int(left.xy[0] + left._width)
        yh = int(left.xy[1] + left._height)
        l_eye = image[y:yh, x:xw] # 1. 눈 영역 자름
        l_eye = cv2.resize(l_eye, (256, 256)) # 2. 사이즈를 256x256로 맞춤
        l_eye = self.sharpen(l_eye) # 3. 선명하게 만듦
        l_eye = tf.image.rgb_to_grayscale(l_eye) # 4. 흑백 이미지로 바꿈

        right = img[1] # 위 과정 반복
        x = int(right.xy[0])
        y = int(right.xy[1])
        xw = int(right.xy[0] + right._width)
        yh = int(right.xy[1] + right._height)
        r_eye = image[y:yh, x:xw]
        r_eye = cv2.resize(r_eye, (256, 256))
        r_eye = self.sharpen(r_eye)
        r_eye = tf.image.rgb_to_grayscale(r_eye)

        return (l_eye, r_eye), (left, right) # 실제 눈 이미지 2개, 눈 위치 정보 반환

    @staticmethod
    def produce_eye_output(model, inputs): # 눈 감김 판단
        outputs = model(inputs)
        print(outputs)
        if tf.argmax(outputs[0]) == 0 or tf.argmax(outputs[1]) == 0:
        # 왼쪽 눈 또는 오른쪽 눈이 특정 클래스(0)로 분류되면 True로 처리
        # 주의! 클래스 0이 "Closed"인지 "Open" 인지는 확인해봐야함
            return True
        else:
            return False

    def eye_classification(self, l_eye, r_eye, model): # 양쪽 눈을 모델에 넣음
        l_eye = tf.expand_dims(l_eye, axis=0) # 1. 배치 차원 추가
        r_eye = tf.expand_dims(r_eye, axis=0)
        print(l_eye.shape, r_eye.shape)
        inputs = tf.concat([l_eye, r_eye], axis=0) # 2. 왼쪽 + 오른쪽 눈을 하나의 입력 묶음으로 합침
        print(inputs.shape)
        output = self.produce_eye_output(model, inputs) # 3. CNN에 입력
        if output:
            return True

    @staticmethod
    def yawn_detection(face, model): # 하품 판정
        inputs = cv2.resize(face, (256, 256)) # 1. 얼굴 이미지를 256x256로 설정
        inputs = tf.expand_dims(inputs, axis=0) # 2. 배치 차원 추가
        output = model(inputs) # 3. 하품 CNN에 입력
        print(output)
        output = np.argmax(np.round(np.squeeze(output))) # 4. 모델의 출력값 중 가장 큰 클래스 번호를 가져옴
        return output

    def drowsiness(self, image): # 전체 코드 연결
        yawn = self.yawn_detection(image, self.models[1]) # 하품 판정
        eyes = self.extract_eye(image, "") # 눈 찾기
        markings = False 
        if not eyes:
            eye = False
        else:
            eye_class = eyes[0]
            markings = eyes[1]
            eye = self.eye_classification(eye_class[0], eye_class[1], self.models[0])
        if yawn == 0: # 하품이면 경고
            print("Danger Sign : Yawn") # add any alarm sounds
        if eye: # 눈 감김이면 경고
            print("Danger Sign : Eyes Closed")  # add any alarm sounds
        if markings is not False:
            return markings

#① drowsiness()
#→ 전체 과정 실행: 하품 판정 → 눈 추출 → 눈 감김 판정 → 경고 출력

#② extract_eye()
#→ MTCNN으로 얼굴을 찾고 양쪽 눈 영역 추출

#③ eye_classification() / produce_eye_output()
#→ 양쪽 눈을 Eye CNN에 넣어 눈 감김 여부 판정

#④ yawn_detection()
#→ 얼굴을 Yawn CNN에 넣어 하품 여부 판정

#⑤ get_models_ready()
#→ 학습된 Eye CNN·Yawn CNN 모델 불러오기