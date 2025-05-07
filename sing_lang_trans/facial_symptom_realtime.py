import cv2
import numpy as np
import time
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
from tflite_runtime.interpreter import Interpreter

# 설정
actions = [
    "답답하다", "땀난다", "베였다", "부었다", "불편하다",
    "아프다", "어지럽다", "열난다", "지속된다", "피곤하다"
]
seq_length = 10
expected_feature_dim = 55  # 학습 시 사용한 피처 수

USED_FACE_INDEXES = list(dict.fromkeys([
    33, 133, 362, 263, 1, 61, 291, 199, 429, 152,
    234, 454, 138, 172, 136, 215, 177, 398, 367,
    10, 66, 105, 107, 55, 285, 336, 296, 300,
    447, 123, 351, 168, 206, 195, 3, 51, 281, 425,
    57, 43, 214, 441, 244, 112, 226, 31, 54, 35,
    143, 366, 70, 63
]))

# 폰트 로드
try:
    font = ImageFont.truetype("fonts/HMKMMAG.TTF", 40)
except:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/AppleGothic.ttf", 40)

def draw_text_on_image(img, text):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    w, h = img_pil.size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    position = ((w - text_w) / 2, h - text_h - 30)

    margin = 10
    draw.rectangle(
        [position[0] - margin, position[1] - margin,
         position[0] + text_w + margin, position[1] + text_h + margin],
        fill=(0, 0, 0, 180)
    )
    draw.text(position, text, font=font, fill=(255, 255, 255))
    return np.array(img_pil)

def load_tflite_model(path):
    interpreter = Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def predict_action(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 입력 데이터 (1, 10, 55)로 고정
    input_data = np.array(input_data, dtype=np.float32).reshape(1, seq_length, expected_feature_dim)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    return interpreter.get_tensor(output_details[0]['index'])[0]

def main():
    interpreter = load_tflite_model('models/multi_face_expression_classifier_vlow.tflite')
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 최소화

    seq = []
    action_seq = []
    last_action = None
    prev_time = time.time()

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            coords = []
            for i in USED_FACE_INDEXES[:expected_feature_dim]:
                if i < len(face.landmark):
                    lm = face.landmark[i]
                    coords.extend([lm.x, lm.y])
                else:
                    coords.extend([0.0, 0.0])

            # 피처 수 불일치 보정
            if len(coords) != expected_feature_dim * 2:
                print(f"[경고] 피처 수 불일치: {len(coords)}")
                coords.extend([0.0] * (expected_feature_dim * 2 - len(coords)))

            input_feature = np.array(coords, dtype=np.float32)

            # 시퀀스 추가
            seq.append(input_feature)
            if len(seq) > seq_length:
                seq.pop(0)  # 오래된 데이터 삭제하여 메모리 절약

            # (1, 10, 55)로 입력 변환
            if len(seq) == seq_length:
                input_data = np.array(seq, dtype=np.float32).reshape(1, 10, 55)
                y_pred = predict_action(interpreter, input_data)

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]
                if conf < 0.9:
                    continue

                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) >= 3 and action_seq[-3:] == [action] * 3:
                    if last_action != action:
                        last_action = action
                        img = draw_text_on_image(img, action)

        # 프레임 제한 (최대 10FPS)
        current_time = time.time()
        elapsed_time = current_time - prev_time
        if elapsed_time < 0.1:
            time.sleep(0.1 - elapsed_time)
        prev_time = current_time

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()