import requests
import os

# 서버 URL
url = "http://127.0.0.1:8000/predict"

# 테스트할 폴더 경로
base_folder_path = './test'  # test 폴더 경로

# 하위 폴더 순회
for category in ['cats', 'dogs']:
    folder_path = os.path.join(base_folder_path, category)

    print(f"\nProcessing category: {category}")
    print("=" * 40)

    # 각 이미지 파일에 대해 예측 요청 보내기
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        # 이미지 파일이 맞는지 확인 (예: jpg, png 확장자)
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            with open(image_path, "rb") as image_file:
                response = requests.post(url, files={"file": image_file})

            # 결과 출력
            print(f"Image: {image_name}")
            print("Category:", category)
            print("Prediction:", response.json())
            print("-" * 30)
