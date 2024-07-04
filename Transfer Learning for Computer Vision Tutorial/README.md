# 1.Load Data

## 이미지 변환
```py
from torchvision import transforms

data_transforms = {
    'train': transforms.Compose([

        # 이미지를 무작위로 자르고 크기를 224x224 픽셀로 조정합니다
        transforms.RandomResizedCrop(224),
        # 이미지를 50% 확률로 좌우 반전합니다
        transforms.RandomHorizontalFlip(),
        # 이미지를 PyTorch 텐서로 변환합니다
        transforms.ToTensor(),
        # 각 채널에 대해 평균과 표준 편차 값으로 이미지를 정규화합니다
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # 이미지의 짧은 측면을 256 픽셀로 크기 조정합니다
        transforms.Resize(256),
        # 이미지의 중심을 224x224 픽셀로 자릅니다
        transforms.CenterCrop(224),
        # 이미지를 PyTorch 텐서로 변환합니다
        transforms.ToTensor(),
        # 각 채널에 대해 평균과 표준 편차 값으로 이미지를 정규화합니다
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

```   
  
- **transforms.RandomResizedCrop(x)**   
    - 이미지를 무작위로 자르고 크기를 x*x 픽셀로 조정  
    - scale, ratio 인수를 사용해 자를 크기 비율의 점위와 가로 세로 비율의 범위도 조율 가능  
    ```py
    transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(0.75, 1.33))
    ```  
<br/>   

- **transforms.Resize(x)**  
    - 이미지의 크기를 지정된 크기로 조정  
    - 이미지의 짧은 쪽을 기준으로 크기를 조정하며, 비율을 유지  
<br/>   

- **_RandomResizedCrop Vs. Resize_**  
    - RandomResizedCrop의 경우 크롭 위치와 크기가 무작위로 선택되어 **데이터 증강 효과** 존재  
        - 데이터 증강을 통해 **모델의 일반화 성능 향상**  
        - 무작위 크롭 후 리사이즈  
    - Resize의 경우 원본 이미지의 비율을 유지하며 크기 조정
<br/>   

- **transforms.CenterCrop(x)**  
    - 이미지 중심에서 지정된 크기만큼 잘라내는 역할   
    - 이미지의 크기를 줄이거나 특정 부분을 강조할 때 사용  
    - 주로 검증 데이터셋에 사용  
    - 직사각형 크기로도 자를 수 있음  
    ```py
    transforms.CenterCrop((224, 256))
    ```  
- **Resize()와 CenterCrop()이 함께 사용되어 원본 이미지를 일관된 크기로 변환**
<br/>  

- **transforms.RandomHorizontalFlip()**  
    - 이미지를 좌우로 무작위로 뒤집으며, 기본적으로 50%의 확률로 뒤집힘  
    - 'p' 매개변수를 통해 사용자가 원하는 확률을 지정할 수도 있음  
    - **데이터 다양성 증가, 모델 일반화 능력 향상, 비대칭적인 특성 보정, 훈련 데이터 증가** 등의 효과 존재  
    ```py
    transforms.RandomHorizontalFlip(p=0.7)  # 70% 확률로 좌우 반전
    ```
<br/>  

- **transforms.ToTensor()**  
    - 입력으로 들어온 데이터의 타입을 확인하고, PyTorch 텐서로 변환  
        - 기본적으로 PIL 이미지는 (높이, 너비, 채널) 순서로 되어 있지만, ToTensor를 적용하면 (채널, 높이, 너비) 순서로 변경  
        - PIL 이미지의 경우 픽셀 값 범위를 0에서 255 사이에서 0에서 1 사이로 조정 (**픽셀 값 범위 조정**)  
            - 넘파이의 경우 이미 0에서 1사이의 범위일 수 있음  
<br/>  

- **transforms.Normalize()**
    - 이미지 데이터 정규화를 위한 모듈의 함수  
    - 입력 이미지의 각 채널을 평균과 표준편차를 사용하여 정규화  
    - 보통 RGB 이미지의 경우 각 채널 별로 평균과 표준편차를 구해 적용
    - 정규화는 일반적으로 학습 및 검증 데이터셋 모두에 적용  
    ```py
    # 정규화에 사용할 평균과 표준편차
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # 변환 함수 정의
    transforms.Normalize(mean=mean, std=std)
    ```  
- _transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])_를 **보편적으로 활용**  
    - 이는 **ImageNet 데이터셋에서 계산된 평균과 표준편차**  
    - ImageNet은 대규모의 이미지 데이터셋으로, 다양한 카테고리와 스타일의 이미지를 포함하고 있어 일종의 표준으로 인식됨  
    - 대부분의 사전 학습된 모델들은 ImageNet 데이터셋으로 사전학습되어 적합하다고 여겨짐
 <br/>  
