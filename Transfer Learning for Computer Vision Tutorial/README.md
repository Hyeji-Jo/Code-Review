# Load Data

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
