# 1.Load Data

## 1)이미지 변환
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
- **transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])를 보편적으로 활용**  
    - 이는 **ImageNet 데이터셋에서 계산된 평균과 표준편차**  
    - ImageNet은 대규모의 이미지 데이터셋으로, 다양한 카테고리와 스타일의 이미지를 포함하고 있어 일종의 표준으로 인식됨  
    - 대부분의 사전 학습된 모델들은 ImageNet 데이터셋으로 사전학습되어 적합하다고 여겨짐
 <br/>  
 
## 2)이미지 로드
```py
# 데이터셋 경로 설정 및 데이터 변환 정의
data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
                  for x in ['train', 'val']}
# 데이터로더 정의
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
              for x in ['train', 'val']}
# 데이터셋 크기 설정
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# 클래스 이름 설정
class_names = image_datasets['train'].classes

# 장치 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```  
- **데이터셋 경로 설정 및 데이터 변환 정의**  
    - data_dir  
        - 이미지 데이터가 저장된 디렉토리 경로  
        ```
        data
        └── hymenoptera_data
            ├── train
            │   ├── ants
            │   │   ├── ant1.jpg
            │   │   └── ...
            │   └── bees
            │       ├── bee1.jpg
            │       └── ...
            └── val
                ├── ants
                │   ├── ant1.jpg
                │   └── ...
                └── bees
                    ├── bee1.jpg
                    └── ...
        ```
    - image_datasets  
        - 'train'과 'val' 두 개의 데이터셋 정의  
        - 각 데이터셋은 datasets.ImageFolder를 통해 로드  
        - 위에서 설정한 data_transforms에 따라 데이터 변환 수행  
<br/>  

- **데이터로더 정의**  
    - **대부분의 딥러닝 모델은** 한 번에 전체 데이터셋을 처리하는 것이 아니라 **미니매치 단위로 데이터 처리**
    - DataLoader는 딥러닝 모델 학습 과정에서 필수적이며, 데이터셋을 효율적으로 관리하고 처리할 수 있는 중요한 도구  
    ```py
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    ```  
    - batch_size: 한 번에 로드할 이미지의 개수  
    - shuffle=True: 데이터를 미니배치로 로드할 때 매번 섞을지 여부  
        - 데이터가 순서대로 정렬되어 있는 경우, 모델의 학습이나 일반화 능력에 부정적인 영향을 미칠 수 있음  
    - num_workers=4: 데이터를 읽어오는 데 사용할 프로세스의 수  
        - 병렬적으로 데이터를 처리함으로 속도 향상에 도움
<br/>  

- **데이터셋 크기 설정**  
    - 데이터셋의 크기를 저장하는 용도로 사용됨  
    - **에폭(epoch)당 반복 횟수 설정**, 학습 진행 상황 모니터링, 데이터셋의 분할 관리, 모델 평가 등 모델 학습 과정에서 사용  
<br/>  

- **클래스 이름 설정**  
    - 모델이 예측한 클래스 인덱스를 해석하는 데 사용  
    ```py
    class_names = image_datasets['train'].classes
    print(class_names) # ['ants', 'bees']
    ```  
<br/>  

- **장치 설정**
    - CUDA 장치가 사용 가능한 경우("cuda:0"), GPU를 사용하도록 설정
    - 그렇지 않은 경우 CPU를 사용
 
# 2.Training the model
```py
import time
import os
from tempfile import TemporaryDirectory

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    # 학습 시작 시간을 기록합니다.
    since = time.time()

    # 임시 디렉토리를 생성하여 학습 중 체크포인트를 저장합니다.
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
    
        # 초기 모델 가중치를 저장합니다.
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0  # 최상의 검증 정확도를 기록할 변수를 초기화합니다.

        # 에폭 반복
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # 학습 단계와 검증 단계를 반복
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # 모델을 학습 모드로 설정
                else:
                    model.eval()   # 모델을 평가 모드로 설정

                running_loss = 0.0  # 현재 에폭 동안의 손실을 누적할 변수 초기화
                running_corrects = 0  # 현재 에폭 동안의 정확한 예측 수를 누적할 변수 초기화

                # 데이터 로더를 사용하여 미니 배치 단위로 데이터를 반복
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)  # 입력 데이터를 GPU 또는 CPU로 이동
                    labels = labels.to(device)  # 라벨 데이터를 GPU 또는 CPU로 이동

                    optimizer.zero_grad()  # 옵티마이저의 그래디언트를 초기화

                    # 학습 단계에서만 그래디언트 계산을 추적
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)  # 모델에 입력 데이터를 통과시켜 출력 계산
                        _, preds = torch.max(outputs, 1)  # 가장 높은 확률을 가진 클래스 인덱스 예측
                        loss = criterion(outputs, labels)  # 손실 함수 계산

                        # 학습 단계에서만 역전파와 옵티마이저 스텝 실행
                        if phase == 'train':
                            loss.backward()  # 그래디언트 계산
                            optimizer.step()  # 옵티마이저를 사용하여 모델 매개변수 갱신

                    # 통계 업데이트
                    running_loss += loss.item() * inputs.size(0)  # 현재 배치의 손실을 누적
                    running_corrects += torch.sum(preds == labels.data)  # 정확한 예측 수를 누적

                # 학습 단계가 끝나면 학습률 스케줄러 업데이트
                if phase == 'train':
                    scheduler.step()

                # 에폭 손실과 정확도 계산
                epoch_loss = running_loss / dataset_sizes[phase]  # 현재 에폭의 평균 손실 계산
                epoch_acc = running_corrects.double() / dataset_sizes[phase]  # 현재 에폭의 평균 정확도 계산

                # 에폭 손실과 정확도 출력
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # 최상의 검증 정확도를 달성한 모델의 가중치를 저장
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            # 에폭 종료 후 줄바꿈
            print()

        # 전체 학습 시간 계산 및 출력
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # 최상의 모델 가중치를 로드합니다.
        model.load_state_dict(torch.load(best_model_params_path))
    
    # 최상의 모델을 반환합니다.
    return model
```
- **에폭 반복**  
    ```py
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
    ```
    - 학습과 검증을 여러번 반복하기 위해 루프를 돌림  
    - 여러번 반복할수록 모델이 데이터셋의 패턴을 더 잘 학습하게되며, **각 에폭 후에 모델의 가중치가 조정**됨  
    - 여러번의 에폭을 거치며 모델이 점진적으로 학습함으로써, **과적합니아 불안정성 문제 완화 가능**  
 <br/>  
 
- **학습과 검증 단계 설정**  
    ```py
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0
    ```  
    - 학습 단계에서는 model.train(), 검증 단계에서는 model.eval()을 호출하여 모델의 모드를 설정  
    - **이전 에폭 값의 영향을 받지 않기 위해 손실과 정확도 초기화 수행** 
    - **손실**은 보통 **실수** 값으로 계산되기에 **'0.0'으로 초기화**하고, **정확도**는 보통 **정수** 값으로 누적되기에 **'0'으로 초기화**하는 것이 일반적
 <br/>  

- **데이터 처리**
    ```py
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)
    
        # zero the parameter gradients
        optimizer.zero_grad()
    
        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
    
            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()
    ```
    - dataloaders[phase]  
        - 위에서 불러온 데이터 로더를 사용하여 phase 값에 따른 데이터 로드  
    - inputs.to(device), labels.to(device)  
        - 위에서 지정된 device로 데이터 이동  
        - **GPU 또는 CPU의 메모리로 이동**시켜 해당 장치에서 연산을 수행할 수 있도록 조치  
        - 모델 파라미터와 데이터가 동일한 장치에 존재해야 연산이 원활하게 이루어짐  
    - **optimizer.zero_grad()**  
        - **그래디언트의 누적방지, 올바른 그래디언트 계산, 학습 안정성 유지**를 위해 옵티마이저의 **그래디언트 초기화**  
        - 만약 그래디언트를 초기화하지 않으면, 이전 미니 배치의 그래디언트가 현재에 영향을 미치게 되어 잘못된 업데이트 가능

    - **_그래디언트(Gradient)_**  
        - 모델의 손실 함수에 대한 파라미터의 미분값을 의미  
        - 딥러닝에서 주어진 입력 데이터와 모델의 파라미터를 사용하여 **손실 함수를 최소화하기 위해 그래디언트 활용**  
        - 함수가 다변수 함수일 경우, **각 변수에 대해 편미분을 계산**하여 그래디언트 벡터 형성

    - **그래디언트 계산**  
        - torch.set_grad_enabled(phase == 'train') : 학습 단계에서 그래디언트 계산 활성화  
            - 모델을 학습할 때는 그래디언트를 계산하고 모델을 업데이트해야 하므로  
        - outputs = model(inputs) : 입력 데이터를 모델에 통과시켜 출력 저장  
        - _, preds = torch.max(outputs, 1)  
            - torch.max(outputs, 1) 함수는 outputs 텐서에서 각 행에 대해 최댓값과 인덱스 반환  
            - _는 최댓값 자체를 나타내며, preds는 최댓값을 가진 클래스의 인덱스  
            - 일반적으로 분류 문제에서는 이 인덱스가 모델이 예측한 클래스  
        - loss = criterion(outputs, labels)  
            - criterion은 손실함수를 나타내며, 출력과 실제 라벨 사이의 손실 계산  
    - **역전파 & 파라미터 업데이트 수행**  
        - phase가 학습 단계일 때만 역전파와 옵티마이저의 파라미터 업데이트 수행  
        - loss.backward()를 통해 그래디언트 계산  
        - optimizer.step()을 통해 모델의 파라미터 업데이트, 가중치 갱신    
        - 마라미터 업데이트는 손실을 줄이는 방향으로 수행되며, 일반적으로 경사 하강법이나 그 변형을 사용하여 이루어짐  
    - **통계 업데이트**  
        - 현재 미니 배치의 손실과 정확한 예측의 수 누적  
        - loss.item()을 통해 손실값 가져와서 inputs.size(0) - 미니 배치의 크기를 곱해 전체 손실 계산  
        - preds == labels.data를 통해 예측값과 실제 라벨의 값의 일치여부 확인 후 torch.sum(...)으로 맞춘 예측 수 계산  
    - **스케줄러 스탭**  
        - 학습 단계가 끝나면 학습률 스케줄러 업데이트  
        - 학습 스케줄러는 일정한 학습 속도를 유지하거나, 진행됨에 따라 속도를 조정하는 기능 제공  
        - 모델이 더 빠르고 정확한 학습을 할 수 있도록 도와줌  
    - **에폭 손실 및 정확도 계산**  
        - 에폭의 손실(epoch_loss)과 정확도(epoch_acc)를 계산하여 출력  
        - 이는 모델이 어떻게 학습하고 성능이 개선되는지를 시각적으로 확인할 수 있음  
    - **최상의 모델 저장**  
        - 검증 단계에서의 최상의 정확도를 달성한 모델의 가중치 저장  
        - 검증 단계에서 모델의 성능이 이전의 최고 정확도보다 더 좋을 경우에만 실행됨  
        - model.state_dict(): 모델의 학습 가능한 모든 파라미터와 버퍼를 포함하는 사전  
        - best_model_params_path: 모델의 상태를 저장할 파일 경로  

