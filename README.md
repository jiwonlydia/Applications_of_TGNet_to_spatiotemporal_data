# Applications_of_TGNet_to_spatiotemporal_data

## Usage
### 1. Run preprocess_fine-dust.py
```
>>> python preprocess_fine-dust.py
```
- Raw 데이터를 3차원 이미지 데이터로 전처리하여 ./data_preprocessed/ 경로 하에 train_image_dust.npy, test_image_dust.npy 파일을 생성한다.

### 2. Run Sampler-01_fine-dust.py
```
>>> python Sampler-01_fine-dust.py
```
- 3차원 이미지 데이터를 sequence data 형태로 전처리하여 ./data/ 경로 하에 STAMP, LAG, STEP에 따른 sequence data를 생성한다. 
- STAMP: 데이터의 시간 단위
ex)  30분 단위의 기존 데이터를 2시간 단위로 sampling하려면 STAMP=2로 설정
- LAG: 학습에 사용할 과거 데이터 개수 
- STEP: multi-step 예측의 단계 

### 3. Run  Model-01_fine-dust.py
```
>>> python Model-01_fine-dust.py –-train –-stamp 2 –-lag 8 –-epoch 1000
```
- TGNet 모델을 훈련 후 recursive multi-step 예측 평가
- training 유무, stamp, lag, epoch 등을 설정 가능
- training 과정 없이 저장된 모델을 불러와 예측만 진행할 경우 –-train 생략

### 4. direct multi-step prediction 및 Seoul floating population data에 대해서도 동일하게 진행한다. 

---
#### References
- 본 프로젝트 및 코드는 논문 [Forecasting taxi demands with fully convolutional networks and temporal guided embedding](https://openreview.net/pdf?id=BygF00DuiX) 및 저자의 [코드](https://github.com/jiwonlydia/TGGNet-keras)를 참조하였다.

@article{lee2019demand,
  title={Demand Forecasting from Spatiotemporal Data with Graph Networks ans Temporal-Guided Embedding},
  author={Lee, Doyup and Jung, Suehun and Cheon, Yeongjae and Kim, Dongil and You, Seungil},
  journal={arXiv preprint arXiv:1905.10709},
  year={2019}
}
