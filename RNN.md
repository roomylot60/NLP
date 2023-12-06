### RNN Text Classification
- Binary Classification
- Multi-Class Classification

### Text Classification using Keras(supervised learning)
- Supervised Learning(지도 학습) : Train Data로 Label이라는 정답을 포함하고 있는 Dataset을 사용하여 학습; 정답을 알고 있는 상태로 훈련
- Validation(검증) : 모든 Sample을 사용하여 학습하지 않고 일정 비율의 데이터를 남기고 이를 예측한 뒤 Label 값과 대조하여 검증하는 데 사용
- Embedding() : 각각의 단어가 정수로 변환된 값(index)를 입력으로 받아 임베딩 작업을 수행; 인덱싱 작업(정수 인코딩)의 방법 중 하나로는 빈도수에 따라 정렬
- 텍스트 분류는 RNN의 Many-to-one 문제에 속하므로, 모든 시점에 대해서 입력을 받지만, 최종시점의 은닉 상태 만을 출력
    * Binary Classification : 출력값의 종류가 두 종류일 경우(loss function = binary_crossentropy)
    * Multi-Class Classificatino : 출력값의 종류가 세 가지 이상(loss function : categorical_crossentropy)
```python
model.add(SimpleRNN(hidden_units, input_shape=(timesteps, input_dim)))
# hidden_units : RNN 출력의 크기; 은닉 상태의 크기
# timesteps : 시점의 수; 각 문서의 단어 수
# input_dim : 입력의 크기; 임베딩 벡터의 차원
```
