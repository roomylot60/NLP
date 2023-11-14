## Deep Learning
- Artifical Neural Network(인공 신경망)의 층을 연속적으로 쌓아올려 데이터를 학습
- 기계가 가중치를 스스로 찾아내도록 자동화 시키는 심층 신경망의 학습

### Perceptron(퍼셉트론)
- 초기 형태의 인공 신경망
- 입력값 x 벡터에 대해 가중치 w를 곱하고 편향 b(-임계값)을 더해 결과값 y를 도출

### Single-Layer Perceptron
- Input layer(입력층)과 Output layer(출력층)으로 구성
### Multi-Layer Perceptron
- Input, Output layer 사이에 Hidden layer(은닉층)을 두어 좀 더 복잡한 문제에 대응
    * Feed-Forward Neural Network(FFNN) : 입력층에서 출력층방향으로 연산이 전개
    * RNN : 은닉층의 출력값을 출력층과 은닉층의 입력으로 사용
    * Fully-connected Layer : 어떤 층의 모든 뉴런이 이전 층의 모든 뉴런과 연결된 층
    * Activation Function(활성화 함수) : 은닉층과 출력층의 뉴런에서 출력값을 결정하는 비선형 함수
        + Step Function(계단 함수)
        + Sigmoid Function : 시그모이드 함수를 활성화 함수로 하는 인공 신경망을 다층으로 쌓을 때 역전파 과정에서 사용하는 경사 하강법에 의해 Vanishing Gradient(기울기 소실)이 발생하여 학습이 잘 이루어지지 않음
        + Hyperbolic tangent function
        + ReLU Function : 음수 입력에 대해 0을 출력하고, 양수는 입력값을 출력
        + Leaky ReLU
        + Softmax Function : 시그모이드 함수처럼 출력층에서 주로 사용되며, 시그모이드가 이진 분류에 사용되는 반면, 소프트맥스는 다중 클래스 분류에 주로 사용

### Forward Propagation(순전파)
- 입력층에서 출력층 방향으로 연산을 진행하는 과정
### BackPropagation(역전파)
- 순전파 과정을 진행하여 예측값과 실제값의 오차를 계산하였을 때, 경사 하강법을 사용하여 학습률을 반영해 가중치를 업데이트하는 과정
### Concept
- Loss function : 실제값과 예측값의 차이를 수치화 해주는 함수
    * MSE
    * Binary Cross-Entropy
    * Categorical Cross-Entropy
- Batch : 가중치 등의 매개 변수의 값을 조정하기 위해 사용하는 데이터의 양
    * Batch Gradient Descent : Optimizer 중 하나로 오차를 구할 때 전체 데이터를 고려
    * Stochastic GD : 랜덤으로 선택한 하나의 데이터에 대해서만 경사 하강법을 계산
    * Mini-Batch GD : 적당한 수의 배치 크기를 지정하여 GD를 실행
- Optimizer : Compile 과정에서 사용
    * Momentum : GD에서 계산된 접선의 기울기에 한 시점 전의 접선의 기울기 값을 일정한 비율만큼 반영
    * Adagrad : 각각의 매개변수에 서로 다른 학습률을 적용
    * RMSprop : Adagrad의 학습률이 지나치게 떨어지는 단점을 보완하기 위해 사용
    * Adam : RMSprop과 Momentum을 합친 방식
- Epoch : 인공 신경망에서 전체 데이터에 대해서 순전파와 역전파가 끝난 상태로 훈련 및 검증 과정이 한차례 끝난 것
- Gradient Vanishing : 역전파 과정에서 입력층으로 갈 수록 기울기가 점차적으로 작아져 입력층에 가까운 층들에서의 업데이트가 제대로 이루어지지 않는 상태
- Gradient Exploding : 기울기가 점차 커져서 가중치들의 값이 발산
    * Gradient Clipping : 기울기 폭주를 막기 위해 임계값을 넘지 않도록 값을 제한
    * Weight initialization
        + Xavier Initialization : 이전 층과 다음 층의 뉴런의 개수를 고려하여, 여러 층의 기울기 분산 사이에 균형을 맞춰서 특정 층이 너무 주목을 받거나 다른 층이 뒤쳐지는 것을 방지(S자 형태의 activation function에서 좋은 성능) ![Xavier initialization_uniform](./img/Xavier_uniform.jpg) ![Xavier initialization_normal](./img/Xavier_normal.jpg)
        + He Initialization : 이전 층의 뉴런의 개수를 반영하여 초기화(ReLU 계열 함수에 효율적) ![He initialization_uniform](./img/He_uniform.jpg) ![He initialization_normal](./img/He_normal.jpg)
    * Batch Normalization(배치 정규화) : 인공 신경망에 들어가는 각 입력을 평균과 분산으로 정규화
        + Internal Covariate Shift(내부 공변량 변화) : 학습에 의해 가중치가 변화하면, 입력 시점에 따라 입력 데이터의 분포가 변화하는 것
    * Layer Normalization(층 정규화)

### 과적합을 방지하는 방법
1. 데이터의 양을 늘리기 : 데이터의 양이 적을 때, 데이터의 특정 패턴이나 노이즈까지 쉽게 암기할 수 있으므로, 데이터의 양을 늘려 일반적인 패턴을 학습(ex - Data Augmentation, Back Translation)

2. 모델의 복잡도 줄이기 : 은닉층의 수, 매개변수의 수를 줄여서 사용

3. Regularization(가중치 규제) 적용하기
    - L1 regularization : 가중치 w들의 절댓값의 합을 비용함수에 추가
    - L2 regularization : 모든 가중치 w들의 제곱합을 비용함수에 추가
4. Dropout : 학습 과정에서 설정한 비율로 랜덤한 신경망을 사용하지 않는 방법으로, 특정 뉴런 또는 특정 조합에 너무 의존적으로 되는 것을 방지