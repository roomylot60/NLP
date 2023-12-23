### RNN을 이용한 Encoder-Decoder

하나의 RNN을 Encoder, 다른 하나를 Decoder로 구현하여 이를 연결하여 하용하는 구조로 입력 문장과 출력 문장의 길이가 다를 경우에 사용(ex - 번역기, 텍스트 요약)

#### Sequence-to-Sequence;Seq2seq
- seq2seq : 입력된 시퀀스로부터 다른 도메인의 시퀀스를 출력하는 모델(ex - chatBoT, 기계 번역, STT, 내용 요약)으로 RNN을 조립하는 방식에 따라 구조를 생성. Encoder와 Decoder 두 개의 모듈로 구성되며, Encoder는 입력 문장의 모든 단어들을 순차적으로 입력받은 뒤에 마지막에 이 모든 단어 정보를 압축해서 하나의 벡터를 생성(Context Vector)하고 이를 Decoder에서 번역된 단어를 한 개씩 순차적으로 출력. 훈련 과정에서는 교사 강요를 사용하여 학습
    * Encoder : RNN 아키텍처로 바닐라 RNN이 아니라 LSTM 혹은 GRU Cell들로 구성되어 Tokenized words를 각각의 시점에 RNN 셀에서 입력으로 받아 마지막 시점의 Hidden State를 출력
    * Context Vector : Encoder에서 출력하는 마지막 시점의 은닉 상태로 Decoder의 첫번째 은닉 상태에 사용
    * Decoder : 기본적 구조는 RNNLM으로 초기 입력으로 문장의 시작을 의미하는 `<sos>`가 입력되면 다음 단어를 예측하고 예측 단어를 다음 시점의 RNN 셀의 입력으로 사용하여 문장의 끝을 의미하는 심볼 `<eos>`가 예측될 때까지 반복

- Bilingual Evaluation Understudy Score;BLEU : 자연어 처리 태스크를 기계적으로 평가할 수 있는 방법