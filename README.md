# 딥러닝 일러스트레이티드

![](https://github.com/rickiepark/dl-illustrated/blob/master/img/cover.jpeg)

이 깃허브는 시그마프레스에서 출간한 <[딥러닝 일러스트레이티드](https://tensorflow.blog/dl-illustrated/)> 책의 코드를 담고 있습니다. 책을 보시기 전에 꼭 [에러타](https://tensorflow.blog/dl-illustrated/)를 확인해 주세요.

이 책의 코드는 최신 텐서플로, 케라스를 사용하며 구글 코랩(Colab)을 사용해 무료로 실행할 수 있습니다. 각 주피터 노트북에 코랩에서 실행할 수 있는 링크가 포함되어 있습니다.

### 1부: 딥러닝 소개

#### 1장: 생물의 눈과 기계의 눈

* 생물의 눈
* 기계의 눈
	* 신인식기
	* LeNet-5
	* 전통적인 머신러닝 방법
	* 이미지넷과 ILSVRC
	* AlexNet
* 텐서플로 플레이그라운드
* Quick, Draw!

#### 2장: 사람의 언어와 기계의 언어

* 자연어 처리를 위한 딥러닝
	* 딥러닝은 자동으로 표현을 학습합니다
    * 자연어 처리
	* 자연어 처리를 위한 딥러닝의 짧은 역사
* 언어의 컴퓨터 표현
	* 단어의 원-핫 표현
	* 단어 벡터
	* 단어 벡터 산술 연산
	* word2viz
	* 지역 표현 vs 분산 표현
* 자연어의 구성 요소
* 구글 듀플렉스

#### 3장: 기계의 예술

* 밤새도록 마시는 술꾼
* 가짜 얼굴 생성
* 스타일 트랜스퍼: 사진을 모네 그림으로 변환하기 (또는 그 반대)
* 스케치를 사진으로 바꾸기
* 텍스트를 사진으로 바꾸기
* 딥러닝을 사용한 이미지 처리

#### 4장: 게임하는 기계

* 딥러닝, 인공 지능 그리고 다른 기술들
	* 인공 지능
	* 머신러닝
	* 표현 학습
	* 인공 신경망
    * 딥러닝
    * 머신 비전
    * 자연어 처리
* 3가지 종류의 머신러닝 문제
	* 지도 학습
	* 비지도 학습
	* 강화 학습
* 심층 강화 학습
* 비디오 게임
* 보드 게임
	* 알파고
	* 알파고 제로
	* 알파제로
* 물체 조작
* 유명한 심층 강화 학습 환경
	* OpenAI 짐
	* 딥마인드 랩
	* 유니티 ML-Agents
* 세 부류의 인공 지능
	* 약 인공 지능
	* 인공 일반 지능
	* 초 인공 지능

### 2부: 핵심 이론

#### 5장: 말(이론)보다 마차(코드)

* 필요한 지식
* 설치
* 케라스로 얕은 신경망 만들기 ([5-1.shallow_net_in_keras.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/5-1.shallow_net_in_keras.ipynb))
	* MNIST 손글씨 숫자 ([5-2.mnist_digit_pixel_by_pixel.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/5-2.mnist_digit_pixel_by_pixel.ipynb))
	* 신경망 구조
	* 데이터 적재
	* 데이터 전처리
	* 신경망 구조 설계
	* 신경망 모델 훈련

#### 6장: 핫도그를 감지하는 인공 뉴런

* 생물학적 신경 구조
* 퍼셉트론
	* 핫도그 감지기
	* 이 책에서 가장 중요한 공식
* 현대적인 뉴런과 활성화 함수
	* 시그모이드 활성화 함수 ([6-1.sigmoid_function.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/6-1.sigmoid_function.ipynb))
	* Tanh 활성화 함수 
	* ReLU 활성화 함수
* 활성화 함수 선택

#### 7장: 인공 신경망

* 입력층
* 밀집 층
* 핫도그 감지 밀집 신경망
	* 첫 번째 은닉층의 정방향 계산
	* 나머지 층의 정방향 계산
* 패스트 푸드 분류 신경망의 소프트맥스 활성화 함수 ([7-1.softmax_demo.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/7-1.softmax_demo.ipynb))
* 얕은 신경망 다시 보기

#### 8장: 심층 신경망 훈련하기

* 비용 함수
	* 이차 비용 함수 ([8-1.quadratic_cost.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/8-1.quadratic_cost.ipynb))
	* 포화된 뉴런
	* 크로스-엔트로피 비용 함수 ([8-2.cross_entropy_cost.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/8-2.cross_entropy_cost.ipynb))
* 최적화: 학습을 통해 비용을 최소화하기
	* 경사 하강법
	* 학습률 ([8-3.measuring_speed_of_learning.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/8-3.measuring_speed_of_learning.ipynb))
	* 배치 크기와 확률적 경사 하강법
	* 지역 최솟값 탈출하기
* 역전파
* 은닉층 개수와 뉴런 개수 튜닝하기
* 케라스로 중간 깊이 신경망 만들기 ([8-4.intermediate_net_in_keras.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/8-4.intermediate_net_in_keras.ipynb))

#### 9장: 심층 신경망 성능 높이기

* 가중치 초기화 ([9-1.weight_initialization.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/9-1.weight_initialization.ipynb))
	* 세이비어 글로럿 분포
* 불안정한 그레이디언트 
	* 그레이디언트 소실
	* 그레이디언트 폭주
	* 배치 정규화
* 모델 일반화 - 과대적합 피하기
	* L1와 L2 규제
	* 드롭아웃
	* 데이터 증식
* 고급 옵티마이저
	* 모멘텀
	* 네스테로프 모멘텀
	* AdaGrad
	* AdaDelta와 RMSProp
	* Adam
* 케라스로 심층 신경망 만들기 ([9-2.deep_net_in_keras.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/9-2.deep_net_in_keras.ipynb))
* 회귀 ([9-3.regression_in_keras.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/9-3.regression_in_keras.ipynb))
* 텐서보드 ([9-4.deep_net_in_keras_with_tensorboard.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/9-4.deep_net_in_keras_with_tensorboard.ipynb))

### 3부: 딥러닝 애플리케이션

#### 10장: 머신 비전

* 합성곱 신경망
	* 시각적 이미지의 2차원 구조
	* 계산 복잡도
	* 합성곱 층
	* 다중 필터
	* 합성곱 예제
	* 합성곱 필터 하이퍼파라미터
	* 스트라이드 크기
	* 패딩
* 풀링 층
* 케라스로 만드는 LeNet-5 ([10-1.lenet_in_keras.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/10-1.lenet_in_keras.ipynb))
* 케라스로 만드는 AlexNet ([10-2.alexnet_in_keras.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/10-2.alexnet_in_keras.ipynb))과 VGGNet ([10-3.vggnet_in_keras.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/10-3.vggnet_in_keras.ipynb))
* 잔차 네트워크
	* 그레이디언트 소멸: 심층 CNN 최대의 적
	* 잔차 연결
    * ResNet
* 머신 비전 애플리케이션
	* 객체 탐지
	* 이미지 분할
	* 전이 학습 ([10-4.transfer_learning_in_keras.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/10-4.transfer_learning_in_keras.ipynb))
	* 캡슐 네트워크

#### 11장: 자연어 처리

* 자연어 데이터 전처리 ([11-1.natural_language_preprocessing.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/11-1.natural_language_preprocessing.ipynb))
	* 토큰화
	* 모든 문자를 소문자로 바꾸기
	* 불용어와 구둣점 삭제
	* 어간 추출: *n*-그램 다루기
	* 전체 말뭉치 전처리하기
* word2vec으로 단어 임베딩 만들기
	* word2vec의 핵심 이론
	* 단어 벡터 평가
	* word2vec 실행하기
	* 단어 벡터 출력하기
* ROC 곡선의 면적
	* 오차 행렬
	* ROC AUC 계산하기
* 신경망으로 영화 리뷰 분류하기
	* IMDB 영화 리뷰 데이터
	* IMDB 데이터 살펴 보기
	* 리뷰 길이 맞추기
	* 밀집 신경망 ([11-2.dense_sentiment_classifier.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/11-2.dense_sentiment_classifier.ipynb))
	* 합성곱 신경망 ([11-3.convolutional_sentiment_classifier.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/11-3.convolutional_sentiment_classifier.ipynb))
* 순차 데이터를 위한 신경망
    * 순환 신경망
	* 케라스로 RNN 구현하기 ([11-4.rnn_sentiment_classifier.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/11-4.rnn_sentiment_classifier.ipynb))
    * LSTM
	* 케라스로 LSTM 구현하기 ([11-5.lstm_sentiment_classifier.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/11-5.lstm_sentiment_classifier.ipynb))
	* 양방향 LSTM ([11-6.bi_lstm_sentiment_classifier.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/11-6.bi_lstm_sentiment_classifier.ipynb))
	* 적층 순환 신경망 ([11-7.stacked_bi_lstm_sentiment_classifier.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/11-7.stacked_bi_lstm_sentiment_classifier.ipynb), [11-8.gru_sentiment_classifier.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/11-8.gru_sentiment_classifier.ipynb))
	* Seq2seq와 어텐션
	* NLP의 전이 학습
* 케라스 함수형 API ([11-9.conv_lstm_stack_sentiment_classifier.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/11-9.conv_lstm_stack_sentiment_classifier.ipynb), [11-10.multi_convnet_sentiment_classifier.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/11-10.multi_convnet_sentiment_classifier.ipynb))

#### 12장: 생성적 적대 신경망

* 핵심 GAN 이론
* _Quick, Draw!_ 데이터셋
* 판별자 신경망
* 생성자 신경망
* 적대 신경망
* GAN 훈련 ([12-1.generative_adversarial_network.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/12-1.generative_adversarial_network.ipynb))

#### 13장: 심층 강화 학습

* 강화 학습의 핵심 이론
	* 카트-폴 게임
	* 마르코프 결정 과정
	* 최적 정책
* 심층 Q-러닝 신경망의 핵심 이론
	* 가치 함수
	* Q-가치 함수
	* 최적의 Q-가치 추정하기
* DQN 에이전트 만들기 ([13-1.cartpole_dqn.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/13-1.cartpole_dqn.ipynb))
	* 파라미터 초기화
	* 에이전트 신경망 모델 만들기
	* 게임 플레이 기억하기
	* 경험 재생을 통해 훈련하기
	* 행동 선택하기
	* 모델 파라미터 저장하고 로드하기
* OpenAI 짐 환경과 연동하기
* SLM Lab을 사용한 하이퍼파라미터 최적화
* DQN 이외의 에이전트
	* 정책 그레이디언트와 REINFORCE 알고리즘
	* 액터-크리틱 알고리즘

### 4부: 나 그리고 AI

#### 14장: 딥러닝 프로젝트 시작하기

* 딥러닝 프로젝트 아이디어
	* 머신 비전과 GAN ([14-1.fashion_mnist_pixel_by_pixel.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/14-1.fashion_mnist_pixel_by_pixel.ipynb))
	* 자연어 처리
	* 심층 강화 학습
	* 기존 머신러닝 프로젝트 변환하기
* 프로젝트를 위한 추가 자료
	* 사회적으로 유익한 프로젝트
* 하이퍼파라미터 튜닝을 포함한 모델링 프로세스
	* 하이퍼파라미터 탐색 자동화
* 딥러닝 라이브러리
	* 케라스와 텐서플로
	* 파이토치 ([14-2.pytorch.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/14-2.pytorch.ipynb))
	* MXNet, CNTK, 카페 등등
* 소프트웨어 2.0
* 인공 일반 지능