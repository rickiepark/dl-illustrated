# Deep Learning Illustrated (2020)

This repository is home to the code that accompanies [Jon Krohn](https://www.jonkrohn.com/), [Grant Beyleveld](http://grantbeyleveld.com/about/) and [Aglaé Bassens](https://www.aglaebassens.com/)' book [Deep Learning Illustrated](https://www.deeplearningillustrated.com/). This visual, interactive guide to artificial neural networks was published on Pearson's Addison-Wesley imprint. 

## Installation

Step-by-step guides for running the code in this repository can be found in the [installation directory](https://github.com/the-deep-learners/deep-learning-illustrated/tree/master/installation). For installation difficulties, please consider visiting our book's [Q&A forum](https://groups.google.com/forum/#!forum/deep-learning-illustrated) instead of creating an _Issue_.

## Notebooks

All of the code covered in the book can be found in [the notebooks directory](https://github.com/the-deep-learners/deep-learning-illustrated/tree/master/notebooks) as [Jupyter notebooks](http://jupyter.org/). 

Below is the book's table of contents with links to all of the individual notebooks. 

*Note that while TensorFlow 2.0 was released after the book had gone to press, as detailed in Chapter 14 (specifically, Example 14.1), all of our notebooks can be trivially converted into TensorFlow 2.x code if desired.*

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

### Part III: Interactive Applications of Deep Learning

#### Chapter 10: Machine Vision

* Convolutional Neural Networks 
	* The Two-Dimensional Structure of Visual Imagery
	* Computational Complexity
	* Convolutional Layers
	* Multiple Filters
	* A Convolutional Example
	* Convolutional Filter Hyperparameters
	* Stride Length
	* Padding
* Pooling Layers
* LeNet-5 in Keras ([10-1.lenet_in_keras.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/10-1.lenet_in_keras.ipynb))
* AlexNet ([10-2.alexnet_in_keras.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/10-2.alexnet_in_keras.ipynb)) and VGGNet ([10-3.vggnet_in_keras.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/10-3.vggnet_in_keras.ipynb))
* Residual Networks 
	* Vanishing Gradients: The Bête Noire of Deep CNNs
	* Residual Connection
* Applications of Machine Vision
	* Object Detection
	* Image Segmentation
	* Transfer Learning ([10-4.transfer_learning_in_keras.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/10-4.transfer_learning_in_keras.ipynb))
	* Capsule Networks

#### Chapter 11: Natural Language Processing

* Preprocessing Natural Language Data ([natural_language_preprocessing.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/natural_language_preprocessing.ipynb))
	* Tokenization
	* Converting all Characters to Lower Case
	* Removing Stop Words and Punctuation
	* Stemming; Handling *n*-grams
	* Preprocessing the Full Corpus
* Creating Word Embeddings with word2vec
	* The Essential Theory Behind word2vec
	* Evaluating Word Vectors
	* Running word2vec
	* Plotting Word Vectors
* The Area Under the ROC Curve
	* The Confusion Matrix
	* Calculating the ROC AUC Metric
* Natural Language Classification with Familiar Networks
	* Loading the IMDB Film Reviews
	* Examining the IMDB Data
	* Standardizing the Length of the Reviews
	* Dense Network ([dense_sentiment_classifier.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/dense_sentiment_classifier.ipynb))
	* Convolutional Networks ([convolutional_sentiment_classifier.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/convolutional_sentiment_classifier.ipynb))
* Networks Designed for Sequential Data 
	* Recurrent Neural Networks ([rnn_sentiment_classifier.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/rnn_sentiment_classifier.ipynb))
	* Long Short-Term Memory Units ([lstm_sentiment_classifier.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/lstm_sentiment_classifier.ipynb))
	* Bidirectional LSTMs ([bi_lstm_sentiment_classifier.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/bi_lstm_sentiment_classifier.ipynb))
	* Stacked Recurrent Models ([stacked_bi_lstm_sentiment_classifier.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/stacked_bi_lstm_sentiment_classifier.ipynb))
	* Seq2seq and Attention
	* Transfer Learning in NLP
* Non-Sequential Architectures: The Keras Functional API ([multi_convnet_sentiment_classifier.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/multi_convnet_sentiment_classifier.ipynb))

#### Chapter 12: Generative Adversarial Networks

* Essential GAN Theory
* The _Quick, Draw!_ Dataset
* The Discriminator Network
* The Generator Network
* The Adversarial Network
* GAN Training ([generative_adversarial_network.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/generative_adversarial_network.ipynb))

#### Chapter 13: Deep Reinforcement Learning

* Essential Theory of Reinforcement Learning 
	* The Cart-Pole Game
	* Markov Decision Processes
	* The Optimal Policy
* Essential Theory of Deep Q-Learning Networks
	* Value Functions
	* Q-Value Functions
	* Estimating an Optimal Q-Value
* Defining a DQN Agent ([cartpole_dqn.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/cartpole_dqn.ipynb))
	* Initialization Parameters
	* Building the Agent’s Neural Network Model
	* Remembering Gameplay
	* Training via Memory Replay
	* Selecting an Action to Take
	* Saving and Loading Model Parameters
* Interacting with an OpenAI Gym Environment
* Hyperparameter Optimization with SLM Lab
* Agents Beyond DQN 
	* Policy Gradients and the REINFORCE Algorithm
	* The Actor-Critic Algorithm

### Part IV: You and AI

#### Chapter 14: Moving Forward with Your Own Deep Learning Projects

* Ideas for Deep Learning Projects
	* Machine Vision and GANs ([fashion_mnist_pixel_by_pixel.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/fashion_mnist_pixel_by_pixel.ipynb))
	* Natural Language Processing
	* Deep Reinforcement Learning
	* Converting an Existing Machine-Learning Project
* Resources for Further Projects 
	* Socially-Beneficial Projects
* The Modeling Process, including Hyperparameter Tuning 
	* Automation of Hyperparameter Search
* Deep Learning Libraries
	* Keras and TensorFlow ([deep_net_in_tensorflow.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/deep_net_in_tensorflow.ipynb))
	* PyTorch ([pytorch.ipynb](https://github.com/rickiepark/dl-illustrated/blob/master/notebooks/pytorch.ipynb))
	* MXNet, CNTK, Caffe, and Beyond
* Software 2.0
* Approaching Artificial General Intelligence

## Book Cover

![](https://github.com/rickiepark/dl-illustrated/blob/master/img/cover.jpeg)

