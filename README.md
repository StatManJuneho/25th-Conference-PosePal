# PosePal

<div align="center">
<h3>25th YBIGTA Conference</h3>

![KakaoTalk_20241224_010135825](https://github.com/user-attachments/assets/b6e591fb-a510-4cb5-a785-587430edafbc)

<em>Your Friendly Fitness Coach: Real-time Exercise Analysis and Feedback System</em>

</div>

## 목차
- [문제 정의](#문제-정의)
- [세부 목표](#세부-목표)
- [접근 방법](#접근-방법)
- [결과 및 주요 기능](#결과-및-주요-기능)
- [팀 구성](#팀-구성)

---

## 문제 정의
이제 PT, 안 받아도 됩니다!

좋은 PT 선생님 고르기 너무 어려우시죠?
나 스스로 내 운동 자세를 교정하고, 
분석해볼 수 있다면 얼마나 좋을까?

PosePal과 함께라면 가능합니다!

사용자의 운동 영상을 실시간으로 분석하여 자세를 교정하고, 개선을 위한 피드백을 제공하는 운동 코칭 시스템,
PosePal을 소개합니다!

---

## 세부 목표

1. 사용자의 운동 자세를 실시간으로 추적하여 주요 관절의 움직임을 분석. 
2. 촬영된 영상을 분석하여 올바르게 운동을 수행하고 있는지 여부를 분류하고 분석한 후, 점수화하여 하나의 리포트 형태로 정리. 
3. 추가적으로 궁금한 점을 실시간으로 챗봇을 통해 질문할 수 있도록 하여, 심화 운동 학습에 도움을 주고자 함.

---

## 접근 방법

### 태스크

1. **운동 자세 분석**
   - MediaPipe를 활용하여 운동 자세의 33개 주요 키포인트를 추적.
   - Multivariate Time-Series Classification을 기반으로 운동 자세 분류.
2. **운동 자세 평가**
   - ROCKET Classifier를 사용하여 주요 관절의 움직임에 대한 시계열 데이터를 효과적으로 분류.
   - Peak Detection 알고리즘을 통해 운동 반복 횟수와 자세 구간 파악.
3. **리포트와 챗봇을 통한 피드백 제공**
   - GPT API를 활용하여 개인 맞춤형 피드백 리포트 생성.
   - Prompt Engineering으로 GPT를 전문 피트니스 코치로 설정, 챗봇으로 추가적인 운동 상담 제공.
4. **Keypoints**
   - Inputs: 실시간 촬영 동영상 + 사용자가 제공하는 운동 영상
   - Outputs: 운동 영상 기반 자세 분석 후 제공되는 피드백
   - Personlized Report: 운동 분석 결과를 시각적이고 문서화된 형태로 사용자에게 제공
   - Chatbot: 추가로 궁금한 점을 실시간으로 질문하여 심화 운동 지식 학습.
![image](https://github.com/user-attachments/assets/f88c4788-4d25-407a-a1e1-5a671f0b67be)

### 데이터셋

- **AI HUB 피트니스 자세 이미지 데이터**
  - 사람의 피트니스 동작 영상 200,000 Clip(건당 15초)을 촬영 및 편집하고, 해당 영상정보에서 총 300만 장의 영상이미지를 추출하여 24개의 Keypoint로 가공한 데이터.
  - 동일 운동동작에 대해 5회 이상 반복동작을 실시. 평균 48명의 동일동작을 촬영하여 데이터셋의 품질 향상. 한국인 평균 체격의 ±20%, 연령대 20~30대, 운동년수 2~5년 남녀 70여 명을 모델로 실내 스튜디오에서 촬영.
  - 각각의 운동에 대하여 정의된 운동 상태의 수에 맞춰서 전체 경우의 수가 설정됨. 예를 들어 푸시업의 경우 5가지의 운동상태를 이용하여 1가지의 바른 자세, 31가지의 틀린 자세에 대해 촬영.

- **커스텀 데이터**
  - 본 프로젝트에서는 AI HUB의 Annotated 데이터를 사용하는 대신 Raw Image를 사용해 MediaPipe로 Pose Estimation 수행, 이에 따른 Output인 Coordinates 값을 학습의 Input으로 사용
  - 운동별 약 10,000 프레임 사용.

### 모델링/아키텍처

- **Intuition**
  - 사람의 운동 동작은 일종의 Time-Series로 볼 수 있다?
  - 운동 동작을 수행하며 관절마다 변화하는 위치 좌표를 Multivariate Time-Series로 나타낼 수 없을까?

- **Multivariate Time-Series**
  - Multiple Variables: 각각의 변수가 시간에 따라 변화하는 feature를 나타냄.
  - Variable Interdependencies and Temporal Dependency
  - Higher Complexity: Univariate time-series에 비해 분석과 모델링이 어려움.
![image](https://github.com/user-attachments/assets/35a415f9-9c26-4687-987a-1f1b35855650)

- **MediaPipe**
  - Google의 Pose Estimation 프레임워크로, BlazePose를 기반으로 하여 높은 정확도 제공.
  - 사람의 어깨, 팔꿈치, 무릎 등 주요 신체 랜드마크를 추적하여 33개의 3D 키포인트제공
  - OpenPose Framework와 비교하여 상대적으로 가벼워 모바일 환경으로의 확장 가능성
  - For Training: AI HUB Raw Image Data -> Pose Estimation -> Training Data Construction
  - For User: Video Data -> Dividing Video into Frames -> Pose Estimation -> Input data for Inference
![image](https://github.com/user-attachments/assets/1856fb5d-0732-4268-afc0-39a64c51f5f1)

- **ROCKET Classifier**
  - An efficient method for time-series classification.
  - Transforms raw time-series data into a feature space using thousands of random convolutional kernels, and feed it into RidgeClassifier.
  - 즉, Random 1-D Kernels와 RidgeClassifier를 조합하여 Multivariate Time-Series 데이터를 효과적으로 분류.
![image](https://github.com/user-attachments/assets/74283725-0f61-4b0f-925f-9477cdac5ac6)

- **GPT API**
  - 사용자의 운동 데이터를 바탕으로 언어 기반의 개인화된 피드백 생성.
  - Exercise Feedback: Generates language feedback based on the model’s analysis.
  - Exercise Counseling: Provides real-time responses to user queries related to exercise and general fitness.

- **Streamlit**
  - 사용자가 운동 영상을 업로드하고, 리포트를 확인한 후 챗봇을 이용할 수 있는 인터페이스 제공.
  - First Page: Service Implementation, enabling taking or uploading exercise videos.
  - Second Page: Exercise Counseling Interface, displaying GPT-powered personalized reports and chatbot for real-time conversation with users.

- **Methodology**
  - Data Smoothing:
     - Time-Series의 noise를 제거하여 매끄럽게 만들기.
     - Peak Detection의 정확도를 높이기 위한 중요한 사전 작업.
  - Peak Detection:
     - Smoothed data 기반으로 Peak 감지하여 운동의 reps를 파악.
     - 각 반복 구간의 시작과 끝을 정의.
     - 가장 변동성이 심한 좌표를 기준으로 삼음
     - e.g. 사레레는 left elbow의 y좌표를 기준으로 삼음.
     - 2 reps마다 데이터를 결합하여 chunking.
  - Point: Time Series Data를 Chunking 후 Rocket Classifier의 Input Data로 넣어 모델 학습 및 추론 진행.
![image](https://github.com/user-attachments/assets/03dd1042-987c-4546-9a4a-02c06a48bdf5)

---

## 결과 및 주요 기능

1. **개인 맞춤형 리포트 생성**
   - Reps마다 운동 자세 상태 클래스 중 하나로 분류
   - 분류된 결과 + 올바른 자세와 부정확한 자세의 관절값을 비교한 Pose Estimation 결과를 바탕으로 한 세트의 운동 전반에 대한 Report를 LLM (GPT 3.5)이 생성
   - Prompt Engineering : Persona 부여 -> GPT에 전문 피트니스 코치라는 역할 부여.
   - Report에는 운동의 각 항목별(예: 무릎 각도, 상체 안정성 등)에 대해 세부 점수를 합산한 총점, 총평 등이 포함됨.
![image](https://github.com/user-attachments/assets/3702bb80-dac4-4061-99c9-2df318be05a7)

2. **실시간 챗봇 기능**
   - 앞에서의 사용자 운동 리포트를 기반으로 GPT-4를 활용해 실시간으로 유저의 질문에 대해 친절한 답변을 제공(?)
   - 챗봇의 응답은 비동기적으로 처리되어, 사용자가 질문을 입력한 후 응답이 준비되는 대로 순차적으로 화면에 표시됨.
![image](https://github.com/user-attachments/assets/fa70b65c-ae4e-4502-8ea0-cc2786e920d6)

3. **Contributions**
   - Time-series를 이용한 Exercise Classification의 신선함(?)
4. **Limitations**
   - 이따금씩 나타나는 Hallucination Problems.
5. **모델의 유연성, 확장 가능성**
   - 새로운 운동 데이터를 쉽게 적용할 수 있는 아키텍처 설계.

---

## 팀 구성

| 이름       | 팀  | 역할                              |
|------------|-----|-----------------------------------|
| 김준호 (팀장) | DS  | Multivariate Time-Series Modeling, Pose Estimation Modeling |
| 김서윤      | DA  | LLM, Reports Generation, Chatbot System |
| 김성환      | DS  | Video Processing, Pose Estimation Modeling |
| 이상헌      | DE  | Fullstack                        |
| 최수현      | DS  | Multivariate Time-Series Modeling, Pose Estimation Modeling |
| 홍규원      | DS  | LLM, Reports Generation, Chatbot System |
