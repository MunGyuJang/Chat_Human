## <p align ="center"> 🕰️ Aiffel Thon </p>

## <p align ="center"> 프로젝트 : 인공지능 비서를 만들어보자! </p> 

### <p align ="center"> 🤖 Team ChatHuman 🤖 </p> 

### <p align ="center"> 팀원 </p>

 <p align ="center"> 🤹‍♂️ 방승욱 🚴‍♂️ 구본회 🏌️‍♂️ 이태환 ⛷️ 장문규 </p>

### <p align ="center"> 이 프로젝트는 22.12.26 ~ 23.02.08일까지 진행되는 아이펠톤 프로젝트 입니다. </p>
---
 
### 세부 일정

|순서|기간|세부 계획|
|:---:|:---:|:---:|
|1번|22.12.26 ~ 22.12.30|팀원들과의 계획 조율|
|2번|23.01.02 ~ 23.02.03|데이터 전처리 및 개발 환경 구축|
|3번|23.01.09 ~ 23.01.13|모델 테스트|
|4번|23.01.09 ~ 23.02.03|웹페이지 구축|
|5번|23.01.16 ~ 23.01.20|모델 연구 및 인퍼런스 코드 작성|
|6번|23.01.16 ~ 23.01.27|모델 학습|
|7번|23.01.18 ~ 23.02.03|모델 다듬기|
|8번|23.02.06 ~ 23.02.07|발표 준비|
|9번|23.02.08|발표|
 
---
### 개요
- 일상생활에 도움을 주며 현재 존재하는 시리, 클로버, 카카오와 같은 인공지능 비서에게 부재된 기능을 탑재하는것을 목표로 함.
  - 부재된 기능이라함은 일상대화 기능에 초점을 맞추며 그 외의 사소한 것들을 추가해볼 예정.   
  
- 기본적인 형태는 챗봇 형태이며 다양한 키워드를 통해 명령을 수행할 수 있도록 할 것.

![service](https://github.com/Ukbang/Aiffel_thon/blob/main/images/service.png)

---

### Requirement
> Python 3.9.7
> 
> Transformer 4.11.3
>  
> Numpy 1.21.4
>  
> PyTorch 1.9.1+cu111

---

### Dataset
- AIHub 에서 제공하는 주제별 텍스트 일상생활 데이터와 한국어 대화 요약을 이용하여 만듦.
- 데이터프레임 형태로 한 대화의 말뭉치를 Conversation column으로 구분하고 각 대화 간 발화자를 `'<usr>'`, `'<sys>'` 토큰으로 구분하였음.
- 약 19만개의 대화를 이용함. 

![data_image](https://github.com/Ukbang/Aiffel_thon/blob/main/images/data_image.jpeg)

---
### 전처리 방식
- modules/preprocessing.py 파일의 clear_sentence 함수를 이용하여 처리.
- #@이름#은 make_name 함수를 이용하여 랜덤한 이름을 생성할 수 있도록 하였음.
- @URL, #@시스템#사진#, #@이모티콘#은 삭제하였고 반복되는 ㅋ,ㅎ,ㅜ,ㅠ,. 과 같은 문자는 2개로 통일하였으며 자주 등장하는 키키 는 ㅋㅋ 로 변경하였음.

![make_name](https://github.com/Ukbang/Aiffel_thon/blob/main/images/make_name.jpeg)
![clear_sentence](https://github.com/Ukbang/Aiffel_thon/blob/main/images/clear_sentence.jpeg)

---

### 모델
- 모델은 🤗Hugging Face에서 제공하는 gpt2 모델을 사용하였음.
- 베이스 모델로 ['skt/kogpt2-base-v2'](https://github.com/SKT-AI/KoGPT2) 을 사용함.   
 
 
<p align ="center"><img src="https://user-images.githubusercontent.com/112140135/216884750-53fb4373-2d9d-4a6a-800b-0062b8b702f5.png" width="800px" height="300px"></p>

---
### 학습 진행과정 리더보드
#### Data type
__Topic = 250000개__
 
 __Topic+kakao Data = 190000개 ('`<usr>`로 끝나는 문구 삭제', 길이 256')__
 
__kakao Data = 65000개__

---

#### Label

__Input = Input과 Label이 동일__
 
 
__-100 = 마지막 `<sys>` 대화를 제외한 -100을 이용한 Masking__

__-100+sys = 모든 `<sys>` 대화를 제외한 모든 대화 -100으로 Masking__
 
 
__Shift = Input은 `<s>` 토큰을 bos_token으로 사용, Label은 `</s>`토큰을 eos_token으로 사용함.__

|index|Model|Epochs|Data type|진행 상황|진행 일시|Label|Loss|Val_Loss|Comment|성능|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-01-31|-100|4.290 -> 3.797 -> 3.340 -> 2.803 -> 2.195|3.821 -> 3.759 -> 3.804 -> 3.938 -> 4.143|단답형이고 대화가 잘 이루어 지지 않음.|[Link](https://github.com/Ukbang/Aiffel_thon/blob/main/chatbot/Test/23-02-01_-100_test.ipynb)|
|2|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-02-01|Input|1.476 -> 1.343 -> 1.270 -> 1.203 -> 1.137|1.486 -> 1.445 -> 1.434 -> 1.441 -> 1.461|현재까지 가장 Best|[Link](https://github.com/Ukbang/Aiffel_thon/blob/main/chatbot/Test/23-02-01_True_test.ipynb)|
|3|skt/kogpt2-base-v2|3|kakao Data|Done|2023-01-30|Input|2.330 -> 2.147 -> 2.084|1.765 -> 1.723 -> 1.704|문장 생성을 eos token 밖에 못함.|[Link](https://github.com/Ukbang/Aiffel_thon/blob/main/chatbot/Test/Inference_code_label_True_len384.ipynb)|
|4|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-02-01|shift|2.140 -> 2.005 -> 1.931 -> 1.864 -> 1.794|2.298 -> 2.236 -> 2.215 -> 2.221 -> 2.246|학습이 전혀 되지 않았음. 폐기|[Link]()|
|5|skt/kogpt2-base-v2|10|Topic+kakao|Done|2023-02-01|Input|1.483 -> 1.352 -> 1.275 -> 1.206 -> 1.135 -> 1.062 -> 0.986 -> 0.908 -> 0.830 -> 0.753|1.504 -> 1.469 -> 1.456 -> 1.463 -> 1.485 -> 1.517 -> 1.562 -> 1.616 -> 1.683 -> 1.759|5epoch 이상부터 학습이 오히려 안됨. 폐기|[Link]()|
|6|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-02-05|Input|1.479 -> 1.380 -> 1.330 -> 1.292 -> 1.260|1.401 -> 1.370 -> 1.357 -> 1.350 -> 1.346|자잘한 코드 수정후 학습하였음. 2번과 성능이 동일함.|[Link]()|
|7|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-02-05|-100|4.269 -> 3.869 -> 3.574 -> 3.296 -> 3.031|4.069 -> 3.997 -> 3.989 -> 4.032 -> 4.106|.....|[Link]()|
|8|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-02-05|-100+sys|3.826 -> |3.352 -> |.....|[Link]()|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|

---

### 추가된 서비스
<p align ="center"><img src="https://user-images.githubusercontent.com/112140135/216884826-5905e7cb-229a-4a53-becd-25508e40fd1d.png" width="600px" height="900px"></p>

---

### 회고

- 커스텀 데이터 로더 
```python
token_ids ====>  tensor([[49186,     2,     4,  ...,     0,     0,     0],
        [49186,     2,     4,  ...,     0,     0,     0],
        [49186,     2,     4,  ...,     0,     0,     0],
        [49186,     2,     4,  ...,     0,     0,     0],
        [49186,     2,     4,  ...,     0,     0,     0],
        [49186,     2,     4,  ...,     0,     0,     0]])
mask =====>  tensor([[2, 2, 2,  ..., 0, 0, 0],
        [2, 2, 2,  ..., 0, 0, 0],
        [2, 2, 2,  ..., 0, 0, 0],
        [2, 2, 2,  ..., 0, 0, 0],
        [2, 2, 2,  ..., 0, 0, 0],
        [2, 2, 2,  ..., 0, 0, 0]])
label =====>  tensor([[-100, -100, -100,  ..., -100, -100, -100],
        [-100, -100, -100,  ..., -100, -100, -100],
        [-100, -100, -100,  ..., -100, -100, -100],
        [-100, -100, -100,  ..., -100, -100, -100],
        [-100, -100, -100,  ..., -100, -100, -100],
        [-100, -100, -100,  ..., -100, -100, -100]])
(tensor([[49186,     2,     4,  ...,     0,     0,     0],
        [49186,     2,     4,  ...,     0,     0,     0],
        [49186,     2,     4,  ...,     0,     0,     0],
        [49186,     2,     4,  ...,     0,     0,     0],
        [49186,     2,     4,  ...,     0,     0,     0],
        [49186,     2,     4,  ...,     0,     0,     0]]), tensor([[2, 2, 2,  ..., 0, 0, 0],
        [2, 2, 2,  ..., 0, 0, 0],
        [2, 2, 2,  ..., 0, 0, 0],
        [2, 2, 2,  ..., 0, 0, 0],
        [2, 2, 2,  ..., 0, 0, 0],
        [2, 2, 2,  ..., 0, 0, 0]]), tensor([[-100, -100, -100,  ..., -100, -100, -100],
        [-100, -100, -100,  ..., -100, -100, -100],
        [-100, -100, -100,  ..., -100, -100, -100],
        [-100, -100, -100,  ..., -100, -100, -100],
        [-100, -100, -100,  ..., -100, -100, -100],
        [-100, -100, -100,  ..., -100, -100, -100]]))
```
첫 설계는 길이/턴 수가 긴 데이터는 뒷부분을 자르고 앞부분을 남기는 방식으로 시도함

`<usr>가나다<sys>라마바<usr>사아자(길이/턴수초과)<sys>차카타` :  
`<usr>가나다<sys>라마바` -- 자를 때는 무조건 sys 대화로 종료

```python
for i in tqdm(range(len(self.data))):
            hists = []
            dials = self.data[i]
            
            for u, utter in enumerate(dials):
                if u % 2 == 0:
                    hists.append(self.usr_token + utter)  # Speaker 1: User
                else:
                    hists.append(self.sys_token + utter)  # Speaker 2: System
            
            max_turn = max(len(hists), self.max_turns) # max_turns을 넘으면 post
            if max_turn % 2 != 0: max_turn -= 1 # user 대화로 끝남 방지
                
            for f in range(max_turn, 1, -2):
                contexts = hists[:f]
                if sum([len(l) for l in contexts]) > self.max_len-2: continue # bos_token와 eos_token 토큰을 추가하기 위해 -2
                contexts[0] = self.bos_token + contexts[0]
                contexts[-1] = contexts[-1] + self.eos_token
                contexts = [tokenizer.encode(ctx) for ctx in contexts]
                
                token_type_id = [[ctx[0]] * len(ctx) if c != 0 else [ctx[1]] * len(ctx) for c, ctx in enumerate(contexts)]
                label = [[-100] * len(ctx) if c != len(contexts)-1 else [-100] + ctx[1:] for c, ctx in enumerate(contexts)]
                
                input_id = list(chain.from_iterable(contexts))
                token_type_id = list(chain.from_iterable(token_type_id))
                label = list(chain.from_iterable(label))
                
                assert len(input_id) == len(token_type_id) == len(label), "There is something wrong in dialogue process."
                
                input_id, token_type_id, label = self.make_padding(input_id, token_type_id, label)
                
                self.input_ids.append(input_id)
                self.token_type_ids.append(token_type_id)
                self.labels.append(label)
                
                break
```
break을 남겨놓을 경우 데이터 하나당 최대 한 개의 데이터를 생성하지만  
break을 제거할 경우 가능한 모든 턴수를 모두 데이터로 사용해 agumentation함

  ex: max_turns = 6  
  `<usr>가나다<sys>라마바<usr>사아자<sys>차카타<usr>파하가<sys>나다라<usr>마바사<sys>아자차` =  
   `<usr>가나다<sys>라마바<usr>사아자<sys>차카타<usr>파하가<sys>나다라`, -- 6턴  
    `<usr>가나다<sys>라마바<usr>사아자<sys>차카타`, -- 4턴  
     `<usr>가나다<sys>라마바`... -- 2  

하지만 학습 시간만 늘어날 뿐 큰 성과는 없었음.

오히려 agumentation을 포기하면서 `__init__`이 아닌 `get_item`에서 작업을 수행할 경우 연산 속도가 더 빠름

- class ChatBot
```python
class ChatBot:
    """
        __init__ : 챗봇 모델 생성
            Args : model, tokenizer, Config
        
        train : 모델 학습 진행
            Args : epochs, train_data, (validation_data), (save)
        
        load_model : 모델 불러오기
            Args : PATH
        
        save_model : 모델 저장하기
            Args : PATH
        
        talk : 챗봇 대화하기
            대화 종료 멘트 : quit
    """
    
    def __init__(self, model, tokenizer, Config):
        """
            Args : model, tokenizer, Config
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = Config.device
        self.name = Config.model_name
        self.optim = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
        
#         self.user_token_id = tokenizer.get_vocab()[Config.usr_token]
#         self.bot_token_id = tokenizer.get_vocab()[Config.sys_token]
        self.user_token_id = tokenizer.PieceToId(Config.usr_token)
        self.bot_token_id = tokenizer.PieceToId(Config.sys_token)
        self.max_len = Config.max_len
        self.max_turns = Config.max_turns
        
        self.losses = []
        self.val_losses = []
    
    def train(self, epochs, train_data, validation_data=None, save=None):
        """
            epochs, train_data, validation_data=None, save=None
            save : epoch마다 모델을 저장할 경로/파일명
        """
        self.model.to(self.device)
        for epoch in range(epochs):
            self.model.train()
            print(f"\n Epoch {epoch+1}/{epochs}", sep="\n")
            starttime = time.time()
            batch_loss = []

            for i, batch in enumerate(train_data):
                input_ids, token_type_ids, labels = batch
                input_ids, token_type_ids, labels = \
                    input_ids.to(self.device), token_type_ids.to(self.device), labels.to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids = token_type_ids,
                    labels = labels
                )
                
                loss, logits = outputs[0], outputs[1]
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                batch_loss.append(loss.item())
                
                print(self.status(i+1, len(train_data), time.time()-starttime, np.mean(batch_loss)), end='\r')

            self.losses.append(np.mean(batch_loss))
            
            if validation_data:
                val_loss = self.validation(validation_data)
                print(self.status(i+1, len(train_data), time.time()-starttime, np.mean(batch_loss)) + \
                      " | val_loss : %.6f"%(val_loss), end='\r')
                self.val_losses.append(val_loss)
            
            if save:
                PATH = f'{save}_epochs-{epoch+1}_loss-{np.mean(batch_loss)}.pth'
                torch.save(self.model.state_dict(), PATH)
                
    def validation(self, validation_data):
        self.model.eval()
        batch_loss = []
        
        with torch.no_grad():
            for i, batch in enumerate(validation_data):
                input_ids, token_type_ids, labels = batch
                input_ids, token_type_ids, labels = \
                    input_ids.to(self.device), token_type_ids.to(self.device), labels.to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids = token_type_ids,
                    labels = labels
                )
                
                loss, logits = outputs[0], outputs[1]
                batch_loss.append(loss.item())
            
            valid_loss = np.mean(batch_loss)
        
        return valid_loss

    @staticmethod
    def status(step, step_len, time, loss):
        return "step : %d/%d - %ds | loss : %.6f | %.2fit/s"%(
            step,
            step_len,
            int(time),
            loss,
            step/time
        )
    
    def load_model(self, PATH):
        """
            PATH : pth 파일이 저장된 경로
        """
        self.model.load_state_dict(torch.load(PATH))
        print("model loaded.")
    
    def save_model(self, PATH=None):
        """
            PATH : 저장할 파일 경로/이름, 생략시 모델 이름과 현재 시간을 파일명으로 지정함
        """
        if not PATH:
            name = self.name.replace("/", "-")
            PATH = f"./{name}_{time.strftime('%Y-%m-%d %H:%M:%S')}.pth"
        torch.save(self.model.state_dict(), PATH)
        print("model saved.")
      
  -- talk 메서드 생략
```
model.fit 외에도 원하는 기능들을 사용하기 위해 허깅페이스의 모델 학습 코드를 공부하기보다 직접 class를 만들어 사용함.

필요했던 기능으로는 간단한 학습 수행 , 체크포인트 저장 메서드, 체크포인트 불러오기 메서드, loss 리스트
      
- SentencePiece
```python
import pandas as pd
import sentencepiece as spm
from tqdm import tqdm

data = pd.read_csv('spacing_data.csv', encoding='utf-8')
conversations = data['conversation'].tolist()

def get_tokenizer(corpus, lang, vocab_size=52000, s_tokens=False):
    temp_file = f"./SentencePiece_{lang}.txt"
    
    with open(temp_file, "w", encoding='utf-8') as f:
        for i in tqdm(range(len(corpus))):
            f.write(corpus[i] + '\n')
    
    print("file saved..")
    
    prefix = f"SentencePiece_{lang}"
    
    spm.SentencePieceTrainer.Train(
        f"--input={temp_file} --model_prefix={prefix} --vocab_size={vocab_size}" + 
        " --model_type=bpe" +
        " --pad_id=0" + # pad (0)
        " --unk_id=1" + # unknown (1)
        " --bos_id=2" + # begin of sequence (2)
        " --eos_id=3"   # end of sequence (3)
    )
    print("train has been finish")

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(f"{prefix}.model")
    print("model loaded..")
    
    if s_tokens:
        tokenizer.set_encode_extra_options("bos:eos")
    
    corpus = [tokenizer.EncodeAsIds(sentence) for sentence in corpus]
    print("tokenizing has been finish")
    
    return corpus, tokenizer

sentences, tokenizer = get_tokenizer(conversations, '249388')
# tokenizer = spm.SentencePieceProcessor()
# tokenizer.load("SentencePiece_02-04.model")

# src_corpus = [tokenizer.EncodeAsIds(sentence) for sentence in corpus['document']]
```

허깅페이스의 토크나이저는 vocabulary가 정해져 있어서 커스텀 데이터에 맞는 커스텀 토크나이저가 있다면 어떨까 시도해봄.

결과는 

```RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
```

오류를 마지막으로 실험 종료.

---
### 참고 문헌
1. [songys/AwesomeKorean_Data: 한국어 데이터 세트 링크](https://github.com/songys/AwesomeKorean_Data)
2. [자유대화형식의 음성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=109)
3. [STT모델 및 TTS모델 개발](https://www.youtube.com/watch?v=WTul6LIjIBA)
4. [온라인 구어체 말뭉치 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=625)
5. [법률 지식 베이스](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=99)
6. [파이썬으로 JSON 파일 다루기](https://www.youtube.com/watch?v=s9D-JIuaFqY&t=433s)
7. [korean SmileStyle Dataset](https://www.google.com/url?q=https://github.com/smilegate-ai/korean_smile_style_dataset&sa=D&source=docs&ust=1672048006339662&usg=AOvVaw2KWZl71R1gdPiznFcT1tkG)
8. [주제별 텍스트 일상생활 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=543)
9. [한국어 대화 요약](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=117)
10. [허깅페이스 모델](https://huggingface.co/lcw99/ko-dialoGPT-korean-chit-chat)
11. [[NLP] 언어모델의 평가지표 'Perplexity' 개념 및 계산방법](https://heytech.tistory.com/344)
12. [무슨 대화든 할 수 있는 에이전트를 향하여](https://brunch.co.kr/@synabreu/35)
13. [PyTorch 2.0 무엇이 다른가?](https://blog.naver.com/october-eight/222948663006)
14. [Tensorflow_KoGPT2_Chabot](https://github.com/ukairia777/tensorflow-kogpt2-chatbot/blob/main/KoGPT2_Chatbot.ipynb)
15. [GPT-2 Fine Tuning ](https://blog.naver.com/ds_penaut/222699897818)
16. [CaFeCoKe/KoGPT2_Chatbot](https://github.com/CaFeCoKe/KoGPT2_Chatbot)
---
### 팀원 깃허브 링크

- [방승욱](https://github.com/Ukbang)
- [구본회](https://github.com/HughBGrant) 
- [이태환](https://github.com/leetaehwan) 
- [장문규](https://github.com/MunGyuJang)

---
### Google Drive
- [구글 드라이브 링크](https://drive.google.com/drive/folders/13xvDPcMMqEe8cVTOg3VBjc0IgSjOAX9E)


 
