## <p align ="center"> π°οΈ Aiffel Thon </p>

## <p align ="center"> νλ‘μ νΈ : μΈκ³΅μ§λ₯ λΉμλ₯Ό λ§λ€μ΄λ³΄μ! </p> 

### <p align ="center"> π€ Team ChatHuman π€ </p> 

### <p align ="center"> νμ </p>

 <p align ="center"> π€ΉββοΈ λ°©μΉμ± π΄ββοΈ κ΅¬λ³Έν ποΈββοΈ μ΄νν β·οΈ μ₯λ¬Έκ· </p>

### <p align ="center"> μ΄ νλ‘μ νΈλ 22.12.26 ~ 23.02.08μΌκΉμ§ μ§νλλ μμ΄ν ν€ νλ‘μ νΈ μλλ€. </p>
---
 
### μΈλΆ μΌμ 

|μμ|κΈ°κ°|μΈλΆ κ³ν|
|:---:|:---:|:---:|
|1λ²|22.12.26 ~ 22.12.30|νμλ€κ³Όμ κ³ν μ‘°μ¨|
|2λ²|23.01.02 ~ 23.02.03|λ°μ΄ν° μ μ²λ¦¬ λ° κ°λ° νκ²½ κ΅¬μΆ|
|3λ²|23.01.09 ~ 23.01.13|λͺ¨λΈ νμ€νΈ|
|4λ²|23.01.09 ~ 23.02.03|μΉνμ΄μ§ κ΅¬μΆ|
|5λ²|23.01.16 ~ 23.01.20|λͺ¨λΈ μ°κ΅¬ λ° μΈνΌλ°μ€ μ½λ μμ±|
|6λ²|23.01.16 ~ 23.01.27|λͺ¨λΈ νμ΅|
|7λ²|23.01.18 ~ 23.02.03|λͺ¨λΈ λ€λ¬κΈ°|
|8λ²|23.02.06 ~ 23.02.07|λ°ν μ€λΉ|
|9λ²|23.02.08|λ°ν|
 
---
### κ°μ
- μΌμμνμ λμμ μ£Όλ©° νμ¬ μ‘΄μ¬νλ μλ¦¬, ν΄λ‘λ², μΉ΄μΉ΄μ€μ κ°μ μΈκ³΅μ§λ₯ λΉμμκ² λΆμ¬λ κΈ°λ₯μ νμ¬νλκ²μ λͺ©νλ‘ ν¨.
  - λΆμ¬λ κΈ°λ₯μ΄λΌν¨μ μΌμλν κΈ°λ₯μ μ΄μ μ λ§μΆλ©° κ·Έ μΈμ μ¬μν κ²λ€μ μΆκ°ν΄λ³Ό μμ .   
  
- κΈ°λ³Έμ μΈ ννλ μ±λ΄ ννμ΄λ©° λ€μν ν€μλλ₯Ό ν΅ν΄ λͺλ Ήμ μνν  μ μλλ‘ ν  κ².

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
- AIHub μμ μ κ³΅νλ μ£Όμ λ³ νμ€νΈ μΌμμν λ°μ΄ν°μ νκ΅­μ΄ λν μμ½μ μ΄μ©νμ¬ λ§λ¦.
- λ°μ΄ν°νλ μ ννλ‘ ν λνμ λ§λ­μΉλ₯Ό Conversation columnμΌλ‘ κ΅¬λΆνκ³  κ° λν κ° λ°νμλ₯Ό `'<usr>'`, `'<sys>'` ν ν°μΌλ‘ κ΅¬λΆνμμ.
- μ½ 19λ§κ°μ λνλ₯Ό μ΄μ©ν¨. 

![data_image](https://github.com/Ukbang/Aiffel_thon/blob/main/images/data_image.jpeg)

---
### μ μ²λ¦¬ λ°©μ
- modules/preprocessing.py νμΌμ clear_sentence ν¨μλ₯Ό μ΄μ©νμ¬ μ²λ¦¬.
- #@μ΄λ¦#μ make_name ν¨μλ₯Ό μ΄μ©νμ¬ λλ€ν μ΄λ¦μ μμ±ν  μ μλλ‘ νμμ.
- @URL, #@μμ€ν#μ¬μ§#, #@μ΄λͺ¨ν°μ½#μ μ­μ νμκ³  λ°λ³΅λλ γ,γ,γ,γ ,. κ³Ό κ°μ λ¬Έμλ 2κ°λ‘ ν΅μΌνμμΌλ©° μμ£Ό λ±μ₯νλ ν€ν€ λ γγ λ‘ λ³κ²½νμμ.

![make_name](https://github.com/Ukbang/Aiffel_thon/blob/main/images/make_name.jpeg)
![clear_sentence](https://github.com/Ukbang/Aiffel_thon/blob/main/images/clear_sentence.jpeg)

---

### λͺ¨λΈ
- λͺ¨λΈμ π€Hugging Faceμμ μ κ³΅νλ gpt2 λͺ¨λΈμ μ¬μ©νμμ.
- λ² μ΄μ€ λͺ¨λΈλ‘ ['skt/kogpt2-base-v2'](https://github.com/SKT-AI/KoGPT2) μ μ¬μ©ν¨.   
 
 
<p align ="center"><img src="https://user-images.githubusercontent.com/112140135/216884750-53fb4373-2d9d-4a6a-800b-0062b8b702f5.png" width="800px" height="300px"></p>

---
### νμ΅ μ§νκ³Όμ  λ¦¬λλ³΄λ
#### Data type
__Topic = 250000κ°__
 
 __Topic+kakao Data = 190000κ° ('`<usr>`λ‘ λλλ λ¬Έκ΅¬ μ­μ ', κΈΈμ΄ 256')__
 
__kakao Data = 65000κ°__

---

#### Label

__Input = Inputκ³Ό Labelμ΄ λμΌ__
 
 
__-100 = λ§μ§λ§ `<sys>` λνλ₯Ό μ μΈν -100μ μ΄μ©ν Masking__

__-100+sys = λͺ¨λ  `<sys>` λνλ₯Ό μ μΈν λͺ¨λ  λν -100μΌλ‘ Masking__
 
 
__Shift = Inputμ `<s>` ν ν°μ bos_tokenμΌλ‘ μ¬μ©, Labelμ `</s>`ν ν°μ eos_tokenμΌλ‘ μ¬μ©ν¨.__

|index|Model|Epochs|Data type|μ§ν μν©|μ§ν μΌμ|Label|Loss|Val_Loss|Comment|μ±λ₯|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-01-31|-100|4.290 -> 3.797 -> 3.340 -> 2.803 -> 2.195|3.821 -> 3.759 -> 3.804 -> 3.938 -> 4.143|λ¨λ΅νμ΄κ³  λνκ° μ μ΄λ£¨μ΄ μ§μ§ μμ.|[Link](https://github.com/Ukbang/Aiffel_thon/blob/main/chatbot/Test/23-02-01_-100_test.ipynb)|
|2|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-02-01|Input|1.476 -> 1.343 -> 1.270 -> 1.203 -> 1.137|1.486 -> 1.445 -> 1.434 -> 1.441 -> 1.461|νμ¬κΉμ§ κ°μ₯ Best|[Link](https://github.com/Ukbang/Aiffel_thon/blob/main/chatbot/Test/23-02-01_True_test.ipynb)|
|3|skt/kogpt2-base-v2|3|kakao Data|Done|2023-01-30|Input|2.330 -> 2.147 -> 2.084|1.765 -> 1.723 -> 1.704|λ¬Έμ₯ μμ±μ eos token λ°μ λͺ»ν¨.|[Link](https://github.com/Ukbang/Aiffel_thon/blob/main/chatbot/Test/Inference_code_label_True_len384.ipynb)|
|4|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-02-01|shift|2.140 -> 2.005 -> 1.931 -> 1.864 -> 1.794|2.298 -> 2.236 -> 2.215 -> 2.221 -> 2.246|νμ΅μ΄ μ ν λμ§ μμμ. νκΈ°|[Link]()|
|5|skt/kogpt2-base-v2|10|Topic+kakao|Done|2023-02-01|Input|1.483 -> 1.352 -> 1.275 -> 1.206 -> 1.135 -> 1.062 -> 0.986 -> 0.908 -> 0.830 -> 0.753|1.504 -> 1.469 -> 1.456 -> 1.463 -> 1.485 -> 1.517 -> 1.562 -> 1.616 -> 1.683 -> 1.759|5epoch μ΄μλΆν° νμ΅μ΄ μ€νλ € μλ¨. νκΈ°|[Link]()|
|6|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-02-05|Input|1.479 -> 1.380 -> 1.330 -> 1.292 -> 1.260|1.401 -> 1.370 -> 1.357 -> 1.350 -> 1.346|μμν μ½λ μμ ν νμ΅νμμ. 2λ²κ³Ό μ±λ₯μ΄ λμΌν¨.|[Link]()|
|7|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-02-05|-100|4.269 -> 3.869 -> 3.574 -> 3.296 -> 3.031|4.069 -> 3.997 -> 3.989 -> 4.032 -> 4.106|.....|[Link]()|
|8|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-02-05|-100+sys|3.826 -> |3.352 -> |.....|[Link]()|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|

---

### μΆκ°λ μλΉμ€
<p align ="center"><img src="https://user-images.githubusercontent.com/112140135/216884826-5905e7cb-229a-4a53-becd-25508e40fd1d.png" width="600px" height="900px"></p>

---

### νκ³ 

- μ»€μ€ν λ°μ΄ν° λ‘λ 
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
μ²« μ€κ³λ κΈΈμ΄/ν΄ μκ° κΈ΄ λ°μ΄ν°λ λ·λΆλΆμ μλ₯΄κ³  μλΆλΆμ λ¨κΈ°λ λ°©μμΌλ‘ μλν¨

`<usr>κ°λλ€<sys>λΌλ§λ°<usr>μ¬μμ(κΈΈμ΄/ν΄μμ΄κ³Ό)<sys>μ°¨μΉ΄ν` :  
`<usr>κ°λλ€<sys>λΌλ§λ°` -- μλ₯Ό λλ λ¬΄μ‘°κ±΄ sys λνλ‘ μ’λ£

```python
for i in tqdm(range(len(self.data))):
            hists = []
            dials = self.data[i]
            
            for u, utter in enumerate(dials):
                if u % 2 == 0:
                    hists.append(self.usr_token + utter)  # Speaker 1: User
                else:
                    hists.append(self.sys_token + utter)  # Speaker 2: System
            
            max_turn = max(len(hists), self.max_turns) # max_turnsμ λμΌλ©΄ post
            if max_turn % 2 != 0: max_turn -= 1 # user λνλ‘ λλ¨ λ°©μ§
                
            for f in range(max_turn, 1, -2):
                contexts = hists[:f]
                if sum([len(l) for l in contexts]) > self.max_len-2: continue # bos_tokenμ eos_token ν ν°μ μΆκ°νκΈ° μν΄ -2
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
breakμ λ¨κ²¨λμ κ²½μ° λ°μ΄ν° νλλΉ μ΅λ ν κ°μ λ°μ΄ν°λ₯Ό μμ±νμ§λ§  
breakμ μ κ±°ν  κ²½μ° κ°λ₯ν λͺ¨λ  ν΄μλ₯Ό λͺ¨λ λ°μ΄ν°λ‘ μ¬μ©ν΄ agumentationν¨

  ex: max_turns = 6  
  `<usr>κ°λλ€<sys>λΌλ§λ°<usr>μ¬μμ<sys>μ°¨μΉ΄ν<usr>ννκ°<sys>λλ€λΌ<usr>λ§λ°μ¬<sys>μμμ°¨` =  
   `<usr>κ°λλ€<sys>λΌλ§λ°<usr>μ¬μμ<sys>μ°¨μΉ΄ν<usr>ννκ°<sys>λλ€λΌ`, -- 6ν΄  
    `<usr>κ°λλ€<sys>λΌλ§λ°<usr>μ¬μμ<sys>μ°¨μΉ΄ν`, -- 4ν΄  
     `<usr>κ°λλ€<sys>λΌλ§λ°`... -- 2  

νμ§λ§ νμ΅ μκ°λ§ λμ΄λ  λΏ ν° μ±κ³Όλ μμμ.

μ€νλ € agumentationμ ν¬κΈ°νλ©΄μ `__init__`μ΄ μλ `get_item`μμ μμμ μνν  κ²½μ° μ°μ° μλκ° λ λΉ λ¦

- class ChatBot
```python
class ChatBot:
    """
        __init__ : μ±λ΄ λͺ¨λΈ μμ±
            Args : model, tokenizer, Config
        
        train : λͺ¨λΈ νμ΅ μ§ν
            Args : epochs, train_data, (validation_data), (save)
        
        load_model : λͺ¨λΈ λΆλ¬μ€κΈ°
            Args : PATH
        
        save_model : λͺ¨λΈ μ μ₯νκΈ°
            Args : PATH
        
        talk : μ±λ΄ λννκΈ°
            λν μ’λ£ λ©νΈ : quit
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
            save : epochλ§λ€ λͺ¨λΈμ μ μ₯ν  κ²½λ‘/νμΌλͺ
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
            PATH : pth νμΌμ΄ μ μ₯λ κ²½λ‘
        """
        self.model.load_state_dict(torch.load(PATH))
        print("model loaded.")
    
    def save_model(self, PATH=None):
        """
            PATH : μ μ₯ν  νμΌ κ²½λ‘/μ΄λ¦, μλ΅μ λͺ¨λΈ μ΄λ¦κ³Ό νμ¬ μκ°μ νμΌλͺμΌλ‘ μ§μ ν¨
        """
        if not PATH:
            name = self.name.replace("/", "-")
            PATH = f"./{name}_{time.strftime('%Y-%m-%d %H:%M:%S')}.pth"
        torch.save(self.model.state_dict(), PATH)
        print("model saved.")
      
  -- talk λ©μλ μλ΅
```
model.fit μΈμλ μνλ κΈ°λ₯λ€μ μ¬μ©νκΈ° μν΄ νκΉνμ΄μ€μ λͺ¨λΈ νμ΅ μ½λλ₯Ό κ³΅λΆνκΈ°λ³΄λ€ μ§μ  classλ₯Ό λ§λ€μ΄ μ¬μ©ν¨.

νμνλ κΈ°λ₯μΌλ‘λ κ°λ¨ν νμ΅ μν λͺλ Ή, μ²΄ν¬ν¬μΈνΈ μ μ₯ λ©μλ, μ²΄ν¬ν¬μΈνΈ λΆλ¬μ€κΈ° λ©μλ, loss λ¦¬μ€νΈ

μ κ° νλ μ€νλ€λ‘λ μ£Όλ‘ μ΄λ€ μ’λ₯μ λ°μ΄ν°λ₯Ό νμ΅ν΄μΌ νμ΅μ΄ μ λλκ°λ₯Ό μμ£Όλ‘ μ€ννλλ° μ’μ μ±κ³Όλ μμμ.
      
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

νκΉνμ΄μ€μ ν ν¬λμ΄μ λ vocabularyκ° μ ν΄μ Έ μμ΄μ μ»€μ€ν λ°μ΄ν°μ λ§λ μ»€μ€ν ν ν¬λμ΄μ κ° μλ€λ©΄ μ΄λ¨κΉ μλν΄λ΄.

κ²°κ³Όλ νμ΅μ λ§μΉ ν λ¬Έμ₯ μμ±μ μλν  κ²½μ°

```RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
```

μ€λ₯λ₯Ό λ§μ§λ§μΌλ‘ μ€ν μ’λ£.

- HTML
```html
<!------
This code is an adaptation of the original work of pablocorezzola 
provided at https://bootsnipp.com/snippets/y8e4W
under the MIT license

Avatar icons were obtained from:

bot:
https://www.flaticon.com/free-icon/bot_1786548?term=bot%20avatar&page=1&position=1&page=1&position=1&related_id=1786548&origin=search

human:
https://www.flaticon.com/premium-icon/man_2202112?term=avatar&page=1&position=2&page=1&position=2&related_id=2202112&origin=search
                
---------->

<!DOCTYPE html>
<html>
    <head>
        <script src="http://code.jquery.com/jquery-1.11.1.min.js"></script>
        <link href="https://use.fontawesome.com/releases/v5.0.7/css/all.css" rel="stylesheet">
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css" rel="stylesheet" id="bootstrap-css">
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/3.0.0/js/bootstrap.min.js"></script>
        <link rel="stylesheet" href="{{ url_for('static', path='/styles.css') }}">
		<style>
			.cont_a{
                position:relative;
                display:flex;
            }
			.col-sm-4.col-sm-offset-4.frame{
				min-width:450px;
                min-height:720px;
			}
            .using_manual{
				background-color:#f0f0f0;
				border-top:20px solid #9999ff;
				border-bottom:20px solid #9999ff;
				border-left:2px solid #9999ff;
				border-right:2px solid #9999ff;
				border-radius:10px;
				position:relative;
                padding:15px;
				margin:20px;
				height:auto;
				box-shadow: 5px 5px 10px;
            }
            body{
                font-size:small;
            }
            
            
		</style>
    </head>
    <body>
        <div class="cont_a">
            <div class="col-sm-4 col-sm-offset-4 frame">
                <ul>
                    {{ chat|safe }}
                </ul>
                <div>
                    <form class="col-sm-12" name="frm1" method="post" action="/">
                        <div class="input-group">
                            <input type="text" class="form-control" name="message" placeholder="Type a message">
                            <span class="btn input-group-addon glyphicon glyphicon-share-alt" onclick="frm1.submit()"></span>
                        </div>
                    </form>            
                </div>
            </div>
			
			<div class="using_manual">
                <center><h4>μ±ν΄λ¨Όνμ μ±λ΄ μ΄λμ μ°Ύμμ£Όμμ κ°μ¬ν©λλ€.</h4></center>
                <p>μ ν¬μκ²λ μ΄λ° κΈ°λ₯λ€μ΄ μμ΅λλ€.</p>
                <p><span><b><u>λ‘λ, λ μ¨, μκ°, μμ΄μ€νμ κ΅¬λΆ, μ£Όμ κ²μ</u></b></span></p>
                <p><span><b>1. μκ°μ μλ €λλ¦½λλ€.</b></span></p>
                <p><span><i>β μ¬μ©λ² : μ΄λμ μ§κΈ λͺμμΌ?</i></span></p>
                <p><span><i>          μ΄λμ μκ° μλ €μ€.</i></span></p>
                <p><span><b>2. λ μ¨λ₯Ό μλ €λλ¦½λλ€.</b></span></p>
                <p><span><i>β μ¬μ©λ² : κ³ μλ λ μ¨ μ΄λ?</i></span></p>
                <p><span><i>(κ³ μλμ΄λΌ ν λ μμ° λ¨μκ΅¬μ κ³ μλμ μ°μ μΌλ‘ κ²μν¨.)</i></span></p>
                <p><span><i>	μΈμ² κ³ μλ λ μ¨ μ΄λ?</i></span></p>
                <p><span><i>(μλ₯Ό κ΅¬λΆν΄μ€μΌλ‘ μΈμ² κ³ μλμ λ μ¨λ₯Ό κ²μ κ°λ₯.)</i></span></p>
                <p><span><b>3. λ‘λλ²νΈλ₯Ό λ½μλλ¦½λλ€.</b></span></p>
                <p><span><i>β μ¬μ©λ² : μ΄λμ λ‘λλ²νΈ λ½μμ€.</i></span></p>
                <p><span><i>          μ΄λμ λ‘λλ²νΈ μΆμ²ν΄μ€.</i></span></p>
                <p><span><b>4. μ€μκ° μ£Όμ κ°κ²©μ μλ €λλ¦½λλ€.</b></span></p>
                <p><span><b>β μ¬μ©λ² : μΌμ±μ μ μ£Όκ° μλ €μ€.</b></span></p>
                <p><span><i>          μΌμ±μ μ μ£Όμ μΌλ§μΌ?</i></span></p>
                <p><span><i>          μΌμ±μ μ μ£Όκ° μΌλ§μΌ?</i></span></p>
                <p><span><b>5. νκΈμ΄ μμ³μ Έλ κ΄μ°?μ΅λλ€.</b></span></p>
                <p><span><i>β μ¬μ©λ² : dk rkqwkrl gksrmfdl dkscuwu('μ κ°μκΈ° νκΈμ΄ μμ³μ Έ')</i></span></p>
                <p><span><i>          dkssud('μλ')</i></span></p>
				<div align="right" style="font-size:8px;">AIFFEL μΈμ²μΊ νΌμ€ 3κΈ° μ±ν΄λ¨Όν</div>
            </div>
        </div>
        <script>
            window.onload = function() {
            var a = document.querySelector("ul");
            a.scrollTop = a.scrollHeight;
			document.frm1.message.focus();
            }
        </script>
    </body>
    <footer>
        <a class="footer" href="https://www.flaticon.com/free-icons" title="icons">
            Icons created by Freepik - Flaticon
        </a>
    </footer>
<html>
```
λ¨μν μ€λͺ λ¬Έκ΅¬λ₯Ό uiλ‘ νννκ³ , λͺ κ°μ§ μ΄μλ₯Ό μμ νλ λ§μ§ μμ μμμ΄ νμνμ§λ§  
HTMLμ λ°°μλ³Έ μ μ΄ μμ΄μ μλΉν λ§μ μκ°μ΄ μμλλ€.

νΉν μ±νμ μΉλ©΄ μ€ν¬λ‘€μ΄ λ§¨ μλ‘ μ¬λΌκ°λ λ¬Έμ , λ§€ μ±νλ§λ€ νμ€νΈ λ°μ€μ ν¬μ»€μ±μ΄ νλ¦¬λ λ¬Έμ λ₯Ό ν΄κ²°νκΈ° μν΄  
μ³μ λ°©λ²μ μλμ§λ§ μ½λλ₯Ό μμ νκΈ°λ³΄λ€ ν¨μλ₯Ό λ?μ΄μμ μ’μ§ λͺ»ν μ½λκ° λ¨.

- FastAPI
```python
import uvicorn
from typing import Optional
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from html_utils import build_html_chat
from model import ChatBot

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import time

app = FastAPI()

# initialises the chatbot model and starts the uvicorn app
chatbot = ChatBot()

# mounts the static folder that contains the css file
app.mount("/static", StaticFiles(directory="static"), name="static")

# locates the template files that will be modified at run time
# with the dialog form the user and bot
templates = Jinja2Templates(directory="templates")

@app.post("/", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
async def root(request:Request, message: Optional[str] = Form(None)):
  # if the Form is not None, then get a reply from the bot
  if message is not None:
  
    # gets a response of the AI bot
    _ = chatbot.get_reply(message)

    # converts the chat history into an HTML dialog
    chat_html = '\n'.join([
      build_html_chat(is_me=i%2==0, text=msg['text'], time=msg['time'])
      for i, msg in enumerate(chatbot.chat_history)
    ])
  
  else: # μ΅μ΄ μ°κ²° μ index.html κ°±μ μ μν λλ―Έ μ νΈμ΄μ§λ§ μλ¬΄κ²λ μλ ₯νμ§ μμ μ±νμ λ³΄λΌ μ μ±ν λ‘κ·Έκ° μ¬λΌμ Έλ³΄μ΄λ λ¬Έμ  λ°μ
    chat_html = ''
    
  message_dict = {
    "request": request,
    "chat": chat_html
  }
  
  # returns the final HTML
  return templates.TemplateResponse("index.html", message_dict)


# initialises the chatbot model and starts the uvicorn app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
μ¬λ¬κ°μ§ μ΄μ μ€ κ°μ₯ ν΄κ²°νκ³  μΆμλ λ¬Έμ λ‘ μ μ  λνκ° λ¨Όμ  μ¬λΌκ°κ³  μμ€ν λνκ° μμ±λλ©΄ μμ°¨μ μΌλ‘ μ¬λΌκ°μ§ μκ³   
μ μ  λνλ₯Ό λ°μΌλ©΄ μμ€ν λνκΉμ§ μμ±ν λ€μ ν λ²μ λ κ°μ λ¬Έμ₯μ΄ μλ°μ΄νΈ λλ μ΄μκ° μμλλ°,

μ λ§ λ§μ μλλ₯Ό ν΄λ΄€μ§λ§ κ²°κ΅­ ν΄κ²°νμ§ λͺ»ν¨.

### μ°Έκ³  λ¬Έν
1. [songys/AwesomeKorean_Data: νκ΅­μ΄ λ°μ΄ν° μΈνΈ λ§ν¬](https://github.com/songys/AwesomeKorean_Data)
2. [μμ λννμμ μμ± λ°μ΄ν°](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=109)
3. [STTλͺ¨λΈ λ° TTSλͺ¨λΈ κ°λ°](https://www.youtube.com/watch?v=WTul6LIjIBA)
4. [μ¨λΌμΈ κ΅¬μ΄μ²΄ λ§λ­μΉ λ°μ΄ν°](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=625)
5. [λ²λ₯  μ§μ λ² μ΄μ€](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=99)
6. [νμ΄μ¬μΌλ‘ JSON νμΌ λ€λ£¨κΈ°](https://www.youtube.com/watch?v=s9D-JIuaFqY&t=433s)
7. [korean SmileStyle Dataset](https://www.google.com/url?q=https://github.com/smilegate-ai/korean_smile_style_dataset&sa=D&source=docs&ust=1672048006339662&usg=AOvVaw2KWZl71R1gdPiznFcT1tkG)
8. [μ£Όμ λ³ νμ€νΈ μΌμμν λ°μ΄ν°](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=543)
9. [νκ΅­μ΄ λν μμ½](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=117)
10. [νκΉνμ΄μ€ λͺ¨λΈ](https://huggingface.co/lcw99/ko-dialoGPT-korean-chit-chat)
11. [[NLP] μΈμ΄λͺ¨λΈμ νκ°μ§ν 'Perplexity' κ°λ λ° κ³μ°λ°©λ²](https://heytech.tistory.com/344)
12. [λ¬΄μ¨ λνλ  ν  μ μλ μμ΄μ νΈλ₯Ό ν₯νμ¬](https://brunch.co.kr/@synabreu/35)
13. [PyTorch 2.0 λ¬΄μμ΄ λ€λ₯Έκ°?](https://blog.naver.com/october-eight/222948663006)
14. [Tensorflow_KoGPT2_Chabot](https://github.com/ukairia777/tensorflow-kogpt2-chatbot/blob/main/KoGPT2_Chatbot.ipynb)
15. [GPT-2 Fine Tuning ](https://blog.naver.com/ds_penaut/222699897818)
16. [CaFeCoKe/KoGPT2_Chatbot](https://github.com/CaFeCoKe/KoGPT2_Chatbot)
---
### νμ κΉνλΈ λ§ν¬

- [λ°©μΉμ±](https://github.com/Ukbang)
- [κ΅¬λ³Έν](https://github.com/HughBGrant) 
- [μ΄νν](https://github.com/leetaehwan) 
- [μ₯λ¬Έκ·](https://github.com/MunGyuJang)

---
### Google Drive
- [κ΅¬κΈ λλΌμ΄λΈ λ§ν¬](https://drive.google.com/drive/folders/13xvDPcMMqEe8cVTOg3VBjc0IgSjOAX9E)


 
