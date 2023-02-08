import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from message_check import message_check
from html_utils import build_html_chat


class ChatBot:
    def __init__(self, model_name='skt/kogpt2-base-v2', model_path='./model.pth'):
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.usr_token = '<usr>'
        self.pad_token = '<pad>'
        self.sys_token = '<sys>'
        self.unk_token = '<unk>'
        self.mask_token = '<mask>'
        self.max_length = 256
        self.max_turns = 8
        self.device = torch.device("cpu")
        self.model_name = "skt/kogpt2-base-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                bos_token=self.bos_token, eos_token=self.eos_token, unk_token=self.unk_token,
                pad_token=self.pad_token, mask_token=self.mask_token, model_max_length=self.max_length)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.history_limit = []
        self.chat_history = []
        

    def get_reply(self, user_message):
        # save message from the user
        self.chat_history.append({
            'text':user_message, 
            'time':str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).time().replace(microsecond=0))
        })

        if len(self.history_limit) == self.max_turns * 2:
            self.history_limit = self.history_limit[2: ]
            self.history_limit[0] = self.bos_token + self.history_limit[0]

        user_message_pull = self.usr_token + user_message + self.sys_token
        
        if len(self.history_limit) == 0:
            user_message_pull = self.bos_token + user_message_pull
        
        result = message_check(user_message)
        if result == '그건 나쁜말이야 그런말은 쓰면 안돼!':
            self.chat_history.append({
                'text':result, 
                'time':str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).time().replace(microsecond=0))
            })
            return 
        
        self.history_limit.append(user_message_pull)
    
        # encode the new user message to be used by our model
        message_ids = self.tokenizer.encode(''.join(self.history_limit), return_tensors='pt').to(self.device)
        
        bad_words = [self.tokenizer.encode(token) for token in [self.unk_token, self.sys_token, self.usr_token, 'ㅋ', 'ㅠ', 'ㅠㅠ', 'ㅜㅜ', 'ㅜ', 'ㅎ', 'ㅠㅜ', 'ㅜㅠ', 'ㅋㅎ', 'ㅎㅋ', '.', '하하']]
        
        if user_message == result:
            with torch.no_grad():
                reply_ids = self.model.generate(
                            message_ids,
                            temperature=0.9,
                            top_k=3,
                            top_p=0.95,
                            num_beams=3,
                            do_sample=True,
                            # max_length=1000,
                            no_repeat_ngram_size=3,
                            repetition_penalty=2.0,
                            penalty_alpha=2.0,
                            max_new_tokens=20,
                            length_penalty= 0.1,
                            early_stopping= True,
                            bos_token_id = self.tokenizer.convert_tokens_to_ids(self.sys_token),
                            eos_token_id = self.tokenizer.convert_tokens_to_ids(self.usr_token),
                            bad_words_ids = bad_words
                        )   

            decoded_ids = reply_ids[0, message_ids.shape[-1]:]
            
            if decoded_ids[-1] == self.tokenizer.eos_token_id:
              decoded_ids = decoded_ids[:-1]

            decoded_message = self.tokenizer.decode(
                decoded_ids
                # skip_special_tokens=True
            )
        else:
            decoded_message = result

        self.history_limit.append(decoded_message)

        # save reply from the bot
        self.chat_history.append({
            'text':decoded_message, 
            'time':str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).time().replace(microsecond=0))
        })

        return decoded_message