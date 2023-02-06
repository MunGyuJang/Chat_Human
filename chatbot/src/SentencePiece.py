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
