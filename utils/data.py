import os, json, re
import numpy as np
import sentencepiece as spm
import tensorflow as tf

from typing import Dict, List
from random import shuffle, randrange, random
from tqdm import tqdm
#from eunjeon import Mecab


def emotion_to_cntxt(emotion_path: str, cntxt_path: str):
    """build emotion data into context data
        
    Args:
        emotion_path (str): emotion data file(.json) path
        cntxt_path (str): context data file(.json) path 
    """

    emotion_f = open(emotion_path, "r", encoding="utf-8")
    cntxt_f = open(cntxt_path, "w", encoding="utf-8")

    emotions = json.load(emotion_f)
    cntxt_f.write("[")
    for i, emotion in tqdm(enumerate(emotions), desc=f"building contexts"):
        cntxt = {
            "cntxt": [
                emotion["talk"]["content"]["HS01"],
                emotion["talk"]["content"]["SS01"],
                emotion["talk"]["content"]["HS02"],
                emotion["talk"]["content"]["SS02"],
                emotion["talk"]["content"]["HS03"],
                emotion["talk"]["content"]["SS03"]
            ]
        }
        cntxt_f.write(json.dumps(cntxt, ensure_ascii=False))

        if i != len(emotions) - 1: cntxt_f.write(",\n")
    cntxt_f.write("]")

    emotion_f.close()
    cntxt_f.close()


def cntxt_to_inst(vocab_path: str,
                  cntxt_path: str,
                  inst_path: Dict,
                  uttr_len: int,
                  cntxt_len: int):
    """ convert cntxt to inst

       Args:
            vocab_path (str): vocabulary file(.json) path 
            cntxt_path (str): context file(.json) path 
            inst_path ({'idx': str, 'token': str}): instance file(.json) path
            uttr_len (int): utterance max length
            cntxt_len (int): context max length
        """
        
    # 0. Load vocabulary
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(vocab_path)

    # 1. uttr -> idx
    cntxt_f = open(cntxt_path, "r", encoding="utf-8")
    cntxts = json.load(cntxt_f)

    idxs = []
    for i, cntxt in tqdm(enumerate(cntxts), desc=f"Tokenizing {cntxt_path}..."):
        idx = []
        for uttr in cntxt["cntxt"]:
            if len(uttr) == 0: continue
            idx.append(tokenizer.encode_as_ids(uttr))
        idxs.append(idx)

    cntxt_f.close()

    # 2. build inst.json(idx -> instance)
    idx_f = open(inst_path['idx'], "w", encoding="utf-8")
    token_f = open(inst_path['token'], "w", encoding="utf-8")

    cntxt_size = len(idxs)
    idx_f.write("["),  token_f.write("[")

    for i, idx in tqdm(enumerate(idxs), desc=f"Padding..."):
        cntxt, idx_len = [], len(idx) - 1

        # uttr_idx
        mask_idx = randrange(0, idx_len) 

        # select mask_uttr
        rand = random()
        if rand < 0.8: # 80% replace with [CLS] [MASK] [SEP]
            mask_uttr = [6]
        elif rand < 0.9: # 10% keep original
            mask_uttr = idx[mask_idx]
        else: # 10% random uttr
            cntxt_idx = i
            while(cntxt_idx == i): cntxt_idx = randrange(0, cntxt_size)
            uttr_idx = randrange(0, len(idxs[cntxt_idx]) - 1)
            mask_uttr = idxs[cntxt_idx][uttr_idx]
        idx.append(mask_uttr)
                
        # uttr padding
        for j, uttr in enumerate(idx):
            if len(uttr) > uttr_len - 2:
                idx[j] = [5] + uttr[:uttr_len - 2] + [4]
            else:
                idx[j] = [5] + uttr + [4] + [0 for _ in range(uttr_len - 2 - len(uttr))]
        
        # resp, mask_uttr, cntxt
        resp, mask_uttr, cntxt = idx[-2], idx[-1], idx[:-2]

        # cntxt padding
        pad_uttr = [0 for _ in range(uttr_len)]
        for _ in range(cntxt_len - idx_len):
            cntxt.append(pad_uttr)

        # shuf
        shuf = [i for i in range(idx_len)]
        shuffle(shuf)
        shuf += [i + idx_len \
            for i in range(cntxt_len - idx_len)]

        token_cntxt = []
        for uttr in cntxt:
            token_cntxt.append(tokenizer.IdToPiece(uttr))

        token = {
            "cntxt": [tokenizer.IdToPiece(uttr) for uttr in cntxt],
            "mask_uttr": tokenizer.IdToPiece(mask_uttr),
            "mask_idx": mask_idx,
            "shuf": shuf,
            "resp": tokenizer.IdToPiece(resp),
        }

        idx = {
            "cntxt": cntxt,
            "mask_uttr": mask_uttr,
            "mask_idx": mask_idx,
            "shuf": shuf,
            "resp": resp,
        }

        idx_f.write(json.dumps(idx, ensure_ascii=False))
        token_f.write(json.dumps(token, ensure_ascii=False))

        if i != len(cntxts) - 1:    
            idx_f.write(",\n")
            token_f.write(",\n")
    idx_f.write("]")
    token_f.write("]")

    idx_f.close()
    token_f.close()


def build_vocab(cntxt_path: str, vocab_path: str, vocab_size: int):
    """build vocab
        
    Args:
        cntxt_path (str): context data file(.json) path 
        vocab_path (str): vocabulary file(.json) path 
        vocab_size (int): vocabulary size
    """

    mecab = Mecab()
    cntxt_f1 = open(cntxt_path['train'], "r", encoding="utf-8")
    cntxt_f2 = open(cntxt_path['val'], "r", encoding="utf-8")
    morphs_f = open("./morphs.txt", "w", encoding="utf-8")

    train_cntxt = json.load(cntxt_f1)
    eval_cntxt = json.load(cntxt_f2)

    # 1. morphs
    cntxts = []
    # 1-1. train data
    for i, cntxt in tqdm(enumerate(train_cntxt), desc=f"building training morpheme..."):
        cntxts += cntxt["cntxt"]
        if i % 1000 == 0 or i + 1 == len(train_cntxt):
            morphs = mecab.morphs(" ".join(cntxts))
            morphs_f.write(" ".join(morphs))
            cntxts = []
    

    # 1-2. eval data
    for i, cntxt in tqdm(enumerate(eval_cntxt), desc=f"building eval morpheme..."):
        cntxts += cntxt["cntxt"]
        if i % 1000 == 0 or i + 1 == len(eval_cntxt):
            morphs = mecab.morphs(" ".join(cntxts))
            morphs_f.write(" ".join(morphs))
            cntxts = []

    # 2. sentencepiece
    spm.SentencePieceTrainer.train(
        f"--input=./morphs.txt --model_prefix={vocab_path} --vocab_size={vocab_size}" + 
        " --model_type=bpe" +
        " --max_sentence_length=99999999" + # max length
        " --pad_id=0 --pad_piece=[PAD]" + # pad (0)
        " --unk_id=1 --unk_piece=[UNK]" + # unknown (1)
        " --bos_id=2 --bos_piece=[BOS]" + # begin of sequence (2)
        " --eos_id=3 --eos_piece=[EOS]" + # end of sequence (3)
        " --user_defined_symbols=[SEP],[CLS],[MASK]"
    ) 

    os.remove('./morphs.txt')


def get_emotion_dataset(inst_file: str, batch_size: int):
    """get emotion numpy dataset
    
    Args:
        inst_file (str): instance file(.json) path
        (default="./data/emtion/inst.json")
        batch_size (int): batch size
    """

    if not os.path.isfile(inst_file):
        raise Exception (f"{inst_file} doesn't exist")

    cntxts = []
    mask_uttrs = []
    mask_idxs = []
    shufs = []
    resps = []

    inst_f = open(inst_file, "r", encoding="utf-8")
    insts = json.load(inst_f)
    for inst in tqdm(insts, desc="Loading instances..."):
        cntxts.append(np.array(inst["cntxt"], dtype=np.float32))
        mask_uttrs.append(np.array(inst["mask_uttr"], dtype=np.float32))
        mask_idxs.append(np.array(inst["mask_idx"], dtype=np.int32))
        shufs.append(np.array(inst["shuf"], dtype=np.int32))
        resps.append(np.array(inst["resp"], dtype=np.float32))
    
    inst_f.close()

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                'cntxts': np.array(cntxts),
                'mask_uttrs': np.array(mask_uttrs),
                'mask_idxs': np.array(mask_idxs),
                'resps': np.array(resps),
                'shufs': np.array(shufs),
            },
        )
    )

    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)

    return dataset


def get_nug_emotion_dataset(inst_file: str, batch_size: int):
    """get nug emotion numpy dataset
    
    Args:
        inst_file (str): instance file(.json) path
        (default="./data/emtion/inst.json")
        batch_size (int): batch size
    """

    if not os.path.isfile(inst_file):
        raise Exception (f"{inst_file} doesn't exist")

    cntxts = []
    resps = []

    inst_f = open(inst_file, "r", encoding="utf-8")
    insts = json.load(inst_f)
    for inst in tqdm(insts, desc="Loading instances..."):
        cntxts.append(np.array(inst["cntxt"], dtype=np.float32))
        resps.append(np.array(inst["resp"], dtype=np.float32))
    
    inst_f.close()

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                'cntxts': np.array(cntxts),
                'resps': np.array(resps)
            },
        )
    )

    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)

    return dataset


def token_to_sentence(token: List[str]):
    """convert token to sentence
    
    Args:
        token (List[str]): token list
    """
    token = "".join(token)
    token = token.replace("▁", " ")
    token = token.replace("[CLS]", "")
    token = token.replace("[PAD]", "")
    sentence = re.sub("\[SEP\].*", "", token)

    return sentence


if __name__=='__main__':
    """1. primitive to context
    emotion_path = "../data/emotion/primitive_val.json"
    cntxt_path = "../data/emotion/cntxt_val.json"
    emotion_to_cntxt(emotion_path, cntxt_path)
    """

    """2. build vocab
    cntxt_path = {
        "train": "../data/emotion/cntxt_train.json",
        "val": "../data/emotion/cntxt_val.json"
    }
    vocab_path = "../data/emotion/spm"
    vocab_size = 32000
    build_vocab(cntxt_path, vocab_path, vocab_size)
    """

    """3. preprocess data
    vocab_path = "../data/emotion/spm.model"
    cntxt_path = "../data/emotion/cntxt_train.json"
    inst_path = {
        "idx": "../data/emotion/idx_train.json",
        "token": "../data/emotion/token_train.json"
    }

    cntxt_to_inst(vocab_path, cntxt_path, inst_path, uttr_len=30, cntxt_len=5)
    """

    """sampling preprocessor data"""
    inst_path = "../data/emotion/token_train.json"
    inst_f = open(inst_path, "r", encoding="utf-8")
    insts = json.load(inst_f)

    a = [i for i in range(5000)]
    shuffle(a)
    print(a[0])
    sample = insts[a[0]]
    for i in range(len(sample["cntxt"])):
        print(f'\nuttr{i}: {sample["cntxt"][i]}')
    print(f'\nresp: {sample["resp"]}')
    print(f'\nmask_idx: {sample["mask_idx"]}')
    print(f'\nmask_uttr: {sample["mask_uttr"]}')
    print(f'\nshuf: {sample["shuf"]}')

    for i in range(len(sample["cntxt"])):
        print(f'\nuttr len{i}: {len(sample["cntxt"][i])}')
    print(f'\nresp len: {len(sample["resp"])}')
    print(f'\nmask_idx len: {sample["mask_idx"]}')
    print(f'\nmask_uttr len: {len(sample["mask_uttr"])}')
    print(f'\nshuf len: {len(sample["shuf"])}')