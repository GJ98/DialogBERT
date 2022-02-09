import json
import tensorflow as tf
import sentencepiece as spm
from typing import Dict
from tqdm import tqdm
from hydra import initialize, compose

from utils.data import token_to_sentence
from model.chatbot import *
from model.model import DialogBERT, Generator


def inference(config_name: str, 
              config_path: str, 
              save_path: Dict, 
              vocab_path: str, 
              data_path: str):
    """inference
    
    Args:
        config_name (str): config file name
        config_path (str): config path
        save_path ({'dialog': str, 'gen': str}): dialogBERT and Generator weights path
        vocab_path (str): vocabulary path
        data_path (str): sample data path
    """
    initialize(config_path)
    cfg = compose(config_name)

    # MODEL
    dialog = DialogBERT(
        cfg.model.vocab_size,
        cfg.processing.uttr_len,
        cfg.processing.cntxt_len,
        cfg.model.d_h,
        cfg.model.head,
        cfg.model.d_ff,
        cfg.model.uttr_layer,
        cfg.model.cntxt_layer,
        cfg.processing.p,
    )

    gen = Generator(
        cfg.model.vocab_size,
        cfg.processing.uttr_len,
        cfg.model.d_h,
        cfg.model.head,
        cfg.model.d_ff,
        cfg.model.uttr_layer,
        cfg.processing.p,
    )

    dialog.load_weights(save_path['dialog'])
    gen.load_weights(save_path['gen'])

    type = input('MODEL TYPE\n' +
    '  1. GreedyChatbot\n' +
    '  2. BeamChatbot\n' +
    '  3. CheatAllChatbot\n' +
    '  4. CheatFirstChatbot\n:')

    if type == '1':
        chatbot = GreedyChatbot(dialog, gen)
    elif type == '2':
        chatbot = BeamChatbot(dialog, gen, 5, cfg.model.vocab_size)
    elif type == '3':
        chatbot = CheatAllChatbot(dialog, gen)
    elif type == '4':
        chatbot = CheatFirstChatbot(dialog, gen)
    else:
        return 

    # TOKENIZER
    sp = spm.SentencePieceProcessor()
    sp.Load(vocab_path)

    # DATA
    inst_f = open(data_path, "r", encoding="utf-8")
    insts = json.load(inst_f)

    cntxts, resps, idxs = [], [], []
    for i, inst in tqdm(enumerate(insts), desc="Loading instances..."):
        cntxts.append(inst['cntxt'])
        resps.append(inst['resp'])
        idx = []
        for uttr in inst['cntxt']:
            idx.append(sp.PieceToId(uttr))
        if type == '3' or type == '4':
            idx.append(sp.PieceToId(inst['resp']))
        idxs.append(idx)

    inputs = tf.convert_to_tensor(idxs, dtype=tf.float32)

    # INFERENCE
    while True:
        i = input('num (exit == -1): ')
        i = int(i)
        if i == -1: exit()

        resp = tf.cast(chatbot(inputs[i]), dtype=tf.int32)

        print('context: ')
        for idx, uttr in enumerate(cntxts[i]):
            uttr = token_to_sentence(uttr)
            print(f'{idx}: {uttr}')

        if type == '1' or type == '3' or type == '4':
            print('predict response: ')
            predict = sp.IdToPiece(resp.numpy().tolist())
            predict = token_to_sentence(predict)
            print(predict)
        
        elif type == '2':
            print('predict response: ')
            predict = sp.IdToPiece(resp[0].numpy().tolist())
            predict = token_to_sentence(predict)
            print(predict)

            print('candidate response: ')
            for idx, cand in enumerate(resp[1:]):
                predict = sp.IdToPiece(cand.numpy().tolist())
                predict = token_to_sentence(predict)
                print(f"cand{idx}: {predict}")

        print('true response: ')
        true = token_to_sentence(resps[i])
        print(true)


if __name__=="__main__":
    version = "base_2_3"
    config_name = f"{version}.yaml"
    config_path = "./configs"
    save_path = {
        "dialog": f"./save/dialog_{version}",
        "gen": f"./save/gen_{version}"
    }
    vocab_path = "./data/emotion/spm.model"
    data_path = "./data/emotion/token_val.json"
 
    inference(config_name, config_path, save_path, vocab_path, data_path)