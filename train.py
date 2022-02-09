import wandb
import tensorflow as tf

from typing import Dict
from tensorflow.keras import optimizers, callbacks
from wandb.keras import WandbCallback
from hydra import initialize, compose

from model.model import DialogNug, DialogAll
from utils.data import get_nug_emotion_dataset, get_emotion_dataset


def pretrain(config_path: str, config_name: str, data_path: Dict, save_path: Dict):
    """pretrain
    
    Args:
        config_name (str): config file name
        config_path (str): config path
        data_path ({'train': str, 'val': str}): train and val data path
        save_path ({'dialog': str, 'gen': str}): dialogBERT and Generator weights path
    """

    # Wandb & Hydra
    initialize(config_path)
    cfg = compose(config_name)
    #wandb.init(project="dialogBERT", entity="gj98", config=cfg)

    # DATASET
    train_dataset = get_emotion_dataset(data_path["train"], cfg.processing.batch_size)
    val_dataset = get_emotion_dataset(data_path["val"], cfg.processing.batch_size) 

    print(f'train dataset: {train_dataset}\n')
    print(f'val dataset: {val_dataset}\n')
    
    # MODEL
    model = DialogAll(
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

    # COMPILE
    model.compile(
        optimizer=optimizers.Adam(
            cfg.training.learning_rate,
            beta_1=cfg.training.beta1,
            beta_2=cfg.training.beta2,
            epsilon=cfg.training.eps),
    )

    # TRAIN
    model.fit(
        train_dataset,
        epochs=cfg.training.epoch,
        validation_data=val_dataset,
        callbacks=[
            #WandbCallback(), 
            callbacks.ReduceLROnPlateau(patience=1),
            callbacks.EarlyStopping(patience=3)
        ]
    )

    # SAVE
    model.dialog.save_weights(save_path['dialog'])
    model.gen.save_weights(save_path['gen'])


def pretrain_tpu(config_path, config_name, data_path, save_path):
    """pretrain
    
    Args:
        config_name (str): config file name
        config_path (str): config path
        data_path (dict): train and val data path
        save_path (dict): dialogBERT and Generator weights path
    """

    # TPU
    print("Tensorflow version " + tf.__version__)

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    except ValueError:
        raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

    # Wandb & Hydra
    initialize(config_path)
    cfg = compose(config_name)
    wandb.init(project="dialogBERT", entity="gj98", config=cfg)

    # DATASET
    train_dataset = get_emotion_dataset(data_path["train"], cfg.processing.batch_size)
    val_dataset = get_emotion_dataset(data_path["val"], cfg.processing.batch_size) 

    print(f'train dataset: {train_dataset}\n')
    print(f'val dataset: {val_dataset}\n')
    
    with tpu_strategy.scope():
        # MODEL
        model = DialogAll(
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

        # COMPILE
        model.compile(
            optimizer=optimizers.Adam(
                cfg.training.learning_rate,
                beta_1=cfg.training.beta1,
                beta_2=cfg.training.beta2,
                epsilon=cfg.training.eps),
        )

    # TRAIN
    model.fit(
        train_dataset,
        epochs=cfg.training.epoch,
        validation_data=val_dataset,
        callbacks=[
            WandbCallback(), 
            callbacks.ReduceLROnPlateau(patience=1),
            callbacks.EarlyStopping(patience=2)
        ]
    )

    # SAVE
    localhost_save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
    model.dialog.save_weights(save_path['dialog'], options=localhost_save_option)
    model.gen.save_weights(save_path['gen'], options=localhost_save_option)


if __name__=="__main__":
    version = "small_3_6"
    config_name = f"{version}.yaml"
    config_path = "./configs"
    data_path = {
        "train": "./data/emotion/idx_train.json",
        "val": "./data/emotion/idx_val.json"
    }
    save_path = {
        "dialog": f"./save/all/dialog_{version}",
        "gen": f"./save/all/gen_{version}"
    }

    type = False
    if type:
        pretrain(config_path, config_name, data_path, save_path)
    else:
        pretrain_tpu(config_path, config_name, data_path, save_path)