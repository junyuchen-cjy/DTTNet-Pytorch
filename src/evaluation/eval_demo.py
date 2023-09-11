from os import listdir
from pathlib import Path
from typing import Optional, List

from concurrent import futures
import hydra
import wandb
import os
import shutil
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.loggers import Logger, WandbLogger

from tqdm import tqdm
import numpy as np
from src.callbacks.wandb_callbacks import get_wandb_logger
from src.evaluation.separate import separate_with_onnx_TDF, separate_with_ckpt_TDF
from src.utils import utils
from src.utils.utils import load_wav, sdr, get_median_csdr, save_results, get_metrics

from src.utils import pylogger
import soundfile as sf
log = pylogger.get_pylogger(__name__)


def evaluation(config: DictConfig, idx):

    assert config.split in ['train', 'valid', 'test']

    data_dir = Path(config.get('eval_dir')).joinpath(config['split'])
    assert data_dir.exists()

    model = hydra.utils.instantiate(config.model)
    target_name = model.target_name
    ckpt_path = Path(config.ckpt_path)
    is_onnx = os.path.split(ckpt_path)[-1].split('.')[-1] == 'onnx'
    shutil.copy(ckpt_path,os.getcwd()) # copy model

    datas = sorted(listdir(data_dir))
    if len(datas) > 27: # if not debugging
        # move idx 27 to head
        datas = [datas[27]] + datas[:27] + datas[28:]


    track = datas[idx]
    track = data_dir.joinpath(track)
    print(track)
    mixture = load_wav(track.joinpath('mixture.wav')) # (c, t)
    target = load_wav(track.joinpath(target_name + '.wav'))
    if model.audio_ch == 1:
        mixture = np.mean(mixture, axis=0, keepdims=True)
        target = np.mean(target, axis=0, keepdims=True)
    #target_hat = {source: separate(config['batch_size'], models[source], onnxs[source], mixture) for source in sources}
    if is_onnx:
        target_hat = separate_with_onnx_TDF(config.batch_size, model, ckpt_path, mixture)
    else:
        target_hat = separate_with_ckpt_TDF(config.batch_size, model, ckpt_path, mixture, config.device, config.double_chunk, overlap_factor=config.overlap_factor)

    bssmetrics, perms, ssdr = get_metrics(target_hat, target, mixture, sr=44100,version=config.bss)
    # dump bssmetrics into pkl
    import pickle
    with open(os.path.join(os.getcwd(),'bssmetrics.pkl'),'wb') as f:
        pickle.dump(bssmetrics,f)

    return bssmetrics






