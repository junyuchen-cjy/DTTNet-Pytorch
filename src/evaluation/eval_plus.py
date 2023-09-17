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
import soundfile as sf

from tqdm import tqdm
import numpy as np
from src.callbacks.wandb_callbacks import get_wandb_logger
from src.evaluation.separate import separate_with_onnx_TDF, separate_with_ckpt_TDF
from src.utils import utils
from src.utils.utils import load_wav, sdr, get_median_csdr, save_results, get_metrics

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def evaluation_plus(config: DictConfig, extra_lst, pos_lst):

    assert config.split in ['train', 'valid', 'test']

    data_dir = Path(config.get('eval_dir')).joinpath(config['split'])
    assert data_dir.exists()

    model = hydra.utils.instantiate(config.model)
    target_name = model.target_name
    ckpt_path = Path(config.ckpt_path)
    is_onnx = os.path.split(ckpt_path)[-1].split('.')[-1] == 'onnx'
    shutil.copy(ckpt_path,os.getcwd()) # copy model

    ssdrs = []
    bss_lst = []
    bss_perms = []
    num_tracks = len(listdir(data_dir))
    target_list = [config.model.target_name,"complement"]


    neg_lst = [x for x in extra_lst if x not in pos_lst]


    pool = futures.ProcessPoolExecutor
    with pool(config.pool_workers) as pool:
        datas = sorted(listdir(data_dir))
        if len(datas) > 27: # if not debugging
            # move idx 27 to head
            datas = [datas[27]] + datas[:27] + datas[28:]
        # iterate datas with batchsize 8
        for k in range(0, len(datas), config.pool_workers):
            batch = datas[k:k + config.pool_workers]
            pendings = []
            for i, track in tqdm(enumerate(batch)):
                folder_name = track
                track = data_dir.joinpath(track)
                mixture = load_wav(track.joinpath('mixture.wav')) # (c, t)
                target = load_wav(track.joinpath(target_name + '.wav'))
                if config.extended_dataset:
                    test_set_path, track_name = os.path.split(track)
                    ext_folder = os.path.join(str(config.extended_dataset), "test", track_name)
                    pos_audios = []
                    neg_audios = []
                    for pos in pos_lst:
                        pos_audios.append(load_wav(os.path.join(ext_folder, pos + '.wav')))
                    for neg in neg_lst:
                        neg_audios.append(load_wav(os.path.join(ext_folder, neg + '.wav')))
                    if len(pos_audios) > 0:
                        pos = np.sum(pos_audios, axis=0)
                    else:
                        pos = np.zeros_like(mixture)
                    if len(neg_audios) > 0:
                        neg = np.sum(neg_audios, axis=0)
                    else:
                        neg = np.zeros_like(mixture)
                    mixture += pos + neg
                    target += pos

                if model.audio_ch == 1:
                    mixture = np.mean(mixture, axis=0, keepdims=True)
                    target = np.mean(target, axis=0, keepdims=True)
                #target_hat = {source: separate(config['batch_size'], models[source], onnxs[source], mixture) for source in sources}
                if is_onnx:
                    target_hat = separate_with_onnx_TDF(config.batch_size, model, ckpt_path, mixture)
                else:
                    target_hat = separate_with_ckpt_TDF(config.batch_size, model, ckpt_path, mixture, config.device, config.double_chunk)
                if config.save_separated:
                    sf.write(rf"G:\sdx2023fork\predictions\{k+i}_{folder_name}.wav", target_hat.T, 44100)


                pendings.append((folder_name, pool.submit(
                    get_metrics, target_hat, target, mixture, sr=44100,version=config.bss)))


            for i, (track_name, pending) in tqdm(enumerate(pendings)):
                pending = pending.result()
                bssmetrics, perms, ssdr = pending
                bss_lst.append(bssmetrics)
                bss_perms.append(perms)
                ssdrs.append(ssdr)


    # save_results(log_dir, bss_lst, target_list, bss_perms)

    cSDR = get_median_csdr(bss_lst)
    uSDR = sum(ssdrs)/num_tracks

    print(f"extra_lst: {extra_lst}, pos_lst: {pos_lst}, neg_lst: {neg_lst}")
    print(f"cSDR: {cSDR}, uSDR: {uSDR}")

    return cSDR, uSDR
