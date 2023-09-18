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


def evaluation(config: DictConfig):

    assert config.split in ['train', 'valid', 'test']

    data_dir = Path(config.get('eval_dir')).joinpath(config['split'])
    assert data_dir.exists()

    # Init Lightning loggers
    loggers: List[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                loggers.append(hydra.utils.instantiate(lg_conf))

        if any([isinstance(l, WandbLogger) for l in loggers]):
            utils.wandb_login(key=config.wandb_api_key)

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

                if model.audio_ch == 1:
                    mixture = np.mean(mixture, axis=0, keepdims=True)
                    target = np.mean(target, axis=0, keepdims=True)
                #target_hat = {source: separate(config['batch_size'], models[source], onnxs[source], mixture) for source in sources}
                if is_onnx:
                    target_hat = separate_with_onnx_TDF(config.batch_size, model, ckpt_path, mixture)
                else:
                    target_hat = separate_with_ckpt_TDF(config.batch_size, model, ckpt_path, mixture, config.device, config.double_chunk, config.overlap_add)


                pendings.append((folder_name, pool.submit(
                    get_metrics, target_hat, target, mixture, sr=44100,version=config.bss)))

                for wandb_logger in [logger for logger in loggers if isinstance(logger, WandbLogger)]:
                    mid = mixture.shape[-1] // 2
                    track = target_hat[:, mid - 44100 * 3:mid + 44100 * 3]
                    wandb_logger.experiment.log(
                        {f'track={k+i}_target={target_name}': [wandb.Audio(track.T, sample_rate=44100)]})


            for i, (track_name, pending) in tqdm(enumerate(pendings)):
                pending = pending.result()
                bssmetrics, perms, ssdr = pending
                bss_lst.append(bssmetrics)
                bss_perms.append(perms)
                ssdrs.append(ssdr)

                for logger in loggers:
                    logger.log_metrics({'song/ssdr': ssdr}, k+i)
                    logger.log_metrics({'song/csdr': get_median_csdr([bssmetrics])}, k+i)

    log_dir = os.getcwd()
    save_results(log_dir, bss_lst, target_list, bss_perms, ssdrs)

    cSDR = get_median_csdr(bss_lst)
    uSDR = sum(ssdrs)/num_tracks
    for logger in loggers:
        logger.log_metrics({'metrics/mean_sdr_' + target_name: sum(ssdrs)/num_tracks})
        logger.log_metrics({'metrics/median_csdr_' + target_name: get_median_csdr(bss_lst)})
        # get the path of the log dir
        if not isinstance(logger, WandbLogger):
            logger.experiment.close()

    if any([isinstance(logger, WandbLogger) for logger in loggers]):
        wandb.finish()

    return cSDR, uSDR
