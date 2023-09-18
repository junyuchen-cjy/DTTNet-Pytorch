import warnings
from copy import deepcopy
from importlib.util import find_spec
from typing import Callable, List

import re
import hydra
import museval
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
import pytorch_lightning as pl
import tempfile
import soundfile as sf
import shutil
import subprocess


from pytorch_lightning.utilities import rank_zero_only

from src.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished or failed
    - Logging the exception if occurs
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):

        # execute the task
        try:

            # apply extra utilities
            extras(cfg)

            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:

            # save exception to `.log` file
            log.exception("")

            # when using hydra plugins like Optuna, you might want to disable raising exception
            # to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:

            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # close loggers (even if exception occurs so multirun won't fail)
            close_loggers()

        return metric_dict, object_dict

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    Logger: List[pl.loggers.Logger] = []

    if not logger_cfg:
        log.warning("No Logger configs found! Skipping...")
        return Logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating Logger <{lg_conf._target_}>")
            Logger.append(hydra.utils.instantiate(lg_conf))

    return Logger


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("trainer.logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    #
    _cfg = deepcopy(cfg)
    OmegaConf.resolve(_cfg)
    hparams["datamodule"] = _cfg["datamodule"]
    hparams["trainer"] = _cfg["trainer"]

    hparams["callbacks"] = _cfg.get("callbacks")
    # hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = _cfg.get("task_name")
    hparams["tags"] = _cfg.get("tags")
    hparams["ckpt_path"] = _cfg.get("ckpt_path")
    hparams["seed"] = _cfg.get("seed")

    # send hparams to all loggers
    for Logger in trainer.loggers:
        Logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


@rank_zero_only
def wandb_login(key):
    wandb.login(key=key)

from functools import reduce
import pandas as pd

def print_single_result(x, target_list, metrics):
    df = pd.DataFrame(x)
    df.columns = target_list
    df.index = metrics
    print(df)


def concat_results(results, reduce_per_song):
    '''
    return: metrics_num x len(target_list) x no.
    '''
    if reduce_per_song:
        medians = []
        for i in range(len(results)):
            medians.append(np.nanmedian(results[i], axis=2, keepdims=True))

        total = reduce(lambda x, y: np.concatenate((x, y), axis=2), medians)
    else:
        total = reduce(lambda x, y: np.concatenate((x, y), axis=2), results)
    return total

import pickle
import os

def save_results(model_path,results,target_list, perms, ssdrs):

    # if the path is dir
    if os.path.isdir(model_path):
        base_name = os.path.join(model_path, "bss_metrics")
    else: # remove suffix
        # base_name = os.path.splitext(model_path)[0]
        base_name = model_path

    # make path unique by assigning id
    id = 0
    save_path = f"{base_name}_{str(id)}.pkl"
    while os.path.isfile(save_path):
        save_path = f"{base_name}_{str(id)}.pkl"
        id = id + 1

    # save into file
    with open(save_path, "wb") as f:
        d = {'results': results,
             'target_list':target_list,
             'metrics':["SDR", "ISR", "SIR", "SAR"],
             'perms':perms,
             'ssdrs':ssdrs}
        pickle.dump(d, f)
        # np.savez(**d)

def get_metrics(target_hat, target, mixture, sr, version="naive"):
    ssdr = sdr(target_hat, target)  # c, t

    target_hat = target_hat.T # (t,c)
    target = target.T # (t,c)
    mixture = mixture.T # (t,c)
    references = np.stack([target, mixture - target]) # (nsrc, nsampl, nchan)

    ests = np.stack([target_hat, mixture - target_hat]) # (nsrc, nsampl, nchan)

    # evaluate
    if version == "official":
        SDR, ISR, SIR, SAR, perms = museval.metrics.bss_eval(references, ests, window=sr, hop=sr)
        bss_ret = np.stack((SDR, ISR, SIR, SAR), axis=0)
    elif version == "fast":
        SDR = simplified_msseval(references, ests, chunk_size=sr)
        bss_ret = np.expand_dims(SDR, axis=0)
        perms = []
    else:
        raise ValueError("version must be official or fast")

    return bss_ret, perms, ssdr

def get_median_csdr(results):
    total = concat_results(results, False)
    return np.nanmedian(total, axis=2)[0,0]

def sdr(est, ref):
    '''
    Args:
        est: (c, t)
        ref: (c, t)
    '''
    ratio = np.sum(ref ** 2) / np.sum((ref - est) ** 2)
    if ratio == 0:
        return np.nan
    return 10 * np.log10(ratio)

def simplified_msseval(references, ests, chunk_size):
    '''
    Args:
        ests: (n, t, c)
        references: (n, t, c)
    Returns:
        results: (src, chunks)
    '''
    results = []
    for i in range(ests.shape[0]):
        e = ests[i] # (t, c)
        r = references[i] # (t, c)
        # pad to multiple of chunk_size
        # e = np.pad(e, ((0, chunk_size - e.shape[0] % chunk_size), (0, 0)))
        # r = np.pad(r, ((0, chunk_size - r.shape[0] % chunk_size), (0, 0)))
        # remove the last chunk if it's not full (to align with museval)
        if e.shape[0] % chunk_size != 0:
            e = e[:-(e.shape[0] % chunk_size)]
            r = r[:-(r.shape[0] % chunk_size)]
        # split into chunks
        e_chunks = np.split(e, e.shape[0] // chunk_size)
        r_chunks = np.split(r, r.shape[0] // chunk_size)
        sdr_ls = []
        for j in range(len(e_chunks)):
            e_chunk = e_chunks[j]
            r_chunk = r_chunks[j]
            sdr_chunk = sdr(e_chunk, r_chunk)
            sdr_ls.append(sdr_chunk)
        results.append(np.array(sdr_ls))
    results = np.array(results) # (src, chunks)

    # remove silent chunks SDR (i.e. to nan, to algin with museval)
    is_nan = np.isnan(results)
    is_nan = np.sum(is_nan, axis=0, keepdims=True)  # to count
    is_nan = is_nan > 0  # to boolean
    is_nan = np.repeat(is_nan,results.shape[0], axis=0)  # (src,chunks)
    results[is_nan == 1] = np.nan
    return results

def load_wav(path, track_length=None, chunk_size=None):
    if track_length is None:
        return sf.read(path, dtype='float32')[0].T # (c, t)
    else:
        s = np.random.randint(track_length - chunk_size)
        return sf.read(path, dtype='float32', start=s, frames=chunk_size)[0].T

def get_unique_save_path(dir_path):
    '''
    Given dir_path, add number suffix to make if unique
    :param dir_path: the directory path or file path without suffix
    :return:
    '''
    # get the parent directory
    parent_dir = os.path.dirname(dir_path)
    # ls the parent directory
    ls = os.listdir(parent_dir)
    # get the folder name
    folder_prefix = os.path.basename(dir_path)
    # filter out the files with the same prefix

    ls = [f for f in ls if re.match(folder_prefix + "_\d+", f)]
    if len(ls) == 0:
        return 0
    else:
        # get the number suffix
        ls = [int(f.split("_")[-1]) for f in ls]
        return max(ls) + 1

# https://github.com/RetroCirce/Zero_Shot_Audio_Source_Separation/blob/main/utils.py
def split_nparray_with_overlap(array, hop_length, overlap_size):
    '''
    Args:
        array: np.array (t, c)
        array_size: n_segs
        overlap_size: int
    Returns:
        [(c, t_seg + overlap_size)]
    '''
    result = []
    n, k, s = array.shape[0], hop_length + overlap_size, hop_length
    p = s - (n - k) % s
    array = np.pad(array, ((0, p), (0, 0)), 'constant', constant_values=0)
    array_size = (n + p - k) // s + 1
    # print(f"padded array size = {array.shape}")
    # no_ovelap_size = int(len(array) / array_size)
    for i in range(array_size):
        offset = int(i * hop_length)
        # last_loop = i == array_size
        chunk = array[offset: offset + k]
        # chunk = chunk.copy()
        # if chunk.shape[0] < no_ovelap_size + overlap_size:
        #     print(f"zero padding = {no_ovelap_size + overlap_size - chunk.shape[0]}")
        #     chunk.resize((no_ovelap_size + overlap_size, 2), refcheck=False)
        # chunk.resize((element_size + overlap_size, 2), refcheck=False)  # zero padding
        result.append(chunk.T)
    return result

# https://github.com/RetroCirce/Zero_Shot_Audio_Source_Separation/blob/main/models/asp_model.py
def join_chunks(tmp_root, chunk_ls, samplerate, overlap_size):
    '''
    Args:
        chunk_ls: list of np.array (chunks, chunk_size, c)
        overlap_size: int
    '''
    # tmp_root = "G:\\sdx2023fork\\tmp"

    # song_dir = tmp_root
    # src_filename = "test.wav"
    # out_path = os.path.join(song_dir, src_filename)

    array_size = len(chunk_ls)
    tmpfolder = tempfile.TemporaryDirectory(dir=tmp_root)
    tmpfolder_name = tmpfolder.name
    tmpfolder.cleanup()

    out_path = os.path.join(tmpfolder_name, "out.wav")

    # print(tmpfolder_name)
    os.mkdir(tmpfolder_name)


    filters = []
    args = ["ffmpeg", "-y", "-loglevel", "quiet"]
    for i in range(array_size):
        # tmpfile = tempfile.NamedTemporaryFile(dir=tmpfolder.name, suffix=".wav",delete=True)
        # os.remove(tmpfile.name)
        tmpfile_name = os.path.join(tmpfolder_name, f"chunk_{i}.wav")
        args.extend(["-i", tmpfile_name])
        sf.write(tmpfile_name, data=chunk_ls[i], samplerate=samplerate, format='WAV')

        if i < array_size - 1:
            filter_cmd = "[" + ("a" if i != 0 else "") + "{0}][{1}]acrossfade=ns={2}:c1=tri:c2=tri".format(i, i + 1,
                                                                                                           overlap_size)

            if i != array_size - 2:
                filter_cmd += "[a{0}];".format(i + 1)

            filters.append(filter_cmd)

    args.extend([
        "-filter_complex",
        "".join(filters),
        "-y",
        out_path
    ])

    # print(" ".join(args))

    try:
        out = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE) # shell=False, otherwise cmd length exceed
        # print(out)
        # print(out.stderr.decode("gbk"))
    except:
        raise Exception("ffmpeg does not exist. Install ffmpeg or set config.overlap_rate to zero.")



    # remove zero padding
    test, sr = sf.read(out_path)
    # print(f"test.shape = {test.shape}")
    # os.remove(out_path)
    shutil.rmtree(tmpfolder_name)
    return test


@rank_zero_only
def wandb_watch_all(wandb_logger, model):
    wandb_logger.watch(model, 'all')

if __name__ == "__main__":
    import  os
    cur_dir = r"/root/unmix_plus/scripts/open-unmix"
    # ls all the files in the dir
    files = os.listdir(cur_dir)
    # filter the files with name start with "vocals"
    files = [file for file in files if file.startswith("vocals")]
    # get the number suffix
    files = [int(file.split("_")[-1].split(".")[0]) for file in files]

    # rename the files with suffix decrease by 1
    for file in files:
        os.rename(os.path.join(cur_dir, f"vocals_{file}.wav"), os.path.join(cur_dir, f"vocals_{file-1}.wav"))
