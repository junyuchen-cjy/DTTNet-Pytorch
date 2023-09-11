import os
from src.utils import get_unique_save_path

cur_suffix = -1

def get_train_log_dir(a, b, *, _parent_):
    global cur_suffix
    logbase = os.environ.get("LOG_DIR", "dtt_logs")
    if not os.path.exists(logbase):
        os.mkdir(logbase)
    dir_path = os.path.join(logbase, f"{a}_{b}")
    if cur_suffix == -1:
        cur_suffix = get_unique_save_path(dir_path)
        return f"{dir_path}_{str(cur_suffix)}"
    else:
        return f"{dir_path}_{str(cur_suffix)}"

cur_suffix1 = -1

def get_eval_log_dir(ckpt_path, *, _parent_):
    # get environment variable logbase
    global cur_suffix1
    logbase = os.environ.get("LOG_DIR", "dtt_logs")
    if not os.path.exists(logbase):
        os.mkdir(logbase)
    ckpt_name = os.path.basename(ckpt_path)
    # remove the suffix
    ckpt_name = ckpt_name.split(".")[0]
    dir_path = os.path.join(logbase, f"eval_{ckpt_name}")
    if cur_suffix1 == -1:
        cur_suffix1 = get_unique_save_path(dir_path)
        return f"{dir_path}_{str(cur_suffix1)}"
    else:
        return f"{dir_path}_{str(cur_suffix1)}"

cur_suffix2 = -1

def get_sweep_log_dir(a, b, *, _parent_):
    global cur_suffix2
    logbase = os.environ.get("LOG_DIR", "dtt_logs")
    if not os.path.exists(logbase):
        os.mkdir(logbase)
    dir_path = os.path.join(logbase, f"m_{a}_{b}")
    if cur_suffix2 == -1:
        cur_suffix2 = get_unique_save_path(dir_path)
        return f"{dir_path}_{str(cur_suffix2)}"
    else:
        return f"{dir_path}_{str(cur_suffix2)}"