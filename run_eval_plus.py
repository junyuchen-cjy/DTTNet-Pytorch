import re

import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from src.evaluation.eval_plus import evaluation_plus

from src.utils.omega_resolvers import get_eval_log_dir
import pandas as pd
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="eval_plus.yaml", version_base='1.1')
def main(config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934

    from src.utils import utils

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # create dataframe with columns: testset, cSDR, uSDR
    df = pd.DataFrame(columns=['testset', 'cSDR', 'uSDR'])

    inst_lst = ["guitar_processed", "horn_processed", "siren_processed", "upfilters_processed", "vocalchops_processed"]

    cSDR, uSDR = evaluation_plus(config, [], [])
    df = df._append({'testset': "org", 'cSDR': cSDR, 'uSDR': uSDR}, ignore_index=True)

    for i in range(0, len(inst_lst)):
        extra_lst = [inst_lst[i]]
        if inst_lst[i] == "vocalchops_processed":
            pos_lst = ["vocalchops_processed"]
        else:
            pos_lst = []
        cSDR, uSDR = evaluation_plus(config, extra_lst, pos_lst)
        df = df._append({'testset': inst_lst[i], 'cSDR': cSDR, 'uSDR': uSDR}, ignore_index=True)

    extra_lst = inst_lst
    pos_lst = ["vocalchops_processed"]
    cSDR, uSDR = evaluation_plus(config, extra_lst, pos_lst)
    df = df._append({'testset': "all", 'cSDR': cSDR, 'uSDR': uSDR}, ignore_index=True)

    log_dir = os.getcwd()
    df.to_csv(os.path.join(log_dir, "eval_plus.csv"), index=False)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("get_eval_log_dir", get_eval_log_dir)
    main()
