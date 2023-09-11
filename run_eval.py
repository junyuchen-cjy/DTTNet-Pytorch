import re

import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pandas as pd
from src.utils.omega_resolvers import get_eval_log_dir

dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="evaluation.yaml", version_base='1.1')
def main(config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934

    from src.utils import utils

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    if not config.single:
        from src.evaluation.eval import evaluation
        cSDR, uSDR = evaluation(config)
        df = pd.DataFrame(columns=['cSDR', 'uSDR'])
        df = df._append({'cSDR': cSDR, 'uSDR': uSDR}, ignore_index=True)
        log_dir = os.getcwd()
        df.to_csv(os.path.join(log_dir, "eval.csv"), index=False)
    else:
        from src.evaluation.eval_demo import evaluation
        evaluation(config, 0)

if __name__ == "__main__":
    OmegaConf.register_new_resolver("get_eval_log_dir", get_eval_log_dir)
    main()
