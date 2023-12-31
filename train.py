import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.omega_resolvers import get_train_log_dir, get_sweep_log_dir
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
from pytorch_lightning.utilities import rank_zero_info

from src.utils import print_config_tree

dotenv.load_dotenv(override=True)

@hydra.main(config_path="configs/", config_name="config.yaml", version_base='1.1')
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.train import train
    from src.utils import utils

    rank_zero_info(OmegaConf.to_yaml(config))

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # - forcing multi-gpu friendly configuration
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        print_config_tree(config, resolve=True)

    # Train model
    train(config)



if __name__ == "__main__":
    # register resolvers with hydra key run.dir
    OmegaConf.register_new_resolver("get_train_log_dir", get_train_log_dir)
    OmegaConf.register_new_resolver("get_sweep_log_dir", get_sweep_log_dir)
    main()
