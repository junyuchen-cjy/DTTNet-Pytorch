
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import soundfile as sf
from src.utils.utils import load_wav, get_unique_save_path
from src.utils.omega_resolvers import get_eval_log_dir
from pathlib import Path

import dotenv
from src.evaluation.separate import separate_with_ckpt_TDF, no_overlap_inference, overlap_inference
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="infer.yaml", version_base='1.1')
def main(config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934

    from src.utils import utils

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)


    model = hydra.utils.instantiate(config.model)
    ckpt_path = Path(config.ckpt_path)
    print(ckpt_path)
    mixture = load_wav(config.mixture_path)
    target_hat = separate_with_ckpt_TDF(config.batch_size, model, ckpt_path, mixture, config.device,
                                        config.double_chunk, config.overlap_add)


    base_name, file_name = os.path.split(config.mixture_path)
    #remove extension
    dir = os.path.splitext(file_name)[0]
    # print(config)

    save_path = os.path.join(config.paths.root_dir, "infer")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    dir = os.path.join(save_path, dir)
    cur_suffix = get_unique_save_path(dir)
    save_path = f"{dir}_{str(cur_suffix)}"

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path = os.path.join(save_path, model.target_name + ".wav")
    sf.write(save_path, target_hat.T, 44100)

    print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()
