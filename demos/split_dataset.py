import dotenv
import hydra
from omegaconf import DictConfig

import os
from os.path import exists


dotenv.load_dotenv(override=True)

@hydra.main(config_path="../configs/datamodule", config_name="musdb_dev14.yaml", version_base='1.1')
def main(config: DictConfig):
    trainset_path = os.path.join(config.data_dir, 'train')
    validset_path = os.path.join(config.data_dir, 'valid')
    validation_set = config.validation_set
    if not exists(validset_path):
        from shutil import move
        os.mkdir(validset_path)
        for track in validation_set:
            # if exist
            old_path = os.path.join(trainset_path,track)
            if os.path.exists(old_path):
                new_path = os.path.join(validset_path,track)
                move(old_path, new_path)
                print(f"Moved {old_path} to {new_path}")
    return

if __name__ == "__main__":
    main()