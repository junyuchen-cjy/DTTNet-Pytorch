# Dual-Path TFC-TDF UNet

A Pytorch Implementation of the Dual-Path TFC-TDF UNet for Music Source Separation. DTTNet achieves 10.11 dB cSDR on vocals with 86% fewer parameters compared to BSRNN (SOTA). Our paper is coming soon.



## Environment Setup (First Time)

1. Download MUSDB18HQ from https://sigsep.github.io/datasets/musdb.html
2. (Optional) Edit the validation_set in configs/datamodule/musdb_dev14.yaml
3. Create Miniconda/Anaconda environment

```
conda env create -f conda_env_gpu.yaml -n DTT
source /root/miniconda3/etc/profile.d/conda.sh
conda activate DTT
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd) # for Windows, replace the 'export' with 'set'
```

4. Edit .env file according to the instructions. It is recommended to use wandb to manage the logs.

```
cp .env.example .env
vim .env
```



## Environment Setup (After First Time)

Once all these settings are configured, the next time you simply need to execute these code snippets to set up the environment

```
source /root/miniconda3/etc/profile.d/conda.sh
conda activate DTT
```



## Inference

1. Download checkpoints from: https://mega.nz/folder/E4c1QD7Z#OkgM_dEK1tC5MzpqEBuxvQ
2. Run code

```
python run_infer.py model=vocals ckpt_path=xxxxx mixture_path=xxxx
```

The files will be saved under the folder ```PROJECT_ROOT\infer\```



Parameter Options:

- model=vocals, model=bass, model=drums, model=other



## Evaluation

```
export ckpt_path=xxx # for Windows, replace the 'export' with 'set'

python run_eval.py model=vocals eval_dir=G:/musdb18hq/ logger.wandb.name=xxxx

# or if you don't want to use logger
python run_eval.py model=vocals eval_dir=G:/musdb18hq/ logger=[]
```

The result will be saved as eval.csv under the folder  ```LOG_DIR\basename(ckpt_path)_suffix```



Parameter Options:

- model=vocals, model=bass, model=drums, model=other



## Train

Note that you will need:

- 1 TB disk space for data augmentation. Otherwise, delete values of the ```aug_params``` in ```configs/datamodule/musdb18_hq.yaml```. This will train the model without data augmentation.
- 2 A40 (48GB). Or equivalently, 4 RTX 3090 (24 GB). Otherwise, change the ```datamodule.batch_size``` to a smaller one and ```trainer.devices``` to 1 in ```configs/experiment/vocals_dis.yaml```.

### 1. Data Partition 
```
python demos/split_dataset.py # data partition
```


### 2. Data Augmentation (Optional)

```
# install aug tools
sudo apt-get update
sudo apt-get install soundstretch

mkdir /root/autodl-tmp/tmp

# perform augumentation
python src/utils/data_augmentation.py --data_dir /root/autodl-tmp/musdb18hq/
```

### 2. Run code

```
python train.py experiment=vocals_dis datamodule=musdb_dev14 trainer=default

# or if you don't want to use logger

python train.py experiment=vocals_dis datamodule=musdb_dev14 trainer=default logger=[]
```



## Referenced Repositories

1. TFC-TDF UNet
   1. https://github.com/kuielab/sdx23
   2. https://github.com/kuielab/mdx-net
   3. https://github.com/ws-choi/sdx23
   4. https://github.com/ws-choi/ISMIR2020_U_Nets_SVS
2. BandSplitRNN
   1. https://github.com/amanteur/BandSplitRNN-Pytorch
3. fast-reid (Sync BN)
   1. https://github.com/JDAI-CV/fast-reid






