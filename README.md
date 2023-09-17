# DTT-Bespoke

To fine-tune the model with the bespoke dataset. For vocals only.



## Generate Bespoke Dataset

1. Prepare a folder named 'samplepacks', with 5 subfolders:
    ```
    .
    ├── guitar
    ├── horn
    ├── siren
    ├── upfilters
    └── vocalchops
    ```

    The subfolders should contain corresponding audio segments.



2. generate bespoke-train, be-spoke-valid, bespoke-test

   ```
   # make the clips has length roughly between 4-8s
   # This will create extra 5 folders with suffix 'processed'
   python src/bespoke/merge_clips.py
   
   # sample the segments and generate bspoke train/val/test
   python src/bespoke/generate_partitions.py
   ```

   Note that you will need to edit:

   - ```samplepacks_root``` in ```merge_clip.py```
   - ```musdb_root```, ```sample_pack_root``` , ```tmp_root``` and ```target_root```.
     - ```musdb_root``` is essential since it provides length information for generating b-spoke val/test
     - ```sample_pack_root``` stores all the processed segments
     - ```tmp_root``` stores temp segments partitions, the partitions will be merged into synthetic songs.
     - ```target_root``` will be the final dataset



## Train

```
export extended_dataset=xxx # target_root
export pretrained_ckpt_path=xxx

python train.py experiment=ft_vc datamodule=musdb_dev14 trainer=default

# or if you don't want to use logger

python train.py experiment=ft_vc datamodule=musdb_dev14 trainer=default logger=[]
```

Params:

- experiment
  - ft_debug: single GPU, just to make sure that the code can run fluently
  - ft_vc: the vocal chops are included in the training set
  - ft_nvc: the vocal chops are not included in the training set



## Evaluation

Change ```pool_workers``` in ```configs\evual_plus```. You can set the number as the number of cores in your CPU.

```
export ckpt_path=xxx # for Windows, replace the 'export' with 'set'
export extended_dataset=xxxx # target_root

python run_eval_plus.py model=vocals logger=[]

# or if you don't want to use logger

python run_eval_plus.py model=vocals eval_dir=G:/musdb18hq/ logger=[]
```

This will test the model on MUSDB18-HQ + 5 extra test sets + 1 fully combined test set.

The result will be saved as eval.csv under the folder  ```LOG_DIR\basename(ckpt_path)_suffix```



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






