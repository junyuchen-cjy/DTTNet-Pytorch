import os
from abc import ABCMeta, ABC
from pathlib import Path

import soundfile
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from tqdm import tqdm
import soundfile as sf

from src.utils.utils import load_wav
import numpy as np

from src import utils
log = utils.get_pylogger(__name__)

def check_target_name(target_name, source_names):
    try:
        assert target_name is not None
    except AssertionError:
        print('[ERROR] please identify target name. ex) +datamodule.target_name="vocals"')
        exit(-1)
    try:
        assert target_name in source_names or target_name == 'all'
    except AssertionError:
        print('[ERROR] target name should one of "bass", "drums", "other", "vocals", "all"')
        exit(-1)


def check_sample_rate(sr, sample_track):
    try:
        sample_rate = soundfile.read(sample_track)[1]
        assert sample_rate == sr
    except AssertionError:
        sample_rate = soundfile.read(sample_track)[1]
        print('[ERROR] sampling rate mismatched')
        print('\t=> sr in Config file: {}, but sr of data: {}'.format(sr, sample_rate))
        exit(-1)


class MusdbDataset(Dataset):
    __metaclass__ = ABCMeta

    def __init__(self, data_dir, chunk_size):
        self.source_names = ['bass', 'drums', 'other', 'vocals']
        self.chunk_size = chunk_size
        self.musdb_path = Path(data_dir)

def get_duration(path):
    return len(sf.SoundFile(path))

def get_samplepack_metadata(root_path):
    meta_data = {}
    folders = os.listdir(root_path)
    for f in folders:
        # list all the files in the folder
        type_folder = os.path.join(root_path, f)
        if not os.path.isdir(type_folder):
            continue
        if f == ".idea":
            continue
        file_lst = os.listdir(type_folder)
        new_lst = []
        for x in file_lst:
            wav_path = os.path.join(type_folder, x)
            dur = get_duration(wav_path)
            new_lst.append((wav_path, dur))
        if f in meta_data:
            meta_data[f] += new_lst
        else:
            meta_data[f] = new_lst
    return meta_data


def pick_from_sample_pack(meta_data, inst_lst, chunk_size):
    # randomly pick a sample type
    picked_type = random.choice(inst_lst)
    # randomly pick one from the folder
    file_path = meta_data[picked_type]
    picked_path, length = random.choice(file_path)

    segment_start = np.random.randint(low=0,
                                      high=abs(chunk_size - length))
    if length > chunk_size:
        # truncate
        segment_end = segment_start + chunk_size
        picked_file, sr = sf.read(picked_path, start=segment_start, stop=segment_end)
    else:
        picked_file, sr = sf.read(picked_path)

    picked_file = picked_file.T

    # normalize clip
    picked_file = picked_file / np.max(np.abs(picked_file))
    picked_file = picked_file * 0.5

    # ensure stereo
    if len(picked_file.shape) == 1:
        picked_file = np.repeat(picked_file[np.newaxis, :], 2, axis=0)

    # pad
    if length < chunk_size:
        arr = np.zeros((2, chunk_size))  # (c, t)
        segment_end = segment_start + length
        arr[:, segment_start:segment_end] = picked_file
        picked_file = arr

    if picked_type == "upfilters" and random.randint(0, 1)==1:
        # randomly reverse the upfilters
        picked_file = np.flip(picked_file, axis=1)

    return picked_file, picked_type

# def pad_or_truncate(picked_arr, picked_length, chunk_size):
#     arr = np.zeros((2, chunk_size)) # (c, t)
#     segment_start = np.random.randint(low=0,
#                                       high=abs(chunk_size - picked_length))
#
#     if picked_length <= chunk_size:
#         segment_end = segment_start + picked_length
#         arr[:, segment_start:segment_end] = picked_arr
#     else:
#         segment_end = segment_start + chunk_size
#         arr[:, :] = picked_arr[:, segment_start:segment_end]
#     return arr

class MusdbTrainDataset(MusdbDataset):
    def __init__(self, data_dir, chunk_size, target_name, aug_params, external_datasets, single_channel, epoch_size, extended_dataset, pos_lst, exclude_lst, **kwargs):
        super(MusdbTrainDataset, self).__init__(data_dir, chunk_size)

        self.single_channel = single_channel
        self.neg_lst = [x for x in self.source_names if x != target_name]

        self.target_name = target_name
        check_target_name(self.target_name, self.source_names)

        if not self.musdb_path.joinpath('metadata').exists():
            os.mkdir(self.musdb_path.joinpath('metadata'))

        splits = ['train']
        if external_datasets is not None:
            splits += external_datasets

        # collect paths for datasets and metadata (track names and duration)
        datasets, metadata_caches = [], []
        raw_datasets = []    # un-augmented datasets
        for split in splits:
            raw_datasets.append(self.musdb_path.joinpath(split))
            max_pitch, max_tempo = aug_params
            for p in range(-max_pitch, max_pitch+1):
                for t in range(-max_tempo, max_tempo+1, 10):
                    aug_split = split if p==t==0 else split + f'_p={p}_t={t}'
                    datasets.append(self.musdb_path.joinpath(aug_split))
                    metadata_caches.append(self.musdb_path.joinpath('metadata').joinpath(aug_split + '.pkl'))

        # collect all track names and their duration
        self.metadata = []
        raw_track_lengths = []   # for calculating epoch size
        for i, (dataset, metadata_cache) in enumerate(tqdm(zip(datasets, metadata_caches))):
            try:
                metadata = torch.load(metadata_cache)
            except FileNotFoundError:
                print('creating metadata for', dataset)
                metadata = [] # [(path, length)]
                for track_name in sorted(os.listdir(dataset)):
                    track_path = dataset.joinpath(track_name)
                    track_length = load_wav(track_path.joinpath('vocals.wav')).shape[-1]
                    metadata.append((track_path, track_length))
                torch.save(metadata, metadata_cache)

            self.metadata += metadata
            if dataset in raw_datasets: # for epoch size
                raw_track_lengths += [length for path, length in metadata]

        self.epoch_size = sum(raw_track_lengths) // self.chunk_size if epoch_size is None else epoch_size
        log.info(f'epoch size: {self.epoch_size}')

        # extended dataset with samplepack
        self.extended_dataset = extended_dataset
        self.extended_metadata = {} # {type: [(path, length)]}
        self.pos_lst = pos_lst
        if self.extended_dataset:
            log.info(f'extended dataset: {self.extended_dataset}')
            ext_meta_path = os.path.join(extended_dataset, "metadata.pkl")
            try:
                self.extended_metadata = torch.load(ext_meta_path)
            except FileNotFoundError:
                print('creating metadata for', extended_dataset)
                self.extended_metadata = get_samplepack_metadata(os.path.join(extended_dataset, "train"))
                torch.save(self.extended_metadata, Path(ext_meta_path))
            self.ext_klst = list(self.extended_metadata.keys())
            self.ext_klst = [x for x in self.ext_klst if x not in exclude_lst]
            log.info(f'extended dataset keys: {self.ext_klst}')

    def __getitem__(self, _):
        sources = []
        for source_name in self.source_names:
            track_path, track_length = random.choice(self.metadata)   # random mixing between tracks
            source = load_wav(track_path.joinpath(source_name + '.wav'),
                              track_length=track_length, chunk_size=self.chunk_size) # (2, times)
            sources.append(source)

        mix = sum(sources) # (c, times)

        if self.target_name == 'all':
            # Targets for models that separate all four sources (ex. Demucs).
            # This adds additional 'source' dimension => batch_shape=[batch, source, channel, time]
            target = sources
        else:
            target = sources[self.source_names.index(self.target_name)]

        if self.extended_dataset:
            picked_arr, picked_type = pick_from_sample_pack(self.extended_metadata, self.ext_klst, self.chunk_size) # (c, t)
            # picked_arr = pad_or_truncate(picked_arr, picked_length, self.chunk_size)
            mix += picked_arr
            if picked_type in self.pos_lst:
                target += picked_arr

        mix, target = torch.tensor(mix), torch.tensor(target)
        if self.single_channel:
            mix = torch.mean(mix, dim=0, keepdim=True)
            target = torch.mean(target, dim=0, keepdim=True)
        return mix, target

    def __len__(self):
        return self.epoch_size


class MusdbValidDataset(MusdbDataset):

    def __init__(self, data_dir, chunk_size, target_name, overlap, batch_size, single_channel, extended_dataset):
        super(MusdbValidDataset, self).__init__(data_dir, chunk_size)

        self.target_name = target_name
        check_target_name(self.target_name, self.source_names)

        self.overlap = overlap
        self.batch_size = batch_size
        self.single_channel = single_channel
        self.extended_dataset = extended_dataset

        musdb_valid_path = self.musdb_path.joinpath('valid')
        self.track_paths = [musdb_valid_path.joinpath(track_name)
                            for track_name in os.listdir(musdb_valid_path)]

    def __getitem__(self, index):
        mix = load_wav(self.track_paths[index].joinpath('mixture.wav')) # (2, time)

        if self.target_name == 'all':
            # Targets for models that separate all four sources (ex. Demucs).
            # This adds additional 'source' dimension => batch_shape=[batch, source, channel, time]
            target = [load_wav(self.track_paths[index].joinpath(source_name + '.wav'))
                      for source_name in self.source_names]
        else:
            target = load_wav(self.track_paths[index].joinpath(self.target_name + '.wav'))

        if self.extended_dataset:
            test_set_path, track_name = os.path.split(self.track_paths[index])
            ext_folder = os.path.join(str(self.extended_dataset), "valid", track_name)
            pos_path = os.path.join(ext_folder, 'pos.wav')
            neg_path = os.path.join(ext_folder, 'neg.wav')
            pos = load_wav(pos_path)
            neg = load_wav(neg_path)
            mix += pos + neg
            target += pos

        chunk_output_size = self.chunk_size - 2 * self.overlap
        left_pad = np.zeros([2, self.overlap])
        right_pad = np.zeros([2, self.overlap + chunk_output_size - (mix.shape[-1] % chunk_output_size)])
        mix_padded = np.concatenate([left_pad, mix, right_pad], 1)

        num_chunks = mix_padded.shape[-1] // chunk_output_size
        mix_chunks = np.array([mix_padded[:, i * chunk_output_size: i * chunk_output_size + self.chunk_size]
                      for i in range(num_chunks)])
        mix_chunk_batches = torch.tensor(mix_chunks, dtype=torch.float32).split(self.batch_size)
        target = torch.tensor(target)

        if self.single_channel:
            mix_chunk_batches = [torch.mean(t, dim=1, keepdim=True) for t in mix_chunk_batches]
            target = torch.mean(target, dim=0, keepdim=True)

        return mix_chunk_batches, target

    def __len__(self):
        return len(self.track_paths)