import soundfile as sf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil

logs = {}


def calculate_duration(root_path, file_lst):
    secs = 0
    for file in file_lst:
        f = sf.SoundFile(root_path + "\\" + file)
        secs += len(f) / f.samplerate
    return secs


def get_samplepack_info(root_path):
    df = pd.DataFrame(columns=["folder", "duration"])

    folders = os.listdir(root_path)
    for f in folders:
        # list all the files in the folder
        if not os.path.isdir(root_path + "\\" + f):
            continue
        if f == ".idea":
            continue
        file_lst = os.listdir(root_path + "\\" + f)
        secs = calculate_duration(root_path + "\\" + f, file_lst)
        df = df.append({"folder": f, "duration": secs}, ignore_index=True)
        print(f, secs)
    return df


def secs_to_time(secs):
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return f"{d:.0f}d {h:.0f}h {m:.0f}m {s:.0f}s"


def get_partition_dur(path):
    folders = os.listdir(path)
    dur = []
    for f in folders:
        if f == "metadata":
            continue
        mixture_path = path + "\\" + f + "\\mixture.wav"
        mix_file = sf.SoundFile(mixture_path)
        dur.append(len(mix_file) / mix_file.samplerate)
    return dur


def get_parition_info(path):
    test_dur = get_partition_dur(path)
    # plot bar
    plt.figure(figsize=(20,10))
    plt.bar(range(1, len(test_dur)+1), test_dur)
    print(path,secs_to_time(sum(test_dur)))
    return sum(test_dur)


def get_samplepack_metadata(root_path):
    meta_data = {}
    folders = os.listdir(root_path)
    for f in folders:
        # list all the files in the folder
        if not os.path.isdir(root_path + "\\" + f):
            continue
        if f == ".idea":
            continue
        file_lst = os.listdir(root_path + "\\" + f)
        if f in meta_data:
            meta_data[f] += file_lst
        else:
            meta_data[f] = file_lst
    return meta_data


def partiton_inst(inst_lst):
    # shuffle the list
    np.random.shuffle(inst_lst)
    inst_len = len(inst_lst)
    train_end = int(np.ceil(5/9 * inst_len))
    val_end = int(np.ceil(6/9 * inst_len))

    train = inst_lst[:train_end]
    val = inst_lst[train_end:val_end]
    test = inst_lst[val_end:]
    assert len(train) + len(val) + len(test) == len(inst_lst)
    return [("train", train), ("valid", val), ("test", test)]


def create_subfolder(root_path, sub_name):
    new_dir = root_path + "\\" + sub_name
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    return new_dir


def get_rate(ext_partition, mus_partition):
    df = get_samplepack_info(ext_partition)
    g_dur = df.loc[0,"duration"] + df.loc[1,"duration"] # use guitar to estimate
    est_dur = g_dur * 5
    return est_dur / sum(get_partition_dur(mus_partition))


def pick_from_sample_pack(ext_partiton_root, inst_lst):
    # randomly pick a sample type
    picked_type = np.random.choice(inst_lst)

    inst_folder = ext_partiton_root + "\\" + picked_type
    inst_tracks = os.listdir(inst_folder)

    # randomly pick one from the folder
    picked_name = np.random.choice(inst_tracks)
    picked_path = inst_folder + "\\" + picked_name
    picked_file, sr = sf.read(picked_path)
    if len(picked_file.shape) == 1:
        # repeat the channel
        picked_file = np.repeat(picked_file[:, np.newaxis], 2, axis=1)
    # normalize clip
    picked_file = picked_file / np.max(np.abs(picked_file))
    picked_file = picked_file * 0.5

    if picked_type == "upfilters":
        if np.random.rand() > 0.5:
            picked_file = np.flip(picked_file, axis=0)

    return picked_file, picked_type, picked_name  # (t, c)


def cur_total_segments(previous_segments):
    total = 0
    for (start, end) in previous_segments:
        total += (end - start + 1)
    return total


def is_overlapping(segment_time, previous_segments):
    segment_start, segment_end = segment_time
    overlap = False

    for previous_start, previous_end in previous_segments: # @KEEP
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True
            break

    return overlap


def get_random_time_segment(segment_length, bg_length):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """
    # pick from [low, high)
    segment_start = np.random.randint(low=0, high=bg_length-segment_length)   # Make sure segment doesn't run past the 10sec background
    segment_end = segment_start + segment_length - 1

    return (segment_start, segment_end)


def insert_audio_clip(bg_lst, audio_clip, picked_type, inst_lst, previous_segments, max_dur):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the
    audio segment does not overlap with existing segments.

    Arguments:
    background -- a 10 second background audio recording.
    audio_clip -- the audio clip to be inserted/overlaid.
    previous_segments -- times where audio segments have already been placed

    Returns:
    new_background -- the updated background audio
    """

    # Get the duration of the audio clip in ms
    if cur_total_segments(previous_segments) > max_dur:
        return False, bg_lst

    segment_length = audio_clip.shape[0]
    bg_length = bg_lst.shape[1]

    segment_time = get_random_time_segment(segment_length, bg_length)

    retry = 5 # @KEEP
    while is_overlapping(segment_time, previous_segments) and retry >= 0:
        segment_time = get_random_time_segment(segment_length, bg_length)
        retry = retry - 1
        #print(segment_time)
    # if last try is not overlaping, insert it to the background
    if not is_overlapping(segment_time, previous_segments):
        # Step 3: Append the new segment_time to the list of previous_segments (≈ 1 line)
        previous_segments.append(segment_time)
        # Step 4: Superpose audio segment and background
        start, end = segment_time

        idx = inst_lst.index(picked_type)
        bg_lst[idx, start:end+1,:] = audio_clip
        return True, bg_lst
    else:
        return False, bg_lst


def insert_samples_to_a_track(musc_partition_root, ext_partiton_root, track, target_folder, inst_lst, par):
    mix_path = musc_partition_root + "\\" + track + "\\mixture.wav"
    mix_file = sf.SoundFile(mix_path)
    cur_length = len(mix_file)
    val_rate = 0.45 # 2 * time(samples)/time(musdb_dataset)
    max_dur = cur_length * val_rate

    previous_segments = []
    # pos_bg = np.zeros((cur_length, 2))
    # neg_bg = np.zeros((cur_length, 2))


    bg_lst = np.zeros((len(inst_lst), cur_length, 2))
    picked_array = None
    while True:
        if picked_array is None:
            picked_array, picked_type, picked_name = pick_from_sample_pack(ext_partiton_root, inst_lst)
        flag, bg_lst = insert_audio_clip(bg_lst,
                                          audio_clip=picked_array,
                                          picked_type=picked_type,
                                          inst_lst=inst_lst,
                                          previous_segments=previous_segments,
                                          max_dur=max_dur)
        if flag:
            picked_array = None
            if track not in logs:
                logs[track] = []
            logs[track].append((picked_type, picked_name))
        if not flag: # if not inserted, break
            break

    save_path = target_folder + "\\" + track
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if par == "valid":
        vc_idx = inst_lst.index("vocalchops_processed")
        pos_bg = bg_lst[vc_idx,:,:]
        others = []
        for i in range(len(inst_lst)):
            if i != vc_idx:
                others.append(bg_lst[i,:,:])
        neg_bg = np.sum(others, axis=0)
        sf.write(save_path + "\\" + "pos.wav", pos_bg, samplerate=44100)
        sf.write(save_path + "\\" + "neg.wav", neg_bg, samplerate=44100)
    elif par == "test":
        for i in range(len(inst_lst)):
            sf.write(save_path + "\\" + inst_lst[i] + ".wav", bg_lst[i,:,:], samplerate=44100)


if __name__ == "__main__":

    musdb_root = "G:\musdb18hq"
    sample_pack_root = "G:\samplepacks"
    tmp_root = r"G:\vcplus_ext"
    if not os.path.exists(tmp_root):
        os.makedirs(tmp_root)
    target_root = "G:\\vcplus_musdb_extension"
    if not os.path.exists(target_root):
        os.makedirs(target_root)
    inst_lst = ["guitar_processed", "siren_processed", "horn_processed", "upfilters_processed","vocalchops_processed"]


    np.random.seed(42)
    s_meta_data = get_samplepack_metadata(sample_pack_root)

    for k in s_meta_data.keys():
        if k not in inst_lst:
            continue
        sample_lst = s_meta_data[k]
        par_lst = partiton_inst(sample_lst)

        for p_name, p_lst in par_lst:
            new_dir = create_subfolder(tmp_root, p_name)
            new_dir = create_subfolder(new_dir, k)
            for inst in p_lst:
                shutil.copyfile(sample_pack_root + "\\" + k + "\\" + inst, new_dir + "\\" + inst)


    np.random.seed(42)
    partition_lst = ["valid", "test"]

    for par in partition_lst:
        musc_partition_root = musdb_root + "\\" + par
        ext_partiton_root = tmp_root + "\\" + par
        target_folder = target_root + "\\" + par
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        track_ls = os.listdir(musc_partition_root)

        for track in track_ls:
            insert_samples_to_a_track(musc_partition_root, ext_partiton_root, track,
                                      target_folder=target_folder, inst_lst=inst_lst, par=par)

    # copy train to target directly
    tmp_train = os.path.join(tmp_root, "train")
    for i in inst_lst:
        shutil.copytree(os.path.join(tmp_train, i), os.path.join(target_root, "train", i))


