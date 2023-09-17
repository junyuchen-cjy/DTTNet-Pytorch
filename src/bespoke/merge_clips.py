import os
import numpy as np
import soundfile as sf

def merge_clips(raw_path, target):
    if not os.path.isdir(target):
        os.mkdir(target)

    sample_lst = os.listdir(raw_path)
    data = []
    for i in sample_lst:
        if i.endswith(".wav"):
            f = sf.SoundFile(raw_path + "\\" + i)
            secs = len(f) / f.samplerate
            data.append((i, secs))
    # sort by duration
    data.sort(key=lambda x: x[1])

    cur_dur = 0
    idx = 0
    cur_data = []
    for i, d in data:
        cur_dur += d
        cur_audio = sf.read(raw_path + "\\" + i)[0]
        # normalize
        cur_audio = cur_audio / np.max(np.abs(cur_audio))
        cur_audio = cur_audio * 0.5
        if len(cur_audio.shape) == 1:
            cur_audio = np.expand_dims(cur_audio, axis=1)
            cur_audio = np.concatenate((cur_audio, cur_audio), axis=1)
        cur_data.append(cur_audio)
        if cur_dur > 4:
            if cur_dur > 8:
                # split
                cur_data = np.concatenate(cur_data, axis=0)
                split_idx = cur_data.shape[0]//2
                sf.write(target + "\\" + str(idx) + ".wav", cur_data[0:split_idx,:], 44100)
                sf.write(target + "\\" + str(idx + 1) + ".wav", cur_data[split_idx:,:], 44100)

                idx += 2
                cur_dur = 0
                cur_data = []
            else:
                cur_data = np.concatenate(cur_data, axis=0)
                sf.write(target + "\\" + str(idx) + ".wav", cur_data, 44100)

                idx += 1
                cur_dur = 0
                cur_data = []

if __name__ == "__main__":
    # guitar = "G:\samplepacks\guitar_merged"
    # target = "G:\samplepacks\guitar_processed"
    # merge_clips(guitar, target)
    samplepacks_root = "G:\samplepacks"

    inst_lst = ["guitar", "siren", "horn", "upfilters", "vocalchops"]
    for i in inst_lst:
        raw_path = os.path.join(samplepacks_root, i)
        target = os.path.join(samplepacks_root, i + "_processed")
        merge_clips(raw_path, target)