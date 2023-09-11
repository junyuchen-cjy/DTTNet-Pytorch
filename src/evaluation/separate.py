from os import listdir
from pathlib import Path

import torch
import numpy as np
import onnxruntime as ort


def separate_with_onnx(batch_size, model, onnx_path: Path, mix):
    n_sample = mix.shape[1]

    trim = model.n_fft // 2
    gen_size = model.sampling_size - 2 * trim
    pad = gen_size - n_sample % gen_size
    mix_p = np.concatenate((np.zeros((2, trim)), mix, np.zeros((2, pad)), np.zeros((2, trim))), 1)

    mix_waves = []
    i = 0
    while i < n_sample + pad:
        waves = np.array(mix_p[:, i:i + model.sampling_size], dtype=np.float32)
        mix_waves.append(waves)
        i += gen_size
    mix_waves_batched = torch.tensor(mix_waves, dtype=torch.float32).split(batch_size)

    tar_signals = []

    with torch.no_grad():
        _ort = ort.InferenceSession(str(onnx_path))
        for mix_waves in mix_waves_batched:
            tar_waves = model.istft(torch.tensor(
                _ort.run(None, {'input': model.stft(mix_waves).numpy()})[0]
            ))
            tar_signals.append(tar_waves[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).numpy())
        tar_signal = np.concatenate(tar_signals, axis=-1)[:, :-pad]

    return tar_signal


def separate_with_ckpt(batch_size, model, ckpt_path: Path, mix, device, double_chunk):
    model = model.load_from_checkpoint(ckpt_path).to(device)
    if double_chunk:
        inf_ck = model.inference_chunk_size
    else:
        inf_ck = model.sampling_size
    true_samples = inf_ck - 2 * model.trim

    right_pad = true_samples + model.trim - ((mix.shape[-1]) % true_samples)
    mixture = np.concatenate((np.zeros((2, model.trim), dtype='float32'),
                              mix,
                              np.zeros((2, right_pad), dtype='float32')),
                             1)
    num_chunks = mixture.shape[-1] // true_samples
    mix_waves_batched = [mixture[:, i * true_samples: i * true_samples + inf_ck] for i in
                         range(num_chunks)]
    mix_waves_batched = torch.tensor(mix_waves_batched, dtype=torch.float32).split(batch_size)

    target_wav_hats = []

    with torch.no_grad():
        model.eval()
        for mixture_wav in mix_waves_batched:
            mix_spec = model.stft(mixture_wav.to(device))
            spec_hat = model(mix_spec)
            target_wav_hat = model.istft(spec_hat)
            target_wav_hat = target_wav_hat.cpu().detach().numpy()
            target_wav_hats.append(target_wav_hat)

        target_wav_hat = np.vstack(target_wav_hats)[:, :, model.trim:-model.trim]
        target_wav_hat = np.concatenate(target_wav_hat, axis=-1)[:, :mix.shape[-1]]
    return target_wav_hat




def separate_with_onnx_TDF(batch_size, model, onnx_path: Path, mix):
    n_sample = mix.shape[1]

    overlap = model.n_fft // 2
    gen_size = model.inference_chunk_size - 2 * overlap
    pad = gen_size - n_sample % gen_size
    mix_p = np.concatenate((np.zeros((2, overlap)), mix, np.zeros((2, pad)), np.zeros((2, overlap))), 1)

    mix_waves = []
    i = 0
    while i < n_sample + pad:
        waves = np.array(mix_p[:, i:i + model.inference_chunk_size], dtype=np.float32)
        mix_waves.append(waves)
        i += gen_size
    mix_waves_batched = torch.tensor(mix_waves, dtype=torch.float32).split(batch_size)

    tar_signals = []

    with torch.no_grad():
        _ort = ort.InferenceSession(str(onnx_path), providers=['CUDAExecutionProvider'])
        for mix_waves in mix_waves_batched:
            tar_waves = model.istft(torch.tensor(
                _ort.run(None, {'input': model.stft(mix_waves).numpy()})[0]
            ))
            tar_signals.append(tar_waves[:, :, overlap:-overlap].transpose(0, 1).reshape(2, -1).numpy())
        tar_signal = np.concatenate(tar_signals, axis=-1)[:, :-pad]

    return tar_signal



def separate_with_ckpt_TDF(batch_size, model, ckpt_path: Path, mix, device, double_chunk):
    '''
    Args:
        batch_size: the inference batch size
        model: the model to be used
        ckpt_path: the path to the checkpoint
        mix: (c, t)
        device: the device to be used
        double_chunk: whether to use double chunk size
    Returns:
        target_wav_hat: (c, t)
    '''
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    # model = model.load_from_checkpoint(ckpt_path).to(device)
    if double_chunk:
        inf_ck = model.inference_chunk_size
    else:
        inf_ck = model.chunk_size


    target_wav_hat = no_overlap_inference(model, mix, device, batch_size, inf_ck)


    return target_wav_hat

def no_overlap_inference(model, mix, device, batch_size, inf_ck):
    true_samples = inf_ck - 2 * model.overlap

    right_pad = true_samples + model.overlap - ((mix.shape[-1]) % true_samples)
    mixture = np.concatenate((np.zeros((model.audio_ch, model.overlap), dtype='float32'),
                              mix,
                              np.zeros((model.audio_ch, right_pad), dtype='float32')),
                             1)
    num_chunks = mixture.shape[-1] // true_samples
    mix_waves_batched = [mixture[:, i * true_samples: i * true_samples + inf_ck] for i in
                         range(num_chunks)]
    mix_waves_batched = torch.tensor(mix_waves_batched, dtype=torch.float32).split(batch_size)

    target_wav_hats = []

    with torch.no_grad():
        model.eval()
        for mixture_wav in mix_waves_batched:
            mix_spec = model.stft(mixture_wav.to(device))
            spec_hat = model(mix_spec)
            target_wav_hat = model.istft(spec_hat)
            target_wav_hat = target_wav_hat.cpu().detach().numpy()
            target_wav_hats.append(target_wav_hat) # (b, c, t)

        target_wav_hat = np.vstack(target_wav_hats)[:, :, model.overlap:-model.overlap] # (sum(b), c, t)
        target_wav_hat = np.concatenate(target_wav_hat, axis=-1)[:, :mix.shape[-1]]
    return target_wav_hat

