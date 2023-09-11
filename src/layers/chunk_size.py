#%%
import numpy as np
import torch



def wave_to_batches(mix, inf_ck, overlap, batch_size):
    '''
    Args:
        mix: (2, N) numpy array
        inf_ck: int, the chunk size as the model input (contains 2*overlap)
                inf_ck = overlap + true_samples + overlap
        overlap: int, the discarded samples at each side
    Returns:
        a tuples of batches, each batch is a (batch, 2, inf_ck) torch tensor
    '''
    true_samples = inf_ck - 2 * overlap
    channels = mix.shape[0]

    right_pad = true_samples + overlap - ((mix.shape[-1]) % true_samples)
    mixture = np.concatenate((np.zeros((channels, overlap), dtype='float32'),
                              mix,
                              np.zeros((channels, right_pad), dtype='float32')),
                             1)

    num_chunks = mixture.shape[-1] // true_samples
    mix_waves_batched = np.array([mixture[:, i * true_samples: i * true_samples + inf_ck] for i in
                         range(num_chunks)]) # (x,2,inf_ck)
    return torch.tensor(mix_waves_batched, dtype=torch.float32).split(batch_size)

def batches_to_wave(target_hat_chunks, overlap, org_len):
    '''
    Args:
        target_hat_chunks: a list of (batch, 2, inf_ck) torch tensors
        overlap: int, the discarded samples at each side
        org_len: int, the original length of the mixture
    Returns:
        (2, N) numpy array
    '''
    target_hat_chunks = [c[..., overlap:-overlap] for c in target_hat_chunks]
    target_hat_chunks = torch.cat(target_hat_chunks)

    # concat all output chunks
    return target_hat_chunks.transpose(0, 1).reshape(2, -1)[..., :org_len].detach().cpu().numpy()

if __name__ == '__main__':
    mix = np.random.rand(2, 14318640)
    inf_ck = 261120
    overlap = 3072
    batch_size = 8
    out = wave_to_batches(mix, inf_ck, overlap, batch_size)
    in_wav = batches_to_wave(out, overlap, mix.shape[-1])
    print(in_wav.shape)