from abc import ABCMeta
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.utils.utils import sdr, simplified_msseval


class AbstractModel(LightningModule):
    __metaclass__ = ABCMeta

    def __init__(self, target_name,
                 lr, optimizer,
                  dim_f, dim_t, n_fft, hop_length, overlap,
                 audio_ch,
                  **kwargs):
        super().__init__()
        self.target_name = target_name
        self.lr = lr
        self.optimizer = optimizer
        self.dim_c_in = audio_ch * 2
        self.dim_c_out = audio_ch * 2
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.hop_length = hop_length
        self.audio_ch = audio_ch

        self.chunk_size = hop_length * (self.dim_t - 1)
        self.inference_chunk_size = hop_length * (self.dim_t*2 - 1)
        self.overlap = overlap
        self.window = nn.Parameter(torch.hann_window(window_length=self.n_fft, periodic=True), requires_grad=False)
        self.freq_pad = nn.Parameter(torch.zeros([1, self.dim_c_out, self.n_bins - self.dim_f, 1]), requires_grad=False)
        self.inference_chunk_shape = (self.stft(torch.zeros([1, audio_ch, self.inference_chunk_size]))).shape


    def configure_optimizers(self):
        if self.optimizer == 'rmsprop':
            print("Using RMSprop optimizer")
            return torch.optim.RMSprop(self.parameters(), self.lr)
        elif self.optimizer == 'adamW':
            print("Using AdamW optimizer")
            return torch.optim.AdamW(self.parameters(), self.lr)

    def comp_loss(self, pred_detail, target_wave):
        pred_detail = self.istft(pred_detail)

        comp_loss = F.l1_loss(pred_detail, target_wave)

        self.log("train/comp_loss", comp_loss, sync_dist=True, on_step=False, on_epoch=True, prog_bar=False)

        return comp_loss


    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        mix_wave, target_wave = args[0] # (batch, c, 261120)

        # input 1
        stft_44k = self.stft(mix_wave) # (batch, c*2, 1044, 256)
        # forward
        t_est_stft = self(stft_44k) # (batch, c, 1044, 256)

        loss = self.comp_loss(t_est_stft, target_wave)

        self.log("train/loss", loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss}


    # Validation SDR is calculated on whole tracks and not chunks since
    # short inputs have high possibility of being silent (all-zero signal)
    # which leads to very low sdr values regardless of the model.
    # A natural procedure would be to split a track into chunk batches and
    # load them on multiple gpus, but aggregation was too difficult.
    # So instead we load one whole track on a single device (data_loader batch_size should always be 1)
    # and do all the batch splitting and aggregation on a single device.
    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        mix_chunk_batches, target = args[0]

        # remove data_loader batch dimension
        # [(b, c, time)], (c, all_times)
        mix_chunk_batches, target = [batch[0] for batch in mix_chunk_batches], target[0]

        # process whole track in batches of chunks
        target_hat_chunks = []
        for batch in mix_chunk_batches:
            # input
            stft_44k = self.stft(batch)  # (batch, c*2, 1044, 256)
            pred_detail = self(stft_44k) # (batch, c, 1044, 256), irm
            pred_detail = self.istft(pred_detail)

            target_hat_chunks.append(pred_detail[..., self.overlap:-self.overlap])
        target_hat_chunks = torch.cat(target_hat_chunks) # (b*len(ls),c,t)

        # concat all output chunks (c, all_times)
        target_hat = target_hat_chunks.transpose(0, 1).reshape(self.audio_ch, -1)[..., :target.shape[-1]]

        ests = target_hat.detach().cpu().numpy()  # (c, all_times)
        references = target.cpu().numpy()
        score = sdr(ests, references)

        # (src, t, c)
        SDR = simplified_msseval(np.expand_dims(references.T, axis=0), np.expand_dims(ests.T, axis=0), chunk_size=44100)
        # self.log("val/sdr", score, sync_dist=True, on_step=False, on_epoch=True, logger=True)

        return {'song': score, 'chunk': SDR}

    def validation_epoch_end(self, outputs) -> None:
        avg_uSDR = torch.Tensor([x['song'] for x in outputs]).mean()
        self.log("val/usdr", avg_uSDR, sync_dist=True, on_step=False, on_epoch=True, logger=True)

        chunks = [x['chunk'][0, :] for x in outputs]
        # concat np array
        chunks = np.concatenate(chunks, axis=0)
        median_cSDR = np.nanmedian(chunks.flatten(), axis=0)
        median_cSDR = float(median_cSDR)
        self.log("val/csdr", median_cSDR, sync_dist=True, on_step=False, on_epoch=True, logger=True)

    def stft(self, x):
        '''
        Args:
            x: (batch, c, 261120)
        '''
        dim_b = x.shape[0]
        x = x.reshape([dim_b * self.audio_ch, -1]) # (batch*c, 261120)
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True) # (batch*c, 3073, 256, 2)
        x = x.permute([0, 3, 1, 2]) # (batch*c, 2, 3073, 256)
        x = x.reshape([dim_b, self.audio_ch, 2, self.n_bins, -1]).reshape([dim_b, self.audio_ch * 2, self.n_bins, -1]) # (batch, c*2, 3073, 256)
        return x[:, :, :self.dim_f] # (batch, c*2, 2048, 256)

    def istft(self, x):
        '''
        Args:
            x: (batch, c*2, 2048, 256)
        '''
        dim_b = x.shape[0]
        x = torch.cat([x, self.freq_pad.repeat([x.shape[0], 1, 1, x.shape[-1]])], -2) # (batch, c*2, 3073, 256)
        x = x.reshape([dim_b, self.audio_ch, 2, self.n_bins, -1]).reshape([dim_b * self.audio_ch, 2, self.n_bins, -1]) # (batch*c, 2, 3073, 256)
        x = x.permute([0, 2, 3, 1]) # (batch*c, 3073, 256, 2)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True) # (batch*c, 261120)
        return x.reshape([dim_b, self.audio_ch, -1]) # (batch,c,261120)

    def demix(self, mix, inf_chunk_size, batch_size=5, inf_overf=4):
        '''
        Args:
            mix: (C, L)
        Returns:
            est: (src, C, L)
        '''

        # batch_size = self.config.inference.batch_size
        #  = self.chunk_size
        # self.instruments = ['bass', 'drums', 'other', 'vocals']
        num_instruments = 1

        inf_hop = inf_chunk_size // inf_overf  # hop size
        L = mix.shape[1]
        pad_size = inf_hop - (L - inf_chunk_size) % inf_hop
        mix = torch.cat([torch.zeros(2, inf_chunk_size - inf_hop), torch.Tensor(mix), torch.zeros(2, pad_size + inf_chunk_size - inf_hop)], 1)
        mix = mix.cuda()

        chunks = []
        i = 0
        while i + inf_chunk_size <= mix.shape[1]:
            chunks.append(mix[:, i:i + inf_chunk_size])
            i += inf_hop
        chunks = torch.stack(chunks)

        batches = []
        i = 0
        while i < len(chunks):
            batches.append(chunks[i:i + batch_size])
            i = i + batch_size

        X = torch.zeros(num_instruments, 2, inf_chunk_size - inf_hop) # (src, c, t)
        X = X.cuda()
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                for batch in batches:
                    x = self.stft(batch)
                    x = self(x)
                    x = self.istft(x) # (batch, c, 261120)
                    # insert new axis, the model only predict 1 src so we need to add axis
                    x = x[:,None, ...] # (batch, 1, c, 261120)
                    x = x.repeat([ 1, num_instruments, 1, 1]) # (batch, src, c, 261120)
                    for w in x: # iterate over batch
                        a = X[..., :-(inf_chunk_size - inf_hop)]
                        b = X[..., -(inf_chunk_size - inf_hop):] + w[..., :(inf_chunk_size - inf_hop)]
                        c = w[..., (inf_chunk_size - inf_hop):]
                        X = torch.cat([a, b, c], -1)

        estimated_sources = X[..., inf_chunk_size - inf_hop:-(pad_size + inf_chunk_size - inf_hop)] / inf_overf

        assert L == estimated_sources.shape[-1]

        return estimated_sources

