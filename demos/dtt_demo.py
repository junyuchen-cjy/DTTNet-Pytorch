import torch
from src.dp_tdf.dp_tdf_net import DPTDFNet
from src.layers.chunk_size import wave_to_batches

import dotenv
from omegaconf import OmegaConf
dotenv.load_dotenv(override=True)

def run_once(src_name):
    # create dummy data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create dummy config
    yaml = rf"G:\sdx2023fork\configs\model\{src_name}.yaml"
    # yaml = r"G:\sdx2023fork\configs\model\TDF_mono.yaml"
    # load yaml into dict
    confg = OmegaConf.load(yaml)

    stft = torch.rand(1, 2 * confg.audio_ch, confg.dim_f, 256).to(device)
    # create model
    model = DPTDFNet(**confg).to(device)

    pred_detail = model(stft)
    print(pred_detail.shape)


def run_train(src_name):
    yaml = rf"G:\sdx2023fork\configs\model\{src_name}.yaml"
    h = OmegaConf.load(yaml)
    device = torch.device('cuda')
    b = 2
    chunk_size = (h.dim_t - 1) * h.hop_length
    print("before ", chunk_size)
    input1 = torch.rand(b, h.audio_ch, chunk_size).to(device)
    input2 = torch.rand(b, h.audio_ch, chunk_size).to(device)

    tdf = DPTDFNet(**h).to(device)

    output = tdf.training_step((input1, input2))

    print(output["loss"])


def run_val(src_name):
    yaml = rf"G:\sdx2023fork\configs\model\{src_name}.yaml"
    h = OmegaConf.load(yaml)
    device = torch.device('cuda')
    b = 2
    chunk_size = (h.dim_t - 1) * h.hop_length
    print("before ", chunk_size)
    input1 = torch.rand(h.audio_ch, 3*chunk_size)
    input1 = wave_to_batches(input1, chunk_size, h.overlap, b)
    input1 = [i.unsqueeze(0).to(device) for i in input1]
    input2 = torch.rand(1, h.audio_ch, 3*chunk_size).to(device)

    tdf = DPTDFNet(**h).to(device)


    output = tdf.validation_step((input1, input2))

    for k, v in output.items():
        print(k, v)


if __name__ == "__main__":
    src_name = "vocals"
    # src_name = "bass"
    # src_name = "drums"
    # src_name = "other"
    run_once(src_name)
    run_train(src_name)
    with torch.no_grad():
        run_val(src_name)
