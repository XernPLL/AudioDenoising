import os
import torchaudio as ta
from Demucs import *
import scipy.io.wavfile
import numpy as np



def create_noise_file(audio_clean,noise_path,output):
    noise, _ = ta.load(str(noise_path), normalize=True)
    audio_length = audio_clean.shape[-1]
    noise_length = noise.shape[-1]
    if noise_length > audio_length:
        noise = noise[..., 0: audio_length]
        clean = np.transpose(audio_clean.detach().cpu().numpy(), (1, 0))
        scipy.io.wavfile.write(output+"_original.wav", samplerate, clean)
    elif noise_length < audio_length:
        audio_clean = audio_clean[..., 0: noise_length]
        clean = np.transpose(audio_clean.detach().cpu().numpy(), (1, 0))
        scipy.io.wavfile.write(output+"_original.wav", samplerate, clean)
        #noise = torch.cat([noise, torch.zeros((noise.shape[0], audio_length - noise_length))], dim=-1)

    return audio_clean + noise_power*noise

def denoise(audio_list,noise_list,output_path):
    for i,audio_path in enumerate(audio_list):
        output = output_path + "/" + str(i)
        wavS, _ = ta.load(str(audio_path), normalize=True)
        if createnoise == 1:
            wavS = create_noise_file(wavS,noise_list[i],output)
            noised = np.transpose(wavS.detach().cpu().numpy(), (1, 0))
            scipy.io.wavfile.write(output+"_noised.wav", samplerate, noised)
        else:
            noised = np.transpose(wavS.detach().cpu().numpy(), (1, 0))
            scipy.io.wavfile.write(output+"_noised.wav", samplerate, noised)

        lenofwave = wavS.shape[1]
        lenofseg = samplerate*segemntlength
        steps = int(lenofwave/lenofseg)
        rest = lenofwave - steps * lenofseg
        listofsplits = []
        for i in range(steps):
            listofsplits.append(wavS[0][i*lenofseg:(i+1)*lenofseg])
        restarr = torch.cat([wavS[0][steps*lenofseg:], torch.zeros(lenofseg - rest)], dim=-1)
        listofsplits.append(restarr)
        stacked = torch.stack(listofsplits)
        denoised = model(stacked.view([-1,1,lenofseg]).to(device))
        denoised = denoised.view([1,1,-1])
        denoised = denoised[0][0][:-int(lenofseg - rest)]
        denoised = np.transpose(denoised.view([1,-1]).detach().cpu().numpy(), (1, 0))
        scipy.io.wavfile.write(output+"_denoised.wav", samplerate, denoised)

if __name__ == "__main__":
    audio_list = ["D:/LibriSpeech/cleandataset/38/4.flac",
                  "D:/LibriSpeech/cleandataset/302/0.flac",
                  "D:/LibriSpeech/cleandataset/369/1.flac",
                  "D:/LibriSpeech/cleandataset/696/1.flac",]
    noise_list = ["D:/musan/musan/noise/free-sound/noise-free-sound-0006.wav",
                  "D:/musan/musan/noise/free-sound/noise-free-sound-0147.wav",
                  "D:/musan/musan/noise/free-sound/noise-free-sound-0155.wav",
                  "D:/musan/musan/noise/free-sound/noise-free-sound-0654.wav",]
    createnoise = 1
    model_path = {"HDemucs_mrstft": "./current_models_Sweeprun_mrstft_HDemucs/G_5_004000.pth",
                  "Demucs_mrstft": "./current_models_Sweeprun_mrstft_Demucs/G_5_004000.pth",
                  "HDemucs_l1": "./current_models_Sweeprun_l1_HDemucs/G_5_004000.pth",
                  "Demucs_l1": "./current_models_Sweeprun_l1_Demucs/G_5_004000.pth",
                  "HDemucs_ranstft": "./current_models_Sweeprun_ranstft_HDemucs/G_5_004000.pth",
                  "Demucs_ranstft": "./current_models_Sweeprun_ranstft_Demucs/G_5_004000.pth"}
    samplerate = 16000
    segemntlength = 2
    noise_power = 0.3
    if not os.path.exists(f'./output'):
        os.mkdir(f'./output')


    for i in model_path:
        output_path = "./output/"+i
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if i[0] == "H":
            model = HDemucs().to(device)
        else:
            model = Demucs().to(device)
        model.eval()
        tempmodel = torch.load(model_path[i], map_location=torch.device('cpu'))
        # G.load_state_dict(torch.load(args.G_path, map_location=torch.device('cpu')), strict=True)
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in tempmodel.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        denoise(audio_list,noise_list,output_path)
