import torch
import auraloss
l1_loss = torch.nn.L1Loss()
l2_loss = torch.nn.MSELoss()
mrstft = auraloss.freq.MultiResolutionSTFTLoss()
ranstft= auraloss.freq.RandomResolutionSTFTLoss(max_fft_size= 16384)
sdstft = auraloss.freq.SumAndDifferenceSTFTLoss()


def hinge_loss(X, positive=True):
    if positive:
        return torch.relu(1-X)
    else:
        return torch.relu(X+1)


def compute_generator_losses2(G, Y, unnoised, args):

    if args.loss == "mrstft":
        L_rec = mrstft(unnoised,Y)
    elif args.loss == "l1":
        L_rec = l1_loss(unnoised, Y)
    elif args.loss == "ranstft":
        L_rec = ranstft(unnoised, Y)
    elif args.loss == "sdstft":
        L_rec = sdstft(unnoised, Y)


    lossG =  L_rec

    return lossG,  L_rec


