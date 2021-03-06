print("started imports")

import argparse
import wandb
import numpy as np

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from Demucs import *
from utils.training.Dataset import Wavset
from utils.training.losses import compute_generator_losses2
torch.autograd.set_detect_anomaly(True)

print("finished imports")


class DataParallel(torch.nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__(module, device_ids, output_device, dim)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)




def train_one_epoch(G: 'generator model',
                    opt_G: "generator opt",
                    scheduler_G: "scheduler G opt",
                    args: 'Args Namespace',
                    dataloader: torch.utils.data.DataLoader,
                    testloader: torch.utils.data.DataLoader,
                    device: 'torch device',
                    epoch: int):

    for iteration, data in enumerate(dataloader):
        start_time = time.time()

        Y, noised = data
        Y[Y != Y] = 0
        noised[noised != noised] = 0


        Y = Y.to(device)
        noised = noised.to(device)

        # generator training
        opt_G.zero_grad()

        unnoised = G(noised)


        lossG, L_rec = compute_generator_losses2(G, Y, unnoised, args)
        lossG.backward()
        opt_G.step()
        if args.scheduler:
            scheduler_G.step()


        batch_time = time.time() - start_time

        if iteration % args.show_step == 0:
            Y = np.transpose(Y[0].detach().cpu().numpy(),(1,0))
            noised = np.transpose(noised[0].detach().cpu().numpy(),(1,0))
            unnoised = np.transpose(unnoised[0].detach().cpu().numpy(), (1, 0))

            if args.use_wandb:
                wandb.log({"Y": wandb.Audio(Y, caption=f"{epoch:03}" + '_' + f"{iteration:06}", sample_rate=16000)})
                wandb.log({"noised": wandb.Audio(noised, caption=f"{epoch:03}" + '_' + f"{iteration:06}", sample_rate=16000)})
                wandb.log(
                    {"unnoised": wandb.Audio(unnoised, caption=f"{epoch:03}" + '_' + f"{iteration:06}", sample_rate=16000)})
            else:
                pass
            print("Before the sleep statement")
            #time.sleep(30)
            print("After the sleep statement")

        if iteration % 10 == 0:
            print(f'epoch: {epoch}    {iteration} / {len(dataloader)}')
            print(f'  L_rec: {L_rec.item()}')
            if args.scheduler:
                print(f'scheduler_G lr: {scheduler_G.get_last_lr()}')

        if args.use_wandb:
            wandb.log({
                       "loss_rec_train": L_rec.item()})

        if iteration % 4000 == 0:
            torch.save(G.state_dict(), f'./saved_models_{args.run_name}_{args.loss}_{args.model}/G_latest.pth')

            torch.save(G.state_dict(),
                       f'./current_models_{args.run_name}_{args.loss}_{args.model}/G_' + str(epoch) + '_' + f"{iteration:06}" + '.pth')

        if iteration % 500 == 0:
            suma = 0
            meanof = 0
            for iteration2, data in enumerate(testloader):
                G.eval()

                Y, noised = data
                Y[Y != Y] = 0
                noised[noised != noised] = 0

                Y = Y.to(device)
                noised = noised.to(device)

                # generator training
                opt_G.zero_grad()

                unnoised = G(noised)

                lossG, L_rec = compute_generator_losses2(G, Y, unnoised, args)
                suma = suma + 1
                meanof = L_rec.item() + meanof

                if iteration2 == 50:
                    break


            if args.use_wandb:
                wandb.log({
                    "loss_rec_test": meanof/suma})

            G.train()


def train(args, device):
    # training params
    batch_size = args.batch_size
    max_epoch = args.max_epoch

    # initializing main models
    if args.model == "HDemucs":
        G = HDemucs().to(device)
    elif args.model == "Demucs":
        G = Demucs().to(device)

    G.train()


    opt_G = optim.Adam(G.parameters(), lr=args.lr_G, betas=(0.9, 0.999))


    if args.scheduler:
        scheduler_G = scheduler.StepLR(opt_G, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    else:
        scheduler_G = None

    if args.pretrained:
        try:
            tempmodel = torch.load(args.G_path, map_location=torch.device('cpu'))
            # G.load_state_dict(torch.load(args.G_path, map_location=torch.device('cpu')), strict=True)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in tempmodel.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            G.load_state_dict(new_state_dict)

        except FileNotFoundError as e:
            print("Not found pretrained weights. Continue without any pretrained weights.")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        G = DataParallel(G)

    dataset = Wavset(args.dataset_path)
    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6, drop_last=True)



    for epoch in range(0, max_epoch):
        train_one_epoch(G,
                        opt_G,
                        scheduler_G,
                        args,
                        dataloader,
                        testloader,
                        device,
                        epoch)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print('cuda is not available. using cpu. check if it\'s ok')

    print("Starting traing")
    train(args, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset params
    parser.add_argument('--dataset_path', default='D:/LibriSpeech/cleandataset',
                        help='Path to the dataset.')

    parser.add_argument('--G_path', default='./current_models_TESTrun/G_0_015000.pth',
                        help='Path to pretrained weights for G. Only used if pretrained=True')

    parser.add_argument('--pretrained', default=False, type=bool,
                        help='If using the pretrained weights for training or not')
    parser.add_argument('--scheduler', default=True, type=bool,
                        help='If True decreasing LR is used for learning of generator and discriminator')
    parser.add_argument('--scheduler_step', default=7000, type=int)
    parser.add_argument('--scheduler_gamma', default=0.5, type=float,
                        help='It is value, which shows how many times to decrease LR')
    # info about this run
    parser.add_argument('--use_wandb', default=True, type=bool, help='Use wandb to track your experiments or not')
    parser.add_argument('--run_name', default="Sweeprun", type=str,
                        help='Name of this run. Used to create folders where to save the weights.')
    parser.add_argument('--wandb_project', default='uncategorized', type=str)
    parser.add_argument('--wandb_entity', default='xernpl', type=str)
    # training params you probably don't want to change
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr_G', default=4e-4, type=float)
    parser.add_argument('--max_epoch', default=6, type=int)
    parser.add_argument('--show_step', default=100, type=int)

    parser.add_argument('--loss', default='ranstft',  nargs='?', choices=['mrstft', 'l1', "ranstft"], help='Loss')
    parser.add_argument('--model', default='HDemucs', nargs='?', choices=['HDemucs', 'Demucs'],
                        help='Model')

    args = parser.parse_args()

    if args.use_wandb == True:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)

        config = wandb.config
        config.dataset_path = args.dataset_path
        config.scheduler = args.scheduler
        config.scheduler_step = args.scheduler_step
        config.scheduler_gamma = args.scheduler_gamma
        config.pretrained = args.pretrained
        config.run_name = args.run_name
        config.G_path = args.G_path
        config.batch_size = args.batch_size
        config.lr_G = args.lr_G
        config.loss = args.loss
        config.model = args.model


    # ?????????????? ??????????, ?????????? ???????? ???????? ?????????????????? ?????????????????? ???????? ??????????????, ?? ?????????? ???????? ?? ???????????? ??????????
    if not os.path.exists(f'./saved_models_{args.run_name}_{args.loss}_{args.model}'):
        os.mkdir(f'./saved_models_{args.run_name}_{args.loss}_{args.model}')
        os.mkdir(f'./current_models_{args.run_name}_{args.loss}_{args.model}')

    main(args)
