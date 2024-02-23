import os
import time
import sys
sys.path.append('..')
import GMI.utils
import torch
from GMI.utils import save_tensor_images, init_dataloader, load_json
from torch.autograd import grad
from GMI.discri import DGWGAN, DGWGAN32, DiscriminatorCIFAR10
from GMI.generator import Generator, GeneratorMNIST, GeneratorCIFAR10
from argparse import ArgumentParser
import datetime
import logging


def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)


def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)


def gradient_penalty(x, y):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = DG(z)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp


def get_logger(logpath):
    logger = logging.getLogger('main')
    logger.setLevel(level=logging.DEBUG)
    formatter1 = logging.Formatter('%(filename)s: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter1)

    formatter2 = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s')
    # os.makedirs(logpath, exist_ok=True)
    file_handler = logging.FileHandler(logpath)
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(formatter2)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s', level=logging.DEBUG,
    #                     filename=logpath, filemode='a')

    return logger

if __name__ == "__main__":
    parser = ArgumentParser(description='Step1: train GAN')
    parser.add_argument('--dataset', default='celeba', help='celeba | mnist | cifar10')
    parser.add_argument('--prior_dataset', default='fmnist', help='fmnist | mnist')

    args = parser.parse_args()

    # log file name define
    timestr = datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S")
    args.timestr = timestr
    log_fname = f'{timestr}-GMI-PDataset({args.prior_dataset})-TDataset({args.dataset}).log'
    logpath = f'./result/{args.timestr}'
    os.makedirs(logpath, exist_ok=True)
    logpath = os.path.join(logpath, log_fname)
    flogger = get_logger(logpath)
    flogger.info(f'timestr is {timestr}')

    ############################# mkdirs ##############################
    save_model_dir = f"result/{args.timestr}/models_gan"
    os.makedirs(save_model_dir, exist_ok=True)
    save_img_dir = f"result/{args.timestr}/imgs_gan"
    os.makedirs(save_img_dir, exist_ok=True)
    ############################# mkdirs ##############################

    file = "./config/" + args.prior_dataset + '_' + args.dataset +".json"
    loaded_args = load_json(json_file=file)
    file_path = loaded_args['dataset']['train_file_path']
    model_name = loaded_args['dataset']['model_name']
    lr = loaded_args[model_name]['lr']
    batch_size = loaded_args[model_name]['batch_size']
    z_dim = loaded_args[model_name]['z_dim']
    epochs = loaded_args[model_name]['epochs']
    n_critic = loaded_args[model_name]['n_critic']

    flogger.info("---------------------Training [%s]------------------------------" % model_name)
    GMI.utils.print_params(loaded_args["dataset"], loaded_args[model_name])

    dataset, dataloader = init_dataloader(loaded_args, file_path, batch_size, mode="gan")

    if args.dataset == 'celeba':
        G = Generator(z_dim)
        DG = DGWGAN(3)

    elif args.dataset == 'mnist':
        G = GeneratorMNIST(z_dim)
        DG = DGWGAN32()

    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()

    # 0.004
    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    step = 0
    for epoch in range(epochs):
        start = time.time()
        for i, imgs in enumerate(dataloader):
            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)
            flogger.info(f'| Epoch {epoch + 1}/{epochs}: {bs*(i+1)}/{len(dataloader.dataset)}')

            freeze(G)
            unfreeze(DG)

            z = torch.randn(bs, z_dim).cuda()
            f_imgs = G(z)

            r_logit = DG(imgs)
            f_logit = DG(f_imgs)

            wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
            gp = gradient_penalty(imgs.data, f_imgs.data)
            dg_loss = - wd + gp * 10.0

            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            # train G
            if step % n_critic == 0:
                freeze(DG)
                unfreeze(G)
                z = torch.randn(bs, z_dim).cuda()
                f_imgs = G(z)
                logit_dg = DG(f_imgs)
                # calculate g_loss
                g_loss = - logit_dg.mean()

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        end = time.time()
        interval = end - start
        flogger.info("Epoch:%d \tTime:%.2f\tD_loss:%.2f\tG_loss:%.2f" % (epoch, interval, dg_loss, g_loss))
        if (epoch + 1) % 10 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            save_tensor_images(fake_image.detach(), os.path.join(save_img_dir, "result_image_{}.png".format(epoch + 1)),
                               nrow=8)

        if epoch + 1 >= 100:
            flogger.info('saving weights file')
            torch.save({'state_dict': G.state_dict()},
                       os.path.join(save_model_dir, f"{args.dataset}_G.tar"))
            torch.save({'state_dict': DG.state_dict()},
                       os.path.join(save_model_dir, f"{args.dataset}_D.tar"))
