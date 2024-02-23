import os
import time
import datetime
import logging
import utils
import torch
from utils import *
from torch.autograd import grad
import torch.nn.functional as F
from discri import DGWGAN, Discriminator, MinibatchDiscriminator
from discri import *
from generator import Generator
from generator import *
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import sys

sys.path.append('../BiDO')
import model


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


def log_sum_exp(x, axis=1):
    m = torch.max(x, dim=1)[0]

    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis))


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
    parser = ArgumentParser(description='Step2: targeted recovery')
    parser.add_argument('--dataset', default='celeba', help='celeba | mnist')
    parser.add_argument('--prior_dataset', default='fmnist', help='fmnist | mnist')
    parser.add_argument('--defense', default='HSIC', help='HSIC | vib')
    parser.add_argument('--root_path', default="./improvedGAN")
    parser.add_argument('--model_path', default='../GMI/target_model')
    parser.add_argument('--targetor_name', default='mnist_MCNN_98.41.tar')
    parser.add_argument('--fl_timestr', default='123456')
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--beta', default=0, type=float)
    parser.add_argument('--acc', default=0, type=float)
    args = parser.parse_args()

    # log file name define
    import datetime
    timestr = datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S")
    args.timestr = timestr
    log_fname = f'{timestr}-DMI-1stStage-PDataset({args.prior_dataset})-TDataset({args.dataset}).log'
    logpath = f'{args.root_path}/{args.fl_timestr}/{args.timestr}-{args.targetor_name}'
    os.makedirs(logpath, exist_ok=True)
    logpath = os.path.join(logpath, log_fname)
    flogger = get_logger(logpath)
    flogger.info(f'fl timestr is {args.fl_timestr}')
    flogger.info(f'timestr is {timestr}')
    flogger.info(f'targetor is {args.targetor_name}')

    if args.gpu:
        torch.cuda.set_device(args.gpu - 1)
    device = 'cuda' if args.gpu else 'cpu'
    args.device = device
    args.device_idx = list((args.gpu-1, ))

    file = "./config/" + args.prior_dataset + '_' + args.dataset + ".json"
    loaded_args = load_json(json_file=file)

    ############################# mkdirs ##############################
    save_model_dir = f'{args.root_path}/{args.dataset}/{args.fl_timestr}/{args.timestr}-PD{args.prior_dataset}-TD{args.dataset}-{args.targetor_name}/models_gan'
    os.makedirs(save_model_dir, exist_ok=True)
    save_img_dir = f'{args.root_path}/{args.dataset}/{args.fl_timestr}/{args.timestr}-PD{args.prior_dataset}-TD{args.dataset}-{args.targetor_name}/imgs_gan'
    os.makedirs(save_img_dir, exist_ok=True)

    ############################# mkdirs ##############################

    file_path = loaded_args['dataset']['gan_file_path']
    stage = loaded_args['dataset']['stage']
    lr = loaded_args[stage]['lr']
    batch_size = loaded_args[stage]['batch_size']
    z_dim = loaded_args[stage]['z_dim']
    epochs = loaded_args[stage]['epochs']
    n_critic = loaded_args[stage]['n_critic']

    utils.print_params(loaded_args["dataset"], loaded_args[stage])
    model_name = loaded_args["dataset"]["model_name"]

    flogger.info(f'epochs is {epochs}')
    flogger.info(f'model_name is {model_name}')

    if args.dataset == 'celeba':
        hp_ac_list = [
            # celeba - hsic
            # (0.01, 0.05, 84.81),
            # (0.01, 0.2, 81.65),
            # (0.01, 0.5, 74.97),

            # (0.05, 0.25, 83.84),
            # (0.05, 0.5, 80.35),
            # (0.05, 1.0, 70.31),
            # (0.05, 2.5, 53.49),
        ]
        for (a1, a2, ac) in hp_ac_list:
            print("a1:", a1, "a2:", a2, "test_acc:", ac)

            T = model.VGG16(1000, True)
            T = torch.nn.DataParallel(T).cuda()
            path_T = os.path.join(args.model_path, f"{args.dataset}/",args.defense,
                                  "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, ac))

            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'], strict=False)

            print("---------------------Training [%s]------------------------------" % stage)

            dataset, dataloader = utils.init_dataloader(loaded_args, file_path, batch_size, mode="gan")

            G = Generator(z_dim)
            DG = MinibatchDiscriminator()
            G = torch.nn.DataParallel(G).cuda()
            DG = torch.nn.DataParallel(DG).cuda()

            dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
            g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

            entropy = HLoss()

            step = 0

            for epoch in range(0, epochs):
                start = time.time()
                _, unlabel_loader1 = init_dataloader(loaded_args, file_path, batch_size, mode="gan", iterator=True)
                _, unlabel_loader2 = init_dataloader(loaded_args, file_path, batch_size, mode="gan", iterator=True)

                for i, imgs in enumerate(dataloader):
                    current_iter = epoch * len(dataloader) + i + 1

                    step += 1
                    imgs = imgs.cuda()
                    bs = imgs.size(0)
                    x_unlabel = unlabel_loader1.next()
                    x_unlabel2 = unlabel_loader2.next()

                    freeze(G)
                    unfreeze(DG)

                    z = torch.randn(bs, z_dim).cuda()
                    f_imgs = G(z)

                    y_prob = T(imgs)[-1]

                    targetprobs = nn.functional.softmax(y_prob, dim=1)
                    # print(entropy(targetprobs))

                    y = torch.argmax(y_prob, dim=1).view(-1)

                    _, output_label = DG(imgs)
                    _, output_unlabel = DG(x_unlabel)
                    _, output_fake = DG(f_imgs)

                    loss_lab = softXEnt(output_label, y_prob)
                    loss_unlab = 0.5 * (torch.mean(F.softplus(log_sum_exp(output_unlabel)))
                                        - torch.mean(log_sum_exp(output_unlabel))
                                        + torch.mean(F.softplus(log_sum_exp(output_fake))))
                    dg_loss = loss_lab + loss_unlab

                    acc = torch.mean((output_label.max(1)[1] == y).float())

                    dg_optimizer.zero_grad()
                    dg_loss.backward()
                    dg_optimizer.step()

                    # train G
                    if step % n_critic == 0:
                        freeze(DG)
                        unfreeze(G)
                        z = torch.randn(bs, z_dim).cuda()
                        f_imgs = G(z)
                        mom_gen, output_fake = DG(f_imgs)
                        mom_unlabel, _ = DG(x_unlabel2)

                        mom_gen = torch.mean(mom_gen, dim=0)
                        mom_unlabel = torch.mean(mom_unlabel, dim=0)

                        Hloss = entropy(output_fake)
                        g_loss = torch.mean((mom_gen - mom_unlabel).abs()) + 1e-4 * Hloss

                        g_optimizer.zero_grad()
                        g_loss.backward()
                        g_optimizer.step()

                end = time.time()
                interval = end - start

                print("Epoch:%d \tTime:%.2f\tD_loss:%.2f\tG_loss:%.2f\t train_acc:%.2f" % (
                    epoch, interval, dg_loss, g_loss,
                    acc))

                if epoch + 1 >= 100 and (epoch + 1) % 10 == 0:
                    Gpath = os.path.join(save_model_dir, "{}_G_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, ac)
                    Dpath = os.path.join(save_model_dir, "{}_D_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, ac)

                    torch.save({'state_dict': G.state_dict()}, Gpath)
                    torch.save({'state_dict': DG.state_dict()}, Dpath)

    elif args.dataset == 'mnist':
        hp_ac_list = [
            # # mnist-coco
            (1, 50, 99.51),
        ]
        for (a1, a2, ac) in hp_ac_list:
            # print("a1:", a1, "a2:", a2, "test_acc:", ac)
            if 'MCNN' in args.targetor_name:
                T = model.MCNN(5)
            elif 'LeNet5' in args.targetor_name:
                T = model.LeNet5(5)
            elif 'LeNet3' in args.targetor_name:
                T = model.LeNet3(5)
            T = torch.nn.DataParallel(T, device_ids=args.device_idx).cuda()

            args.model_path = f'../FL/targ_model/{args.dataset}/{args.fl_timestr}'
            path_T = os.path.join(args.model_path, args.targetor_name)
            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'], strict=False)
            dataset, dataloader = utils.init_dataloader(loaded_args, file_path, batch_size, mode="gan")

            G = GeneratorMNIST(z_dim)
            G = torch.nn.DataParallel(G, device_ids=args.device_idx).cuda()
            DG = MinibatchDiscriminator_MNIST()
            DG = torch.nn.DataParallel(DG, device_ids=args.device_idx).cuda()

            dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
            g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

            entropy = HLoss()

            step = 0
            for epoch in range(0, epochs):
                start = time.time()
                _, unlabel_loader1 = init_dataloader(loaded_args, file_path, batch_size, mode="gan", iterator=True)
                _, unlabel_loader2 = init_dataloader(loaded_args, file_path, batch_size, mode="gan", iterator=True)

                for i, imgs in enumerate(dataloader):
                    current_iter = epoch * len(dataloader) + i + 1

                    step += 1
                    imgs = imgs.cuda()
                    bs = imgs.size(0)
                    x_unlabel = unlabel_loader1.next()
                    x_unlabel2 = unlabel_loader2.next()

                    freeze(G)
                    unfreeze(DG)

                    z = torch.randn(bs, z_dim).cuda()
                    f_imgs = G(z)

                    y_prob = T(imgs)[-1]

                    targetprobs = nn.functional.softmax(y_prob, dim=1)
                    # print(entropy(targetprobs))

                    y = torch.argmax(y_prob, dim=1).view(-1)

                    _, output_label = DG(imgs)
                    _, output_unlabel = DG(x_unlabel)
                    _, output_fake = DG(f_imgs)

                    loss_lab = softXEnt(output_label, y_prob)
                    loss_unlab = 0.5 * (torch.mean(F.softplus(log_sum_exp(output_unlabel)))
                                        - torch.mean(log_sum_exp(output_unlabel))
                                        + torch.mean(F.softplus(log_sum_exp(output_fake))))
                    dg_loss = loss_lab + loss_unlab

                    acc = torch.mean((output_label.max(1)[1] == y).float())

                    dg_optimizer.zero_grad()
                    dg_loss.backward()
                    dg_optimizer.step()

                    # train G
                    if step % n_critic == 0:
                        freeze(DG)
                        unfreeze(G)
                        z = torch.randn(bs, z_dim).cuda()
                        f_imgs = G(z)
                        mom_gen, output_fake = DG(f_imgs)
                        mom_unlabel, _ = DG(x_unlabel2)

                        mom_gen = torch.mean(mom_gen, dim=0)
                        mom_unlabel = torch.mean(mom_unlabel, dim=0)

                        Hloss = entropy(output_fake)
                        g_loss = torch.mean((mom_gen - mom_unlabel).abs()) + 1e-4 * Hloss

                        g_optimizer.zero_grad()
                        g_loss.backward()
                        g_optimizer.step()

                end = time.time()
                interval = end - start

                flogger.info("Epoch:%d \tTime:%.2f\tD_loss:%.2f\tG_loss:%.2f\t train_acc:%.2f" % (
                    epoch, interval, dg_loss, g_loss,
                    acc))

                if (epoch + 1) % 10 == 0:
                    z = torch.randn(32, z_dim).cuda()
                    fake_image = G(z)
                    save_tensor_images(fake_image.detach(),
                                       os.path.join(save_img_dir, "result_image_{}.png".format(epoch + 1)),
                                       nrow=8)

                if epoch + 1 == epochs or (epoch+1 >= 100 and (epoch+1)%50 == 0):
                    flogger.info(f'saving weights file to {save_model_dir}')

                    Gpath = os.path.join(save_model_dir, f"G_Eps{epoch+1}_{args.targetor_name}")
                    Dpath = os.path.join(save_model_dir, f"D_Eps{epoch+1}_{args.targetor_name}")

                    torch.save({'state_dict': G.state_dict()}, Gpath)
                    torch.save({'state_dict': DG.state_dict()}, Dpath)




