import utils
from utils import *
from generator import *
from discri import *
import torch.nn as nn
import torch.optim as optim
import torch, time, time, os, logging, statistics
from torch.autograd import Variable
import numpy as np
from generator import Generator
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from recover_vib import reparameterize, dist_inversion
from fid_score import calculate_fid_given_paths
from fid_score_raw import calculate_fid_given_paths
from resnet import  ResNet18


import sys

sys.path.append('../BiDO')
sys.path.append('../')
from BiDO.hsic import hsic_objective, coco_objective, hsic_single_objective, coco_single_objective
import model


def dist_inversion_hsicgan(args, G, D, T, E, iden, lr=2e-2, lamda=100, iter_times=1500, clip_range=1,
                   improved=False, num_seeds=5, verbose=False):
    iden = iden.view(-1).long().to('cuda')
    criterion = nn.CrossEntropyLoss().to('cuda')
    bs = iden.shape[0]

    G.eval()
    D.eval()
    T.eval()
    E.eval()

    tf = time.time()

    # NOTE
    mu = Variable(torch.zeros(bs, 100), requires_grad=True)
    log_var = Variable(torch.zeros(bs, 100), requires_grad=True)

    params = [mu, log_var]
    solver = optim.Adam(params, lr=lr)

    for i in range(iter_times):
        z = reparameterize(mu, log_var).to('cuda')
        fake = G(z)
        if improved == True:
            _, label = D(fake)
        else:
            label = D(fake)

        out = T(fake)[-1]

        for p in params:
            if p.grad is not None:
                p.grad.data.zero_()

        if improved:
            Prior_Loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
        else:
            Prior_Loss = - label.mean()

        Iden_Loss = criterion(out, iden)

        Total_Loss = Prior_Loss + lamda * Iden_Loss

        hsic_loss = 0
        seg_num = int(bs / 2)
        # print(seg_num, fake.size())
        h_target = fake[0:seg_num].view(seg_num, -1)
        h_data = fake[seg_num:2*seg_num].view(seg_num, -1)

        # print(seg_num, fake.size()
        if args.attack_improve == 'BATCHHSIC':
            hsic_loss = hsic_single_objective(
                h_target=h_target,
                h_data=h_data,
                sigma=args.sigma,
                ktype = args.kernelfunc
            )
        elif args.attack_improve == 'BATCHCOCO':
            hsic_loss = coco_single_objective(
                h_target=h_target,
                h_data=h_data,
                sigma=args.sigma,
                ktype=args.kernelfunc
            )

        hsic_loss = args.lamda * hsic_loss

        Total_Loss += hsic_loss

        Total_Loss.backward()
        solver.step()

        z = torch.clamp(z.detach(), -clip_range, clip_range).float()

        Prior_Loss_val = Prior_Loss.item()
        Iden_Loss_val = Iden_Loss.item()

        if (i + 1) % 500 == 0 and verbose:
            fake_img = G(z.detach())

            if args.dataset == 'celeba':
                eval_prob = E(utils.low2high(fake_img))[-1]
            else:
                eval_prob = E(fake_img)[-1]

            eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
            acc = iden.eq(eval_iden.long()).sum().item() * 100.0 / bs
            print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tHSIC Loss:{:.2f}\tAttack Acc:{:.4f}".format(i + 1, Prior_Loss_val,
                                                                                                Iden_Loss_val, hsic_loss, acc), flush=True)

    if verbose:
        interval = time.time() - tf
        print("Time:{:.2f}".format(interval), flush=True)

    res = []
    res5 = []
    seed_acc = torch.zeros((bs, num_seeds))

    for random_seed in range(num_seeds):
        tf = time.time()
        z = reparameterize(mu, log_var).to('cuda')
        fake = G(z)

        if args.dataset == 'celeba':
            eval_prob = E(utils.low2high(fake))[-1]
        else:
            eval_prob = E(fake)[-1]

        eval_iden = torch.argmax(eval_prob, dim=1).view(-1)

        cnt, cnt5 = 0, 0
        for i in range(bs):
            gt = iden[i].item()
            sample = fake[i]
            save_tensor_images(sample.detach(),
                               os.path.join(args.save_img_dir, "attack_iden_{:03d}|{}.png".format(gt, random_seed + 1)))

            if eval_iden[i].item() == gt:
                seed_acc[i, random_seed] = 1
                cnt += 1
                ims = G(z)
                best_img = ims[i]
                save_tensor_images(best_img.detach(), os.path.join(args.success_dir,
                                                                   "attack_iden_{:03d}|{}.png".format(gt,
                                                                                                      random_seed + 1)))
            _, top5_idx = torch.topk(eval_prob[i], 5)
            if gt in top5_idx:
                cnt5 += 1

        interval = time.time() - tf
        if verbose:
            print("Time:{:.2f}\tSeed:{}\tAcc:{:.4f}\t".format(interval, random_seed, cnt * 100.0 / bs), flush=True)
        res.append(cnt * 100.0 / bs)
        res5.append(cnt5 * 100.0 / bs)

        torch.cuda.empty_cache()

    acc, acc_5 = statistics.mean(res), statistics.mean(res5)
    acc_var = statistics.stdev(res)
    acc_var5 = statistics.stdev(res5)

    if verbose:
        print(f"Acc:{acc:.4f}\tAcc_5:{acc_5:.4f}\tAcc_var:{acc_var:.4f}\tAcc_var5:{acc_var5:.4f}", flush=True)

    return acc, acc_5, acc_var, acc_var5

if __name__ == "__main__":
    parser = ArgumentParser(description='Step2: targeted recovery')
    parser.add_argument('--dataset', default='celeba', help='celeba | mnist | cifar')
    parser.add_argument('--priordata', default='mnist', help='emnist | fmnist')
    parser.add_argument('--defense', default='reg', help='HSIC | COCO | reg')
    parser.add_argument('--attack_improve', default='BATCHHSIC', help='BATCHHSIC | BATCHCOCO')
    parser.add_argument('--lamda', type=float, default='0.05')
    parser.add_argument('--sigma', type=float, default='5.0', help='kernal width')
    parser.add_argument('--kernelfunc', type=str, default='gaussian', help='gaussian |  linear | IMQ')
    parser.add_argument('--times', default=5, type=int)
    parser.add_argument('--iter', default=5000, type=int)
    parser.add_argument('--improved_flag', action='store_true', default=True, help='use improved k+1 GAN')
    parser.add_argument('--root_path', default="./improvedGAN")
    parser.add_argument('--model_path', default='../BiDO/target_model')
    parser.add_argument('--targetor_name', default='mnist_MCNN_98.41.tar')
    parser.add_argument('--g_name', default='mnist_MCNN_98.41.tar')
    parser.add_argument('--d_name', default='mnist_MCNN_98.41.tar')
    parser.add_argument('--save_img_dir', default='./attack_res/')
    parser.add_argument('--success_dir', default='')
    parser.add_argument('--verbose', default=False, type=bool)
    parser.add_argument('--seeds', default=5, type=int)
    parser.add_argument('--FLID', default='Dec-20-2022-00-45-14')
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--gpus', default='0,')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # if args.gpu:
    #     torch.cuda.set_device(args.gpu - 1)
    device = 'cuda' if args.gpu else 'cpu'
    args.device = device
    # args.device_idx = list((args.gpu-1, ))
    args.device_idx = list((int(args.gpu - 1),))

    z_dim = 100
    ############################# mkdirs ##############################
    save_model_dir = os.path.join(args.root_path, args.priordata, args.defense, args.FLID)
    # save_model_dir = os.path.join(args.root_path, args.dataset)

    # args.save_img_dir = os.path.join(args.save_img_dir, args.dataset, args.defense)
    args.save_img_dir = os.path.join(args.save_img_dir, args.dataset, args.priordata, 'FL-' + args.FLID, args.targetor_name, f'G({args.g_name})-M({args.attack_improve})-Ts({args.times})-lamda({args.lamda})-Sigma({args.sigma})-seeds{args.seeds}-iter{args.iter}')
    args.success_dir = args.save_img_dir + "/res_success"
    os.makedirs(args.success_dir, exist_ok=True)

    args.save_img_dir = os.path.join(args.save_img_dir, 'all')
    os.makedirs(args.save_img_dir, exist_ok=True)
    ############################# mkdirs ##############################
    file = "./config/" + args.dataset + ".json"
    loaded_args = load_json(json_file=file)
    stage = loaded_args["dataset"]["stage"]
    model_name = loaded_args["dataset"]["model_name"]

    if args.dataset == 'celeba':
        hp_ac_list = [
            #HSIC
            # (0, 0, 85.31),
            #
            (0.05, 0.5, 80.35),
            # (0.05, 1., 70.31),
            # (0.05, 2.5, 53.49),
            #
            # (0, 1, 64.73),
            # (0.1, 0, 0.83),
            # (0.1, 1, 76.36),
        ]

        for (a1, a2, ac) in hp_ac_list:
            hp_set = "a1 = {:.3f}|a2 = {:.3f}, test_acc={:.2f}".format(a1, a2, ac)
            print(hp_set)

            G = Generator(z_dim)
            G = torch.nn.DataParallel(G).cuda()
            D = MinibatchDiscriminator()
            D = torch.nn.DataParallel(D).cuda()

            path_G = os.path.join(save_model_dir, args.g_name)
            path_D = os.path.join(save_model_dir, args.d_name)

            ckp_G = torch.load(path_G)
            G.load_state_dict(ckp_G['state_dict'], strict=False)
            ckp_D = torch.load(path_D)
            D.load_state_dict(ckp_D['state_dict'], strict=False)

            if args.defense == 'reg':
                T = model.VGG16(1000)
            else:
                T = model.VGG16(1000, True)

            path_T = os.path.join('../FL', 'targ_model', f"{args.dataset}", args.FLID, args.targetor_name)

            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'], strict=False)

            E = model.FaceNet(1000)
            E = torch.nn.DataParallel(E).cuda()
            path_E = './eval_model/FaceNet_95.88.tar'
            ckp_E = torch.load(path_E)
            E.load_state_dict(ckp_E['state_dict'], strict=False)

            ############         attack     ###########
            aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
            # evaluate on the first 300 identities only
            ids = 300
            times = args.times
            ids_per_time = ids // times
            iden = torch.from_numpy(np.arange(ids_per_time))
            for idx in range(times):
                if args.verbose:
                    print("--------------------- Attack batch [%s]------------------------------" % idx, flush=True)

                acc, acc5, var, var5 = dist_inversion_hsicgan(args, G, D, T, E, iden, lr=2e-2, lamda=100,
                                                      iter_times=args.iter, clip_range=1, improved=args.improved_flag,
                                                      num_seeds=args.seeds, verbose=args.verbose)

                iden = iden + ids_per_time
                aver_acc += acc / times
                aver_acc5 += acc5 / times
                aver_var += var / times
                aver_var5 += var5 / times

            print("Avg acc:{:.2f}\tAvg acc5:{:.2f}\tAvg acc_var:{:.4f}\tAvg acc_var5:{:.4f}".format(
                aver_acc,
                aver_acc5,
                aver_var,
                aver_var5), flush=True)

            fid_value = calculate_fid_given_paths(args.dataset,
                                                  [f'../attack_dataset/{args.dataset}/trainset/',
                                                   f'attack_res/{args.dataset}/{args.priordata}/FL-{args.FLID}/{args.targetor_name}/G({args.g_name})-M({args.attack_improve})-Ts({args.times})-lamda({args.lamda})-Sigma({args.sigma})-seeds{args.seeds}-iter{args.iter}/all/'],
                                                  50, args.gpu, 2048)
            print(f'FID:{fid_value:.4f}', flush=True)



    elif args.dataset == 'mnist':
        hp_ac_list = [
            # # mnist-coco
            # (1, 50, 99.51),
            (2, 20, 99.61),
        ]
        for (a1, a2, ac) in hp_ac_list:
            # hp_set = "a1 = {:.3f}|a2 = {:.3f}, test_acc={:.2f}".format(a1, a2, ac)
            # print(hp_set)

            G = GeneratorMNIST(z_dim)
            G = torch.nn.DataParallel(G, device_ids=args.device_idx).cuda()
            D = MinibatchDiscriminator_MNIST()
            D = torch.nn.DataParallel(D, device_ids=args.device_idx).cuda()

            path_G = os.path.join(save_model_dir, f"G_{args.targetor_name}.tar")
            path_D = os.path.join(save_model_dir, f"D_{args.targetor_name}.tar")

            ckp_G = torch.load(path_G)
            G.load_state_dict(ckp_G['state_dict'], strict=False)
            ckp_D = torch.load(path_D)
            D.load_state_dict(ckp_D['state_dict'], strict=False)

            T = model.MCNN(5)
            T.cuda()
            # T = torch.nn.DataParallel(T, device_ids=args.device_idx).cuda()
            path_T = os.path.join('../FL', 'targ_model', f"{args.dataset}", args.FLID, args.targetor_name)
            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'], strict=False)

            E = model.SCNN(10)
            # path_E = './eval_model/mnist_SCNN.tar'
            path_E = './eval_model/SCNN_99.42.tar'
            ckp_E = torch.load(path_E)
            E = nn.DataParallel(E, device_ids=args.device_idx).cuda()
            E.load_state_dict(ckp_E['state_dict'])

            aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
            fid = []
            res_all = []

            K = 3
            for i in range(K):
                if args.verbose:
                    print(f'{i}-th time-------------------------')
                iden = torch.from_numpy(np.arange(5))
                acc, acc5, var, var5 = dist_inversion_hsicgan(args, G, D, T, E, iden, lr=2e-2, lamda=100, iter_times=args.iter,
                                                      clip_range=1, improved=True, num_seeds=args.seeds, verbose=args.verbose)

                res_all.append([acc, acc5, var, var5])

            res = np.array(res_all).mean(0)
            # avg_fid, var_fid = statistics.mean(fid), statistics.stdev(fid)
            print("Avg acc:{:.2f}\tAvg acc5:{:.2f}\tAvg acc_var:{:.4f}\tAvg acc_var5:{:.4f}".format(
                res[0],
                res[1],
                res[2],
                res[3]), flush=True)

            fid_value = calculate_fid_given_paths(args.dataset,
                                                  [f'../attack_dataset/{args.dataset}/trainset/',
                                                   f'attack_res/{args.dataset}/{args.priordata}/FL-{args.FLID}/{args.targetor_name}/G({args.g_name})-M({args.attack_improve})-Ts({args.times})-lamda({args.lamda})-Sigma({args.sigma})-seeds{args.seeds}-iter{args.iter}/all/'],
                                                  50, args.gpu, 2048)
            print(f'FID:{fid_value:.4f}', flush=True)

   