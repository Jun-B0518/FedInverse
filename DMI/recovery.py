import utils
from utils import *
from generator import *
from discri import *
import torch.nn as nn
import torch, time, time, os, logging, statistics
import numpy as np
from generator import Generator
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from recover_vib import reparameterize, dist_inversion
from fid_score import calculate_fid_given_paths
from fid_score_raw import calculate_fid_given_paths


import sys

sys.path.append('../BiDO')
import model

if __name__ == "__main__":
    parser = ArgumentParser(description='Step2: targeted recovery')
    parser.add_argument('--dataset', default='celeba', help='celeba | mnist | cifar')
    parser.add_argument('--defense', default='HSIC', help='HSIC | COCO')
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
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--gpus', default='1,2,3')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # if args.gpu:
    #     torch.cuda.set_device(args.gpu - 1)
    device = 'cuda' if args.gpu else 'cpu'
    args.device = device
    args.device_idx = list((args.gpu-1, ))

    z_dim = 100
    ############################# mkdirs ##############################
    save_model_dir = os.path.join(args.root_path, args.dataset, args.defense)
    # save_model_dir = os.path.join(args.root_path, args.dataset)

    # args.save_img_dir = os.path.join(args.save_img_dir, args.dataset, args.defense)
    args.save_img_dir = os.path.join(args.save_img_dir, args.dataset, args.targetor_name+f'-seeds{args.seeds}-iter{args.iter}')
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

            T = torch.nn.DataParallel(T).cuda()
            path_T = os.path.join(args.model_path, f"{args.dataset}", args.defense,
                                  args.targetor_name)

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
            times = 5
            ids_per_time = ids // times
            iden = torch.from_numpy(np.arange(ids_per_time))
            for idx in range(times):
                if args.verbose:
                    print("--------------------- Attack batch [%s]------------------------------" % idx)

                acc, acc5, var, var5 = dist_inversion(args, G, D, T, E, iden, lr=2e-2, lamda=100,
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
                aver_var5))

            fid_value = calculate_fid_given_paths(args.dataset,
                                                  [f'../attack_dataset/{args.dataset}/trainset/',
                                                   f'attack_res/{args.dataset}/{args.targetor_name}-seeds{args.seeds}-iter{args.iter}/all/'],
                                                  50, args.gpu, 2048)
            print(f'FID:{fid_value:.4f}')



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

            epochs = 101
            path_G = os.path.join(save_model_dir, f"{model_name}_G_Eps{epochs}_{args.targetor_name}")
            path_D = os.path.join(save_model_dir, f"{model_name}_D_Eps{epochs}_{args.targetor_name}")

            ckp_G = torch.load(path_G)
            G.load_state_dict(ckp_G['state_dict'], strict=False)
            ckp_D = torch.load(path_D)
            D.load_state_dict(ckp_D['state_dict'], strict=False)

            T = model.MCNN(5)
            T = torch.nn.DataParallel(T, device_ids=args.device_idx).cuda()
            path_T = os.path.join(args.model_path, args.targetor_name)
            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'], strict=False)

            E = model.SCNN(10)
            path_E = '../GMI/eval_model/mnist_SCNN.tar'
            ckp_E = torch.load(path_E)
            E = nn.DataParallel(E, device_ids=args.device_idx).cuda()
            E.load_state_dict(ckp_E['state_dict'])

            aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
            fid = []
            res_all = []

            K = 1
            for i in range(K):
                if args.verbose:
                    print('-------------------------')
                iden = torch.from_numpy(np.arange(5))
                acc, acc5, var, var5 = dist_inversion(args, G, D, T, E, iden, lr=2e-2, lamda=100, iter_times=args.iter,
                                                      clip_range=1, improved=True, num_seeds=args.seeds, verbose=args.verbose)


                # fid_value = calculate_fid_given_paths(args.dataset,
                #                                       [f'attack_res/{args.dataset}/trainset/',
                #                                        f'attack_res/{args.dataset}/{args.defense}/all/'],
                #                                       50, 1, 2048)
                # print(f'FID:{fid_value:.4f}')


                # fid.append(fid_value)
                res_all.append([acc, acc5, var, var5])
                # fid.append(fid_value)

            res = np.array(res_all).mean(0)
            # avg_fid, var_fid = statistics.mean(fid), statistics.stdev(fid)
            print(f"Acc:{res[0]:.4f} (+/- {res[2]:.4f}), Acc5:{res[1]:.4f} (+/- {res[3]:.4f})")
            # print(f'FID:{avg_fid:.4f} (+/- {var_fid:.4f})')

