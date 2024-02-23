import torch, os, time, random, generator, discri
import numpy as np
import torch.nn as nn
import statistics
from argparse import ArgumentParser
from fid_score import calculate_fid_given_paths
from collections import OrderedDict
# from fid_score_raw import calculate_fid_given_paths0
import torch.nn.functional as F

device = "cuda"

import sys
sys.path.append('../BiDO/')
sys.path.append('../')

import model, utils
from utils import save_tensor_images
from BiDO.hsic import hsic_objective, coco_objective, hsic_single_objective, coco_single_objective
from resnet import ResNet18

def inversion(args, G, D, T, E, iden, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500,
              clip_range=1, num_seeds=10, verbose=False):
    iden = iden.view(-1).long().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    bs = iden.shape[0]

    G.eval()
    D.eval()
    T.eval()
    E.eval()

    flag = torch.zeros(bs)

    res = []
    res5 = []
    for random_seed in range(num_seeds):
        tf = time.time()

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        z = torch.randn(bs, 100).cuda().float()
        z.requires_grad = True
        v = torch.zeros(bs, 100).cuda().float()

        for i in range(iter_times):
            fake = G(z)
            label = D(fake)
            out = T(fake)[-1]

            if z.grad is not None:
                z.grad.data.zero_()

            Prior_Loss = - label.mean()
            Iden_Loss = criterion(out, iden)
            Total_Loss = Prior_Loss + lamda * Iden_Loss

            hsic_loss = 0
            seg_num = int(bs/2)
            h_target = fake[0:seg_num].view(seg_num, -1)
            h_data = fake[seg_num:2*seg_num].view(seg_num, -1)
            # print(f'fake size is {fake.size()}')
            # print(f'h_target size is {h_target.size()}')
            # print(f'h_data size is {h_data.size()}')

            if args.attack_improve == 'BATCHHSIC':
                hsic_loss = hsic_single_objective(
                    h_target=h_target,
                    h_data=h_data,
                    sigma=args.sigma,
                    ktype=args.kernelfunc
                )
            elif args.attack_improve == 'BATCHCOCO':
                hsic_loss = coco_single_objective(
                    h_target=h_target,
                    h_data=h_data,
                    sigma=args.sigma,
                    ktype=args.kernelfunc
                )

            hsic_loss = args.lamda * hsic_loss

            Total_Loss +=  hsic_loss

            Total_Loss.backward()

            v_prev = v.clone()
            gradient = z.grad.data
            v = momentum * v - lr * gradient
            z = z + (- momentum * v_prev + (1 + momentum) * v)
            z = torch.clamp(z.detach(), -clip_range, clip_range).float()
            z.requires_grad = True

            Prior_Loss_val = Prior_Loss.item()
            Iden_Loss_val = Iden_Loss.item()

            if verbose:
                if (i + 1) % 500 == 0:
                    fake_img = G(z.detach())
                    print(fake_img.shape)

                    if args.dataset == 'celeba':
                        eval_prob = E(utils.low2high(fake_img))[-1]
                    elif args.dataset == 'mnist':
                        eval_prob = E(fake_img)[-1]


                    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                    acc = iden.eq(eval_iden.long()).sum().item() * 100.0 / bs
                    print("Seeds: {}/{}\tIteration:{}/{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tHSIC Loss:{:.2f}\tAttack Acc:{:.2f}".format(
                        random_seed+1, num_seeds,i + 1, iter_times,
                                                                                                        Prior_Loss_val,
                                                                                                        Iden_Loss_val,
                                                                                                        hsic_loss,
                                                                                                        acc))

            torch.cuda.empty_cache()
        fake = G(z)
        if args.dataset == 'celeba':
            eval_prob = E(utils.low2high(fake))[-1]
        elif args.dataset == 'mnist':
            eval_prob = E(fake)[-1]

        eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
        cnt, cnt5 = 0, 0
        for i in range(bs):
            gt = iden[i].item()

            sample = fake[i]
            save_tensor_images(sample.detach(),
                               os.path.join(args.save_img_dir,
                                            "attack_iden_{:03d}|{}.png".format(gt + 1, random_seed + 1)))

            if eval_iden[i].item() == gt:
                cnt += 1
                flag[i] = 1
                ims = G(z)
                best_img = ims[i]
                save_tensor_images(best_img.detach(),
                                   os.path.join(args.success_dir,
                                                "attack_iden_{:03d}|{}.png".format(gt + 1, random_seed + 1)))

            _, top5_idx = torch.topk(eval_prob[i], 5)
            if gt in top5_idx:
                cnt5 += 1

        res.append(cnt * 100.0 / bs)
        res5.append(cnt5 * 100.0 / bs)
        interval = time.time() - tf
        print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 100.0 / bs))

        # del z, v, fake, v_prev, fake_img, h_target, h_data, gradient
        torch.cuda.empty_cache()
        # del z, v, fake, v_prev, fake_img, h_target, h_data, gradient

    acc = statistics.mean(res)
    acc_5 = statistics.mean(res5)
    acc_var = statistics.stdev(res)
    acc_var5 = statistics.stdev(res5)
    print("Acc:{:.2f}\tAcc_5:{:.2f}\tAcc_var:{:.4f}\tacc_var5{:.4f}".format(acc, acc_5, acc_var, acc_var5))

    return acc, acc_5, acc_var, acc_var5


if __name__ == '__main__':
    parser = ArgumentParser(description='Step2: targeted recovery')
    parser.add_argument('--dataset', default='celeba', help='celeba | cxr | mnist')
    parser.add_argument('--defense', default='reg', help='reg | vib | HSIC')
    parser.add_argument('--attack_improve', default='BATCHHSIC', help='BATCHHSIC | BATCHCOCO')
    parser.add_argument('--lamda', type=float, default='0.05')
    parser.add_argument('--sigma', type=float, default='5.0', help='kernal width')
    parser.add_argument('--kernelfunc', type=str, default='gaussian', help='gaussian |  linear | IMQ')
    parser.add_argument('--times', default=5, type=int)
    parser.add_argument('--ids', default=300, type=int)
    parser.add_argument('--save_img_dir', default='./attack_res/')
    parser.add_argument('--success_dir', default='')
    parser.add_argument('--model_path', default='./targ_model')
    parser.add_argument('--targetor_name', default='mnist_MCNN_98.41.tar')
    parser.add_argument('--verbose', action='store_true', help='')
    parser.add_argument('--seeds', default=10, type=int)
    parser.add_argument('--iter', default=2000, type=int)
    parser.add_argument('--FLID', default='Dec-20-2022-00-45-14')
    parser.add_argument('--gantimestr', type=str, default='', help="which gan for GMI")
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

    ############################# mkdirs ##############################
    args.save_img_dir = os.path.join(args.save_img_dir, args.dataset, 'FL-' + args.FLID, args.targetor_name, f'M({args.attack_improve})-Ts({args.times})-lamda({args.lamda})-Sigma({args.sigma})-seeds{args.seeds}-iter{args.iter}')
    args.success_dir = args.save_img_dir + "/res_success"
    os.makedirs(args.success_dir, exist_ok=True)
    args.save_img_dir = os.path.join(args.save_img_dir, 'all')
    os.makedirs(args.save_img_dir, exist_ok=True)

    eval_path = "./eval_model"
    ############################# mkdirs ##############################

    if args.dataset == 'celeba':
        model_name = "VGG16"
        num_classes = 1000

        e_path = os.path.join(eval_path, "FaceNet_95.88.tar")
        E = model.FaceNet(num_classes)
        E = nn.DataParallel(E).cuda()
        ckp_E = torch.load(e_path)
        E.load_state_dict(ckp_E['state_dict'], strict=False)

        g_path = "./result/models_celeba_gan/celeba_G_300.tar"
        G = generator.Generator()

        G = nn.DataParallel(G).cuda()
        ckp_G = torch.load(g_path)
        G.load_state_dict(ckp_G['state_dict'], strict=False)

        d_path = "./result/models_celeba_gan/celeba_D_300.tar"
        D = discri.DGWGAN()
        D = nn.DataParallel(D).cuda()
        ckp_D = torch.load(g_path)
        D.load_state_dict(ckp_D['state_dict'], strict=False)

        if args.defense == 'HSIC' or args.defense == 'COCO':
            hp_ac_list = [
                # HSIC
                # 1
                # (0.05, 0.5, 80.35),
                # (0.05, 1.0, 70.08),
                # (0.05, 2.5, 56.18),
                # 2
                # (0.05, 0.5, 78.89),
                # (0.05, 1.0, 69.68),
                # (0.05, 2.5, 56.62),
                # no defense
                (0.0, 0.0, 87.63),
            ]
            for (a1, a2, ac) in hp_ac_list:
                print("a1:", a1, "a2:", a2, "test_acc:", ac)

                T = model.VGG16(num_classes, True)
                T = nn.DataParallel(T).cuda()

                # model_tar = f"{model_name}_{a1:.3f}&{a2:.3f}_{ac:.2f}.tar"
                model_tar = args.targetor_name
                # path_T = os.path.join(args.model_path, args.dataset, args.defense, model_tar)

                path_T = os.path.join(args.model_path, args.dataset, args.FLID, args.targetor_name)

                ckp_T = torch.load(path_T)
                T.load_state_dict(ckp_T)
    

                res_all = []
                # ids = 300
                ids = args.ids
                times = args.times
                ids_per_time = ids // times
                iden = torch.from_numpy(np.arange(ids_per_time))
                for idx in range(times):
                    print("--------------------- Attack batch [%s]------------------------------" % idx)
                    res = inversion(args, G, D, T, E, iden, iter_times=args.iter, num_seeds=args.seeds, verbose=True)
                    res_all.append(res)
                    iden = iden + ids_per_time

                res = np.array(res_all).mean(0)
                # print(f"Acc:{res[0]:.4f} (+/- {res[2]:.4f}), Acc5:{res[1]:.4f} (+/- {res[3]:.4f})")
                print(f"Acc:{res[0]:.2f}+-{res[2]:.2f}        Acc5:{res[1]:.2f}+-{res[3]:.2f}")

                fid_value = calculate_fid_given_paths(args.dataset,
                                                      [f'../attack_dataset/{args.dataset}/trainset/',
                                                       f'attack_res/{args.dataset}/FL-{args.FLID}/{args.targetor_name}/M({args.attack_improve})-Ts({args.times})-lamda({args.lamda})-Sigma({args.sigma})-seeds{args.seeds}-iter{args.iter}/all/'],
                                                      50, args.gpu, 2048)
                print(f'FID:{fid_value:.4f}')

        else:
            if args.defense == "vib":
                path_T_list = [
                    os.path.join(args.model_path, args.dataset, args.defense, "VGG16_beta0.003_77.59.tar"),
                    os.path.join(args.model_path, args.dataset, args.defense, "VGG16_beta0.010_67.72.tar"),
                    os.path.join(args.model_path, args.dataset, args.defense, "VGG16_beta0.020_59.24.tar"),
                ]
                for path_T in path_T_list:
                    T = model.VGG16_vib(num_classes)
                    T = nn.DataParallel(T).cuda()

                    checkpoint = torch.load(path_T)
                    ckp_T = torch.load(path_T)
                    T.load_state_dict(ckp_T['state_dict'])

                    res_all = []
                    ids = 300
                    times = 5

                    ids_per_time = ids // times
                    iden = torch.from_numpy(np.arange(ids_per_time))
                    for idx in range(times):
                        print("--------------------- Attack batch [%s]------------------------------" % idx)
                        res = inversion(args, G, D, T, E, iden, iter_times=2000, verbose=True)
                        res_all.append(res)
                        iden = iden + ids_per_time

                    res = np.array(res_all).mean(0)
                    fid_value = calculate_fid_given_paths(args.dataset,
                                                          [f'attack_res/{args.dataset}/trainset/',
                                                           f'attack_res/{args.dataset}/{args.defense}/all/'],
                                                          50, 1, 2048)
                    print(f"Acc:{res[0]:.4f} (+/- {res[2]:.4f}), Acc5:{res[1]:.4f} (+/- {res[3]:.4f})")
                    print(f'FID:{fid_value:.4f}')

            elif args.defense == 'reg':
                path_T = os.path.join(args.model_path, args.dataset, args.FLID, args.targetor_name)
                # path_T = os.path.join(args.model_path, args.dataset, args.defense, "VGG16_reg_87.27.tar")
                T = model.VGG16(num_classes)

                T = nn.DataParallel(T).cuda()

                # checkpoint = torch.load(path_T)
                ckp_T = torch.load(path_T)
                T.load_state_dict(ckp_T)
                # T.load_state_dict(old_state_T)

                # T.load_state_dict(ckp_T['state_dict'])

                res_all = []
                # ids = 300
                ids = args.ids
                times = args.times
                ids_per_time = ids // times
                iden = torch.from_numpy(np.arange(ids_per_time))
                for idx in range(times):
                    print("--------------------- Attack batch [%s]------------------------------" % idx)
                    res = inversion(args, G, D, T, E, iden, lr=2e-2, iter_times=args.iter, num_seeds=args.seeds, verbose=True)
                    res_all.append(res)
                    iden = iden + ids_per_time

                res = np.array(res_all).mean(0)
                print(f"Acc:{res[0]:.2f}+-{res[2]:.2f}        Acc5:{res[1]:.2f}+-{res[3]:.2f}")

                fid_value = calculate_fid_given_paths(args.dataset,
                                                      [f'../attack_dataset/{args.dataset}/trainset/',
                                                       f'attack_res/{args.dataset}/FL-{args.FLID}/{args.targetor_name}/M({args.attack_improve})-Ts({args.times})-lamda({args.lamda})-Sigma({args.sigma})-seeds{args.seeds}-iter{args.iter}/all/'],
                                                      50, args.gpu, 2048)
                print(f'FID:{fid_value:.4f}')

    elif args.dataset == 'mnist':
        num_classes = 5

        e_path = os.path.join(eval_path, "mnist_SCNN.tar")
        E = model.SCNN(10)
        # E.cuda()
        E = nn.DataParallel(E, device_ids=args.device_idx).cuda()
        ckp_E = torch.load(e_path)
        E.load_state_dict(ckp_E['state_dict'])


        g_path = f"./result/{args.gantimestr}/models_gan/mnist_G.tar"
        G = generator.GeneratorMNIST()
        G = nn.DataParallel(G, device_ids=args.device_idx).cuda()
        ckp_G = torch.load(g_path)
        G.load_state_dict(ckp_G['state_dict'])
        # G.to(device)

        d_path = f"./result/{args.gantimestr}/models_gan/mnist_D.tar"
        D = discri.DGWGAN32()
        D = nn.DataParallel(D, device_ids=args.device_idx).cuda()
        ckp_D = torch.load(d_path)
        D.load_state_dict(ckp_D['state_dict'])
        # D.to(device)

        if args.defense == "HSIC":
            pass
        elif args.defense == 'reg':

            path_T = os.path.join(args.model_path, args.dataset, args.FLID, args.targetor_name)
            if 'MCNN' in args.targetor_name:
                T = model.MCNN(num_classes)
            elif 'LeNet5' in args.targetor_name:
                T = model.LeNet5(num_classes)
            # T = nn.DataParallel(T).cuda()
            T.cuda()
            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'])
            # T.to(device)

            aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
            K = 1
            for i in range(K):
                print(f'| Start the {i+1}/{K} GMI attack', flush=True)
                iden = torch.from_numpy(np.arange(5))
                acc, acc5, var, var5 = inversion(args, G, D, T, E, iden, lr=0.01, lamda=100,
                                                 iter_times=args.iter, num_seeds=args.seeds, verbose=args.verbose)
                aver_acc += acc / K
                aver_acc5 += acc5 / K
                aver_var += var / K
                aver_var5 += var5 / K

            print(f"Acc:{aver_acc:.2f}+-{aver_var:.2f}, Acc5:{aver_acc5:.2f}+-{aver_var5:.2f}")

            fid_value = calculate_fid_given_paths(args.dataset,
                                                  [f'../attack_dataset/mnist/trainset/',
                                                   f'attack_res/{args.dataset}/FL-{args.FLID}/{args.targetor_name}/M({args.attack_improve})-Ts({args.times})-lamda({args.lamda})-Sigma({args.sigma})-seeds{args.seeds}-iter{args.iter}/all/'],
                                                  50, args.gpu, 2048)


            print(f'FID:{fid_value:.4f}')

