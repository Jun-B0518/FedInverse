import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import datetime

import torch
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,'
import sys
sys.path.append('../..')
sys.path.append('../../GMI')
sys.path.append('../../BiDO')
sys.path.append('../../VMI')
from options import args_parser
from update import LocalUpdate, test_inference, test_inference_celeba
from models import MLP, MLP_v2, CNNMnist, CNNMnist_v2, CNNFashion_Mnist, CNNCifar, MCNN, LeNet5, LeNet3
from utils import get_dataset, average_weights, exp_details, get_logger
from GMI import attack_fl
from BiDO import model
from BiDO.train_HSIC import load_feature_extractor
from BiDO.utils import load_json
from VMI.classify_mnist import ResNetCls1



if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()

    # log file name define
    log_fname = f'{args.timestr}-M({args.model})-D({args.dataset})-' \
                f'iid({args.iid})-Net({args.net})-Me({args.measure})-Lx({args.lamdax})-Ly({args.lamday})-K({args.num_users})-C({args.frac})-R({args.epochs})-' \
                f'E({args.local_ep})-B({args.local_bs})-lr({args.lr}).log'
    logpath = '../logs'
    os.makedirs(logpath, exist_ok=True)
    logpath = os.path.join(logpath, log_fname)
    flogger = get_logger(logpath)
    args.flogger = flogger

    exp_details(args)
    flogger.info(f'timestr is {args.timestr}')

    # loss logger
    path_project = os.path.abspath('.')
    logger = None

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # if args.gpu:
    #     torch.cuda.set_device(args.gpu-1)
    device = 'cuda' if args.gpu else 'cpu'

    if args.dataset == 'celeba':
        # load configure file
        if args.net == 'VGG16':
            file = os.path.join('../config', args.dataset + ".json")
            args.loaded_args = load_json(json_file=file)
        elif args.net == 'ResNet':
            file = os.path.join('../config', args.dataset + "_resnet.json")
            args.loaded_args = load_json(json_file=file)

    # load dataset and user groups
    flogger.info('To get dataset...')
    train_dataset, test_dataset, user_groups = get_dataset(args)
    flogger.info('Finish getting dataset!')
    flogger.handlers[0].flush()
    flogger.handlers[1].flush()

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            if args.net == 'MCNN':
                global_model = MCNN(5)
            elif args.net == 'LeNet5':
                global_model = LeNet5(5)
            elif args.net == 'LeNet3':
                global_model = LeNet3(5)

        elif args.dataset == 'celeba':
            if args.net == 'VGG16':
                if args.measure == 'HSIC' or args.measure == 'COOC':
                    global_model = model.VGG16(args.loaded_args["dataset"]["n_classes"], hsic_training=args.hsic_training, dataset=args.dataset)

                    load_pretrained_feature_extractor = True
                    if load_pretrained_feature_extractor:
                        pretrained_model_ckpt = "../../BiDO/target_model/vgg16_bn-6c64b313.pth"
                        checkpoint = torch.load(pretrained_model_ckpt)
                        load_feature_extractor(global_model, checkpoint)
                else:
                    global_model = model.VGG16(args.loaded_args["dataset"]["n_classes"])
                    args.flogger.info(global_model.state_dict().keys())
                    load_pretrained_feature_extractor = True
                    if load_pretrained_feature_extractor:
                        pretrained_model_ckpt = "../../BiDO/target_model/vgg16_bn-6c64b313.pth"
                        checkpoint = torch.load(pretrained_model_ckpt)
                        load_feature_extractor(global_model, checkpoint)
            elif args.net == 'ResNet':
                model_name = args.loaded_args["dataset"]["model_name"]
                if model_name == 'ResNetCls1':
                    global_model = ResNetCls1(3, zdim=args.loaded_args[model_name]["latent_dim"], imagesize=64, nclass=args.loaded_args["dataset"]["n_classes"],
                               resnetl=args.loaded_args[model_name]["resnetl"], dropout=args.loaded_args[model_name]["dropout"])

    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    # iterate based on specified rounds
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        flogger.info(f'\n | Global Training Round : {epoch+1} |\n')
        flogger.handlers[0].flush()
        flogger.handlers[1].flush()

        global_model.train()

        # select the clients, proportion: C
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        flogger.info(f'Clients in epoch {epoch+1}:{idxs_users}')

        # local training in each client
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, userid=idx, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            if args.dataset == 'celeba':
                w, loss = local_model.update_weights_celeba(
                    model=copy.deepcopy(global_model), global_round=epoch, client_id=idx)
            else:
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch, client_id=idx, test_dataset=test_dataset)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            # save the client model
            clt_save_path = f"../targ_clientmodel/{args.dataset}/{args.timestr}/R{epoch+1}/User{idx}"
            os.makedirs(clt_save_path, exist_ok=True)
            torch.save(w, os.path.join(clt_save_path, f'user{idx}-R{epoch+1}-{args.net}.tar'))

        global_weights = average_weights(local_weights)

        # update global model
        global_model.load_state_dict(global_weights)

        # record the train loss at each round
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        flogger.info(f'| Global Round[{epoch+1}]:global average training loss is {loss_avg}')
        # logger.add_scalar('TrainLoss vs CommunicationRounds', loss_avg, global_step=epoch+1)

        flogger.info(f' \nAvg Training Stats after {epoch+1} global rounds:')
        flogger.info(f'Training Loss : {np.mean(np.array(train_loss))}')

        # Test inference after each round
        if args.dataset == 'celeba':
            test_acc, test_loss = test_inference_celeba(args, global_model)
            test_acc = test_acc/100.0
        else:
            test_acc, test_loss = test_inference(args, global_model, test_dataset)

        flogger.info(f'| Global Round[{epoch + 1}]:global average test loss is {test_loss}')
        flogger.info(f'| Global Round[{epoch + 1}]:global average test acc is {test_acc}')

        # logger.flush()
        flogger.handlers[0].flush()
        flogger.handlers[1].flush()

        # save the global model in each epoch
        targ_path = f"../targ_model/{args.dataset}/{args.timestr}"
        os.makedirs(targ_path, exist_ok=True)
        targetor_name = f'{args.dataset}_{args.net}_idd[{args.iid}]_R[{epoch+1}]_C[{args.frac}]_E[{args.local_ep}]_B[{args.local_bs}]_Acc[{str(test_acc*100)[:5]}].tar'
        torch.save({'state_dict': global_model.state_dict()},
                   os.path.join(targ_path, targetor_name))
        args.targetor_name = targetor_name

    flogger.info('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    flogger.info(f'Log No.: {args.timestr}')

