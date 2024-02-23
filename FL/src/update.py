#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.append('../../BiDO')
sys.path.append('../../VIM')
from BiDO.engine import *
from BiDO.utils import init_dataloader
from BiDO import engine
from VMI.classify_mnist import vmi_train_reg, vmi_test_reg

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, userid, dataset, idxs, logger):
        self.id = userid
        self.args = args
        self.logger = logger
        self.n_classes = 5
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        if args.lossfunc == 'NLL':
            self.criterion = nn.NLLLoss().to(self.device)
        elif args.lossfunc == 'CE':
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        else:
            self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        # idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_train = idxs[:int(len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        if self.args.dataset == 'celeba':
            train_file_path = os.path.join('../data/celeba', self.args.timestr, f'user{self.id}')
            os.makedirs(train_file_path, exist_ok=True)
            train_file = os.path.join(train_file_path, 'trainset.txt')
            f = open(train_file, 'w')
            for idx in idxs_train:
                f.write(dataset[idx])
                f.write('\n')
            f.close()
            trainloader = init_dataloader(self.args.loaded_args, train_file, mode="train")

            test_file = self.args.loaded_args['dataset']['test_file']
            testloader = init_dataloader(self.args.loaded_args, test_file, mode="test")

            validloader = testloader

        else:
            if self.args.local_bs == -1:
                trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                         batch_size=int(len(idxs_train)), shuffle=True)
            else:
                trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                     batch_size=self.args.local_bs, shuffle=True)

            validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                     batch_size=int(len(idxs_val)/10), shuffle=False)
            testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                    batch_size=int(len(idxs_test)/10), shuffle=False)

        return trainloader, validloader, testloader

    def update_weights(self, model, global_round, client_id, test_dataset=None):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0)

        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        # minibatch training
        for iter in range(self.args.local_ep):
            batch_loss = []
            lxz, lyz, cel = [], [], []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                if len(labels) < 2:
                    continue
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()

                if self.args.measure == 'None':
                    fts, outs = model(images)
                    loss = self.criterion(outs, labels)
                elif self.args.measure == 'HSIC' or self.args.measure == 'COCO':   # if measure='COCO', hsic_training=False, it means MID defense training
                    a1, a2 = self.args.lamdax, self.args.lamday
                    loss, cross_loss, out_digit, hx_l_list, hy_l_list = multilayer_hsic(model, self.criterion, images, labels,
                                                                                        a1, a2,
                                                                                        self.n_classes, self.args.ktype, self.args.hsic_training,
                                                                                        self.args.measure)

                    self.args.flogger.info(f'### [{self.args.measure}] batch loss : {loss}')
                    self.args.flogger.info(f'### [{self.args.measure}] batch cross_loss : {cross_loss}')
                    self.args.flogger.info(f'### [{self.args.measure}] batch hx_l_list : {hx_l_list}')
                    self.args.flogger.info(f'### [{self.args.measure}] batch hx_l_list : {hy_l_list}')

                    lxz.append(sum(hx_l_list) / len(hx_l_list))
                    lyz.append(sum(hy_l_list) / len(hy_l_list))
                    cel.append(cross_loss.item())
                else:
                    fts, outs = model(images)
                    loss = self.criterion(outs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.args.flogger.info('| Global Round : {} | Client ID: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\t BatchLoss: {:.6f}'.format(
                    global_round+1, client_id, iter+1, (batch_idx+1) * len(images),
                    len(self.trainloader.dataset),
                    100. * batch_idx / len(self.trainloader), loss.item()))
                # self.logger.add_scalar('loss', loss.item(), 1)
                batch_loss.append(loss.item())

            self.args.flogger.info(f'### batch losses after epoch{iter+1}: {batch_loss}')
            self.args.flogger.info(f'### [{self.args.measure}] batch lxz after epoch{iter + 1}: {lxz}')
            self.args.flogger.info(f'### [{self.args.measure}] batch lyz after epoch{iter + 1}: {lyz}')
            self.args.flogger.info(f'### [{self.args.measure}] batch ce loss after epoch{iter + 1}: {cel}')


            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            self.args.flogger.info(
                '| Global Round : {} | Client ID: {} | Local Epoch : {} | EpochAvgLoss: {:.6f}'.format(
                    global_round + 1, client_id, iter + 1, sum(batch_loss)/len(batch_loss)))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_celeba(self, model, global_round, client_id):
        n_classes = self.args.loaded_args["dataset"]["n_classes"]
        model_name = self.args.loaded_args["dataset"]["model_name"]
        weight_decay = self.args.loaded_args[model_name]["weight_decay"]
        momentum = self.args.loaded_args[model_name]["momentum"]
        # n_epochs = self.args.loaded_args[model_name]["epochs"]
        n_epochs = self.args.local_ep
        lr = self.args.loaded_args[model_name]["lr"]
        milestones = self.args.loaded_args[model_name]["adjust_epochs"]
        gamma = self.args.loaded_args[model_name]["gamma"]
        epoch_loss = []

        if self.args.measure == 'HSIC' or self.args.measure == 'COCO':

            criterion = nn.CrossEntropyLoss().cuda()

            optimizer = torch.optim.Adam(model.parameters(), lr)

            model = torch.nn.DataParallel(model).cuda()
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

            best_ACC = -1
            for epoch in range(n_epochs):
                self.args.flogger.info('=>[Round%d, user%d] Epoch: [%d | %d] LR: %f' % (global_round, self.id, epoch + 1, n_epochs, optimizer.param_groups[0]['lr']))
                train_loss, train_acc = engine.train_HSIC(model, criterion, optimizer, self.trainloader, self.args.lamdax, self.args.lamday, n_classes,
                                                          ktype=self.args.ktype,
                                                          hsic_training=self.args.hsic_training)
                epoch_loss.append(train_loss)

                self.args.flogger.info(
                    f'=>[Round{global_round}, user{self.id}] trainloss:{train_loss},trainacc:{train_acc}')
                    
                test_loss, test_acc = engine.test_HSIC(model, criterion, self.testloader, self.args.lamdax, self.args.lamday, n_classes,
                                                       ktype=self.args.ktype,
                                                       hsic_training=self.args.hsic_training)
                self.args.flogger.info(f'=>[Round{global_round}, user{self.id}] testloss:{test_loss},testacc:{test_acc}')

                if test_acc > best_ACC:
                    best_ACC = test_acc
                    best_model = deepcopy(model)
                scheduler.step()
            self.args.flogger.info(f'=>[Round{global_round}, user{self.id}] best acc:{best_ACC}')

        elif self.args.measure == 'reg':
            optimizer = torch.optim.SGD(params=model.parameters(),
                                        lr=lr,
                                        momentum=momentum,
                                        weight_decay=weight_decay,
                                        nesterov=True
                                        )

            scheduler = MultiStepLR(optimizer, milestones, gamma=gamma)
            criterion = nn.CrossEntropyLoss().cuda()
            model = torch.nn.DataParallel(model).to(device)

            best_acc = -1
            for epoch in range(n_epochs):
                self.args.flogger.info('=>[Round%d, user%d] Epoch: [%d | %d] LR: %f' % (global_round, self.id, epoch + 1, n_epochs, optimizer.param_groups[0]['lr']))
                if self.args.net == 'VGG16':
                    train_loss, train_acc = engine.train_reg(model, criterion, optimizer, self.trainloader)
                elif self.args.net == 'ResNet':
                    train_loss, train_acc = vmi_train_reg(model, optimizer, self.trainloader, epoch)
                epoch_loss.append(train_loss)
                self.args.flogger.info(
                    f'=>[Round{global_round}, user{self.id}] trainloss:{train_loss},trainacc:{train_acc}')
                if self.args.net == 'VGG16':
                    test_loss, test_acc = engine.test_reg(model, criterion, self.testloader)
                elif self.args.net == 'ResNet':
                    test_loss, test_acc = vmi_test_reg(model, self.testloader, epoch)

                self.args.flogger.info(
                    f'=>[Round{global_round}, user{self.id}] testloss:{test_loss},testacc:{test_acc}')

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_model = deepcopy(model)

                scheduler.step()
                self.args.flogger.handlers[0].flush()
                self.args.flogger.handlers[1].flush()

                # save client models every .. epoches
                if self.args.save_every_epoch > 0:
                    if (epoch+1) % self.args.save_every_epoch == 0:
                        # save the model
                        targ_path = f"../targ_model/{self.args.dataset}/{self.args.timestr}"
                        os.makedirs(targ_path, exist_ok=True)
                        targetor_name = f'{self.args.dataset}_{self.args.net}_E[{epoch + 1}]Acc[{str(test_acc * 100)[:5]}].tar'
                        torch.save({'state_dict': model.state_dict()},
                                   os.path.join(targ_path, targetor_name))

            if self.args.save_every_epoch > 0:
                exit(0)
            self.args.flogger.info(f'=>[Round{global_round}, user{self.id}] best acc:{best_acc}')

        return model.state_dict(), sum(epoch_loss)/len(epoch_loss)


    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

def test_inference_celeba(args, model):
    n_classes = args.loaded_args["dataset"]["n_classes"]
    test_file = args.loaded_args['dataset']['test_file']
    testloader = init_dataloader(args.loaded_args, test_file, mode="test")

    if args.measure == 'HSIC' or args.measure == 'COCO':
        criterion = nn.CrossEntropyLoss().cuda()

        model = torch.nn.DataParallel(model).cuda()

        test_loss, test_acc = engine.test_HSIC(model, criterion, testloader, args.lamdax,
                                               args.lamday, n_classes,
                                               ktype=args.ktype,
                                               hsic_training=args.hsic_training)
        args.flogger.info(f'=>[server] testloss:{test_loss},testacc:{test_acc}')

    elif args.measure == 'reg':

        criterion = nn.CrossEntropyLoss().cuda()
        model = torch.nn.DataParallel(model).to(device)

        if args.net == 'VGG16':
            test_loss, test_acc = engine.test_reg(model, criterion, testloader)
        elif args.net == 'ResNet':
            test_loss, test_acc = vmi_test_reg(model, testloader, -1)

        args.flogger.info(
            f'=>[server] testloss:{test_loss},testacc:{test_acc}')


    return test_acc, test_loss

def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'

    if args.lossfunc == 'NLL':
        criterion = nn.NLLLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    testloader = DataLoader(test_dataset, batch_size=1,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        # filter out 5 6 7 8 9 digits
        if labels >= 5:
            continue

        # Inference
        fts, outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss/total
