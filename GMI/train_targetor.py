import argparse
from torchvision import datasets, transforms
import torch
import tqdm
import os
import sys
sys.path.append('..')
sys.path.append('../BiDO')
import time
from torch.utils.data import DataLoader, Dataset
from BiDO.model import SCNN, MCNN


def test_inference(args, model, testloader):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.lossfun == 'NLL':
        criterion = torch.nn.NLLLoss().to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        if args.model == 'evaluator':
            features, outputs = model(images)
        elif args.model == 'targetor':
            features, outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss/total

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        pass
    elif args.dataset == 'mnist':
        data_dir = '../attack_dataset/mnist_tmp/'
        apply_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))])
            #transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

    trainloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)

    return trainloader, testloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist',
                        help="mnist")
    parser.add_argument('--model', type=str, default='evaluator',
                        help="evaluator || targetor")
    parser.add_argument('--epochs', type=int, default=100,
                        help="")
    parser.add_argument('--batchsize', type=int, default=32,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lossfun', type=str, default='CE',
                        help="CE || NLL")
    parser.add_argument('--tacc', type=float, default=0.98,
                        help='target acc')

    args = parser.parse_args()

    lr = args.lr
    batch_size = args.batchsize
    epochs = args.epochs

    print("---------------------Training [%s]------------------------------" % args.model)

    trainloader, testloader= get_dataset(args)

    if args.dataset == 'celeba':
        pass
    elif args.dataset == 'mnist':
        if args.model == 'evaluator':
            train_model = SCNN(10)
        elif args.model == 'targetor':
            train_model = MCNN(10)

    train_model = torch.nn.DataParallel(train_model).cuda()

    if args.lossfun == 'NLL':
        criterion = torch.nn.NLLLoss().cuda()
    elif args.lossfun == 'CE':
        criterion = torch.nn.CrossEntropyLoss().cuda()

    # 0.004
    optimizer = torch.optim.Adam(train_model.parameters(), lr=lr, betas=(0.5, 0.999))

    step = 0
    for epoch in range(args.epochs):
        # start = time.time()
        epoch_loss = []
        for i, (images, labels) in enumerate(trainloader):
            start = time.time()
            step += 1
            images, labels = images.cuda(), labels.cuda()
            bs = images.size(0)


            optimizer.zero_grad()
            features, out = train_model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

            end = time.time()
            interval = end - start
            print("Epoch:%d Run:[%d/%d] \tTime:%.2f\tloss:%.2f" % (epoch+1, bs*i, len(trainloader.dataset), interval, loss))

        test_acc, test_loss = test_inference(args, train_model, testloader)
        print("Test Accuracy: {:.2f}% at Epoch {}".format(100 * test_acc, epoch + 1))
        if test_acc >= args.tacc:
            print('saving weights file')
            if args.model == 'evaluator':
                eval_path = "./eval_model"
                os.makedirs(eval_path, exist_ok = True)
                torch.save({'state_dict': train_model.state_dict()},
                       os.path.join(eval_path, f"{args.dataset}_SCNN_{str(test_acc*100)[:5]}.tar"))
            elif args.model == 'targetor':
                targ_path = "./targ_model"
                os.makedirs(targ_path, exist_ok=True)
                torch.save({'state_dict': train_model.state_dict()},
                           os.path.join(targ_path, f"{args.dataset}_MCNN_{str(test_acc*100)[:5]}.tar"))
            print('succeed saving weights file')
            break

