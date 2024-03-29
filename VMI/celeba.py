import torch
import numpy as np
import pickle
import os
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pylab as plt
import PIL
import pandas

# CELEBA_IMGS = 'data/img_align_celeba'
CELEBA_IMGS = r'../attack_dataset/celeba/img_align_celeba'
CELEBA_ROOT = '../attack_dataset/celeba'



def create_split_by_pid_frequencies(subset):
    # Get PIDs
    with open(os.path.join(CELEBA_ROOT, 'identity_CelebA.txt'), 'r') as f:
        lines = f.readlines()

        def _process_line(line):
            A, B = line.strip().split(' ')
            # Make the idxs start from 0, hence -1
            A = int(A.split('.')[0]) - 1
            B = int(B) - 1
            print(A, B)
            exit()
            return A, B

        imgid2personid = dict([_process_line(line) for line in lines])

    print("===> Calculating PID frequencies")
    # Look at personid frequencies
    unique_pids = np.arange(10177)
    all_pids = np.array(list(imgid2personid.values()))
    all_iids = np.array(list(imgid2personid.keys()))
    freqs = [(all_pids == i).sum() for i in unique_pids]

    # Split by frequency
    sorted_pids = np.argsort(freqs)
    top_pids = sorted_pids[-1000:]
    aux_pids = sorted_pids[:-1000]

    if subset == 'target':
        candidate_pids = top_pids
        cutoffN = 25
    elif subset == 'aux':
        candidate_pids = aux_pids
        cutoffN = -1
    elif subset == 'all':
        candidate_pids = all_pids
        cutoffN = -1

    train_ids, test_ids = [], []
    for y, pid in tqdm(enumerate(candidate_pids), desc='PIDs'):
        eids = np.where(all_pids == pid)[0]
        for j, eid in enumerate(eids):
            id = all_iids[eid]
            T = len(eids) * 8 // 10 if cutoffN == -1 else cutoffN
            if j < T:
                train_ids.append(id)
            else:
                test_ids.append(id)
    return train_ids, test_ids


def get_celeba_dataset(subset, crop=True):
    assert subset in ['target', 'aux', 'all']
    print("===> Loading Images")
    if crop:
        cache_path = "celeba1k-Feb25-64x64-crop.npz"
    else:
        cache_path = "celeba1k-Feb25-64x64.npz"
    if os.path.exists(cache_path):
        print("===> Loading cache")
        X = np.load(open(cache_path, 'rb')).astype('float32')
        X = torch.from_numpy(X / 255.)
    else:
        tr = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.CenterCrop((128 if not crop else 114)),
            transforms.Resize(64),
            transforms.ToTensor()
        ])
        # Get Images
        print("=====> Loading all image")
        X = []
        # Img_names = []
        celeba_imgdir = CELEBA_IMGS
        n_imgs = 202599
        for i in tqdm(range(1, n_imgs + 1)):
            im = PIL.Image.open(os.path.join(celeba_imgdir, f"{i:06d}.jpg"))
            im = tr(im)
            X.append(im)
            # Img_names.append(f"{i:06d}.png")
        X = torch.stack(X)
        print("DONE loading")

        # Cache
        np.save(open(cache_path, 'wb'), (X.numpy() * 255).astype('uint8'))

    ###
    # Get Images
    # print("=====> Loading all image names")
    # Img_names = []
    # n_imgs = 202599
    # for i in tqdm(range(1, n_imgs + 1)):
    #     Img_names.append(f"{i:06d}.png")
    # print("DONE loading")
    ###

    X = X.float() * 2 - 1
    print("min-max value of the data:", X.min(), X.max())
    print("===> Getting PIDs")
    # Get PIDs
    with open(os.path.join(CELEBA_ROOT, 'identity_CelebA.txt'), 'r') as f:
        lines = f.readlines()

        def _process_line(line):
            A, B = line.strip().split(' ')
            # Make the idxs start from 0, hence -1
            A = int(A.split('.')[0]) - 1
            B = int(B) - 1
            return A, B

        imgid2personid = dict([_process_line(line) for line in lines])

    print("===> Calculating PID frequencies")
    # Look at personid frequencies
    unique_pids = np.arange(10177)
    all_pids = np.array(list(imgid2personid.values()))
    freqs = [(all_pids == i).sum() for i in unique_pids]

    # Split by frequency
    sorted_pids = np.argsort(freqs)
    top_pids = sorted_pids[-1000:]
    aux_pids = sorted_pids[:-1000]

    if subset == 'target':
        candidate_pids = top_pids
        cutoffN = 25
    elif subset == 'aux':
        candidate_pids = aux_pids
        cutoffN = -1
    elif subset == 'all':
        candidate_pids = all_pids
        cutoffN = -1

    train_x, train_y, test_x, test_y = [], [], [], []
    # train_imgs, test_imgs = [], []
    for y, pid in tqdm(enumerate(candidate_pids), desc='PIDs'):
        eids = np.where(all_pids == pid)[0]
        for j, eid in enumerate(eids):
            x = X[eid]
            # img_names = Img_names[eid]
            T = len(eids) * 8 // 10 if cutoffN == -1 else cutoffN
            if j < T:
                train_x.append(x)
                train_y.append(y)
                # train_imgs.append(img_names)
            else:
                test_x.append(x)
                test_y.append(y)
                # test_imgs.append(img_names)

    # vim_train = os.path.join(CELEBA_ROOT, 'vim_train.txt')
    # vim_test = os.path.join(CELEBA_ROOT, 'vim_test.txt')
    # f_vim_train = open(vim_train, "w")
    # f_vim_test = open(vim_test, "w")
    # 
    # for (img_name, label) in zip(train_imgs, train_y):
    #     line = f"{img_name} {label}\n"
    #     f_vim_train.write(line)
    #
    # for (img_name, label) in zip(test_imgs, test_y):
    #     line = f"{img_name} {label}\n"
    #     f_vim_test.write(line)
    #
    train_x = torch.stack(train_x)
    test_x = torch.stack(test_x)
    train_y = torch.LongTensor(train_y)
    test_y = torch.LongTensor(test_y)

    return train_x, train_y, test_x, test_y


def main(batch_size):
    train_x, train_y, test_x, test_y = get_celeba_dataset('target')
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        train_x, train_y), batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        test_x, test_y), batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


if __name__ == '__main__':
    train_x, _, test_x, _ = get_celeba_dataset('target', crop=False)
