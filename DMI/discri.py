import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import LinearWeightNorm
import torch.nn.init as init


class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        init.normal_(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)  # NxB, subtract self distance
        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)
        return x


class MinibatchDiscriminator(nn.Module):
    def __init__(self, in_dim=3, dim=64, n_classes=1000):
        super(MinibatchDiscriminator, self).__init__()
        self.n_classes = n_classes

        def conv_ln_lrelu(in_dim, out_dim, k, s, p):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, k, s, p),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.layer1 = conv_ln_lrelu(in_dim, dim, 5, 2, 2)
        self.layer2 = conv_ln_lrelu(dim, dim * 2, 5, 2, 2)
        self.layer3 = conv_ln_lrelu(dim * 2, dim * 4, 5, 2, 2)
        self.layer4 = conv_ln_lrelu(dim * 4, dim * 4, 3, 2, 1)
        self.mbd1 = MinibatchDiscrimination(dim * 4 * 4 * 4, 64, 50)
        self.fc_layer = nn.Linear(dim * 4 * 4 * 4 + 64, self.n_classes)

    def forward(self, x):
        bs = x.shape[0]
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        feat = feat4.view(bs, -1)
        mb_out = self.mbd1(feat)  # Nx(A+B)
        y = self.fc_layer(mb_out)

        return feat, y


class MinibatchDiscriminator_MNIST(nn.Module):
    def __init__(self, in_dim=1, dim=64, n_classes=5):
        super(MinibatchDiscriminator_MNIST, self).__init__()
        self.n_classes = n_classes

        def conv_ln_lrelu(in_dim, out_dim, k, s, p):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, k, s, p),
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.layer1 = conv_ln_lrelu(in_dim, dim, 5, 2, 2)
        self.layer2 = conv_ln_lrelu(dim, dim * 2, 5, 2, 2)
        self.layer3 = conv_ln_lrelu(dim * 2, dim * 4, 5, 2, 2)
        self.mbd1 = MinibatchDiscrimination(4096, 64, 50)
        self.fc_layer = nn.Linear(4096 + 64, self.n_classes)

    def forward(self, x):
        bs = x.shape[0]
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat = feat3.view(bs, -1)
        mb_out = self.mbd1(feat)  # Nx(A+B)
        y = self.fc_layer(mb_out)

        return feat, y


class Discriminator(nn.Module):
    def __init__(self, in_dim=3, dim=64, n_classes=1000):
        super(Discriminator, self).__init__()
        self.n_classes = n_classes

        def conv_ln_lrelu(in_dim, out_dim, k, s, p):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, k, s, p),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.layer1 = conv_ln_lrelu(in_dim, dim, 5, 2, 2)
        self.layer2 = conv_ln_lrelu(dim, dim * 2, 5, 2, 2)
        self.layer3 = conv_ln_lrelu(dim * 2, dim * 4, 5, 2, 2)
        self.layer4 = conv_ln_lrelu(dim * 4, dim * 4, 3, 2, 1)
        self.fc_layer = nn.Linear(dim * 4 * 4 * 4, self.n_classes)

    def forward(self, x):
        bs = x.shape[0]
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        feat = feat4.view(bs, -1)
        y = self.fc_layer(feat)

        return feat, y


class DiscriminatorMNIST(nn.Module):
    def __init__(self, d_input_dim=1024, n_classes=5):
        super(DiscriminatorMNIST, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, n_classes)

    # forward method
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        y = self.fc4(x)
        y = y.view(-1)

        return x, y


class DGWGAN32(nn.Module):
    def __init__(self, in_dim=1, dim=64):
        super(DGWGAN32, self).__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.layer1 = nn.Sequential(nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2))
        self.layer2 = conv_ln_lrelu(dim, dim * 2)
        self.layer3 = conv_ln_lrelu(dim * 2, dim * 4)
        self.layer4 = nn.Conv2d(dim * 4, 1, 4)

    def forward(self, x):
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        y = self.layer4(feat3)
        y = y.view(-1)
        return y


class DGWGAN(nn.Module):
    def __init__(self, in_dim=3, dim=64):
        super(DGWGAN, self).__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4))

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y


class DLWGAN(nn.Module):
    def __init__(self, in_dim=3, dim=64):
        super(DLWGAN, self).__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.layer1 = nn.Sequential(nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2))
        self.layer2 = conv_ln_lrelu(dim, dim * 2)
        self.layer3 = conv_ln_lrelu(dim * 2, dim * 4)
        self.layer4 = nn.Conv2d(dim * 4, 1, 4)

    def forward(self, x):
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        y = self.layer4(feat3)
        return y


class DiscriminatorCIFAR10(nn.Module):
    def __init__(self):
        super(DiscriminatorCIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(4*4*256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)

        x = x.view(-1, 4*4*256)
        x = self.fc1(x).view(-1)
        return x


class MinibatchDiscriminatorCIFAR10(nn.Module):
    def __init__(self):
        super(MinibatchDiscriminatorCIFAR10, self).__init__()
        self.dim = 64
        self.n_classes = 5
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(4*4*256, 1)
        self.mbd1 = MinibatchDiscrimination(4096, 64, 50)
        self.fc_layer = nn.Linear(self.dim * 4 * 4 * 4 + 64, self.n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)

        feat = x.view(-1, 4*4*256)
        # feat = self.fc1(x).view(-1)

        mb_out = self.mbd1(feat)  # Nx(A+B)
        y = self.fc_layer(mb_out)

        return feat, y