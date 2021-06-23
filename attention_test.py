import torch
import torchvision
from torchvision import transforms

from squeeze_layer import squeeze
from inv_attention import InvAttention_dot, InvAttention_concat, InvAttention_gaussian, InvAttention_embedded_gaussian
import argparse
import time
from torch.autograd import Variable
import numpy as np
import os
from spectral_norm_fc import spectral_norm_fc
from celebA import CelebADataset

class Attention_TestConcat(torch.nn.Module):
    def __init__(self, convGamma=False):
        super(Attention_TestConcat, self).__init__()
        self.squeeze_layer = squeeze(2)
        self.attention_layer = InvAttention_concat(12, convGamma=convGamma)
    def forward(self, x):
        x = self.squeeze_layer.forward(x)
        x = self.attention_layer.forward(x, ignore_logdet=True)[0]
        return x
    def inverse(self, x, maxIter=100):
        x = self.attention_layer.inverse(x, maxIter=maxIter)
        x = self.squeeze_layer.inverse(x)
        return x
    def inspect_lip(self, x, eps = 0.00001):
        x = self.squeeze_layer(x)
        dx = x * eps
        y1 = self.attention_layer.res_branch.forward(x)
        y2 = self.attention_layer.res_branch.forward(x + dx)
        lip = torch.dist(y2, y1) / torch.dist((x + dx), x)
        return lip

class Attention_TestDot(torch.nn.Module):
    def __init__(self, convGamma=False):
        super(Attention_TestDot, self).__init__()
        self.squeeze_layer = squeeze(2)
        self.attention_layer = InvAttention_dot(12, convGamma=convGamma)
    def forward(self, x):
        x = self.squeeze_layer.forward(x)
        x = self.attention_layer.forward(x, ignore_logdet=True)[0]
        return x
    def inverse(self, x, maxIter=100):
        x = self.attention_layer.inverse(x, maxIter=maxIter)
        x = self.squeeze_layer.inverse(x)
        return x
    def inspect_lip(self, x, eps=0.00001):
        x = self.squeeze_layer(x)
        dx = x * eps
        y1 = self.attention_layer.res_branch.forward(x)
        y2 = self.attention_layer.res_branch.forward(x + dx)
        lip = torch.dist(y2, y1) / torch.dist((x + dx), x)
        return lip

class Attention_TestGaussian(torch.nn.Module):
    def __init__(self, convGamma=False):
        super(Attention_TestGaussian, self).__init__()
        self.squeeze_layer = squeeze(2)
        self.attention_layer = InvAttention_gaussian(12, convGamma=convGamma)
    def forward(self, x):
        x = self.squeeze_layer.forward(x)
        x = self.attention_layer.forward(x, ignore_logdet=True)[0]
        return x
    def inverse(self, x, maxIter=100):
        x = self.attention_layer.inverse(x, maxIter=maxIter)
        x = self.squeeze_layer.inverse(x)
        return x
    def inspect_lip(self, x, eps=0.00001):
        x = self.squeeze_layer(x)
        dx = x * eps
        y1 = self.attention_layer.res_branch.forward(x)
        y2 = self.attention_layer.res_branch.forward(x + dx)
        lip = torch.dist(y2, y1) / torch.dist((x + dx), x)
        return lip

class Attention_TestEmbeddedGaussian(torch.nn.Module):
    def __init__(self, convGamma):
        super(Attention_TestEmbeddedGaussian, self).__init__()
        self.squeeze_layer = squeeze(2)
        self.attention_layer = InvAttention_embedded_gaussian(12, convGamma=convGamma)
    def forward(self, x):
        x = self.squeeze_layer.forward(x)
        x = self.attention_layer.forward(x, ignore_logdet=True)[0]
        return x
    def inverse(self, x, maxIter=100):
        x = self.attention_layer.inverse(x, maxIter=maxIter)
        x = self.squeeze_layer.inverse(x)
        return x
    def inspect_lip(self, x, eps=0.00001):
        x = self.squeeze_layer(x)
        dx = x * eps
        y1 = self.attention_layer.res_branch.forward(x)
        y2 = self.attention_layer.res_branch.forward(x + dx)
        lip = torch.dist(y2, y1) / torch.dist((x + dx), x)
        return lip


class Conv_Test(torch.nn.Module):
    def __init__(self, use_cuda):
        super(Conv_Test, self).__init__()
        self.squeeze_layer = squeeze(2)
        self.conv_layer = self._spec_norm_wrapper(torch.nn.Conv2d(in_channels=12, out_channels=12, kernel_size=1))
        if use_cuda:
            self.conv_layer = self.conv_layer.cuda()
            self.conv_layer.weight = self.conv_layer.weight.cuda()
    def forward(self, x):
        x = self.squeeze_layer.forward(x)
        Fx =  self.conv_layer.forward(x)
        x = x + Fx
        return x
    def inverse(self, y, maxIter=100):
        x = y
        for i in range(maxIter):
            x = y - self.conv_layer.forward(x)
        x = self.squeeze_layer.inverse(x)
        return x
    def inspect_lip(self, x, eps=0.00001):
        x = self.squeeze_layer(x)
        dx = x * eps
        y1 = self.conv_layer(x)
        y2 = self.conv_layer(x + dx)
        lip = torch.dist(y2, y1) / torch.dist((x + dx), x)
        return lip
    def _spec_norm_wrapper(self, layer):
        return spectral_norm_fc(layer, coeff=.9, n_power_iterations=5)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--save_dir', type=str, default='results/invattention_test')
parser.add_argument('--show_image', type=bool, default=True)
parser.add_argument('--model', type=str, default='concat')
parser.add_argument('--inverse', type=int, default=50)
parser.add_argument('--convGamma', type=bool, default=True)
parser.add_argument('--fullscale', type=bool, default=False)
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

def try_make_dir(d):
    if not os.path.isdir(d):
        os.mkdir(d)

def main():
    args = parser.parse_args()
    if args.fullscale and args.dataset != 'celebA':
        print("Fullscale only supported for celebA")
        exit()
    if args.fullscale and args.batch != 1:
        print("Fullscale only support batch = 1")
        exit()
    try_make_dir(args.save_dir)
    use_cuda = torch.cuda.is_available()
    dens_est_chain = [
        lambda x: (255. * x) + torch.zeros_like(x).uniform_(0., 1.),
        lambda x: x / 256.,
        lambda x: x - 0.5
    ]
    test_chain = [transforms.ToTensor()]
    train_chain = [transforms.ToTensor()]
    transform_train = transforms.Compose(train_chain + dens_est_chain)
    transform_test = transforms.Compose(test_chain + dens_est_chain)

    inverse_den_est_chain = [
        lambda x: x + 0.5
    ]
    inverse_den_est = transforms.Compose(inverse_den_est_chain)


    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        train_subset = torch.utils.data.Subset(trainset, list(range(1000)))
        test_subset = torch.utils.data.Subset(testset, list(range(1000)))
    elif args.dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(
            root='./data', split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.SVHN(
            root='./data', split='test', download=True, transform=transform_test)
        train_subset = torch.utils.data.Subset(trainset, list(range(1000)))
        test_subset = torch.utils.data.Subset(testset, list(range(1000)))
    elif args.dataset == 'celebA':
        resize_chain = [transforms.CenterCrop((178, 178)), transforms.Resize((32, 32))]
        if not args.fullscale:
            transform_train = transforms.Compose(train_chain + dens_est_chain +  resize_chain)
        else:
            transform_train = transforms.Compose(train_chain + dens_est_chain)
        dataset = CelebADataset('./data', transform=transform_train)
        length = len(dataset)
        train_subset = torch.utils.data.Subset(dataset, list(range(1000)))
        test_subset = torch.utils.data.Subset(dataset, [length - i - 1 for i in list(range(1000))])


    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch,
                                              shuffle=True, num_workers=2,drop_last=True,
                                              worker_init_fn=np.random.seed(1234))
    testloader = torch.utils.data.DataLoader(test_subset, batch_size=args.batch,
                                             shuffle=False, num_workers=2,drop_last=True,
                                             worker_init_fn=np.random.seed(1234))
    if args.model == 'dot':
        model = Attention_TestDot(convGamma=args.convGamma)
    elif args.model == 'gaussian':
        model = Attention_TestGaussian()
    elif args.model == 'concat':
        model = Attention_TestConcat(convGamma=args.convGamma)
    elif args.model == 'embedded':
        model = Attention_TestEmbeddedGaussian(convGamma=args.convGamma)
    else:
        model = Conv_Test(use_cuda)

    if use_cuda:
        model = model.cuda()

    if not args.fullscale:
        target = torch.randn([args.batch, 12, 16, 16])
    else:
        target = torch.randn([args.batch, 12, 109, 89])
    # if args.dataset == 'celebA':
    #     target = torch.randn([args.batch, 12, 109, 89])
    #target = Variable(target)
    if use_cuda:
        target = target.cuda()


    criterion = torch.nn.MSELoss()

    optim = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0)

    elapsed_time = 0.
    if args.dataset != 'celebA':
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()

            for batch_idx, (inputs, _) in enumerate(trainloader):
                optim.zero_grad()
                if use_cuda:
                    inputs = inputs.cuda()
                inputs = Variable(inputs, requires_grad=True)
                output = model.forward(inputs)
                loss = criterion(output, target)
                loss.backward()
                optim.step()

            epoch_time = time.time() - start_time
            elapsed_time += epoch_time
            print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))
    else:
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            for batch_idx, inputs in enumerate(trainloader):
                optim.zero_grad()
                if use_cuda:
                    inputs = inputs.cuda()
                inputs = Variable(inputs, requires_grad=True)
                output = model.forward(inputs)
                loss = criterion(output, target)
                loss.backward()
                optim.step()

            epoch_time = time.time() - start_time
            elapsed_time += epoch_time
            print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))


    try_make_dir(args.save_dir)
    model = model.eval()
    data_dir = os.path.join(args.save_dir, 'data')
    recon_dir = os.path.join(args.save_dir, 'recon')
    try_make_dir(data_dir)
    try_make_dir(recon_dir)

    if args.dataset != 'celebA':
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch = Variable(inputs, requires_grad=True)
            if use_cuda:
                batch = batch.cuda()
            output = model(batch)
            inverse_input = model.inverse(output, maxIter=args.inverse)
            batch = inverse_den_est(batch)
            inverse_input = inverse_den_est(inverse_input)
            for i in range(args.batch):
                index = batch_idx * args.batch + i
                torchvision.utils.save_image(batch[i].cpu(),
                                                os.path.join(data_dir, "data_" +str(index)+ ".jpg"), normalize=False)
                torchvision.utils.save_image(inverse_input[i].cpu(),
                                                os.path.join(recon_dir, "recon_" + str(index) + ".jpg"), normalize=False)
    else:
        if not args.fullscale:
            for batch_idx, inputs in enumerate(testloader):
                batch = Variable(inputs, requires_grad=True)
                if use_cuda:
                    batch = batch.cuda()
                output = model(batch)
                inverse_input = model.inverse(output, maxIter=args.inverse)
                batch = inverse_den_est(batch)
                inverse_input = inverse_den_est(inverse_input)
                for i in range(args.batch):
                    index = batch_idx * args.batch + i
                    torchvision.utils.save_image(batch[i].cpu(),
                                                    os.path.join(data_dir, "data_" + str(index) + ".jpg"), normalize=False)
                    torchvision.utils.save_image(inverse_input[i].cpu(),
                                                    os.path.join(recon_dir, "recon_" + str(index) + ".jpg"), normalize=False)
        else:
            for batch_idx, inputs in enumerate(testloader):
                batch = inputs
                batch = Variable(batch, requires_grad=True)
                if use_cuda:
                    batch = batch.cuda()
                output = model(batch)
                inverse_input = model.inverse(output, maxIter=args.inverse)
                batch = inverse_den_est(batch)
                inverse_input = inverse_den_est(inverse_input)
                if args.show_image:
                    torchvision.utils.save_image(batch[0].cpu(),
                                                os.path.join(data_dir, "data_" + str(batch_idx) + ".jpg"),
                                                normalize=False)
                    torchvision.utils.save_image(inverse_input[0].cpu(),
                                                os.path.join(recon_dir, "recon_" + str(batch_idx) + ".jpg"),
                                                normalize=False)


if __name__ == '__main__':
    main()
