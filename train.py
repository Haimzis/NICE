"""Training procedure for NICE.
"""

import argparse
import numpy as np
import torch
import torchvision
from torchvision import transforms
from collections import defaultdict
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import nice


def train(flow, trainloader, optimizer, epoch):
    flow.train()  # set to training mode
    epoch_loss = 0
    for inputs, _ in tqdm(trainloader, desc=f"Training Epoch {epoch}:", total=len(trainloader)):
        optimizer.zero_grad() 
        inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3])  # change  shape from BxCxHxW to Bx(C*H*W)
        inputs = inputs.to(flow.device)
        loss = -flow(inputs).mean() 
        loss.backward()
        optimizer.step() 
        epoch_loss += loss.item()
    return epoch_loss / len(trainloader)


def test(flow, testloader, filename, epoch, sample_shape):
    flow.eval()  # set to inference mode
    with torch.no_grad():
        samples = flow.sample(100).cpu()
        # a, b = samples.min(), samples.max()
        # samples = (samples - a) / (b - a + 1e-10)
        samples = samples.view(-1, sample_shape[0], sample_shape[1], sample_shape[2])
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     './samples/' + filename + 'epoch%d.png' % epoch)
        epoch_loss = 0
        for inputs, _ in testloader:
            inputs = inputs.to(flow.device)
            inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3])
            loss = -flow(inputs).mean()
            epoch_loss += loss.item()
        epoch_loss /= len(testloader)
    return epoch_loss

def add_noise(x):
    return x + torch.zeros_like(x).uniform_(0., 1. / 256.)

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_shape = [1, 28, 28]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.,)),
        add_noise  # dequantization
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
                                              train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
                                                     train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                    train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        raise ValueError('Dataset not implemented')

    model_save_filename = '%s_' % args.dataset \
                          + 'batch%d_' % args.batch_size \
                          + 'coupling%d_' % args.coupling \
                          + 'coupling_type%s_' % args.coupling_type \
                          + 'mid%d_' % args.mid_dim \
                          + 'hidden%d_' % args.hidden \
                          + '.pt'

    flow = nice.NICE(
        prior=args.prior,
        coupling=args.coupling,
        coupling_type=args.coupling_type,
        in_out_dim=np.prod(sample_shape),
        mid_dim=args.mid_dim,
        hidden=args.hidden,
        device=device).to(device)

    optimizer = torch.optim.Adam(
        flow.parameters(), lr=args.lr)

    train_loss_lst, test_loss_lst = [], []
    for epoch in trange(args.epochs):
        train_loss = train(flow, trainloader, optimizer, epoch)
        test_loss = test(flow, testloader, model_save_filename, epoch, sample_shape)
        train_loss_lst.append(train_loss)
        test_loss_lst.append(test_loss)
        print(f'Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}')
        torch.save(flow.state_dict(), './models/' + model_save_filename)

    fig, ax = plt.subplots()
    ax.plot(train_loss_lst)
    ax.plot(test_loss_lst)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(['Train Loss','Test Loss'])
    fig.savefig('./plots/'+ model_save_filename + '.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--prior',
                        help='latent distribution.',
                        type=str,
                        default='logistic')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--coupling-type',
                        help='.',
                        type=str,
                        default='additive')
    parser.add_argument('--coupling',
                        help='.',
                        type=int,
                        default=4)
    parser.add_argument('--mid-dim',
                        help='.',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='.',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    parser.add_argument('--num_workers',
                        help='num_workers for dataloading',
                        type=int,
                        default=2)

    args = parser.parse_args()
    main(args)
