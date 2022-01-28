import os
from statistics import mean
import matplotlib.pyplot as plt
import torch
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
from scipy.interpolate import interp1d


def data(args):
    if args.data == 'mnist':
        train_data = datasets.MNIST('./data', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((args.norm_mean,), (args.norm_std,))
                                    ]))

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((args.norm_mean,), (args.norm_std,))
            ])),
            batch_size=args.test_batch_size, shuffle=False)
    else:
        train_data = datasets.CIFAR10('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((args.norm_mean,), (args.norm_std,))
                                      ]))

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((args.norm_mean,), (args.norm_std,))
            ])),
            batch_size=args.test_batch_size, shuffle=False)
    return train_data, test_loader


def data_split(data, amount, args):
    # split train, validation
    train_data, val_data = torch.utils.data.random_split(data, [len(data) - amount, amount])
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False)

    # input, output sizes
    in_channels, dim1, dim2 = data[0][0].shape  # images are dim1 x dim2 pixels
    input = dim1 * dim2 if args.model == 'mlp' or args.model == 'linear' else in_channels
    output = len(data.classes)  # number of classes

    return input, output, train_data, val_loader


def train_one_epoch(train_loader, model,
                    optimizer, creterion,
                    device, iterations):
    model.train()
    losses = []
    if iterations:
        local_iteration = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        # send to device
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = creterion(output, label)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if iterations:
            local_iteration += 1
            if local_iteration == iterations:
                break
    return mean(losses)


def test(test_loader, model, creterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)  # send to device

        output = model(data)
        test_loss += creterion(output, label).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


def initializations(args):
    #  reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    #  documentation
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')

    best_val_acc = np.NINF
    path_best_model = 'checkpoints/' + args.exp_name + '/model.best.t7'

    return boardio, textio, best_val_acc, path_best_model


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def plots(*args):
    my_extra_linestyles = {'densely dotted': (0, (1, 1)), 'densely dashed': (0, (5, 1)),
                           'densely dashdotted': (0, (3, 1, 1, 1))}
    x = np.arange(start=1, stop=9)
    if args:
        my_linestyles = ['-', '-*', '-^', '--', '-.', ':']
        for idx, arg in enumerate(args):
            plt.plot(x, arg, my_linestyles[idx], markersize=4)

        plt.legend([r'JoPEQ ($\epsilon=3$)', r'JoPEQ ($\epsilon=3.5$)', r'JoPEQ ($\epsilon=4$)',
                    r'FL+Lap+SDQ ($\epsilon=3$)', r'FL+Lap+SDQ ($\epsilon=3.5$)', r'FL+Lap+SDQ ($\epsilon=4$)'],
                   bbox_to_anchor=(0.565, 0.385, 0, 0))

        plt.xlabel('Bit-rate')
        plt.ylabel('SNR [dB]')
        plt.grid()
        plt.savefig('./checkpoints/fig2/fig2.pdf', transperent=True, bbox_inches='tight')
    else:
        x = np.arange(95)
        X_ = np.linspace(x.min(), x.max(), 500)

        y1 = np.load('./checkpoints/fig1/FL/val_acc_list.npy')[:95]
        cubic_interpolation_model = interp1d(x, y1, kind="cubic")
        Y_ = cubic_interpolation_model(X_)
        plt.plot(X_, Y_, ':')

        y2 = np.load('./checkpoints/fig1/FL+SDQ/val_acc_list.npy')[:95]
        cubic_interpolation_model = interp1d(x, y2, kind="cubic")
        Y_ = cubic_interpolation_model(X_)
        plt.plot(X_, Y_, '--')

        y3 = np.load('./checkpoints/fig1/FL+Lap/val_acc_list.npy')[:95]
        cubic_interpolation_model = interp1d(x, y3, kind="cubic")
        Y_ = cubic_interpolation_model(X_)
        plt.plot(X_, Y_, '-.')

        y4 = np.load('./checkpoints/fig1/FL+Lap+SDQ/val_acc_list.npy')[:95]
        cubic_interpolation_model = interp1d(x, y4, kind="cubic")
        Y_ = cubic_interpolation_model(X_)
        plt.plot(X_, Y_, linestyle=my_extra_linestyles['densely dotted'], markersize=3)

        y5 = np.load('./checkpoints/fig1/JoPEQ/val_acc_list.npy')[:95]
        cubic_interpolation_model = interp1d(x, y5, kind="cubic")
        Y_ = cubic_interpolation_model(X_)
        plt.plot(X_, Y_, '-')

        plt.legend(['FL', 'FL+SDQ', 'FL+Lap',
                    'FL+Lap+SDQ', 'JoPEQ'])

        plt.xlabel('Global iteration')
        plt.ylabel('Accuracy')

        plt.grid()
        plt.savefig('./checkpoints/fig1/fig1.pdf', transperent=True, bbox_inches='tight')


from matplotlib import cm


def plot3d():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.01f}')

    ax.set_xlabel(r'$q_1$')
    ax.set_ylabel(r'$q_2$')
    ax.set_zlabel(r'RMSE$(\mathbf{R})$')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.7, aspect=7, location='left')

    plt.savefig('./checkpoints/fig1/mnist.pdf', transperent=True, bbox_inches='tight')


if __name__ == '__main__':
    # plots()

    y1 = [-16.77, -16.784, -16.31, -15.988, -15.771, -15.697, -15.573, -15.634]
    y2 = [-19.454, -19.51, -17.952, -16.713, -16.002, -15.788, -15.648, -15.649]

    y3 = [-14.3, -14.276, -13.764, -13.358, -13.048, -13.018, -12.987, -12.921]
    y4 = [-18.049, -17.915, -16.017, -14.353, -13.474, -13.171, -12.976, -12.937]

    y5 = [-12.206, -12.089, -11.736, -11.141, -10.781, -10.695, -10.659, -10.62]
    y6 = [-16.814, -16.58, -14.305, -12.42, -11.304, -10.917, -10.769, -10.625]

    plots(y1, y3, y5, y2, y4, y6)
