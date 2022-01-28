from torch.distributions.laplace import Laplace
import torch
import numpy as np

class privacy():  #
    def __init__(self, args):
        b_lap = 2 / args.epsilon
        if args.privacy_noise == 'laplace':
            self.noise = Laplace(loc=0, scale=b_lap)
        else:
            assert args.epsilon < (np.sqrt(24)*(2**args.R)-1)/(2*args.R), "reformulate eps and R to hold theorem 2 from the paper" # if condition returns False, AssertionError is raised:
            delta = (2 * args.gamma) / (2 ** args.R)
            var_PPN = 2 * (b_lap ** 2) - ((delta ** 2) / 12)
            b_PPN = np.sqrt(var_PPN / 2)
            self.noise = Laplace(loc=0, scale=b_PPN)

    def __call__(self, input):
        privacy_noise = self.noise.sample(input.shape)
        return input + privacy_noise.to(input.dtype).to(input.device)


def PPN(R, eps):
    b = 2 / eps
    delta = (2 * R - 1) / (2 ** R)

    N = 101
    w = np.linspace(-np.pi, np.pi, N)
    t, T = np.linspace(-6, 6, N, retstep=True)

    FFT = (1 / T) * ((1 + (b * w) ** 2) ** (-1)) * np.sinc(delta * w / (2 * np.pi)) ** (-1)  # DTFT=(1/T)*FT
    PPN = np.fft.fftshift(abs(np.fft.ifft(np.fft.ifftshift(FFT))))  # IF

    # fit a triangular distribution
    middle_idx = int(np.floor(N / 2))
    left_idx = max(np.where(PPN[:middle_idx] <= 1e-02*max(PPN))[0])  # half life value
    right_idx = middle_idx + (middle_idx - left_idx)

    plt.plot(t, PPN)
    plt.hist(np.random.triangular(t[left_idx], t[middle_idx], t[right_idx], 10000), density=True, histtype='step')

    var_PPN = 2 * (b ** 2) - ((delta ** 2) / 12)
    b_PPN = np.sqrt(var_PPN / 2)
    plt.hist(Laplace(0, b_PPN).sample([10000]), density=True, histtype='step')
    a_PPN = np.sqrt(6*var_PPN)
    plt.hist(np.random.triangular(-a_PPN, 0, a_PPN, 10000), density=True, histtype='step')

    plt.legend(['PPN', 'triangular', 'lap', 'tri'])
    plt.grid()
    plt.show()

    return t[left_idx], t[middle_idx], t[right_idx]


def PPN_time_domain(delta, b):
    N = 101
    w = np.linspace(-np.pi, np.pi, N)
    t, T = np.linspace(-6, 6, N, retstep=True)

    FFT = (1 / T) * ((1 + (b * w) ** 2) ** (-1)) * np.sinc(delta * w / (2 * np.pi)) ** (-1)  # DTFT=(1/T)*FT
    PPN = np.fft.fftshift(abs(np.fft.ifft(np.fft.ifftshift(FFT))))  # IF

    # plotting
    lap = (1 / (2 * b)) * np.exp(-abs(t) / b)
    plt.plot(t, PPN)
    plt.plot(t, lap)
    plt.legend(['PPN' + ' auc=' + '{:.2f}'.format(np.sum(PPN) * T),
                'Laplace' + ' auc=' + '{:.2f}'.format(np.sum(lap) * T)])
    plt.xlabel('t')
    plt.grid()
    # plt.title('$\epsilon=$' + '{:.0f}'.format(2 / b))
    # plt.savefig('../figs/time_eps'+'{:.0f}'.format(2 / b)+'.pdf', transperent=True, bbox_inches='tight')
    plt.show()


def PPN_freq_domain(Delta, b):
    N = 201
    w = np.linspace(-16.5 * np.pi, 16.5 * np.pi, N)
    t, T = np.linspace(-5, 5, N, retstep=True)
    Lap = (1 / T) * ((1 + (b * w) ** 2) ** (-1))
    Dither = np.sinc(Delta * w / (2 * np.pi)) ** (-1)
    plt.plot(w, Lap)
    plt.plot(w, Dither)
    plt.plot(w, Lap * Dither)
    plt.legend([r'$\mathcal{F}(f_{Lap})$',
                r'$\frac{1}{\mathcal{F}(f_{Rect})}$',
                r'$\mathcal{F}(f_{PPN})='
                r'\frac{\mathcal{F}(f_{Lap})}{\mathcal{F}(f_{Rect})}$'])
    plt.ylim(-2 * max(Lap), 2 * max(Lap))
    plt.xlabel(r'$\omega$')
    # plt.title('$\epsilon=$' + '{:.0f}'.format(2 / b))
    plt.grid()
    # plt.show()
    plt.savefig('../figs/freq_eps' + '{:.0f}'.format(2 / b) + '.pdf', transperent=True, bbox_inches='tight')


def PPN_estimation(R, eps):
    b_lap = 2/eps
    delta = (2*(2 * R + (1 / eps)))/(2**R)
    var_PPN = 2*(b_lap**2) - ((delta**2)/12)
    b_PPN = np.sqrt(var_PPN/2)
    print(f'lap: {2*(b_lap**2)}, rect: {(delta**2)/12}, PPN: {2*(b_PPN**2)}')

    N = 10000000
    plt.hist(Laplace(0, b_lap).sample([N]), bins=100, density=True)  # , histtype='step')
    plt.hist(Laplace(0, b_PPN).sample([N]), bins=100, density=True)  # , histtype='step')
    plt.legend(['lap', 'PPN'])
    plt.grid()
    plt.show()


if __name__=='__main__':
    import matplotlib.pyplot as plt
    PPN_estimation(1, 3)