from torch.distributions.laplace import Laplace
import numpy as np


class Privacy:
    def __init__(self, args):
        b_lap = 2 / args.epsilon
        if args.privacy_noise == 'laplace':
            self.noise = Laplace(loc=0, scale=b_lap)
        else:
            assert args.epsilon < (np.sqrt(24) * (2 ** args.R) - 1) / (
                        2 * args.R), "reformulate eps and R to hold theorem 2 " \
                                     "from the paper"  # if condition returns
            # False, AssertionError is raised:
            delta = (2 * args.gamma) / (2 ** args.R)
            var_PPN = 2 * (b_lap ** 2) - ((delta ** 2) / 12)
            b_PPN = np.sqrt(var_PPN / 2)
            self.noise = Laplace(loc=0, scale=b_PPN)

    def __call__(self, input):
        privacy_noise = self.noise.sample(input.shape)
        return input + privacy_noise.to(input.dtype).to(input.device)
