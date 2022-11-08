import torch
from torch.distributions.laplace import Laplace
import numpy as np
from scipy import stats


class Privacy:
    def __init__(self, args, dither_var):
        self.privacy_noise = args.privacy_noise

        if self.privacy_noise == 'laplace' or self.privacy_noise == 'jopeq_scalar':
            b_lap = 2 / args.epsilon
            self.noise = Laplace(loc=0, scale=b_lap)
            if self.privacy_noise == 'jopeq_scalar':
                var_PPN = 2 * (b_lap ** 2) - dither_var
                self.noise.scale = np.sqrt(var_PPN / 2)

        if self.privacy_noise == 't' or self.privacy_noise == 'jopeq_vector':
            self.sigma_squared, self.nu = args.sigma_squared, args.nu
            cov_PPN = self.sigma_squared * np.eye(2)  # 't'
            self.noise = stats.multivariate_t(shape=cov_PPN, df=self.nu)
            if args.privacy_noise == 'jopeq_vector':  # subtract the cov_dither
                self.noise.shape = ((self.nu - 2) / self.nu) * ((self.nu / (self.nu - 2)) * cov_PPN - dither_var)

    def __call__(self, input):
        if self.privacy_noise == 'laplace' or self.privacy_noise == 'jopeq_scalar':
            privacy_noise = self.noise.sample(input.shape)
        else:
            privacy_noise = torch.from_numpy(self.noise.rvs(size=input.shape[-1]).T)
        return input + privacy_noise.to(input.dtype).to(input.device)