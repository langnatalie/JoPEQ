import torch
import numpy as np


class lattice_quantization:
    def __init__(self, args):
        self.lattice_dim = args.lattice_dim
        self.dither = args.dither
        self.subtract_dither = args.subtract_dither

        if self.lattice_dim > 1:
            gen_mat = torch.tensor([[2, 0], [1, np.sqrt(3)]]).T.to(torch.float32).to(
                args.device)  # lattice generating matrix
            gen_mat = gen_mat / (torch.sqrt(torch.det(gen_mat)) * (np.floor(2 ** args.R)))  # scale generator matrix
            self.gen_mat = gen_mat

    def __call__(self, input):
        # reshape into vector
        input_vec = input.view(-1)

        # Zero pad if needed
        modulo = len(input_vec) % self.lattice_dim
        if modulo:
            pad_with = self.lattice_dim - modulo
            input_vec = torch.cat((input_vec, torch.zeros(pad_with).to(input.dtype).to(input.device)))
        else:
            pad_with = 0

        # encoder
        input_vec = input_vec.view(self.lattice_dim, -1)  # divide input into blocks

        # hexagonal lattice
        # lattice = torch.zeros((lattice_dim, int((2*max_dim+1)**lattice_dim))).to(input.dtype).to(input.device)
        # idx = 0
        # for kk in np.arange(start=-max_dim, stop=max_dim):
        #     for ll in np.arange(start=-max_dim, stop=max_dim):
        #         lattice[:, idx] = torch.tensor([kk, ll]).T
        #         idx += 1

        dither = torch.zeros_like(input_vec, dtype=input.dtype)
        if self.dither:
            dither = torch.matmul(self.gen_mat, 0.5 * (dither.uniform_() - 0.5))  # generate dither

        # quantize
        input_vec = torch.matmul(self.gen_mat,
                                 torch.round(torch.matmul(torch.inverse(self.gen_mat), input_vec + dither)))

        # decoder
        if self.subtract_dither:
            input_vec = (input_vec - dither).view(-1)  # subtracting dither
        input_vec = input_vec[:-pad_with] if pad_with else input_vec  # remove zero padding
        output = input_vec.reshape(input.shape)
        return output


class scalar_quantization:
    def __init__(self, args):
        self.dither = args.dither
        self.subtract_dither = args.subtract_dither
        self.gamma = args.gamma
        self.delta = (2 * self.gamma) / np.floor(2 ** args.R)  # quantization levels spacing
        self.quantizer_type = args.quantizer_type

    def __call__(self, input):
        # decode
        dither = torch.zeros_like(input, dtype=torch.float32)
        if self.dither:
            dither = dither.uniform_(-self.delta / 2, self.delta / 2)  # generate dither
        input = input + dither

        # quantize
        if self.quantizer_type == 'mid-tread':
            input = self.delta * torch.round(input / self.delta)
            input[input >= self.gamma] = self.gamma
            input[input <= -self.gamma] = -self.gamma
        else:
            input = self.delta * (torch.floor(input / self.delta) + 0.5)
            input[input >= self.gamma] = self.gamma - (self.delta / 2)
            input[input <= -self.gamma] = -(self.gamma - (self.delta / 2))

        # encode
        if self.subtract_dither:
            output = input - dither  # subtracting dither
        else:
            output = input
        return output

