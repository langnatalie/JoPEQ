import torch
import torch.optim as optim
import copy
import math
from quantization import LatticeQuantization, ScalarQuantization
from privacy import Privacy

def federated_setup(global_model, train_data, args):
    # create a dict of dict s (local users), i.e. {'1': {'data':..., 'model':..., 'opt':...}, ...}
    indexes = torch.randperm(len(train_data))
    user_data_len = math.floor(len(train_data) / args.num_users)
    local_models = {}
    for user_idx in range(args.num_users):
        user = {'data': torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_data,
                                    indexes[user_idx * user_data_len:(user_idx + 1) * user_data_len]),
            batch_size=args.train_batch_size, shuffle=True),
            'model': copy.deepcopy(global_model)}
        user['opt'] = optim.SGD(user['model'].parameters(), lr=args.lr,
                                momentum=args.momentum) if args.optimizer == 'sgd' \
            else optim.Adam(user['model'].parameters(), lr=args.lr)
        if args.lr_scheduler:
            user['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(user['opt'], patience=10, factor=0.1, verbose=True)
        local_models[user_idx] = user
    return local_models


def distribute_model(local_models, global_model):
    for user_idx in range(len(local_models)):
        local_models[user_idx]['model'].load_state_dict(copy.deepcopy(global_model.state_dict()))


def aggregate_models(local_models, global_model, mechanism):  # FeaAvg
    mean = lambda x: sum(x) / len(x)
    state_dict = copy.deepcopy(global_model.state_dict())
    SNR_layers = []
    for key in state_dict.keys():
        local_weights_average = torch.zeros_like(state_dict[key])
        SNR_users = []
        for user_idx in range(0, len(local_models)):
            local_weights_orig = local_models[user_idx]['model'].state_dict()[key] - state_dict[key]
            local_weights = mechanism(local_weights_orig)
            SNR_users.append(torch.var(local_weights_orig) / torch.var(local_weights_orig - local_weights))
            local_weights_average += local_weights
        SNR_layers.append(mean(SNR_users))
        state_dict[key] += (local_weights_average / len(local_models)).to(state_dict[key].dtype)
    global_model.load_state_dict(copy.deepcopy(state_dict))
    return mean(SNR_layers)


class JoPEQ:  # Privacy Quantization class
    def __init__(self, args):
        self.vec_normalization = args.vec_normalization
        dither_var = None
        if args.quantization:
            if args.lattice_dim > 1:
                self.quantizer = LatticeQuantization(args)
                dither_var = self.quantizer.P0_cov
            else:
                self.quantizer = ScalarQuantization(args)
                dither_var = (self.quantizer.delta ** 2) / 12
        else:
            self.quantizer = None
        if args.privacy:
            self.privacy = Privacy(args, dither_var)
        else:
            self.privacy = None

    def divide_into_blocks(self, input, dim=2):
        # Zero pad if needed
        modulo = len(input) % dim
        if modulo:
            pad_with = dim - modulo
            input_vec = torch.cat((input, torch.zeros(pad_with).to(input.dtype).to(input.device)))
        else:
            pad_with = 0
        input_vec = input.view(dim, -1)  # divide input into blocks
        return input_vec, pad_with,

    def __call__(self, input):
        original_shape = input.shape
        input = input.view(-1)
        if self.vec_normalization:  # normalize
            input, pad_with = self.divide_into_blocks(input)

        mean = torch.mean(input, dim=-1, keepdim=True)
        std = torch.norm(input - mean) / (input.shape[-1] ** 0.5)

        std = 3 * std
        input = (input - mean) / std

        if self.privacy is not None:
            input = self.privacy(input)

        if self.quantizer is not None:
            input = self.quantizer(input)

        # denormalize
        input = (input * std) + mean

        if self.vec_normalization:
            input = input.view(-1)[:-pad_with] if pad_with else input  # remove zero padding

        input = input.reshape(original_shape)

        return input
