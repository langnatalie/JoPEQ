import gc
import sys
from statistics import mean
import time
import torch
import torch.optim as optim
from configurations import args_parser
from tqdm import tqdm
import utils
import models
import federated_utils
from torchinfo import summary
import numpy as np

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    boardio, textio, best_val_acc, path_best_model = utils.initializations(args)
    textio.cprint(str(args))

    # data
    train_data, test_loader = utils.data(args)
    input, output, train_data, val_loader = utils.data_split(train_data, len(test_loader.dataset), args)

    # model
    if args.model == 'mlp':
        global_model = models.FC2Layer(input, output)
    elif args.model == 'cnn2':
        global_model = models.CNN2Layer(input, output)
    elif args.model == 'cnn3':
        global_model = models.CNN3Layer()
    else:
        global_model = models.Linear(input, output)
    textio.cprint(str(summary(global_model)))
    global_model.to(args.device)

    train_creterion = torch.nn.CrossEntropyLoss(reduction='mean')
    test_creterion = torch.nn.CrossEntropyLoss(reduction='sum')

    # learning curve
    train_loss_list = []
    val_acc_list = []

    #  inference
    if args.eval:
        global_model.load_state_dict(torch.load(path_best_model))
        test_acc = utils.test(test_loader, global_model, args.device)
        textio.cprint(f'eval test_acc: {test_acc:.0f}%')
        gc.collect()
        sys.exit()

    # training loops
    if not args.federated_learning:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)

        optimizer = optim.SGD(global_model.parameters(), lr=args.lr, momentum=args.momentum) \
            if args.optimizer == 'sgd' \
            else optim.Adam(global_model.parameters(), lr=args.lr)

        for global_epoch in tqdm(range(args.global_epochs)):
            train_loss = utils.train_one_epoch(train_loader, global_model, optimizer, args.device)
            val_acc = utils.test(val_loader, global_model, args.device)
            train_loss_list.append(train_loss)
            val_acc_list.append(val_acc)
            print(f'epoch: {global_epoch} | train_loss: {train_loss:.2f} | val_acc: {val_acc:.0f}%')
        test_acc = utils.test(test_loader, global_model, args.device)
        print(f'final centralized test_acc: {test_acc:.0f}%')
    else:
        local_models = federated_utils.federated_setup(global_model, train_data, args)
        JoPEQ = federated_utils.JoPEQ(args)
        SNR_list = []

        for global_epoch in tqdm(range(0, args.global_epochs)):
            federated_utils.distribute_model(local_models, global_model)
            users_loss = []

            for user_idx in range(args.num_users):
                user_loss = []
                for local_epoch in range(0, args.local_epochs):
                    user = local_models[user_idx]
                    train_loss = utils.train_one_epoch(user['data'], user['model'], user['opt'],
                                                       train_creterion, args.device, args.local_iterations)
                    user['scheduler'].step(train_loss)
                    user_loss.append(train_loss)
                users_loss.append(mean(user_loss))

            train_loss = mean(users_loss)
            SNR = federated_utils.aggregate_models(local_models, global_model, JoPEQ)  # FeaAvg
            SNR_list.append(SNR)
            val_acc = utils.test(val_loader, global_model, test_creterion, args.device)

            train_loss_list.append(train_loss)
            val_acc_list.append(val_acc)

            boardio.add_scalar('train', train_loss, global_epoch)
            boardio.add_scalar('validation', val_acc, global_epoch)
            gc.collect()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(global_model.state_dict(), path_best_model)

            test_acc = utils.test(test_loader, global_model, test_creterion, args.device)
            textio.cprint(f'epoch: {global_epoch} | train_loss: {train_loss:.2f} | '
                          f'val_acc: {val_acc:.0f}% | SNR: {20 * torch.log10(SNR):.3f}')

        textio.cprint(f'avg SNR: {20 * torch.log10(sum(SNR_list) / len(SNR_list)):.3f}')

    np.save(f'checkpoints/{args.exp_name}/train_loss_list.npy', train_loss_list)
    np.save(f'checkpoints/{args.exp_name}/val_acc_list.npy', val_acc_list)
    elapsed_min = (time.time() - start_time) / 60
    textio.cprint(f'total execution time: {elapsed_min:.0f} min')
