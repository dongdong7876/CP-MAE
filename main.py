import os
import random
import argparse
import configparser

import numpy as np
import pandas as pd
from torch.backends import cudnn
import torch

from solver import Solver
import sys


class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass

def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        os.mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    else:
        solver.train(resume=True)
        results_pd = solver.test()

        result = np.array(
            [config.dataset, 'CP-MAE', config.num_patch])
        column_names = ['data_name', 'algo', 'num_patch']
        result_df = pd.DataFrame([result], columns=column_names)

        if results_pd is not None:
            results = pd.concat([result_df, results_pd.reset_index(drop=True)], axis=1)
        else:
            results = result_df

        result_path = os.path.join('results', 'results_CP-MAE.csv')
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        file_exists = os.path.exists(result_path)
        results.to_csv(result_path, mode='a', header=not file_exists, index=False)
        print(f"Results saved to {result_path}")
    return solver


if __name__ == '__main__':
    def list_of_ints(arg):
        return [int(x) for x in arg.split(',')]

    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument('--dataset', type=str, default='SWaT', help='Dataset name (e.g., SMD, SWaT)')
    initial_parser.add_argument('--config', type=str, default='', help='Path to configuration file')

    init_args, _ = initial_parser.parse_known_args()

    data_name = init_args.dataset
    config_file = init_args.config if init_args.config else f"config/{data_name}.conf"

    fileconfig = configparser.ConfigParser()
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    fileconfig.read(config_file)

    parser = argparse.ArgumentParser(description="CP-MAE Model Pipeline")

    parser.add_argument('--dataset', type=str, default=data_name)
    parser.add_argument('--config', type=str, default=config_file)

    # =========================
    # Basic / Training
    # =========================
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--train_split', type=float, default=0.6)
    parser.add_argument('--lr', type=float, default=fileconfig.getfloat('train', 'lr', fallback=0.0001))
    parser.add_argument('--gpu', type=str, default=fileconfig.get('train', 'gpu', fallback='0'))
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--anomaly_ratio', type=float, default=fileconfig.getfloat('train', 'ar', fallback=1.0))
    parser.add_argument('--batch_size', type=int, default=fileconfig.getint('train', 'bs', fallback=512))
    parser.add_argument('--seed', type=int, default=fileconfig.getint('train', 'seed', fallback=0))

    # =========================
    # Data
    # =========================
    parser.add_argument('--win_size', type=int, default=fileconfig.getint('data', 'win_size'))
    parser.add_argument('--input_c', type=int, default=fileconfig.getint('data', 'input_c'))
    parser.add_argument('--output_c', type=int, default=fileconfig.getint('data', 'output_c'))
    parser.add_argument('--data_path', type=str, default=fileconfig.get('data', 'data_path'))

    # =========================
    # Model backbone
    # =========================
    parser.add_argument('--d_model', type=int, default=fileconfig.getint('param', 'd_model'))
    parser.add_argument('--num_patch', type=list_of_ints,
                        default=list_of_ints(fileconfig.get('param', 'num_patch')))
    parser.add_argument('--num_patches_tf', type=list_of_ints,
                        default=list_of_ints(fileconfig.get('param', 'num_patches_tf')))
    parser.add_argument('--e_layers', type=int, default=fileconfig.getint('param', 'e_layers'))
    parser.add_argument('--dropout', type=float, default=fileconfig.getfloat('param', 'dropout'))
    parser.add_argument('--alpha', type=float, default=fileconfig.getfloat('param', 'alpha'))
    parser.add_argument('--beta', type=float, default=fileconfig.getfloat('param', 'beta'))
    parser.add_argument('--gamma', type=float, default=fileconfig.getfloat('param', 'gamma'))

    parser.add_argument('--st_mask_ratio', type=float, default=fileconfig.getfloat('param', 'st_mask_ratio'))
    parser.add_argument('--tf_mask_ratio', type=float, default=fileconfig.getfloat('param', 'tf_mask_ratio'))

    parser.add_argument('--mc_samples', type=int, default=fileconfig.getint('param', 'mc_samples'))
    parser.add_argument('--mc_mask_ratio_time', type=float,
                        default=fileconfig.getfloat('param', 'mc_mask_ratio_time'))
    parser.add_argument('--mc_mask_ratio_freq', type=float,
                        default=fileconfig.getfloat('param', 'mc_mask_ratio_freq'))

    parser.add_argument('--mode', type=str, default=fileconfig.get('model', 'mode', fallback='train'))
    parser.add_argument('--model_save_path', type=str, default=fileconfig.get('model', 'msp', fallback='cpt'))

    config = parser.parse_args()

    config.model_save_path = f"{config.model_save_path}_{config.seed}"

    args = vars(config)

    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True

    os.makedirs("result", exist_ok=True)

    sys.stdout = Logger("result/" + config.dataset + ".log", sys.stdout)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    main(config)
