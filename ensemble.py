import argparse
import os
from pathlib import Path

import pandas as pd

from logger import setup_logging, get_logger
from utils import msg_box


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ensemble')
    parser.add_argument('--k_fold', default=3, type=int)
    parser.add_argument('--metric_dir', default="saved/dataset_model/0/metrics_best", type=str)
    parser.add_argument('--log_dir', default="saved/dataset_model/0/log", type=str)
    parser.add_argument('--log_name', default="info.log", type=str)
    args = parser.parse_args()

    log_means = pd.DataFrame()
    for k in range(1, args.k_fold + 1):
        read_path = os.path.join(args.metric_dir, f'fold_{k}.pkl')
        log_mean = pd.read_pickle(read_path)
        log_means = pd.concat([log_means, log_mean], axis=1)

    mean = log_means.mean(axis=1)
    std = log_means.std(axis=1)
    result = pd.concat([mean, std], axis=1)
    result.columns = ['mean', 'std']

    log_path = Path(args.log_dir)
    setup_logging(log_path, filename=args.log_name)
    logger = get_logger('ensemble')
    k_fold_msg = msg_box(f"{args.k_fold}-fold cross validation averaged result")
    logger.info(f"{k_fold_msg}\n{result}")
