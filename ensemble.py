import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ensemble')
    parser.add_argument('-k', default=3, type=int)
    parser.add_argument('--log_dir', default="saved/dataset_model/0/log", type=str)
    parser.add_argument('--metric_dir', default="metrics_best", type=str)
    args = parser.parse_args()
