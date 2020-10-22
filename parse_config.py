import argparse
import yaml

args = argparse.ArgumentParser()
args.add_argument('-c', '--config', default='config.yaml', type=str)
config = args.parse_args()

