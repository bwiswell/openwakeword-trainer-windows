from argparse import ArgumentParser
import os
import warnings

os.environ['HF_HUB_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore')

from .data_manager import DataManager
from .runner import Runner


parser = ArgumentParser()
parser.add_argument('model')
parser.add_argument('-d', '--data-dir', default=DataManager.DEFAULT_DATA_PATH)
parser.add_argument('-o', '--output-dir', default=DataManager.DEFAULT_OUTPUT_PATH)


if __name__ == '__main__':
    args = parser.parse_args()

    r = Runner(args.model, args.data_dir, args.output_dir)
    r.run()