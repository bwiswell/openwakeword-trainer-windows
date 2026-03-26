from argparse import ArgumentParser
import os
import warnings

os.environ['HF_HUB_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore')

from .data_manager import DataManager
from .pipeline_step import PipelineStep
from .runner import Runner


parser = ArgumentParser()
parser.add_argument('model')
parser.add_argument('-d', '--data-dir', default=DataManager.DEFAULT_DATA_PATH)
parser.add_argument('-o', '--output-dir', default=DataManager.DEFAULT_OUTPUT_PATH)
parser.add_argument('-s', '--start-from', default='ensure')
parser.add_argument('-e', '--end-at', default='export')
parser.add_argument('-i', '--do-only', default=None)


if __name__ == '__main__':
    args = parser.parse_args()

    r = Runner(args.model, args.data_dir, args.output_dir)
    r.run(
        PipelineStep[args.start_from.upper()],
        PipelineStep[args.end_at.upper()],
        None if args.do_only is None else PipelineStep[args.do_only.upper()]
    )