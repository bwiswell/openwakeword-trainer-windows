import os
import sys
import warnings

from .runner import Runner

os.environ['HF_HUB_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    r = Runner(sys.argv[1])
    r.run()