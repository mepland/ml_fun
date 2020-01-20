# commonly used in notebooks, just import them here
import os
import sys
import gc
import json
from collections import OrderedDict
import datetime as dt

import numpy as np
import pandas as pd

from natsort import natsorted
from tqdm import tqdm
import humanize

########################################################
# setup my own large number formatter for convenience and tweakability
def my_large_num_formatter(value, e_precision=3):
	if value > 1000:
		return f'{value:.{e_precision}e}'
	else:
		return f'{value:.0f}'
