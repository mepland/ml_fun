import os
import argparse
from natsort import natsorted
# from tqdm import tqdm

from PIL import Image

# setup resapling algos as dict TODO test all
resampling_algos = {
'bicubic': Image.BICUBIC,
'bilinear': Image.BILINEAR,
'box': Image.BOX,
'hamming': Image.HAMMING,
'lanczos': Image.LANCZOS,
'nearest': Image.NEAREST,
}

########################################################
# setup args
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
	pass

parser = argparse.ArgumentParser(description='Author: Matthew Epland', formatter_class=lambda prog: CustomFormatter(prog, max_help_position=30))

parser.add_argument('-i', '--input_path', dest='input_path', type=str, default='./train', help='Path to top level directory containing imagenet synset (class) subdirectories.')
parser.add_argument('-o', '--output_path', dest='output_path', type=str, default='./processed_images/train', help='Path to output directory.')
parser.add_argument('-m', '--map_path', dest='map_path', type=str, default='./ILSVRC2017_devkit/devkit/data/map_clsloc.txt', help='Path to map_clsloc.txt map file in devkit.')
parser.add_argument('-s', '--size', dest='target_res', type=int, default=128, help='Size of output image (128 produces a 128x128 image)')
parser.add_argument('-a', '--algorithm', dest='resampling_algo', type=str, default='bicubic', help='Algorithm used for resampling: bicubic, bilinear, box, hamming, lanczos, nearest')
parser.add_argument('-j', '--processes', dest='n_processes', type=int, default=1, help='Number of sub-processes to process different sysnet (class) subdirectories in parallel')
parser.add_argument('--allow_upscale', dest='allow_upscale', action='count', default=0, help='Enable upscaling.')
# parser.add_argument('-v','--verbose', dest='verbose', action='count', default=0, help='Enable verbose output.')

# parse the arguments, throw errors if missing any
args = parser.parse_args()

# assign to normal variables for convenience
input_path = args.input_path
output_path = args.output_path
map_path = args.map_path
target_res = args.target_res
resampling_algo = args.resampling_algo
n_processes = args.n_processes
allow_upscale = args.allow_upscale

# do some sanity checking
if target_res < 16 or 2048 < target_res:
	raise ValueError('Are you sure you want to run with an output image size of {target_res}? If so, you will have to edit the code...')

if resampling_algo not in resampling_algos.keys():
	raise ValueError('Unknown resampling algorithm {resampling_algo}!')

########################################################
# build helper dicts from map_clsloc.txt
class_dir_map = {}
# id_class_map = {}

with open(map_path, 'rb') as f:
	for row in f.readlines():
		row = row.strip()
		arr = row.decode('utf-8').split(' ')

		dir_name = arr[0]
		class_id = int(arr[1])
		class_name = arr[2]

		if class_id == 429 and class_name == 'crane':
			class_name = 'crane_bird'
		elif class_id == 782 and class_name == 'maillot':
			class_name = 'maillot_1'
		elif class_id == 977 and class_name == 'maillot':
			class_name = 'maillot_2'

		class_dir_map[dir_name] = class_name
		# id_class_map[class_id] = class_name
except:
	raise IOError(f'Could not find map_clsloc.txt at provided path: {map_path}')

########################################################
# start processing files
input_class_dirs = [_dir for _dir in natsorted(os.listdir(input_path)) if os.path.isdir(os.path.join(input_path, _dir))]
if len(input_class_dirs) == 0:
	raise IOError(f'Provided input path {input_path} contained no subdirectories')

for i, src_class_dir in enumerate(input_class_dirs):
	out_class_dirs = class_dir_map.get(src_class_dir, None)
	if out_class_dirs is None:
		print('ERROR could not find the class name for {src_class_dir}!!')
		with open('./preprocessing.log', 'a') as f:
			f.write(f'Error could not find the class name for {src_class_dir}\n')
		continue

	if not os.path.exists(os.path.join(output_path, out_class_dirs)):
		os.makedirs(os.path.join(output_path, out_class_dirs))

	fnames = os.listdir(os.path.join(input_path, src_class_dir))
	if len(fnames) is None:
		print('ERROR {src_class_dir} contained no files!!')
		with open('./preprocessing.log', 'a') as f:
			f.write(f'Error {src_class_dir} contained no files\n')
		continue

	for fname in fnames:
		try:
			im = Image.open(os.path.join(input_path, src_class_dir, fname))

			width, height = im.size
			min_side = min(width, height)
			if not allow_upscale and min_side < target_res:
				continue

			im = im.resize((target_res, target_res), resampling_algos[resampling_algo])

			# drop extension to be safe
			fname_base, _ = os.path.splitext(fname)
			im.save(os.path.join(output_path, out_class_dirs, f'{fname_base}.jpg'), 'JPEG')

		except OSError as err:
			print(f'Error processing {os.path.join(input_path, src_class_dir, fname)}!')
			with open('./preprocessing.log', 'a') as f:
				f.write(f'Error processing {os.path.join(input_path, src_class_dir, fname)}!\n')
