import os
import argparse
import warnings
warnings.filterwarnings('error')
from functools import partial
from tqdm import tqdm

import multiprocessing as mp

from PIL import Image
# import piexif

########################################################
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
# function to process one image
def process_im(target_res, allow_upscale, resampling_algo, in_path, out_path, fname):
	try:
		f_path = os.path.join(in_path, fname)
		# piexif.remove(f_path) # drop any exif data, at least one file has issues with corrupted data - very slow!

		im = Image.open(f_path)

		if not allow_upscale:
			width, height = im.size
			min_side = min(width, height)
			if min_side < target_res:
				return

		im = im.resize((target_res, target_res), resampling_algos[resampling_algo])

		# quick check that the image is not empty
		if im.getbbox() is None:
			print(f"Can not save {(out_path, f'{fname_base}.jpg')} as it has no bounding box / is completely empty!")
			return None

		# drop alpha channel to be safe
		im = im.convert('RGB')

		# drop extension to be safe
		fname_base, _ = os.path.splitext(fname)
		im.save(os.path.join(out_path, f'{fname_base}.jpg'), 'JPEG')

	except (Exception, Warning) as err:
		error_msg = f'Error processing {os.path.join(in_path, fname)}!\n{str(err)}'
		print(error_msg)
		with open('./preprocessing.log', 'a') as f:
			f.write(f'{error_msg}\n')

########################################################
# function to process one dir, in parallel
def process_dir(target_res, allow_upscale, resampling_algo, input_path, output_path, class_dir_map, src_class_dir):
	out_class_dirs = class_dir_map.get(src_class_dir, None)
	if out_class_dirs is None:
		print(f'ERROR could not find the class name for {src_class_dir}!!')
		with open('./preprocessing.log', 'a') as f:
			f.write(f'Error could not find the class name for {src_class_dir}\n')
		return

	os.makedirs(os.path.join(output_path, out_class_dirs), exist_ok=True)

	fnames = [fname for fname in os.listdir(os.path.join(input_path, src_class_dir)) if not os.path.isdir(os.path.join(input_path, src_class_dir, fname))]

	if len(fnames) is None:
		print(f'ERROR {src_class_dir} contained no files!!')
		with open('./preprocessing.log', 'a') as f:
			f.write(f'Error {src_class_dir} contained no files\n')
		return

	for fname in fnames:
			process_im(target_res, allow_upscale, resampling_algo, os.path.join(input_path, src_class_dir), os.path.join(output_path, out_class_dirs), fname)

########################################################
########################################################
if __name__ == '__main__':

	########################################################
	# setup args
	class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
		pass

	parser = argparse.ArgumentParser(description='Author: Matthew Epland', formatter_class=lambda prog: CustomFormatter(prog, max_help_position=30))

	parser.add_argument('-i', '--input_path', dest='input_path', type=str, default='C:/imagenet/ILSVRC2012_img/train', help='Path to top level directory containing imagenet synset (class) subdirectories.')
	parser.add_argument('-o', '--output_path', dest='output_path', type=str, default='C:/imagenet/processed_images/train', help='Path to output directory.')
	parser.add_argument('-m', '--map_path', dest='map_path', type=str, default='C:\imagenet\ILSVRC2017_devkit\data\map_clsloc.txt', help='Path to map_clsloc.txt map file in devkit.')
	parser.add_argument('-s', '--size', dest='target_res', type=int, default=128, help='Size of output image (128 produces a 128x128 image)')
	parser.add_argument('-a', '--algorithm', dest='resampling_algo', type=str, default='bicubic', help='Algorithm used for resampling: bicubic, bilinear, box, hamming, lanczos, nearest')
	parser.add_argument('--flat_dir', dest='flat_dir', action='count', default=0, help='Input directory has no subdirectory structure, just images.')
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
	flat_dir = args.flat_dir
	n_processes = args.n_processes
	allow_upscale = args.allow_upscale

	# do some sanity checking
	if target_res < 16 or 2048 < target_res:
		raise ValueError('Are you sure you want to run with an output image size of {target_res}? If so, you will have to edit the code...')

	if resampling_algo not in resampling_algos.keys():
		raise ValueError('Unknown resampling algorithm {resampling_algo}!')

	n_cores = mp.cpu_count()
	if n_processes > n_cores:
		raise ValueError('Trying to use {n_processes} processes but only have {n_cores} cores!')

	########################################################
	# build helper dicts from map_clsloc.txt
	class_dir_map = {}
	# id_class_map = {}

	if not flat_dir:
		try:
			with open(map_path, 'rb') as f:
				for row in f.readlines():
					row = row.strip()
					arr = row.decode('utf-8').split(' ')

					WNID = arr[0]
					class_id = int(arr[1])
					class_name = arr[2]

					if class_id == 429 and class_name == 'crane':
						class_name = 'crane_bird'
					elif class_id == 782 and class_name == 'maillot':
						class_name = 'maillot_1'
					elif class_id == 977 and class_name == 'maillot':
						class_name = 'maillot_2'

					class_dir_map[WNID] = class_name
					# id_class_map[class_id] = class_name
		except:
			raise IOError(f'Could not find map_clsloc.txt at provided path: {map_path}')

	########################################################
	# actually run
	print(f'n_cores = {n_cores}, using n_processes = {n_processes}')
	pool = mp.Pool(processes=n_processes)

	if flat_dir:
		fnames = [fname for fname in os.listdir(input_path) if not os.path.isdir(os.path.join(input_path, fname))]
		if len(fnames) == 0:
			raise IOError(f'Provided input path {input_path} contained no images')

		os.makedirs(output_path, exist_ok=True)

		process_im_partial = partial(process_im, target_res, allow_upscale, resampling_algo, input_path, output_path)

		for _ in tqdm(pool.imap_unordered(process_im_partial, fnames), total=len(fnames), desc='Images'):
		    pass

	else:
		input_class_dirs = [_dir for _dir in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, _dir))]

		if len(input_class_dirs) == 0:
			raise IOError(f'Provided input path {input_path} contained no subdirectories')

		process_dir_partial = partial(process_dir, target_res, allow_upscale, resampling_algo, input_path, output_path, class_dir_map)

		for _ in tqdm(pool.imap_unordered(process_dir_partial, input_class_dirs), total=len(input_class_dirs), desc='Class subdirectories'):
		    pass

	pool.close()
	pool.join()
