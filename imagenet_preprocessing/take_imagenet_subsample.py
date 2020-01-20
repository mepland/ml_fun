import os
import shutil
import argparse
from tqdm import tqdm

########################################################
# from https://stackoverflow.com/a/15824216
def recursive_overwrite(src, dest, ignore=None):
	if os.path.isdir(src):
		if not os.path.isdir(dest):
			os.makedirs(dest)
		files = os.listdir(src)
		if ignore is not None:
			ignored = ignore(src, files)
		else:
			ignored = set()
		for f in files:
			if f not in ignored:
				recursive_overwrite(os.path.join(src, f), os.path.join(dest, f), ignore)
	else:
		shutil.copyfile(src, dest)

########################################################
########################################################
if __name__ == '__main__':

	########################################################
	# setup args
	class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
		pass

	parser = argparse.ArgumentParser(description='Author: Matthew Epland', formatter_class=lambda prog: CustomFormatter(prog, max_help_position=30))

	parser.add_argument('-i', '--input_path', dest='input_path', type=str, default='C:/imagenet/processed_images/train', help='Path to top level directory containing imagenet synset (class) subdirectories.')
	parser.add_argument('-o', '--output_path', dest='output_path', type=str, default='C:/imagenet/processed_images/train_subset_of_classes', help='Path to output directory.')
	# parser.add_argument('-v','--verbose', dest='verbose', action='count', default=0, help='Enable verbose output.')

	# parse the arguments, throw errors if missing any
	args = parser.parse_args()

	# assign to normal variables for convenience
	input_path = args.input_path
	output_path = args.output_path

	########################################################
	# hard coded lists...
	other_classes = ['dogsled', 'schooner', 'bottlecap']

	imagenet_dog_classes = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier', 'appenzeller', 'australian_terrier', 'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog', 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick', 'border_collie', 'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel', 'collie', 'coyote', 'curly-coated_retriever', 'dalmatian', 'dandie_dinmont', 'dhole', 'dingo', 'doberman', 'english_foxhound', 'english_setter', 'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog', 'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane', 'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog', 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog', 'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki', 'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier', 'timber_wolf', 'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier', 'whippet', 'white_wolf', 'wire-haired_fox_terrier', 'yorkshire_terrier']

	subset_of_classes = other_classes + imagenet_dog_classes

	########################################################
	# actually run

	dir_names = [dir_name for dir_name in subset_of_classes if os.path.isdir(os.path.join(input_path, dir_name))]
	if len(dir_names) != len(subset_of_classes):
		print('Missing some of the source subset_of_classes dirs:')
		print(list(set(subset_of_classes)-set(dir_names)))

	os.makedirs(output_path, exist_ok=True)

	for dir_name in tqdm(dir_names, desc='Class Dirs'):
		recursive_overwrite(src=os.path.join(input_path, dir_name), dest=os.path.join(output_path, dir_name))
