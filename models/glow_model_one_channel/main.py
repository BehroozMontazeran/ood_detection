import argparse
# import warnings
from os import path, makedirs
import json
from models.glow_model_one_channel.train import train
from models.glow_model_one_channel.generate import generate
from utilities.routes import  DATAROOT, OUTPUT_DIR
from utilities.utils import dataset_names


def parser():
	parser = argparse.ArgumentParser(description="Glow Normalizing Flow model Parameters")
	subparsers = parser.add_subparsers(help="The mode of the program", dest="mode")
	parser_train = subparsers.add_parser("train", help="Train the Glow model")
	parser_generate = subparsers.add_parser("generate", help="Use trained Glow model to generate samples")
	
	parser_train.add_argument('--dataset', type=str, choices=dataset_names, default='mnist')
	parser_train.add_argument('--batch_size', type=int, default=128)
	parser_train.add_argument('--conditional', type=bool, default=False)
	parser_train.add_argument('--lr', type=float, default=1e-3)
	parser_train.add_argument('--weight_decay', type=float, default=1e-4)
	parser_train.add_argument('--epochs', type=int, default=250)
	parser_train.add_argument('--hidden_channels', type=int, default=1000)
	parser_train.add_argument('--num_flow_steps', type=int, default=10)
	parser_train.add_argument('--ACL_layers', type=int, default=10)
	parser_train.add_argument('--num_levels', type=int, default=1)
	parser_train.add_argument('--num_splits', type=int, default=2)
	parser_train.add_argument('--chunk_size', type=int, default=1)
	parser_train.add_argument('--dataroot', type=str, default=DATAROOT)
	parser_train.add_argument('--download', type=bool, default=True)
	parser_train.add_argument('--transform', type=bool, default=None)
	parser_train.add_argument('--augment', type=bool, default=False)
	parser_train.add_argument('--n_workers', type=int, default=6)
	parser_train.add_argument('--output_dir', type=str, default=OUTPUT_DIR)



	parser_generate.add_argument('--dataset', type=str, choices=dataset_names, default='mnist')
	parser_generate.add_argument('--checkpoint', type=int, default=10)
	# parser_generate.add_argument('--conditional', type=bool, default=False)
	parser_generate.add_argument("--num_samples", type=int, default=15, help="Number of samples to be generated")
	# parser_generate.add_argument('--hidden_channels', type=int, default=1000)
	# parser_generate.add_argument('--num_flow_steps', type=int, default=10)
	# parser_generate.add_argument('--ACL_layers', type=int, default=10)
	# parser_generate.add_argument('--num_levels', type=int, default=1)
	# parser_generate.add_argument('--num_splits', type=int, default=2)
	# parser_generate.add_argument('--chunk_size', type=int, default=1)
	# parser_train.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
	return parser.parse_args()

def main():
	args = parser()
	if (args.mode=='train'):
		if not path.isdir(f'{args.output_dir}/glow_{args.dataset}'):
			makedirs(f'{args.output_dir}/glow_{args.dataset}')
		with open(path.join(f'{args.output_dir}/glow_{args.dataset}', "hparams.json"), "w", encoding='utf-8') as fp:
			json.dump(vars(args), fp, sort_keys=True, indent=4)
		train(args)
	elif (args.mode=='generate'):
		generate(args)

if __name__ == "__main__":
	main()