# import os
# import argparse
import json

import matplotlib.pyplot as plt
# import numpy as np 

import torch
# import torch.nn as nn
# import torch.nn.functional as F

from models.glow_model_one_channel.utils.glow import Glow as Glow_uncond
from models.glow_model_one_channel.utils.glow_cond import Glow as Glow_cond
from utilities.routes import  OUTPUT_DIR

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_samples(samples, ds, chkp, samples_per_row=5):	
	# Plotting function
	n_rows = len(samples) // samples_per_row
	n_tot = int(n_rows*samples_per_row)
	samples = samples[:n_tot]
	fig = plt.figure(figsize=(2*samples_per_row, 2*n_rows))
	for i, out in enumerate(samples):
		a = fig.add_subplot(n_rows, samples_per_row, i+1)
		out_sh = out.permute(1,2,0)
		plt.imshow(out_sh, cmap='gray')
		a.axis("off")
	plt.savefig(f'./images/samples/{ds}_samples_checkpoint_{chkp}.png')
	print(f'Samples saved in ./images/samples/{ds}_samples_checkpoint_{chkp}.png')
	plt.show()

def generate(args):
	data_shape=(1,28,28)
	num_labels = 10
	model_chkp = f'{OUTPUT_DIR}/glow_{args.dataset}/glow_checkpoint_{args.checkpoint}.pt'
	with open(f'{OUTPUT_DIR}/glow_{args.dataset}/' + 'hparams.json', encoding='utf-8') as json_file:
		hparams = json.load(json_file)
	if hparams['conditional'] is True:
		model = Glow_cond(data_shape, hparams['hidden_channels'], hparams['num_flow_steps'], hparams['ACL_layers'], hparams['num_levels'], hparams['num_splits'], num_labels, device, hparams['chunk_size'])
		model.load_state_dict(torch.load(model_chkp), strict=False)
		model.eval()
		samples = model.sample(torch.tensor([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4]).to(device), hparams['num_samples']).cpu().detach()
	else:
		model = Glow_uncond(data_shape, hparams['hidden_channels'], hparams['num_flow_steps'], hparams['ACL_layers'], hparams['num_levels'], hparams['num_splits'], device, hparams['chunk_size'])
		model.load_state_dict(torch.load(model_chkp))
		model.eval()
		samples = model.sample(args.num_samples).cpu().detach()

	plot_samples(samples, args.dataset, args.checkpoint)

