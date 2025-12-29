"""
Docstring for sockeye_scripts.multi_signal_llm

This is a script form of content responsible for training CellUntangler and saving its embedding, found in `notebooks/multi_signal_celluntangler.py`
"""
import sys
import os

# Compute project root (go up one level)
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
# Add project root to path
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import scanpy as sc
import pandas as pd
import numpy as np
import torch

from src.data.umi_data import UMIVaeDataset
from src.celluntangler import utils
from src.celluntangler.models import Trainer
from src.celluntangler.models.nb_vae import NBVAE

# TODO - update all paths for remote
"""
CONFIGURATION CONSTANTS
"""
EPOCHS = 50
EPOCHS_SAVE_EVERY = 10
EPOCH_EMBEDDINGS_SAVE_PATH = '/scratch/st-jiaruid-1/jerome/experiments/multi_signal_12_28'

"""
READING DATA
"""

adata = sc.read_h5ad("../../../data/HGSOC/ALL_CELLS/all_cells.h5ad")

"""
PRE-PROCESSING
"""
# REQUIREMENT: Get unnormalized gene expressionr reads because CellUntangler requires gene-expression counts
adata = adata.raw.to_adata()

adata.var["gene_symbols"] = adata.var["feature_name"]
cell_cycle_genes_path = "../../../genes/celluntangler_human_cell_cycle_genes.tsv"
interferon_genes_path = "../../../genes/HGSOC/multi_signal/llm_11_2025/human_interferon_genes.tsv" 
dissociation_genes_path = "../../../genes/HGSOC/multi_signal/llm_11_2025/human_cell_dissociation_genes.tsv" 

cell_cycle_genes = pd.read_csv(cell_cycle_genes_path, header = None, sep="\t")
interferon_genes = pd.read_csv(interferon_genes_path, header = None, sep="\t") # PROGRESS!!!
dissociation_genes = pd.read_csv(dissociation_genes_path, header = None, sep = "\t")

cell_cycle_genes_set = set(cell_cycle_genes.iloc[:,0])
interferon_genes_set = set(interferon_genes.iloc[:,0]) 
dissociation_genes_set = set(dissociation_genes.iloc[:,0])

contained_genes_cc = adata.var["gene_symbols"].isin(cell_cycle_genes[0])
contained_genes_interferon = adata.var["gene_symbols"].isin(interferon_genes[0])
contained_genes_dissociation = adata.var["gene_symbols"].isin(dissociation_genes[0])

# Order Genes in adata such that genes appear in following order: cell cycle genes, interferon genes, dissociation genes
cc_indices = np.where(contained_genes_cc)[0]
interferon_indices = np.where(contained_genes_interferon)[0]
dissociation_indices = np.where(contained_genes_dissociation)[0]

all_known_indices = np.hstack((cc_indices, interferon_indices, dissociation_indices))
remaining_indices = np.setdiff1d(np.arange(adata.shape[1]), all_known_indices)

rearranged_indices = np.hstack((cc_indices, interferon_indices, dissociation_indices, remaining_indices))
rearranged_indices = rearranged_indices.astype(int)

adata = adata[:, rearranged_indices] # .copy() is safer in AnnData
adata.uns["new_gene_ordering"] = rearranged_indices # NB: this takes a while to run!

"""
ANNOTATIONS: calculate and add gene enrichment scores
"""
from anndata.utils import make_index_unique

# 1. Get the gene symbols and convert to a standard pandas Index
temp_index = pd.Index(adata.var['gene_symbols'].astype(str))

# 2. Use the anndata utility to make the index unique (appends -1, -2, etc. to duplicates)
adata.var_names = make_index_unique(temp_index)
adata.var_names.name = None
# TODO - removed `adata.raw = adata` - try running/testing on Jupyter notebook

sc.tl.score_genes(
    adata,
    gene_list=interferon_genes_set,
    score_name="interferon_score"
)
sc.tl.score_genes(
    adata,
    gene_list=dissociation_genes_set,
    score_name="dissociation_score"
)

"""
MODEL CONFIGURATION
"""
model_name = "r2, e2, e2, e10"

from src.celluntangler.models.get_config import get_config
config = get_config()
config.model_name = model_name
config.seed = 68715
config.init = "custom"
# NB: `epochs` and `max_epochs` based on what Sarah set for her interferon subspace experiment
config.max_epochs = EPOCHS 
config.epochs = EPOCHS

if config.seed:  
    torch.manual_seed(config.seed)
    np.random.seed(config.seed) 
    np.random.default_rng(config.seed)
components = utils.parse_components(model_name, config.fixed_curvature) 

"""
CREATING TRAINING DATASET
"""
if not "batch" in adata.obs.columns.tolist(): 
    adata.obs["batch"] = adata.obs.index.str.split("_").str[0]
    adata.obs["batch"] = adata.obs["batch"].astype("category")

batch = adata.obs.batch.cat.codes.values 
batch = batch[:,None]
batch = batch.astype(np.int64) 
# One entry in n_batch for each "factor of batch effects"
config.n_batch = [ len(np.unique(batch)) ]

batch_counts = np.bincount(batch[:,0])
batch_counts = batch_counts[1:]

# Creating Training Dataset
x = adata.X.todense().astype(np.double)

# y holds the batch vector for the dataset
y = batch 
in_dim = x.shape[1]
batch_size = config.batch_size
dataset = UMIVaeDataset(batch_size=batch_size, in_dim=in_dim)
# Create the dataset loaders
train_loader = dataset.create_loaders(x, y, seed=config.seed)

# Creating (Disjoint) Masks
mask_cyc = np.zeros(adata.n_vars)
mask_cyc[adata.var["gene_symbols"].isin(cell_cycle_genes[0])] = 1
mask_interferon = np.zeros(adata.n_vars)
mask_interferon[adata.var["gene_symbols"].isin(interferon_genes[0])] = 1
mask_dissociation = np.zeros(adata.n_vars) 
mask_dissociation[adata.var["gene_symbols"].isin(dissociation_genes[0])] = 1

mask_all = np.ones(adata.n_vars)
mask_all[adata.var["gene_symbols"].isin(cell_cycle_genes[0])] = 0 
mask_all[adata.var["gene_symbols"].isin(interferon_genes[0])] = 0 
mask_all[adata.var["gene_symbols"].isin(dissociation_genes[0])] = 0 

mask = torch.tensor([mask_cyc, mask_interferon, mask_dissociation, mask_all])

epoch_embeddings_save_path = EPOCH_EMBEDDINGS_SAVE_PATH
visualize_information={}
# The epochs to save the intermediate embeddings for
visualize_information["epochs"]=[i for i in range(0, EPOCHS, EPOCHS_SAVE_EVERY)] 
visualize_information["x"]=x
visualize_information["y"]=y
visualize_information["embeddings_save_path"]=epoch_embeddings_save_path
visualize_information["model_name"]=config.model_name
visualize_information["device"]=config.device

"""
MODEL CREATION
"""
use_gpu = True
if use_gpu:
  print("Using GPU")
  config.device = torch.device("cuda")
else:
  print("Using CPU")
  config.device = torch.device("cpu")

torch.set_default_dtype(torch.float64)

component_subspaces = {0 :[0], 1: [0,1], 2:[0,1,2], 3: [0,1,2,3]}
component_no_grads = {1: [0], 2: [0,1], 3: [0,1,2]}
component_batch =  {0: True, 1: True, 2: True, 3: True }

model = NBVAE(h_dim=config.h_dim,
              components=components,
              mask=mask,
              dataset=dataset,
              config=config,
              component_subspaces=component_subspaces,
              component_no_grads=component_no_grads,
              component_batch=component_batch).to(config.device)

"""
TRAINING
"""
trainer = Trainer(model)

optimizer = trainer.build_optimizer(learning_rate=config.learning_rate,
                                        fixed_curvature=config.fixed_curvature,
                                        use_adamw=config.use_adamw,
                                        weight_decay=config.weight_decay)

betas = utils.linear_betas(config.start,
                           config.end,
                           end_epoch=config.end_epoch,
                           epochs=config.epochs)

trainer.train_epochs(optimizer=optimizer,
                       train_data=train_loader,
                       betas=betas,
                       likelihood_n=0,
                       max_epochs=config.max_epochs,
                       visualize_information=visualize_information)

"""
SAVING EMBEDDINGS
"""
embeddings_save_path = epoch_embeddings_save_path
a = trainer.model(torch.log1p(torch.tensor(x, device=config.device)), torch.tensor(y, device=config.device))
np.savetxt(os.path.join(embeddings_save_path, f'{model_name}_all_encode_v63_z_params.txt'), a[4].detach().to(torch.device("cpu")).numpy())