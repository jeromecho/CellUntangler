import geomstats.visualization as visualization
import math as math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os as os
import scanpy as sc
import pandas as pd

from matplotlib.patches import Circle
from scipy.signal import savgol_filter

from .helpers import lorentz_to_poincare


tableau_colors = list(mcolors.TABLEAU_COLORS.keys())
COLOR_NAMES = tableau_colors + list(mcolors.CSS4_COLORS.keys())

# Added: custom UMAP function for visualizing higher dimensional Euclidean space 
def visualize_embedding(
    embedding,              # numpy array (N x 2)
    obs_values,             # pandas Series from adata.obs
    cmap=plt.cm.viridis,    # colormap for continuous or default
    cat_colors=None,        # optional dict for categorical colors
    grid_lines=False,       # optional
    c_bar_label=None,
    bbox_to_anchor=(1.05, 1.0),
    point_size=8,
    alpha=0.9,
    title = "",
    figsize=(7,7)
):
    """
    Generic scatter plot for 2D embeddings
    Supports both continuous and categorical obs values.
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Detect if categorical or continuous
    is_categorical = pd.api.types.is_categorical_dtype(obs_values)

    # ---------------------
    # Categorical plotting
    # ---------------------
    if is_categorical:
        categories = obs_values.cat.categories

        # Use user-provided categorical colors or generate automatically
        if cat_colors is None:
            # map categories to colors using the given cmap
            cat_codes = np.arange(len(categories))
            colors = cmap(cat_codes / (len(categories)-1 if len(categories) > 1 else 1))
            cat_colors = dict(zip(categories, colors))

        for cat in categories:
            mask = obs_values == cat
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                label=str(cat),
                s=point_size,
                alpha=alpha,
                color=cat_colors[cat]
            )

        # Legend
        ax.legend(
            title=obs_values.name,
            bbox_to_anchor=bbox_to_anchor,
            loc="upper left"
        )

    # ---------------------
    # Continuous plotting
    # ---------------------
    else:
        sc = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=obs_values.values,
            cmap=cmap,
            s=point_size,
            alpha=alpha
        )
        cbar = plt.colorbar(sc, ax=ax)
        if c_bar_label:
            cbar.set_label(c_bar_label)

    # Optional grid lines
    if grid_lines:
        ax.grid(True, color='lightgray', linestyle='--', linewidth=0.5)

    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    title_ax = title if title != "" else f"Embedding colored by {obs_values.name}"
    ax.set_title(title_ax)

    plt.tight_layout()


def visualize_poincare_from_lorentz(embeddings,
                                    desired_obs_all,
                                    embedding_type="discrete",
                                    curvature=-1.0,
                                    cmap=plt.cm.viridis,
                                    grid_lines=False,
                                    origin=(0, 0),
                                    cat_colors=None,
                                    c_bar_label="",
                                    bbox_to_anchor=None,
                                    s=None):
  """
  Creates a plot of the visualization of the Poincare coordinates from the Lorentz coordinates.
  embeddings: The Lorentz embeddings.
  desired_obs_all: The obeservations to color the cells by.
  embedding_type: "discrete" if the observations are discrete and "continuous" otherwise.
  curvature: The curvature of the Lorentz mode.
  cmap: The colormap to use if the observations are continuous.
  grid_lines: True to show the values for the x-axis and y-axis. False otherwise.
  origin: The origin of the Poincare disc.
  cat_colors: A list of colors for each observation if the observations are discrete.
  c_bar_label: The label of the color bar if the observations are continuous.
  bbox_to_anchor: The coordinates of the legend.
  s: The size of the points.
  """
  poincare_coordinates = lorentz_to_poincare(embeddings, curvature)
  fig = visualize_poincare(poincare_coordinates, desired_obs_all, curvature, embedding_type, cmap, grid_lines, cat_colors=cat_colors, c_bar_label=c_bar_label, bbox_to_anchor=bbox_to_anchor, s=s)
  return fig

def visualize_poincare(poincare_coordinates,
                       desired_obs_all,
                       curvature=-1.0,
                       embedding_type="discrete",
                       cmap=plt.cm.viridis,
                       grid_lines=False,
                       origin=(0, 0),
                       cat_colors=None,
                       c_bar_label="",
                       bbox_to_anchor=None,
                       s=None):
  """
  Visualization of the Poincare coordinates colored by a desired observation.
  """
  x_p_1 = poincare_coordinates[:, 0]
  x_p_2 = poincare_coordinates[:, 1]
  circle = visualization.PoincareDisk(coords_type="ball")
  # print(round(curvature, 1))
  circle = Circle(origin, radius=1/math.sqrt(abs(curvature)), color='black', fill=False)
  # print(round(1/math.sqrt(abs(curvature)),1))
  # circle.set_origin((0.5, 0.4))
  fig, ax = plt.subplots(figsize=(8, 8))
  ax.axes.xaxis.set_visible(grid_lines)
  ax.axes.yaxis.set_visible(grid_lines)
  # ax.grid(grid_lines, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

  ax.set_xlim((-1/math.sqrt(abs(curvature)), 1/math.sqrt(abs(curvature))))
  ax.set_ylim((-1/math.sqrt(abs(curvature)), 1/math.sqrt(abs(curvature))))

  # circle.set_ax(ax)
  # circle.draw(ax=ax)
  ax.add_artist(circle)

  if embedding_type == "discrete":
    # categories = np.unique(desired_obs_all)
    categories = desired_obs_all.cat.categories
    print(categories)
    if cat_colors is None:
      cat_colors = COLOR_NAMES[0:len(categories)]
    for i, cat in enumerate(categories):
      cat_indices = desired_obs_all == cat
      if s:
        ax.scatter(x_p_1[cat_indices], x_p_2[cat_indices], label=cat, color=cat_colors[i], s=s)
      else:
        ax.scatter(x_p_1[cat_indices], x_p_2[cat_indices], label=cat, color=cat_colors[i])
  else:
    scatter = ax.scatter(x_p_1, x_p_2, c=desired_obs_all, cmap=cmap)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(c_bar_label)
  if bbox_to_anchor is not None:
    ax.legend(loc='upper right', bbox_to_anchor=bbox_to_anchor)
  else:
    ax.legend()
  # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
  # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
  return fig

def compute_umap(adata,
                  l_neighbors,
                  color,
                  embeddings_key,
                  n_pcs=None,
                  use_original_umap=False,
                  save_figure=False,
                  additional_save_information=[],
                  palette=None,
                  title=""):
  """
  l_neighbors: The number of neighbors to use when creating the neighborhood map.
  n_pcs: Dimension to use when computing neighborhood map.
  color: A list of colors to display the UMAP in.
  embeddings_key: The key to use to get the embeddings to use to compute the neighborhood map.
  use_original_umap: True to use the existing UMAP in the adata. False to compute it and use the computed UMAP.
  save_figure: True to save the figure. False otherwie.
  additional_save_information: A list of strings to include in the save name for the figure.
  palette: A palette specifying the color for the observition.
  title: A title for the figure.
  """
  if n_pcs is None:
      n_pcs = adata.obsm[embeddings_key].shape[1]
  if not use_original_umap:
    sc.pp.neighbors(adata, n_neighbors=l_neighbors, n_pcs=n_pcs, use_rep=embeddings_key)
    sc.tl.umap(adata)

  save = None
  if save_figure:
    information = [str(n_pcs),
                      embeddings_key,
                      str(l_neighbors),
                    "original" if use_original_umap else "computed"]+additional_save_information
    save = "_" + "_".join(information) + ",.png"
  
  sc.pl.umap(adata, color=color, save=save, palette=palette, title=title)


def plot_gene_change(ordered_adata, phase, gene_name, colors=None, labels=None, save_path=None, ccPhase_palette=None, layer=""):
    """
    Plots the gene expression for a single gene.
    ordered_adata: The data with the cells ordered by pseudotime.
    phase: The phase of the gene.
    gene_name: The name of the gene.
    colors: Colors for each cell.
    labels: The label of the phase for each cell.
    save_path: The path to save the plot to.
    ccPhase_palette: A dictionary where the keys are the cell cycle phase and the values are the color for the cell cycle phase.
    layer: The empty string to get the expression values from .X. Otherwise, a string that specifies which layer to use.
    """
    if layer:
        gene_expression = ordered_adata[:, gene_name].layers[layer].squeeze(-1)
    else:
        gene_expression = ordered_adata[:, gene_name].X.squeeze(-1)
    # print(gene_expression)
    if labels is not None:
        categories = labels.cat.categories

        for i, cat in enumerate(categories):
            cat_indices = labels.values == cat

            plt.vlines(np.where(cat_indices==True),
                       np.zeros(np.sum(cat_indices==True)),
                       gene_expression[cat_indices],
                       color=colors[cat_indices],
                       label=cat)
    else:
        plt.vlines(np.arange(0, len(ordered_adata)), np.zeros(len(ordered_adata)), gene_expression, colors=colors)

    if ccPhase_palette:
        color = ccPhase_palette[phase.replace("_", ".")]
    else:
        color = 'r'

    plt.plot(savgol_filter(gene_expression,70,2), color=color)

    plt.ylabel(f'${gene_name}$ expression')
    plt.xlabel('Ordering along pseudotime')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.21, 1))
    plt.tight_layout()
    # plt.savefig(os.path.join(save_path, f'savgol_{gene_name}_{phase}_ordered_x_mb_normalize_log_italicize.png'))
    plt.savefig(os.path.join(save_path, f'savgol_{gene_name}_{phase}_ordered_normalize_log_layer{layer}.png'))
    plt.show()
    plt.close()
    # sc.pp.pca(adata)
    # sc.pl.pca(adata, color=['CCNE2','ccPhase'])