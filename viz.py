import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imageio.v3 as iio
from pygifsicle import optimize

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from umap.umap_ import UMAP

from typing import Union, Optional, Any
# import os
from pathlib import Path
import re

import matplotlib.style as mplstyle
mplstyle.use('fast')  # TODO: correct place?


def read_dir_to_gif(
        dir_path: str,
        save_path: str,
        fps: int = 10,
        sort_by: Optional[str] = r'epoch=(\d+)',
        file_ext: str = '.png',
    ):
    '''
    Reads images from a directory and compiles these into a GIF.
    Images are sorted before compilation. `sort_by` expects a 
    regex pattern to extract a sortable value from image filenames.

    Args:
        dir_path: Directory containing image files.
        save_path: Path to save the GIF.
        fps: Frames per second for the GIF.
        sort_by: Regex pattern to sort images by, e.g. 'epoch=(\d+)'.
        file_ext: File extension of the images to include in the GIF.

    Raises:
        AssertionError: If the directory does not exist.
        ValueError: If no files with the specified extension are found in the directory.
    
    Example:
        read_dir_to_gif(
            dir_path='path/to/images',
            save_path='path/to/save/animation.gif',
            fps=10,
            sort_by=r'epoch=(\d+)',
            file_ext='.png'
        )
    '''
    dir_path = Path(dir_path)
    save_path = Path(save_path)

    assert dir_path.is_dir(), f"Directory {dir_path} does not exist"

    if not save_path.suffix == '.gif':
        save_path = save_path.with_suffix('.gif')

    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # if not os.path.exists(os.path.dirname(save_path)):
    #     os.makedirs(os.path.dirname(save_path))

    image_files = list(dir_path.glob(f'*{file_ext}'))
    if not image_files:
        raise ValueError(f"No files with extension {file_ext} found in {dir_path}")
    
    # image_files = [f for f in os.listdir(dir_path) if f.endswith(file_ext)]
    # if not image_files:
    #     raise ValueError(f"No files with extension {file_ext} found in {dir_path}")

    image_files.sort(key=lambda x: int(re.search(sort_by, x).group(1)))
    frames = []
    for image_file in image_files:
        image_path = dir_path / image_file
        frames.append(iio.imread(image_path))

    # image_files.sort(key=lambda x: int(re.search(sort_by, x).group(1)))
    # frames = []
    # for image_file in image_files:
    #     image_path = os.path.join(dir_path, image_file)
    #     frames.append(iio.imread(image_path))

    iio.imwrite(save_path, frames, extension=".gif", duration=1/fps, loop=0)
    optimize(save_path)

def plot_reduced_embeddings(
        reduced_embeddings: np.ndarray,
        labels: np.ndarray,
        n_components: int,
        cmap: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = False,
        fig_kw: dict = {},
        ax_kw: dict = {},
        ax_title: Optional[str] = None,
    ):
    '''
    '''
    assert labels.ndim == 1, \
        "labels should be a vector of shape (n_embeddings,)"
    assert reduced_embeddings.shape[0] == labels.shape[0], \
        "Number of embeddings and labels should match"
    assert n_components in (2, 3), \
        "n_components should be either 2 or 3 for visualization"
    assert reduced_embeddings.shape[1] == n_components, \
        "reduced_embeddings should have shape (n_samples, n_components)"

    reduced_embedding_axes = [reduced_embeddings[:,k] \
                                  for k in range(n_components)]

    fig, ax = plt.subplots(
        figsize=(8, 6),
        subplot_kw={'projection': '3d'} if n_components == 3 else {},
        **fig_kw,
    )
    ax.scatter(
        *reduced_embedding_axes,
        c=labels,
        cmap=cmap,
        **ax_kw,
    )
    ax.set_xlabel(f'component 1')
    ax.set_ylabel(f'component 2')
    if n_components == 3:
        ax.set_zlabel(f'component 3')
    if ax_title is not None:
        ax.set_title(ax_title)

    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)

    plt.close(fig)
    return fig

def reduce_embeddings(
        embeddings: np.ndarray,
        transform: Union[PCA, UMAP, TSNE, Isomap, Any],
        n_components: int = 2,
    ):
    '''
    Applies a dimensionality reduction transform to the embeddings.

    Args:
        embeddings: A 2D tensor or ndarray of shape (n_samples, n_features).
        transform: An instance of a dimensionality reduction class with a fit_transform method.
        n_components: Number of components to reduce to. Must be 2 or 3.

    Returns:
        reduced_embeddings: A 2D ndarray of shape (n_samples, n_components).
        n_components: The number of components reduced to in the transformation.

    Raises:
        AssertionError: If `embeddings` is not 2D; if n_components is not 2 or 3;
            or if `transform` does not have a fit_transform method.
    '''
    assert embeddings.ndim == 2, \
        "embeddings should be a 2D tensor of shape (n_samples, n_features)"
    assert hasattr(transform, 'fit_transform'), \
        "transform should have a 'fit_transform' method"
    assert n_components in (2, 3), \
        "n_components should be either 2 or 3 for visualization"
    
    embeddings = preprocessing.normalize(embeddings, axis=1)
    reduced_embeddings = transform.fit_transform(embeddings)
    return reduced_embeddings

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
        https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.
        https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts