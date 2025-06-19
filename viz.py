import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from pygifsicle import optimize

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from umap.umap_ import UMAP

from typing import Union, Optional, Any
# import os
from pathlib import Path
import re


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

    if not save_path.exists():
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

def plot_transformed_embeddings(
        transformed_embeddings: np.ndarray,
        labels: Union[torch.Tensor, np.ndarray],
        n_components: int,
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
    assert transformed_embeddings.shape[0] == labels.shape[0], \
        "Number of transformed embeddings and labels should match"
    assert n_components in (2, 3), \
        "n_components should be either 2 or 3 for visualization"
    assert transformed_embeddings.shape[1] == n_components, \
        "transformed_embeddings should have shape (n_samples, n_components)"

    if type(labels) is torch.Tensor:
        labels = labels.cpu().numpy()

    transformed_embedding_axes = [transformed_embeddings[:,k] \
                                  for k in range(n_components)]

    fig, ax = plt.subplots(
        figsize=(8, 6),
        subplot_kw={'projection': '3d'} if n_components == 3 else {},
        **fig_kw,
    )
    ax.scatter(
        *transformed_embedding_axes,
        c=labels,
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

def _validate(
        embeddings: Union[torch.Tensor, np.ndarray],
        transform: Any,
        n_components: int = 2,
    ):
    assert embeddings.ndim == 2, \
        "embeddings should be a 2D tensor of shape (n_samples, n_features)"
    assert hasattr(transform, 'fit_transform'), \
        "transform should have a 'fit_transform' method"
    assert n_components in (2, 3), \
        "n_components should be either 2 or 3 for visualization"

def transform(
        embeddings: Union[torch.Tensor, np.ndarray],
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
        transformed_embeddings: A 2D ndarray of shape (n_samples, n_components).
        n_components: The number of components reduced to in the transformation.

    Raises:
        AssertionError: If `embeddings` is not 2D; if n_components is not 2 or 3;
            or if `transform` does not have a fit_transform method.
    '''
    _validate(embeddings, transform, n_components)
    if type(embeddings) is torch.Tensor:
        embeddings = embeddings.cpu().numpy()
    transformed_embeddings = transform.fit_transform(embeddings)
    return transformed_embeddings, n_components