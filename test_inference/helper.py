import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

import requests
from PIL import Image


def plot_attn_map(attn, stride=1):
    attn = torch.mean(attn, dim=0)  # Shape: (n_tokens, n_tokens)
    attn = torch.nn.functional.avg_pool2d(attn.unsqueeze(0).unsqueeze(0), stride, stride).squeeze(0).squeeze(0)
    attn = attn.cpu().detach().numpy()  

    cmap = plt.cm.get_cmap("viridis")
    plt.figure(figsize=(5, 5))
    log_norm = LogNorm(vmin=attn[-1].min(), vmax=attn[-1].max())
    ax = sns.heatmap(attn, cmap=cmap, norm=log_norm)
    plt.savefig('VLM_attention_map.pdf', format='pdf')
    plt.show()


def plot_heatmap(tensor,file_name):
    """
    Plot a heatmap of the input tensor and save it to a file if file_name is provided.
    """
    if tensor.dim() != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError("Input tensor must be of shape [n, n]")
    
    tensor = tensor.cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(tensor, cmap='viridis', cbar=True)
    plt.title("Heatmap of Tensor")
    plt.xlabel("Index")
    plt.ylabel("Index")
    
    if file_name:
        plt.savefig(file_name)
    else:
        plt.show()

def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    This is done by calculating the effective and wasted resolution for each possible resolution.

    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).
        possible_resolutions (list):
            A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

    Returns:
        tuple: The best fit resolution in the format (height, width).
    """
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


def get_random_image(h=400,w=400,show=False):
    url = f"https://picsum.photos/{h}/{w}"
    response = requests.get(url)
    if response.status_code == 200:
        with open('image.jpg', 'wb') as file:
            file.write(response.content)
    else:
        print(f"Failed to retrieve image. HTTP Status code: {response.status_code}")
    raw_image = Image.open('image.jpg')
    if show:
        plt.imshow(raw_image)
        plt.show()
    return raw_image

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_topk_attention_map_in_nxn_grid(attention_scores, topk=100):
    """
    Plot the attention map in a n x n grid
    for topk tokens and rest are zeros
    """
    num_tokens = attention_scores.shape[0]
    num_token_grid = int(np.sqrt(num_tokens))
    
    # Flatten the attention scores
    flattened_scores = attention_scores.flatten()
    
    # Get the indices of the topk values
    topk_indices = np.argpartition(flattened_scores, -topk)[-topk:]
    
    # Create a mask to zero out other values
    binary_mask = np.zeros_like(flattened_scores)
    binary_mask[topk_indices] = 1
    
    # Reshape the binary mask back to the grid shape
    binary_mask = binary_mask.reshape((num_token_grid, num_token_grid))
    
    # Plotting the attention heatmap
    plt.figure(figsize=(5, 5))
    sns.heatmap(binary_mask, annot=False, cmap='viridis', cbar=False)
    plt.show()

def plot_attentin_map_in_nxn_grid(attention_scores,topk=100):
    """
    Plot the attention map in a n x n grid
    for topk tokens and rest are zeros
    """
    num_tokens = attention_scores.shape[0]
    num_token_grid = int(np.sqrt(num_tokens))

    # plot the attention map in a n x n grid
    attention_scores = attention_scores.reshape((num_token_grid, num_token_grid))
    
    # Plotting the attention heatmap
    plt.figure(figsize=(5, 5))
    # sns.heatmap(attention_scores, annot=False, cmap='viridis', norm=LogNorm(vmin=attention_scores.min(), vmax=attention_scores.max()),cbar=False)
    sns.heatmap(attention_scores, annot=False, cmap='viridis',cbar=False)
    plt.show()


def plot_attentin_map_in_nxn_grid(attention_scores):
    """
    Plot the attention map in a n x n grid
    assume num_token_patches is a perfect square
    attention_scores: torch tensor [num_token_patches]
    """
    num_tokens = attention_scores.shape[0]
    num_token_grid = int(np.sqrt(num_tokens))

    # plot the attention map in a n x n grid
    attention_scores = attention_scores.reshape((num_token_grid, num_token_grid))
    
    # Plotting the attention heatmap
    plt.figure(figsize=(5, 5))
    # sns.heatmap(attention_scores, annot=False, cmap='viridis', norm=LogNorm(vmin=attention_scores.min(), vmax=attention_scores.max()),cbar=False)
    sns.heatmap(attention_scores, annot=False, cmap='viridis',cbar=False)
    plt.show()

def count_top_npercents_contribution(vit_attention, topk=100):
    """
    Count the contribution of topk tokens to the cls token in the attention map.
    vit_attention: tuple of torch tensor [1, num_heads, n_all_tokens, n_all_tokens]
    topk: int top k tokens to consider
    """
    num_layers = len(vit_attention)
    n_all_tokens = vit_attention[0].shape[-1]
    
    for layer in range(num_layers):
        attn = vit_attention[layer][0] # [num_heads, n_all_tokens, n_all_tokens]
        attn = attn.sum(dim=0) # [n_all_tokens, n_all_tokens]
        cls_attn = attn[0, 1:] # [n_all_tokens]
        
        cls_attn_sorted, _ = torch.sort(cls_attn, descending=True)
        # sum of topk tokens
        cls_attn_topk_sum = cls_attn_sorted[:topk].sum()
        # sum of all tokens
        cls_attn_sum = cls_attn.sum()

        # attein by topk
        print(f"Layer {layer}: {cls_attn_topk_sum/cls_attn_sum*100:.2f}%")