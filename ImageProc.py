#!/usr/bin/env python3
"""
tsp_portrait.py

Produce TSP art from a portrait image.

Dependencies:
    pip install pillow numpy scipy matplotlib

Example:
    python tsp_portrait.py --input portrait.jpg --points 3000 --out out.svg
"""

import argparse
import math
import random
import sys
from time import time

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev


# -------------------------
# Utilities: importance map
# -------------------------
def compute_importance_map(img_gray: np.ndarray, edge_weight=1.0, darkness_weight=1.0, blur_sigma=0.0, brightness_cutoff=0.0):
    """
    img_gray: 2D numpy array with values 0..255 (uint8)
    brightness_cutoff: fraction (0..1). If brightness > cutoff (i.e. too bright), no points will be sampled there.
    """
    im = img_gray.astype(np.float64) / 255.0  # brightness 0..1
    darkness = 1.0 - im  # dark=1, bright=0
    
    # apply cutoff: zero out all areas that are too bright
    mask = darkness >= brightness_cutoff
    darkness = np.where(mask, darkness, 0.0)
    
    # apply non-linear emphasis
    darkness = darkness ** darkness_weight
    
    # Sobel edges
    sx = ndimage.sobel(im, axis=1, mode='reflect')
    sy = ndimage.sobel(im, axis=0, mode='reflect')
    edges = np.hypot(sx, sy)
    if edges.max() > 0:
        edges /= edges.max()
    # also mask edges so bright areas don't count
    edges = np.where(mask, edges, 0.0)

    combined = darkness_weight * darkness + edge_weight * edges

    if blur_sigma > 0:
        combined = ndimage.gaussian_filter(combined, sigma=blur_sigma)

    combined += 1e-12
    combined = combined / combined.sum()
    return combined

# -------------------------
# Sampling
# -------------------------
def sample_points_from_importance(importance_map, n_points, jitter=True, rng=None):
    """
    importance_map: normalized 2D array summing to 1
    returns Nx2 array of coordinates in image space (x, y) where x in [0, W), y in [0, H)
    """
    if rng is None:
        rng = np.random.default_rng()
    H, W = importance_map.shape
    flat = importance_map.ravel()
    # choose pixel indices with probability
    idx = rng.choice(flat.size, size=n_points, replace=True, p=flat)
    ys, xs = np.divmod(idx, W)
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    if jitter:
        # jitter uniformly inside pixel
        xs += rng.random(n_points)
        ys += rng.random(n_points)
    pts = np.column_stack([xs, ys])
    return pts

# -------------------------
# TSP heuristics: NN + 2-opt
# -------------------------
def nearest_neighbor_order(points, start_index=0):
    """Return an order using nearest neighbor heuristic (fast via KD-tree)."""
    n = len(points)
    tree = cKDTree(points)
    print(tree.data.shape)
    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=int)
    cur = start_index
    for i in range(n):
        order[i] = cur
        visited[cur] = True
        if i == n - 1:
            break
        # query for next nearest that is not visited
        # we progressively ask for k nearest
        k = 1
        found = False
        while not found:
            dists, idxs = tree.query(points[cur], k=k)
            if np.isscalar(idxs):
                idxs = [idxs]
            for cand in np.atleast_1d(idxs):
                if not visited[cand]:
                    cur = int(cand)
                    found = True
                    break
            k *= 2
            if k > n:
                # fallback: linear scan
                unvisited = np.where(~visited)[0]
                # pick the closest
                dists = np.sum((points[unvisited] - points[cur]) ** 2, axis=1)
                cur = int(unvisited[np.argmin(dists)])
                found = True
    return order

def tour_length(points, order):
    p = points[order]
    diffs = p[np.roll(np.arange(len(p)), -1)] - p
    return np.hypot(diffs[:,0], diffs[:,1]).sum()

def two_opt(points, order, max_iters=50000, improvement_threshold=1e-9):
    """
    Classic 2-opt. Returns improved order.
    Uses index-based operations on numpy arrays.
    """
    n = len(order)
    if n <= 2:
        return order
    pts = points
    best = order.copy()
    best_len = tour_length(pts, best)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # iterate i,j
        for i in range(1, n - 2):
            a = best[i - 1]
            b = best[i]
            pa = pts[a]
            pb = pts[b]
            for j in range(i + 1, n - 1):
                c = best[j]
                d = best[j + 1]
                pc = pts[c]
                pd = pts[d]
                # compute gain from swapping (i..j)
                old = np.hypot(*(pa - pb)) + np.hypot(*(pc - pd))
                new = np.hypot(*(pa - pc)) + np.hypot(*(pb - pd))
                if new + improvement_threshold < old:
                    # perform 2-opt: reverse segment i..j
                    best[i:j+1] = best[i:j+1][::-1]
                    best_len = best_len - (old - new)
                    improved = True
                    break
            if improved:
                break
    return best

# -------------------------
# Drawing & saving
# -------------------------
def draw_tsp(points, order, img_shape=None, linewidth=0.3, figsize=(10,10), dpi=300, save_png=None, save_svg=None, background='white'):
    """
    points: Nx2
    order: permutation array
    img_shape: (H, W) to set axis limits
    """
    p = points[order]
    # append the first to close loop visually (optional). For TSP-art, many prefer not to close; we won't close.
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_facecolor(background)
    ax.plot(p[:,0], p[:,1], linewidth=linewidth, solid_capstyle='round')
    ax.set_aspect('equal')
    ax.invert_yaxis()  # image coordinates: y goes down
    ax.axis('off')
    # set limits
    if img_shape is not None:
        H, W = img_shape
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
    else:
        ax.axis('tight')

    if save_png:
        plt.savefig(save_png, bbox_inches='tight', pad_inches=0, dpi=dpi)
    if save_svg:
        plt.savefig(save_svg, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

from matplotlib.path import Path
import matplotlib.patches as patches

def draw_tsp_smooth(points, order, img_shape=None, linewidth=0.3,
                    figsize=(10,10), dpi=300, save_png=None, save_svg=None, background='white', smoothing=0.01):
    p = points[order]
    x = p[:,0]
    y = p[:,1]

    # Parametrize by cumulative distance
    dist = np.cumsum(np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2))
    dist /= dist[-1]  # normalize 0..1

    # B-spline parametrization
    tck, u = splprep([x, y], u=dist, s=smoothing*len(points))  # s controls smoothness
    u_new = np.linspace(0, 1, len(points)*10)
    x_new, y_new = splev(u_new, tck)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_facecolor(background)
    ax.plot(x_new, y_new, linewidth=linewidth, solid_capstyle='round', color='black')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')

    if img_shape is not None:
        H, W = img_shape
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

    if save_png:
        plt.savefig(save_png, bbox_inches='tight', pad_inches=0, dpi=dpi)
    if save_svg:
        plt.savefig(save_svg, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# -------------------------
# Main pipeline
# -------------------------
def run_pipeline(input_path, n_points=3000, random_seed=None,
                 darkness_weight=1.0, edge_weight=1.5, blur_sigma=1.0,
                 do_2opt=True, twoopt_iters=20000,
                 out_svg='out.svg', out_png='out.png', linewidth=0.25):
    rng = np.random.default_rng(random_seed)
    img = Image.open(input_path).convert('RGB')
    W, H = img.size
    print(f"Loaded image {input_path} (W={W}, H={H})")
    # convert to grayscale
    img_gray = np.asarray(img.convert('L'))
    corner_vals = [
        img_gray[0, 0],          # top-left
        img_gray[0, -1]          # top-right
    ]
    bg_brightness = np.mean(img_gray) / 255.0
    mask = (img_gray.astype(np.float64) / 255.0) < (bg_brightness - 0.4)

    # compute importance
    imp = compute_importance_map(img_gray, edge_weight=edge_weight, darkness_weight=darkness_weight, blur_sigma=blur_sigma)
    imp *= mask
    imp /= imp.sum()
    print("Computed importance map.")
    plt.imshow(mask)
    plt.title("Mask")
    plt.show()
    plt.imshow(imp)
    plt.title("Importance Map")
    plt.show()
    # sample points
    pts = sample_points_from_importance(imp, n_points, jitter=True, rng=rng)
    print(f"Sampled {n_points} points.")

    # remove duplicates (rare) and do a small random shuffle for variety
    # ensure dtype for KD-tree usage
    # Solve TSP
    t0 = time()
    start_index = 0
    nn_order = nearest_neighbor_order(pts, start_index=start_index)
    nn_len = tour_length(pts, nn_order)
    print(f"Nearest-neighbor tour length: {nn_len:.2f} (time {time()-t0:.2f}s)")

    if do_2opt:
        t1 = time()
        improved = two_opt(pts, nn_order, max_iters=twoopt_iters)
        improved_len = tour_length(pts, improved)
        print(f"2-opt improved: {nn_len:.2f} -> {improved_len:.2f} (time {time()-t1:.2f}s)")
        order = improved
    else:
        order = nn_order

    # draw and save
    print(f"Rendering and saving to {out_svg} and {out_png} ...")
    #draw_tsp(pts, order, img_shape=(H, W), linewidth=linewidth, save_png=out_png, save_svg=out_svg)
    draw_tsp_smooth(pts, order, img_shape=(H, W), linewidth=linewidth, save_png=out_png, save_svg=out_svg, smoothing=0.03)
    print("Done.")

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Create TSP art from a portrait.")
    p.add_argument("--input", "-i", required=True, help="Input image file (portrait).")
    p.add_argument("--points", "-p", type=int, default=3000, help="Number of sample points (try 1000..10000).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--darkness-weight", type=float, default=10.0)
    p.add_argument("--edge-weight", type=float, default=1.5)
    p.add_argument("--blur-sigma", type=float, default=1.0)
    p.add_argument("--no-2opt", action='store_true', help="Disable 2-opt step (faster).")
    p.add_argument("--out-svg", default="out.svg")
    p.add_argument("--out-png", default="out.png")
    p.add_argument("--linewidth", type=float, default=2)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.input, n_points=args.points, random_seed=args.seed,
                 darkness_weight=args.darkness_weight, edge_weight=args.edge_weight, blur_sigma=args.blur_sigma,
                 do_2opt=not args.no_2opt, twoopt_iters=200,
                 out_svg=args.out_svg, out_png=args.out_png, linewidth=args.linewidth)
