import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from river.tree.nodes.branch import NumericBinaryBranch

class Centroid:
    """Class that stores a centroid's attributes."""
    def __init__(self, centre=None, class_label=None, std_dev=None):
        self.centre = centre
        self.class_label = class_label
        self.std_dev = std_dev


def calculate_centroids(X, y, n_clusters_per_class, reference_centroids=None, min_cluster_distance=0.5):
    centroids = []
    for label in sorted(y.unique()):
        instances = X[y == label]
        if len(instances) < n_clusters_per_class:
            class_centroids = [instances.mean().to_numpy()]
        else:
            kmeans = KMeans(n_clusters=n_clusters_per_class, random_state=42, n_init=10)
            kmeans.fit(instances)
            class_centroids = list(kmeans.cluster_centers_)
            if len(class_centroids) > 1:
                silhouette_avg = silhouette_score(instances, kmeans.labels_)
                distances = np.linalg.norm(
                    np.array(class_centroids)[:, None] - np.array(class_centroids), axis=-1
                )
                np.fill_diagonal(distances, np.inf)
                if silhouette_avg < 0.2 or distances.min() < min_cluster_distance:
                    class_centroids = [instances.mean().to_numpy()]
        for centre in class_centroids:
            centroids.append((label, centre))

    if reference_centroids and len(centroids) == len(reference_centroids):
        ref_coords = [(rc.centre if isinstance(rc, Centroid) else rc) for rc in reference_centroids]
        ordered, remaining = [], centroids.copy()
        for rc in ref_coords:
            idx = min(range(len(remaining)), key=lambda i: np.linalg.norm(remaining[i][1] - rc))
            ordered.append(remaining.pop(idx))
        centroids = ordered + remaining

    return [Centroid(centre, label) for label, centre in centroids]


def traverse_tree(node):
    boundaries = []
    if hasattr(node, 'feature'):
        boundaries.append((node.feature, node.threshold))
        if hasattr(node, 'children') and isinstance(node.children, list):
            for child in node.children:
                boundaries.extend(traverse_tree(child))
    return boundaries

def format_dataframe(df):
    DIM = df.shape[1] - 1 
    dim_columns = [f"dim_{i}" for i in range(DIM)]
    column_names = dim_columns + ["class"]
    df.columns = column_names
    df.reset_index(drop=True, inplace=True)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    return X,y,df











#### Obsolete functions, kept for future use ####
def plot_frames_continuous(df, sub_window_size, models, output_path, n_dimensions, font_size=16):
    print("[plot_utils] Plotting frames...")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    dims = [f'dim_{i}' for i in range(n_dimensions)]
    classes = df['class'].unique()
    num_frames = len(df) // sub_window_size

    # color map
    if len(classes) == 2:
        red = mcolors.to_rgba('#FF383B')
        blue = mcolors.to_rgba('#3399FF')
        cmap = {classes[0]: red, classes[1]: blue}
    else:
        cmap_vals = plt.cm.get_cmap('Set1', len(classes))
        cmap = {cls: cmap_vals(i) for i, cls in enumerate(classes)}

    for i in range(num_frames):
        start, end = i*sub_window_size, (i+1)*sub_window_size
        sub = df.iloc[start:end]
        fig, ax = plt.subplots(figsize=(6, 6))
        # plot scatter
        for cls in classes:
            pts = sub[sub['class'] == cls]
            if not pts.empty:
                ax.scatter(pts[dims[0]], pts[dims[1]],
                           color=cmap[cls], alpha=0.8, edgecolor='k', linewidth=0.5)
        # decision boundary
        if i>0:
            xx, yy = np.meshgrid(
                np.linspace(sub[dims[0]].min(), sub[dims[0]].max(), 200),
                np.linspace(sub[dims[1]].min(), sub[dims[1]].max(), 200)
            )
            grid = np.c_[xx.ravel(), yy.ravel()]
            preds = np.array([
                models[i-1].predict_one({dims[0]: x, dims[1]: y}) for x,y in grid
            ]).reshape(xx.shape)
            classes_ = np.unique(preds)
            cmap_list = [cmap[c] for c in classes_]
            lc = ListedColormap(cmap_list)
            ax.imshow(preds, extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                      origin='lower', cmap=lc, alpha=0.35)
        ax.set_xlabel(dims[0], fontsize=font_size)
        ax.set_ylabel(dims[1], fontsize=font_size)
        ax.tick_params(labelsize=font_size)
        frame_file = os.path.join(output_path, f"frame_{i:03d}.pdf")
        fig.savefig(frame_file, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    print("[plot_utils] Frames saved to", output_path)


def plot_frames(df, window_size, models, decision_boundaries, output_path, n_dimensions=2, font_size=16):
    plot_frames_continuous(df, window_size, models, output_path, n_dimensions, font_size)


def save_tree_plot(model, output_path, filename="tree", fmt="pdf"):
    dot = model.draw()
    os.makedirs(output_path, exist_ok=True)
    dot.render(filename=filename, directory=output_path, format=fmt, cleanup=True, quiet=True)


def traverse_tree_partial(root, restrictions=None):
    if restrictions is None:
        restrictions = []
    boundaries = []
    if isinstance(root, NumericBinaryBranch):
        boundaries.append((root.feature, root.threshold, list(restrictions)))
        left_restr = restrictions + [(root.feature, '<=', root.threshold)]
        boundaries += traverse_tree_partial(root.children[0], left_restr)
        right_restr = restrictions + [(root.feature, '>', root.threshold)]
        boundaries += traverse_tree_partial(root.children[1], right_restr)
    return boundaries


def restrictions_to_box(restrictions):
    min_x = max_x = min_y = max_y = None
    for feat, op, thr in restrictions:
        idx = int(feat.split('_')[1])
        if idx == 0:
            if op == '<=': max_x = thr if max_x is None or thr < max_x else max_x
            else: min_x = thr if min_x is None or thr > min_x else min_x
        else:
            if op == '<=': max_y = thr if max_y is None or thr < max_y else max_y
            else: min_y = thr if min_y is None or thr > min_y else min_y
    return (min_x, max_x, min_y, max_y)


def plot_partial_boundary(ax, feature, threshold, box):
    min_x, max_x, min_y, max_y = box
    idx = int(feature.split('_')[1])
    if idx == 0:
        ax.vlines(threshold, min_y or ax.get_ylim()[0], max_y or ax.get_ylim()[1],
                  linestyles='--', linewidth=2, color='red')
    else:
        ax.hlines(threshold, min_x or ax.get_xlim()[0], max_x or ax.get_xlim()[1],
                  linestyles='--', linewidth=2, color='red')
