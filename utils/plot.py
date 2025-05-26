import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import colors as mcolors


# NOTE:
# We recommend to set the limits of the axes to a precise range depending on the visualization goal. 

def plot_frames(df, sub_window_size, models, decision_boundaries=None, output_path=None, n_dimensions=2, font_size=16):
    print("[plot_utils] Plotting frames...")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    dims = [f'dim_{i}' for i in range(n_dimensions)]
    classes = df['class'].unique()
    num_frames = len(df) // sub_window_size

    if len(classes) == 2:
        cmap = {classes[0]: mcolors.to_rgba('#FF383B'), classes[1]: mcolors.to_rgba('#3399FF')}
    else:
        cmap_vals = plt.cm.get_cmap('Set1', len(classes))
        cmap = {cls: cmap_vals(i) for i, cls in enumerate(classes)}

    for i in range(num_frames):
        start, end = i * sub_window_size, (i + 1) * sub_window_size
        sub = df.iloc[start:end]

        fig, ax = plt.subplots(figsize=(6, 6))

        for cls in classes:
            pts = sub[sub['class'] == cls]
            if not pts.empty:
                ax.scatter(pts[dims[0]], pts[dims[1]],
                           color=cmap[cls], alpha=0.8, edgecolor='k', linewidth=0.5)

        if models is not None and i > 0 and (i - 1) < len(models):
            xx, yy = np.meshgrid(
                np.linspace(-6, 6, 200),
                np.linspace(-6, 6, 200)
            )
            grid = np.c_[xx.ravel(), yy.ravel()]
            preds = np.array([
                models[i - 1].predict_one({dims[0]: x, dims[1]: y}) for x, y in grid
            ]).reshape(xx.shape)
            classes_ = np.unique(preds)
            lc = ListedColormap([cmap[c] for c in classes_])
            ax.imshow(preds, extent=(-6, 6, -6, 6),
                      origin='lower', cmap=lc, alpha=0.35)


        # TO modify as needed
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)



        if decision_boundaries is not None and i < len(decision_boundaries):
            for feat, threshold in decision_boundaries[i]:
                idx = int(feat.split('_')[1])
                if idx == 0:
                    ax.axvline(threshold, linestyle='--', linewidth=1.5, color='red')
                elif idx == 1:
                    ax.axhline(threshold, linestyle='--', linewidth=1.5, color='red')

        ax.set_xlabel(dims[0], fontsize=font_size)
        ax.set_ylabel(dims[1], fontsize=font_size)
        ax.tick_params(labelsize=font_size)

        frame_file = os.path.join(output_path, f"frame_{i:03d}.pdf")
        fig.savefig(frame_file, bbox_inches='tight')
        plt.close(fig)

    print("[plot_utils] Frames saved to", output_path)

