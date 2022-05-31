# !/usr/bin/env python3

# Author: burcakotlu

# Contact: burcakotlu@eng.ucsd.edu

# source code for plot_occupancy_legend is copied from Combined_PCAWG_nonPCAWG_Occupancy_ReplicationTime_Processivity.py
# updated for immediate need.

from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
import os


def plot_occupancy_legend(output_dir):
    fig, ax = plt.subplots(figsize=(10, 1))

    legend_elements = [
        Line2D([0], [2], linestyle="-", lw=5, color='royalblue', label='Real Subs Average Signal',
               markerfacecolor='royalblue', markersize=30),
        # Line2D([0], [2], linestyle="-", lw=5, color='crimson', label='Real Dinucs Average Signal',
        #        markerfacecolor='crimson', markersize=30),
        Line2D([0], [2], linestyle="-", lw=5, color='darkgreen', label='Real Indels Average Signal',
               markerfacecolor='darkgreen', markersize=30),
        Line2D([0], [2], linestyle="--", lw=5, color='gray', label='Simulations Average Signal', markerfacecolor='gray',
               markersize=30)]

    plt.legend(handles=legend_elements, handlelength=5, ncol=1, loc="center", bbox_to_anchor=(0.5, 0.5), fontsize=30)
    plt.gca().set_axis_off()

    filename = 'Occupancy_Legend.png'
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=100, bbox_inches="tight")

    plt.cla()
    plt.close(fig)


output_dir = os.path.join('/Users/burcakotlu/Documents/AlexandrovLab/BurcakOtlu_Papers/Topography_of_Mutational_Processes_In_Human_Cancer/Figures_discreet_mode/Figure5_Epigenomics_Chromatin_Organization')
plot_occupancy_legend(output_dir)