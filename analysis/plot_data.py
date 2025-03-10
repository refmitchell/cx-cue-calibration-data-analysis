
"""
plot_data.py

Script responsible for producing change-in-bearing plots along with associated
statistics.
"""
import json
import numpy as np
import itertools
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import pandas as pd

# import dict_key_definitions as keydefs
# from dict_key_definitions import statskeys, simkeys
# from dict_key_definitions import modality_transfer_reskeys as reskeys

from test_utilities import circ_scatter, confidence_interval, angular_deviation, circmean, angular_difference, v_test


def produce_plot(filename):
    # Load stats data from file
    datasheet = pd.read_csv(filename)

    # Dacke et al. (2019) show the change in bearing between rolls 1 and 4
    # for each individual. Extract exit angles for each individual for roll 1.
    # Then extract exit angles for each individual for roll 4. Round to nearest
    # five degrees and compute the difference.

    # Exit angles, should be one entry for each agent (n = 40).
    # Each agent has a list of exits so these should be flattened.
    
    r1 = np.radians(datasheet["r1"])
    r2 = np.radians(datasheet["r2"])
    r3 = np.radians(datasheet["r3"])
    r4 = np.radians(datasheet["r4"])

    changes_dict = dict()
    changes_dict["r1r2"] = [angular_difference(f,l, signed=True) for (f,l) in zip(r1,r2)]
    changes_dict["r1r3"] = [angular_difference(f,l, signed=True) for (f,l) in zip(r1,r3)]
    changes_dict["r1r4"] = [angular_difference(f,l, signed=True) for (f,l) in zip(r1,r4)]

    for k in changes_dict.keys():
        # Rounding done for plotting and has no significant effect on stats.
        # FP errors mean that the stacked circ-scatter plots don't work unless you
        # remove some precision.
        changes_dict[k] = [float(int(x * 1000))/1000 for x in changes_dict[k]]

    # Load in Dacke et al. 2019 data
    dacke_changes = pd.read_csv(
        "csvs/mt_changes_dacke2019.csv"
    )["change"].to_numpy()

    dacke_changes = np.radians(dacke_changes)
    changes_dict["dacke"] = dacke_changes

    means = dict()
    v_tests = dict()
    circ_scatter_coordinates = dict()
    ci95s = dict()
    csd_arcs = dict()
    mean_line_rs = dict()
    p_str_dict = dict()

    for k in changes_dict.keys():
        n = len(changes_dict[k])
        radial_base = 1
        means[k] = circmean(changes_dict[k])
        v_tests[k] = v_test(changes_dict[k], 0)
        circ_scatter_coordinates[k] = circ_scatter(changes_dict[k], radial_interval=-0.1, radial_base=radial_base)
        ci95s[k] = confidence_interval(means[k][0], 0.05, n)
        half_arc = ci95s[k]/2
        csd_arcs[k] = np.linspace(means[k][1] - half_arc, means[k][1] + half_arc, 100)
        mean_line_rs[k] = np.linspace(0, radial_base + 0.1, 100)

        p = v_tests[k][0]
        p_str_dict[k] = ""
        thresholds = [0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001, 0.000001]
        thresholds.reverse()
        for t in thresholds:
            if p < t:
                p_str_dict[k] = "$p$ < {}".format(t)
                break

        if p_str_dict[k] == "":
            p_str_dict[k] = "n.s."
        

    title_dict = dict()
    title_dict["r1r2"] = r"A) Exit 1 $\rightarrow$ Exit 2"
    title_dict["r1r3"] = r"B) Exit 1 $\rightarrow$ Exit 3"
    title_dict["r1r4"] = r"C) Exit 1 $\rightarrow$ Exit 4"
    title_dict["dacke"] = "D) Dacke et al. (2019)"

    fig, axs = plt.subplot_mosaic([["r1r2","r1r3"],
                                   ["r1r4", "dacke"]],
                                  subplot_kw={"projection":"polar"},
                                  figsize=(6,6)
    )

    # Plotting
    for k in axs.keys():
        # Plot data
        mc = 'tab:blue'
        if k == "dacke":
            mc = 'tab:red'
        axs[k].scatter(circ_scatter_coordinates[k][1],
                       circ_scatter_coordinates[k][0],
                       s=30,
                       facecolor=mc,
                       edgecolor='k')
        axs[k].plot(np.zeros(len(mean_line_rs[k])) + means[k][1],
                    mean_line_rs[k],
                    color='k')
        axs[k].plot(csd_arcs[k],
                    np.zeros(len(csd_arcs[k])) + 1.1,
                    color='k')

        # Make pretty
        axs[k].set_theta_direction(-1)
        axs[k].set_theta_zero_location("N")
        axs[k].set_xticks([0,np.pi/2, np.pi, 3*np.pi/2],
                             labels=["0$^\degree$",
                                     "90$^\degree$",
                                     "180$^\degree$",
                                     "-90$^\degree$"],
                             fontsize=14)
        axs[k].set_ylim([0,radial_base + 0.15])
        axs[k].set_yticks([])
        axs[k].set_title(title_dict[k])
        s0 = angular_deviation(means[k][0])
        axs[k].text(0,
                    0,
                    "$\mu$ = {:.2f}$^\degree$\n$s0$ = {:.2f}$^\degree$\n$V = {:.2f}$\n{}"
                    .format(np.degrees(means[k][1]), np.degrees(s0), v_tests[k][1], p_str_dict[k]),
                    ha="center",
                    va="center",
                    bbox=dict(facecolor='1', edgecolor='k', pad=1.5)
        )
        

    plt.tight_layout()

    outfile_name = filename.split(".")[0].split("/")[1]
    outfile = "svg/" + outfile_name + ".svg"
    
    plt.savefig(outfile, bbox_inches="tight", dpi=400)


if __name__ == "__main__":
    produce_plot("csvs/left_hand.csv")
    produce_plot("csvs/right_hand.csv")
    produce_plot("csvs/alternate.csv")
    produce_plot("csvs/no_norm.csv")
                 
