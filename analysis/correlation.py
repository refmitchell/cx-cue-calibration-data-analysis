"""
correlation.py

Create correlation plots for angles encoded by each neural population during
dances.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import pandas as pd
from test_utilities import angular_difference

from scipy.stats import circmean

def generate_plot_data(data, dk, clean_borders=True):
    """
    Generate the average EPG and R correlations for dance dk over all beetles.
    The sets returned will have one entry for each timestep and are averaged over
    all beetles.

    :param data: The full dataset
    :param dk: The target dance
    :param clean_borders: Clean instances of jumps over the zero line
    :return: a dictionary with keys "epg", "r1", and "r2" containing the appropriate means
    """
    # Linear search for minimum dance length. Will be around
    # 180.
    min_duration = len(data["b001"][dk]["epg"])
    for k in data.keys():
        if len(data[k][dk]["epg"]) < min_duration:
            min_duration = len(data[k][dk]["epg"])

    # One 'x' per timestep, one 'y' per beetle per dance
    container_shape = (len(data.keys()), min_duration)
    norm_epg_angles = np.empty(container_shape)
    norm_r1_angles = np.empty(container_shape)
    norm_r2_angles = np.empty(container_shape)
    norm_pegL_angles = np.empty(container_shape) 
    norm_pegR_angles = np.empty(container_shape) 
    norm_penL_angles = np.empty(container_shape) 
    norm_penR_angles = np.empty(container_shape) 
    norm_d7_angles = np.empty(container_shape) 
    
    
    for kdx in range(len(data.keys())):
        k = list(data.keys())[kdx]

        dance_data = data[k][dk]
        epg_angles = np.array(dance_data["epg"])
        r1_angles = np.array(dance_data["r1"])
        r2_angles = np.array(dance_data["r2"])
        pegL_angles = np.array(dance_data["pegL"])
        pegR_angles = np.array(dance_data["pegR"])
        penL_angles = np.array(dance_data["penL"])
        penR_angles = np.array(dance_data["penR"])
        d7_angles = np.array(dance_data["d7"])        

        # Align representations
        epg_angles -= epg_angles[0]
        r1_angles -= r1_angles[0]
        r2_angles -= r2_angles[0]
        pegL_angles -= pegL_angles[0] 
        pegR_angles -= pegR_angles[0] 
        penL_angles -= penL_angles[0]
        penR_angles -= penR_angles[0] 
        d7_angles -= d7_angles[0] 

        epg_angles = epg_angles % (2*np.pi)
        r1_angles = r1_angles % (2*np.pi)
        r2_angles = r2_angles % (2*np.pi)
        pegL_angles = pegL_angles % (2*np.pi) 
        pegR_angles = pegR_angles % (2*np.pi) 
        penL_angles = penL_angles % (2*np.pi)
        penR_angles = penR_angles % (2*np.pi) 
        d7_angles = d7_angles % (2*np.pi)         

        # Drop some entries from the end of the array to even up
        # the lengths.
        norm_epg_angles[kdx] = epg_angles[:min_duration]
        norm_r1_angles[kdx] = r1_angles[:min_duration]
        norm_r2_angles[kdx] = r2_angles[:min_duration]
        norm_pegL_angles[kdx] = pegL_angles[:min_duration] 
        norm_pegR_angles[kdx] = pegR_angles[:min_duration] 
        norm_penL_angles[kdx] = penL_angles[:min_duration]
        norm_penR_angles[kdx] = penR_angles[:min_duration] 
        norm_d7_angles[kdx] = d7_angles[:min_duration]         


    mean_dict = dict.fromkeys(["epg", "r1", "r2", "pegL", "pegR", "penL", "penR", "d7"])
    mean_dict["epg"] = circmean(norm_epg_angles, axis=0)
    mean_dict["r1"] = circmean(norm_r1_angles, axis=0, nan_policy='omit')
    mean_dict["r2"] = circmean(norm_r2_angles, axis=0, nan_policy='omit')
    mean_dict["pegL"] = circmean(norm_pegL_angles, axis=0)
    mean_dict["pegR"] = circmean(norm_pegR_angles, axis=0)
    mean_dict["penL"] = circmean(norm_penL_angles, axis=0)
    mean_dict["penR"] = circmean(norm_penR_angles, axis=0)
    mean_dict["d7"] = circmean(norm_d7_angles, axis=0)    

    # This step depends on dance direction
    if clean_borders:
        for k in mean_dict.keys():
            for idx in range(len(mean_dict[k]) - 1):
                current = mean_dict[k][idx]
                nxt = mean_dict[k][idx + 1]

                # Continuity corrections
                if (current < np.pi/2) and (nxt > 3*np.pi/2):
                    # Check for jumps from <90deg to >270deg
                    # Shift angle so that it's negatively represented
                    # if such a jump occurs.
                    mean_dict[k][idx + 1] -= 2*np.pi

                if (current > 3*np.pi/2) and (nxt < np.pi/2):
                    # Check for jumps from >270deg to <90deg
                    mean_dict[k][idx + 1] += 2*np.pi                    

    return mean_dict

def compute_derivatives(mean_dict, degrees=True, seconds=True):
    derivatives = dict()
    for neur_k in mean_dict.keys():
        total_change = 0

        for idx in range(len(mean_dict[neur_k]) - 1):
            # Accumulate inner angles between each step
            a1 = mean_dict[neur_k][idx]
            a2 = mean_dict[neur_k][idx+1]
            diff = angular_difference(a1,a2,signed=False)
            total_change += diff

        total_change /= len(mean_dict[neur_k])

        # Store mean change per timestep
        if degrees:
            total_change = np.degrees(total_change)
        if seconds:
            total_change *= 5 # five timesteps per second
        derivatives[neur_k] = total_change

    return derivatives

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, action='store', dest="path")
    parser.add_argument("--output", required=True, action='store', dest='output')
    parser.add_argument("-d", action='store_true', dest='store_derivatives')
    args = parser.parse_args()
    path = args.path
    outfile = args.output
    store_derivatives = args.store_derivatives

    with open(path, 'r') as f:
        data = json.load(f)

    dance_means = dict.fromkeys(["d1", "d2", "d3", "d4"])
    derivatives = dict.fromkeys(dance_means.keys())
    title_dict = dict()
    title_dict["d1"] = "Dance 1"
    title_dict["d2"] = "Dance 2"
    title_dict["d3"] = "Dance 3"
    title_dict["d4"] = "Dance 4"
    
    for k in dance_means.keys():
        dance_means[k] = generate_plot_data(data, k)
        derivatives[k] = compute_derivatives(dance_means[k])

    if store_derivatives:
        deriv_dict = pd.DataFrame.from_dict(derivatives)
        deriv_filename = outfile.split(".")[0] + ".csv"
        deriv_dict.to_csv(deriv_filename, float_format="%.2f")

    mosaic = [["d1", "d2"],
              ["d3", "d4"]]
    left_plots = ["d1", "d3"]
    bottom_plots = ["d3", "d4"]

    fig, axs = plt.subplot_mosaic(mosaic, figsize=(7,7))
    
    for k in axs.keys():
        ax = axs[k]
        lw = 1
        a = 1
        ax.plot(dance_means[k]["epg"],color='tab:purple', linewidth=lw, alpha=a, label='EPG',linestyle='',marker='+',zorder=0)
        ax.plot(dance_means[k]["r1"],color='tab:red', linewidth=lw, alpha=a, label='R1')
        ax.plot(dance_means[k]["r2"],color='tab:blue', linewidth=lw, alpha=a, label='R2')
        ax.plot(dance_means[k]["pegR"],color='tab:pink', linewidth=lw, alpha=a, label='PEG(R)', linestyle='dashed')
        ax.plot(dance_means[k]["penR"],color='tab:green', linewidth=lw, alpha=a, label='PEN(R)', linestyle='dashdot')
        ax.plot(dance_means[k]["d7"],color='tab:olive', linewidth=lw, alpha=a, label=r'$\Delta$7')                
        ax.set_title(title_dict[k])
        ax.set_yticks([0, 2*np.pi], labels=[r'$0^\degree$','$360^\degree$'])
        ax.set_xticks([0, len(dance_means[k]["epg"])], ["0", "{:.0f}".format(np.round(len(dance_means[k]["epg"])/5))])

        if k == "d4":
            ax.legend()
        if k in left_plots:
            ax.set_ylabel("Encoded angular change")
        if k in bottom_plots:
            ax.set_xlabel("Time (s)")

    # plt.show()    
    plt.savefig(outfile, dpi=400, bbox_inches='tight')
