## Data/code repository for "A robot model of compass cue calibration in the insect brain"
This repository contains the raw data and analysis code for the manuscript "A robot model of compass cue calibration in the insect brain". 

Code for the robot and the neural model are **not** included in this repository but can be found [here]().

If you want to inspect/make use of this data or analysis and it is not clear how to do so, please feel free to reach out via the correspondence address in the manuscript. 

### Basic data and analysis
In the `analysis` directory, there are three subdirectories:

- `csvs` contains the exit angles for each experimental condition (alternating dances, left-hand dances, right-hand dances, and normalisation disabled). It also contains the relevant data against which we compared from [Dacke et al. (2019)](https://www.pnas.org/doi/10.1073/pnas.1904308116).
- `json` contains the processed model recordings from each experiment. Each file contains an entry for each "individual" which stores the angles encoded by each neural population over the course of each dance.
- `svg` contains svg files output by the analysis/plotting routines. These raw files were neatened up using Inkscape before inclusion in the manuscript.

In addition there are a set of python scripts:
- `plot_data.py` is used to produce the change in bearing plots for each condition.
- `correlation.py` is used to produce the neural recording plots. 
- `dances_to_json.py` is used to convert raw model recordings (rosbag format) into json files which contain the angles encoded during each dance (the processed json files in the `json` directory).
- `test_utilities.py` contains useful utilities and stats functions for working with circular data.

### Raw model recordings
Raw model recordings are available in `recordings/dataset` with one subdirectory for each condition. There is one recording for each beetle. 

These recordings were performed using [rosbag](https://wiki.ros.org/rosbag) and can be played back using rosbag. The recordings can also be accessed using the rosbag library for python. This should not require a full ROS installation.

### Robot code
The full robot codebase is available [here](https://github.com/refmitchell/beetlebot_software/tree/master).

Some code has been included here for ease of reference. The `robot` directory contains:

- `extended_ring_model.py` this is the core neural model (including steering circuit) used for this work. 
- `dict_key_definitions.py` this is an ancillary file which contains dictionary keys used in the ring model.
- `mt_experiment` this python script contains the routine used to run the robot in the arena. *It will not run* without ROS or the extra framework provided by the full beetlebot codebase. It is here for reference only.