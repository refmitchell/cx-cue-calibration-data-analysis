"""
dances_to_json.py

Takes a directory containing rosbag files, extracts the angles encoded by each
neural population during dances, and then packs these into a JSON file.

JSON will be of the form. 
{"bXXX": {
  "d1":{
    "epg":[ epg angles over dance 1 ]
    "r1": [ r1 angles over dance 1 ]
    "r2": [ r2 angles over dance 1 ]
  },
  "d2":{
...
}
}

where "bXXX" is the id of the beetlebot run beetle_001.bag becomes b001 for 
example.

"""


import numpy as np
import matplotlib.pyplot as plt
import rosbag
import os
import argparse
import sys
import json


def decode_population(rates):
    """
    Given a population of Rs or EPGs, decode to get a single angle
    based on preset preferred angles taken from the ring model.
    :param rates: the neural rates, expected to be of size 8
    :return: angle encoded by population in radians
    """
    if sum(rates) == 0:
        # Rates are always strictly positive so will only be 0 if the population
        # is flat. NAN indicates that the population was off for this segment.
        return np.NAN

    prefs = np.linspace(0,2*np.pi,8, endpoint=False)
    euler_rep = [r * np.exp(-1j * p) for (r,p) in zip(rates,prefs)]

    return np.angle(np.sum(euler_rep))

if __name__ == "__main__":
    path = ""
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', required=True, action='store')
    parser.add_argument('--output', dest='outfile', required=True, action='store')
    args = parser.parse_args()
    path = args.path
    outfile = args.outfile

    print("Target: {}".format(path))
    print("Outfile: {}".format(outfile))

    if not  os.path.isdir(path):
        print("Specified path is not a directory.")
        sys.exit()

    # Move to target diretory and check for bagfiles
    cwd = os.getcwd()
    os.chdir(path)
    filenames = os.listdir()
    filenames = [x for x in filenames if ".bag" in x]
    if filenames == []:
        print("Specified path does not contain any bagfiles")
        sys.exit()

    filenames = [x for x in filenames if "beetle_" in x]
    if filenames == []:
        print("Specified path does not contain any bagfiles of known format")
        sys.exit()    

    full_dict = dict()
    for bagfile in filenames:
        # For a given beetle
        bot_idx_string = "b" +  bagfile.split(".")[0].split("_")[1]
        print(bot_idx_string)
        bag = rosbag.Bag(bagfile)

        context_data = []
        r1_angles = []
        r2_angles = []
        epg_angles = []
        pegL_angles = []
        pegR_angles = []
        penL_angles = []
        penR_angles = []
        d7_angles = []

        # Extract relevant ROS data
        for topic, msg, t in bag.read_messages(topics=["erm_status", "context"]):
            if topic == "context":
                context_data.append(msg.data)
            if topic == "erm_status":
                r1_angles.append(decode_population(msg.r1))
                r2_angles.append(decode_population(msg.r2))
                epg_angles.append(decode_population(msg.epg))
                pegL_angles.append(decode_population(msg.peg[:8]))
                pegR_angles.append(decode_population(msg.peg[8:]))
                penL_angles.append(decode_population(msg.pen[:8]))
                penR_angles.append(decode_population(msg.pen[8:]))
                d7_angles.append(decode_population(msg.d7))

        last_context = "stopped"
        dance_counter = 0

        dance_data = dict()
        dance_contexts = ["dance", "just_dance"]

        for i in range(len(context_data)):
            context = context_data[i]
            if last_context == "stopped" and (context in dance_contexts):
                dance_counter += 1
                dance_key = "d" + str(dance_counter)
                dance_data[dance_key] = dict()
                dance_data[dance_key]["epg"] = []
                dance_data[dance_key]["r1"] = []
                dance_data[dance_key]["r2"] = []
                dance_data[dance_key]["pegL"] = []
                dance_data[dance_key]["pegR"] = []
                dance_data[dance_key]["penL"] = []
                dance_data[dance_key]["penR"] = []
                dance_data[dance_key]["d7"] = []                

            if context in dance_contexts:
                dance_data[dance_key]["epg"].append(epg_angles[i])
                dance_data[dance_key]["r1"].append(r1_angles[i])
                dance_data[dance_key]["r2"].append(r2_angles[i])
                dance_data[dance_key]["pegL"].append(pegL_angles[i])
                dance_data[dance_key]["pegR"].append(pegR_angles[i])
                dance_data[dance_key]["penL"].append(penL_angles[i])
                dance_data[dance_key]["penR"].append(penR_angles[i])
                dance_data[dance_key]["d7"].append(d7_angles[i])   
                

            last_context = context

        full_dict[bot_idx_string] = dance_data

    os.chdir(cwd)
    with open(outfile, 'w') as f:
        print("writing json")        
        json.dump(full_dict, f)

    print("done")
