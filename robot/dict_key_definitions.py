"""
dict_key_definitions.py

Copied from Mitchell et al. (2023) to facilitate use of their cue
integration model.

References:
Mitchell et al. (2023) - A model of cue integration as vector summation in
                         the insect brain.
"""

"""
Keys for parameter dictionary used by the RingModel class in
extended_ring_model.py
"""
class constkeys_rm_param_dict():
    __slots__ = ()
    lr = "lr"
    r_threshold = "r_threshold"
    epg_threshold = "epg_threshold"
    n_r1 = "n_r1"
    n_r2 = "n_r2"
    n_r = "n_r"
    w_r_epg = "w_r_epg"
    w_epg_peg = "w_epg_peg"
    w_epg_pen = "w_epg_pen"
    w_epg_d7 = "w_epg_d7"
    w_d7_peg = "w_d7_peg"
    w_d7_pen = "w_d7_pen"
    w_d7_d7 = "w_d7_d7"
    w_peg_epg = "w_peg_epg"
    w_pen_epg = "w_pen_epg"
    w_sm_pen = "w_sm_pen"
    w_epg_pfl3 = "w_epg_pfl3"
    w_fc2_pfl3 = "w_efc2_pfl3"
    r_slope = "r_slope"
    r_bias = "r_bias"
    epg_slope = "epg_slope"
    epg_bias = "epg_bias"
    d7_slope = "d7_slope"
    d7_bias = "d7_bias"
    peg_slope = "peg_slope"
    peg_bias = "peg_bias"
    pen_slope = "pen_slope"
    pen_bias = "pen_bias"
    pfl3_slope = "pfl3_slope"
    pfl3_bias = "pfl3_bias"
    fc2_slope = "fc2_slope"
    fc2_bias = "fc2_bias"
    d_w1 = "d_w1"
    d_w2 = "d_w2"
    dynamic_r_inhibition = "dynamic_r_inhibition"
    r_inhibition = "r_inhibition"
    show_inputs = "show_inputs"
    verbose = "verbose"

"""
Keys for the decoding dictionary used by the RingModel class in
extended_ring_model.py (RingModel.decode(self))
"""
class constkeys_rm_decode_dict():
        __slots__ = ()
        r1 = "r1"
        r2 = "r2"
        epg = "epg"
        d7 = "d7"
        peg = "peg"
        pen = "pen"
        pfl3L = "pfl3L"
        pfl3R = "pfl3R"
        fc2 = "fc2"
        r1_epg = "r1_epg"
        r2_epg = "r2_epg"


#
# Instantiations - to be imported with the module
#

# Ring model keys
rmkeys = constkeys_rm_param_dict()
decodekeys = constkeys_rm_decode_dict()


