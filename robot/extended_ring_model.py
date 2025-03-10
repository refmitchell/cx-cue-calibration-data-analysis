"""
extended_ring_model.py

This module contains the orientation cue integration model from
Mitchell et al. (2023) with some adaptations:

1. R -> E-PG plasticity can now be linearly scaled according to
   the current angular velocity input.
2. If (1) is enabled, R influence over E-PG neurons is scaled 
   according to ngular velocity as suggested in the Discussion 
   from Mitchell et al. (2023).
3. A simple steering method has been added which will generate a
   steering command when given a goal direction. This can either
   use a basic vector calculation or a bio-inspired (but not anatomical)
   steering circuit.

For more information on the basic model and further reading/references,
please see Mitchell et al. (2023).

References:
Mitchell et al. (2023) - A model of cue integration as vector summation
                         in the insect brain.

"""

from dict_key_definitions import rmkeys, decodekeys
import numpy as np
import matplotlib.pyplot as plt

class RingModel():
    def __init__(self, params=dict()):
        """
        Constructor.

        :param params: Full argument dictionary. All arguments have
        sensible defaults so any ommitted arguments are assumed to
        be unmodified. Invalid arguments are ignored.
        """
        # Learning parameters
        self.learning_rate = params.get(rmkeys.lr, 0.1)
        self.r_threshold = params.get(rmkeys.r_threshold, 0.7)
        self.epg_threshold = params.get(rmkeys.epg_threshold, 0.9)

        # Number of R neuronsc
        self.n_r1 = params.get(rmkeys.n_r1, 8)
        self.n_r2 = params.get(rmkeys.n_r2, 8)
        self.n_r = params.get(rmkeys.n_r, 0) # Set both, default zero.

        # R scaling 
        self.scalar_r_epg = params.get(rmkeys.w_r_epg, -1.4) # E-PG -> R

        # E-PG out
        self.scalar_epg_peg = params.get(rmkeys.w_epg_peg, 1.2) # E-PG -> P-EG
        self.scalar_epg_pen = params.get(rmkeys.w_epg_pen, 0.8) # E-PG -> P-EN
        self.scalar_epg_d7 = params.get(rmkeys.w_epg_d7, 0.5) # E-PG -> D7
        self.scalar_epg_pfl3 = params.get(rmkeys.w_epg_pfl3, 0.2) # E-PG -> PFL3

        # D7 out
        self.scalar_d7_peg = params.get(rmkeys.w_d7_peg, -0.3) # D7 -> P-EG
        self.scalar_d7_pen = params.get(rmkeys.w_d7_pen, -0.6) # D7 -> P-EG
        self.scalar_d7_d7 = params.get(rmkeys.w_d7_d7, 0.1) # D7 -> D7

        # PB out
        self.scalar_peg_epg = params.get(rmkeys.w_peg_epg, 0.6) # P-EG -> E-PG
        self.scalar_pen_epg = params.get(rmkeys.w_pen_epg, 1.7) # P-EN -> E-PG

        # SM
        self.scalar_sm_pen = params.get(rmkeys.w_sm_pen, 1) # SM -> P-EN

        # FC2 -> PFL3
        self.scalar_fc2_pfl3 = params.get(rmkeys.w_fc2_pfl3, 1) # FC2 -> PFL3


        # Rate parameters
        self.r_slope = params.get(rmkeys.r_slope, 4)  # R neurons
        self.r_bias = params.get(rmkeys.r_bias, 1.8)
        self.epg_slope = params.get(rmkeys.epg_slope, 4) # E-PGs
        self.epg_bias = params.get(rmkeys.epg_bias, 1.8)
        self.d7_slope = params.get(rmkeys.d7_slope, 3)  # D7s
        self.d7_bias = params.get(rmkeys.d7_bias, 3)
        self.peg_slope = params.get(rmkeys.peg_slope, 4) # P-EGs
        self.peg_bias = params.get(rmkeys.peg_bias, 3)
        self.pen_slope = params.get(rmkeys.pen_slope, 4) # P-ENs
        self.pen_bias = params.get(rmkeys.pen_bias, 5)
        self.pfl3_slope = params.get(rmkeys.pfl3_slope, 4) # PFL3s
        self.pfl3_bias = params.get(rmkeys.pfl3_bias, 3)
        self.fc2_slope = params.get(rmkeys.fc2_bias, 4) # FC2s
        self.fc2_bias = params.get(rmkeys.fc2_slope, 1)

        # Misc.
        # Default R -> E-PG adjacency modifiers
        self.d_w1 = params.get(rmkeys.d_w1, 0.5)
        self.d_w2 = params.get(rmkeys.d_w2, 0.5)

        # Dynamic inhibition
        self.dynamic_r_inhibition = params.get(rmkeys.dynamic_r_inhibition, True)
        self.r_inhibition = params.get(rmkeys.r_inhibition, 0.2) # R influence scaling

        # Show input arrays to neuron populations
        self.show_inputs = params.get(rmkeys.show_inputs, False)
        self.verbose = params.get(rmkeys.verbose, False)
        if self.verbose: self.print_spec()

        # R neuron configuration
        # If n_r is set, then this overrides any individual group size setting.
        if self.n_r != 0:
            self.n_r1 = self.n_r
            self.n_r2 = self.n_r

        self.r1_preferences = np.linspace(0, 2*np.pi, self.n_r1, endpoint=False)
        self.r2_preferences = np.linspace(0, 2*np.pi, self.n_r2, endpoint=False)


        # Non-R neuron counts.
        self.n_epg = 8
        self.n_d7 = 8
        self.n_pen = 16
        self.n_peg = 16
        self.n_fc2 = 8
        self.n_pfl3 = 8 # Number of PFL3s going to one LAL

        # All neural rate containers as explicit column vectors
        # R neurons
        self.r1_rates = np.zeros((self.n_r1,1))
        self.r2_rates = np.zeros((self.n_r2,1))

        # Delta7 neurons
        self.d7_rates = np.zeros((self.n_d7,1))

        # E-PG neurons
        self.epg_rates = np.zeros((self.n_epg,1))

        # P-EG neurons
        self.peg_rates = np.zeros((self.n_peg,1))

        # P-EN neurons
        self.pen_rates = np.zeros((self.n_pen,1))

        # FC2 (goal) neurons
        self.fc2_rates = np.zeros((self.n_fc2, 1))
        
        # PFL3 Left (steering) neurons
        self.pfl3L_rates = np.zeros((self.n_pfl3, 1))

        # PFL3 Right (steering) neurons
        self.pfl3R_rates = np.zeros((self.n_pfl3, 1))
        
        # Adjacency matrices
        # R -> E-PG
        self.epg_preferences = np.linspace(0, 2*np.pi, self.n_epg, endpoint=False)

        self.w_r1_epg = self.d_w1 * generate_mapping(self.n_r1,
                                                     self.r1_preferences,
                                                     self.n_epg,
                                                     self.epg_preferences)
        self.w_r2_epg = self.d_w2 * generate_mapping(self.n_r2,
                                                     self.r2_preferences,
                                                     self.n_epg,
                                                     self.epg_preferences)


        self.fc2_preferences = np.linspace(0, 2*np.pi, self.n_fc2, endpoint=False)
        
        weight_onto_epg = np.sum(self.w_r1_epg, axis=1) + np.sum(self.w_r2_epg, axis=1)
        self.w_r1_epg = (self.w_r1_epg.T / weight_onto_epg).T
        self.w_r2_epg = (self.w_r2_epg.T / weight_onto_epg).T

        # E-PG -> Delta7
        self.w_epg_d7 = (np.identity(self.n_d7) - 1) * -self.scalar_epg_d7

        # E-PG -> P-EN
        idents = (np.identity(self.n_epg), np.identity(self.n_epg))
        self.w_epg_pen = np.concatenate(idents, axis=0) * self.scalar_epg_pen

        # P-EN -> E-PG
        idents_l = np.roll(np.identity(self.n_epg),-1,axis=0) # leftshift
        idents_r = np.roll(np.identity(self.n_epg),1,axis=0) # rightshift
        idents = (idents_l, idents_r)
        self.w_pen_epg = np.concatenate(idents, axis=1) * self.scalar_pen_epg

        # E-PG -> P-EG
        idents = (np.identity(self.n_epg), np.identity(self.n_epg))
        self.w_epg_peg = np.concatenate(idents, axis=0) * self.scalar_epg_peg

        # E-PG -> PFL3 Left
        self.w_epg_pfl3L = self.scalar_epg_pfl3 * np.identity(self.n_epg)

        # E-PG -> PFL3 Right
        # (inputs in the E-PGs are right-shifted by one column
        self.w_epg_pfl3R = self.scalar_epg_pfl3 * np.roll(
            np.identity(self.n_epg), 1, axis=1
        )

        # P-EG -> E-PG
        idents = (np.identity(self.n_epg), np.identity(self.n_epg))
        self.w_peg_epg = np.concatenate(idents, axis=1) * self.scalar_peg_epg

        # Delta7 -> P-EN
        idents = (np.identity(self.n_d7), np.identity(self.n_d7))
        self.w_d7_pen = np.concatenate(idents, axis=0) * self.scalar_d7_pen

        # Delta7 -> P-EG
        idents = (np.identity(self.n_d7), np.identity(self.n_d7))
        self.w_d7_peg = np.concatenate(idents, axis=0) * self.scalar_d7_peg

        # Delta7 -> Delta7 (RA dynamics)
        self.w_d7_d7 = (np.identity(self.n_d7) - 1) * self.scalar_d7_d7

        # FC2 -> PFL3
        # (FC2 neurons connect to the same neurons in the left and right populations)
        self.w_fc2_pfl3 = self.scalar_fc2_pfl3 * np.identity(self.n_fc2)

        # Last steering output for logging/visualisation.
        self.last_steering_output = 0

        # AV regulated plasticity additional parameters
        self.sm_fixed_point = 20.9 # Decrease increases learning rate
        self.av_learning_regulation_constant = 0.1 # Increase increases learning rate
        self.r_inh_slope = 5/6 # Increase increases R inhibition prop. to angular velocity


    def reset_rates(self):
        """
        Zero all neural rates.
        """
        self.r1_rates = np.zeros((self.n_r1,1))
        self.r2_rates = np.zeros((self.n_r2,1))
        self.d7_rates = np.zeros((self.n_d7,1))
        self.epg_rates = np.zeros((self.n_epg,1))
        self.peg_rates = np.zeros((self.n_peg,1))
        self.pen_rates = np.zeros((self.n_pen,1))

    def update_state(self, a1, a2, sm, w1=0.5, w2=0.5,
                     plasticity=False, initialisation=False,
                     av_plasticity=False):
        """
        Compute the state of the network given some angular input to
        the R neurons. The effect is cumulative (intended to be run from
        within a simulation loop). The function does not return anythin,g
        callers are expected to extract the information they need at any
        given timestep.

        :param a1: Cue 1 input angle (degrees)
        :param a2: Cue 2 input angle (degrees)
        :param sm: Self-motion (angular velocity).
        :param w1: The weight of cue one
        :param w2: The weight of cue two (w1 + w2 should be 1)
        :param plasticity: Enable or disable R -> E-PG plasticity
        :param initialisation: Flag that this method is being called by an init
                               routine

        """
        # Scaling factors arbitrarily chosen so that the vector stored in
        # each R layer scales from 0 to 1
        cue_one = np.deg2rad(a1)
        cue_two = np.deg2rad(a2)

        # If DI enabled and plasticity is enabled
        inhibit_rs = (plasticity & self.dynamic_r_inhibition)

        self.r1_output(cue_one, r1_scale=0.5, weight=w1) # Angular input
        self.r2_output(cue_two, r2_scale=0.5, weight=w2)
        self.pen_output(sm) # Self-motion and previous E-PG integration
        self.epg_output(inhibit_rs=inhibit_rs,
                        initialisation=initialisation,
                        w1=w1,
                        av_inhibition=av_plasticity,
                        av=abs(sm)
        ) # Input from sm, R, and P-EG
        self.d7_output() # Update ring attractor
        self.peg_output() # Output E-PG and D7 to P-EG

        if plasticity:
            self.update_weights()

        if av_plasticity and not plasticity: # options should be mutually exclusive
            # Change learning rate
            self.learning_rate = (self.av_learning_regulation_constant/self.sm_fixed_point) * np.clip(abs(sm), 0, self.sm_fixed_point)
            self.update_weights()
    #
    # Neuron outputs
    #
    def r1_output(self, cue_one=0, r1_scale=0, weight=0.5):
        """
        Compute the R neuron output.

        :param cue_one: The angular position of cue one.
        :param r1_scale: Scaling parameter for the input
        :param weight: The weight (output scaling)
        """
        # Modified R neuron encoding - cosine + 1
        r_input = r1_scale * np.array(
            [np.cos(x - cue_one) + 1 for x in self.r1_preferences]
        )
        if self.show_inputs: print(" R1IN: {}".format(r_input.reshape((self.n_r1,))))
        self.r1_rates = sigmoid(r_input,
                                self.r_slope,
                                self.r_bias).reshape((self.n_r1, 1)) * weight

    def r2_output(self, cue_two=0, r2_scale=0, weight=0.5):
        """
        Compute the R neuron output.

        :param cue_two: The angular position of cue one.
        :param r2_scale: Scaling parameter for the input
        :param weight: The weight (output scaling)
        """
        # Modified R neuron encoding - cosine + 1
        r_input = r2_scale * np.array(
            [np.cos(x - cue_two) + 1 for x in self.r2_preferences]
        )
        if self.show_inputs: print(" R2IN: {}".format(r_input.reshape((self.n_r2,))))
        self.r2_rates = sigmoid(r_input,
                                self.r_slope,
                                self.r_bias).reshape((self.n_r2, 1)) * weight

    def pen_output(self, self_motion=0):
        """
        P-EN neurons are updated both by both a direct self-motion signal
        and the E-PGs.

        :param self_motion: the angular velocity experienced by the agent
        """
        # Map self-motion roughly into 0,1
        scale = 0.5/self.sm_fixed_point

        scaled_sm = np.clip((scale * self_motion) + 0.5, 0, 1)

        # Counter self-motion is 1 - self-motion so overall activity
        # remains the same
        contra_sm = 1 - scaled_sm

        self.scalar_sm_pen = 1
        epg_input = np.dot(self.w_epg_pen, self.epg_rates)
        d7_input = np.dot(self.w_d7_pen, self.d7_rates)

        sm_input = np.zeros((16,1))
        if self_motion > 0: # Right turn
            sm_input[0:8] = self.scalar_sm_pen * contra_sm
            sm_input[8:] = self.scalar_sm_pen * scaled_sm
        elif self_motion < 0: # Left turn
            sm_input[0:8] = self.scalar_sm_pen * contra_sm
            sm_input[8:] = self.scalar_sm_pen * scaled_sm
        else: # Static
            sm_input[:] = 0.5


        pen_input = d7_input + epg_input + sm_input

        if self.show_inputs:
            r = pen_input.reshape((16,))
            print("PENIN: {}".format(r))
            print("  MID: {}".format( 0.5*(max(r) - min(r)) + min(r) ))

        self.pen_rates = sigmoid(pen_input,
                                 self.pen_slope,
                                 self.pen_bias)

    def peg_output(self):
        """
        P-EGs provide a recurrent signal to maintain E-PG activity.
        They receive input only from the E-PGs.
        """
        epg_input = np.dot(self.w_epg_peg, self.epg_rates)
        d7_input = np.dot(self.w_d7_peg, self.d7_rates)
        peg_input = epg_input + d7_input

        if self.show_inputs: print("PEGIN: {}".format(peg_input.reshape((16,))))
        self.peg_rates = sigmoid(peg_input,
                                 self.peg_slope,
                                 self.peg_bias)

    def epg_output(self,
                   inhibit_rs=False,
                   initialisation=False,
                   w1=0.5,
                   av_inhibition=False,
                   av=0):
        """
        E-PGs are the compass layer. They receive inputs from the R neurons,
        P-ENs and P-EGs. Note that the parameters 'initialisation' and 'w1' are
        currently not used but they are useful to have specified for debugging
        purposes.

        :param inhibit_rs: Apply inhibition to the R input (used during learning)
        :param initialisation: Flag call during init process
        :param w1: The weight given to cue one.
        """
        # Input from R neurons
        r_inputs = np.dot(self.w_r1_epg, self.r1_rates) + np.dot(self.w_r2_epg, self.r2_rates)

        r_inputs = self.scalar_r_epg*r_inputs

        # Input from P-EG neurons
        peg_inputs = np.dot(self.w_peg_epg, self.peg_rates)

        # Input from P-EN neurons
        pen_inputs = np.dot(self.w_pen_epg, self.pen_rates)

        # Dynamic R inhibition - reduce R input if plasticity is enabled
        if inhibit_rs:
            r_inputs = self.r_inhibition*r_inputs

        if av_inhibition and not inhibit_rs:
            # R inhibition tied to angular velocity.
            av_r_inhibition = 1 - np.clip(
                (self.r_inh_slope)*abs(av), 0, 0.8)#  0.8)

            r_inputs = av_r_inhibition*r_inputs


        total_input = r_inputs + peg_inputs + pen_inputs

        self.epg_rates = sigmoid(total_input,
                                 self.epg_slope,
                                 self.epg_bias)

        if self.show_inputs:
            print("EPGI: {}".format(total_input.reshape(8,)))



    def d7_output(self):
        """
        Compute the output of the Delta7 neurons.
        """
        epg_input = np.dot(self.w_epg_d7, self.epg_rates)
        d7_input = np.dot(self.w_d7_d7, self.d7_rates)
        total_input = epg_input + d7_input


        if self.show_inputs:
            r = total_input.reshape(8,)
            print(" D7I: {}".format(r))

        self.d7_rates = sigmoid(total_input,
                                self.d7_slope,
                                self.d7_bias)

        
    def fc2_output(self, goal_direction):
        """
        Compute FC2 neuron activity for a given goal direction (in radians).
        Note that this goal is not necessarily coupled to the world but is coupled
        to the compass. (As in Beetz et al. (2023) and Mussells Pires et al. (2023))

        :param goal_direction: The desired goal direction
        """
        fc2_inputs = [np.cos(goal_direction - x) for x in self.fc2_preferences]
        fc2_inputs = np.array(fc2_inputs).reshape((self.n_fc2,1))
        self.fc2_rates = sigmoid(fc2_inputs, self.fc2_slope, self.fc2_bias)

    def pfl3_output(self):
        """
        Compute PFL3 neuron activity in the Left and Right PFL populations.
        These differ in that PFL3 Right cells sample one column to the right
        of the PFL3 Left neurons in the E-PGs. This is bio-inspired but not
        anatomically correct.
        """
        pfl3L_inputs = np.dot(self.w_fc2_pfl3, self.fc2_rates)
        pfl3L_inputs += np.dot(self.w_epg_pfl3L, self.epg_rates)
        
        pfl3R_inputs = np.dot(self.w_fc2_pfl3, self.fc2_rates)
        pfl3R_inputs += np.dot(self.w_epg_pfl3R, self.epg_rates)

        self.pfl3L_rates = sigmoid(pfl3L_inputs, self.pfl3_slope, self.pfl3_bias)
        self.pfl3R_rates = sigmoid(pfl3R_inputs, self.pfl3_slope, self.pfl3_bias)

    def motor_output(self):
        """
        Computes a steering command based on PFL3 left and right population 
        activity imbalances.
        """
        return np.sum(self.pfl3R_rates) - np.sum(self.pfl3L_rates)


    def update_weights(self):
        """
        Update the weights between the R and E-PG rings.
        """
        self.r1_threshold = self.r_threshold * max(self.r1_rates)
        self.r2_threshold = self.r_threshold * max(self.r2_rates)

        r1_updates = np.dot(
            (self.epg_rates - self.epg_threshold).reshape((self.n_epg,1)),
            (self.r1_rates - self.r1_threshold).reshape((1,self.n_r1))
        ) * self.learning_rate

        r2_updates = np.dot(
            (self.epg_rates - self.epg_threshold).reshape((self.n_epg,1)),
            (self.r2_rates - self.r2_threshold).reshape((1,self.n_r2))
        ) * self.learning_rate

        # Anti-hebbian learning
        r1_updates *= -1
        r2_updates *= -1

        # Apply additive weight update
        self.w_r1_epg += r1_updates
        self.w_r1_epg = np.clip(self.w_r1_epg, 0, 1)

        self.w_r2_epg += r2_updates
        self.w_r2_epg = np.clip(self.w_r2_epg, 0, 1)

        # Normalisation - all synapses *onto* an E-PG must sum to 1
        weight_onto_epg = np.sum(self.w_r1_epg, axis=1) + np.sum(self.w_r2_epg, axis=1)
        self.w_r1_epg = (self.w_r1_epg.T / weight_onto_epg).T
        self.w_r2_epg = (self.w_r2_epg.T / weight_onto_epg).T


    def steering(self, goal_direction, bio=False):
        """
        Generates a left-right steering command which compares a goal direction
        to the current heading stored in the E-PG population.

        If bio=False, then the perp dot product will be used to determine the
        correct steering command based on the E-PG population.

        If bio=True, then a bio-inspired steering circuit will be used instead.
        Note that this circuit is bio-inspired but not anatomical; it operates
        on the same principle as the steering circuit described by Mussells 
        Pires et al. (2023; biorxiv). See the Thesis for more information.

        :param goal_direction: The goal direction we want in the E-PG population.
        :param bio: Use the bio inspired circuit for steering
        :return: a left-right steering command. It is up to the caller to
                 determine the correct motor commands to issue to the robot.
        """
        if not bio:
            # Current EPG heading in [-np.pi, np.pi]
            current_theta = self.decode()[decodekeys.epg][0]

            # Signed error between current E-PG angle and desired
            desired = np.array([np.cos(goal_direction), np.sin(goal_direction)])
            internal = np.array([np.cos(current_theta), np.sin(current_theta)])
            x_proj = internal[0]*desired[0] + internal[1]*desired[1]
            y_proj = -internal[0]*desired[1] + internal[1]*desired[0]

            # Gives positive counter clockwise angle (as per standard convention)
            # of internal from desired. If this angle is positive, the corrective turn
            # (clockwise) should be negative. 
            error = np.arctan2(y_proj, x_proj)
            print("{}:{}:{}".format(np.degrees(current_theta),
                                    np.degrees(np.arctan2(desired[1], desired[0])),
                                    np.degrees(error)))

            # Store for logging/vis.
            self.last_steering_output = error
            return self.last_steering_output

        # Bio-inspired version
        # Compute FC2 activity for this goal angle
        # Compute PFL3L and PFL3R activity for this goal and current EPG
        # Then compute the motor output from the imbalance in PFL3 populations
        self.fc2_output(goal_direction)
        self.pfl3_output()
        self.last_steering_output = self.motor_output()
        return self.last_steering_output

    def randomise_weights(self):
        """
        Randomises the R -> E-PG mappings. In the original paper 'flat' mappings
        (all connections equal) but randomisation seems to be the more standard 
        technique.
        """
        self.w_r1_epg = np.random.uniform(size=(self.n_r1, self.n_epg))
        self.w_r2_epg = np.random.uniform(size=(self.n_r2, self.n_epg))
                                          
        weight_onto_epg = np.sum(self.w_r1_epg, axis=1) + np.sum(self.w_r2_epg, axis=1)
        self.w_r1_epg = (self.w_r1_epg.T / weight_onto_epg).T
        self.w_r2_epg = (self.w_r2_epg.T / weight_onto_epg).T

    def decode(self):
        """
        Return a dictionary containing the angle stored in each of the neural
        populations.
        """
        ret = dict()
        ret[decodekeys.r1] = self.__decode_layer(self.r1_preferences,
                                                 self.r1_rates)
        ret[decodekeys.r2] = self.__decode_layer(self.r2_preferences,
                                                 self.r2_rates)
        ret[decodekeys.epg] = self.__decode_layer(self.epg_preferences,
                                                  self.epg_rates)
        ret[decodekeys.d7] = self.__decode_layer(self.epg_preferences,
                                                 self.d7_rates)
        ret[decodekeys.fc2] = self.__decode_layer(self.fc2_preferences,
                                                  self.fc2_rates)
        ret[decodekeys.pfl3L] = self.__decode_layer(self.epg_preferences,
                                                    self.pfl3L_rates)
        ret[decodekeys.pfl3R] = self.__decode_layer(self.epg_preferences,
                                                    self.pfl3R_rates)
        ret[decodekeys.pen] = self.__decode_layer(self.epg_preferences,
                                                  self.pen_rates[:8])
        ret[decodekeys.peg] = self.__decode_layer(self.epg_preferences,
                                                  self.peg_rates[:8])

        # R -> E-PG map decoding (to which directions do R neurons map)
        # R1s
        w_r1_epg_array = []
        for r_idx in range(self.n_r1):
            neuron_paths = self.w_r1_epg[:, r_idx]
            val = self.__decode_layer(self.epg_preferences, neuron_paths)
            w_r1_epg_array.append(val)

        # R2s
        w_r2_epg_array = []
        for r_idx in range(self.n_r2):
            neuron_paths = self.w_r2_epg[:, r_idx]
            val = self.__decode_layer(self.epg_preferences, neuron_paths)
            w_r2_epg_array.append(val)

        ret[decodekeys.r1_epg] = w_r1_epg_array
        ret[decodekeys.r2_epg] = w_r2_epg_array

        return ret

    def initialise(self, c1=0, c2=0, w1=0.5, w2=0.5, time=500, velocity=24, use_defaults=False):
        """
        Initialisation routine. Should be called before any experiment to place
        the model in a known start state.

        :param c1: Cue one start position
        :param c2: Cue two start position
        :param w1: Cue one weight
        :param w2: Cue two weight
        :param time: Initialisation duration
        :param velocity: Angular velocity during init process
        :return: Unused
        """
        change = velocity
        self.randomise_weights()
        for t in range(time):
            c1+=change
            c2+=change
            self.update_state(c1,
                              c2,
                              sm=change,
                              w1=w1,
                              w2=w2,
                              plasticity=False,
                              initialisation=True,
                              av_plasticity=True
            )

        if use_defaults:
            self.w_r1_epg = self.d_w1 * generate_mapping(self.n_r1,
                                                         self.r1_preferences,
                                                         self.n_epg,
                                                         self.epg_preferences)
            self.w_r2_epg = self.d_w2 * generate_mapping(self.n_r2,
                                                         self.r2_preferences,
                                                         self.n_epg,
                                                         self.epg_preferences)

        
            weight_onto_epg = np.sum(self.w_r1_epg, axis=1) + np.sum(self.w_r2_epg, axis=1)
            self.w_r1_epg = (self.w_r1_epg.T / weight_onto_epg).T
            self.w_r2_epg = (self.w_r2_epg.T / weight_onto_epg).T
            

        return c1, c2 # Return cue positions at the end of initialisation


    def __decode_layer(self, prefs, rates):
        """
        Compute the angle stored in a population given a sequence of preferred
        angles and neural rates for those angles. Note that for a network using
        learned connections, the E-PG preferences will not be correct as the
        learning process can introduce an offset between R inputs and E-PGs.

        :param prefs: The preferred angles of each neuron.
        :param rates: The neural rates for neach neuron.
        :return: Average angle theta and mean vector length R.
        """
        rates = rates.reshape((len(rates),))
        polar = list(zip(rates, prefs))
        x = [ r*np.cos(t) for (r, t) in polar]
        y = [r*np.sin(t) for (r, t) in polar]
        avg_x = sum(x) / len(x)
        avg_y = sum(y) / len(y)
        return np.arctan2(avg_y, avg_x), np.sqrt(avg_x**2 + avg_y**2)

    def print_spec(self):
        print("=== Ring model specification ===")
        print("= Learning parameters =")
        print("learning_rate: {}".format(self.learning_rate))
        print("r_threshold:   {}".format(self.r_threshold))
        print("epg_threshold: {}".format(self.epg_threshold))
        print("")
        print("= # of R neurons =")
        print("n_r1: {}".format(self.n_r1))
        print("n_r2: {}".format(self.n_r2))
        print("n_r:  {}".format(self.n_r))
        print("")
        print("= Weight parameters =")
        print("R -> E-PG:    {}".format(self.scalar_r_epg))
        print("E-PG -> P-EG: {}".format(self.scalar_epg_peg))
        print("E-PG -> P-EN: {}".format(self.scalar_epg_pen))
        print("E-PG -> D7:   {}".format(self.scalar_epg_d7))
        print("D7 -> P-EG:     {}".format(self.scalar_d7_peg))
        print("D7 -> P-EN:   {}".format(self.scalar_d7_pen))
        print("D7 -> D7:     {}".format(self.scalar_d7_d7))
        print("P-EG -> E-PG: {}".format(self.scalar_peg_epg))
        print("P-EN -> E-PG: {}".format(self.scalar_pen_epg))
        print("SM -> P-EN:   {}".format(self.scalar_sm_pen))
        print("")
        print("= Rate parameters=")
        # Rate parameters
        print("R slope:    {}".format(self.r_slope))
        print("R bias:     {}".format(self.r_bias))
        print("E-PG slope: {}".format(self.epg_slope))
        print("E-PG bias:  {}".format(self.epg_bias))
        print("D7 slope:   {}".format(self.d7_slope))
        print("D7 bias:    {}".format(self.d7_bias))
        print("P-EG slope: {}".format(self.peg_slope))
        print("P-EG bias:  {}".format(self.peg_bias))
        print("P-EN slope: {}".format(self.pen_slope))
        print("P-EN bias:  {}".format(self.pen_bias))
        print("")

        # Misc.
        # Default R -> E-PG adjacency modifiers

        print("= Misc. =")
        print("Default W1: {}".format(self.d_w1))
        print("Default W2: {}".format(self.d_w2))
        print("")


def sigmoid(x, slope, bias):
    """
    Sigmoid activation function

    :param x: The input
    :param slope: The steepness of the curve
    :param bias: A left/right shift of the curve
    """
    gaussian_noise = np.random.normal(0, 0.001, size=x.shape)
    activated_input = 1 / (1 + np.exp(-(x * slope - bias)))
    return activated_input + gaussian_noise



def mag(vec):
    """
    Compute the magnitude of a cartesian vector.
    :param vec: A cartesian vector of arbitrary dimension
    :return: The magnitude of the vector
    """
    s = 0
    for element in vec:
        s += element**2
    return np.sqrt(s)

def generate_mapping(n_r,
                     r_prefs,
                     n_epg,
                     epg_preferences):
    """
    Generate a mapping between R and E-PG neurons where
    the number of R and E-PG neurons is not necessarily
    equal. Each R maps to any E-PG with 'preferred direction'
    within +/-90 degrees with weight inversely proportional
    to the difference in preferred direction.

    Number of E-PGs is assumed to be fixed at eight but
    this algorithm generalises to any number of E-PGs/R
    neurons.

    :param n_r: The number of R neurons
    :param r_prefs: The R neuron preferred directions
    :param n_epg: The number of epg_neurons
    :param epg_preferences: The 'preferred directions' (misnomer) of the
                            e-pg neurons
    :return: The R->E-PG weight matrix
    """
    # The E-PG "preferred directions" and count.
    w = np.zeros((len(epg_preferences), len(r_prefs)))

    for i in range(len(r_prefs)):
        indices = [] # any epg indices with rf within 90deg
        r = r_prefs[i]

        for j in range(len(epg_preferences)):
            e = epg_preferences[j]
            r_cart = np.array([np.cos(r), np.sin(r)])
            e_cart = np.array([np.cos(e), np.sin(e)])

            # Rounding error causing issues (1.0000000000000002), generating nans
            arg = np.clip(np.dot(r_cart, e_cart) / (mag(r_cart) * mag(e_cart)), -1, 1)
            angle = np.arccos(arg)

            info = (j, angle)
            w[j,i] = (np.pi) - angle

        # Normalise column, weights from each R neuron sum to 1.
        w[:,i] = w[:,i] / sum(w[:,i])

    return w
