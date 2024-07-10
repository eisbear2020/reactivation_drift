########################################################################################################################
# PARAMETER FILE
#
# Peter's data: cheese board with long sleep
#
# ATTENTION:    only contains one hemisphere (16 electrodes --> the data from the other hemisphere has not been
#               clustered) --> here mjc163R3R: last R stands for right hemisphere
#
########################################################################################################################


class SessionParameters():
    """class that session specific parameters"""

    def __init__(self):

        # --------------------------------------------------------------------------------------------------------------
        # Data description
        # --------------------------------------------------------------------------------------------------------------

        # important: need to be named in a way to contain the following keywords: exploration, task, sleep
        self.data_description_dictionary = {
            "exploration_familiar": "1",
            "sleep_familiar": "2",
            "exploration_cheeseboard": "3",
            "sleep_cheeseboard_1": "4",
            "learning_cheeseboard_1": "5",
            "sleep_long_1": "6",
            "sleep_long_2": "7",
            "sleep_long_3": "8",
            "sleep_long_4": "9",
            "sleep_long_5": "10",
            "learning_cheeseboard_2": "11",
            "sleep_cheeseboard_2": "12",
            "post_probe": "13",
            "exploration_familiar_post": "14"
        }

        # --------------------------------------------------------------------------------------------------------------
        # Data description:
        #
        # defines all additional parameters of the data that might be relevant (e.g. if cells need to be split further
        # to separate left/right hemisphere, coordinates of start box etc.)
        # --------------------------------------------------------------------------------------------------------------

        self.data_params_dictionary = {
            "data_dir": "/mnt/hdl1/01_across_regions/02_data/02_Peter",
            # "data_dir": "../02_data/02_peter",
            "session_name": "mjc163R3L_0114",
            # select cell type:
            # p1: pyramidal cells of the HPC
            "cell_type_array": ["p1", "b1"],
            # defines whether cells need to be further divided
            "cell_type_sub_div": None,
            # boundaries of start box: x_min, x_max, y_min, y_max
            "start_box_coordinates": [25, 75, 125, 190],
            # experiment phase has to be defined in main
            "experiment_phase": "",
            # data description --> initialize empty, DO NOT INPUT A VALUE!
            "data_description": "",
            # spatial factor: cm/a.u. --> defines how many cm one arbitrary unit from the .whl file corresponds to
            "spatial_factor": 0.45
        }

        # --------------------------------------------------------------------------------------------------------------
        # Analysis parameters
        # --------------------------------------------------------------------------------------------------------------

        self.session_name = self.data_params_dictionary["session_name"]
        self.goal_locations = [[211.774, 243.179], [282.135, 131.950],
                       [241.337, 52.5184], [161.811, 81.1764]]
        self.long_sleep_experiment_phases = ["sleep_long_1", "sleep_long_2",
                                             "sleep_long_3", "sleep_long_4", "sleep_long_5"]

        # learning cheeseboard 1
        # --------------------------------------------------------------------------------------------------------------
        self.lcb_1_nr_trials_last_6_min = range(39, 58)
        self.lcb_1_last_6_min_phmm_modes = 40
        self.lcb_1_last_6_min_phmm_modes_old = 30
        self.lcb_1_nr_trials_learned_locations = None
        # which data to use by default
        self.default_trials_lcb_1 = self.lcb_1_nr_trials_last_6_min

        # learning cheeseboard 2
        # --------------------------------------------------------------------------------------------------------------
        self.lcb_2_nr_trials_first_6_min = range(0,17)
        self.lcb_2_first_6_min_phmm_modes = 45
        self.lcb_2_first_6_min_phmm_modes_old = 30
        # which data to use by default
        self.default_trials_lcb_2 = self.lcb_2_nr_trials_first_6_min

        # for decoding analysis
        # --------------------------------------------------------------------------------------------------------------
        self.sleep_phase_speed_threshold = 30
        self.sleep_compression_factor_12spikes_100ms = 0.75
        self.sleep_compression_factor_12spikes_100ms_stable_cells = 4
        self.sleep_compression_factor_12spikes_100ms_decreasing_cells = 1.33
        self.sleep_compression_factor_12spikes_100ms_increasing_cells = 1.26
        self.default_exp_fam_1_model = "mjc163R3L_0114_1_p1_51_modes"
        self.default_exp_fam_2_model = "mjc163R3L_0114_14_p1_61_modes"
        self.default_pre_phmm_model = "mjc163R3L_0114_5_p1_trials_39_57_40_modes"
        self.default_pre_phmm_model_stable = "mjc163R3L_0114_5_p1_trials_39_57_21_modes"
        self.default_pre_phmm_model_dec = "mjc163R3L_0114_5_p1_trials_39_57_41_modes"
        self.default_pre_phmm_model_inc = "mjc163R3L_0114_5_p1_trials_39_57_28_modes"
        self.default_post_phmm_model = "mjc163R3L_0114_11_p1_trials_0_16_45_modes"
        self.default_pre_ising_model = "mjc163R3L_0114_5_5cm_bins_p1_trials_39_57"
        self.default_post_ising_model = "mjc163R3L_0114_11_5cm_bins_p1_trials_0_16"
