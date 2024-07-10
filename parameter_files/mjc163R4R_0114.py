########################################################################################################################
# PARAMETER FILE
#
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

        # important: need to be named in a way to contain the following keywords: EXPLORATION, TASK, SLEEP
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
        # Data params:
        #
        # defines all additional parameters of the data that might be relevant (e.g. if cells need to be split further
        # to separate left/right hemisphere, coordinates of start box etc.)
        # --------------------------------------------------------------------------------------------------------------

        self.data_params_dictionary = {
            "data_dir": "/mnt/hdl1/01_across_regions/02_data/02_Peter",
            # "data_dir": "../02_data/02_peter",
            "session_name": "mjc163R4R_0114",
            # select cell type:
            # p1: pyramidal cells of the HPC
            "cell_type_array": ["p1", "b1"],
            # defines whether cells need to be further divided
            "cell_type_sub_div": None,
            # boundaries of start box: x_min, x_max, y_min, y_max
            "start_box_coordinates": [25, 75, 120, 170],
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
        self.goal_locations = [[200.449, 246.704], [280.068, 136.773],
                               [206.093, 45.1413], [197.181, 146.616]]
        self.long_sleep_experiment_phases = ["sleep_long_1", "sleep_long_2", "sleep_long_3", "sleep_long_4",
                                             "sleep_long_5"]

        # learning cheeseboard 1
        # --------------------------------------------------------------------------------------------------------------
        self.lcb_1_nr_trials_last_5_min = range(30, 43)
        self.lcb_1_nr_trials_last_10_min = range(13, 43)
        self.lcb_1_last_5_min_phmm_modes = 35
        self.lcb_1_last_5_min_phmm_modes_old = 25
        # which data to use by default
        self.default_trials_lcb_1 = self.lcb_1_nr_trials_last_5_min

        # learning cheeseboard 2
        # --------------------------------------------------------------------------------------------------------------
        self.lcb_2_nr_trials_first_5_min = range(0, 14)
        self.lcb_2_first_5_min_phmm_modes = 30
        self.lcb_2_first_5_min_phmm_modes_old = 25
        self.default_trials_lcb_2 = self.lcb_2_nr_trials_first_5_min

        # for decoding analysis
        # --------------------------------------------------------------------------------------------------------------
        self.sleep_phase_speed_threshold = 35
        self.sleep_compression_factor_12spikes_100ms = 0.66
        self.sleep_compression_factor_12spikes_100ms_stable_cells = 12
        self.sleep_compression_factor_12spikes_100ms_decreasing_cells = 1.2
        self.sleep_compression_factor_12spikes_100ms_increasing_cells = 0.63
        self.default_exp_fam_1_model = "mjc163R4R_0114_1_p1_56_modes"
        self.default_exp_fam_2_model = "mjc163R4R_0114_14_p1_56_modes"
        self.default_pre_phmm_model = "mjc163R4R_0114_5_p1_trials_30_43_35_modes"
        self.default_pre_phmm_model_stable = "mjc163R4R_0114_5_p1_trials_30_42_14_modes"
        self.default_pre_phmm_model_dec = "mjc163R4R_0114_5_p1_trials_30_42_31_modes"
        self.default_pre_phmm_model_inc = "mjc163R4R_0114_5_p1_trials_30_42_11_modes"
        self.default_post_phmm_model = "mjc163R4R_0114_11_p1_trials_0_13_30_modes"
        self.default_pre_ising_model = "mjc163R4R_0114_5_5cm_bins_p1_trials_30_43"
        self.default_post_ising_model = "mjc163R4R_0114_11_5cm_bins_p1_trials_0_13"
