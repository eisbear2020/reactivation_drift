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
            "exploration_familiar_post": "13",
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
            "session_name": "mjc148R4R_0113",
            # select cell type:
            # p1: pyramidal cells of the HPC
            "cell_type_array": ["p1", "b1"],
            # experiment phase has to be defined in main
            "experiment_phase": "",
            # data description --> initialize empty, DO NOT INPUT A VALUE!
            "data_description": "",
            # defines whether cells need to be further divided
            "cell_type_sub_div": None,
            # boundaries of start box: x_min, x_max, y_min, y_max
            "start_box_coordinates": [295, 350, 110, 170],
            # spatial factor: cm/a.u. --> defines how many cm one arbitrary unit from the .whl file corresponds to
            "spatial_factor": 0.45
        }

        # --------------------------------------------------------------------------------------------------------------
        # Analysis parameters
        # --------------------------------------------------------------------------------------------------------------

        self.session_name = self.data_params_dictionary["session_name"]
        self.goal_locations = [[176.920, 53.602], [135.888, 136.594], [225.860, 181.799], [93.363, 226.540]]
        self.long_sleep_experiment_phases = ["sleep_long_1", "sleep_long_2", "sleep_long_3", "sleep_long_4",
                                             "sleep_long_5"]

        # learning cheeseboard 1
        # --------------------------------------------------------------------------------------------------------------
        self.lcb_1_nr_trials_last_5_min = range(3, 15)
        self.lcb_1_last_5_min_phmm_modes = 35
        self.lcb_1_last_5_min_phmm_modes_old = 30
        self.lcb_1_nr_trials_learned_locations = None
        self.default_trials_lcb_1 = self.lcb_1_nr_trials_last_5_min

        # learning cheeseboard 2
        # --------------------------------------------------------------------------------------------------------------
        self.lcb_2_nr_trials_first_5_min = range(0, 16)
        self.lcb_2_first_5_min_phmm_modes = 55
        self.lcb_2_first_5_min_phmm_modes_old = 35

        # which data to use by default
        self.default_trials_lcb_2 = self.lcb_2_nr_trials_first_5_min

        # for decoding analysis
        # --------------------------------------------------------------------------------------------------------------
        self.sleep_phase_speed_threshold = 15
        self.sleep_compression_factor_12spikes_100ms = 0.46
        self.sleep_compression_factor_12spikes_100ms_stable_cells = 12
        self.sleep_compression_factor_12spikes_100ms_decreasing_cells = 0.8
        self.sleep_compression_factor_12spikes_100ms_increasing_cells = 0.4
        self.default_exp_fam_1_model = "mjc148R4R_0113_1_p1_46_modes"
        self.default_exp_fam_2_model = "mjc148R4R_0113_13_p1_51_modes"
        self.default_pre_phmm_model = "mjc148R4R_0113_5_p1_trials_3_14_35_modes"
        self.default_pre_phmm_model_stable = "mjc148R4R_0113_5_p1_trials_3_14_21_modes"
        self.default_pre_phmm_model_dec = "mjc148R4R_0113_5_p1_trials_3_14_39_modes"
        self.default_pre_phmm_model_inc = "mjc148R4R_0113_5_p1_trials_3_14_28_modes"
        self.default_post_phmm_model = "mjc148R4R_0113_11_p1_trials_0_15_55_modes"
        self.default_pre_ising_model = "mjc148R4R_0113_5_5cm_bins_p1_trials_3_14"
        self.default_post_ising_model = "mjc148R4R_0113_11_5cm_bins_p1_trials_0_15"
