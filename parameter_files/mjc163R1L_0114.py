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
            # where to find the data
            "data_dir": "/mnt/hdl1/01_across_regions/02_data/02_Peter",
            # "data_dir": "../02_data/02_peter",
            "session_name": "mjc163R1L_0114",
            # select cell type:
            # p1: pyramidal cells of the HPC
            "cell_type_array": ["p1", "b1"],
            # defines whether cells need to be further divided
            "cell_type_sub_div": None,
            # boundaries of start box: x_min, x_max, y_min, y_max
            "start_box_coordinates": [25, 90, 115, 170],
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
        self.goal_locations = [[177.378, 207.229], [215.393, 130.925],
                       [162.691, 64.054], [304.9, 175.994]]
        self.long_sleep_experiment_phases = ["sleep_long_1", "sleep_long_2", "sleep_long_3",
                                             "sleep_long_4", "sleep_long_5"]

        # learning cheeseboard 1
        # --------------------------------------------------------------------------------------------------------------
        self.lcb_1_nr_trials_last_5_min = range(25, 33)
        self.lcb_1_nr_trials_last_10_min = range(18, 33)
        self.lcb_1_last_10_min_phmm_modes = 55
        self.lcb_1_last_10_min_phmm_modes_old = 40
        self.lcb_1_nr_trials_learned_locations = range(18, 33)
        # which data to use by default
        self.default_trials_lcb_1 = self.lcb_1_nr_trials_last_10_min

        # learning cheeseboard 2
        # --------------------------------------------------------------------------------------------------------------
        self.lcb_2_nr_trials_first_5_min = range(0, 7)
        self.lcb_2_nr_trials_first_10_min = range(0, 17)
        self.lcb_2_first_10_min_phmm_modes = 50
        self.lcb_2_first_10_min_phmm_modes_old = 40

        # which data to use by default
        self.default_trials_lcb_2 = self.lcb_2_nr_trials_first_10_min

        # for decoding analysis
        # --------------------------------------------------------------------------------------------------------------
        self.sleep_phase_speed_threshold = 21
        # self.sleep_phase_speed_threshold = 16
        self.sleep_compression_factor_12spikes_100ms = 0.4
        self.sleep_compression_factor_12spikes_100ms_stable_cells = 12
        self.sleep_compression_factor_12spikes_100ms_decreasing_cells = 0.57
        self.sleep_compression_factor_12spikes_100ms_increasing_cells = 0.38
        self.default_exp_fam_1_model = "mjc163R1L_0114_1_p1_76_modes"
        self.default_exp_fam_2_model = "mjc163R1L_0114_14_p1_61_modes"
        self.default_pre_phmm_model = "mjc163R1L_0114_5_p1_trials_18_32_55_modes"
        self.default_pre_phmm_model_stable = "mjc163R1L_0114_5_p1_trials_18_32_26_modes"
        self.default_pre_phmm_model_dec = "mjc163R1L_0114_5_p1_trials_18_32_50_modes"
        self.default_pre_phmm_model_inc = "mjc163R1L_0114_5_p1_trials_18_32_36_modes"
        self.default_post_phmm_model = "mjc163R1L_0114_11_p1_trials_0_16_50_modes"
        self.default_pre_ising_model = "mjc163R1L_0114_5_5cm_bins_p1_trials_18_32"
        self.default_post_ising_model = "mjc163R1L_0114_11_5cm_bins_p1_trials_0_16"