########################################################################################################################
#
#   SESSIONS
#
#   Description:    - SingleSession:    class that contains all info about one session
#                                       (parameters, which method can be used etc.)
#
#                                       one session contains several experiments phases (e.g. sleep, task, exploration)
#
#                   - MultipleSessions: class that bundles multiple sessions to compute things using results from
#                                       from multiple sessions
#
#   Author: Lars Bollmann
#
#   Created: 30/04/2021
#
#   Structure:
#
########################################################################################################################

import importlib
import os
from .single_phase import Sleep, Exploration, Cheeseboard
from .multiple_phases import (LongSleep, PrePostCheeseboard, PreLongSleepPost, \
     ExplFamPrePostCheeseboardExplFam, AllData, PreProbPrePostPostProb, SleepBeforeSleep, SleepBeforePreSleep, \
     PreProbPre, TwoPopLongSleep, SleepLongSleep, CheeseboardLongSleep, ExplorationLongSleep, ExplorationCheeseboard,
                              SleepSleepLongSleep, ExplorationExplorationCheeseboard, ExplorationExploration)
from matplotlib.colors import LogNorm, PowerNorm, FuncNorm
from .load_data import LoadData
from .support_functions import moving_average, NonLinearNormalize
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import fdrcorrection
import numpy as np
import scipy
import pickle
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.stats import pearsonr, entropy, spearmanr, sem, mannwhitneyu, wilcoxon, ks_2samp, multivariate_normal, \
      zscore, f_oneway, ttest_ind, binom_test, ttest_1samp, ttest_rel
import seaborn as sns
import matplotlib

# define default location to save plots
save_path = os.path.dirname(os.path.realpath(__file__)) + "/../plots"
plt.rcParams.update({'font.size': 18})


class SingleSession:
    """class for single session"""

    def __init__(self, session_name, cell_type, params):
        """

        :param session_name: name of session (e.g. "mjc169R1R_0114")
        :type session_name: str
        :param cell_type: which cell_type to analyze (e.g. "p1" for HPC pyramidal cells)
        :type cell_type: str
        :param params: parameters for analysis
        :type params: Parameters class
        """

        # import analysis parameters that are specific for the current session
        # --------------------------------------------------------------------------------------------------------------
        session_parameter_file = importlib.import_module("parameter_files."+session_name)

        self.session_params = session_parameter_file.SessionParameters()
        self.cell_type = cell_type
        self.session_name = session_name
        self.params = params
        self.params.cell_type = cell_type

    def load_data(self, experiment_phase, data_to_use):
        """
        loads data for the current experiment phase

        :param experiment_phase: which experiment phase data to load (e.g. ["learning_cheeseboard_1"]
        :type experiment_phase: list
        :param data_to_use: which data to use ("std": only standard data, "ext": lfp data in addition)
        :type data_to_use:
        :return: data dict (pre-processed data for experimental phase)
        :rtype: dict
        """
        data_obj = LoadData(session_name=self.session_name,
                            experiment_phase=experiment_phase,
                            cell_type=self.cell_type,
                            pre_proc_dir=self.params.pre_proc_dir)
        # write experiment phase id (_1, _2, ..) and experiment_phase data to params
        self.session_params.experiment_phase_id = data_obj.get_experiment_phase_id()
        self.session_params.experiment_phase = data_obj.get_experiment_phase()

        # check whether standard or extended data (incl. lfp) needs to be used
        if data_to_use == "std":
            data_dic = data_obj.get_standard_data()
            self.params.data_to_use = "std"
        elif data_to_use == "ext":
            data_dic = data_obj.get_extended_data()
            self.params.data_to_use = "ext"
        elif data_to_use == "ext_eegh":
            data_dic = data_obj.get_extended_data(which_file="eegh")
            self.params.data_to_use = "ext"
        else:
            raise Exception("Either use std or ext data")
        return data_dic

    def load_data_object(self, experiment_phase):
        """
        loads data object (LoadData object)

        :param experiment_phase: which experiment phase data to load (e.g. ["learning_cheeseboard_1"]
        :type experiment_phase: list
        :return: data_obj -- LoadData object
        :rtype: class
        """
        data_obj = LoadData(session_name=self.session_name,
                            experiment_phase=experiment_phase,
                            cell_type=self.cell_type,
                            pre_proc_dir=self.params.pre_proc_dir)
        # write experiment phase id (_1, _2, ..) and experiment_phase data to params
        self.session_params.experiment_phase_id = data_obj.get_experiment_phase_id()
        self.session_params.experiment_phase = data_obj.get_experiment_phase()

        return data_obj

    # <editor-fold desc="Single experiment phases">

    def sleep(self, experiment_phase, data_to_use="std"):
        """
        Loads data and returns sleep object

        :param experiment_phase: which sleep to use (e.g. "sleep_1")
        :type experiment_phase: list
        :param data_to_use: "std" for normal data or "ext" to include lfp
        :type data_to_use: str
        :return: Sleep object
        :rtype: class
        """
        data_dic = self.load_data(experiment_phase=experiment_phase, data_to_use=data_to_use)

        return Sleep(data_dic=data_dic, cell_type=self.cell_type, params=self.params,
                     session_params=self.session_params, experiment_phase=experiment_phase)

    def exploration(self, experiment_phase, data_to_use="std"):
        """
        Loads data and returns exploration object

        :param experiment_phase: which exploration to use (e.g. "exploration_familiar")
        :type experiment_phase: list
        :param data_to_use: "std" for normal data or "ext" to include lfp
        :type data_to_use: str
        :return: Exploration object
        :rtype: class
        """

        data_dic = self.load_data(experiment_phase=experiment_phase, data_to_use=data_to_use)

        return Exploration(data_dic=data_dic, cell_type=self.cell_type, params=self.params,
                           session_params=self.session_params, experiment_phase=experiment_phase)

    def cheese_board(self, experiment_phase, data_to_use="std"):
        """
        Loads data and returns cheeseboard object

        :param experiment_phase: which cheeseboard phase to use (e.g. "learning_cheeseboard_1")
        :type experiment_phase: list
        :param data_to_use: "std" for normal data or "ext" to include lfp
        :type data_to_use: str
        :return: Exploration object
        :rtype: class
        """

        data_dic = self.load_data(experiment_phase=experiment_phase, data_to_use=data_to_use)

        return Cheeseboard(data_dic=data_dic, cell_type=self.cell_type, params=self.params,
                           session_params=self.session_params, experiment_phase=experiment_phase)

    # </editor-fold>

    # <editor-fold desc="Multiple experiment phases">

    def exp_fam_pre_post_exp_fam(self):
        """
        Loads data and returns ExplFamPrePostCheeseboardExplFam

        :return: ExplFamPrePostCheeseboardExplFam object
        :rtype: class
        """

        exp_fam_1 = self.exploration(experiment_phase=["exploration_familiar"])
        pre = self.cheese_board(experiment_phase=["learning_cheeseboard_1"])
        post = self.cheese_board(experiment_phase=["learning_cheeseboard_2"])
        exp_fam_2 = self.exploration(experiment_phase=["exploration_familiar_post"])

        return ExplFamPrePostCheeseboardExplFam(exp_fam_1=exp_fam_1, pre=pre, post=post, exp_fam_2=exp_fam_2,
                                                params=self.params, session_params=self.session_params)

    def pre_prob_pre_post_post_prob(self):
        """
        Loads data and returns PreProbPrePostPostProb

        :return: PreProbPrePostPostProb object
        :rtype: class
        """

        pre_probe = self.exploration(experiment_phase=["exploration_cheeseboard"])
        pre = self.cheese_board(experiment_phase=["learning_cheeseboard_1"])
        post = self.cheese_board(experiment_phase=["learning_cheeseboard_2"])
        try:
            post_probe = self.exploration(experiment_phase=["post_probe"])
        except:
            post_probe = None
            print(self.session_name+": post_probe not found!")

        return PreProbPrePostPostProb(pre_probe=pre_probe, pre=pre, post=post, post_probe=post_probe,
                                      params=self.params, session_params=self.session_params)

    def long_sleep(self, data_to_use="std", subset_of_sleep=None):
        """
        Loads data and returns LongSleep object (concatenates single sleep files)

        :param data_to_use: "std" for normal data or "ext" to include lfp
        :type data_to_use: str
        :param subset_of_sleep: only use a single sleep file (for debugging)
        :type subset_of_sleep: int
        :return: LongSleep object
        :rtype: class
        """

        # get all experiment phases that define long sleep
        long_sleep_exp_phases = self.session_params.long_sleep_experiment_phases
        if subset_of_sleep is not None:
            long_sleep_exp_phases = [long_sleep_exp_phases[subset_of_sleep]]
        data_obj = self.load_data_object(experiment_phase=long_sleep_exp_phases)
        self.params.data_to_use = data_to_use
        return LongSleep(sleep_data_obj=data_obj, params=self.params, session_params=self.session_params)

    def sleep_long_sleep(self, data_to_use="std", subset_of_sleep=None,
                         sleep_experiment_phase=["sleep_cheeseboard_1"]):
        """
        Loads data and returns SleepLongSleep object (concatenates single sleep files + sleep)

        :param data_to_use: "std" for normal data or "ext" to include lfp
        :type data_to_use: str
        :param subset_of_sleep: only use a single sleep file (for debugging)
        :type subset_of_sleep: int
        :return: LongSleep object
        :rtype: class
        """

        # get all experiment phases that define long sleep
        long_sleep_exp_phases = self.session_params.long_sleep_experiment_phases
        if subset_of_sleep is not None:
            long_sleep_exp_phases = [long_sleep_exp_phases[subset_of_sleep]]
        data_obj = self.load_data_object(experiment_phase=long_sleep_exp_phases)

        data_dic = self.load_data(experiment_phase=sleep_experiment_phase, data_to_use=data_to_use)

        sleep = Sleep(data_dic=data_dic, cell_type=self.cell_type, params=self.params,
                      session_params=self.session_params, experiment_phase=sleep_experiment_phase)

        # don't want to load lfp data for long sleep
        self.params.data_to_use = "std"

        return SleepLongSleep(sleep_data_obj=data_obj, params=self.params, session_params=self.session_params,
                              sleep=sleep)
    def sleep_sleep_long_sleep(self, data_to_use="std", subset_of_sleep=None,
                         sleep_1_experiment_phase=["sleep_familiar"],
                         sleep_2_experiment_phase=["sleep_cheeseboard_1"]):
        """
        Loads data and returns SleepLongSleep object (concatenates single sleep files + sleep + sleep)

        :param data_to_use: "std" for normal data or "ext" to include lfp
        :type data_to_use: str
        :param subset_of_sleep: only use a single sleep file (for debugging)
        :type subset_of_sleep: int
        :return: LongSleep object
        :rtype: class
        """

        # get all experiment phases that define long sleep
        long_sleep_exp_phases = self.session_params.long_sleep_experiment_phases
        if subset_of_sleep is not None:
            long_sleep_exp_phases = [long_sleep_exp_phases[subset_of_sleep]]
        data_obj = self.load_data_object(experiment_phase=long_sleep_exp_phases)

        data_dic_sleep_1 = self.load_data(experiment_phase=sleep_1_experiment_phase, data_to_use=data_to_use)

        sleep_1 = Sleep(data_dic=data_dic_sleep_1, cell_type=self.cell_type, params=self.params,
                      session_params=self.session_params, experiment_phase=sleep_1_experiment_phase)

        data_dic_sleep_2 = self.load_data(experiment_phase=sleep_2_experiment_phase, data_to_use=data_to_use)

        sleep_2 = Sleep(data_dic=data_dic_sleep_2, cell_type=self.cell_type, params=self.params,
                        session_params=self.session_params, experiment_phase=sleep_2_experiment_phase)

        # don't want to load lfp data for long sleep
        self.params.data_to_use = "std"

        return SleepSleepLongSleep(sleep_data_obj=data_obj, params=self.params, session_params=self.session_params,
                              sleep_1=sleep_1, sleep_2=sleep_2)

    def cheese_board_long_sleep(self, data_to_use="std", subset_of_sleep=None,
                                cheeseboard_experiment_phase=["learning_cheeseboard_1"]):
        """
        Loads data and returns CheeseboardLongSleep object (concatenates single sleep files + cheeseboard)

        :param data_to_use: "std" for normal data or "ext" to include lfp
        :type data_to_use: str
        :param subset_of_sleep: only use a single sleep file (for debugging)
        :type subset_of_sleep: int
        :return: LongSleep object
        :rtype: class
        """

        # get all experiment phases that define long sleep
        long_sleep_exp_phases = self.session_params.long_sleep_experiment_phases
        if subset_of_sleep is not None:
            long_sleep_exp_phases = [long_sleep_exp_phases[subset_of_sleep]]
        data_obj = self.load_data_object(experiment_phase=long_sleep_exp_phases)
        self.params.data_to_use = data_to_use

        data_dic = self.load_data(experiment_phase=cheeseboard_experiment_phase, data_to_use=data_to_use)

        cb = Cheeseboard(data_dic=data_dic, cell_type=self.cell_type, params=self.params,
                           session_params=self.session_params, experiment_phase=cheeseboard_experiment_phase)
        self.params.data_to_use = data_to_use
        return CheeseboardLongSleep(sleep_data_obj=data_obj, params=self.params, session_params=self.session_params,
                              cheeseboard=cb)

    def exploration_long_sleep(self, data_to_use="std", subset_of_sleep=None,
                                exploration_experiment_phase=["exploration_familiar"]):
        """
        Loads data and returns CheeseboardLongSleep object (concatenates single sleep files + cheeseboard)

        :param data_to_use: "std" for normal data or "ext" to include lfp
        :type data_to_use: str
        :param subset_of_sleep: only use a single sleep file (for debugging)
        :type subset_of_sleep: int
        :return: LongSleep object
        :rtype: class
        """

        # get all experiment phases that define long sleep
        long_sleep_exp_phases = self.session_params.long_sleep_experiment_phases
        if subset_of_sleep is not None:
            long_sleep_exp_phases = [long_sleep_exp_phases[subset_of_sleep]]
        data_obj = self.load_data_object(experiment_phase=long_sleep_exp_phases)
        self.params.data_to_use = data_to_use

        data_dic = self.load_data(experiment_phase=exploration_experiment_phase, data_to_use=data_to_use)

        exploration = Exploration(data_dic=data_dic, cell_type=self.cell_type, params=self.params,
                           session_params=self.session_params, experiment_phase=exploration_experiment_phase)

        return ExplorationLongSleep(sleep_data_obj=data_obj, params=self.params, session_params=self.session_params,
                              exploration=exploration)

    def exploration_cheeseboard(self, data_to_use="std", cheeseboard_experiment_phase=["learning_cheeseboard_1"],
                                exploration_experiment_phase=["exploration_familiar"]):
        """
        Loads data and returns CheeseboardLongSleep object (concatenates single sleep files + cheeseboard)

        :param data_to_use: "std" for normal data or "ext" to include lfp
        :type data_to_use: str
        :param subset_of_sleep: only use a single sleep file (for debugging)
        :type subset_of_sleep: int
        :return: LongSleep object
        :rtype: class
        """

        data_dic_ex = self.load_data(experiment_phase=exploration_experiment_phase, data_to_use=data_to_use)

        exploration = Exploration(data_dic=data_dic_ex, cell_type=self.cell_type, params=self.params,
                           session_params=self.session_params, experiment_phase=exploration_experiment_phase)

        data_dic_cb = self.load_data(experiment_phase=cheeseboard_experiment_phase, data_to_use=data_to_use)

        cheeseboard =  Cheeseboard(data_dic=data_dic_cb, cell_type=self.cell_type, params=self.params,
                           session_params=self.session_params, experiment_phase=cheeseboard_experiment_phase)

        return ExplorationCheeseboard(cheeseboard=cheeseboard, params=self.params, session_params=self.session_params,
                              exploration=exploration)

    def pre_post(self):
        """
        Loads data and returns PrePostCheeseboard object

        :return: PrePostCheeseboard object
        :rtype: class
        """

        pre = self.cheese_board(experiment_phase=["learning_cheeseboard_1"])
        post = self.cheese_board(experiment_phase=["learning_cheeseboard_2"])

        return PrePostCheeseboard(pre=pre, post=post, params=self.params,
                                  session_params=self.session_params)

    def exploration_exploration_cheeseboard(self, expl_phase_1 =["exploration_familiar"],
                                            expl_phase_2=["exploration_cheeseboard"]):
        """
        Loads data and returns PrePostCheeseboard object plus exploration

        :return: PrePostCheeseboard object
        :rtype: class
        """

        cb = self.cheese_board(experiment_phase=["learning_cheeseboard_1"])
        exp1 = self.exploration(experiment_phase=expl_phase_1)
        exp2 = self.exploration(experiment_phase=expl_phase_2)

        return ExplorationExplorationCheeseboard(exp1=exp1, exp2=exp2, cb=cb, params=self.params,
                                  session_params=self.session_params)

    def exploration_exploration(self, expl_phase_1 =["exploration_familiar"],
                                            expl_phase_2=["exploration_familiar_post"]):
        """
        Loads data and returns PrePostCheeseboard object plus exploration

        :return: PrePostCheeseboard object
        :rtype: class
        """

        exp1 = self.exploration(experiment_phase=expl_phase_1)
        exp2 = self.exploration(experiment_phase=expl_phase_2)

        return ExplorationExploration(exp1=exp1, exp2=exp2, params=self.params,
                                  session_params=self.session_params)

    def sleep_before_sleep(self, data_to_use="std"):
        """
        Loads data and returns SleepBeforeSleep

        :param data_to_use: "std" for normal data or "ext" to include lfp
        :type data_to_use: str
        :return: SleepBeforeSleep object
        :rtype: class
        """

        sleep_before = self.sleep(experiment_phase=["sleep_cheeseboard_1"], data_to_use=data_to_use)
        sleep = self.sleep(experiment_phase=["sleep_long_1"], data_to_use=data_to_use)

        return SleepBeforeSleep(sleep_before=sleep_before, sleep=sleep, params=self.params,
                                session_params=self.session_params)

    def sleep_before_pre_sleep(self, data_to_use="std"):
        """
        Loads data and returns SleepBeforePreSleep

        :param data_to_use: "std" for normal data or "ext" to include lfp
        :type data_to_use: str
        :return: SleepBeforePreSleep object
        :rtype: class
        """

        sleep_before = self.sleep(experiment_phase=["sleep_cheeseboard_1"], data_to_use=data_to_use)
        sleep = self.sleep(experiment_phase=["sleep_long_1"], data_to_use=data_to_use)
        pre = self.cheese_board(experiment_phase=["learning_cheeseboard_1"])

        return SleepBeforePreSleep(sleep_before=sleep_before, sleep=sleep, pre=pre, params=self.params,
                                   session_params=self.session_params)

    def pre_prob_pre(self):
        """
        Loads data and returns PreProbPre

        :return: PreProbPre object
        :rtype: class
        """

        pre_probe = self.exploration(experiment_phase=["exploration_cheeseboard"])
        pre = self.cheese_board(experiment_phase=["learning_cheeseboard_1"])

        return PreProbPre(pre_probe=pre_probe, pre=pre, params=self.params, session_params=self.session_params)

    def pre_long_sleep_post(self, data_to_use_cb="std", data_to_use_long_sleep="std"):
        """
        Loads data and returns PreLongSleepPost

        :return: PreLongSleepPost object
        :rtype: class
        """

        # get pre and post phase
        pre = self.cheese_board(experiment_phase=["learning_cheeseboard_1"], data_to_use=data_to_use_cb)
        post = self.cheese_board(experiment_phase=["learning_cheeseboard_2"], data_to_use=data_to_use_cb)
        # get all experiment phases that define long sleep
        long_sleep_exp_phases = self.session_params.long_sleep_experiment_phases
        sleep_data_obj = self.load_data_object(experiment_phase=long_sleep_exp_phases)
        self.params.data_to_use = data_to_use_long_sleep
        self.params.session_name = self.session_name
        return PreLongSleepPost(sleep_data_obj=sleep_data_obj, pre=pre, post=post, params=self.params,
                                session_params=self.session_params)

    def all_data(self):
        """
        Loads data and returns AllData (does not separate into experimental phases)

        :return: AllData object
        :rtype: class
        """

        return AllData(params=self.params, session_params=self.session_params)

    # </editor-fold>


class TwoPopSingleSession:
    """class for single session"""

    def __init__(self, session_name, cell_type_1, cell_type_2, params):
        # --------------------------------------------------------------------------------------------------------------
        # args: - session_name, str: name of session
        # --------------------------------------------------------------------------------------------------------------

        # import analysis parameters that are specific for the current session
        # --------------------------------------------------------------------------------------------------------------
        session_parameter_file = importlib.import_module("parameter_files." + session_name)

        self.session_params = session_parameter_file.SessionParameters()

        self.cell_type_1 = cell_type_1
        self.cell_type_2 = cell_type_2
        self.session_name = session_name
        self.params = params
        self.params.cell_type_1 = cell_type_1
        self.params.cell_type_2 = cell_type_2

    def load_data(self, experiment_phase, data_to_use):
        data_obj = LoadData(session_name=self.session_name,
                            experiment_phase=experiment_phase,
                            pre_proc_dir=self.params.pre_proc_dir)
        # write experiment phase id (_1, _2, ..) and experiment_phase data to params
        self.session_params.experiment_phase_id = data_obj.get_experiment_phase_id()
        self.session_params.experiment_phase = data_obj.get_experiment_phase()

        # check whether standard or extended data (incl. lfp) needs to be used
        if data_to_use == "std":
            data_dic = data_obj.get_standard_data()
            self.params.data_to_use = "std"
        elif data_to_use == "ext":
            data_dic = data_obj.get_extended_data()
            self.params.data_to_use = "ext"
        return data_dic

    def load_data_object(self, experiment_phase, cell_type):

        data_obj = LoadData(session_name=self.session_name,
                            experiment_phase=experiment_phase,
                            pre_proc_dir=self.params.pre_proc_dir, cell_type=cell_type)
        # write experiment phase id (_1, _2, ..) and experiment_phase data to params
        self.session_params.experiment_phase_id = data_obj.get_experiment_phase_id()
        self.session_params.experiment_phase = data_obj.get_experiment_phase()

        return data_obj

    """#################################################################################################################
    #  analyzing single experiment phase for two population
    #################################################################################################################"""

    def long_sleep(self, data_to_use="std"):

        long_sleep_exp_phases = self.session_params.long_sleep_experiment_phases
        # if subset_of_sleep is not None:
        #     long_sleep_exp_phases = [long_sleep_exp_phases[subset_of_sleep]]
        # load data object for first cell type

        self.params.data_to_use = data_to_use

        data_obj_1 = self.load_data_object(experiment_phase=long_sleep_exp_phases, cell_type=self.cell_type_1)
        data_obj_2 = self.load_data_object(experiment_phase=long_sleep_exp_phases, cell_type=self.cell_type_2)

        ls_pop_1 = LongSleep(sleep_data_obj=data_obj_1, params=self.params, session_params=self.session_params)
        ls_pop_2 = LongSleep(sleep_data_obj=data_obj_2, params=self.params, session_params=self.session_params)

        return TwoPopLongSleep(long_sleep_pop_1=ls_pop_1, long_sleep_pop_2=ls_pop_2,
                           params=self.params, session_params=self.session_params)


class TwoPopMultipleSessions:
    """class for multiple sessions"""

    def __init__(self, session_names, cell_type_1, cell_type_2, params):
        """
        Initializes multiple sessions

        :param session_names: name of sessions (e.g. ["mjc163R4R_0114", "mjc163R2R_0114"])
        :type session_names: list
        :param cell_type: which cell_type to analyze (e.g. "p1" for HPC pyramidal cells)
        :type cell_type: str
        :param params: parameters for analysis
        :type params: Parameters class
        """
        self.params = params

        # initialize all sessions
        self.session_list = []
        for session_name in session_names:
            self.session_list.append(TwoPopSingleSession(session_name=session_name, cell_type_1=cell_type_1,
                                                         cell_type_2=cell_type_2, params=params))

    def memory_drift_delta_score_interneuron_firing(self, use_abs_delta_score=False, save_fig=False,
                                                    invert_rem_sign=False, control_mul_comp=False, plotting=False):
        """
        Computes distribution of pearson r (interneuron firing, delta score) for all sessions

        :param use_abs_delta_score: whether to use absolute (True) or normal delta score
        :type use_abs_delta_score: bool
        """
        corr_ds_mean_nrem = []
        corr_ds_mean_rem = []
        corr_ds_mean_nrem_rem = []
        corr_ds_nrem_delta_fir = []
        corr_ds_rem_delta_fir = []
        p_nrem = []
        p_rem = []
        p_nrem_rem = []
        p_nrem_delta_fir = []
        p_rem_delta_fir = []

        for session in self.session_list:
            c_nrem, c_rem, c_nrem_rem, p_nrem_, p_rem_, p_nrem_rem_ = \
                session.long_sleep().memory_drift_delta_score_interneuron_firing(return_p_values=True, plotting=False,
                                                                                use_abs_delta_score=use_abs_delta_score,
                                                                                 invert_rem_sign=invert_rem_sign)
            c_nrem_delta, c_rem_delta, _, p_nrem_delta, p_rem_delta,_ = \
                session.long_sleep().memory_drift_delta_score_delta_interneuron_firing(return_p_values=True,
                                                                                       plotting=False,
                                                                                       use_abs_delta_score=
                                                                                       use_abs_delta_score)

            corr_ds_mean_nrem.append(c_nrem)
            corr_ds_mean_rem.append(c_rem)
            corr_ds_mean_nrem_rem.append(c_nrem_rem)
            p_nrem.append(p_nrem_)
            p_rem.append(p_rem_)
            p_nrem_rem.append(p_nrem_rem_)
            corr_ds_nrem_delta_fir.append(c_nrem_delta)
            corr_ds_rem_delta_fir.append(c_rem_delta)
            p_nrem_delta_fir.append(p_nrem_delta)
            p_rem_delta_fir.append(p_rem_delta)

        corr_ds_mean_nrem = np.hstack(corr_ds_mean_nrem)
        corr_ds_mean_rem = np.hstack(corr_ds_mean_rem)
        corr_ds_mean_nrem_rem = np.hstack(corr_ds_mean_nrem_rem)
        p_nrem = np.hstack(p_nrem)
        p_rem = np.hstack(p_rem)
        p_nrem_rem = np.hstack(p_nrem_rem)
        corr_ds_nrem_delta_fir = np.hstack(corr_ds_nrem_delta_fir)
        corr_ds_rem_delta_fir = np.hstack(corr_ds_rem_delta_fir)
        p_nrem_delta_fir = np.hstack(p_nrem_delta_fir)
        p_rem_delta_fir = np.hstack(p_rem_delta_fir)

        if plotting or save_fig:
            # correlation with mean firing
            # --------------------------------------------------------------------------------------------------------------
            if control_mul_comp:
                corr_ds_mean_nrem_sign = corr_ds_mean_nrem[p_nrem < (0.05/corr_ds_mean_nrem.shape[0])]
            else:
                corr_ds_mean_nrem_sign = corr_ds_mean_nrem[p_nrem<0.05]

            # run binomial test
            print("NREM:")
            print(scipy.stats.binom_test(corr_ds_mean_nrem_sign.shape[0], corr_ds_mean_nrem.shape[0], alternative="greater"))

            nrem_perc_sign = corr_ds_mean_nrem_sign.shape[0]/corr_ds_mean_nrem.shape[0]
            if save_fig:
                plt.style.use('default')
            plt.figure(figsize=(4,4))
            plt.hist(corr_ds_mean_nrem, bins=18, color="lightblue", label="All interneurons")
            plt.hist(corr_ds_mean_nrem_sign, bins=18, color="blue", alpha=0.7,
                     label="Sign. correlations ("+str(np.round(nrem_perc_sign*100,2))+"%)")
            plt.ylabel("#Interneurons")
            if use_abs_delta_score:
                plt.xlabel("Pearson R (Interneuron firing rate, abs. delta_score)")
            else:
                plt.xlabel("Pearson R (Interneuron firing rate, delta_score)")
            # plt.text(-0.25, 2, str(np.round(nrem_perc_sign*100,2))+"% significant")
            plt.xlim(-0.6, 0.6)
            plt.ylim(0,19)
            plt.legend(loc=2)
            plt.title("NREM")
            plt.tight_layout()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "corr_interneuron_fir_delta_score_nrem.svg"), transparent="True")
                plt.close()
            else:
                plt.show()

            if control_mul_comp:
                corr_ds_mean_rem_sign = corr_ds_mean_rem[p_rem < (0.05/corr_ds_mean_rem.shape[0])]
            else:
                corr_ds_mean_rem_sign = corr_ds_mean_rem[p_rem<0.05]
            rem_perc_sign = corr_ds_mean_rem_sign.shape[0]/corr_ds_mean_rem.shape[0]

            # run binomial test
            print(scipy.stats.binom_test(corr_ds_mean_rem_sign.shape[0], corr_ds_mean_rem.shape[0], alternative="greater"))

            if save_fig:
                plt.style.use('default')
            plt.figure(figsize=(4,4))
            plt.hist(corr_ds_mean_rem, color="salmon", bins=18, label="All interneurons")
            plt.hist(corr_ds_mean_rem_sign, color="red", bins=18, alpha=0.7, label="Sign. correlations ("+str(np.round(rem_perc_sign*100,2))+"%)")
            # plt.text(-0.3, 2, str(np.round(rem_perc_sign*100,2))+"% significant")
            plt.ylabel("#Interneurons")
            if use_abs_delta_score:
                plt.xlabel("Pearson R (Interneuron firing rate, abs. delta_score)")
            else:
                plt.xlabel("Pearson R (Interneuron firing rate, delta_score)")
            plt.xlim(-0.6, 0.6)
            plt.ylim(0, 24)
            plt.legend(loc=2)
            plt.title("REM")
            plt.tight_layout()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "corr_interneuron_fir_delta_score_rem.svg"), transparent="True")
                plt.close()
            else:
                plt.show()

            if control_mul_comp:
                corr_ds_mean_nrem_rem_sign = corr_ds_mean_nrem_rem[p_nrem_rem < (0.05/corr_ds_mean_nrem_rem.shape[0])]
            else:
                corr_ds_mean_nrem_rem_sign = corr_ds_mean_nrem_rem[p_nrem_rem<0.05]
            nrem_rem_perc_sign = corr_ds_mean_nrem_rem_sign.shape[0]/corr_ds_mean_nrem_rem.shape[0]

            if save_fig:
                plt.style.use('default')
            plt.figure(figsize=(4,4))
            plt.hist(corr_ds_mean_nrem_rem, color="lightgrey", bins=25, label="All interneurons")
            plt.hist(corr_ds_mean_nrem_rem_sign, color="dimgray", bins=25, alpha=0.7,
                     label="Sign. correlations ("+str(np.round(nrem_rem_perc_sign*100,2))+"%)")
            # plt.text(-0.3, 2, str(np.round(nrem_rem_perc_sign*100,2))+"% significant")
            plt.ylabel("#Interneurons")
            plt.title("REM & NREM")
            if use_abs_delta_score:
                plt.xlabel("Pearson R (Interneuron firing rate, abs. delta_score)")
            else:
                plt.xlabel("Pearson R (Interneuron firing rate, delta_score)")
            # plt.xlim(-0.45, 0.45)
            plt.xlim(-0.6, 0.6)
            plt.ylim(0,26)
            plt.legend(loc=2)
            plt.tight_layout()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "corr_interneuron_fir_delta_score_rem_nrem_dist.svg"), transparent="True")
                plt.close()
            else:
                plt.show()

            plt.figure(figsize=(4,4))
            plt.scatter(corr_ds_mean_rem, corr_ds_mean_nrem, color="gray", edgecolors="white")
            plt.text(-0.2, -0.2, "R="+str(np.round(pearsonr(corr_ds_mean_rem, corr_ds_mean_nrem)[0],2)))
            plt.text(-0.2, -0.25, "p=" + str(np.round(pearsonr(corr_ds_mean_rem, corr_ds_mean_nrem)[1], 6)))
            plt.xlabel("Pearson R (Interneuron firing rate,\n abs. delta_score) REM")
            plt.ylabel("Pearson R (Interneuron firing rate,\n abs. delta_score) NREM")
            plt.tight_layout()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "corr_interneuron_fir_delta_score_rem_vs_nrem.svg"), transparent="True")
                plt.close()
            else:
                plt.show()

            # correlation with delta firing
            # --------------------------------------------------------------------------------------------------------------
            corr_ds_nrem_delta_fir_sign = corr_ds_nrem_delta_fir[p_nrem_delta_fir<0.05]
            nrem_delta_fir_perc_sign = corr_ds_nrem_delta_fir_sign.shape[0]/corr_ds_nrem_delta_fir.shape[0]
            if save_fig:
                plt.style.use('default')
            plt.hist(corr_ds_nrem_delta_fir_sign, density=True, bins=10, color="lightblue")
            plt.ylabel("Density")
            plt.xlabel("Pearson R (Delta Interneuron firing, delta_score)")
            plt.text(-0.25, 2, str(np.round(nrem_delta_fir_perc_sign*100,2))+"% significant")
            plt.xlim(-0.4, 0.4)
            plt.tight_layout()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "corr_interneuron_delta_fir_delta_score_nrem.svg"), transparent="True")
                plt.close()
            else:
                plt.show()

            corr_ds_rem_delta_fir_sign = corr_ds_rem_delta_fir[p_rem_delta_fir<0.05]
            rem_delta_fir_perc_sign = corr_ds_rem_delta_fir_sign.shape[0]/corr_ds_rem_delta_fir.shape[0]

            if save_fig:
                plt.style.use('default')
            plt.hist(corr_ds_rem_delta_fir, density=True, color="salmon", bins=10)
            plt.text(-0.3, 4, str(np.round(rem_delta_fir_perc_sign*100,2))+"% significant")
            plt.ylabel("Density")
            plt.xlabel("Pearson R (Delta Interneuron firing, delta_score)")
            plt.xlim(-0.5, 0.5)
            plt.tight_layout()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "corr_interneuron_delta_fir_delta_score_rem.svg"), transparent="True")
                plt.close()
            else:
                plt.show()

            slope, intercept, r, p, stderr = scipy.stats.linregress(corr_ds_mean_rem, corr_ds_mean_nrem)

            if save_fig:
                line_col = "black"
                plt.style.use('default')
            else:
                line_col = "white"
            plt.figure(figsize=(4,4))
            plt.plot(corr_ds_mean_rem, intercept + slope * corr_ds_mean_rem, color=line_col)
            plt.scatter(corr_ds_mean_rem, corr_ds_mean_nrem, color="gray", edgecolor="white")
            plt.text(-0.4, -0.2, "R="+str(np.round(pearsonr(corr_ds_mean_rem, corr_ds_mean_nrem)[0],2)))
            plt.text(-0.4, -0.25, "p=" + str(np.round(pearsonr(corr_ds_mean_rem, corr_ds_mean_nrem)[1], 6)))
            plt.xlabel("Pearson R (Interneuron firing rate, delta_score) REM")
            plt.ylabel("Pearson R (Interneuron firing rate, delta_score) NREM")
            plt.xlim(-0.5, 0.3)
            plt.ylim(-0.5, 0.3)
            plt.xticks([-0.4, -0.2, 0.0, 0.2])
            plt.yticks([-0.4, -0.2, 0.0, 0.2])
            plt.gca().set_aspect("equal")
            plt.tight_layout()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "corr_interneuron_fir_delta_score_rem_nrem.svg"),
                            transparent="True")
                plt.close()
            else:
                plt.show()

        return corr_ds_mean_nrem_rem

    def memory_drift_delta_score_delta_interneuron_firing(self, use_abs_delta_score=False, save_fig=False,
                                                    invert_rem_sign=False, control_mul_comp=False, plotting=False):
        """
        Computes distribution of pearson r (interneuron firing, delta score) for all sessions

        :param use_abs_delta_score: whether to use absolute (True) or normal delta score
        :type use_abs_delta_score: bool
        """
        corr_ds_mean_nrem = []
        corr_ds_mean_rem = []
        corr_ds_mean_nrem_rem = []
        p_rem = []
        p_nrem = []
        p_rem_nrem = []

        for session in self.session_list:
            c_nrem, c_rem, c_nrem_rem, p_nrem_, p_rem_, p_rem_nrem_ = \
                session.long_sleep().memory_drift_delta_score_delta_interneuron_firing(return_p_values=True)

            corr_ds_mean_nrem.append(c_nrem)
            corr_ds_mean_rem.append(c_rem)
            corr_ds_mean_nrem_rem.append(c_nrem_rem)
            p_nrem.append(p_nrem_)
            p_rem.append(p_rem_)
            p_rem_nrem.append(p_rem_nrem_)

        corr_ds_mean_nrem = np.hstack(corr_ds_mean_nrem)
        corr_ds_mean_rem = np.hstack(corr_ds_mean_rem)
        corr_ds_mean_nrem_rem = np.hstack(corr_ds_mean_nrem_rem)
        p_nrem = np.hstack(p_nrem)
        p_rem = np.hstack(p_rem)
        p_rem_nrem = np.hstack(p_rem_nrem)

        if plotting or save_fig:
            # correlation with mean firing
            # --------------------------------------------------------------------------------------------------------------
            if control_mul_comp:
                corr_ds_mean_nrem_sign = corr_ds_mean_nrem[p_nrem < (0.05/corr_ds_mean_nrem.shape[0])]
            else:
                corr_ds_mean_nrem_sign = corr_ds_mean_nrem[p_nrem<0.05]

            # run binomial test
            print("NREM:")
            print(scipy.stats.binom_test(corr_ds_mean_nrem_sign.shape[0], corr_ds_mean_nrem.shape[0], alternative="greater"))

            nrem_perc_sign = corr_ds_mean_nrem_sign.shape[0]/corr_ds_mean_nrem.shape[0]
            if save_fig:
                plt.style.use('default')
            plt.figure(figsize=(4,4))
            plt.hist(corr_ds_mean_nrem, bins=12, color="lightblue", label="All interneurons")
            plt.hist(corr_ds_mean_nrem_sign, bins=4, color="blue", alpha=0.7,
                     label="Sign. correlations ("+str(np.round(nrem_perc_sign*100,2))+"%)")
            plt.ylabel("#Interneurons")
            if use_abs_delta_score:
                plt.xlabel("Pearson R (Delta interneuron firing rate, abs. delta_score)")
            else:
                plt.xlabel("Pearson R (Delta interneuron firing rate, delta_score)")
            # plt.text(-0.25, 2, str(np.round(nrem_perc_sign*100,2))+"% significant")
            plt.xlim(-0.6, 0.6)
            plt.ylim(0,25)
            plt.legend(loc=2)
            plt.title("NREM")
            plt.tight_layout()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "corr_delta_interneuron_fir_delta_score_nrem.svg"), transparent="True")
                plt.close()
            else:
                plt.show()

            if control_mul_comp:
                corr_ds_mean_rem_sign = corr_ds_mean_rem[p_rem < (0.05/corr_ds_mean_rem.shape[0])]
            else:
                corr_ds_mean_rem_sign = corr_ds_mean_rem[p_rem<0.05]
            rem_perc_sign = corr_ds_mean_rem_sign.shape[0]/corr_ds_mean_rem.shape[0]

            # run binomial test
            print(scipy.stats.binom_test(corr_ds_mean_rem_sign.shape[0], corr_ds_mean_rem.shape[0], alternative="greater"))

            if save_fig:
                plt.style.use('default')
            plt.figure(figsize=(4,4))
            plt.hist(corr_ds_mean_rem, color="salmon", bins=12, label="All interneurons")
            plt.hist(corr_ds_mean_rem_sign, color="red", bins=6, alpha=0.7, label="Sign. correlations ("+str(np.round(rem_perc_sign*100,2))+"%)")
            # plt.text(-0.3, 2, str(np.round(rem_perc_sign*100,2))+"% significant")
            plt.ylabel("#Interneurons")
            if use_abs_delta_score:
                plt.xlabel("Pearson R (Interneuron firing rate, abs. delta_score)")
            else:
                plt.xlabel("Pearson R (Delta interneuron firing rate, delta_score)")
            plt.xlim(-0.6, 0.6)
            plt.ylim(0, 28)
            plt.legend(loc=2)
            plt.title("REM")
            plt.tight_layout()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "corr_delta_interneuron_fir_delta_score_rem.svg"), transparent="True")
                plt.close()
            else:
                plt.show()

            if control_mul_comp:
                corr_ds_mean_nrem_rem_sign = corr_ds_mean_nrem_rem[p_rem_nrem < (0.05/corr_ds_mean_nrem_rem.shape[0])]
            else:
                corr_ds_mean_nrem_rem_sign = corr_ds_mean_nrem_rem[p_rem_nrem<0.05]
            nrem_rem_perc_sign = corr_ds_mean_nrem_rem_sign.shape[0]/corr_ds_mean_nrem_rem.shape[0]

            if save_fig:
                plt.style.use('default')
            plt.figure(figsize=(4,4))
            plt.hist(corr_ds_mean_nrem_rem, color="lightgrey", bins=18, label="All interneurons")
            plt.hist(corr_ds_mean_nrem_rem_sign, color="dimgray", bins=13, alpha=0.7,
                     label="Sign. correlations ("+str(np.round(nrem_rem_perc_sign*100,2))+"%)")
            # plt.text(-0.3, 2, str(np.round(nrem_rem_perc_sign*100,2))+"% significant")
            plt.ylabel("#Interneurons")
            plt.title("REM & NREM")
            if use_abs_delta_score:
                plt.xlabel("Pearson R (Delta interneuron firing rate, abs. delta_score)")
            else:
                plt.xlabel("Pearson R (Delta interneuron firing rate, delta_score)")
            # plt.xlim(-0.45, 0.45)
            plt.xlim(-0.6, 0.6)
            plt.ylim(0,37)
            plt.legend(loc=2)
            plt.tight_layout()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "corr_delta_interneuron_fir_delta_score_rem_nrem_dist.svg"), transparent="True")
                plt.close()
            else:
                plt.show()

        return corr_ds_mean_nrem_rem

    def memory_drift_delta_score_vs_abs_delta_score_interneuron_firing(self):

        corr_ds_mean_nrem_rem = self.memory_drift_delta_score_interneuron_firing(use_abs_delta_score=False)
        corr_abs_ds_mean_nrem_rem = self.memory_drift_delta_score_interneuron_firing(use_abs_delta_score=True)

        plt.figure(figsize=(8,8))
        plt.scatter(corr_ds_mean_nrem_rem, corr_abs_ds_mean_nrem_rem)
        plt.xlabel("Pearson R (interneuron_fir, delta_score")
        plt.ylabel("Pearson R (interneuron_fir, ABS. delta_score")
        plt.text(-0.4, -0.2, "Pearson stats="+str(np.round(pearsonr(corr_ds_mean_nrem_rem, corr_abs_ds_mean_nrem_rem), 2)))
        plt.tight_layout()
        plt.show()

    def memory_drift_firing_prob_interneuron_firing(self, save_fig=False, partial_corr=False):
        corr_rem_dec = []
        corr_rem_inc = []
        corr_nrem_dec = []
        corr_nrem_inc = []

        for session in self.session_list:
            corr_rem_dec_, corr_rem_inc_, corr_nrem_dec_, corr_nrem_inc_ = \
                session.long_sleep().memory_drift_interneuron_vs_firing_prob(plotting=False, partial_corr=partial_corr)
            corr_rem_dec.append(corr_rem_dec_)
            corr_rem_inc.append(corr_rem_inc_)
            corr_nrem_dec.append(corr_nrem_dec_)
            corr_nrem_inc.append(corr_nrem_inc_)

        corr_rem_dec = np.hstack(corr_rem_dec)
        corr_rem_inc = np.hstack(corr_rem_inc)
        corr_nrem_dec = np.hstack(corr_nrem_dec)
        corr_nrem_inc = np.hstack(corr_nrem_inc)

        plt.scatter(corr_rem_inc, corr_nrem_inc)
        if partial_corr:
            plt.xlabel("Part. corr: inter firing vs. delta_firing_prob inc REM")
            plt.ylabel("Part. corr: inter firing vs. delta_firing_prob inc NREM")
        else:
            plt.xlabel("Corr: inter firing vs. delta_firing_prob inc REM")
            plt.ylabel("Corr: inter firing vs. delta_firing_prob inc NREM")
        plt.text(-0.4, -0.2, "R=" + str(pearsonr(corr_rem_inc, corr_nrem_inc)[0]))
        plt.show()

        if save_fig:
            plt.style.use('default')
        plt.scatter(corr_rem_dec, corr_nrem_dec, color="gray")
        if partial_corr:
            plt.xlabel("Partial corr. (interneuron firing rate, \n delta firing_prob dec cells), REM")
            plt.ylabel("Partial corr. (interneuron firing rate, \n delta firing_prob dec cells), NREM")
        else:
            plt.xlabel("Pearson R (interneuron firing rate, \n delta firing_prob dec cells), REM")
            plt.ylabel("Pearson R (interneuron firing rate, \n delta firing_prob dec cells), NREM")
        plt.text(-0.4, -0.2, "R=" + str(np.round(pearsonr(corr_rem_dec, corr_nrem_dec)[0],2)))
        plt.text(-0.4, -0.25, "p=" + str(pearsonr(corr_rem_dec, corr_nrem_dec)[1]))
        plt.gca().set_aspect("equal")
        plt.xlim(-0.5, 0.5)
        plt.ylim(-0.5, 0.5)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "interneuron_firing_vs_delta_firing_prob_dec.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def memory_drift_interneuron_vs_firing_rate(self, save_fig=False):
        corr_rem = []
        corr_nrem = []
        for session in self.session_list:
            corr_rem_, corr_nrem_ = session.long_sleep().memory_drift_interneuron_vs_firing_rate(plotting=False)
            corr_rem.append(corr_rem_)
            corr_nrem.append(corr_nrem_)
        corr_rem = np.hstack(corr_rem)
        corr_nrem = np.hstack(corr_nrem)

        if save_fig:
            plt.style.use('default')
        plt.scatter(corr_rem, corr_nrem, color="gray")
        plt.xlabel("Pearson R (interneuron firing, mean firing dec. cells) REM")
        plt.ylabel("Pearson R (interneuron firing, mean firing dec. cells) NREM")
        plt.text(-0, -0.3, "R=" + str(np.round(pearsonr(corr_rem, corr_nrem)[0],2)))
        plt.text(-0, -0.35, "p=" + str(pearsonr(corr_rem, corr_nrem)[1]))
        plt.gca().set_aspect("equal")
        plt.xlim(-0.6, 0.9)
        plt.ylim(-0.6, 0.9)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "interneuron_firing_vs_firing_rate_dec.svg"), transparent="True")
            plt.close()
        else:
            plt.show()


class MultipleSessions:
    """class for multiple sessions"""

    def __init__(self, session_names, cell_type, params):
        """
        Initializes multiple sessions

        :param session_names: name of sessions (e.g. ["mjc163R4R_0114", "mjc163R2R_0114"])
        :type session_names: list
        :param cell_type: which cell_type to analyze (e.g. "p1" for HPC pyramidal cells)
        :type cell_type: str
        :param params: parameters for analysis
        :type params: Parameters class
        """
        self.params = params

        # initialize all sessions
        self.session_list = []
        for session_name in session_names:
            self.session_list.append(SingleSession(session_name=session_name, cell_type=cell_type, params=params))

    # <editor-fold desc = "Learning">

    def learning_map_dynamics(self, adjust_pv_size=False):
        """
        Checks remapping of cells during learning
        @param adjust_pv_size: whether to subsample pv for decreasing/increasing/stable cells to have same number
        of cells
        @type adjust_pv_size: bool

        """

        # go trough all sessions to collect results
        remapping_stable = []
        remapping_shuffle_stable = []
        remapping_dec = []
        remapping_shuffle_dec = []
        remapping_pv_stable = []
        remapping_pv_stable_shuffle = []
        remapping_pv_dec = []
        remapping_pv_dec_shuffle = []
        for session in self.session_list:
            rs, rss, rd, rds, rpvs, rpvss, rpvd, rpvds = session.cheese_board(
                experiment_phase=["learning_cheeseboard_1"]).map_dynamics_learning(plot_results=False,
                                                                                   adjust_pv_size=adjust_pv_size)
            remapping_stable.append(rs)
            remapping_shuffle_stable.append(rss.flatten())
            remapping_dec.append(rd)
            remapping_shuffle_dec.append(rds.flatten())
            remapping_pv_stable.append(rpvs)
            remapping_pv_stable_shuffle.append(rpvss)
            remapping_pv_dec.append(rpvd)
            remapping_pv_dec_shuffle.append(rpvds)

        # ----------------------------------------------------------------------------------------------------------
        # compute for single cells
        # ----------------------------------------------------------------------------------------------------------

        # stable cells
        remapping_stable = np.expand_dims(np.hstack(remapping_stable), 1).flatten()
        remapping_shuffle_stable = np.expand_dims(np.hstack(remapping_shuffle_stable), 1).flatten()

        remapping_dec = np.expand_dims(np.hstack(remapping_dec), 1).flatten()
        remapping_shuffle_dec = np.expand_dims(np.hstack(remapping_shuffle_dec), 1).flatten()

        c = "white"
        res = [remapping_stable, remapping_shuffle_stable, remapping_dec, remapping_shuffle_dec]
        bplot = plt.boxplot(res, positions=[1, 2, 3, 4], patch_artist=True,
                            labels=["Stable", "Stable shuffle", "Dec", "Dec shuffle"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["magenta", 'magenta', "blue", "blue"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.title("Remapping using single cells")
        plt.ylabel("Pearson R: PRE map - POST map")
        plt.grid(color="grey", axis="y")
        plt.show()

        remapping_stable_sorted = np.sort(remapping_stable)
        remapping_stable_shuffle_sorted = np.sort(remapping_shuffle_stable)
        remapping_dec_sorted = np.sort(remapping_dec)
        remapping_dec_shuffle_sorted = np.sort(remapping_shuffle_dec)

        # plot on population vector level
        p_stable = 1. * np.arange(remapping_stable_sorted.shape[0]) / (remapping_stable_sorted.shape[0] - 1)
        p_stable_shuffle = 1. * np.arange(remapping_stable_shuffle_sorted.shape[0]) / (
                    remapping_stable_shuffle_sorted.shape[0] - 1)

        p_dec = 1. * np.arange(remapping_dec_sorted.shape[0]) / (remapping_dec_sorted.shape[0] - 1)
        p_dec_shuffle = 1. * np.arange(remapping_dec_shuffle_sorted.shape[0]) / (
                    remapping_dec_shuffle_sorted.shape[0] - 1)

        plt.plot(remapping_stable_sorted, p_stable, label="Stable", color="magenta")
        plt.plot(remapping_stable_shuffle_sorted, p_stable_shuffle, label="Stable shuffle", color="darkmagenta")

        plt.plot(remapping_dec_sorted, p_dec, label="Dec", color="aquamarine")
        plt.plot(remapping_dec_shuffle_sorted, p_dec_shuffle, label="Dec shuffle", color="lightseagreen")
        plt.legend()
        plt.ylabel("CDF")
        plt.xlabel("PEARSON R")
        plt.title("Per cell.")
        plt.show()

        # ----------------------------------------------------------------------------------------------------------
        # compute for population vectors
        # ----------------------------------------------------------------------------------------------------------

        # stable cells
        remapping_pv_stable = np.hstack(remapping_pv_stable)
        remapping_pv_stable_shuffle = np.hstack(remapping_pv_stable_shuffle)

        remapping_pv_stable = remapping_pv_stable[~np.isnan(remapping_pv_stable)]
        remapping_pv_stable_shuffle = remapping_pv_stable_shuffle[~np.isnan(remapping_pv_stable_shuffle)]

        # decreasing cells
        remapping_pv_dec = np.hstack(remapping_pv_dec)
        remapping_pv_dec_shuffle = np.hstack(remapping_pv_dec_shuffle)

        remapping_pv_dec = remapping_pv_dec[~np.isnan(remapping_pv_dec)]
        remapping_pv_dec_shuffle = remapping_pv_dec_shuffle[~np.isnan(remapping_pv_dec_shuffle)]

        # print("MWU for PV remapping:"+str(mannwhitneyu(remapping_pv, remapping_pv_shuffle)[1]))
        c = "white"

        plt.figure(figsize=(4, 5))
        res = [remapping_pv_stable, remapping_pv_stable_shuffle, remapping_pv_dec, remapping_pv_dec_shuffle]
        bplot = plt.boxplot(res, positions=[1, 2, 3, 4], patch_artist=True,
                            labels=["stable", "stable shuffle", "dec", "dec shuffle"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["magenta", 'magenta', "blue", "blue"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Pearson R: First trial vs. last trial")
        # plt.yticks([-0.5, 0, 0.5, 1])
        # plt.ylim(-0.5, 1)
        plt.grid(color="grey", axis="y")

        plt.title("Remapping using Population Vectors")
        plt.show()

        remapping_pv_stable_sorted = np.sort(remapping_pv_stable)
        remapping_pv_stable_shuffle_sorted = np.sort(remapping_pv_stable_shuffle)
        remapping_pv_dec_sorted = np.sort(remapping_pv_dec)
        remapping_pv_dec_shuffle_sorted = np.sort(remapping_pv_dec_shuffle)

        # plot on population vector level
        p_pv_stable = 1. * np.arange(remapping_pv_stable_sorted.shape[0]) / (remapping_pv_stable_sorted.shape[0] - 1)
        p_pv_stable_shuffle = 1. * np.arange(remapping_pv_stable_shuffle_sorted.shape[0]) / (
                    remapping_pv_stable_shuffle_sorted.shape[0] - 1)

        p_pv_dec = 1. * np.arange(remapping_pv_dec_sorted.shape[0]) / (remapping_pv_dec_sorted.shape[0] - 1)
        p_pv_dec_shuffle = 1. * np.arange(remapping_pv_dec_shuffle_sorted.shape[0]) / (
                    remapping_pv_dec_shuffle_sorted.shape[0] - 1)

        plt.plot(remapping_pv_stable_sorted, p_pv_stable, label="Stable", color="magenta")
        plt.plot(remapping_pv_stable_shuffle_sorted, p_pv_stable_shuffle, label="Stable shuffle", color="darkmagenta")

        plt.plot(remapping_pv_dec_sorted, p_pv_dec, label="Dec", color="aquamarine")
        plt.plot(remapping_pv_dec_shuffle_sorted, p_pv_dec_shuffle, label="Dec shuffle", color="lightseagreen")
        plt.legend()
        plt.ylabel("CDF")
        plt.xlabel("PEARSON R")
        plt.title("Per pop. vec.")
        plt.show()

    def learning_mean_firing(self, absolute_value=False, filter_low_firing=False):
        """
        Checks firing rate remapping of cells during learning
        @param absolute_value: weather to use absolute or relative value
        @type absolute_value: bool

        """

        # go trough all sessions to collect results
        diff_stable = []
        diff_dec = []

        for session in self.session_list:
            d_s, d_d = session.cheese_board(
                experiment_phase=["learning_cheeseboard_1"]).learning_mean_firing_rate(plotting=False,
                                                                                       absolute_value=absolute_value,
                                                                                       filter_low_firing=filter_low_firing)
            diff_stable.append(d_s)
            diff_dec.append(d_d)

        diff_stable = np.hstack(diff_stable)
        diff_dec = np.hstack(diff_dec)

        diff_stable_sorted = np.sort(diff_stable)
        diff_dec_sorted = np.sort(diff_dec)

        p_diff_stable = 1. * np.arange(diff_stable.shape[0]) / (diff_stable.shape[0] - 1)

        p_diff_dec = 1. * np.arange(diff_dec.shape[0]) / (diff_dec.shape[0] - 1)

        plt.plot(diff_stable_sorted, p_diff_stable, label="stable")
        plt.plot(diff_dec_sorted, p_diff_dec, label="dec")
        if absolute_value:
            plt.xlabel("Abs. relative Difference firing rates")
        else:
            plt.xlabel("Rel. difference firing rates")
        plt.ylabel("cdf")
        plt.title("Change in mean firing rates through learning")
        plt.legend()
        plt.show()
        print(mannwhitneyu(diff_stable, diff_dec))

    def learning_start_end_map_stability(self):
        """
        Checks remapping of cells by comparing initial trials vs. last trials

        """

        # go trough all sessions to collect results
        initial_pop_vec_sim_stable = []
        initial_pop_vec_sim_dec = []
        late_pop_vec_sim_stable = []
        late_pop_vec_sim_dec = []

        for session in self.session_list:
            i_s, i_d, l_s, l_d = session.cheese_board(
                experiment_phase=["learning_cheeseboard_1"]).map_heterogenity(plotting=False)
            initial_pop_vec_sim_stable.append(i_s)
            initial_pop_vec_sim_dec.append(i_d)
            late_pop_vec_sim_stable.append(l_s)
            late_pop_vec_sim_dec.append(l_d)

        initial_pop_vec_sim_stable = np.hstack(initial_pop_vec_sim_stable)
        initial_pop_vec_sim_dec = np.hstack(initial_pop_vec_sim_dec)
        late_pop_vec_sim_stable = np.hstack(late_pop_vec_sim_stable)
        late_pop_vec_sim_dec = np.hstack(late_pop_vec_sim_dec)

        # ----------------------------------------------------------------------------------------------------------
        # compute for single cells
        # ----------------------------------------------------------------------------------------------------------

        initial_pop_vec_sim_stable_sorted = np.sort(initial_pop_vec_sim_stable)
        initial_pop_vec_sim_dec_sorted = np.sort(initial_pop_vec_sim_dec)
        late_pop_vec_sim_stable_sorted = np.sort(late_pop_vec_sim_stable)
        late_pop_vec_sim_dec_sorted = np.sort(late_pop_vec_sim_dec)

        # plot on population vector level
        p_stable_init = 1. * np.arange(initial_pop_vec_sim_stable.shape[0]) / (initial_pop_vec_sim_stable.shape[0] - 1)
        p_dec_init = 1. * np.arange(initial_pop_vec_sim_dec.shape[0]) / (
                    initial_pop_vec_sim_dec.shape[0] - 1)

        p_stable_late = 1. * np.arange(late_pop_vec_sim_stable.shape[0]) / (late_pop_vec_sim_stable.shape[0] - 1)
        p_dec_late = 1. * np.arange(late_pop_vec_sim_dec.shape[0]) / (
                late_pop_vec_sim_dec.shape[0] - 1)

        plt.plot(initial_pop_vec_sim_stable_sorted, p_stable_init, label="Early-Stable", color="magenta",
                 linestyle="dashed")
        plt.plot(initial_pop_vec_sim_dec_sorted, p_dec_init, label="Early-Dec", color="darkmagenta")

        plt.plot(late_pop_vec_sim_stable_sorted, p_stable_late, label="Late-Stable", color="aquamarine",
                 linestyle="dashed")
        plt.plot(late_pop_vec_sim_dec_sorted, p_dec_late, label="Late_dec", color="lightseagreen")
        plt.legend()
        plt.ylabel("CDF")
        plt.xlabel("Cosine distance")
        plt.title("Pairwise PVs")
        plt.show()

    def learning_place_field_peak_shift(self, spatial_resolution=1):
        """
        Checks remapping of cells during learning using place field peak shift

        @param spatial_resolution: spatial bin size in cm2
        @type spatial_resolution: int

        """

        # go trough all sessions to collect results
        shift_stable = []
        shift_dec = []

        for session in self.session_list:
            s_s, s_d = \
                session.cheese_board(experiment_phase=["learning_cheeseboard_1"]).learning_place_field_peak_shift(
                    plotting=False, spatial_resolution=spatial_resolution)
            shift_stable.append(s_s)
            shift_dec.append(s_d)

        # ----------------------------------------------------------------------------------------------------------
        # compute for single cells
        # ----------------------------------------------------------------------------------------------------------

        shift_stable = np.hstack(shift_stable)
        shift_dec = np.hstack(shift_dec)
        shift_stable_sorted = np.sort(shift_stable)
        shift_dec_sorted = np.sort(shift_dec)

        # plot on population vector level
        p_stable = 1. * np.arange(shift_stable.shape[0]) / (shift_stable.shape[0] - 1)

        p_dec = 1. * np.arange(shift_dec.shape[0]) / (shift_dec.shape[0] - 1)

        plt.plot(shift_stable_sorted, p_stable, label="Stable", color="magenta")
        plt.plot(shift_dec_sorted, p_dec, label="Dec", color="aquamarine")
        plt.legend()
        plt.ylabel("CDF")
        plt.xlabel("Place field peak shift / cm")
        plt.title("Place field peak shift during learning")
        plt.show()

    def learning_error_stable_vs_decreasing(self, nr_of_trials=10):
        """
        compares decoding error by cross-validation (first n trials to train, last n to test)

        @param nr_of_trials: how many trials to use for training/testing
        @type nr_of_trials: int
        """
        # go trough all sessions to collect results
        error_stable = []
        error_dec = []

        for session in self.session_list:
            e_s, e_d = session.cheese_board(
             experiment_phase=["learning_cheeseboard_1"]).decoding_error_stable_vs_decreasing(plotting=False,
                                                                                              nr_of_trials=nr_of_trials)
            error_stable.append(e_s)
            error_dec.append(e_d)

        error_stable = np.hstack(error_stable)
        error_stable_sorted = np.sort(error_stable)
        p_error_stable = 1. * np.arange(error_stable.shape[0]) / (error_stable.shape[0] - 1)

        error_dec = np.hstack(error_dec)
        error_dec_sorted = np.sort(error_dec)
        p_error_dec = 1. * np.arange(error_dec.shape[0]) / (error_dec.shape[0] - 1)

        plt.plot(error_stable_sorted, p_error_stable, label="stable")
        plt.plot(error_dec_sorted, p_error_dec, label="dec")
        plt.legend()
        plt.show()

    def cheeseboard_place_field_goal_distance_temporal(self, save_fig=False, mean_firing_threshold=1,
                                                       nr_trials=4, pre_or_post="pre",
                                                       spatial_resolution_rate_maps=5, min_nr_spikes=20):
        """
        computes distance between place field peak and closest goal -- compares end of learning vs. after learning

        @param save_fig: save as .svg
        @param mean_firing_threshold: threshold to exclude low firing cells (in Hz)
        @param nr_trials: how many trials to use
        @type nr_trials: int
        @param pre_or_post: whether to use pre or post
        @type pre_or_post: str
        """

        dist_stable_learning_1 = []
        dist_stable_learning_2 = []
        dist_stable_learning_3 = []
        dist_dec_learning_1 = []
        dist_dec_learning_2 = []
        dist_dec_learning_3 = []
        dist_inc_learning_1 = []
        dist_inc_learning_2 = []
        dist_inc_learning_3 = []

        if pre_or_post == "pre":
            experiment_phase = ["learning_cheeseboard_1"]
        elif pre_or_post == "post":
            experiment_phase = ["learning_cheeseboard_2"]
        else:
            raise Exception("Need to choose either pre or post")

        for session in self.session_list:
            nr_total_trials = session.cheese_board(experiment_phase=experiment_phase).nr_trials
            # first n trials
            d_s_learning, d_d_learning, d_i_learning = \
                session.cheese_board(
                    experiment_phase=experiment_phase).place_field_goal_distance(
                    spatial_resolution_rate_maps=spatial_resolution_rate_maps,
                    plotting=False, trials_to_use=range(nr_trials),
                    min_nr_spikes=min_nr_spikes)

            dist_stable_learning_1.append(d_s_learning)
            dist_dec_learning_1.append(d_d_learning)
            dist_inc_learning_1.append(d_i_learning)

            # middle n trials
            d_s_learning, d_d_learning, d_i_learning = \
                session.cheese_board(experiment_phase=experiment_phase).place_field_goal_distance(
                    plotting=False, trials_to_use=range(int(nr_total_trials/2)-int(nr_trials/2),
                                        int(nr_total_trials/2)+int(nr_trials/2)),spatial_resolution_rate_maps=
                    spatial_resolution_rate_maps, min_nr_spikes=min_nr_spikes)
            dist_stable_learning_2.append(d_s_learning)
            dist_dec_learning_2.append(d_d_learning)
            dist_inc_learning_2.append(d_i_learning)

            # last n trials
            d_s_learning, d_d_learning, d_i_learning = \
                session.cheese_board(
                    experiment_phase=experiment_phase).place_field_goal_distance(
                    plotting=False,
                    trials_to_use=range(nr_total_trials-nr_trials, nr_total_trials))

            dist_stable_learning_3.append(d_s_learning)
            dist_dec_learning_3.append(d_d_learning)
            dist_inc_learning_3.append(d_i_learning)

        dist_stable_learning_1 = np.hstack(dist_stable_learning_1)
        dist_stable_learning_2 = np.hstack(dist_stable_learning_2)
        dist_stable_learning_3 = np.hstack(dist_stable_learning_3)
        dist_dec_learning_1 = np.hstack(dist_dec_learning_1)
        dist_dec_learning_2 = np.hstack(dist_dec_learning_2)
        dist_dec_learning_3 = np.hstack(dist_dec_learning_3)
        dist_inc_learning_1 = np.hstack(dist_inc_learning_1)
        dist_inc_learning_2 = np.hstack(dist_inc_learning_2)
        dist_inc_learning_3 = np.hstack(dist_inc_learning_3)

        if pre_or_post == "pre":

            if save_fig:
                plt.style.use('default')
                c = "black"
            else:
                c = "white"

            res = [dist_stable_learning_1, dist_stable_learning_2, dist_stable_learning_3]

            plt.figure(figsize=(2, 3))
            bplot = plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                                labels=["first "+str(nr_trials)+" trials", "middle "+str(nr_trials)+" trials", 
                                        "last "+str(nr_trials)+" trials"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            colors = ["magenta", "magenta", "magenta"]
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            plt.ylabel("Distance to closest goal (cm)")
            # plt.grid(color="grey", axis="y")
            y_base = 60
            plt.hlines(y_base+5, 1, 1.9, color=c)
            if mannwhitneyu(dist_stable_learning_1, dist_stable_learning_2)[1] < 0.001/2:
                plt.text(1.3, y_base+5, "***")
            elif mannwhitneyu(dist_stable_learning_1, dist_stable_learning_2)[1] < 0.01/2:
                plt.text(1.3, y_base+5, "**")
            elif mannwhitneyu(dist_stable_learning_1, dist_stable_learning_2)[1] < 0.05/2:
                plt.text(1.3, y_base+5, "*")
            else:
                plt.text(1.3, y_base+5, "n.s.")

            plt.hlines(y_base, 1, 3, color=c)
            if mannwhitneyu(dist_stable_learning_1, dist_stable_learning_3)[1] < 0.001/2:
                plt.text(2, y_base, "***")
            elif mannwhitneyu(dist_stable_learning_1, dist_stable_learning_3)[1] < 0.01/2:
                plt.text(2, y_base, "**")
            elif mannwhitneyu(dist_stable_learning_1, dist_stable_learning_3)[1] < 0.05/2:
                plt.text(2, y_base, "*")
            else:
                plt.text(2, y_base, "n.s.")
            plt.ylim(0, 70)
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "place_field_to_goal_distance_pre_stable.svg"), transparent="True")
            else:
                plt.show()

            res = [dist_dec_learning_1, dist_dec_learning_2, dist_dec_learning_3]

            plt.figure(figsize=(2, 3))
            bplot = plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                                labels=["first "+str(nr_trials)+" trials", "middle "+str(nr_trials)+" trials", 
                                        "last "+str(nr_trials)+" trials"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            colors = ['turquoise', 'turquoise', 'turquoise']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            plt.ylabel("Distance to closest goal (cm)")
            y_base = 60
            plt.hlines(y_base+5, 1, 1.9)
            if mannwhitneyu(dist_dec_learning_1, dist_dec_learning_2)[1] < 0.001/2:
                plt.text(1.3, y_base+5, "***")
            elif mannwhitneyu(dist_dec_learning_1, dist_dec_learning_2)[1] < 0.01/2:
                plt.text(1.3, y_base+5, "**")
            elif mannwhitneyu(dist_dec_learning_1, dist_dec_learning_2)[1] < 0.05/2:
                plt.text(1.3, y_base+5, "*")
            else:
                plt.text(1.3, y_base+5, "n.s.")

            plt.hlines(y_base, 1, 3)
            if mannwhitneyu(dist_dec_learning_1, dist_dec_learning_3)[1] < 0.001/2:
                plt.text(2, y_base, "***")
            elif mannwhitneyu(dist_dec_learning_1, dist_dec_learning_3)[1] < 0.01/2:
                plt.text(2, y_base, "**")
            elif mannwhitneyu(dist_dec_learning_1, dist_dec_learning_3)[1] < 0.05/2:
                plt.text(2, y_base, "*")
            else:
                plt.text(2, y_base, "n.s.")
            plt.ylim(0, 70)
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "place_field_to_goal_distance_pre_dec.svg"), transparent="True")
            else:
                plt.show()
        elif pre_or_post == "post":

            if save_fig:
                plt.style.use('default')
                c = "black"
            else:
                c = "white"

            res = [dist_stable_learning_1, dist_stable_learning_2, dist_stable_learning_3]

            plt.figure(figsize=(2, 3))
            bplot = plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                                labels=["first " + str(nr_trials) + " trials", "middle " + str(nr_trials) + " trials",
                                        "last " + str(nr_trials) + " trials"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            colors = ["magenta", "magenta", "magenta"]
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            plt.ylabel("Distance to closest goal (cm)")
            # plt.grid(color="grey", axis="y")
            y_base = 60
            plt.hlines(y_base + 5, 1, 1.9, color=c)
            if mannwhitneyu(dist_stable_learning_1, dist_stable_learning_2)[1] < 0.001/2:
                plt.text(1.3, y_base + 5, "***")
            elif mannwhitneyu(dist_stable_learning_1, dist_stable_learning_2)[1] < 0.01/2:
                plt.text(1.3, y_base + 5, "**")
            elif mannwhitneyu(dist_stable_learning_1, dist_stable_learning_2)[1] < 0.05/2:
                plt.text(1.3, y_base + 5, "*")
            else:
                plt.text(1.3, y_base + 5, "n.s.")

            plt.hlines(y_base, 1, 3, color=c)
            if mannwhitneyu(dist_stable_learning_1, dist_stable_learning_3)[1] < 0.001/2:
                plt.text(2, y_base, "***")
            elif mannwhitneyu(dist_stable_learning_1, dist_stable_learning_3)[1] < 0.01/2:
                plt.text(2, y_base, "**")
            elif mannwhitneyu(dist_stable_learning_1, dist_stable_learning_3)[1] < 0.05/2:
                plt.text(2, y_base, "*")
            else:
                plt.text(2, y_base, "n.s.")
            plt.ylim(0, 70)
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "place_field_to_goal_distance_post_stable.svg"), transparent="True")
            else:
                plt.show()

            res = [dist_inc_learning_1, dist_inc_learning_2, dist_inc_learning_3]

            plt.figure(figsize=(2, 3))
            bplot = plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                                labels=["first "+str(nr_trials)+" trials", "middle "+str(nr_trials)+" trials",
                                        "last "+str(nr_trials)+" trials"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            colors = ['orange', 'orange', 'orange']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            plt.ylabel("Distance to closest goal (cm)")
            y_base = 60
            plt.hlines(y_base+5, 1, 1.9)
            if mannwhitneyu(dist_inc_learning_1, dist_inc_learning_2)[1] < 0.001/2:
                plt.text(1.3, y_base+5, "***")
            elif mannwhitneyu(dist_inc_learning_1, dist_inc_learning_2)[1] < 0.01/2:
                plt.text(1.3, y_base+5, "**")
            elif mannwhitneyu(dist_inc_learning_1, dist_inc_learning_2)[1] < 0.05/2:
                plt.text(1.3, y_base+5, "*")
            else:
                plt.text(1.3, y_base+5, "n.s.")

            plt.hlines(y_base, 1, 3)
            if mannwhitneyu(dist_inc_learning_1, dist_inc_learning_3)[1] < 0.001/2:
                plt.text(2, y_base, "***")
            elif mannwhitneyu(dist_inc_learning_1, dist_inc_learning_3)[1] < 0.01/2:
                plt.text(2, y_base, "**")
            elif mannwhitneyu(dist_inc_learning_1, dist_inc_learning_3)[1] < 0.05/2:
                plt.text(2, y_base, "*")
            else:
                plt.text(2, y_base, "n.s.")
            plt.ylim(0, 70)
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "place_field_to_goal_distance_post_inc.svg"), transparent="True")
            else:
                plt.show()

    def cheeseboard_place_field_goal_distance_first_x_vs_rest(self, save_fig=False,
                                                       first_x_trials=4, pre_or_post="pre",
                                                       spatial_resolution_rate_maps=5, min_nr_spikes=20):
        """
        computes distance between place field peak and closest goal -- compares end of learning vs. after learning

        @param save_fig: save as .svg
        @param mean_firing_threshold: threshold to exclude low firing cells (in Hz)
        @param nr_trials: how many trials to use
        @type nr_trials: int
        @param pre_or_post: whether to use pre or post
        @type pre_or_post: str
        """

        dist_stable_learning_1 = []
        dist_stable_learning_2 = []
        dist_dec_learning_1 = []
        dist_dec_learning_2 = []
        dist_inc_learning_1 = []
        dist_inc_learning_2 = []


        if pre_or_post == "pre":
            experiment_phase = ["learning_cheeseboard_1"]
        elif pre_or_post == "post":
            experiment_phase = ["learning_cheeseboard_2"]
        else:
            raise Exception("Need to choose either pre or post")

        for session in self.session_list:
            dist_stable_first, dist_stable, dist_dec_first, dist_dec, dist_inc_first, dist_inc = \
                session.cheese_board(
                    experiment_phase=experiment_phase).place_field_goal_distance_first_x_vs_rest(
                    spatial_resolution_rate_maps=spatial_resolution_rate_maps,
                    plotting=False, first_x=first_x_trials,
                    min_nr_spikes=min_nr_spikes)

            dist_stable_learning_1.append(dist_stable_first)
            dist_dec_learning_1.append(dist_dec_first)
            dist_inc_learning_1.append(dist_inc_first)
            dist_stable_learning_2.append(dist_stable)
            dist_dec_learning_2.append(dist_dec)
            dist_inc_learning_2.append(dist_inc)

        dist_stable_learning_1 = np.hstack(dist_stable_learning_1)
        dist_stable_learning_2 = np.hstack(dist_stable_learning_2)
        dist_dec_learning_1 = np.hstack(dist_dec_learning_1)
        dist_dec_learning_2 = np.hstack(dist_dec_learning_2)
        dist_inc_learning_1 = np.hstack(dist_inc_learning_1)
        dist_inc_learning_2 = np.hstack(dist_inc_learning_2)

        if pre_or_post == "pre":

            if save_fig:
                plt.style.use('default')
                c = "black"
            else:
                c = "white"

            res = [dist_stable_learning_1, dist_stable_learning_2]

            plt.figure(figsize=(4, 5))
            bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                                labels=["First trial", "Rest"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            colors = ["magenta","magenta"]
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

            plt.ylabel("Distance to closest goal (cm)")
            # plt.grid(color="grey", axis="y")
            y_base = 60
            plt.hlines(y_base+5, 1, 1.9)
            if mannwhitneyu(dist_stable_learning_1, dist_stable_learning_2)[1] < 0.001:
                plt.text(1.3, y_base+5, "***")
            elif mannwhitneyu(dist_stable_learning_1, dist_stable_learning_2)[1] < 0.01:
                plt.text(1.3, y_base+5, "**")
            elif mannwhitneyu(dist_stable_learning_1, dist_stable_learning_2)[1] < 0.05:
                plt.text(1.3, y_base+5, "*")
            else:
                plt.text(1.3, y_base+5, "n.s.")
            plt.ylim(0, 70)
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "place_field_to_goal_distance_first_vs_rest_pre_stable.svg"), transparent="True")
            else:
                plt.show()


            res = [dist_dec_learning_1, dist_dec_learning_2]

            plt.figure(figsize=(4, 5))
            bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                                labels=["First trial", "Rest"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            colors = ['turquoise', 'turquoise']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

            plt.ylabel("Distance to closest goal (cm)")
            # plt.grid(color="grey", axis="y")
            y_base = 60
            plt.hlines(y_base+5, 1, 1.9)
            if mannwhitneyu(dist_dec_learning_1, dist_dec_learning_2)[1] < 0.001:
                plt.text(1.3, y_base+5, "***")
            elif mannwhitneyu(dist_dec_learning_1, dist_dec_learning_2)[1] < 0.01:
                plt.text(1.3, y_base+5, "**")
            elif mannwhitneyu(dist_dec_learning_1, dist_dec_learning_2)[1] < 0.05:
                plt.text(1.3, y_base+5, "*")
            else:
                plt.text(1.3, y_base+5, "n.s.")
            plt.ylim(0, 70)
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "place_field_to_goal_distance_first_vs_rest_pre_dec.svg"), transparent="True")
            else:
                plt.show()

        elif pre_or_post == "post":

            if save_fig:
                plt.style.use('default')
                c = "black"
            else:
                c = "white"

            res = [dist_stable_learning_1, dist_inc_learning_1, dist_stable_learning_2, dist_inc_learning_2]

            plt.figure(figsize=(4, 5))
            bplot = plt.boxplot(res, positions=[1, 2, 3, 4], patch_artist=True,
                                labels=["Stable", "Inc", "Stable", "Inc"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            colors = ["magenta", 'orange', "magenta", 'orange']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            plt.ylabel("Distance to closest goal (cm)")
            plt.grid(color="grey", axis="y")
            print(mannwhitneyu(dist_stable_learning_1, dist_stable_learning_2))
            print(mannwhitneyu(dist_inc_learning_1, dist_inc_learning_2))
            plt.tight_layout()
            plt.ylim(0, 60)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "place_field_to_goal_distance_post.svg"), transparent="True")
            else:
                plt.show()


    def learning_rate_map_corr_stable_dec(self, spatial_resolution=2):
        stable = []
        dec = []
        for session in self.session_list:

            # first n trials
            remap_stable, remap_dec = \
                session.cheese_board(experiment_phase=["learning_cheeseboard_1"]).learning_rate_map_corr_stable_dec(spatial_resolution=
                                                                                                 spatial_resolution)
            stable.append(remap_stable)
            dec.append(remap_dec)

        plt.hist(np.hstack(stable).flatten(), density=True, color="blue", label="stable")
        plt.hist(np.hstack(dec).flatten(), density=True, color="red", alpha=0.7, label="decreasing")
        plt.legend()
        plt.xlabel("Pearson R (first 5 trials vs. last 5 trials")
        plt.ylabel("Density")
        plt.show()

        res = [np.hstack(stable), np.hstack(dec)]
        c="white"
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["stable", "dec"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.ylabel("Pearson R (first 5 trials vs. last 5 trials)")
        plt.show()
        print(mannwhitneyu(np.hstack(stable), np.hstack(dec)))

    def learning_pv_corr_stable_dec(self, spatial_resolution=5, save_fig=False):
        stable = []
        dec = []
        for session in self.session_list:

            # first n trials
            remap_stable, remap_dec = \
                session.cheese_board(experiment_phase=["learning_cheeseboard_1"]).learning_pv_corr_stable_dec(spatial_resolution=
                                                                                                 spatial_resolution, plotting=False)
            stable.append(remap_stable)
            dec.append(remap_dec)

        res = [np.hstack(dec), np.hstack(stable)]
        plt.figure(figsize=(3, 5))
        if save_fig:
            c = "black"
            plt.style.use('default')
        else:
            c = "white"
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["decreasing", "persistent"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.ylabel("PV correlation (first 5 trials vs. last 5 trials)")
        plt.ylim(0, 1.19)
        plt.grid(color="grey", axis="y")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "learning_pv_correlations.svg"), transparent="True")
        else:
            plt.show()
        print(mannwhitneyu(np.hstack(stable), np.hstack(dec)))

    def learning_multinom_log_reg_stable_dec_inc(self, time_bin_size=None):
        stable = []
        dec = []
        inc = []
        for session in self.session_list:

            # first n trials
            res_dic = \
                session.cheese_board(experiment_phase=["learning_cheeseboard_1"]).learning_multinom_log_reg_stable_dec_inc(time_bin_size=
                                                                                                 time_bin_size)
            stable.append(res_dic["stable"])
            dec.append(res_dic["decreasing"])
            inc.append(res_dic["increasing"])

        dec = np.hstack(dec)
        inc = np.hstack(inc)
        stable = np.hstack(stable)

        labels = ["Stable", "Decreasing", "Increasing"]
        c = "white"
        res = [stable, dec, inc]
        bplot = plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                            labels=labels,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.ylabel("Score (Multinom. Log. Reg.)")
        if time_bin_size is None:
            plt.title(str(self.params.time_bin_size)+"s time bins")
        else:
            plt.title(str(time_bin_size)+"s time bins")
        plt.show()
        print("HERE")

    def learning_svm_first_last_trials_stable_dec_inc(self, time_bin_size=None):
        stable = []
        dec = []
        inc = []
        for session in self.session_list:

            # first n trials
            res_dic = \
                session.cheese_board(experiment_phase=["learning_cheeseboard_1"]).learning_svm_first_trials_last_trials_stable_dec_inc(time_bin_size=
                                                                                                 time_bin_size)
            stable.append(res_dic["stable"])
            dec.append(res_dic["decreasing"])
            inc.append(res_dic["increasing"])

        dec = np.hstack(dec)
        inc = np.hstack(inc)
        stable = np.hstack(stable)

        labels = ["Stable", "Decreasing", "Increasing"]
        c = "white"
        res = [stable, dec, inc]
        bplot = plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                            labels=labels,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.ylabel("Score (SVM) - first vs. last trials")
        if time_bin_size is None:
            plt.title(str(self.params.time_bin_size)+"s time bins")
        else:
            plt.title(str(time_bin_size)+"s time bins")
        plt.ylim(0,1)
        plt.show()
        print(mannwhitneyu(stable, dec))

    def learning_predict_bin_progression(self, time_bin_size=2, save_fig=False):
        stable = []
        dec = []
        for session in self.session_list:
            r2_stable = \
                session.cheese_board(experiment_phase=["learning_cheeseboard_1"]).learning_predict_bin_progression(time_bin_size=
                                                                                                 time_bin_size, cells_to_use="stable",
                                                                                                                   plotting=False)
            r2_dec = \
                session.cheese_board(experiment_phase=["learning_cheeseboard_1"]).learning_predict_bin_progression(time_bin_size=
                                                                                                 time_bin_size, cells_to_use="decreasing",
                                                                                                                   plotting=False)
            stable.append(r2_stable)
            dec.append(r2_dec)

        stable = np.array(stable)
        dec = np.array(dec)
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        diff = dec - stable

        plt.figure(figsize=(2, 6))
        bplot = plt.boxplot(diff, positions=[1], patch_artist=True,
                            labels=[""],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.ylabel("r2_decreasing - r2_persistent")
        plt.hlines(0, 0.5, 1.5, color="gray", linestyles="--")
        plt.ylim(-1, 1)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "learning_predicting_time_bin_dec_vs_stable.svg"), transparent="True")
        else:
            plt.show()

        print(ttest_1samp(diff, 0))

    def learning_correlations(self, save_fig=False):
        stable = []
        diff_stable = []
        dec = []
        diff_dec = []
        for session in self.session_list:
            corr_corr_stable, corr_corr_dec, diff_s, diff_d = \
                session.cheese_board(experiment_phase=["learning_cheeseboard_1"]).learning_correlations(plotting=False)

            stable.append(corr_corr_stable)
            dec.append(corr_corr_dec)
            diff_stable.append(diff_s)
            diff_dec.append(diff_d)

        diff_dec = np.hstack(diff_dec)
        diff_stable = np.hstack(diff_stable)

        plt.hist(diff_stable, color="blue", density=True, label="stable", bins=20)
        plt.hist(diff_dec, color="red", alpha=0.6, density=True, label="dec", bins=20)
        plt.legend()
        plt.xlabel("Diff. correlations (end-beginning)")
        plt.ylabel("Density")
        plt.show()

        plt.hist(np.abs(diff_stable), color="blue", density=True, label="stable", bins=20)
        plt.hist(np.abs(diff_dec), color="red", alpha=0.6, density=True, label="dec", bins=20)
        plt.legend()
        plt.xlabel("Abs. diff. correlations (end-beginning)")
        plt.ylabel("Density")
        plt.show()


        stable = np.array(stable)
        dec = np.array(dec)
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        bplot = plt.boxplot([stable, dec], positions=[1,2], patch_artist=True,
                            labels=["Stable", "Decreasing"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.ylabel("Pearson R (corr. 1st 5 trials vs. corr last 5 trials)")
        print(mannwhitneyu(stable, dec))
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "learning_predicting_time_bin_dec_vs_stable.svg"), transparent="True")
        else:
            plt.show()

    def learning_correlations_of_rate_maps(self, save_fig=False, spatial_resolution=5):
        stable = []
        diff_stable = []
        dec = []
        diff_dec = []
        r_values_initial_stable = []
        r_values_initial_dec = []
        r_values_last_stable = []
        r_values_last_dec = []
        for session in self.session_list:
            corr_corr_stable, corr_corr_dec, diff_s, diff_d, init_stable, init_dec, last_stable, last_dec = \
                session.cheese_board(experiment_phase=
                                     ["learning_cheeseboard_1"]).learning_rate_map_corr_across_cells(plotting=False,
                                                                                                     spatial_resolution=
                                                                                                     spatial_resolution)

            stable.append(corr_corr_stable)
            dec.append(corr_corr_dec)
            diff_stable.append(diff_s)
            diff_dec.append(diff_d)
            r_values_initial_stable.append(init_stable)
            r_values_initial_dec.append(init_dec)
            r_values_last_stable.append(last_stable)
            r_values_last_dec.append(last_dec)

        diff_dec = np.hstack(diff_dec)
        diff_stable = np.hstack(diff_stable)

        r_values_initial_stable = np.hstack(r_values_initial_stable)
        r_values_initial_dec = np.hstack(r_values_initial_dec)
        r_values_last_stable = np.hstack(r_values_last_stable)
        r_values_last_dec = np.hstack(r_values_last_dec)

        plt.hist(r_values_initial_dec, color="#00A79D", density=True, label="dec", bins=20)
        plt.hist(r_values_initial_stable, color="#91268F", density=True, alpha=0.6, label="stable", bins=20)
        plt.legend()
        plt.xlabel("Rate map correlations")
        plt.ylabel("Density")
        plt.title("Beginning of acquisition")
        plt.xlim(-0.5, 1)
        plt.show()


        plt.hist(r_values_last_dec, color="#00A79D", density=True, label="dec", bins=20)
        plt.hist(r_values_last_stable, color="#91268F", density=True, alpha=0.6, label="stable", bins=20)
        plt.legend()
        plt.xlabel("Rate map correlations")
        plt.ylabel("Density")
        plt.title("End of acquisition")
        plt.xlim(-0.5, 1)
        plt.show()

        plt.hist(r_values_initial_stable, color="grey", density=True, label="beginning", bins=20)
        plt.hist(r_values_last_stable, color="#00A79E", density=True, alpha=0.4, label="end", bins=20)
        plt.legend()
        plt.xlabel("Rate map correlations")
        plt.ylabel("Density")
        plt.title("Stable cells")
        plt.xlim(-0.5, 1)
        plt.show()

        plt.hist(r_values_initial_dec, color="grey", density=True, label="beginning", bins=20, zorder=-100)
        plt.hist(r_values_last_dec, color="#91268F", density=True, alpha=0.4, label="end", bins=20, zorder=1000)
        plt.legend()
        plt.xlabel("Rate map correlations")
        plt.ylabel("Density")
        plt.title("Decreasing cells")
        plt.xlim(-0.5, 1)
        plt.show()

        if save_fig:
            plt.style.use('default')

        plt.hist(diff_dec, color="#00A79D", density=True, label="dec", bins=20)
        plt.hist(diff_stable, color="#91268F", density=True, alpha=0.6, label="stable", bins=20)
        plt.legend()
        plt.xlabel("Diff. correlations (end-beginning)")
        plt.xlim(-1,1)
        plt.ylim(0,2)
        plt.vlines(0,0,2, color="gray", linestyles="--")
        plt.ylabel("Density")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "learning_diff_rate_map_correlations.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        print("All sessions:")
        print(mannwhitneyu(diff_stable, diff_dec))
        plt.figure(figsize=(5,5))
        plt.hist(np.abs(diff_dec), color="#00A79D", density=True, label="dec", bins=20)
        plt.hist(np.abs(diff_stable), color="#9E1F62", density=True, alpha=0.6, label="stable", bins=20)
        plt.legend()
        plt.xlabel("Abs. diff. correlations (end-beginning)")
        plt.xlim(0, 1)
        plt.ylabel("Density")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "learning_abs_diff_rate_map_correlations.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        print("All sessions (abs):")
        print(mannwhitneyu(np.abs(diff_stable), np.abs(diff_dec)))

        stable = np.array(stable)
        dec = np.array(dec)
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        bplot = plt.boxplot([stable, dec], positions=[1,2], patch_artist=True,
                            labels=["Stable", "Decreasing"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.ylabel("Pearson R (corr. 1st 5 trials vs. corr last 5 trials)")
        print(mannwhitneyu(stable, dec))
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "learning_predicting_time_bin_dec_vs_stable.svg"), transparent="True")
        else:
            plt.show()


    # </editor-fold>

    # <editor-fold desc = "PRE (after learning)">

    def pre_decode_single_goals(self, save_fig=False, subset_range=None, nr_splits=20, nr_subsets=10):
        """
        tries to decode single goals using population vectors and SVM using different number of cells

        @param save_fig: save .svg file
        @param subset_range: nr. of cells to use (as a list)
        @param nr_splits: how often to split for cross-validation
        @param nr_subsets: how many times to subsample (how many different subsets to use)
        """
        print("Identifying single goals using SVM ...")

        if subset_range is None:
            subset_range = [4, 8, 12, 18]

        stable = []
        decreasing = []
        increasing = []

        for session in self.session_list:
            s, i, d = session.cheese_board(
                experiment_phase=["learning_cheeseboard_1"]).identify_single_goal_multiple_subsets(
                subset_range=subset_range, nr_splits=nr_splits, nr_subsets=nr_subsets, plotting=False)
            stable.append(s)
            decreasing.append(d)
            increasing.append(i)

        stable = np.hstack(stable)
        dec = np.hstack(decreasing)
        inc = np.hstack(increasing)

        stable_mean = np.mean(stable, axis=1)
        dec_mean = np.mean(dec, axis=1)

        stable_std = np.std(stable, axis=1)
        dec_std = np.std(dec, axis=1)

        if save_fig:
            plt.style.use('default')

        plt.errorbar(x=subset_range, y=stable_mean, yerr=stable_std, label="stable", ls="--", fmt="o", capsize=5)
        # plt.errorbar(x=np.array(subset_range)+0.1, y=inc_mean, yerr=inc_std, label="inc")
        plt.errorbar(x=np.array(subset_range), y=dec_mean, yerr=dec_std, label="dec", ls="--", fmt="o",
                     capsize=5)
        plt.hlines(0.25, 4, 18, linestyles="--", colors="gray", label="chance")
        plt.ylabel("Mean accuracy - multiclass SVM (mean,std)")
        plt.xlabel("#cells")
        plt.legend()

        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "identifying_goals.svg"), transparent="True")
        else:
            plt.show()

    def pre_goal_related_activity(self, save_fig=False, subset_range=None, nr_splits=20, radius=15):
        """
        Tries to decode activity around goals and seperate it from activity away from goals

        @param save_fig: save .svg file
        @param subset_range: nr. of cells to use (as a list)
        @param nr_splits: how often to split for cross-validation
        @param radius: what radius around goals (in cm) to consider as goal related
        """

        if subset_range is None:
            subset_range = [4, 8, 12, 18]

        stable = []
        decreasing = []
        increasing = []

        for session in self.session_list:
            s,i,d = session.cheese_board(
                experiment_phase=["learning_cheeseboard_1"]).detect_goal_related_activity_using_multiple_subsets(
                subset_range=subset_range, nr_splits=nr_splits, plotting=False, radius=radius)
            stable.append(s)
            decreasing.append(d)
            increasing.append(i)

        stable = np.hstack(stable)
        dec = np.hstack(decreasing)
        inc = np.hstack(increasing)

        stable_mean = np.mean(stable, axis=1)
        dec_mean = np.mean(dec, axis=1)
        inc_mean = np.mean(inc, axis=1)

        stable_std = np.std(stable, axis=1)
        dec_std = np.std(dec, axis=1)
        inc_std = np.std(inc, axis=1)

        if save_fig:
            plt.style.use('default')

        plt.errorbar(x=subset_range, y=stable_mean, yerr=stable_std, label="stable")
        plt.errorbar(x=np.array(subset_range)+0.1, y=inc_mean, yerr=inc_std, label="inc")
        plt.errorbar(x=np.array(subset_range)+0.2, y=dec_mean, yerr=dec_std, label="dec")
        plt.ylabel("Mean accuracy - SVM (mean,std)")
        plt.xlabel("#cells")
        plt.legend()

        if save_fig:
            plt.savefig(os.path.join(save_path, "cell_classification_numbers.svg"), transparent="True")
        else:
            plt.show()

    # </editor-fold>

    # <editor-fold desc="Cheeseboard (PRE or POST)">

    def cheeseboard_cross_val_phmm(self, cells_to_use="all_cells", cl_ar=np.arange(1, 50, 5), pre_or_post="pre"):
        """
        cross validation of PHMM

        :param cells_to_use: whether to use "all_cells" or a subset ("decreasing", "increasing", "stable")
        :type cells_to_use: str
        :param cl_ar: which number of states to fit
        :type cl_ar: numpy.arange
        :param pre_or_post: whether to use "pre" or "post" to fit phmm model
        :type pre_or_post: str
        """
        if pre_or_post == "pre":
            exp_phase = ["learning_cheeseboard_1"]
        elif pre_or_post == "post":
            exp_phase = ["learning_cheeseboard_2"]
        else:
            raise Exception("Need to define pre or post")

        # go through sessions and cross-validate phmm
        for session in self.session_list:
            print("\nCross-val pHMM, session: "+session.session_name+"(cells:"+cells_to_use+") ...\n")
            session.cheese_board(experiment_phase=exp_phase).cross_val_poisson_hmm(cells_to_use=cells_to_use,
                                                                                                    cl_ar=cl_ar)
            print(" \n... done with cross-val pHMM, session: " + session.session_name+"\n")

    def cheeseboard_find_and_fit_optimal_phmm(self, cells_to_use="all_cells", cl_ar_init=np.arange(1, 50, 5),
                                              pre_or_post="pre"):
        """
        cross validation of pHMM to find optimal number of states (automatically does a coarse grid of nr_of_states,
        and refines this grid after)

        :param cl_ar_init: for which number of states to fit pHMM
        :type cl_ar_init: numpy.arange
        :param cells_to_use: whether to use "all_cells" or a subset ("decreasing", "increasing", "stable")
        :type cells_to_use: str
        :param pre_or_post: whether to use "pre" or "post" to fit phmm model
        :type pre_or_post: str
        """
        if pre_or_post == "pre":
            exp_phase = ["learning_cheeseboard_1"]
        elif pre_or_post == "post":
            exp_phase = ["learning_cheeseboard_2"]
        else:
            raise Exception("Select either pre or post")
        # go through sessions and cross-validate phmm
        for session in self.session_list:
            print("\nCross-val pHMM, session: "+session.session_name+"(cells:"+cells_to_use+") ...\n")
            session.cheese_board(experiment_phase=exp_phase).find_and_fit_optimal_number_of_modes(cells_to_use=
                                                                                                  cells_to_use,
                                                                                                  cl_ar_init=cl_ar_init)
            print(" \n... done with cross-val pHMM, session: " + session.session_name+"\n")

    def cheeseboard_evaluate_poisson_hmm(self, pre_or_post="pre", save_fig=False):
        """
        evaluate basic statistics of model and compare to data
        """

        if pre_or_post == "pre":
            exp_phase = ["learning_cheeseboard_1"]
        elif pre_or_post == "post":
            exp_phase = ["learning_cheeseboard_2"]
        else:
            raise Exception("Select either pre or post")

        mean_r = []
        corr_r = []

        for session in self.session_list:
            mean_dic, corr_dic, k_dic = session.cheese_board(experiment_phase=exp_phase).evaluate_poisson_hmm()
            mean_r.append(mean_dic["corr"][0])
            corr_r.append(corr_dic["corr"][0])

        mean_r = np.hstack(mean_r)
        corr_r = np.hstack(corr_r)

        print("Min. Pearson R for mean firing:")
        print(np.min(mean_r))

        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(3,3))
        plt.scatter(np.arange(1, mean_r.shape[0]+1), mean_r, color="gray")
        # plt.yscale("log")
        plt.ylim(0,1.1)
        plt.xlabel("Session")
        plt.ylabel("Pearson R - Mean firing")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "model_quality_mean_firing_all_sessions.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        print("Min. Pearson R for correlations:")
        print(np.min(corr_r))

        plt.figure(figsize=(3,3))
        plt.scatter(np.arange(1, corr_r.shape[0]+1), corr_r, color="gray")
        # plt.yscale("log")
        plt.ylim(0,1.1)
        plt.xlabel("Session")
        plt.ylabel("Pearson R - Correlations")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "model_quality_correlations_all_sessions.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        print("HERE")

    def cheeseboard_phmm_goal_coding(self, pre_or_post="pre", save_fig=False, thr_close_to_goal=10):
        """
        evaluate basic statistics of model and compare to data
        """

        if pre_or_post == "pre":
            exp_phase = ["learning_cheeseboard_1"]
        elif pre_or_post == "post":
            exp_phase = ["learning_cheeseboard_2"]
        else:
            raise Exception("Select either pre or post")

        fraction_close_to_goals = []

        for session in self.session_list:
            frac_per_mode = \
                session.cheese_board(experiment_phase=exp_phase).analyze_all_modes_goal_coding(
                    thr_close_to_goal=thr_close_to_goal)
            fraction_close_to_goals.append(frac_per_mode)

        fraction_close_to_goals = np.hstack(fraction_close_to_goals)

        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(4,3))
        plt.hist(fraction_close_to_goals, bins=25, edgecolor='white', color="gray")
        # plt.yscale("log")
        # plt.ylim(0,1.1)
        plt.xlabel("Proportion State active around goals")
        plt.ylabel("Number of states")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "goal_coding_all_modes.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def cheeseboard_phmm_spatial_information(self, pre_or_post="pre", save_fig=False):
        """
        evaluate basic statistics of model and compare to data
        """

        if pre_or_post == "pre":
            exp_phase = ["learning_cheeseboard_1"]
        elif pre_or_post == "post":
            exp_phase = ["learning_cheeseboard_2"]
        else:
            raise Exception("Select either pre or post")

        median_distance = []

        for session in self.session_list:
            md = \
                session.cheese_board(experiment_phase=exp_phase).analyze_all_modes_spatial_information()
            median_distance.append(md)

        median_distance = np.hstack(median_distance)

        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(4,3))
        plt.hist(median_distance, bins=25, edgecolor='white', color="gray")
        # plt.yscale("log")
        # plt.ylim(0,1.1)
        plt.xlabel("Median distance between activations (cm)")
        plt.ylabel("Number of states")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "median_distance_all_modes.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def cheeseboard_place_field_goal_distance(self, save_fig=False, mean_firing_threshold=None, pre_or_post="pre"):
        """
        computes distance between place field peak and closest goal

        @param pre_or_post: whether to look at goal distance in PRE or POST
        :type pre_or_post: str
        @param save_fig: save as .svg
        @param mean_firing_threshold: threshold to exclude low firing cells (in Hz)
        :type mean_firing_threshold: int
        """

        if pre_or_post == "pre":
            experiment_phase = ["learning_cheeseboard_1"]
        elif pre_or_post == "post":
            experiment_phase = ["learning_cheeseboard_2"]
        else:
            raise Exception("Need to define pre or post")

        dist_stable = []
        dist_dec = []
        dist_inc = []

        for session in self.session_list:
            d_s, d_d, d_i = session.cheese_board(experiment_phase=
                                                 experiment_phase).place_field_goal_distance(plotting=False,
                                                                                             mean_firing_threshold=
                                                                                             mean_firing_threshold)
            dist_stable.append(d_s)
            dist_dec.append(d_d)
            dist_inc.append(d_i)

        dist_stable = np.hstack(dist_stable)
        dist_dec = np.hstack(dist_dec)
        dist_inc = np.hstack(dist_inc)

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        if pre_or_post == "pre":
            plt.figure(figsize=(4, 5))
            res = [dist_stable, dist_dec]
            bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                                labels=["Stable", "Decreasing"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            colors = ["magenta", 'turquoise']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            plt.ylabel("Distance to closest goal (cm)")
            plt.grid(color="grey", axis="y")
            print(mannwhitneyu(dist_stable, dist_dec, alternative="less"))
            plt.ylim(0,60)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "place_field_to_goal_distance_pre.svg"), transparent="True")
            else:
                plt.show()

        elif pre_or_post == "post":
            plt.figure(figsize=(4, 5))
            res = [dist_stable, dist_inc]
            bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                                labels=["Stable", "Increasing"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            colors = ["magenta", 'orange']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            plt.ylabel("Distance to closest goal (cm)")
            plt.grid(color="grey", axis="y")
            print(mannwhitneyu(dist_stable, dist_inc, alternative="less"))
            plt.ylim(0,60)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "place_field_to_goal_distance_post.svg"), transparent="True")
            else:
                plt.show()

    def cheeseboard_firing_rates_during_swr(self, time_bin_size=0.01, experiment_phase=None):
        """
        compares firing rates during SWR for stable/decreasing/increasing cells

        @param time_bin_size: which time bin size (in s) to use for the computation of firing rates
        :type time_bin_size: float
        @param experiment_phase: PRE (["learning_cheeseboard_1"]) or POST (["learning_cheeseboard_2"])
        :type experiment_phase: str
        """

        if experiment_phase is None:
            experiment_phase = ["learning_cheeseboard_1"]

        swr_mean_firing_rates_stable = []
        swr_mean_firing_rates_dec = []
        swr_mean_firing_rates_inc = []

        for session in self.session_list:
            stable, dec, inc = session.cheese_board(experiment_phase=
                                                    experiment_phase,
                                                    data_to_use="ext").firing_rates_during_swr(time_bin_size=time_bin_size,
                                                                                               plotting=False)
            swr_mean_firing_rates_stable.append(stable)
            swr_mean_firing_rates_dec.append(dec)
            swr_mean_firing_rates_inc.append(inc)

        swr_mean_firing_rates_stable = np.hstack(swr_mean_firing_rates_stable)
        swr_mean_firing_rates_dec = np.hstack(swr_mean_firing_rates_dec)
        swr_mean_firing_rates_inc = np.hstack(swr_mean_firing_rates_inc)

        p_stable = 1. * np.arange(swr_mean_firing_rates_stable.shape[0]) / (swr_mean_firing_rates_stable.shape[0] - 1)
        p_inc = 1. * np.arange(swr_mean_firing_rates_inc.shape[0]) / (swr_mean_firing_rates_inc.shape[0] - 1)
        p_dec = 1. * np.arange(swr_mean_firing_rates_dec.shape[0]) / (swr_mean_firing_rates_dec.shape[0] - 1)

        plt.plot(np.sort(swr_mean_firing_rates_stable), p_stable, color="violet", label="stable")
        plt.plot(np.sort(swr_mean_firing_rates_dec), p_dec, color="turquoise", label="dec")
        plt.plot(np.sort(swr_mean_firing_rates_inc), p_inc, color="orange", label="inc")
        plt.ylabel("cdf")
        plt.xlabel("Mean firing rate during SWRs")
        plt.legend()
        plt.show()

    def cheeseboard_firing_rates_gain_during_swr(self, time_bin_size=0.01, threshold_stillness=10,
                                                 experiment_phase=None, save_fig=False, threshold_firing=0.1):
        """
        Checks how much cells increase their firing rates during SWR

        :param time_bin_size: which time bin size to use
        :type time_bin_size: float
        :param threshold_stillness: speed threshold (cm/s) to detect stillness periods
        :type threshold_stillness: int
        :param experiment_phase: which experiment phase to use (e.g. ["learning_cheeseboard_1"]
        :type experiment_phase: list
        :param save_fig: whether to save figure as .svg
        :type save_fig: bool
        :param threshold_firing: cells below this firing threshold (in Hz) are excluded
        :type threshold_firing: int
        """
        if experiment_phase is None:
            experiment_phase = ["learning_cheeseboard_1"]

        swr_gain_stable = []
        swr_gain_dec = []
        swr_gain_inc = []

        for session in self.session_list:
            stable, dec, inc = session.cheese_board(experiment_phase=experiment_phase,
                                                    data_to_use="ext").firing_rates_gain_during_swr(
                time_bin_size=time_bin_size, plotting=False, threshold_stillness=threshold_stillness,
                threshold_firing=threshold_firing)
            swr_gain_stable.append(stable)
            swr_gain_dec.append(dec)
            swr_gain_inc.append(inc)

        swr_gain_stable = np.hstack(swr_gain_stable)
        swr_gain_dec = np.hstack(swr_gain_dec)
        swr_gain_inc = np.hstack(swr_gain_inc)

        # filter out nan
        swr_gain_stable = swr_gain_stable[~np.isnan(swr_gain_stable)]
        swr_gain_dec = swr_gain_dec[~np.isnan(swr_gain_dec)]
        swr_gain_inc = swr_gain_inc[~np.isnan(swr_gain_inc)]

        # swr_gain_stable_log = np.log(swr_gain_stable+1e-15)
        # swr_gain_dec_log = np.log(swr_gain_dec+1e-15)
        # swr_gain_inc_log = np.log(swr_gain_inc+1e-15)

        swr_gain_stable_mean = np.mean(swr_gain_stable)
        swr_gain_stable_std = np.std(swr_gain_stable)
        swr_gain_dec_mean = np.mean(swr_gain_dec)
        swr_gain_dec_std = np.std(swr_gain_dec)
        swr_gain_inc_mean = np.mean(swr_gain_inc)
        swr_gain_inc_std = np.std(swr_gain_inc)
        if save_fig:
            plt.style.use('default')
            c="black"
        else:
            c="white"
        plt.figure(figsize=(3,4))
        plt.bar([1,2,3], [swr_gain_stable_mean, swr_gain_dec_mean, swr_gain_inc_mean],
                color=["violet", "turquoise", "orange"],
                yerr=[swr_gain_stable_std/np.sqrt(swr_gain_stable.shape[0]),
                      swr_gain_dec_std/np.sqrt(swr_gain_dec.shape[0]),
                      swr_gain_inc_std/np.sqrt(swr_gain_inc.shape[0])],
                ecolor=c, capsize=10)
        plt.xticks([1,2,3],["persistent", "decreasing", "increasing"], rotation=45)
        plt.ylabel("SWR Firing gain \n (mean+-sem)")
        plt.ylim(0, 7.5)
        plt.hlines(6.5, 1, 1.9, color=c)
        plt.text(1.3, 6.6, "n.s.")
        plt.hlines(6.5, 2.1, 3, color=c)
        plt.text(2.3, 6.6, "n.s.")
        plt.hlines(7, 1, 3, color=c)
        plt.text(1.8, 7.1, "n.s.")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "within_swr_gain.svg"), transparent="True")
        else:
            plt.show()

        p_stable = 1. * np.arange(swr_gain_stable.shape[0]) / (swr_gain_stable.shape[0] - 1)
        p_inc = 1. * np.arange(swr_gain_inc.shape[0]) / (swr_gain_inc.shape[0] - 1)
        p_dec = 1. * np.arange(swr_gain_dec.shape[0]) / (swr_gain_dec.shape[0] - 1)

        print("Two-sided: stable vs. inc:")
        print(mannwhitneyu(swr_gain_stable, swr_gain_inc))
        print("Two-sided: stable vs. dec:")
        print(mannwhitneyu(swr_gain_stable, swr_gain_dec))
        print("Two-sided: inc vs. dec:")
        print(mannwhitneyu(swr_gain_inc, swr_gain_dec))


        plt.plot(np.sort(swr_gain_stable), p_stable, color="violet", label="stable")
        plt.plot(np.sort(swr_gain_dec), p_dec, color="turquoise", label="dec")
        plt.plot(np.sort(swr_gain_inc), p_inc, color="orange", label="inc")
        plt.ylabel("cdf")
        plt.xlabel("Within ripple - firing rate gain")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def cheeseboard_burstiness(self, window_s=0.008, experiment_phase=None, save_fig=False):
        """
        Checks how much cells increase their firing rates during SWR

        :param time_bin_size: which time bin size to use
        :type time_bin_size: float
        :param threshold_stillness: speed threshold (cm/s) to detect stillness periods
        :type threshold_stillness: int
        :param experiment_phase: which experiment phase to use (e.g. ["learning_cheeseboard_1"]
        :type experiment_phase: list
        :param save_fig: whether to save figure as .svg
        :type save_fig: bool
        :param threshold_firing: cells below this firing threshold (in Hz) are excluded
        :type threshold_firing: int
        """
        if experiment_phase is None:
            experiment_phase = ["learning_cheeseboard_1"]

        burstiness_stable = []
        burstiness_dec = []
        burstiness_inc = []

        for session in self.session_list:
            stable, dec, inc = session.cheese_board(experiment_phase=experiment_phase).get_burstiness_subsets(plot_for_control=False,
                                                                                                              window_s=window_s)
            burstiness_stable.append(stable)
            burstiness_dec.append(dec)
            burstiness_inc.append(inc)

        burstiness_stable = np.hstack(burstiness_stable)
        burstiness_dec = np.hstack(burstiness_dec)
        burstiness_inc = np.hstack(burstiness_inc)

        # filter out nan
        burstiness_stable = burstiness_stable[~np.isnan(burstiness_stable)]
        burstiness_dec = burstiness_dec[~np.isnan(burstiness_dec)]
        burstiness_inc = burstiness_inc[~np.isnan(burstiness_inc)]

        # swr_gain_stable_log = np.log(swr_gain_stable+1e-15)
        # swr_gain_dec_log = np.log(swr_gain_dec+1e-15)
        # swr_gain_inc_log = np.log(swr_gain_inc+1e-15)

        if save_fig:
            plt.style.use('default')
            c="black"
        else:
            c="white"
        res =[burstiness_stable, burstiness_dec, burstiness_inc]
        plt.figure(figsize=(4,4))
        bplot=plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                          labels=["persistent", "decreasing", "increasing"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        plt.ylim(-0.01,0.1)
        y_base=0.06
        n_comp =3
        plt.hlines(y_base, 1, 1.9, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/n_comp:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/n_comp:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/n_comp:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/n_comp:
            plt.text(1.4, y_base, "*", color=c)
        plt.hlines(y_base, 2.1, 2.9, color=c)
        if mannwhitneyu(res[2], res[1])[1] > 0.05/n_comp:
            plt.text(2.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[2], res[1])[1] < 0.001/n_comp:
            plt.text(2.4, y_base, "***", color=c)
        elif mannwhitneyu(res[2], res[1])[1] < 0.01/n_comp:
            plt.text(2.4, y_base, "**", color=c)
        elif mannwhitneyu(res[2], res[1])[1] < 0.05/n_comp:
            plt.text(2.4, y_base, "*", color=c)
        # first vs. second half NREM
        y_base=0.08
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05/n_comp:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001/n_comp:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01/n_comp:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05/n_comp:
            plt.text(2, y_base, "*", color=c)
        for patch, color in zip(bplot['boxes'], ["violet", "turquoise", "orange"]):
            patch.set_facecolor(color)

        plt.ylabel("burstiness")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("burstiness_subsets_"+experiment_phase[0]+".svg", transparent="True")
        else:
            plt.show()

    def phase_preference_analysis(self, oscillation="theta", tetrode=10, sleep_or_awake="awake"):
        """
        Checks firing preference with a certain oscillation

        :param oscillation: which oscillation to look at
        :type oscillation: str
        :param tetrode: which tetrode to use
        :type tetrode: int
        :param sleep_or_awake: analysis during sleep or awake
        :type sleep_or_awake: str
        """
        all_positive_angles_stable = []
        all_positive_angles_dec = []
        all_positive_angles_inc = []

        for session in self.session_list:
            if sleep_or_awake == "awake":
                stable, dec, inc = \
                    session.cheese_board(experiment_phase=["learning_cheeseboard_1"],
                                         data_to_use="ext").phase_preference_analysis(tetrode=tetrode,
                                                                                      plotting=False,
                                                                                      oscillation=oscillation)
            elif sleep_or_awake == "sleep":
                stable, dec, inc = \
                    session.sleep(experiment_phase=["sleep_long_1"],
                                  data_to_use="ext").phase_preference_analysis(tetrode=tetrode,
                                                                               plotting=False,
                                                                               oscillation=oscillation)
            else:
                raise Exception("Need to define sleep or awake")

            all_positive_angles_stable.append(stable)
            all_positive_angles_dec.append(dec)
            all_positive_angles_inc.append(inc)

        all_positive_angles_stable = np.hstack(all_positive_angles_stable)
        all_positive_angles_dec = np.hstack(all_positive_angles_dec)
        all_positive_angles_inc = np.hstack(all_positive_angles_inc)

        bins_number = 10  # the [0, 360) interval will be subdivided into this
        # number of equal bins
        bins = np.linspace(0.0, 2 * np.pi, bins_number + 1)
        angles = all_positive_angles_stable
        n, _, _ = plt.hist(angles, bins, density=True)
        plt.title("stable")
        plt.show()
        plt.clf()
        width = 2 * np.pi / bins_number
        ax = plt.subplot(1, 1, 1, projection='polar')
        bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
        for bar in bars:
            bar.set_alpha(0.5)
        ax.set_title("stable cells")
        plt.show()

        angles = all_positive_angles_dec
        n, _, _ = plt.hist(angles, bins, density=True)
        plt.title("dec")
        plt.show()
        plt.clf()
        width = 2 * np.pi / bins_number
        ax = plt.subplot(1, 1, 1, projection='polar')
        bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
        for bar in bars:
            bar.set_alpha(0.5)
        ax.set_title("dec. cells")
        plt.show()

        angles = all_positive_angles_inc
        n, _, _ = plt.hist(angles, bins, density=True)
        plt.title("inc")
        plt.show()

        plt.clf()
        width = 2 * np.pi / bins_number
        ax = plt.subplot(1, 1, 1, projection='polar')
        bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
        for bar in bars:
            bar.set_alpha(0.5)
        ax.set_title("inc. cells")
        plt.show()

    def cheeseboard_post_occupancy_around_goals(self, save_fig=False):
        """
        Occupancy around goals in POST

        :param save_fig: save as .svg
        :type save_fig: bool
        """
        around_goals = []
        wo_goals = []

        for session in self.session_list:
            occ_around_goals_per_cm2, occ_wo_goals_per_cm2 = session.cheese_board(experiment_phase=
                                                                                  ["learning_cheeseboard_2"]).occupancy_around_goals()
            around_goals.append(occ_around_goals_per_cm2)
            wo_goals.append(occ_wo_goals_per_cm2)

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        plt.figure(figsize=(4, 5))
        res = [around_goals, wo_goals]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["Around goals", "Away from goals"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["gray", 'gray']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Occupancy per spatial bin (s/m2)")
        plt.grid(color="grey", axis="y")
        print(mannwhitneyu(around_goals, wo_goals))
        plt.ylim(0, 0.3)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "occupancy_post_all_sessions.svg"), transparent="True")
        else:
            plt.show()

    def cheeseboard_decode_location_phmm_cross_validated(self, pre_or_post="pre", save_fig=False, trials_for_test=1):
        """
        evaluate basic statistics of model and compare to data
        """
        # check if correct time bin size is used
        if not self.params.time_bin_size==0.1:
            raise Exception("Cannot decode location with time_bin_size != 0.1s")


        if pre_or_post == "pre":
            exp_phase = ["learning_cheeseboard_1"]
        elif pre_or_post == "post":
            exp_phase = ["learning_cheeseboard_2"]
        else:
            raise Exception("Select either pre or post")

        errors = []

        for session in self.session_list:
            er = \
                session.cheese_board(experiment_phase=exp_phase).decode_location_phmm_cross_validated(
                    trials_for_test=trials_for_test)
            errors.append(er)

        errors = np.hstack(errors)
        errors = errors[~np.isnan(errors)]

        cmap_errors = matplotlib.cm.get_cmap('Reds_r')

        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(4,4))
        N, bins, patches = plt.hist(errors, bins=25, edgecolor='white', color="gray", density=True)
        # plt.yscale("log")
        # plt.ylim(0,1.1)
        plt.xlabel("Error, cross-validated (cm)")
        plt.ylabel("Density")
        for i, error_amount in enumerate(np.linspace(0, np.max(errors), 25)):
            patches[i].set_facecolor(cmap_errors((error_amount-np.min(errors))/(np.max(errors-np.min(errors)))))
        plt.xlim(0, 100)
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "phmm_spatial_decoding_error.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        return errors

    def cheeseboard_decode_location_bayes_cross_validated(self, pre_or_post="pre", save_fig=False,
                                                          trials_for_test=1, spatial_resolution_rate_map=1, n_bins=40):
        """
        evaluate basic statistics of model and compare to data
        """

        if pre_or_post == "pre":
            exp_phase = ["learning_cheeseboard_1"]
        elif pre_or_post == "post":
            exp_phase = ["learning_cheeseboard_2"]
        else:
            raise Exception("Select either pre or post")

        errors = []

        for session in self.session_list:
            er = \
                session.cheese_board(experiment_phase=exp_phase).decode_location_bayes_cross_validated(
                    trials_for_test=trials_for_test, spatial_resolution_rate_map=spatial_resolution_rate_map)
            errors.append(er)

        errors = np.hstack(errors)

        errors = errors[~np.isnan(errors)]

        cmap_errors = matplotlib.cm.get_cmap('Reds_r')

        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(4,4))
        N, bins, patches = plt.hist(errors, bins=n_bins, edgecolor='white', color="gray", density=True)
        # plt.yscale("log")
        # plt.ylim(0,1.1)
        plt.xlabel("Error, cross-validated (cm)")
        plt.ylabel("Density")
        plt.xlim(0, 100)
        plt.tight_layout()
        for i, error_amount in enumerate(np.linspace(0, np.max(errors), n_bins)):
            patches[i].set_facecolor(cmap_errors((error_amount-np.min(errors))/(np.max(errors-np.min(errors)))))

        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "bayes_spatial_decoding_error.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        return errors

    def cheeseboard_decode_bayes_vs_phmm(self):

        errors_phmm = self.cheeseboard_decode_location_phmm_cross_validated()
        errors_bayes = self.cheeseboard_decode_location_bayes_cross_validated()

        plt.figure(figsize=(2,4))
        c = "white"
        if save_fig:
            plt.style.use('default')
            c = "black"
        res = [errors_phmm, errors_bayes]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["pHMM", "Bayes"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.ylim(-1, 80)
        colors = ["magenta", 'gray']
        plt.xticks(rotation=45)
        plt.ylabel("Decoding error (cm)")
        plt.tight_layout()
        y_base=70
        plt.hlines(y_base, 1,2)
        if mannwhitneyu(errors_phmm, errors_bayes)[1] < 0.01:
            plt.text(1.4, y_base, "**")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("location_decoding_phmm_bayes_stats.svg", transparent="True")
        else:
            plt.show()


    # </editor-fold>

    # <editor-fold desc="PRE and POST">
    def pre_post_cheeseboard_remapping_stable_cells(self, lump_data_together=True, save_fig=False):
        """
        Checks remapping of stable cells between PRE and POST using (spatial) population vector similarity or single
        cell rate maps

        :param lump_data_together: lump data from all sessions together
        :type lump_data_together: bool
        :param save_fig: whether to save figure
        :type save_fig: bool
        """
        if lump_data_together:
            # go trough all sessions to collect results
            remapping_stable_list = []
            remapping_shuffle_list = []
            remapping_pv_list = []
            remapping_pv_shuffle_list = []
            for session in self.session_list:
                stable, shuffle, pv, pv_shuffle = session.pre_post().remapping(plot_results=False,
                                                                                           return_distribution=True)
                remapping_stable_list.append(stable)
                remapping_shuffle_list.append(shuffle.flatten())
                remapping_pv_list.append(pv)
                remapping_pv_shuffle_list.append(pv_shuffle)

            # ----------------------------------------------------------------------------------------------------------
            # compute for population vectors
            # ----------------------------------------------------------------------------------------------------------
            remapping_pv = np.hstack((remapping_pv_list))
            remapping_pv_shuffle = np.hstack((remapping_pv_shuffle_list))

            remapping_pv = remapping_pv[~np.isnan(remapping_pv)]
            remapping_pv_shuffle = remapping_pv_shuffle[~np.isnan(remapping_pv_shuffle)]

            print("MWU for PV remapping:"+str(mannwhitneyu(remapping_pv, remapping_pv_shuffle)[1]))
            c = "white"
            if save_fig:
                plt.style.use('default')
                c = "black"
            plt.figure(figsize=(4, 5))
            res = [remapping_pv, remapping_pv_shuffle]
            bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                                labels=["Population vectors stable cells", "Shuffle"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            colors = ["magenta", 'gray']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            plt.ylabel("Pearson R: PRE map - POST map")
            plt.yticks([-0.5, 0, 0.5, 1])
            plt.ylim(-0.5, 1)
            plt.grid(color="grey", axis="y")

            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "persistent_map.svg"), transparent="True")
            else:
                plt.title("Remapping using Population Vectors")
                plt.show()

            # ----------------------------------------------------------------------------------------------------------
            # compute for single cells
            # ----------------------------------------------------------------------------------------------------------
            remapping_stable_arr = np.expand_dims(np.hstack((remapping_stable_list)),1)
            remapping_shuffle_arr = np.expand_dims(np.hstack((remapping_shuffle_list)),1)

            remapping_stable_arr = remapping_stable_arr[~np.isnan(remapping_stable_arr)]
            remapping_shuffle_arr = remapping_shuffle_arr[~np.isnan(remapping_shuffle_arr)]

            c = "black"
            res = [remapping_stable_arr, remapping_shuffle_arr]
            bplot = plt.boxplot(res, positions=[1,2], patch_artist=True,
                                labels=["Stable", "Shuffle"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c),showfliers=False
                                )
            colors = ["magenta", 'gray']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            plt.title("Remapping using single cells")
            plt.ylabel("Pearson R: PRE map - POST map")
            plt.grid(color="grey", axis="y")
            plt.show()

        else:

            # go trough all sessions to collect results
            perc_stable = []
            p_value_ks = []
            for session in self.session_list:
                perc, p = session.pre_post().remapping(plot_results=False)
                perc_stable.append(perc)
                p_value_ks.append(p)
                # res.append(session.pre_post().pre_post_firing_rates())

            perc_stable = np.vstack(perc_stable)
            plt.scatter(range(len(self.session_list)), perc_stable)
            plt.xlabel("SESSION ID")
            plt.ylabel("%STABLE CELLS WITH CONSTANT RATE MAPS")
            plt.title("RATE MAP STABILITY PRE - POST: %STABLE")
            plt.grid(c="gray")
            plt.ylim(0,100)
            plt.show()

            p_value_ks = np.vstack(p_value_ks)
            plt.scatter(range(len(self.session_list)), p_value_ks)
            plt.xlabel("SESSION ID")
            plt.ylabel("P-VALUE KS")
            plt.title("RATE MAP STABILITY PRE - POST: KS")
            plt.grid(c="gray")
            plt.ylim(0,max(p_value_ks))
            plt.show()

    def pre_post_cheeseboard_remapping_all_subsets(self, save_fig=False, spatial_resolution=10,
                                                   normalized=False, min_mean_firing_rate=None):
        """
        Checks remapping of cells between PRE and POST using (spatial) population vector similarity or single
        cell rate maps

        :param save_fig: whether to save figure
        :type save_fig: bool
        """

        # go trough all sessions to collect results
        remapping_pv_stable = []
        remapping_pv_dec = []
        remapping_pv_inc = []
        remapping_stable = []
        remapping_dec = []
        remapping_inc = []
        stable_rate_map_stable = []
        stable_rate_map_dec = []
        stable_rate_map_inc = []

        for session in self.session_list:
            remapping_pv_stable_, remapping_pv_dec_, remapping_pv_inc_, remapping_stable_, remapping_dec_, \
                remapping_inc_, stable_rate_map_stable_, stable_rate_map_dec_, stable_rate_map_inc_ = \
                session.pre_post().remapping_pre_post_all_subsets(plotting=False, spatial_resolution=spatial_resolution,
                                                                  normalized=normalized,
                                                                  min_mean_firing_rate=min_mean_firing_rate)
            remapping_pv_stable.append(remapping_pv_stable_)
            remapping_pv_dec.append(remapping_pv_dec_)
            remapping_pv_inc.append(remapping_pv_inc_)
            remapping_stable.append(remapping_stable_)
            remapping_dec.append(remapping_dec_)
            remapping_inc.append(remapping_inc_)
            stable_rate_map_stable.append(stable_rate_map_stable_)
            stable_rate_map_dec.append(stable_rate_map_dec_)
            stable_rate_map_inc.append(stable_rate_map_inc_)

        # ----------------------------------------------------------------------------------------------------------
        # compute for population vectors
        # ----------------------------------------------------------------------------------------------------------
        remapping_pv_stable = np.hstack(remapping_pv_stable)
        remapping_pv_dec = np.hstack(remapping_pv_dec)
        remapping_pv_inc = np.hstack(remapping_pv_inc)
        remapping_stable = np.hstack(remapping_stable)
        remapping_dec = np.hstack(remapping_dec)
        remapping_inc = np.hstack(remapping_inc)
        stable_rate_map_stable = np.hstack(stable_rate_map_stable)
        stable_rate_map_dec = np.hstack(stable_rate_map_dec)
        stable_rate_map_inc = np.hstack(stable_rate_map_inc)

        if save_fig:
            c = "black"
            plt.style.use('default')
        else:
            c = "white"
        plt.figure(figsize=(3, 4))
        bplot = plt.boxplot([remapping_pv_stable, remapping_pv_dec, remapping_pv_inc],
                            positions=[1, 2, 3], patch_artist=True,
                            labels=["stable", "dec", "inc"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),showfliers=False)
        colors = ["violet", 'turquoise', "orange"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.grid(color="grey", axis="y")
        if normalized:
            plt.ylim(-0.6, 1.4)
            plt.ylabel("PV correlation (rate normalized) \n Acquisition - Recall")
        else:
            plt.ylim(0, 1.2)
            plt.ylabel("PV correlation: \n Acquisition - Recall")
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            if normalized:
                plt.savefig(os.path.join(save_path, "pv_correlation_pre_post_normalized.svg"), transparent="True")
            else:
                plt.savefig(os.path.join(save_path, "pv_correlation_pre_post.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # stats

        print("stable vs. dec:")
        print(mannwhitneyu(remapping_pv_stable, remapping_pv_dec)[1]*3)
        print("stable vs. inc:")
        print(mannwhitneyu(remapping_pv_stable, remapping_pv_inc)[1]*3)
        print("inc vs. dec:")
        print(mannwhitneyu(remapping_pv_inc, remapping_pv_dec)[1]*3)

        plt.figure(figsize=(3, 4))
        bplot = plt.boxplot([remapping_stable, remapping_dec, remapping_inc],
                            positions=[1, 2, 3], patch_artist=True,
                            labels=["stable", "dec", "inc"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        colors = ["violet", 'turquoise', "orange"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Rate map correlation: \n Acquisition - Recall")
        plt.grid(color="grey", axis="y")
        plt.xticks(rotation=45)
        plt.ylim(-1, 1)
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "rate_map_correlation_pre_post.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        print("Stable Place field, stable cells:")
        print(str(stable_rate_map_stable)+"%")
        print("Stable Place field, dec cells:")
        print(str(stable_rate_map_dec)+"%")
        print("Stable Place field, inc cells:")
        print(str(stable_rate_map_inc)+"%")

    def pre_post_cheeseboard_goal_coding(self):
        """
        Checks goal coding feature of different subsets

        """
        # go trough all sessions to collect results
        pre_res = []
        post_res = []
        pre_dec = []
        post_inc = []
        session_name_strings =[]
        for session in self.session_list:
            pre, post, pre_d, post_i = session.pre_post().goal_coding(plotting=False)
            pre_res.append(pre)
            post_res.append(post)
            pre_dec.append(pre_d)
            post_inc.append(post_i)
            # res.append(session.pre_post().pre_post_firing_rates())
            session_name_strings.append(session.session_name)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(range(len(self.session_list)), pre_res, label ="STABLE", color="y")
        plt.scatter(range(len(self.session_list)), pre_dec, label ="DEC", color="b")
        plt.xticks(range(len(self.session_list)), session_name_strings, rotation='vertical')
        plt.ylabel("MEDIAN GOAL CODING INDEX")
        plt.title("GOAL CODING PRE: STABLE VS. DEC")
        plt.grid(c="gray")
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(range(len(self.session_list)), post_res, label ="STABLE", color="y")
        plt.scatter(range(len(self.session_list)), post_inc, label ="INC", color="r")
        plt.xticks(range(len(self.session_list)), session_name_strings, rotation='vertical')
        plt.ylabel("MEDIAN GOAL CODING INDEX")
        plt.title("GOAL CODING POST: STABLE VS. INC")
        plt.grid(c="gray")
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(range(len(self.session_list)), pre_res, label ="PRE", color="r")
        plt.scatter(range(len(self.session_list)), post_res, label ="POST", color="b")
        plt.xticks(range(len(self.session_list)), session_name_strings, rotation='vertical')
        plt.ylabel("MEDIAN GOAL CODING INDEX")
        plt.title("GOAL CODING STABLE CELLS: PRE VS. POST")
        plt.grid(c="gray")
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()

    def pre_post_cheeseboard_goal_coding_stable_cells_phmm(self):
        """
        Analysis of goal coding of stable cells using pHMM modes

        """
        # go trough all sessions to collect results
        gain_res = []
        session_name_strings = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for session in self.session_list:
            gain = session.pre_post().pre_post_models_goal_coding(gc_threshold=0.9)
            gain_res.append(gain)
            # res.append(session.pre_post().pre_post_firing_rates())
            session_name_strings.append(session.session_name)

        plt.scatter(range(len(self.session_list)), gain_res, color="r")
        plt.xticks(range(len(self.session_list)), session_name_strings, rotation='vertical')
        plt.ylabel("GAIN IN GOAL CODING (POST - PRE)")
        plt.title("GOAL CODING GAIN FROM PRE TO POST \n USING LAMBDA AND MEAN FIRING RATE")
        plt.grid(c="gray")
        plt.show()

    def pre_post_place_field_goal_distance_stable_cells(self, save_fig=False, mean_firing_threshold=1, nr_trials=4):
        """
        computes distance between place field peak and closest goal

        @param save_fig: save as .svg
        :type: save_fig: bool
        @param mean_firing_threshold: threshold to exclude low firing cells (in Hz)
        :type mean_firing_threshold: float
        @param nr_trials: how many trials to use
        :type nr_trials: int
        """

        dist_stable_pre = []
        dist_stable_post = []

        for session in self.session_list:
            total_nr_trials = session.cheese_board(experiment_phase=["learning_cheeseboard_1"]).nr_trials
            d_s, _, _ = \
                session.cheese_board(experiment_phase=
                    ["learning_cheeseboard_1"]).place_field_goal_distance(plotting=False, mean_firing_threshold=
                                                                           mean_firing_threshold, trials_to_use=
                                                                           range(total_nr_trials-nr_trials,
                                                                           total_nr_trials))
            dist_stable_pre.append(d_s)

        for session in self.session_list:
            d_s, _, _ = \
                session.cheese_board(experiment_phase=
                    ["learning_cheeseboard_2"]).place_field_goal_distance(plotting=False, mean_firing_threshold=
                                                                          mean_firing_threshold, trials_to_use=
                                                                          range(nr_trials))
            dist_stable_post.append(d_s)

        dist_stable_pre = np.hstack(dist_stable_pre)
        dist_stable_post = np.hstack(dist_stable_post)

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        plt.figure(figsize=(4, 5))
        res = [dist_stable_pre, dist_stable_post]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["Last "+str(nr_trials)+"\n trials PRE", "First "+str(nr_trials)+"\n trials POST"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["gray", 'black']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Distance to closest goal (cm)")
        plt.grid(color="grey", axis="y")
        print(mannwhitneyu(dist_stable_pre, dist_stable_post, alternative="less"))
        plt.ylim(0, 60)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "place_field_to_goal_distance_pre_post_stable.svg"), transparent="True")
        else:
            plt.show()

    def pre_post_cheeseboard_nr_goals(self, single_cells=True, mean_firing_thresh=None):
        """
        Checks how many goals a cell codes for

        :param single_cells: whether to look at single cells or all cells together
        :type single_cells: bool
        :param mean_firing_thresh: cells with firing rates (Hz) below this threshold are excluded
        :type mean_firing_thresh: float
        """
        # go trough all sessions to collect results
        pre_stable = []
        pre_dec = []
        post_stable = []
        post_inc = []
        session_name_strings = []

        for session in self.session_list:
            pre_s, pre_d, post_s, post_i = session.pre_post().nr_goals_coded(plotting=False, single_cells=single_cells,
                                                                             mean_firing_thresh=mean_firing_thresh)
            pre_stable.append(pre_s)
            pre_dec.append(pre_d)
            post_stable.append(post_s)
            post_inc.append(post_i)

            # res.append(session.pre_post().pre_post_firing_rates())
            session_name_strings.append(session.session_name)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(range(len(self.session_list)), pre_stable, label ="STABLE", color="y")
        plt.scatter(range(len(self.session_list)), pre_dec, label ="DEC", color="b")
        plt.xticks(range(len(self.session_list)), session_name_strings, rotation='vertical')
        if single_cells:
            plt.ylabel("MEDIAN NR. GOALS CODED")
            plt.title("NR. GOALS CODED PER CELL: PRE")
        else:
            plt.ylabel("NR. GOALS CODED")
            plt.title("NR. GOALS FOR SUBSET: PRE")
        plt.grid(c="gray")
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(range(len(self.session_list)), post_stable, label="STABLE", color="y")
        plt.scatter(range(len(self.session_list)), post_inc, label="INC", color="r")
        plt.xticks(range(len(self.session_list)), session_name_strings, rotation='vertical')
        if single_cells:
            plt.ylabel("MEDIAN NR. GOALS CODED")
            plt.title("NR. GOALS CODED PER CELL: POST")
        else:
            plt.ylabel("NR. GOALS CODED")
            plt.title("NR. GOALS FOR SUBSET: POST")
        plt.grid(c="gray")
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()

    def pre_post_cheeseboard_spatial_information(self, spatial_resolution=2, info_measure="sparsity",
                                                 min_firing_rate=0.1, save_fig=False, plotting=True):
        """
        computes spatial information of different subsets of cells (stable, increasing, decreasing)


        :param plotting: whether to plot results
        :type plotting: bool
        :param mean_firing_thresh: cells with mean firing rate below (Hz) are excluded
        :type mean_firing_thresh: float
        :param remove_nan: whether to remove nan
        :type remove_nan: bool
        :param spatial_resolution: in cm to use for spatial binning
        :type spatial_resolution: int
        :param info_measure: which spatial information measure to use ("sparsity", "skaggs_info")
        :type info_measure: str
        :param lump_data_together: lump cells from all sessions together or keep them separate
        :type lump_data_together: bool
        :param save_fig: whether to save figure
        :type save_fig: bool
        """

        # number of comparisons --> for Bonferoni correction
        nr_comp = 9

        pre_stable_list = []
        post_stable_list = []
        pre_dec_list = []
        post_inc_list = []
        post_dec_list = []
        pre_inc_list = []
        mean_firing_pre_stable = []
        mean_firing_pre_dec = []
        mean_firing_pre_inc = []
        mean_firing_post_stable = []
        mean_firing_post_dec = []
        mean_firing_post_inc = []
        for session in self.session_list:
            pre_stable, post_stable, pre_dec, post_inc, post_dec, pre_inc, _, _, \
                _ = \
                session.pre_post().spatial_information_per_cell(plotting=False,
                                                                spatial_resolution=spatial_resolution,
                                                                info_measure=info_measure,
                                                                min_firing_rate=min_firing_rate)

            pre_stable_list.append(pre_stable)
            post_stable_list.append(post_stable)
            pre_dec_list.append(pre_dec)
            post_inc_list.append(post_inc)
            post_dec_list.append(post_dec)
            pre_inc_list.append(pre_inc)

        pre_stable_arr = np.hstack(pre_stable_list)
        post_stable_arr = np.hstack(post_stable_list)
        pre_dec_arr = np.hstack(pre_dec_list)
        post_inc_arr = np.hstack(post_inc_list)
        post_dec_arr = np.hstack(post_dec_list)
        pre_inc_arr = np.hstack(pre_inc_list)

        # do stats tests
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        plt.figure(figsize=(5,5))

        bplot = plt.boxplot([pre_stable_arr, pre_dec_arr, pre_inc_arr, post_stable_arr,
                             post_dec_arr, post_inc_arr], positions=[1, 2, 3, 4, 5, 6], patch_artist=True,
                            labels=["Acq_pers", "Acq_dec", "Acq_inc", "Rec_pers", "Rec_dec", "Rec_inc"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["violet", 'turquoise', "orange", "violet", 'turquoise', "orange"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel(info_measure)
        # plt.grid(color="grey", axis="y")
        plt.xticks(rotation=45)
        plt.tight_layout()
        if info_measure == "skaggs_second":
            y_base = 2.6
            plt.hlines(y_base, 1, 1.95, colors=c)
            if mannwhitneyu(pre_stable_arr, pre_dec_arr)[1] < 0.001/nr_comp:
                plt.text(1.4, y_base+0.05, "***")
            elif mannwhitneyu(pre_stable_arr, pre_dec_arr)[1] < 0.01/nr_comp:
                plt.text(1.4, y_base+0.05, "**")
            elif mannwhitneyu(pre_stable_arr, pre_dec_arr)[1] < 0.05/nr_comp:
                plt.text(1.4, y_base+0.05, "*")
            else:
                plt.text(1.4, y_base+0.05, "n.s.")
            plt.hlines(y_base, 2.05, 3, colors=c)
            if mannwhitneyu(pre_dec_arr, pre_inc_arr)[1] < 0.001/nr_comp:
                plt.text(2.4, y_base+0.05, "***")
            elif mannwhitneyu(pre_dec_arr, pre_inc_arr)[1] < 0.01/nr_comp:
                plt.text(2.4, y_base+0.05, "**")
            elif mannwhitneyu(pre_dec_arr, pre_inc_arr)[1] < 0.05/nr_comp:
                plt.text(2.4, y_base+0.05, "*")
            else:
                plt.text(2.4, y_base+0.05, "n.s.")
            plt.hlines(y_base+0.2, 1, 3, colors=c)
            if mannwhitneyu(pre_stable_arr, pre_inc_arr)[1] < 0.001/nr_comp:
                plt.text(1.9, y_base+0.25, "***")
            elif mannwhitneyu(pre_stable_arr, pre_inc_arr)[1] < 0.01/nr_comp:
                plt.text(1.9, y_base+0.25, "**")
            elif mannwhitneyu(pre_stable_arr, pre_inc_arr)[1] < 0.05/nr_comp:
                plt.text(1.9, y_base+0.25, "*")
            else:
                plt.text(1.9, y_base+0.25, "n.s.")
            plt.hlines(y_base, 4, 4.95, colors=c)
            if mannwhitneyu(post_stable_arr, post_dec_arr)[1] < 0.001/nr_comp:
                plt.text(4.4, y_base+0.05, "***")
            elif mannwhitneyu(post_stable_arr, post_dec_arr)[1] < 0.01/nr_comp:
                plt.text(4.4, y_base+0.05, "**")
            elif mannwhitneyu(post_stable_arr, post_dec_arr)[1] < 0.05/nr_comp:
                plt.text(4.4, y_base+0.05, "*")
            else:
                plt.text(4.4, y_base+0.05, "n.s.")
            plt.hlines(y_base, 5.05, 6, colors=c)
            if mannwhitneyu(post_stable_arr, post_inc_arr)[1] < 0.001/nr_comp:
                plt.text(5.4, y_base+0.05, "***")
            elif mannwhitneyu(post_stable_arr, post_inc_arr)[1] < 0.01/nr_comp:
                plt.text(5.4, y_base+0.05, "**")
            elif mannwhitneyu(post_stable_arr, post_inc_arr)[1] < 0.05/nr_comp:
                plt.text(5.4, y_base+0.05, "*")
            else:
                plt.text(5.4, y_base+0.05, "n.s.")
            plt.hlines(y_base+0.2, 4, 6, colors=c)
            if mannwhitneyu(post_stable_arr, post_inc_arr)[1] < 0.001/nr_comp:
                plt.text(4.9, y_base+0.25, "***")
            elif mannwhitneyu(post_stable_arr, post_inc_arr)[1] < 0.01/nr_comp:
                plt.text(4.9, y_base+0.25, "**")
            elif mannwhitneyu(post_stable_arr, post_inc_arr)[1] < 0.05/nr_comp:
                plt.text(4.9, y_base+0.25, "*")
            else:
                plt.text(4.9, y_base+0.25, "n.s.")

            plt.hlines(y_base+0.4, 1, 4, colors=c)
            if mannwhitneyu(pre_stable_arr, post_stable_arr)[1] < 0.001/nr_comp:
                plt.text(2.4, y_base+0.45, "***")
            elif mannwhitneyu(pre_stable_arr, post_stable_arr)[1] < 0.01/nr_comp:
                plt.text(2.4, y_base+0.45, "**")
            elif mannwhitneyu(pre_stable_arr, post_stable_arr)[1] < 0.05/nr_comp:
                plt.text(2.4, y_base+0.45, "*")
            else:
                plt.text(2.4, y_base+0.45, "n.s.")
            plt.hlines(y_base+0.6, 2, 5, colors=c)
            if mannwhitneyu(pre_dec_arr, post_dec_arr)[1] < 0.001/nr_comp:
                plt.text(3.4, y_base+0.65, "***")
            elif mannwhitneyu(pre_dec_arr, post_dec_arr)[1] < 0.01/nr_comp:
                plt.text(3.4, y_base+0.65, "**")
            elif mannwhitneyu(pre_dec_arr, post_dec_arr)[1] < 0.05/nr_comp:
                plt.text(3.4, y_base+0.65, "*")
            else:
                plt.text(3.4, y_base+0.65, "n.s.")
            plt.hlines(y_base+0.8, 3, 6, colors=c)
            if mannwhitneyu(pre_inc_arr, post_inc_arr)[1] < 0.001/nr_comp:
                plt.text(4.4, y_base+0.85, "***")
            elif mannwhitneyu(pre_inc_arr, post_inc_arr)[1] < 0.01/nr_comp:
                plt.text(4.4, y_base+0.85, "**")
            elif mannwhitneyu(pre_inc_arr, post_inc_arr)[1] < 0.05/nr_comp:
                plt.text(4.4, y_base+0.85, "*")
            else:
                plt.text(4.4, y_base+0.85, "n.s.")

        elif info_measure == "sparsity":
            y_base = 1
            plt.hlines(y_base, 1, 1.95, colors=c)
            if mannwhitneyu(pre_stable_arr, pre_dec_arr)[1] < 0.001:
                plt.text(1.4, y_base + 0.02, "***")
            elif mannwhitneyu(pre_stable_arr, pre_dec_arr)[1] < 0.01:
                plt.text(1.4, y_base + 0.02, "**")
            elif mannwhitneyu(pre_stable_arr, pre_dec_arr)[1] < 0.05:
                plt.text(1.4, y_base + 0.02, "*")
            else:
                plt.text(1.4, y_base + 0.02, "n.s.")
            plt.hlines(y_base, 2.05, 3, colors=c)
            if mannwhitneyu(pre_dec_arr, pre_inc_arr)[1] < 0.001:
                plt.text(2.4, y_base + 0.02, "***")
            elif mannwhitneyu(pre_dec_arr, pre_inc_arr)[1] < 0.01:
                plt.text(2.4, y_base + 0.02, "**")
            elif mannwhitneyu(pre_dec_arr, pre_inc_arr)[1] < 0.05:
                plt.text(2.4, y_base + 0.02, "*")
            else:
                plt.text(2.4, y_base + 0.02, "n.s.")
            plt.hlines(y_base + 0.1, 1, 3, colors=c)
            if mannwhitneyu(pre_stable_arr, pre_inc_arr)[1] < 0.001:
                plt.text(1.9, y_base + 0.12, "***")
            elif mannwhitneyu(pre_stable_arr, pre_inc_arr)[1] < 0.01:
                plt.text(1.9, y_base + 0.12, "**")
            elif mannwhitneyu(pre_stable_arr, pre_inc_arr)[1] < 0.05:
                plt.text(1.9, y_base + 0.12, "*")
            else:
                plt.text(1.9, y_base + 0.02, "n.s.")
            plt.hlines(y_base, 4, 4.95, colors=c)
            if mannwhitneyu(post_stable_arr, post_dec_arr)[1] < 0.001:
                plt.text(4.4, y_base + 0.02, "***")
            elif mannwhitneyu(post_stable_arr, post_dec_arr)[1] < 0.01:
                plt.text(4.4, y_base + 0.02, "**")
            elif mannwhitneyu(post_stable_arr, post_dec_arr)[1] < 0.05:
                plt.text(4.4, y_base + 0.02, "*")
            else:
                plt.text(4.4, y_base + 0.02, "n.s.")
            plt.hlines(y_base, 5.05, 6, colors=c)
            if mannwhitneyu(post_stable_arr, post_inc_arr)[1] < 0.001:
                plt.text(5.4, y_base + 0.02, "***")
            elif mannwhitneyu(post_stable_arr, post_inc_arr)[1] < 0.01:
                plt.text(5.4, y_base + 0.02, "**")
            elif mannwhitneyu(post_stable_arr, post_inc_arr)[1] < 0.05:
                plt.text(5.4, y_base + 0.02, "*")
            else:
                plt.text(5.4, y_base + 0.02, "n.s.")
            plt.hlines(y_base + 0.1, 4, 6, colors=c)
            if mannwhitneyu(post_stable_arr, post_inc_arr)[1] < 0.001:
                plt.text(4.9, y_base + 0.12, "***")
            elif mannwhitneyu(post_stable_arr, post_inc_arr)[1] < 0.01:
                plt.text(4.9, y_base + 0.12, "**")
            elif mannwhitneyu(post_stable_arr, post_inc_arr)[1] < 0.05:
                plt.text(4.9, y_base + 0.12, "*")
            else:
                plt.text(4.9, y_base + 0.12, "n.s.")

            plt.hlines(y_base + 0.2, 1, 4, colors=c)
            if mannwhitneyu(pre_stable_arr, post_stable_arr)[1] < 0.001:
                plt.text(2.4, y_base + 0.22, "***")
            elif mannwhitneyu(pre_stable_arr, post_stable_arr)[1] < 0.01:
                plt.text(2.4, y_base + 0.22, "**")
            elif mannwhitneyu(pre_stable_arr, post_stable_arr)[1] < 0.05:
                plt.text(2.4, y_base + 0.22, "*")
            else:
                plt.text(2.4, y_base + 0.22, "n.s.")
            plt.hlines(y_base + 0.3, 2, 5, colors=c)
            if mannwhitneyu(pre_dec_arr, post_dec_arr)[1] < 0.001:
                plt.text(3.4, y_base + 0.32, "***")
            elif mannwhitneyu(pre_dec_arr, post_dec_arr)[1] < 0.01:
                plt.text(3.4, y_base + 0.32, "**")
            elif mannwhitneyu(pre_dec_arr, post_dec_arr)[1] < 0.05:
                plt.text(3.4, y_base + 0.32, "*")
            else:
                plt.text(3.4, y_base + 0.32, "n.s.")
            plt.hlines(y_base + 0.4, 3, 6, colors=c)
            if mannwhitneyu(pre_inc_arr, post_inc_arr)[1] < 0.001:
                plt.text(4.4, y_base + 0.42, "***")
            elif mannwhitneyu(pre_inc_arr, post_inc_arr)[1] < 0.01:
                plt.text(4.4, y_base + 0.42, "**")
            elif mannwhitneyu(pre_inc_arr, post_inc_arr)[1] < 0.05:
                plt.text(4.4, y_base + 0.42, "*")
            else:
                plt.text(4.4, y_base + 0.42, "n.s.")
            plt.ylim(0,1.5)
            plt.yticks([0,0.2, 0.4, 0.6, 0.8, 1])
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "spatial_information_all_sessions_"+info_measure+".svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # compute for stable cells only
        p_stable_mwu = mannwhitneyu(np.nan_to_num(post_stable_arr), np.nan_to_num(pre_stable_arr))[1]

        # compare decreasing and increasing
        p_dec_inc_mwu = mannwhitneyu(np.nan_to_num(post_inc_arr), np.nan_to_num(pre_dec_arr))[1]
        print("p-value INC-DEC: " + str(p_dec_inc_mwu))
        # sort for CDF
        pre_stable_sorted = np.sort(pre_stable_arr)
        post_stable_sorted = np.sort(post_stable_arr)
        pre_dec_sorted = np.sort(pre_dec_arr)
        post_inc_sorted = np.sort(post_inc_arr)
        pre_inc_sorted = np.sort(pre_inc_arr)
        post_dec_sorted = np.sort(post_dec_arr)

        if plotting or save_fig:
            if save_fig:
                plt.style.use('default')

            # ----------------------------------------------------------------------------------------------------------
            # PRE
            # ----------------------------------------------------------------------------------------------------------
            p_pre_stable = 1. * np.arange(pre_stable_sorted.shape[0]) / (pre_stable_sorted.shape[0] - 1)
            p_pre_dec = 1. * np.arange(pre_dec_sorted.shape[0]) / (pre_dec_sorted.shape[0] - 1)
            p_pre_inc = 1. * np.arange(pre_inc_sorted.shape[0]) / (pre_inc_sorted.shape[0] - 1)
            # plt.hlines(0.5, -0.02, 0.85, color="gray", linewidth=0.5)
            plt.plot(pre_stable_sorted, p_pre_stable, color="magenta", label="Stable")
            plt.plot(pre_dec_sorted, p_pre_dec, color="turquoise", label="Decreasing")
            # plt.plot(pre_inc_sorted, p_pre_inc, color="orange", label="Increasing")
            # plt.yticks([0, 0.25, 0.5, 0.75, 1])
            # plt.xticks([0, 0.45, 0.9])
            # plt.xlim(-0.05, .95)
            # plt.ylim(-0.05, 1.05)
            plt.ylabel("cdf")
            plt.xlabel(info_measure)
            plt.legend()
            if save_fig:
                if info_measure == "sparsity":
                    plt.xlim(-0.02, 1.02)
                elif info_measure == "skaggs_second":
                    plt.xlim(-0.2, 5.2)
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, info_measure+"_pre.svg"), transparent="True")
                plt.close()
            else:
                plt.title(
                    "PRE: STABLE CELLS vs. DEC. CELLS \n" + "MWU , p-value = " + str(np.round(p_pre_mwu_one_sided, 5)))
                plt.show()

            # ----------------------------------------------------------------------------------------------------------
            # POST
            # ----------------------------------------------------------------------------------------------------------
            # calculate the proportional values of samples
            p_post_stable = 1. * np.arange(post_stable_sorted.shape[0]) / (post_stable_sorted.shape[0] - 1)
            p_post_inc = 1. * np.arange(post_inc_sorted.shape[0]) / (post_inc_sorted.shape[0] - 1)
            p_post_dec = 1. * np.arange(post_dec_sorted.shape[0]) / (post_dec_sorted.shape[0] - 1)
            plt.plot(post_stable_sorted, p_post_stable, color="magenta", label="Stable")
            plt.plot(post_inc_sorted, p_post_inc, color="orange", label="Increasing")
            # plt.plot(post_dec_sorted, p_post_dec, color="turquoise", label="Decreasing")
            plt.ylabel("cdf")
            # plt.yticks([0, 0.25, 0.5, 0.75, 1])
            # plt.xticks([0, 0.45, 0.9])
            # plt.xlim(-0.05, 0.95)
            # plt.ylim(-0.02, 1.02)
            plt.xlabel(info_measure)
            # plt.hlines(0.5, -0.02, 0.87, color="gray", linewidth=0.5)
            plt.legend()
            if save_fig:
                if info_measure == "sparsity":
                    plt.xlim(-0.02, 1.02)
                elif info_measure == "skaggs_second":
                    plt.xlim(-0.4, 5.4)
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, info_measure+"_post.svg"), transparent="True")
                plt.close()
            else:
                plt.title(
                    "POST: STABLE CELLS vs. INC. CELLS \n" + "MWU , p-value = " + str(np.round(p_post_mwu_one_sided, 5)))
                plt.show()

            # spatial information PRE - POST for stable cells
            plt.plot(pre_stable_sorted, p_pre_stable, color="red", label="PRE")
            plt.plot(post_stable_sorted, p_post_stable, color="blue", label="POST")
            plt.ylabel("cdf")
            plt.xlabel(info_measure)
            plt.legend()
            if save_fig:
                if info_measure == "sparsity":
                    plt.xlim(-0.02, 1.02)
                elif info_measure == "skaggs_second":
                    plt.xlim(-0.2, 5.2)
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, info_measure+"_stable_pre_post.svg"), transparent="True")
                plt.close()
            else:
                plt.title("PRE-POST, STABLE CELLS \n" + "MWU , p-value = " + str(np.round(p_stable_mwu, 5)))
                plt.show()

            # spatial information increasing / decreasing cells
            # spatial information PRE - POST for stable cells
            plt.plot(pre_dec_sorted, p_pre_dec, color="#a0c4e4", label="Decreasing (PRE)")
            plt.plot(post_inc_sorted, p_post_inc, color="#f7959c", label="Increasing (POST)")
            plt.ylabel("cdf")
            # plt.yticks([0, 0.25, 0.5, 0.75, 1])
            # plt.xticks([0, 0.45, 0.9])
            # plt.xlim(-0.05, 0.95)
            plt.xlabel(info_measure)
            plt.legend()
            if save_fig:
                if info_measure == "sparsity":
                    plt.xlim(-0.02, 1.02)
                elif info_measure == "skaggs_second":
                    plt.xlim(-0.4, 5.4)
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, info_measure+"_dec_inc.svg"), transparent="True")
                plt.close()
            else:
                plt.title("INC-DEC \n" + "MWU , p-value = " + str(np.round(p_dec_inc_mwu, 5)))
                plt.show()

        return pre_stable_arr, post_stable_arr, pre_dec_arr, post_inc_arr, post_dec_arr, pre_inc_arr

    def pre_post_cheeseboard_firing_rate_changes(self, mean_or_max="mean"):
        """
        Computes firing rates in pre and post

        :param mean_or_max: whether to use max or mean firing
        :type mean_or_max: str
        """
        pre_stable_list = []
        pre_dec_list = []
        post_stable_list = []
        post_inc_list = []
        for session in self.session_list:
            pre_stable, pre_dec, post_stable, post_inc = \
                session.pre_post().firing_rate_changes(plotting=False, mean_or_max=mean_or_max)
            pre_stable_list.append(pre_stable)
            pre_dec_list.append(pre_dec)
            post_stable_list.append(post_stable)
            post_inc_list.append(post_inc)

        pre_stable_arr = np.hstack(pre_stable_list)
        post_stable_arr = np.hstack(post_stable_list)
        pre_dec_arr = np.hstack(pre_dec_list)
        post_inc_arr = np.hstack(post_inc_list)

        # sort for CDF
        pre_stable_sorted = np.sort(pre_stable_arr)
        post_stable_sorted = np.sort(post_stable_arr)
        pre_dec_sorted = np.sort(pre_dec_arr)
        post_inc_sorted = np.sort(post_inc_arr)

        p_pre_stable = 1. * np.arange(pre_stable_sorted.shape[0]) / (pre_stable_sorted.shape[0] - 1)
        p_pre_dec = 1. * np.arange(pre_dec_sorted.shape[0]) / (pre_dec_sorted.shape[0] - 1)
        # plt.hlines(0.5, -0.02, 0.85, color="gray", linewidth=0.5)
        plt.plot(pre_stable_sorted, p_pre_stable, color="#ffdba1", label="STABLE")
        plt.plot(pre_dec_sorted, p_pre_dec, color="#a0c4e4", label="DECREASING")
        # plt.yticks([0, 0.5, 1])
        # plt.xticks([0, 0.415, 0.83])
        # plt.xlim(-0.02, 0.85)
        # plt.ylim(-0.02, 1.02)
        plt.ylabel("CDF")
        plt.xlabel(mean_or_max+" firing")

        plt.legend()
        plt.show()

        p_post_stable = 1. * np.arange(post_stable_sorted.shape[0]) / (post_stable_sorted.shape[0] - 1)
        p_post_inc = 1. * np.arange(post_inc_sorted.shape[0]) / (post_inc_sorted.shape[0] - 1)
        plt.plot(post_stable_sorted, p_post_stable, color="#ffdba1", label="STABLE")
        plt.plot(post_inc_sorted, p_post_inc, color="#f7959c", label="INCREASING")
        plt.ylabel("CDF")
        # plt.yticks([0, 0.5, 1])
        # plt.xticks([0, 0.425, 0.85])
        # plt.xlim(-0.02, 0.87)
        # plt.ylim(-0.02, 1.02)
        plt.xlabel(mean_or_max+" firing")
        # plt.hlines(0.5, -0.02, 0.87, color="gray", linewidth=0.5)
        plt.legend()
        plt.show()

    def pre_post_cheeseboard_firing_around_goals(self):
        """
        Checks firing around goals in PRE and POST for stable, dec and increasing cells

        """
        pre_stable_list = []
        pre_dec_list = []
        post_stable_list = []
        post_inc_list = []
        for session in self.session_list:
            pre_stable, pre_dec, post_stable, post_inc = \
                session.pre_post().distance_peak_firing_closest_goal(plotting=False)
            pre_stable_list.append(pre_stable)
            pre_dec_list.append(pre_dec)
            post_stable_list.append(post_stable)
            post_inc_list.append(post_inc)

        pre_stable_arr = np.hstack(pre_stable_list)
        post_stable_arr = np.hstack(post_stable_list)
        pre_dec_arr = np.hstack(pre_dec_list)
        post_inc_arr = np.hstack(post_inc_list)

        # sort for CDF
        pre_stable_sorted = np.sort(pre_stable_arr)
        post_stable_sorted = np.sort(post_stable_arr)
        pre_dec_sorted = np.sort(pre_dec_arr)
        post_inc_sorted = np.sort(post_inc_arr)

        p_pre_stable = 1. * np.arange(pre_stable_sorted.shape[0]) / (pre_stable_sorted.shape[0] - 1)
        p_pre_dec = 1. * np.arange(pre_dec_sorted.shape[0]) / (pre_dec_sorted.shape[0] - 1)
        plt.plot(pre_stable_sorted, p_pre_stable, color="#ffdba1", label="STABLE")
        plt.plot(pre_dec_sorted, p_pre_dec, color="#a0c4e4", label="DECREASING")
        plt.ylabel("CDF")
        plt.xlabel("Min. distance: peak firing loc. to closest goal / cm")
        plt.title("PRE")
        plt.legend()
        plt.show()

        p_post_stable = 1. * np.arange(post_stable_sorted.shape[0]) / (post_stable_sorted.shape[0] - 1)
        p_post_inc = 1. * np.arange(post_inc_sorted.shape[0]) / (post_inc_sorted.shape[0] - 1)
        plt.plot(post_stable_sorted, p_post_stable, color="#ffdba1", label="STABLE")
        plt.plot(post_inc_sorted, p_post_inc, color="#f7959c", label="INCREASING")
        plt.ylabel("CDF")
        plt.xlabel("Min. distance: peak firing loc. to closest goal / cm")
        plt.title("POST")
        plt.legend()
        plt.show()

        plt.plot(pre_dec_sorted, p_pre_dec, color="#a0c4e4", label="DECREASING")
        plt.plot(post_inc_sorted, p_post_inc, color="#f7959c", label="INCREASING")
        plt.ylabel("CDF")
        plt.xlabel("Min. distance: peak firing loc. to closest goal / cm")
        plt.title("PRE-POST")
        plt.legend()
        plt.show()

        plt.plot(pre_stable_sorted, p_pre_stable, color="b", label="STABLE PRE")
        plt.plot(post_stable_sorted, p_post_stable, color="r", label="STABLE POST")
        plt.ylabel("CDF")
        plt.xlabel("Min. distance: peak firing loc. to closest goal / cm")
        plt.title("STABLE CELLS: PRE-POST")
        plt.legend()
        plt.show()

    def pre_post_cheeseboard_learning_vs_drift_stable_cells(self, spatial_resolution=2):
        """
        Compares rate map changes due to learning vs. rate map changes due to drift (from PRE to POST) for stable cells

        :param spatial_resolution: in cm2
        :type spatial_resolution: int
        """
        # learning - stable
        remapping_per_cell_learn_dec = []
        remapping_per_cell_shuffle_learn_dec = []
        remapping_pv_learn_dec = []
        remapping_pv_shuffle_learn_dec = []
        # learning - dec
        remapping_per_cell_learn = []
        remapping_per_cell_shuffle_learn = []
        remapping_pv_learn = []
        remapping_pv_shuffle_learn = []
        # drift (PRE - POST)
        remapping_per_cell_drift = []
        remapping_per_cell_shuffle_drift = []
        remapping_pv_drift = []
        remapping_pv_shuffle_drift = []

        for session in self.session_list:
            # get drift results
            remap_cell_, remap_cell_shuffle_, remap_pv_, remap_pv_shuffle_ =\
                session.pre_post().remapping_pre_post_stable(plot_results=False,
                                                                      spatial_resolution=spatial_resolution,
                                                                      nr_trials_to_use=5, return_distribution=True)

            remapping_per_cell_drift.append(remap_cell_)
            remapping_per_cell_shuffle_drift.append(remap_cell_shuffle_)
            remapping_pv_drift.append(remap_pv_)
            remapping_pv_shuffle_drift.append(remap_pv_shuffle_)

            # get learning results
            remap_cell, remap_cell_shuffle, remap_cell_dec, remap_cell_shuffle_dec, remap_pv, remap_pv_shuffle, \
            remap_pv_dec, remap_pv_shuffle_dec = \
                session.cheese_board(
                    experiment_phase=["learning_cheeseboard_1"]).map_dynamics_learning(plot_results=False,
                                                                                       adjust_pv_size=False,
                                                                                       spatial_resolution=spatial_resolution)
            # stable cells
            remapping_per_cell_learn.append(remap_cell)
            remapping_per_cell_shuffle_learn.append(remap_cell_shuffle)
            remapping_pv_learn.append(remap_pv)
            remapping_pv_shuffle_learn.append(remap_pv_shuffle)
            # decreasing cells
            remapping_per_cell_learn_dec.append(remap_cell_dec)
            remapping_per_cell_shuffle_learn_dec.append(remap_cell_shuffle_dec)
            remapping_pv_learn_dec.append(remap_pv_dec)
            remapping_pv_shuffle_learn_dec.append(remap_pv_shuffle_dec)

        # pre-process results from learning - stable
        remapping_per_cell_learn = np.hstack(remapping_per_cell_learn)
        remapping_per_cell_shuffle_learn = np.vstack(remapping_per_cell_shuffle_learn).flatten()
        remapping_pv_learn = np.hstack(remapping_pv_learn)
        remapping_pv_cell_shuffle_learn = np.vstack(remapping_per_cell_shuffle_learn).flatten()

        # pre-process results from learning - dec
        remapping_per_cell_learn_dec = np.hstack(remapping_per_cell_learn_dec)
        remapping_per_cell_shuffle_learn_dec = np.vstack(remapping_per_cell_shuffle_learn_dec).flatten()
        remapping_pv_learn_dec = np.hstack(remapping_pv_learn_dec)
        remapping_pv_cell_shuffle_learn_dec = np.vstack(remapping_per_cell_shuffle_learn_dec).flatten()

        # remove nans
        remapping_per_cell_learn = remapping_per_cell_learn[~np.isnan(remapping_per_cell_learn)]
        remapping_per_cell_shuffle_learn = remapping_per_cell_shuffle_learn[~np.isnan(remapping_per_cell_shuffle_learn)]
        remapping_pv_learn = remapping_pv_learn[~np.isnan(remapping_pv_learn)]
        remapping_pv_cell_shuffle_learn = remapping_pv_cell_shuffle_learn[~np.isnan(remapping_pv_cell_shuffle_learn)]

        remapping_per_cell_learn_sorted = np.sort(remapping_per_cell_learn)
        remapping_per_cell_shuffle_learn_sorted = np.sort(remapping_per_cell_shuffle_learn)
        remapping_pv_learn_sorted = np.sort(remapping_pv_learn)
        remapping_pv_cell_shuffle_learn_sorted = np.sort(remapping_pv_cell_shuffle_learn)

        p_remapping_per_cell_learn = 1. * np.arange(remapping_per_cell_learn.shape[0]) / (remapping_per_cell_learn.shape[0] - 1)
        p_remapping_per_cell_learn_shuffle = 1. * np.arange(remapping_per_cell_shuffle_learn.shape[0]) / (remapping_per_cell_shuffle_learn.shape[0] - 1)
        p_remapping_pv_learn = 1. * np.arange(remapping_pv_learn.shape[0]) / (remapping_pv_learn.shape[0] - 1)
        p_remapping_pv_cell_shuffle_learn = 1. * np.arange(remapping_pv_cell_shuffle_learn.shape[0]) / (remapping_pv_cell_shuffle_learn.shape[0] - 1)

        # pre-process results from drift
        remapping_per_cell_drift = np.hstack(remapping_per_cell_drift)
        remapping_per_cell_shuffle_drift = np.vstack(remapping_per_cell_shuffle_drift).flatten()
        remapping_pv_drift = np.hstack(remapping_pv_drift)
        remapping_pv_cell_shuffle_drift = np.vstack(remapping_per_cell_shuffle_drift).flatten()

        # remove nans
        remapping_pv_drift = remapping_pv_drift[~np.isnan(remapping_pv_drift)]
        remapping_pv_cell_shuffle_drift = remapping_pv_cell_shuffle_drift[~np.isnan(remapping_pv_cell_shuffle_drift)]
        remapping_per_cell_drift = remapping_per_cell_drift[~np.isnan(remapping_per_cell_drift)]
        remapping_per_cell_shuffle_drift = remapping_per_cell_shuffle_drift[~np.isnan(remapping_per_cell_shuffle_drift)]

        remapping_per_cell_drift_sorted = np.sort(remapping_per_cell_drift)
        remapping_per_cell_shuffle_drift_sorted = np.sort(remapping_per_cell_shuffle_drift)
        remapping_pv_drift_sorted = np.sort(remapping_pv_drift)
        remapping_pv_cell_shuffle_drift_sorted = np.sort(remapping_pv_cell_shuffle_drift)

        p_remapping_per_cell_drift = 1. * np.arange(remapping_per_cell_drift.shape[0]) / (remapping_per_cell_drift.shape[0] - 1)
        p_remapping_per_cell_drift_shuffle = 1. * np.arange(remapping_per_cell_shuffle_drift.shape[0]) / (remapping_per_cell_shuffle_drift.shape[0] - 1)
        p_remapping_pv_drift = 1. * np.arange(remapping_pv_drift.shape[0]) / (remapping_pv_drift.shape[0] - 1)
        p_remapping_pv_cell_shuffle_drift = 1. * np.arange(remapping_pv_cell_shuffle_drift.shape[0]) / (remapping_pv_cell_shuffle_drift.shape[0] - 1)


        res_cells = [remapping_per_cell_learn, remapping_per_cell_learn_dec, remapping_per_cell_drift]
        res_pv = [remapping_pv_learn, remapping_pv_learn_dec, remapping_pv_drift]

        c = "black"
        bplot = plt.boxplot(res_cells, positions=[1, 2, 3], patch_artist=True,
                            labels=["Learning stable", "Learning decreasing", "PRE-POST stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),
                            )
        colors = ["yellow", 'blue', "green"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.title("Correlation of cell maps")
        plt.ylabel("Pearson R")
        plt.grid(color="grey", axis="y")
        plt.show()

        c = "black"
        bplot = plt.boxplot(res_pv, positions=[1, 2, 3], patch_artist=True,
                            labels=["Learning stable", "Learning decreasing", "PRE-POST stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),
                            )
        colors = ["yellow", 'blue', "green"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.title("Correlation of PV maps")
        plt.ylabel("Pearson R")
        plt.grid(color="grey", axis="y")
        plt.show()



        plt.plot(remapping_per_cell_learn_sorted, p_remapping_per_cell_learn, label="Stable - Learning", color="magenta")
        plt.plot(remapping_per_cell_shuffle_learn_sorted, p_remapping_per_cell_learn_shuffle, label="Shuffle - Learning",
                 color="plum",linestyle="--")
        plt.plot(remapping_per_cell_drift_sorted, p_remapping_per_cell_drift, label="Stable - Drift", color="blue")
        plt.plot(remapping_per_cell_shuffle_drift_sorted, p_remapping_per_cell_drift_shuffle, label="Shuffle - Drift",
                 color="aliceblue",linestyle="--")
        plt.title("Per cell")
        plt.xlabel("Pearson R")
        plt.ylabel("CDF")
        plt.legend()
        plt.show()

        plt.plot(remapping_pv_learn_sorted, p_remapping_pv_learn, label="Stable - Learning", color="magenta")
        plt.plot(remapping_pv_cell_shuffle_learn_sorted, p_remapping_pv_cell_shuffle_learn, label="Shuffle - Learning",
                 color="plum",linestyle="--")
        plt.plot(remapping_pv_drift_sorted, p_remapping_pv_drift, label="Stable - Drift", color="blue")
        plt.plot(remapping_pv_cell_shuffle_drift_sorted, p_remapping_pv_cell_shuffle_drift, label="Shuffle - Drift",
                 color="aliceblue",linestyle="--")

        plt.title("PV")
        plt.xlabel("Pearson R")
        plt.ylabel("CDF")
        plt.legend()
        plt.show()

    def pre_post_cheeseboard_learning_vs_drift(self, spatial_resolution=5, save_fig=False):
        """
        Compares rate map changes due to learning vs. rate map changes due to drift (from PRE to POST)

        :param save_fig: save fig as .svg
        :type save_fig: bool
        :param spatial_resolution: in cm2
        :type spatial_resolution: int
        """

        # learning - stable
        remapping_pv_learning = []
        remapping_pv_drift = []
        remapping_rm_learning = []
        remapping_rm_drift = []

        for session in self.session_list:
            # get drift results
            remapping_pv_learning_, remapping_pv_drift_, remapping_rm_learning_, remapping_rm_drift_ =\
                session.pre_post().remapping_learning_vs_drift(plotting=False, spatial_resolution=spatial_resolution,
                                                               nr_trials_to_use=5)
            remapping_pv_learning.append(remapping_pv_learning_)
            remapping_pv_drift.append(remapping_pv_drift_)
            remapping_rm_learning.append(remapping_rm_learning_)
            remapping_rm_drift.append(remapping_rm_drift_)

        remapping_pv_learning = np.hstack(remapping_pv_learning)
        remapping_pv_drift = np.hstack(remapping_pv_drift)
        remapping_rm_learning = np.hstack(remapping_rm_learning)
        remapping_rm_drift = np.hstack(remapping_rm_drift)

        print("pv")
        print(mannwhitneyu(remapping_pv_learning, remapping_pv_drift, alternative="two-sided"))
        print("rm")
        print(mannwhitneyu(remapping_rm_learning, remapping_rm_drift, alternative="two-sided"))

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        bplot = plt.boxplot([remapping_pv_learning, remapping_pv_drift], positions=[1, 2], patch_artist=True,
                            labels=["Learning", "PRE-POST"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),
                            )
        colors = ["dimgrey", 'lightgrey']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Pop. vec. correlations (Pearson R)")
        plt.grid(color="grey", axis="y")
        plt.ylim(-0.2, 1.19)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "learning_vs_drfit_pvs.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        bplot = plt.boxplot([remapping_rm_learning, remapping_rm_drift], positions=[1, 2], patch_artist=True,
                            labels=["Learning", "PRE-POST"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),
                            )
        colors = ["dimgrey", 'lightgrey']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Rate map correlations (Pearson R)")
        plt.grid(color="grey", axis="y")
        plt.ylim(-0.2,1)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "learning_vs_drfit_ratemaps.svg"), transparent="True")
        else:
            plt.show()

    def pre_post_cheeseboard_subset_pre_post_decoding(self, save_fig=False, cells_to_use="stable"):
        """
        Trains Bayesian decoder in PRE and tests it in POST

        :param save_fig: save figure as .svg
        :type save_fig: bool
        """
        # pre decoding error
        pre = []
        # post decoding error
        post = []
        # post decoding error shuffle
        post_shuffle = []

        for session in self.session_list:
            # get drift results
            error_pre, error_post, error_post_shuffle=\
                session.pre_post().location_decoding_subset_of_cells(plotting=False, cells_to_use=cells_to_use)

            pre.append(error_pre)
            post.append(error_post)
            post_shuffle.append(error_post_shuffle)

        # pre-process results from learning
        pre = np.hstack(pre)
        post = np.hstack(post)
        post_shuffle = np.hstack(post_shuffle)

        # stats
        p_pre_shuffle = mannwhitneyu(pre, post_shuffle)[1]
        print("PRE vs. Shuffle, p = " +str(p_pre_shuffle))

        p_post_shuffle = mannwhitneyu(post, post_shuffle, alternative="less")[1]
        print("POST vs. Shuffle, p = "+str(p_post_shuffle))

        pre_sorted = np.sort(pre)
        post_sorted = np.sort(post)
        post_shuffle_sorted = np.sort(post_shuffle)

        p_pre = 1. * np.arange(pre.shape[0]) / (pre.shape[0] - 1)
        p_post = 1. * np.arange(post.shape[0]) / (post.shape[0] - 1)
        p_post_shuffle = 1. * np.arange(post_shuffle.shape[0]) / (post_shuffle.shape[0] - 1)

        if save_fig:
            plt.style.use('default')

        # want error in cm and not arbitrary units
        pre_sorted = pre_sorted
        post_shuffle_sorted =post_shuffle_sorted
        post_sorted = post_sorted
        plt.plot(pre_sorted, p_pre, label="PRE", color="blue")
        plt.plot(post_sorted, p_post, label="POST",
                 color="magenta")
        plt.plot(post_shuffle_sorted, p_post_shuffle, label="POST Shuffle", color="plum", linestyle="--")
        plt.xlabel("Decoding error (cm)")
        plt.ylabel("cdf")
        plt.xticks([0, 25, 50, 75, 100, 125])
        plt.xlim(-10, 135)
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "decoding_error_pre_post.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def pre_post_cheeseboard_subset_pre_post_decoding_all_cells(self, save_fig=False, use_first_25_percent_of_post=False):
        """
        Trains Bayesian decoder in PRE and tests it in POST

        :param save_fig: save figure as .svg
        :type save_fig: bool
        """
        # pre decoding error
        pre_stable = []
        pre_inc = []
        pre_dec = []
        # post decoding error
        post_stable = []
        post_inc = []
        post_dec = []
        # post decoding error shuffle
        post_shuffle_stable = []
        post_shuffle_dec = []
        post_shuffle_inc = []

        for session in self.session_list:
            # get drift results
            error_pre_s, error_post_s, error_post_shuffle_s=\
                session.pre_post().location_decoding_subset_of_cells(plotting=False, cells_to_use="stable",
                                                            use_first_25_percent_of_post=use_first_25_percent_of_post)
            error_pre_d, error_post_d, error_post_shuffle_d=\
                session.pre_post().location_decoding_subset_of_cells(plotting=False, cells_to_use="decreasing",
                                                            use_first_25_percent_of_post=use_first_25_percent_of_post)
            error_pre_i, error_post_i, error_post_shuffle_i=\
                session.pre_post().location_decoding_subset_of_cells(plotting=False, cells_to_use="increasing",
                                                            use_first_25_percent_of_post=use_first_25_percent_of_post)

            pre_stable.append(error_pre_s)
            pre_inc.append(error_pre_i)
            pre_dec.append(error_pre_d)
            # post decoding error
            post_stable.append(error_post_s)
            post_inc.append(error_post_i)
            post_dec.append(error_post_d)
            # post decoding error shuffle
            post_shuffle_stable.append(error_post_shuffle_s)
            post_shuffle_dec.append(error_post_shuffle_d)
            post_shuffle_inc.append(error_post_shuffle_i)

        # pre-process results from learning
        pre_stable = np.hstack(pre_stable)
        post_stable = np.hstack(post_stable)
        post_shuffle_stable = np.hstack(post_shuffle_stable)
        pre_dec = np.hstack(pre_dec)
        post_dec = np.hstack(post_dec)
        post_shuffle_dec = np.hstack(post_shuffle_dec)
        pre_inc = np.hstack(pre_inc)
        post_inc = np.hstack(post_inc)
        post_shuffle_inc = np.hstack(post_shuffle_inc)

        plt.figure(figsize=(4, 4))
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        res = [pre_stable, post_stable, post_shuffle_stable, pre_dec, post_dec, post_shuffle_dec,
               pre_inc, post_inc, post_shuffle_inc]
        bplot = plt.boxplot(res, positions=[1, 2, 3, 4, 5, 6, 7, 8, 9], patch_artist=True,
                            labels=["Acq.", "Recall", "Shuffle", "Acq.", "Recall", "Shuffle","Acq.", "Recall", "Shuffle"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["violet", 'violet', "violet", "turquoise", "turquoise", "turquoise", "orange", "orange", "orange"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Decoding error (cm)")
        plt.xticks(rotation=45)
        y_base = 130
        # stable cells
        plt.hlines(y_base, 1, 1.9, colors=c)
        if mannwhitneyu(pre_stable, post_stable)[1] < 0.001:
            plt.text(1.3, y_base, "***")
        elif mannwhitneyu(pre_stable, post_stable)[1] < 0.01:
            plt.text(1.3, y_base, "**")
        elif mannwhitneyu(pre_stable, post_stable)[1] < 0.05:
            plt.text(1.3, y_base, "*")
        else:
            plt.text(1.3, y_base, "n.s.")
        plt.hlines(y_base, 2.1, 3, colors=c)
        if mannwhitneyu(post_stable, post_shuffle_stable)[1] < 0.001:
            plt.text(2.3, y_base, "***")
        elif mannwhitneyu(post_stable, post_shuffle_stable)[1] < 0.01:
            plt.text(2.3, y_base, "**")
        elif mannwhitneyu(post_stable, post_shuffle_stable)[1] < 0.05:
            plt.text(2.3, y_base, "*")
        else:
            plt.text(2.3, y_base, "n.s.")
        # decreasing
        # -----------------------------------------------------------------------------------------------
        plt.hlines(y_base, 4, 4.9, colors=c)
        if mannwhitneyu(pre_dec, post_dec)[1] < 0.001:
            plt.text(4.3, y_base, "***")
        elif mannwhitneyu(pre_dec, post_dec)[1] < 0.01:
            plt.text(4.3, y_base, "**")
        elif mannwhitneyu(pre_dec, post_dec)[1] < 0.05:
            plt.text(4.3, y_base, "*")
        else:
            plt.text(4.3, y_base, "n.s.")
        plt.hlines(y_base, 5.1, 6, colors=c)
        if mannwhitneyu(post_dec, post_shuffle_dec)[1] < 0.001:
            plt.text(5.3, y_base, "***")
        elif mannwhitneyu(post_dec, post_shuffle_dec)[1] < 0.01:
            plt.text(5.3, y_base, "**")
        elif mannwhitneyu(post_dec, post_shuffle_dec)[1] < 0.05:
            plt.text(5.3, y_base, "*")
        else:
            plt.text(4.3, y_base, "n.s.")
        # increasing
        # -----------------------------------------------------------------------------------------------
        plt.hlines(y_base, 7, 7.9, colors=c)
        if mannwhitneyu(pre_inc, post_inc)[1] < 0.001:
            plt.text(7.3, y_base, "***")
        elif mannwhitneyu(pre_inc, post_inc)[1] < 0.01:
            plt.text(7.3, y_base, "**")
        elif mannwhitneyu(pre_inc, post_inc)[1] < 0.05:
            plt.text(7.3, y_base, "*")
        else:
            plt.text(7.3, y_base, "n.s.")
        plt.hlines(y_base, 8.1, 9, colors=c)
        if mannwhitneyu(post_inc, post_shuffle_inc)[1] < 0.001:
            plt.text(8.3, y_base, "***")
        elif mannwhitneyu(post_inc, post_shuffle_inc)[1] < 0.01:
            plt.text(8.3, y_base, "**")
        elif mannwhitneyu(post_inc, post_shuffle_inc)[1] < 0.05:
            plt.text(8.3, y_base, "*")
        else:
            plt.text(7.3, y_base, "n.s.")

        # across cells: acquisition
        # -----------------------------------------------------------------------------------------------
        plt.hlines(y_base + 15, 1, 3.9, colors=c)
        if mannwhitneyu(pre_stable, pre_dec)[1] < 0.001:
            plt.text(2.5, y_base + 15, "***")
        elif mannwhitneyu(pre_stable, pre_dec)[1] < 0.01:
            plt.text(2.5, y_base + 15, "**")
        elif mannwhitneyu(pre_stable, pre_dec)[1] < 0.05:
            plt.text(2.5, y_base + 15, "*")
        else:
            plt.text(2.5, y_base + 15, "n.s.")
        plt.hlines(y_base + 15, 4.1, 7, colors=c)
        if mannwhitneyu(pre_dec, pre_inc)[1] < 0.001:
            plt.text(5.5, y_base + 15, "***")
        elif mannwhitneyu(pre_dec, pre_inc)[1] < 0.01:
            plt.text(5.5, y_base + 15, "**")
        elif mannwhitneyu(pre_dec, pre_inc)[1] < 0.05:
            plt.text(5.5, y_base + 15, "*")
        else:
            plt.text(5.5, y_base + 15, "n.s.")

        plt.hlines(y_base + 30, 1, 7, colors=c)
        if mannwhitneyu(pre_stable, pre_inc)[1] < 0.001:
            plt.text(3.8, y_base + 30, "***")
        elif mannwhitneyu(pre_stable, pre_inc)[1] < 0.01:
            plt.text(3.8, y_base + 30, "**")
        elif mannwhitneyu(pre_stable, pre_inc)[1] < 0.05:
            plt.text(3.8, y_base + 30, "*")
        else:
            plt.text(3.8, y_base + 30, "n.s.")

        # across cells: recall
        # -----------------------------------------------------------------------------------------------
        plt.hlines(y_base + 45, 2, 4.9, colors=c)
        if mannwhitneyu(post_stable, post_dec)[1] < 0.001:
            plt.text(3.5, y_base + 45, "***")
        elif mannwhitneyu(post_stable, post_dec)[1] < 0.01:
            plt.text(3.5, y_base + 45, "**")
        elif mannwhitneyu(post_stable, post_dec)[1] < 0.05:
            plt.text(3.5, y_base + 45, "*")
        else:
            plt.text(3.5, y_base + 45, "n.s.")
        plt.hlines(y_base + 45, 5.1, 8, colors=c)
        if mannwhitneyu(post_dec, post_inc)[1] < 0.001:
            plt.text(6.5, y_base + 45, "***")
        elif mannwhitneyu(post_dec, post_inc)[1] < 0.01:
            plt.text(6.5, y_base + 45, "**")
        elif mannwhitneyu(post_dec, post_inc)[1] < 0.05:
            plt.text(6.5, y_base + 45, "*")
        else:
            plt.text(6.5, y_base + 45, "n.s.")

        plt.hlines(y_base + 60, 2, 8, colors=c)
        if mannwhitneyu(post_stable, post_inc)[1] < 0.001:
            plt.text(4.8, y_base + 60, "***")
        elif mannwhitneyu(post_stable, post_inc)[1] < 0.01:
            plt.text(4.8, y_base + 60, "**")
        elif mannwhitneyu(post_stable, post_inc)[1] < 0.05:
            plt.text(4.8, y_base + 60, "*")
        else:
            plt.text(4.8, y_base + 60, "n.s.")

        plt.ylim(-1, 210)
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            if use_first_25_percent_of_post:
                plt.savefig(os.path.join(save_path, "decoding_error_pre_post_all_cells_25_percent_of_post.svg"),
                            transparent="True")
            else:
                plt.savefig(os.path.join(save_path, "decoding_error_pre_post_all_cells.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def pre_post_likelihoods(self, cells_to_use="stable", save_fig=False):
        """
        uses PRE phmm to decode activity in PRE and pre_probe to compare likelihoods

        :param cells_to_use: "all", "stable", "decreasing", "increasing"
        :type cells_to_use: str
        :param save_fig: save as .svg
        :type save_fig: bool
        """
        pre_probe_median_log_likeli = []
        pre_median_log_likeli = []

        for session in self.session_list:
            pre_probe, pre = \
                session.pre_post().phmm_modes_likelihoods(plotting=False, cells_to_use=cells_to_use)
            pre_probe_median_log_likeli.append(pre_probe)
            pre_median_log_likeli.append(pre)

        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(4, 6))
        # plt.figure(figsize=(3,4))
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        for session_id, (first, second) in enumerate(zip(pre_probe_median_log_likeli, pre_median_log_likeli)):
            plt.scatter([0.1, 0.2], [first, second], label=str(session_id), color=col[session_id], zorder=session_id)
            plt.plot([0.1, 0.2], [first, second], color=col[session_id], zorder=session_id)
            plt.xticks([0.1, 0.2], ["Pre_probe", "PRE"])
        plt.ylabel("Median Log-likelihood of PRE states")
        plt.grid(axis="y", color="gray")
        plt.title(cells_to_use)
        plt.ylim(-17, -7)
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "pre_probe_pre_likeli_" + cells_to_use + ".svg"), transparent="True")
        else:
            plt.show()

    def pre_post_optimal_number_states(self, save_fig=False):

        pre_optimal_number_states = []
        post_optimal_number_states = []

        for session in self.session_list:
            pre_optimal_number_states_, post_optimal_number_states_ = \
                session.pre_post().get_optimal_number_states()
            pre_optimal_number_states.append(pre_optimal_number_states_)
            post_optimal_number_states.append(post_optimal_number_states_)

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        plt.figure(figsize=(4, 4))
        plt.hist(pre_optimal_number_states, label="Acquisition", color="grey")
        plt.hist(post_optimal_number_states, label="Recall", color="orange", alpha=0.6)
        plt.xlim(0, 65)
        plt.ylabel("Number of sessions")
        plt.xlabel("Optimal number of pHMM states")
        plt.legend(loc=2)
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "optimal_#states_pre_post.svg"), transparent="True")
        else:
            plt.show()

        print(mannwhitneyu(pre_optimal_number_states, post_optimal_number_states))

    def pre_post_cheesebaord_firing_rate_ratios_all_cells(self, measure="mean", save_fig=False, chunks_in_min=2,
                                                          use_chunks=True):
        """
        Computes ratio of firing between Sleep/PRE and Sleep/POST

        :param measure: "mean" or "max" firing rates
        :type measure: str
        :param save_fig: save as .svg
        :type save_fig: bool
        :param chunks_in_min: in which chunks to compute mean or max firing rates
        :type chunks_in_min: int
        """
        ratio_stable = []
        ratio_dec = []
        ratio_inc = []


        for session in self.session_list:
            s, d, i = \
                session.pre_post().firing_rate_ratios_pre_post_all_cells(plotting=False, measure=measure,
                                                                         chunks_in_min=chunks_in_min,
                                                                         use_chunks=use_chunks)

            ratio_stable.append(s)
            ratio_dec.append(d)
            ratio_inc.append(i)


        ratio_stable = np.hstack(ratio_stable)
        ratio_dec = np.hstack(ratio_dec)
        ratio_inc = np.hstack(ratio_inc)


        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        # print(mannwhitneyu(ratio_pre_stable, ratio_post_stable))

        y_dat = [ratio_stable, ratio_dec, ratio_inc]
        plt.figure(figsize=(3, 4))
        bplot = plt.boxplot(y_dat, positions=[1, 2, 3], patch_artist=True,
                            labels=["Persistent", "Decreasing", "Increasing"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False, widths=(0.25, 0.25, 0.25))
        colors = ["violet", 'turquoise', "orange"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        plt.ylabel("Mean_firing_Recall - Mean_firing_Recall")
        plt.xticks(rotation=45)
        plt.ylim(-1.2,1.5)
        plt.hlines(0, 0.5, 3.5, color="gray", linestyles=":", zorder=-1000)
        base_y = 1.1
        plt.hlines(base_y, 1, 1.9, color=c)
        if mannwhitneyu(ratio_stable, ratio_dec)[1] < 0.001/3:
            plt.text(1.2, base_y, "***")
        elif mannwhitneyu(ratio_stable, ratio_dec)[1] < 0.01/3:
            plt.text(1.2, base_y, "**")
        elif mannwhitneyu(ratio_stable, ratio_dec)[1] < 0.05/3:
            plt.text(1.2, base_y, "*")
        else:
            plt.text(1.2, base_y, "n.s.")
        plt.hlines(base_y, 2.1, 3, color=c)
        if mannwhitneyu(ratio_inc, ratio_dec)[1] < 0.001/3:
            plt.text(2.3, base_y, "***")
        elif mannwhitneyu(ratio_inc, ratio_dec)[1] < 0.01/3:
            plt.text(2.2, base_y, "**")
        elif mannwhitneyu(ratio_inc, ratio_dec)[1] < 0.05/3:
            plt.text(2.2, base_y, "*")
        else:
            plt.text(2.2, base_y, "n.s.")
        plt.hlines(base_y+0.2, 1, 3, color=c)
        if mannwhitneyu(ratio_inc, ratio_stable)[1] < 0.001/3:
            plt.text(1.8, base_y+0.2, "***")
        elif mannwhitneyu(ratio_inc, ratio_stable)[1] < 0.01/3:
            plt.text(1.8, base_y+0.2, "**")
        elif mannwhitneyu(ratio_inc, ratio_stable)[1] < 0.05/3:
            plt.text(1.9, base_y+0.2, "*")
        else:
            plt.text(2.2, base_y+0.2, "n.s.")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_pre_post_ratio_" + measure + "_all_cells.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def pre_post_cheeseboard_modes_cell_contribution(self, save_fig=False):

        ratio_stable = []
        ratio_decreasing = []
        ratio_increasing = []

        for session in self.session_list:
            res = session.pre_post().pre_post_modes_cell_contribution(plotting=False)
            ratio_stable.append(res[0])
            ratio_decreasing.append(res[1])
            ratio_increasing.append(res[2])

        res = [np.hstack(ratio_stable), np.hstack(ratio_decreasing), np.hstack(ratio_increasing)]

        print("Persistent vs. decreasing:")
        print(mannwhitneyu(res[0], res[1]))
        print(" -> corrrected mul. comp:")
        print(mannwhitneyu(res[0], res[1])[1]*3)
        print("Persistent vs. increasing:")
        print(mannwhitneyu(res[0], res[2]))
        print(" -> corrrected mul. comp:")
        print(mannwhitneyu(res[0], res[2])[1]*3)
        print("Increasing vs. decreasing:")
        print(mannwhitneyu(res[1], res[2]))
        print(" -> corrrected mul. comp:")
        print(mannwhitneyu(res[1], res[2])[1]*3)


        plt.figure(figsize=(2, 3))
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        bplot = plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                            labels=["persistent", "decreasing", "increasing"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.ylabel("Mean rate Recall states - Mean rate Acquisition states\n"
                   "Mean rate Recall states + Mean rate Acquisition states")
        colors = ["magenta", 'turquoise', "orange"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.xticks(rotation=45)
        plt.ylim(-1.1, 1.5)
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "pre_post_states_cell_subset_contribution.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def pre_post_model_decoding(self, save_fig=False):

        pre_pre = []
        pre_post = []
        post_pre =[]
        post_post = []

        for i, session in enumerate(self.session_list):
            pre_pre_, pre_post_, post_pre_, post_post_ = \
                session.pre_post().pre_post_model_decoding(plotting=False)
            pre_pre.append(pre_pre_)
            pre_post.append(pre_post_)
            post_pre.append(post_pre_)
            post_post.append(post_post_)



        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        plt.figure(figsize=(4,4))
        # first and second half for acquisition
        res = [np.hstack(pre_pre), np.hstack(pre_post), np.hstack(post_pre), np.hstack(post_post)]

        bplot=plt.boxplot(res, positions=[1, 2, 3, 4], patch_artist=True,
                          labels=["Acquisition", "Recall", "Acquisition", "Recall"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        plt.ylim(-60,5)
        y_base=-12
        plt.hlines(y_base, 1, 1.9, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/5:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/5:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/5:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/5:
            plt.text(1.4, y_base, "*", color=c)
        plt.hlines(y_base, 2.1, 2.9, color=c)
        if mannwhitneyu(res[2], res[1])[1] > 0.05/5:
            plt.text(2.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[2], res[1])[1] < 0.001/5:
            plt.text(2.4, y_base, "***", color=c)
        elif mannwhitneyu(res[2], res[1])[1] < 0.01/5:
            plt.text(2.4, y_base, "**", color=c)
        elif mannwhitneyu(res[2], res[1])[1] < 0.05/5:
            plt.text(2.4, y_base, "*", color=c)
        plt.hlines(y_base, 3.1, 4, color=c)
        if mannwhitneyu(res[2], res[3])[1] > 0.05/5:
            plt.text(3.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.001/5:
            plt.text(3.4, y_base, "***", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.01/5:
            plt.text(3.4, y_base, "**", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.05/5:
            plt.text(3.4, y_base, "*", color=c)
        # first vs. second half NREM
        y_base=-6
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05/5:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001/5:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01/5:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05/5:
            plt.text(2, y_base, "*", color=c)
        # second half
        y_base=0
        plt.hlines(y_base, 2, 4, color=c)
        if mannwhitneyu(res[1], res[3])[1] > 0.05/5:
            plt.text(3, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.001/5:
            plt.text(3, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.01/5:
            plt.text(3, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.05/5:
            plt.text(3, y_base, "*", color=c)

        plt.ylabel("Log-likelihood")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("likelihood_pre_model_post_model_pre_post.svg", transparent="True")
        else:
            plt.show()

    # </editor-fold>

    # <editor-fold desc="PRE, long sleep, POST">

    def pre_long_sleep_post_drift_vs_correlations(self, template_type="phmm", n_smoothing=1000):

        sim_ratio_mean = []
        sim_ratio_std = []
        sim_ratio_correlations = []
        for session in self.session_list:
            s_m, s_d = \
                session.pre_long_sleep_post().drift_correlation_structure_all_cells(plotting=False,
                                                                                    n_smoothing=n_smoothing)
            res = session.long_sleep().memory_drift_plot_temporal_trend(template_type=template_type,
                                                                        n_moving_average_pop_vec=n_smoothing,
                                                                        plotting=False)
            sim_ratio_mean.append(s_m)
            sim_ratio_std.append(s_d)
            sim_ratio_correlations.append(res)

        print("HERE")

    def pre_long_sleep_post_drift_around_and_away_from_goals(self):
        """
        Checks goal coding of over-expressed modes during sleep when using only a subset of cells for decoding

        @param cells_to_compare: which cells to use ("stable", "increasing", "decreasing")
        :type cells_to_compare: str
        """
        # go trough all sessions to collect results
        ds_rem_around_goals = []
        ds_rem_away_from_goals = []
        ds_nrem_around_goals = []
        ds_nrem_away_from_goals = []
        for session in self.session_list:
            ds_rem_around_goals_, ds_rem_away_from_goals_, ds_nrem_around_goals_, ds_nrem_away_from_goals_ = \
                session.pre_long_sleep_post().memory_drift_around_goals_and_away()
            ds_rem_around_goals.append(ds_rem_around_goals_)
            ds_rem_away_from_goals.append(ds_rem_away_from_goals_)
            ds_nrem_around_goals.append(ds_nrem_around_goals_)
            ds_nrem_away_from_goals.append(ds_nrem_away_from_goals_)

        print("HERE")

        c = "white"
        res = [ds_rem_around_goals, ds_rem_away_from_goals, ds_nrem_around_goals, ds_nrem_away_from_goals]
        bplot = plt.boxplot(res, positions=[1, 2, 3, 4], patch_artist=True,
                            labels=["REM around", "REM away", "NREM around", "NREM away"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["magenta", 'magenta', "blue", "blue"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.title("Delta score: around & away from goals")
        plt.ylabel("Delta score")
        plt.grid(color="grey", axis="y")
        plt.show()
        print("REM")
        print(mannwhitneyu(ds_rem_around_goals, ds_rem_away_from_goals))
        print("NREM")
        print(mannwhitneyu(ds_nrem_around_goals, ds_nrem_away_from_goals))

    def pre_long_sleep_post_over_expressed_modes_goal_coding(self, cells_to_compare="stable"):
        """
        Checks goal coding of over-expressed modes during sleep when using only a subset of cells for decoding

        @param cells_to_compare: which cells to use ("stable", "increasing", "decreasing")
        :type cells_to_compare: str
        """
        # go trough all sessions to collect results
        res = []
        session_name_strings = []
        for session in self.session_list:
            gc_increase_stable_rem, gc_stable_rem, gc_increase_stable_nrem, gc_stable_nrem = \
                session.pre_long_sleep_post().over_expressed_modes_goal_coding(template_type="phmm", plotting=False,
                                                                               post_or_pre="pre",
                                                                               cells_to_compare=cells_to_compare)
            res.append([gc_increase_stable_rem, gc_stable_rem, gc_increase_stable_nrem, gc_stable_nrem])
            session_name_strings.append(session.session_name)

        res = np.array(res)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for id_sess, res_sess in enumerate(res):
            plt.scatter(id_sess, res_sess[0]*100, label="INC. WITH " +cells_to_compare+ " CELLS", color="r")
            plt.scatter(id_sess, res_sess[1]*100, label="CONST./DEC. WITH "+cells_to_compare+" CELLS", color="b")
            plt.ylabel("GOAL CODING")
        plt.xticks(range(res.shape[0]), session_name_strings, rotation='vertical')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("REM (PRE): INCREASE REACTIVATION OF MODES \n WITH ONLY "+cells_to_compare+" CELLS")
        plt.ylim(0,100)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for id_sess,res_sess in enumerate(res):
            plt.scatter(id_sess, res_sess[2]*100, label="INC. WITH "+cells_to_compare+" CELLS", color="r")
            plt.scatter(id_sess, res_sess[3]*100, label="CONST./DEC. WITH "+cells_to_compare+" CELLS", color="b")
            plt.ylabel("GOAL CODING")
        plt.xticks(range(res.shape[0]), session_name_strings, rotation='vertical')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("NREM (PRE): INCREASE REACTIVATION OF MODES \n WITH ONLY "+cells_to_compare+" CELLS")
        plt.ylim(0,100)
        plt.show()

        res = []
        for session in self.session_list:
            gc_increase_stable_rem, gc_stable_rem, gc_increase_stable_nrem, gc_stable_nrem = \
                session.pre_long_sleep_post().over_expressed_modes_goal_coding(template_type="phmm", plotting=False,
                                                                               post_or_pre="post",
                                                                               cells_to_compare=cells_to_compare)
            res.append([gc_increase_stable_rem, gc_stable_rem, gc_increase_stable_nrem, gc_stable_nrem])

        res = np.array(res)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for id_sess, res_sess in enumerate(res):
            plt.scatter(id_sess, res_sess[0] * 100, label="INC. WITH "+cells_to_compare+" CELLS", color="r")
            plt.scatter(id_sess, res_sess[1] * 100, label="CONST./DEC. WITH "+cells_to_compare+" CELLS", color="b")
            plt.ylabel("GOAL CODING")
        plt.xticks(range(res.shape[0]), session_name_strings, rotation='vertical')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("REM (POST): INCREASE REACTIVATION OF MODES \n WITH ONLY "+cells_to_compare+" CELLS")
        plt.ylim(0, 100)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for id_sess, res_sess in enumerate(res):
            plt.scatter(id_sess, res_sess[2] * 100, label="INC. WITH "+cells_to_compare+" CELLS", color="r")
            plt.scatter(id_sess, res_sess[3] * 100, label="CONST./DEC. WITH "+cells_to_compare+" CELLS", color="b")
            plt.ylabel("GOAL CODING")
        plt.xticks(range(res.shape[0]), session_name_strings, rotation='vertical')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("NREM (POST): INCREASE REACTIVATION OF MODES \n WITH ONLY "+cells_to_compare+" CELLS")
        plt.ylim(0, 100)
        plt.show()

    def pre_long_sleep_post_over_expressed_modes_spatial_info(self, cells_to_compare="stable"):
        """
        Checks spatial information of over-expressed modes during sleep when using only a subset of cells for decoding

        @param cells_to_compare: which cells to use ("stable", "increasing", "decreasing")
        :type cells_to_compare: str
        """
        # go trough all sessions to collect results
        res = []
        session_name_strings = []
        for session in self.session_list:
            gc_increase_stable_rem, gc_stable_rem, gc_increase_stable_nrem, gc_stable_nrem = \
                session.pre_long_sleep_post().over_expressed_modes_spatial_information(template_type="phmm", plotting=False,
                                                                               post_or_pre="pre",
                                                                               cells_to_compare=cells_to_compare)
            res.append([gc_increase_stable_rem, gc_stable_rem, gc_increase_stable_nrem, gc_stable_nrem])
            session_name_strings.append(session.session_name)

        res = np.array(res)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for id_sess, res_sess in enumerate(res):
            plt.scatter(id_sess, res_sess[0], label="INC. WITH " +cells_to_compare+ " CELLS", color="r")
            plt.scatter(id_sess, res_sess[1], label="CONST./DEC. WITH "+cells_to_compare+" CELLS", color="b")
            plt.ylabel("MEAN(MEDIAN DISTANCE)")
        plt.xticks(range(res.shape[0]), session_name_strings, rotation='vertical')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("REM (PRE): INCREASE REACTIVATION OF MODES \n WITH ONLY "+cells_to_compare+" CELLS")
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for id_sess,res_sess in enumerate(res):
            plt.scatter(id_sess, res_sess[2], label="INC. WITH "+cells_to_compare+" CELLS", color="r")
            plt.scatter(id_sess, res_sess[3], label="CONST./DEC. WITH "+cells_to_compare+" CELLS", color="b")
            plt.ylabel("MEAN(MEDIAN DISTANCE)")
        plt.xticks(range(res.shape[0]), session_name_strings, rotation='vertical')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("NREM (PRE): INCREASE REACTIVATION OF MODES \n WITH ONLY "+cells_to_compare+" CELLS")
        plt.show()

        res = []
        for session in self.session_list:
            gc_increase_stable_rem, gc_stable_rem, gc_increase_stable_nrem, gc_stable_nrem = \
                session.pre_long_sleep_post().over_expressed_modes_spatial_information(template_type="phmm", plotting=False,
                                                                               post_or_pre="post",
                                                                               cells_to_compare=cells_to_compare)
            res.append([gc_increase_stable_rem, gc_stable_rem, gc_increase_stable_nrem, gc_stable_nrem])

        res = np.array(res)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for id_sess, res_sess in enumerate(res):
            plt.scatter(id_sess, res_sess[0], label="INC. WITH "+cells_to_compare+" CELLS", color="r")
            plt.scatter(id_sess, res_sess[1], label="CONST./DEC. WITH "+cells_to_compare+" CELLS", color="b")
            plt.ylabel("MEAN(MEDIAN DISTANCE)")
        plt.xticks(range(res.shape[0]), session_name_strings, rotation='vertical')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("REM (POST): INCREASE REACTIVATION OF MODES \n WITH ONLY "+cells_to_compare+" CELLS")
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for id_sess, res_sess in enumerate(res):
            plt.scatter(id_sess, res_sess[2], label="INC. WITH "+cells_to_compare+" CELLS", color="r")
            plt.scatter(id_sess, res_sess[3], label="CONST./DEC. WITH "+cells_to_compare+" CELLS", color="b")
            plt.ylabel("MEAN(MEDIAN DISTANCE)")
        plt.xticks(range(res.shape[0]), session_name_strings, rotation='vertical')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("NREM (POST): INCREASE REACTIVATION OF MODES \n WITH ONLY "+cells_to_compare+" CELLS")
        plt.show()

    def pre_long_sleep_post_goal_reactivations_occupancy_post(self, cells_to_use="all", sleep_phase="rem",
                                                              first_half_of_post=False, third_of_sleep=None):
        """
        Check goal reactivations during sleep and occupancy in post

        """
        # go trough all sessions to collect results
        goal_decoded = []
        occ_post_amount = []

        for session in self.session_list:
            gd, occp = \
                session.pre_long_sleep_post().goal_reactivations_occupancy_post(cells_to_use=cells_to_use,
                                                                                sleep_phase=sleep_phase,
                                                                                first_half_of_post=first_half_of_post,
                                                                                third_of_sleep=third_of_sleep)

            goal_decoded.append(gd)
            occ_post_amount.append(occp)

        all_goal_decoded = np.hstack(goal_decoded)
        all_occ_post_amount = np.hstack(occ_post_amount)
        plt.scatter(all_goal_decoded, all_occ_post_amount)
        plt.text(np.min(all_goal_decoded)+(np.max(all_goal_decoded)-np.min(all_goal_decoded))/2,
                 np.min(all_occ_post_amount)+(np.max(all_occ_post_amount)-np.min(all_occ_post_amount))/2,
                 "R="+str(np.round(pearsonr(all_goal_decoded, all_occ_post_amount)[0],2)))
        plt.xlabel("#goal reactivated during sleep (normalized)")
        plt.ylabel("occupancy per bin \n goal recall (normalized")
        plt.show()

    def pre_long_sleep_post_goal_reactivations_temporal(self, cells_to_use="all", sleep_phase="rem"):
        """
        Check goal reactivations during sleep and occupancy in post

        """
        # go trough all sessions to collect results

        for session in self.session_list:
            session.pre_long_sleep_post().goal_reactivations_strength_temporal(cells_to_use=cells_to_use,
                                                                                sleep_phase=sleep_phase)

    def pre_long_sleep_post_goal_reactivations_change_in_goal_coding(self, cells_to_use="all", sleep_phase="rem",
                                                                     third_of_sleep=None):
        """
        Check goal reactivations during sleep and occupancy in post

        """
        # go trough all sessions to collect results

        goal_decoded_normalized = []
        all_goals_mean = []

        for session in self.session_list:
            gd, gm = session.pre_long_sleep_post().goal_reactivations_change_in_goal_coding(cells_to_use=cells_to_use,
                                                                                sleep_phase=sleep_phase,
                                                                                            third_of_sleep=third_of_sleep)
            goal_decoded_normalized.append(gd)
            all_goals_mean.append(gm)

        goal_decoded_normalized = np.hstack(goal_decoded_normalized)
        all_goals_mean = np.hstack(all_goals_mean)

        plt.scatter(goal_decoded_normalized, all_goals_mean)
        plt.text(np.min(goal_decoded_normalized) + (np.max(goal_decoded_normalized) - np.min(goal_decoded_normalized)) / 2,
                 np.min(all_goals_mean) + (np.max(all_goals_mean) - np.min(all_goals_mean)) / 2,
                 "R=" + str(np.round(pearsonr(goal_decoded_normalized, all_goals_mean)[0], 2)))
        print(pearsonr(goal_decoded_normalized, all_goals_mean))
        plt.xlabel("#times goal decoded in sleep (normalized)")
        plt.ylabel("PV correlation around goal (PRE-POST)")
        plt.show()

    def pre_long_sleep_post_reactivation_stability_vs_goal_coding(self, save_fig=False,distance_metric="correlation",
                                                                  thresh_stab=0, parts_div_distance=3,
                                                                  thr_close_to_goal=10):

        min_distance_per_pre_mode = []
        coeff_mode_pre = []
        min_distance_per_post_mode = []
        coeff_mode_post = []
        pre_mode_ids = []
        post_mode_ids = []

        pre_modes_goal_coding = []
        post_modes_goal_coding = []

        for i, session in enumerate(self.session_list):
            md_pre, c_pre, md_post, c_post, pre_m_ids, post_m_ids = \
                session.long_sleep().memory_drift_pre_post_mode_probability_stability_vs_similarity(plotting=False,
                                                                                                    distance_metric=
                                                                                                    distance_metric,
                                                                                                    thresh_stab=
                                                                                                    thresh_stab, return_mode_ids=True)

            frac_per_mode_pre = \
                    session.cheese_board(experiment_phase=["learning_cheeseboard_1"]).analyze_all_modes_goal_coding(thr_close_to_goal= thr_close_to_goal)

            frac_per_mode_post = \
                    session.cheese_board(experiment_phase=["learning_cheeseboard_2"]).analyze_all_modes_goal_coding(thr_close_to_goal=thr_close_to_goal)

            pre_mode_ids.append(pre_m_ids)
            post_mode_ids.append(post_m_ids)
            pre_modes_goal_coding.append(frac_per_mode_pre)
            post_modes_goal_coding.append(frac_per_mode_post)
            min_distance_per_pre_mode.append(md_pre)
            coeff_mode_pre.append(c_pre)
            min_distance_per_post_mode.append(md_post)
            coeff_mode_post.append(c_post)


        interval_size = 1/parts_div_distance

        coeff_interval_pre_all_sessions = []

        for interval_id in range(parts_div_distance):
            # go through all the sessions and chose pre/post modes
            coeff_interval_pre = []
            for sess_id, (pre_modes_goal_coding_sess, coeff_mode_pre_sess, mode_ids) in enumerate(zip(pre_modes_goal_coding, coeff_mode_pre,
                                                                                 pre_mode_ids)):
                # select modes for goal coding
                gc_selection = pre_modes_goal_coding_sess[mode_ids]
                coeff_interval_pre.append(coeff_mode_pre_sess[np.logical_and(interval_id * interval_size < gc_selection,
                                                                             (interval_id * interval_size) < gc_selection)])

            coeff_interval_pre_all_sessions.append(coeff_interval_pre)

                # plt.scatter(gc_selection, coeff_mode_pre_sess)
                # plt.show()

        coeff_interval_pre_all_sessions = [np.hstack(x) for x in coeff_interval_pre_all_sessions]

        c = "white"
        bplot = plt.boxplot(coeff_interval_pre_all_sessions, positions=np.arange(parts_div_distance), patch_artist=True,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xlabel("Interval of goal coding")
        plt.title("PRE modes")
        plt.ylabel("Stability of reactivation")
        plt.show()

        coeff_interval_post_all_sessions = []

        for interval_id in range(parts_div_distance):
            # go through all the sessions and chose pre/post modes
            coeff_interval_post = []
            for sess_id, (post_modes_goal_coding_sess, coeff_mode_post_sess, mode_ids) in enumerate(zip(post_modes_goal_coding, coeff_mode_post,
                                                                                 post_mode_ids)):
                # select modes for goal coding
                gc_selection = post_modes_goal_coding_sess[mode_ids]
                coeff_interval_post.append(coeff_mode_post_sess[np.logical_and(interval_id * interval_size < gc_selection,
                                                                             (interval_id * interval_size) < gc_selection)])

            coeff_interval_post_all_sessions.append(coeff_interval_post)

                # plt.scatter(gc_selection, coeff_mode_pre_sess)
                # plt.show()

        coeff_interval_post_all_sessions = [np.hstack(x) for x in coeff_interval_post_all_sessions]

        c = "white"
        bplot = plt.boxplot(coeff_interval_post_all_sessions, positions=np.arange(parts_div_distance), patch_artist=True,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xlabel("Interval of goal coding")
        plt.title("POST modes")
        plt.ylabel("Stability of reactivation")
        plt.show()

        distance_pre_with_post_all_sessions = []

        for interval_id in range(parts_div_distance):
            # go through all the sessions and chose pre/post modes
            distance_pre_with_post = []
            # similarity PRE - POST vs. goal coding
            for sess_id, (pre_modes_goal_coding_sess, min_dist_per_pre_mode_sess, mode_ids) in enumerate(zip(pre_modes_goal_coding, min_distance_per_pre_mode, pre_mode_ids)):
                # select modes for goal coding
                gc_selection = pre_modes_goal_coding_sess[mode_ids]
                distance_pre_with_post.append(min_dist_per_pre_mode_sess[np.logical_and(interval_id * interval_size < gc_selection,
                                                        (interval_id * interval_size) < gc_selection)])

            distance_pre_with_post_all_sessions.append(distance_pre_with_post)


        distance_pre_with_post_all_sessions = [np.hstack(x) for x in distance_pre_with_post_all_sessions]

        c = "white"
        bplot = plt.boxplot(distance_pre_with_post_all_sessions, positions=np.arange(parts_div_distance), patch_artist=True,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xlabel("Interval of goal coding")
        plt.title("PRE modes")
        plt.ylabel("Min. distance to any POST modes")
        plt.show()

        distance_post_with_pre_all_sessions = []

        for interval_id in range(parts_div_distance):
            # go through all the sessions and chose pre/post modes
            distance_post_with_pre = []
            # similarity PRE - POST vs. goal coding
            for sess_id, (post_modes_goal_coding_sess, min_dist_per_post_mode_sess, mode_ids) in enumerate(zip(post_modes_goal_coding, min_distance_per_post_mode, post_mode_ids)):
                # select modes for goal coding
                gc_selection = post_modes_goal_coding_sess[mode_ids]
                distance_post_with_pre.append(min_dist_per_post_mode_sess[np.logical_and(interval_id * interval_size < gc_selection,
                                                        (interval_id * interval_size) < gc_selection)])

            distance_post_with_pre_all_sessions.append(distance_post_with_pre)


        distance_post_with_pre_all_sessions = [np.hstack(x) for x in distance_post_with_pre_all_sessions]

        c = "white"
        bplot = plt.boxplot(distance_post_with_pre_all_sessions, positions=np.arange(parts_div_distance), patch_artist=True,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xlabel("Interval of goal coding")
        plt.title("POST modes")
        plt.ylabel("Min. distance to any PRE modes")
        plt.show()

    def pre_long_sleep_post_cell_type_correlations(self):
        """
        Computes correlations of sleep activity with PRE and POST behavioral activity

        """
        stable_pre = []
        stable_post = []
        inc_post = []
        dec_pre = []
        stable_inc_post = []
        stable_dec_pre = []
        for session in self.session_list:
            spre, spost, ipost, dpost, sipost, sdpre = \
                session.pre_long_sleep_post().cell_type_correlations(plotting=False)
            stable_pre.append(spre)
            stable_post.append(spost)
            inc_post.append(ipost)
            dec_pre.append(dpost)
            stable_inc_post.append(sipost)
            stable_dec_pre.append(sdpre)

        # plot stable pre/post
        for st_pre, st_post in zip(stable_pre, stable_post):
            plt.plot(st_pre, color="green", label="PRE")
            plt.plot(st_post, color="orange", label="POST")
            plt.ylabel("PEARSON R")
        plt.title("STABLE")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()

        # plot inc post
        for inc_po in inc_post:
            # plt.ylim([0, y_max])
            plt.plot(inc_po)
        plt.title("INCREASING")
        plt.show()

        # plot dec pre
        for de_pre in dec_pre:
            # plt.ylim([0, y_max])
            plt.plot(de_pre)
        plt.title("DECREASING")
        plt.show()

        # plot stable inc post
        for si_post in stable_inc_post:
            plt.plot(si_post)
        plt.title("STABLE-INC")
        plt.show()

        # plot stable dec pre
        for sd_post in stable_dec_pre:
            plt.plot(sd_post)
        plt.title("STABLE-DEC")
        plt.show()

    def pre_long_sleep_post_firing_rates(self, cells_to_use="stable", save_fig=True, separate_sleep_phase=True):
        """
        Compares firing rates in PRE, sleep and POST

        :param cells_to_use: which cells to use ("stable", "decreasing", "increasing")
        :type cells_to_use: str
        :param save_fig: save as .svg
        :type save_fig: bool
        :param separate_sleep_phase: separate sleep into nrem, rem
        :type separate_sleep_phase: bool
        """
        if separate_sleep_phase:
            firing_pre_norm = []
            firing_sleep_rem_norm = []
            firing_sleep_nrem_norm = []
            firing_post_norm = []

            for session in self.session_list:
                pre_, rem_, nrem_, post_ = \
                    session.pre_long_sleep_post().firing_rate_distributions(cells_to_use=cells_to_use, plotting=False,
                                                                            separate_sleep_phases=True)

                firing_pre_norm.append(pre_)
                firing_sleep_rem_norm.append(rem_)
                firing_sleep_nrem_norm.append(nrem_)
                firing_post_norm.append(post_)

            firing_pre_norm = np.hstack(firing_pre_norm)
            firing_sleep_rem_norm = np.hstack(firing_sleep_rem_norm)
            firing_sleep_nrem_norm = np.hstack(firing_sleep_nrem_norm)
            firing_post_norm = np.hstack(firing_post_norm)

            p_pre_norm = 1. * np.arange(firing_pre_norm.shape[0]) / (firing_pre_norm.shape[0] - 1)
            p_sleep_nrem_norm = 1. * np.arange(firing_sleep_nrem_norm.shape[0]) / (firing_sleep_nrem_norm.shape[0] - 1)
            p_sleep_rem_norm = 1. * np.arange(firing_sleep_rem_norm.shape[0]) / (firing_sleep_rem_norm.shape[0] - 1)
            p_post_norm = 1. * np.arange(firing_post_norm.shape[0]) / (firing_post_norm.shape[0] - 1)

            if save_fig:
                plt.close()
                plt.style.use('default')

            plt.plot(np.sort(firing_pre_norm), p_pre_norm, label="PRE")
            plt.plot(np.sort(firing_sleep_rem_norm), p_sleep_rem_norm, label="REM")
            plt.plot(np.sort(firing_sleep_nrem_norm), p_sleep_nrem_norm, label="NREM")
            plt.plot(np.sort(firing_post_norm), p_post_norm, label="POST")
            plt.title(cells_to_use)
            plt.xlabel("Mean firing rate / normalized")
            plt.ylabel("CDF")
            plt.legend()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "firing_rates_pre_sleep_post_" + cells_to_use + ".svg"),
                            transparent="True")
            else:
                plt.title(cells_to_use)
                plt.show()

        else:
            pre = []
            sleep = []
            post = []

            for session in self.session_list:
                pre_, sleep_, post_ = \
                    session.pre_long_sleep_post().firing_rate_distributions(cells_to_use=cells_to_use, plotting=False,
                                                                            separate_sleep_phases=False)

                pre.append(pre_)
                sleep.append(sleep_)
                post.append(post_)

            pre = np.hstack(pre)
            sleep = np.hstack(sleep)
            post = np.hstack(post)

            # stats
            p_pre_sleep = mannwhitneyu(pre, sleep, alternative="less")[1]
            p_post_sleep = mannwhitneyu(post, sleep, alternative="less")[1]
            print("PRE-sleep, p-value = "+str(p_pre_sleep))
            print("POST-sleep, p-value = " + str(p_post_sleep))

            p_pre_stable = 1. * np.arange(pre.shape[0]) / (pre.shape[0] - 1)
            p_sleep_stable = 1. * np.arange(sleep.shape[0]) / (sleep.shape[0] - 1)
            p_post_stable = 1. * np.arange(post.shape[0]) / (post.shape[0] - 1)

            if save_fig:
                plt.close()
                plt.style.use('default')
            plt.plot(np.sort(pre), p_pre_stable, label="PRE")
            plt.plot(np.sort(sleep), p_sleep_stable, label="Sleep")
            plt.plot(np.sort(post), p_post_stable, label="POST")
            plt.xlabel("Mean firing rate / normalized")
            plt.ylabel("CDF")
            plt.legend()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "firing_rates_pre_sleep_post_"+cells_to_use+".svg"),
                            transparent="True")
            else:
                plt.title(cells_to_use)
                plt.show()

    def pre_long_sleep_post_firing_rates_stable_vs_decreasing(self, save_fig=True):
        """
        compares firing rates of stable and decreasing cells

        :param save_fig: save as .svg
        :type save_fig: bool
        """
        # data for stable cells
        firing_pre_norm_stable = []
        firing_sleep_rem_norm_stable = []
        firing_sleep_nrem_norm_stable = []
        firing_post_norm_stable = []
        # data for decreasing cells
        firing_pre_norm_dec = []
        firing_sleep_rem_norm_dec = []
        firing_sleep_nrem_norm_dec = []
        firing_post_norm_dec = []

        for session in self.session_list:
            pre_, rem_, nrem_, post_ = \
                session.pre_long_sleep_post().firing_rate_distributions(cells_to_use="stable", plotting=False,
                                                                        separate_sleep_phases=True)

            firing_pre_norm_stable.append(pre_)
            firing_sleep_rem_norm_stable.append(rem_)
            firing_sleep_nrem_norm_stable.append(nrem_)
            firing_post_norm_stable.append(post_)

            pre_, rem_, nrem_, post_ = \
                session.pre_long_sleep_post().firing_rate_distributions(cells_to_use="decreasing", plotting=False,
                                                                        separate_sleep_phases=True)

            firing_pre_norm_dec.append(pre_)
            firing_sleep_rem_norm_dec.append(rem_)
            firing_sleep_nrem_norm_dec.append(nrem_)
            firing_post_norm_dec.append(post_)

        firing_pre_norm_stable = np.hstack(firing_pre_norm_stable)
        firing_sleep_rem_norm_stable = np.hstack(firing_sleep_rem_norm_stable)
        firing_sleep_nrem_norm_stable = np.hstack(firing_sleep_nrem_norm_stable)
        firing_post_norm_stable = np.hstack(firing_post_norm_stable)

        p_pre_norm_stable = 1. * np.arange(firing_pre_norm_stable.shape[0]) / (firing_pre_norm_stable.shape[0] - 1)
        p_sleep_nrem_norm_stable = 1. * np.arange(firing_sleep_nrem_norm_stable.shape[0]) / (firing_sleep_nrem_norm_stable.shape[0] - 1)
        p_sleep_rem_norm_stable = 1. * np.arange(firing_sleep_rem_norm_stable.shape[0]) / (firing_sleep_rem_norm_stable.shape[0] - 1)
        p_post_norm_stable = 1. * np.arange(firing_post_norm_stable.shape[0]) / (firing_post_norm_stable.shape[0] - 1)

        firing_pre_norm_dec = np.hstack(firing_pre_norm_dec)
        firing_sleep_rem_norm_dec = np.hstack(firing_sleep_rem_norm_dec)
        firing_sleep_nrem_norm_dec = np.hstack(firing_sleep_nrem_norm_dec)
        firing_post_norm_dec = np.hstack(firing_post_norm_dec)

        p_pre_norm_dec = 1. * np.arange(firing_pre_norm_dec.shape[0]) / (firing_pre_norm_dec.shape[0] - 1)
        p_sleep_nrem_norm_dec = 1. * np.arange(firing_sleep_nrem_norm_dec.shape[0]) / (firing_sleep_nrem_norm_dec.shape[0] - 1)
        p_sleep_rem_norm_dec = 1. * np.arange(firing_sleep_rem_norm_dec.shape[0]) / (firing_sleep_rem_norm_dec.shape[0] - 1)
        p_post_norm_dec = 1. * np.arange(firing_post_norm_dec.shape[0]) / (firing_post_norm_dec.shape[0] - 1)

        if save_fig:
            plt.close()
            plt.style.use('default')

        plt.plot(np.sort(firing_sleep_rem_norm_stable), p_sleep_rem_norm_stable, label="stable: REM", color="violet")
        plt.plot(np.sort(firing_sleep_nrem_norm_stable), p_sleep_nrem_norm_stable, label="stable: NREM", color="violet",
                 linestyle="--")
        plt.plot(np.sort(firing_sleep_rem_norm_dec), p_sleep_rem_norm_dec, label="dec: REM", color="turquoise")
        plt.plot(np.sort(firing_sleep_nrem_norm_dec), p_sleep_nrem_norm_dec, label="dec: NREM", color="turquoise",
                 linestyle="--")
        plt.xlabel("Mean firing rate / normalized")
        plt.ylabel("CDF")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_rates_pre_sleep_post_stable_vs_decreasing.svg"),
                        transparent="True")
        else:
            plt.show()

    def pre_long_sleep_plot_cell_classification_mean_firing_rates_awake(self, save_fig=False, log_scale=False,
                                                                        normalize=False, smooth_sleep=False,
                                                                        n_chunks_for_sleep=10, n_smoothing=2,
                                                                        power_scale=None, midnorm_scale=False,
                                                                        firing_rate_cap=10):
        """
        Plots imshow plot of mean firing rates for classified cells

        :param save_fig: save as .svg
        :type save_fig: bool
        """
        raster_ds_smoothed_stable = []
        pre_raster_mean_stable = []
        post_raster_mean_stable = []
        raster_ds_smoothed_decreasing = []
        pre_raster_mean_decreasing = []
        post_raster_mean_decreasing = []
        raster_ds_smoothed_increasing = []
        pre_raster_mean_increasing = []
        post_raster_mean_increasing = []

        for session in self.session_list:
            s_s, pre_s, post_s, s_d, pre_d, post_d, s_i, pre_i, post_i = \
                session.pre_long_sleep_post().cell_classification_mean_firing_rates_awake(plotting=False,
                                                    normalize=normalize, smooth_sleep=smooth_sleep, save_fig=False,
                                                    n_chunks_for_sleep=n_chunks_for_sleep, n_smoothing=n_smoothing)

            raster_ds_smoothed_stable.append(s_s)
            pre_raster_mean_stable.append(pre_s)
            post_raster_mean_stable.append(post_s)
            raster_ds_smoothed_decreasing.append(s_d)
            pre_raster_mean_decreasing.append(pre_d)
            post_raster_mean_decreasing.append(post_d)
            raster_ds_smoothed_increasing.append(s_i)
            pre_raster_mean_increasing.append(pre_i)
            post_raster_mean_increasing.append(post_i)

        raster_ds_smoothed_stable = np.vstack(raster_ds_smoothed_stable)
        pre_raster_mean_stable = np.hstack(pre_raster_mean_stable)
        post_raster_mean_stable = np.hstack(post_raster_mean_stable)
        raster_ds_smoothed_decreasing = np.vstack(raster_ds_smoothed_decreasing)
        pre_raster_mean_decreasing = np.hstack(pre_raster_mean_decreasing)
        post_raster_mean_decreasing = np.hstack(post_raster_mean_decreasing)
        raster_ds_smoothed_increasing = np.vstack(raster_ds_smoothed_increasing)
        pre_raster_mean_increasing = np.hstack(pre_raster_mean_increasing)
        post_raster_mean_increasing = np.hstack(post_raster_mean_increasing)

        raster_ds_smoothed_stable_sorted = raster_ds_smoothed_stable[pre_raster_mean_stable.argsort(), :]
        pre_raster_mean_stable_sorted = pre_raster_mean_stable[pre_raster_mean_stable.argsort()]
        post_raster_mean_stable_sorted = post_raster_mean_stable[pre_raster_mean_stable.argsort()]

        raster_ds_smoothed_increasing_sorted = raster_ds_smoothed_increasing[pre_raster_mean_increasing.argsort(), :]
        pre_raster_mean_increasing_sorted = pre_raster_mean_increasing[pre_raster_mean_increasing.argsort()]
        post_raster_mean_increasing_sorted = post_raster_mean_increasing[pre_raster_mean_increasing.argsort()]

        raster_ds_smoothed_decreasing_sorted = raster_ds_smoothed_decreasing[pre_raster_mean_decreasing.argsort(), :]
        pre_raster_mean_decreasing_sorted = pre_raster_mean_decreasing[pre_raster_mean_decreasing.argsort()]
        post_raster_mean_decreasing_sorted = post_raster_mean_decreasing[pre_raster_mean_decreasing.argsort()]

        # stack them back together for plotting
        pre_raster_mean_sorted = np.hstack(
            (pre_raster_mean_stable_sorted, pre_raster_mean_increasing_sorted, pre_raster_mean_decreasing_sorted))
        post_raster_mean_sorted = np.hstack(
            (post_raster_mean_stable_sorted, post_raster_mean_increasing_sorted, post_raster_mean_decreasing_sorted))

        raster_ds_smoothed_sorted = np.vstack((raster_ds_smoothed_stable_sorted, raster_ds_smoothed_increasing_sorted,
                                               raster_ds_smoothed_decreasing_sorted))

        nr_stable_cells = raster_ds_smoothed_stable_sorted.shape[0]
        nr_dec_cells = raster_ds_smoothed_decreasing_sorted.shape[0]
        nr_inc_cells = raster_ds_smoothed_increasing_sorted.shape[0]

        pre_data_raw = np.expand_dims(pre_raster_mean_sorted, 1)
        post_data_raw = np.expand_dims(post_raster_mean_sorted, 1)
        sleep_data_raw = raster_ds_smoothed_sorted

        # cap firing rates for better visualization
        # --------------------------------------------------------------------------------------------------------------
        all_data = np.hstack((pre_data_raw, sleep_data_raw, post_data_raw))
        all_data[all_data > firing_rate_cap] = firing_rate_cap
        all_data = all_data - np.min(all_data)

        # good cells
        good_cells = np.count_nonzero(all_data, axis=1) > 0.75*all_data.shape[1]
        # need to adjust number of cells per subset
        nr_dec_cells = np.count_nonzero(good_cells[(nr_stable_cells+nr_inc_cells):])
        nr_inc_cells = np.count_nonzero(good_cells[nr_stable_cells:(nr_stable_cells+nr_inc_cells)])
        nr_stable_cells = np.count_nonzero(good_cells[:nr_stable_cells])

        # remove cells that do not fire in one of the episodes
        all_data = all_data[good_cells,:]
        all_data += 10e-200

        v_max = np.max(all_data)
        v_min = np.min(all_data)

        pre_data = np.expand_dims(all_data[:, 0], 1)
        post_data = np.expand_dims(all_data[:, -1], 1)
        sleep_data = all_data[:,1:-1]

        # new_norm = NonLinearNormalize(vmin=np.min(all_data), vmax=np.max(all_data), a1=10, a2=0.5)
        # plt.figure(figsize=(5, 10))
        # plt.imshow(all_data, interpolation="nearest", aspect ="auto", norm=new_norm)
        # a = plt.colorbar()
        # plt.show()

        if save_fig:
            plt.style.use('default')

        fig = plt.figure(figsize=(4, 12))
        # fig = plt.figure(figsize=(8, 12))
        gs = fig.add_gridspec(15, 20)
        ax1 = fig.add_subplot(gs[:-3, :2])
        ax2 = fig.add_subplot(gs[:-3, 2:-2])
        ax3 = fig.add_subplot(gs[:-3, -2:])
        ax4 = fig.add_subplot(gs[-1, :10])

        if log_scale:
            ax1.imshow(pre_data, norm=LogNorm(0.01, v_max), interpolation='nearest', aspect='auto')
        elif midnorm_scale:
            ax1.imshow(pre_data, norm=NonLinearNormalize(vmin=v_min, vmax=v_max,  a1=7, a2=0.25),
                       interpolation='nearest', aspect='auto')
        elif power_scale is not None:
            ax1.imshow(pre_data, norm=PowerNorm(gamma=power_scale, vmin=v_min, vmax=v_max),
                       interpolation='nearest', aspect='auto')
        else:
            ax1.imshow(pre_data, vmin=v_min, vmax=v_max, interpolation='nearest', aspect='auto')
        ax1.hlines(nr_stable_cells, -0.5, 0.5, color="red")
        ax1.hlines(nr_stable_cells + nr_inc_cells, -0.5, 0.5, color="red")
        ax1.set_xticks([])
        ax1.set_ylabel("Cells")
        ax1.set_ylim(nr_inc_cells + nr_stable_cells + nr_dec_cells, 0)
        if log_scale:
            cax = ax2.imshow(sleep_data, norm=LogNorm(0.01, v_max),
                             interpolation='none', aspect='auto')
        elif midnorm_scale:
            cax = ax2.imshow(sleep_data, norm=NonLinearNormalize(vmin=v_min, vmax=v_max, a1=7, a2=0.25),
                             interpolation='nearest', aspect='auto')
        elif power_scale is not None:
            cax = ax2.imshow(sleep_data, norm=PowerNorm(gamma=power_scale, vmin=v_min, vmax=v_max),
                             interpolation='none', aspect='auto')
        else:
            cax = ax2.imshow(sleep_data, vmin=v_min, vmax=v_max, interpolation='nearest', aspect='auto')
        ax2.hlines(nr_stable_cells, -0.5, raster_ds_smoothed_sorted.shape[1] - 0.5, color="red")
        ax2.hlines(nr_stable_cells + nr_inc_cells, -0.5,
                   raster_ds_smoothed_sorted.shape[1] - 0.5,
                   color="red")
        ax2.set_yticks([])

        ax2.set_xlabel("Normalized sleep duration")
        if log_scale:
            ax3.imshow(post_data, norm=LogNorm(0.01, v_max), interpolation='nearest',
                       aspect='auto')
        elif midnorm_scale:
            ax3.imshow(post_data, norm=NonLinearNormalize(vmin=v_min, vmax=v_max, a1=7, a2=0.25), interpolation='nearest',
                       aspect='auto')
        elif power_scale is not None:
            ax3.imshow(post_data, norm=PowerNorm(gamma=power_scale, vmin=v_min, vmax=v_max), interpolation='nearest',
                       aspect='auto')
        else:
            ax3.imshow(post_data, vmin=v_min, vmax=v_max, interpolation='nearest',
                       aspect='auto')
        ax3.hlines(nr_stable_cells, -0.5, 0.5, color="red")
        ax3.hlines(nr_stable_cells + nr_inc_cells, -0.5, 0.5, color="red")
        ax3.set_yticks([])
        ax3.set_xticks([])
        a = fig.colorbar(mappable=cax, cax=ax4, orientation="horizontal")
        # a.ax.set_xticklabels(["0", "1"])
        # a.ax.set_xlabel("Normalized firing rate")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            if log_scale:
                plt.savefig(os.path.join(save_path, "cell_classification_all_cells_log_scale.svg"), transparent="True")
            elif midnorm_scale:
                plt.savefig(os.path.join(save_path, "cell_classification_all_cells_midnorm_scale.svg"), transparent="True")
        else:
            plt.show()

    def pre_long_sleep_post_firing_rates_all_cells(self, measure="mean", save_fig=False, plotting=True,
                                                   chunks_in_min=2, z_score=False, use_log_scale=False):
        """
        Computes firing rates in PRE, long sleep and POST for all cells

        :param measure: "mean" or "max" firing rates
        :type measure: str
        :param save_fig: save as .svg
        :type save_fig: bool
        :param plotting: whether to plot results
        :type plotting: bool
        :param chunks_in_min: in what chunks of data to compute mean or max firing rate
        :type chunks_in_min: int
        :return: arrays for each subset of cells and experiment phase
        :rtype: np.array
        """
        firing_pre_stable = []
        firing_pre_dec = []
        firing_pre_inc = []
        firing_sleep_stable = []
        firing_sleep_dec = []
        firing_sleep_inc = []
        firing_post_stable = []
        firing_post_dec = []
        firing_post_inc = []

        for session in self.session_list:
            pre_s, pre_d, pre_i, s_s, s_d, s_i, post_s, post_d, post_i = \
                session.pre_long_sleep_post().firing_rate_distributions_all_cells(plotting=False, measure=measure,
                                                                                  chunks_in_min=chunks_in_min,
                                                                                  z_score=z_score)

            firing_pre_stable.append(pre_s)
            firing_pre_dec.append(pre_d)
            firing_pre_inc.append(pre_i)
            firing_sleep_stable.append(s_s)
            firing_sleep_dec.append(s_d)
            firing_sleep_inc.append(s_i)
            firing_post_stable.append(post_s)
            firing_post_dec.append(post_d)
            firing_post_inc.append(post_i)

        firing_pre_stable = np.hstack(firing_pre_stable)
        firing_pre_dec = np.hstack(firing_pre_dec)
        firing_pre_inc = np.hstack(firing_pre_inc)
        firing_sleep_stable = np.hstack(firing_sleep_stable)
        firing_sleep_dec = np.hstack(firing_sleep_dec)
        firing_sleep_inc = np.hstack(firing_sleep_inc)
        firing_post_stable = np.hstack(firing_post_stable)
        firing_post_dec = np.hstack(firing_post_dec)
        firing_post_inc = np.hstack(firing_post_inc)

        # first for PRE
        # --------------------------------------------------------------------------------------------------------------
        plt.figure(figsize=(2, 3))

        if use_log_scale:
            plt.yscale("log")
            y_base=10
            y_offset=5
        else:
            y_base=8
            y_offset =1
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        # filter low firing cells so that whiskers do not extend towards 0
        firing_pre_stable_fil = firing_pre_stable[firing_pre_stable > 0.01]
        firing_pre_inc_fil = firing_pre_inc[firing_pre_inc > 0.01]
        firing_pre_dec_fil = firing_pre_dec[firing_pre_dec > 0.01]

        bplot = plt.boxplot([firing_pre_stable_fil, firing_pre_dec_fil, firing_pre_inc_fil],
                            positions=[1, 2, 3], patch_artist=True,
                            labels=["persistent", "decreasing", "increasing"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["violet", 'turquoise', "orange"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        # plt.yscale("log")
        # plt.ylim(10-3,10)
        plt.ylabel("Mean firing rate (Hz)")
        plt.xticks(rotation=45)
        plt.hlines(y_base, 1, 1.9, colors=c)
        if mannwhitneyu(firing_pre_stable, firing_pre_dec)[1] < 0.001/3:
            plt.text(1.3, y_base, "***")
        elif mannwhitneyu(firing_pre_stable, firing_pre_dec)[1] < 0.01/3:
            plt.text(1.3, y_base, "**")
        elif mannwhitneyu(firing_pre_stable, firing_pre_dec)[1] < 0.05/3:
            plt.text(1.3, y_base, "*")
        else:
            plt.text(1.3, y_base, "n.s.")
        plt.hlines(y_base, 2.1, 3, colors=c)
        if mannwhitneyu(firing_pre_inc, firing_pre_dec)[1] < 0.001/3:
            plt.text(2.3, y_base, "***")
        elif mannwhitneyu(firing_pre_inc, firing_pre_dec)[1] < 0.01/3:
            plt.text(2.3, y_base, "**")
        elif mannwhitneyu(firing_pre_inc, firing_pre_dec)[1] < 0.05/3:
            plt.text(2.3, y_base, "*")
        else:
            plt.text(2.3, y_base, "n.s.")
        plt.hlines(y_base + y_offset, 1, 3, colors=c)
        if mannwhitneyu(firing_pre_inc, firing_pre_stable)[1] < 0.001/3:
            plt.text(1.8, y_base + y_offset, "***")
        elif mannwhitneyu(firing_pre_inc, firing_pre_stable)[1] < 0.01/3:
            plt.text(1.8, y_base + y_offset, "**")
        elif mannwhitneyu(firing_pre_inc, firing_pre_stable)[1] < 0.05/3:
            plt.text(1.8, y_base + y_offset, "*")
        else:
            plt.text(1.8, y_base + y_offset, "n.s.")
        plt.ylim(10e-3, 30)
        plt.title("Acquisition")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_rate_" + measure +"PRE_all_cells.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        # Sleep
        # --------------------------------------------------------------------------------------------------------------
        plt.figure(figsize=(2, 3))

        if use_log_scale:
            plt.yscale("log")
            y_base=10
            y_offset=5
        else:
            y_base=8
            y_offset =1
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        # filter low firing cells so that whiskers do not extend towards 0
        firing_sleep_stable_fil = firing_sleep_stable[firing_sleep_stable > 0.01]
        firing_sleep_inc_fil = firing_sleep_inc[firing_sleep_inc > 0.01]
        firing_sleep_dec_fil = firing_sleep_dec[firing_sleep_dec > 0.01]

        bplot = plt.boxplot([firing_sleep_stable_fil, firing_sleep_dec_fil, firing_sleep_inc_fil],
                            positions=[1, 2, 3], patch_artist=True,
                            labels=["persistent", "decreasing", "increasing"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["violet", 'turquoise', "orange"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        # plt.yscale("log")
        # plt.ylim(10-3,10)
        plt.ylabel("Mean firing rate (Hz)")
        plt.xticks(rotation=45)
        plt.hlines(y_base, 1, 1.9, colors=c)
        if mannwhitneyu(firing_sleep_stable, firing_sleep_dec)[1] < 0.001/3:
            plt.text(1.3, y_base, "***")
        elif mannwhitneyu(firing_sleep_stable, firing_sleep_dec)[1] < 0.01/3:
            plt.text(1.3, y_base, "**")
        elif mannwhitneyu(firing_sleep_stable, firing_sleep_dec)[1] < 0.05/3:
            plt.text(1.3, y_base, "*")
        else:
            plt.text(1.3, y_base, "n.s.")
        plt.hlines(y_base, 2.1, 3, colors=c)
        if mannwhitneyu(firing_sleep_inc, firing_sleep_dec)[1] < 0.001/3:
            plt.text(2.3, y_base, "***")
        elif mannwhitneyu(firing_sleep_inc, firing_sleep_dec)[1] < 0.01/3:
            plt.text(2.3, y_base, "**")
        elif mannwhitneyu(firing_sleep_inc, firing_sleep_dec)[1] < 0.05/3:
            plt.text(2.3, y_base, "*")
        else:
            plt.text(2.3, y_base, "n.s.")
        plt.hlines(y_base + y_offset, 1, 3, colors=c)
        if mannwhitneyu(firing_sleep_inc, firing_sleep_stable)[1] < 0.001/3:
            plt.text(1.8, y_base + y_offset, "***")
        elif mannwhitneyu(firing_sleep_inc, firing_sleep_stable)[1] < 0.01/3:
            plt.text(1.8, y_base + y_offset, "**")
        elif mannwhitneyu(firing_sleep_inc, firing_sleep_stable)[1] < 0.05/3:
            plt.text(1.8, y_base + y_offset, "*")
        else:
            plt.text(1.8, y_base + y_offset, "n.s.")
        plt.ylim(10e-3, 30)
        plt.title("Sleep")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_rate_" + measure +"_SLEEP_all_cells.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        # Recall
        # --------------------------------------------------------------------------------------------------------------
        plt.figure(figsize=(2, 3))

        if use_log_scale:
            plt.yscale("log")
            y_base=10
            y_offset=5
        else:
            y_base=8
            y_offset =1
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        # filter low firing cells so that whiskers do not extend towards 0
        firing_post_stable_fil = firing_post_stable[firing_post_stable > 0.01]
        firing_post_inc_fil = firing_post_inc[firing_post_inc > 0.01]
        firing_post_dec_fil = firing_post_dec[firing_post_dec > 0.01]

        bplot = plt.boxplot([firing_post_stable_fil, firing_post_dec_fil, firing_post_inc_fil],
                            positions=[1, 2, 3], patch_artist=True,
                            labels=["persistent", "decreasing", "increasing"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["violet", 'turquoise', "orange"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        # plt.yscale("log")
        # plt.ylim(10-3,10)
        plt.ylabel("Mean firing rate (Hz)")
        plt.xticks(rotation=45)
        plt.hlines(y_base, 1, 1.9, colors=c)
        if mannwhitneyu(firing_post_stable, firing_post_dec)[1] < 0.001/3:
            plt.text(1.3, y_base, "***")
        elif mannwhitneyu(firing_post_stable, firing_post_dec)[1] < 0.01/3:
            plt.text(1.3, y_base, "**")
        elif mannwhitneyu(firing_post_stable, firing_post_dec)[1] < 0.05/3:
            plt.text(1.3, y_base, "*")
        else:
            plt.text(1.3, y_base, "n.s.")
        plt.hlines(y_base, 2.1, 3, colors=c)
        if mannwhitneyu(firing_post_inc, firing_post_dec)[1] < 0.001/3:
            plt.text(2.3, y_base, "***")
        elif mannwhitneyu(firing_post_inc, firing_post_dec)[1] < 0.01/3:
            plt.text(2.3, y_base, "**")
        elif mannwhitneyu(firing_post_inc, firing_post_dec)[1] < 0.05/3:
            plt.text(2.3, y_base, "*")
        else:
            plt.text(2.3, y_base, "n.s.")
        plt.hlines(y_base + y_offset, 1, 3, colors=c)
        if mannwhitneyu(firing_post_inc, firing_post_stable)[1] < 0.001/3:
            plt.text(1.8, y_base + y_offset, "***")
        elif mannwhitneyu(firing_post_inc, firing_post_stable)[1] < 0.01/3:
            plt.text(1.8, y_base + y_offset, "**")
        elif mannwhitneyu(firing_post_inc, firing_post_stable)[1] < 0.05/3:
            plt.text(1.8, y_base + y_offset, "*")
        else:
            plt.text(1.8, y_base + y_offset, "n.s.")
        plt.ylim(10e-3, 30)
        plt.title("Recall")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_rate_" + measure +"_RECALL_all_cells.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        p_pre_stable = 1. * np.arange(firing_pre_stable.shape[0]) / (firing_pre_stable.shape[0] - 1)
        p_sleep_stable = 1. * np.arange(firing_sleep_stable.shape[0]) / (firing_sleep_stable.shape[0] - 1)
        p_post_stable = 1. * np.arange(firing_post_stable.shape[0]) / (firing_post_stable.shape[0] - 1)

        p_pre_dec = 1. * np.arange(firing_pre_dec.shape[0]) / (firing_pre_dec.shape[0] - 1)
        p_sleep_dec = 1. * np.arange(firing_sleep_dec.shape[0]) / (firing_sleep_dec.shape[0] - 1)
        p_post_dec = 1. * np.arange(firing_post_dec.shape[0]) / (firing_post_dec.shape[0] - 1)

        p_pre_inc = 1. * np.arange(firing_pre_inc.shape[0]) / (firing_pre_inc.shape[0] - 1)
        p_sleep_inc = 1. * np.arange(firing_sleep_inc.shape[0]) / (firing_sleep_inc.shape[0] - 1)
        p_post_inc = 1. * np.arange(firing_post_inc.shape[0]) / (firing_post_inc.shape[0] - 1)

        plt.plot(np.sort(firing_pre_stable), p_pre_stable)
        plt.plot(np.sort(firing_sleep_stable), p_sleep_stable)
        plt.show()

        print("PRE:")
        print(" - dec vs. stable")
        print(mannwhitneyu(firing_pre_dec, firing_pre_stable))
        print("POST:")
        print(" - inc vs. stable")
        print(mannwhitneyu(firing_post_inc, firing_post_stable))

        fig = plt.figure(figsize=(10,6))
        fig.add_subplot(1, 3, 1)
        plt.title("PRE (Stable)")
        plt.hist(firing_pre_stable, orientation="horizontal", density=True, bins=150)
        plt.ylim(-2, 5)
        plt.ylabel("z-scored max firing")
        fig.add_subplot(1, 3, 2)
        plt.title("Sleep")
        plt.hist(firing_sleep_stable, orientation="horizontal", density=True, bins=150)
        plt.ylim(-2, 5)
        fig.add_subplot(1, 3, 3)
        plt.title("POST")
        plt.hist(firing_post_stable, orientation="horizontal", density=True, bins=150)
        plt.ylim(-2, 5)
        plt.show()

        fig = plt.figure(figsize=(10,6))
        fig.add_subplot(1, 3, 1)
        plt.title("PRE (Dec)")
        plt.hist(firing_pre_dec, orientation="horizontal", density=True, bins=150)
        plt.ylim(-2, 10)
        plt.ylabel("z-scored max firing")
        fig.add_subplot(1, 3, 2)
        plt.title("Sleep")
        plt.hist(firing_sleep_dec, orientation="horizontal", density=True, bins=150)
        plt.ylim(-2, 10)
        fig.add_subplot(1, 3, 3)
        plt.title("POST")
        plt.hist(firing_post_dec, orientation="horizontal", density=True, bins=150)
        plt.ylim(-2, 10)
        plt.show()

        fig = plt.figure(figsize=(10,6))
        fig.add_subplot(1, 3, 1)
        plt.title("PRE (inc)")
        plt.hist(firing_pre_inc, orientation="horizontal", density=True, bins=150)
        plt.ylim(-2, 15)
        plt.ylabel("z-scored max firing")
        fig.add_subplot(1, 3, 2)
        plt.title("Sleep")
        plt.hist(firing_sleep_inc, orientation="horizontal", density=True, bins=150)
        plt.ylim(-2, 15)
        fig.add_subplot(1, 3, 3)
        plt.title("POST")
        plt.hist(firing_post_inc, orientation="horizontal", density=True, bins=150)
        plt.ylim(-2, 15)
        plt.show()

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        print("Stable: PRE-sleep")
        print(mannwhitneyu(firing_pre_stable, firing_sleep_stable, alternative="less"))

        print("Stable: sleep-POST")
        print(mannwhitneyu(firing_sleep_stable, firing_post_stable, alternative="greater"))

        print("Stable: PRE-POST")
        print(mannwhitneyu(firing_pre_stable, firing_post_stable))

        y_dat = [firing_pre_stable, firing_sleep_stable, firing_post_stable]
        plt.figure(figsize=(2,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2,3], patch_artist=True,
                            labels=["PRE", "Sleep", "POST"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        # whis = [0.01, 99.99]
        plt.title("Stable")
        # plt.yscale("symlog")
        if measure == "max":
            plt.ylim(-2, 2.5)
            plt.ylabel("Max firing rate (z-scored)")
        elif measure == "mean":
            plt.ylim(-1.3, 1.3)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_rate_" + measure + "_stable.svg"), transparent="True")
            plt.close()
        else:
            plt.show()


        print("Inc: PRE-sleep")
        print(mannwhitneyu(firing_pre_inc, firing_sleep_inc))

        print("Inc: PRE-POST")
        print(mannwhitneyu(firing_pre_inc, firing_post_inc))

        print("Inc: sleep-POST")
        print(mannwhitneyu(firing_sleep_inc, firing_post_inc, alternative="less"))
        y_dat = [firing_pre_inc, firing_sleep_inc, firing_post_inc]
        plt.figure(figsize=(2,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2,3], patch_artist=True,
                            labels=["PRE", "Sleep", "POST"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        plt.title("Inc")
        # plt.yscale("symlog")
        if measure == "max":
            plt.ylim(-2.8, 5.2)
            plt.ylabel("Max firing rate (z-scored)")
        elif measure == "mean":
            plt.ylim(-2.2, 6.2)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_rate_" + measure + "_increasing.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        print("Dec: PRE-sleep")
        print(mannwhitneyu(firing_pre_dec, firing_sleep_dec, alternative="greater"))

        print("Dec: PRE-POST")
        print(mannwhitneyu(firing_pre_dec, firing_post_dec))

        print("Dec: sleep-POST")
        print(mannwhitneyu(firing_sleep_dec, firing_post_dec, alternative="greater"))
        y_dat = [firing_pre_dec, firing_sleep_dec, firing_post_dec]
        plt.figure(figsize=(2,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2,3], patch_artist=True,
                            labels=["PRE", "Sleep", "POST"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        plt.title("Dec")
        # plt.yscale("symlog")
        if measure == "max":
            plt.ylim(-2, 4.8)
            plt.ylabel("Max firing rate (z-scored)")
        elif measure == "mean":
            plt.ylim(-1.5, 4.8)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_rate_" + measure + "_decreasing.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        if save_fig or plotting:

            if save_fig:
                plt.style.use('default')

            plt.plot(np.sort(firing_pre_stable), p_pre_stable, color="magenta", label="stable")
            plt.plot(np.sort(firing_pre_inc), p_pre_inc, color="orange", label="inc")
            plt.plot(np.sort(firing_pre_dec), p_pre_dec, color="turquoise", label="dec")
            plt.xlabel(measure + " firing rate (Hz)")
            plt.ylabel("CDF")
            plt.legend()
            plt.title("PRE")
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "firing_rate_"+measure+"_pre.svg"), transparent="True")
                plt.close()
            else:
                plt.show()

            plt.plot(np.sort(firing_sleep_stable), p_sleep_stable, color="magenta", label="stable")
            plt.plot(np.sort(firing_sleep_inc), p_sleep_inc, color="orange", label="inc")
            plt.plot(np.sort(firing_sleep_dec), p_sleep_dec, color="turquoise", label="dec")
            plt.xlabel(measure + " firing rate (Hz)")
            plt.ylabel("CDF")
            plt.legend()
            plt.title("Sleep")
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "firing_rate_"+measure+"_sleep.svg"), transparent="True")
                plt.close()
            else:
                plt.show()

            plt.plot(np.sort(firing_post_stable), p_post_stable, color="magenta", label="stable")
            plt.plot(np.sort(firing_post_inc), p_post_inc, color="orange", label="inc")
            plt.plot(np.sort(firing_post_dec), p_post_dec, color="turquoise", label="dec")
            plt.xlabel(measure + " firing rate (Hz)")
            plt.ylabel("CDF")
            plt.legend()
            plt.title("Post")
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "firing_rate_"+measure+"_post.svg"), transparent="True")
                plt.close()
            else:
                plt.show()
        else:
            return firing_pre_stable, firing_pre_dec, firing_pre_inc, firing_sleep_stable, firing_sleep_dec, \
                   firing_sleep_inc, firing_post_stable, firing_post_dec, firing_post_inc

    def pre_long_sleep_post_firing_rate_ratios_all_cells(self, measure="mean", save_fig=False, chunks_in_min=2):
        """
        Computes ratio of firing between Sleep/PRE and Sleep/POST

        :param measure: "mean" or "max" firing rates
        :type measure: str
        :param save_fig: save as .svg
        :type save_fig: bool
        :param chunks_in_min: in which chunks to compute mean or max firing rates
        :type chunks_in_min: int
        """
        ratio_pre_stable = []
        ratio_pre_dec = []
        ratio_pre_inc = []
        ratio_post_stable = []
        ratio_post_dec = []
        ratio_post_inc = []

        for session in self.session_list:
            pre_s, post_s, pre_d, post_d, pre_i, post_i = \
                session.pre_long_sleep_post().firing_rate_ratios_all_cells(plotting=False, measure=measure,
                                                                                  chunks_in_min=chunks_in_min)

            ratio_pre_stable.append(pre_s)
            ratio_post_stable.append(post_s)
            ratio_pre_dec.append(pre_d)
            ratio_post_dec.append(post_d)
            ratio_pre_inc.append(pre_i)
            ratio_post_inc.append(post_i)

        ratio_pre_stable = np.hstack(ratio_pre_stable)
        ratio_pre_dec = np.hstack(ratio_pre_dec)
        ratio_pre_inc = np.hstack(ratio_pre_inc)
        ratio_post_stable = np.hstack(ratio_post_stable)
        ratio_post_dec = np.hstack(ratio_post_dec)
        ratio_post_inc = np.hstack(ratio_post_inc)

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        print("Stable")
        print(mannwhitneyu(ratio_pre_stable, ratio_post_stable))

        y_dat = [ratio_pre_stable, ratio_post_stable]
        plt.figure(figsize=(3, 5))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["Acquisition", "Recall"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False, widths=(0.25, 0.25))
        # whis = [0.01, 99.99]
        plt.title("Persistent")
        plt.xticks(rotation=45)
        plt.tight_layout()
        if measure == "mean":
            plt.ylim(-1.4, 1.4)
            plt.ylabel("Mean firing rate ratio")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_rate_ratio_" + measure + "_stable.svg"), transparent="True")
            plt.close()
        else:
            plt.show()


        print("Inc")
        print(mannwhitneyu(ratio_pre_inc, ratio_post_inc))

        y_dat = [ratio_pre_inc, ratio_post_inc]
        plt.figure(figsize=(3,5))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["Acquisition", "Recall"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False, widths=(0.25, 0.25))
        plt.title("Increasing")
        plt.xticks(rotation=45)
        plt.tight_layout()
        # plt.yscale("symlog")
        # if measure == "max":
        #     plt.ylim(-2.8, 5.2)
        #     plt.ylabel("Max firing rate (z-scored)")
        if measure == "mean":
            plt.ylim(-1.4, 1.4)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_rate_ratio_" + measure + "_increasing.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        print("Dec")
        print(mannwhitneyu(ratio_pre_dec, ratio_post_dec))
        y_dat = [ratio_pre_dec, ratio_post_dec]
        plt.figure(figsize=(3,5))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["Acquisition", "Recall"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False, widths=(0.25, 0.25))
        plt.title("Decreasing")
        plt.xticks(rotation=45)
        plt.tight_layout()
        # plt.yscale("symlog")
        # if measure == "max":
        #     plt.ylim(-2, 4.8)
        #     plt.ylabel("Max firing rate (z-scored)")
        if measure == "mean":
            plt.ylim(-1.4, 1.4)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_rate_ratio_" + measure + "_decreasing.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def pre_long_sleep_drift_correlation_structure_equalized_firing_rates(self, save_fig=False, n_smoothing=40,
                                                                          plot_mean=False, cells_to_use="stable"):
        """
        Looks at drift of correlation structure comparing sleep correlation matrices with PRE and POST. Firing rates
        of cells are equalized to exclude influence of single cell firing rate changes

        :param save_fig: as .svg
        :type save_fig: bool
        :param n_smoothing: how much smoothing to apply to result
        :type n_smoothing: int
        :param plot_mean: if mean is supposed to be plotted
        :type plot_mean: bool
        :param cells_to_use: which cells to use ("stable", "decreasing", "increasing")
        :type cells_to_use:
        """
        sim_ratio = []
        for session in self.session_list:
            s_r = session.pre_long_sleep_post().drift_correlation_structure_equalized_firing_rates(plotting=False,
                                                                                                   cells_to_use=
                                                                                                   cells_to_use)
            sim_ratio.append(s_r)

        if save_fig:
            plt.style.use('default')

        max_y = 0
        fig = plt.figure(figsize=(4,3))
        # plt.figure(figsize=(3,4))
        ax = fig.add_subplot()
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        first_val = []
        last_val = []
        sim_ratios_smooth = []
        sim_ratios_xaxis = []
        for i, session_sim_ratio in enumerate(sim_ratio):
            session_sim_ratio = np.array(session_sim_ratio)
            # smoothing
            session_sim_ratio = moving_average(a=session_sim_ratio, n=n_smoothing)
            sim_ratios_smooth.append(session_sim_ratio)
            sim_ratios_xaxis.append(np.linspace(0, 1, session_sim_ratio.shape[0]))
            first_val.append(session_sim_ratio[0])
            last_val.append(session_sim_ratio[-1])
            max_y = np.max(np.append(np.abs(session_sim_ratio), max_y))
            ax.plot(np.linspace(0, 1, session_sim_ratio.shape[0]),session_sim_ratio, linewidth=2, c=col[i], label=str(i))

        if plot_mean:
            nr_x_ticks = [x.shape[0] for x in sim_ratios_xaxis]
            x_axis_new = sim_ratios_xaxis[np.argmax(nr_x_ticks)]
            sim_ratios_smooth_equal_length = []
            for session_sim_ratio_smooth, x_axis in zip(sim_ratios_smooth, sim_ratios_xaxis):
                sim_ratios_smooth_equal_length.append(np.interp(x=x_axis_new, xp=x_axis, fp=session_sim_ratio_smooth))
            sim_ratios_smooth_equal_length = np.vstack(sim_ratios_smooth_equal_length)
            mean_sim_ratios_smooth = np.mean(sim_ratios_smooth_equal_length, axis=0)

            ax.plot(np.linspace(0, 1, mean_sim_ratios_smooth.shape[0]), mean_sim_ratios_smooth, linewidth=2, c="gray",
                    label="mean")
        plt.legend()
        # plt.ylim(-max_y-0.1, max_y+0.1)
        plt.ylim(-1,1)
        plt.xlim(-0.05,1.05)
        plt.ylabel("sim_ratio correlations")
        plt.xlabel("Normalized sleep duration")
        plt.grid(axis='y', color="gray")
        plt.yticks([-1,-0.5, 0, 0.5,1], ["-1", "-0.5", "0", "0.5", "-1"])
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_correlations_stable_equalized_firing_"+cells_to_use+".svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        # check if drift is significant
        init_vals = []
        end_vals = []

        for sess_data in sim_ratios_smooth:
            init_vals.append(sess_data[0])
            end_vals.append(sess_data[-1])
            # init_vals.append(np.mean(sess_data[:10]))
            # end_vals.append(np.mean(sess_data[-10:]))

        print("T-test (init vals vs. end vals):")
        print(ttest_ind(init_vals, end_vals))
        print("T-test (init vals < 0):")
        print(ttest_1samp(a=init_vals, popmean=0, alternative="less"))
        print("T-test (end vals > 0):")
        print(ttest_1samp(a=end_vals, popmean=0, alternative="greater"))

        plt.title("Correlation similarity (eq. firing):\n"
                  +cells_to_use+" cells\nT-test, p="+str(np.round(ttest_ind(init_vals, end_vals)[1], 3)))
        plt.scatter(np.zeros(7), init_vals)
        plt.scatter(np.ones(7), end_vals)
        plt.xticks([0,1], ["start", "end"])
        plt.xlim(-1,2)
        plt.ylabel("Sim. ratio correlations")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_correlations_stable_equalized_firing_"+cells_to_use+"_stats.svg"),
                        transparent="True")
        else:
            plt.show()

    def pre_long_sleep_drift_correlation_structure_equalized_firing_rates_all_cells(self, save_fig=False, n_smoothing=40,
                                                                          plot_mean=False, n_equalizing=10):
        """
        Looks at drift of correlation structure comparing sleep correlation matrices with PRE and POST. Firing rates
        of cells are equalized to exclude influence of single cell firing rate changes

        :param save_fig: as .svg
        :type save_fig: bool
        :param n_smoothing: how much smoothing to apply to result
        :type n_smoothing: int
        :param plot_mean: if mean is supposed to be plotted
        :type plot_mean: bool
        :param cells_to_use: which cells to use ("stable", "decreasing", "increasing")
        :type cells_to_use:
        """
        sim_ratio = []
        for session in self.session_list:
            s_r = \
                session.pre_long_sleep_post().drift_correlation_structure_equalized_firing_rates_all_cells(
                    plotting=False, n_equalizing=n_equalizing, n_smoothing=1)
            sim_ratio.append(s_r)

        if save_fig:
            plt.style.use('default')

        max_y = 0
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot()
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        first_val = []
        last_val = []
        sim_ratios_smooth = []
        sim_ratios_xaxis = []
        for i, session_sim_ratio in enumerate(sim_ratio):
            session_sim_ratio = session_sim_ratio[0]
            # smoothing
            session_sim_ratio = moving_average(a=session_sim_ratio, n=n_smoothing)
            sim_ratios_smooth.append(session_sim_ratio)
            sim_ratios_xaxis.append(np.linspace(0, 1, session_sim_ratio.shape[0]))
            first_val.append(session_sim_ratio[0])
            last_val.append(session_sim_ratio[-1])
            max_y = np.max(np.append(np.abs(session_sim_ratio), max_y))
            ax.plot(np.linspace(0, 1, session_sim_ratio.shape[0]),session_sim_ratio, linewidth=2, c=col[i], label=str(i))

        if plot_mean:
            nr_x_ticks = [x.shape[0] for x in sim_ratios_xaxis]
            x_axis_new = sim_ratios_xaxis[np.argmax(nr_x_ticks)]
            sim_ratios_smooth_equal_length = []
            for session_sim_ratio_smooth, x_axis in zip(sim_ratios_smooth, sim_ratios_xaxis):
                sim_ratios_smooth_equal_length.append(np.interp(x=x_axis_new, xp=x_axis, fp=session_sim_ratio_smooth))
            sim_ratios_smooth_equal_length = np.vstack(sim_ratios_smooth_equal_length)
            mean_sim_ratios_smooth = np.mean(sim_ratios_smooth_equal_length, axis=0)

            ax.plot(np.linspace(0, 1, mean_sim_ratios_smooth.shape[0]), mean_sim_ratios_smooth, linewidth=2, c="gray",
                    label="mean")
        plt.legend()
        # plt.ylim(-max_y-0.1, max_y+0.1)
        plt.ylim(-1,1)
        # plt.xlim(-0.05,1.05)
        plt.ylabel("sim_ratio correlations")
        plt.xlabel("Normalized sleep duration")
        plt.grid(axis='y', color="gray")
        plt.yticks([-1,-0.5, 0, 0.5,1], ["-1", "-0.5", "0", "0.5", "-1"])
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_correlations_stable_equalized_firing_all_cells.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        # check if drift is significant
        init_vals = []
        end_vals = []

        for sess_data in sim_ratio:
            init_vals.append(sess_data[0])
            end_vals.append(sess_data[-1])
            # init_vals.append(np.mean(sess_data[:10]))
            # end_vals.append(np.mean(sess_data[-10:]))
        init_vals = np.hstack(init_vals)
        end_vals = np.hstack(end_vals)
        print("T-test (init vals vs. end vals):")
        print(ttest_ind(init_vals, end_vals))
        print("T-test (init vals < 0):")
        print(ttest_1samp(a=init_vals, popmean=0, alternative="less"))
        print("T-test (end vals > 0):")
        print(ttest_1samp(a=end_vals, popmean=0, alternative="greater"))

        plt.title("Correlation similarity (eq. firing):all cells\nT-test, p="+str(np.round(ttest_ind(init_vals, end_vals)[1], 3)))
        plt.scatter(np.zeros(7), init_vals)
        plt.scatter(np.ones(7), end_vals)
        plt.xticks([0,1], ["start", "end"])
        plt.xlim(-1,2)
        plt.ylabel("Sim. ratio correlations")
        plt.show()

    def pre_long_sleep_drift_correlation_structure_equalized_firing_rates_only_sleep(self, save_fig=False, n_equalizing=1,
                                                                                    n_smoothing=20, plot_mean=False,
                                                                                     cells_to_use="stable"):
        """
        Looks at drift of correlation structure comparing sleep correlation matrices with PRE and POST. Firing rates
        of cells are equalized to exclude influence of single cell firing rate changes

        :param save_fig: as .svg
        :type save_fig: bool
        :param n_smoothing: how much smoothing to apply to result
        :type n_smoothing: int
        :param plot_mean: if mean is supposed to be plotted
        :type plot_mean: bool
        :param cells_to_use: which cells to use ("stable", "decreasing", "increasing")
        :type cells_to_use:
        """
        sim_ratio_mean = []
        sim_ratio_std = []
        for session in self.session_list:
            s_m, s_d = \
                session.pre_long_sleep_post().drift_correlation_structure_equalized_firing_rates_only_sleep(plotting=False,
                                                                                                           n_smoothing=n_smoothing,
                                                                                                           n_equalizing=n_equalizing,
                                                                                                            cells_to_use=cells_to_use)
            sim_ratio_mean.append(s_m)
            sim_ratio_std.append(s_d)

        if save_fig:
            plt.style.use('default')

        max_y = 0
        plt.figure(figsize=(4,6))
        # plt.figure(figsize=(3,4))
        fig = plt.figure()
        ax = fig.add_subplot()
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        init_vals = []
        end_vals = []
        sim_ratios_smooth = []
        sim_ratios_xaxis = []
        for i, (session_sim_ratio_mean, session_sim_ratio_std) in enumerate(zip(sim_ratio_mean, sim_ratio_std)):
            sim_ratios_xaxis.append(np.linspace(0, 1, session_sim_ratio_mean.shape[1]))
            init_vals.append(session_sim_ratio_mean.flatten()[0])
            end_vals.append(session_sim_ratio_mean.flatten()[-1])
            max_y = np.max(np.append(np.abs(session_sim_ratio_mean), max_y))
            if n_equalizing > 1:
                ax.errorbar(x=np.linspace(0, 1, session_sim_ratio_mean.shape[1]),y=session_sim_ratio_mean,
                            yerr=session_sim_ratio_std, linewidth=2, c=col[i], label=str(i))
            else:
                ax.plot(np.linspace(0, 1, session_sim_ratio_mean.shape[1]), session_sim_ratio_mean.flatten(),
                        linewidth=2, c=col[i], label=str(i))

        if plot_mean:
            nr_x_ticks = [x.shape[0] for x in sim_ratios_xaxis]
            x_axis_new = sim_ratios_xaxis[np.argmax(nr_x_ticks)]
            sim_ratios_smooth_equal_length = []
            for session_sim_ratio_smooth, x_axis in zip(sim_ratios_smooth, sim_ratios_xaxis):
                sim_ratios_smooth_equal_length.append(np.interp(x=x_axis_new, xp=x_axis, fp=session_sim_ratio_smooth))
            sim_ratios_smooth_equal_length = np.vstack(sim_ratios_smooth_equal_length)
            mean_sim_ratios_smooth = np.mean(sim_ratios_smooth_equal_length, axis=0)

            ax.plot(np.linspace(0, 1, mean_sim_ratios_smooth.shape[0]), mean_sim_ratios_smooth, linewidth=2, c="gray",
                    label="mean")
        plt.legend()
        # plt.ylim(-max_y-0.1, max_y+0.1)
        plt.ylim(-1,1)
        plt.xlim(0,1)
        plt.ylabel("Pearson R with first part of sleep")
        plt.xlabel("Normalized sleep duration")
        plt.grid(axis='y', color="gray")
        plt.yticks([-1,-0.5, 0, 0.5,1], ["-1", "-0.5", "0", "0.5", "-1"])
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_correlations_stable_equalized_firing_"+cells_to_use+"_only_sleep.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        # # check if drift is significant
        # init_vals = []
        # end_vals = []
        #
        # for sess_data in sim_ratios_smooth:
        #     init_vals.append(sess_data[0])
        #     end_vals.append(sess_data[-1])
        #     # init_vals.append(np.mean(sess_data[:10]))
        #     # end_vals.append(np.mean(sess_data[-10:]))

        print("T-test (init vals vs. end vals):")
        print(ttest_ind(init_vals, end_vals))
        print("T-test (init vals < 0):")
        print(ttest_1samp(a=init_vals, popmean=0, alternative="less"))
        print("T-test (end vals > 0):")
        print(ttest_1samp(a=end_vals, popmean=0, alternative="greater"))

        plt.title("Correlation similarity (eq. firing):\n"
                  +"all cells\nT-test, p="+str(np.round(ttest_ind(init_vals, end_vals)[1], 3)))
        plt.scatter(np.zeros(7), init_vals)
        plt.scatter(np.ones(7), end_vals)
        plt.xticks([0,1], ["start", "end"])
        plt.xlim(-1,2)
        plt.ylabel("Pearson R with first part of sleep")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_correlations_stable_equalized_firing_"+cells_to_use+"_only_sleep_stats.svg"),
                        transparent="True")
        else:
            plt.show()

    def pre_long_sleep_drift_correlation_structure_all_cells(self, save_fig=False, n_equalizing=1,
                                                                                    n_smoothing=20, plot_mean=False):
        """
        Looks at drift of correlation structure comparing sleep correlation matrices with PRE and POST. Firing rates
        of cells are equalized to exclude influence of single cell firing rate changes

        :param save_fig: as .svg
        :type save_fig: bool
        :param n_smoothing: how much smoothing to apply to result
        :type n_smoothing: int
        :param plot_mean: if mean is supposed to be plotted
        :type plot_mean: bool
        :param cells_to_use: which cells to use ("stable", "decreasing", "increasing")
        :type cells_to_use:
        """
        sim_ratio_mean = []
        sim_ratio_std = []
        for session in self.session_list:
            s_m, s_d = \
                session.pre_long_sleep_post().drift_correlation_structure_all_cells(plotting=False, 
                                                                                    n_smoothing=n_smoothing)
            sim_ratio_mean.append(s_m)
            sim_ratio_std.append(s_d)

        if save_fig:
            plt.style.use('default')

        max_y = 0
        plt.figure(figsize=(4,6))
        # plt.figure(figsize=(3,4))
        fig = plt.figure()
        ax = fig.add_subplot()
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        init_vals = []
        end_vals = []
        sim_ratios_smooth = []
        sim_ratios_xaxis = []
        for i, (session_sim_ratio_mean, session_sim_ratio_std) in enumerate(zip(sim_ratio_mean, sim_ratio_std)):
            sim_ratios_xaxis.append(np.linspace(0, 1, session_sim_ratio_mean.shape[1]))
            init_vals.append(session_sim_ratio_mean.flatten()[0])
            end_vals.append(session_sim_ratio_mean.flatten()[-1])
            max_y = np.max(np.append(np.abs(session_sim_ratio_mean), max_y))
            if n_equalizing > 1:
                ax.errorbar(x=np.linspace(0, 1, session_sim_ratio_mean.shape[1]),y=session_sim_ratio_mean,
                            yerr=session_sim_ratio_std, linewidth=2, c=col[i], label=str(i))
            else:
                ax.plot(np.linspace(0, 1, session_sim_ratio_mean.shape[1]), session_sim_ratio_mean.flatten(),
                        linewidth=2, c=col[i], label=str(i))

        if plot_mean:
            nr_x_ticks = [x.shape[0] for x in sim_ratios_xaxis]
            x_axis_new = sim_ratios_xaxis[np.argmax(nr_x_ticks)]
            sim_ratios_smooth_equal_length = []
            for session_sim_ratio_smooth, x_axis in zip(sim_ratios_smooth, sim_ratios_xaxis):
                sim_ratios_smooth_equal_length.append(np.interp(x=x_axis_new, xp=x_axis, fp=session_sim_ratio_smooth))
            sim_ratios_smooth_equal_length = np.vstack(sim_ratios_smooth_equal_length)
            mean_sim_ratios_smooth = np.mean(sim_ratios_smooth_equal_length, axis=0)

            ax.plot(np.linspace(0, 1, mean_sim_ratios_smooth.shape[0]), mean_sim_ratios_smooth, linewidth=2, c="gray",
                    label="mean")
        plt.legend()
        # plt.ylim(-max_y-0.1, max_y+0.1)
        plt.ylim(-1,1)
        plt.xlim(0,1)
        plt.ylabel("sim_ratio correlations")
        plt.xlabel("Normalized sleep duration")
        plt.grid(axis='y', color="gray")
        plt.yticks([-1,-0.5, 0, 0.5,1], ["-1", "-0.5", "0", "0.5", "-1"])
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_correlations_stable_equalized_firing_all_cells.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        # # check if drift is significant
        # init_vals = []
        # end_vals = []
        #
        # for sess_data in sim_ratios_smooth:
        #     init_vals.append(sess_data[0])
        #     end_vals.append(sess_data[-1])
        #     # init_vals.append(np.mean(sess_data[:10]))
        #     # end_vals.append(np.mean(sess_data[-10:]))

        print("T-test (init vals vs. end vals):")
        print(ttest_ind(init_vals, end_vals))
        print("T-test (init vals < 0):")
        print(ttest_1samp(a=init_vals, popmean=0, alternative="less"))
        print("T-test (end vals > 0):")
        print(ttest_1samp(a=end_vals, popmean=0, alternative="greater"))

        plt.title("Correlation similarity:\n"
                  +"all cells\nT-test, p="+str(np.round(ttest_ind(init_vals, end_vals)[1], 3)))
        plt.scatter(np.zeros(len(self.session_list)), init_vals)
        plt.scatter(np.ones(len(self.session_list)), end_vals)
        plt.xticks([0,1], ["start", "end"])
        plt.xlim(-1,2)
        plt.ylabel("Sim. ratio correlations")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_correlations_stable_equalized_firing_all_cells_stats.svg"),
                        transparent="True")
        else:
            plt.show()

    def pre_long_sleep_drift_correlation_structure_equalized_firing_rates_all_cells_only_sleep(self, save_fig=False, n_equalizing=1,
                                                                                    n_smoothing=20, plot_mean=False):
        """
        Looks at drift of correlation structure comparing sleep correlation matrices with PRE and POST. Firing rates
        of cells are equalized to exclude influence of single cell firing rate changes

        :param save_fig: as .svg
        :type save_fig: bool
        :param n_smoothing: how much smoothing to apply to result
        :type n_smoothing: int
        :param plot_mean: if mean is supposed to be plotted
        :type plot_mean: bool
        :param cells_to_use: which cells to use ("stable", "decreasing", "increasing")
        :type cells_to_use:
        """
        sim_ratio_mean = []
        sim_ratio_std = []
        for session in self.session_list:
            s_m, s_d = \
                session.pre_long_sleep_post().drift_correlation_structure_equalized_firing_rates_all_cells_only_sleep(plotting=False,
                                                                                                           n_smoothing=n_smoothing,
                                                                                                           n_equalizing=n_equalizing)
            sim_ratio_mean.append(s_m)
            sim_ratio_std.append(s_d)

        if save_fig:
            plt.style.use('default')

        max_y = 0
        plt.figure(figsize=(4,6))
        # plt.figure(figsize=(3,4))
        fig = plt.figure()
        ax = fig.add_subplot()
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        init_vals = []
        end_vals = []
        sim_ratios_smooth = []
        sim_ratios_xaxis = []
        for i, (session_sim_ratio_mean, session_sim_ratio_std) in enumerate(zip(sim_ratio_mean, sim_ratio_std)):
            sim_ratios_xaxis.append(np.linspace(0, 1, session_sim_ratio_mean.shape[1]))
            init_vals.append(session_sim_ratio_mean.flatten()[0])
            end_vals.append(session_sim_ratio_mean.flatten()[-1])
            max_y = np.max(np.append(np.abs(session_sim_ratio_mean), max_y))
            if n_equalizing > 1:
                ax.errorbar(x=np.linspace(0, 1, session_sim_ratio_mean.shape[1]),y=session_sim_ratio_mean,
                            yerr=session_sim_ratio_std, linewidth=2, c=col[i], label=str(i))
            else:
                ax.plot(np.linspace(0, 1, session_sim_ratio_mean.shape[1]), session_sim_ratio_mean.flatten(),
                        linewidth=2, c=col[i], label=str(i))

        if plot_mean:
            nr_x_ticks = [x.shape[0] for x in sim_ratios_xaxis]
            x_axis_new = sim_ratios_xaxis[np.argmax(nr_x_ticks)]
            sim_ratios_smooth_equal_length = []
            for session_sim_ratio_smooth, x_axis in zip(sim_ratios_smooth, sim_ratios_xaxis):
                sim_ratios_smooth_equal_length.append(np.interp(x=x_axis_new, xp=x_axis, fp=session_sim_ratio_smooth))
            sim_ratios_smooth_equal_length = np.vstack(sim_ratios_smooth_equal_length)
            mean_sim_ratios_smooth = np.mean(sim_ratios_smooth_equal_length, axis=0)

            ax.plot(np.linspace(0, 1, mean_sim_ratios_smooth.shape[0]), mean_sim_ratios_smooth, linewidth=2, c="gray",
                    label="mean")
        plt.legend()
        # plt.ylim(-max_y-0.1, max_y+0.1)
        plt.ylim(-1,1)
        plt.xlim(0,1)
        plt.ylabel("Pearson R with first part of sleep")
        plt.xlabel("Normalized sleep duration")
        plt.grid(axis='y', color="gray")
        plt.yticks([-1,-0.5, 0, 0.5,1], ["-1", "-0.5", "0", "0.5", "-1"])
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_correlations_stable_equalized_firing_all_cells_only_sleep.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        # # check if drift is significant
        # init_vals = []
        # end_vals = []
        #
        # for sess_data in sim_ratios_smooth:
        #     init_vals.append(sess_data[0])
        #     end_vals.append(sess_data[-1])
        #     # init_vals.append(np.mean(sess_data[:10]))
        #     # end_vals.append(np.mean(sess_data[-10:]))

        print("T-test (init vals vs. end vals):")
        print(ttest_ind(init_vals, end_vals))
        print("T-test (init vals < 0):")
        print(ttest_1samp(a=init_vals, popmean=0, alternative="less"))
        print("T-test (end vals > 0):")
        print(ttest_1samp(a=end_vals, popmean=0, alternative="greater"))

        plt.title("Correlation similarity (eq. firing):\n"
                  +"all cells\nT-test, p="+str(np.round(ttest_ind(init_vals, end_vals)[1], 3)))
        plt.scatter(np.zeros(7), init_vals)
        plt.scatter(np.ones(7), end_vals)
        plt.xticks([0,1], ["start", "end"])
        plt.xlim(-1,2)
        plt.ylabel("Pearson R with first part of sleep")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_correlations_stable_equalized_firing_all_cells_only_sleep_stats.svg"),
                        transparent="True")
        else:
            plt.show()

    def pre_long_sleep_drift_correlation_structure_all_cells_only_sleep(self, save_fig=False, n_smoothing=20,
                                                                        plot_mean=False):
        """
        Looks at drift of correlation structure comparing sleep correlation matrices with PRE and POST. Firing rates
        of cells are equalized to exclude influence of single cell firing rate changes

        :param save_fig: as .svg
        :type save_fig: bool
        :param n_smoothing: how much smoothing to apply to result
        :type n_smoothing: int
        :param plot_mean: if mean is supposed to be plotted
        :type plot_mean: bool
        :param cells_to_use: which cells to use ("stable", "decreasing", "increasing")
        :type cells_to_use:
        """
        sim_ratio_mean = []
        sim_ratio_std = []
        for session in self.session_list:
            s_m, s_d = \
                session.pre_long_sleep_post().drift_correlation_structure_all_cells_only_sleep(plotting=False,
                                                                                               n_smoothing=n_smoothing)
            sim_ratio_mean.append(s_m)
            sim_ratio_std.append(s_d)

        if save_fig:
            plt.style.use('default')

        max_y = 0
        plt.figure(figsize=(4,6))
        # plt.figure(figsize=(3,4))
        fig = plt.figure()
        ax = fig.add_subplot()
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        init_vals = []
        end_vals = []
        sim_ratios_smooth = []
        sim_ratios_xaxis = []
        for i, (session_sim_ratio_mean, session_sim_ratio_std) in enumerate(zip(sim_ratio_mean, sim_ratio_std)):
            sim_ratios_xaxis.append(np.linspace(0, 1, session_sim_ratio_mean.shape[1]))
            init_vals.append(session_sim_ratio_mean.flatten()[0])
            end_vals.append(session_sim_ratio_mean.flatten()[-1])
            max_y = np.max(np.append(np.abs(session_sim_ratio_mean), max_y))

            ax.plot(np.linspace(0, 1, session_sim_ratio_mean.shape[1]), session_sim_ratio_mean.flatten(),
                    linewidth=2, c=col[i], label=str(i))

        if plot_mean:
            nr_x_ticks = [x.shape[0] for x in sim_ratios_xaxis]
            x_axis_new = sim_ratios_xaxis[np.argmax(nr_x_ticks)]
            sim_ratios_smooth_equal_length = []
            for session_sim_ratio_smooth, x_axis in zip(sim_ratios_smooth, sim_ratios_xaxis):
                sim_ratios_smooth_equal_length.append(np.interp(x=x_axis_new, xp=x_axis, fp=session_sim_ratio_smooth))
            sim_ratios_smooth_equal_length = np.vstack(sim_ratios_smooth_equal_length)
            mean_sim_ratios_smooth = np.mean(sim_ratios_smooth_equal_length, axis=0)

            ax.plot(np.linspace(0, 1, mean_sim_ratios_smooth.shape[0]), mean_sim_ratios_smooth, linewidth=2, c="gray",
                    label="mean")
        plt.legend()
        # plt.ylim(-max_y-0.1, max_y+0.1)
        plt.ylim(-1,1)
        plt.xlim(0,1)
        plt.ylabel("Pearson R with first part of sleep")
        plt.xlabel("Normalized sleep duration")
        plt.grid(axis='y', color="gray")
        plt.yticks([-1,-0.5, 0, 0.5,1], ["-1", "-0.5", "0", "0.5", "-1"])
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_correlations_stable_equalized_firing_all_cells_only_sleep.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        # # check if drift is significant
        # init_vals = []
        # end_vals = []
        #
        # for sess_data in sim_ratios_smooth:
        #     init_vals.append(sess_data[0])
        #     end_vals.append(sess_data[-1])
        #     # init_vals.append(np.mean(sess_data[:10]))
        #     # end_vals.append(np.mean(sess_data[-10:]))

        print("T-test (init vals vs. end vals):")
        print(ttest_ind(init_vals, end_vals))
        print("T-test (init vals < 0):")
        print(ttest_1samp(a=init_vals, popmean=0, alternative="less"))
        print("T-test (end vals > 0):")
        print(ttest_1samp(a=end_vals, popmean=0, alternative="greater"))

        plt.title("Correlation similarity (eq. firing):\n"
                  +"all cells\nT-test, p="+str(np.round(ttest_ind(init_vals, end_vals)[1], 3)))
        plt.scatter(np.zeros(7), init_vals)
        plt.scatter(np.ones(7), end_vals)
        plt.xticks([0,1], ["start", "end"])
        plt.xlim(-1,2)
        plt.ylabel("Pearson R with first part of sleep")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_correlations_stable_equalized_firing_all_cells_only_sleep_stats.svg"),
                        transparent="True")
        else:
            plt.show()

    def pre_long_sleep_drift_correlation_structure_all_cells_only_sleep_equalized_vs_non_equalized(self, save_fig=False,
                                                                                                   n_smoothing=80,
                                                                                                   plot_mean=False,
                                                                                                   n_equalizing=10):

        sim_ratio_mean = []
        sim_ratio_std = []
        sim_ratio_mean_eq = []
        sim_ratio_std_eq = []
        for session in self.session_list:
            s_m, s_d = \
                session.pre_long_sleep_post().drift_correlation_structure_all_cells_only_sleep(plotting=False,
                                                                                               n_smoothing=n_smoothing)
            sim_ratio_mean.append(s_m)
            sim_ratio_std.append(s_d)
            s_m_eq, s_d_eq = \
                session.pre_long_sleep_post().drift_correlation_structure_equalized_firing_rates_all_cells_only_sleep(
                    plotting=False, n_smoothing=n_smoothing, n_equalizing=n_equalizing)
            sim_ratio_mean_eq.append(s_m_eq)
            sim_ratio_std_eq.append(s_d_eq)


        plt.style.use('default')

        max_y = 0
        plt.figure(figsize=(4,6))
        # plt.figure(figsize=(3,4))
        fig = plt.figure()
        ax = fig.add_subplot()
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        cmap = matplotlib.cm.get_cmap('Greys')

        col_eq = [cmap(x) for x in [0.15, 0.3, 0.45, 0.6, 0.75, 0.86, 0.99]]
        init_vals = []
        end_vals = []
        sim_ratios_smooth = []
        sim_ratios_xaxis = []
        for i, (session_sim_ratio_mean, session_sim_ratio_std) in enumerate(zip(sim_ratio_mean, sim_ratio_std)):
            sim_ratios_xaxis.append(np.linspace(0, 1, session_sim_ratio_mean.shape[1]))
            init_vals.append(np.mean(session_sim_ratio_mean.flatten()[:10]))
            end_vals.append(np.mean(session_sim_ratio_mean.flatten()[-10:]))
            max_y = np.max(np.append(np.abs(session_sim_ratio_mean), max_y))
            ax.plot(np.linspace(0, 1, session_sim_ratio_mean.shape[1]), session_sim_ratio_mean.flatten(),
                    linewidth=2, c=col[i], label=str(i))
        init_vals_eq = []
        end_vals_eq = []
        for i, (session_sim_ratio_mean_eq, session_sim_ratio_std_eq) in enumerate(zip(sim_ratio_mean_eq, sim_ratio_std_eq)):
            ax.plot(np.linspace(0, 1, session_sim_ratio_mean_eq.shape[0]), session_sim_ratio_mean_eq.flatten(),
                    linewidth=2, c=col_eq[i], label=str(i))
            init_vals_eq.append(np.mean(session_sim_ratio_mean_eq.flatten()[:10]))
            end_vals_eq.append(np.mean(session_sim_ratio_mean_eq.flatten()[-10:]))

        init_vals_eq = np.hstack(init_vals_eq)
        init_vals = np.hstack(init_vals)
        end_vals_eq = np.hstack(end_vals_eq)
        end_vals = np.hstack(end_vals)

        diff = init_vals - end_vals
        diff_eq = init_vals_eq - end_vals_eq

        if plot_mean:
            nr_x_ticks = [x.shape[0] for x in sim_ratios_xaxis]
            x_axis_new = sim_ratios_xaxis[np.argmax(nr_x_ticks)]
            sim_ratios_smooth_equal_length = []
            for session_sim_ratio_smooth, x_axis in zip(sim_ratios_smooth, sim_ratios_xaxis):
                sim_ratios_smooth_equal_length.append(np.interp(x=x_axis_new, xp=x_axis, fp=session_sim_ratio_smooth))
            sim_ratios_smooth_equal_length = np.vstack(sim_ratios_smooth_equal_length)
            mean_sim_ratios_smooth = np.mean(sim_ratios_smooth_equal_length, axis=0)

            ax.plot(np.linspace(0, 1, mean_sim_ratios_smooth.shape[0]), mean_sim_ratios_smooth, linewidth=2, c="gray",
                    label="mean")
        plt.legend(ncol=3)
        # plt.ylim(-max_y-0.1, max_y+0.1)
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.ylabel("Pearson R with first part of sleep")
        plt.xlabel("Normalized sleep duration")
        plt.grid(axis='y', color="gray")
        plt.yticks([0, 0.5,1])
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_correlations_stable_equalized_firing_all_cells_only_sleep.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        c = "black"
        bplot = plt.boxplot([diff, diff_eq], positions=[0,1], patch_artist=True,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            labels=["non-equalized", "equalized"],
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.ylabel("R_beginning - R_end")
        plt.show()
        print("EQ vs. NON-EQ:")
        print(mannwhitneyu(diff, diff_eq, alternative="greater"))

        # # check if drift is significant
        # init_vals = []
        # end_vals = []
        #
        # for sess_data in sim_ratios_smooth:
        #     init_vals.append(sess_data[0])
        #     end_vals.append(sess_data[-1])
        #     # init_vals.append(np.mean(sess_data[:10]))
        #     # end_vals.append(np.mean(sess_data[-10:]))

        print("T-test (init vals vs. end vals):")
        print(ttest_ind(init_vals, end_vals))
        print("T-test (init vals < 0):")
        print(ttest_1samp(a=init_vals, popmean=0, alternative="less"))
        print("T-test (end vals > 0):")
        print(ttest_1samp(a=end_vals, popmean=0, alternative="greater"))

        plt.title("Correlation similarity (eq. firing):\n"
                  +"all cells\nT-test, p="+str(np.round(ttest_ind(init_vals, end_vals)[1], 3)))
        plt.scatter(np.zeros(7), init_vals)
        plt.scatter(np.ones(7), end_vals)
        plt.xticks([0,1], ["start", "end"])
        plt.xlim(-1,2)
        plt.ylabel("Pearson R with first part of sleep")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_correlations_stable_equalized_firing_all_cells_only_sleep_stats.svg"),
                        transparent="True")
        else:
            plt.show()

    def pre_long_sleep_nrem_rem_autocorrelation_spikes_likelihood_vectors(self, template_type="phmm", save_fig=False,
                                                                          nr_pop_vecs=5, exclude_zero_shift=True):
        """
        Auto-correlaion of likelihood vectors (result of sleep decoding)

        :param template_type: "phmm" or "ising"
        :type template_type: str
        :param save_fig: save as .svg
        :type save_fig: bool
        :param nr_pop_vecs: how many population vectors to shift left and right (e.g. 5 --> -5 to 5)
        :type nr_pop_vecs: int
        :param exclude_zero_shift: whether to plot zero shift value
        :type exclude_zero_shift: bool
        """
        # sleep data first
        # --------------------------------------------------------------------------------------------------------------

        rem = []
        nrem = []
        rem_exp = []
        nrem_exp = []

        for i, session in enumerate(self.session_list):
            re, nre, re_exp, nre_exp = \
                session.long_sleep().memory_drift_rem_nrem_autocorrelation_spikes_likelihood_vectors(plotting=False,
                                                                                                     template_type=template_type,
                                                                                                     nr_pop_vecs=nr_pop_vecs)
            rem_exp.append(re_exp)
            nrem_exp.append(nre_exp)
            rem.append(re)
            nrem.append(nre)

        rem = np.array(rem)
        nrem = np.array(nrem)
        rem_exp = np.array(rem_exp)
        nrem_exp = np.array(nrem_exp)

        rem_mean = np.mean(rem, axis=0)
        nrem_mean = np.mean(nrem, axis=0)
        rem_std = np.std(rem, axis=0)
        nrem_std = np.std(nrem, axis=0)

        # awake data
        # --------------------------------------------------------------------------------------------------------------

        awake = []
        awake_exp = []

        for i, session in enumerate(self.session_list):
            _, dat, a_exp = \
                session.cheese_board(experiment_phase=
                                     ["learning_cheeseboard_1"]).decode_awake_activity_autocorrelation_spikes_likelihood_vectors(plotting=False, nr_pop_vecs=nr_pop_vecs)

            awake.append(dat)
            awake_exp.append(a_exp)

        awake = np.array(awake)
        awake_exp = np.array(awake_exp)
        awake_mean = np.mean(awake, axis=0)
        awake_std = np.std(awake, axis=0)

        # plotting
        # --------------------------------------------------------------------------------------------------------------

        shift_array = np.arange(-1*int(nr_pop_vecs), int(nr_pop_vecs)+1)

        if exclude_zero_shift:
            rem_mean[int(rem_mean.shape[0] / 2)] = np.nan
            nrem_mean[int(nrem_mean.shape[0] / 2)] = np.nan
            awake_mean[int(awake_mean.shape[0] / 2)] = np.nan

        if save_fig:
            plt.style.use('default')
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.fill_between(shift_array*self.params.spikes_per_bin, rem_mean+rem_std ,rem_mean-rem_std, facecolor="salmon")
        ax.plot(shift_array*self.params.spikes_per_bin, rem_mean, c="red", label="REM")
        ax.fill_between(shift_array*self.params.spikes_per_bin, nrem_mean+nrem_std ,nrem_mean-nrem_std, facecolor="skyblue", alpha=0.6)
        ax.plot(shift_array*self.params.spikes_per_bin, nrem_mean, c="blue", label="NREM")

        ax.fill_between(shift_array*self.params.spikes_per_bin, awake_mean+awake_std ,awake_mean-awake_std, facecolor="lemonchiffon", alpha=0.7)
        ax.plot(shift_array*self.params.spikes_per_bin, awake_mean, c="yellow", label="Awake")
        plt.xlabel("Shift (#spikes)")
        plt.ylabel("Avg. Pearson correlation of likelihood vectors")
        plt.legend()

        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            if exclude_zero_shift:
                plt.savefig(os.path.join(save_path, "auto_corr_likelihood_vectors_wo_zero.svg"), transparent="True")
            else:
                plt.savefig(os.path.join(save_path, "auto_corr_likelihood_vectors.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        y_dat = np.vstack((nrem_exp, rem_exp, awake_exp))

        plt.figure(figsize=(2,3))
        bplot = plt.boxplot(y_dat.T, positions=[1, 2, 3], patch_artist=True,
                            labels=["NREM", "REM", "AWAKE"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),
                            )
        colors = ["blue", 'red', "yellow"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Exponential coefficient k")
        plt.grid(color="grey", axis="y")
        degrees = 45
        plt.xticks(rotation=degrees)
        # plt.yscale("symlog")
        plt.yticks([np.median(rem_exp), 0.1, -1, np.median(nrem_exp), np.median(awake_exp)])
        # plt.text(-0.001, np.median(rem_exp), np.str(np.round(np.median(rem_exp), 2)))
        # plt.text(-0.001, np.median(nrem_exp), np.str(np.round(np.median(nrem_exp), 2)))
        # plt.text(-0.001, np.median(awake_exp), np.str(np.round(np.median(awake_exp), 2)))
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "exponential_coeff_likelihood_vec.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        print("REM vs. NREM")
        print(mannwhitneyu(rem_exp, nrem_exp))
        print("REM vs. AWAKE")
        print(mannwhitneyu(rem_exp, awake_exp))
        print("NREM vs. AWAKE")
        print(mannwhitneyu(nrem_exp, awake_exp))

    def pre_long_sleep_sparsity_and_distance_of_modes(self):

        sparsity_pre = []
        sparsity_post = []
        sparsity_reactivations_rem_pre = []
        sparsity_reactivations_nrem_pre = []
        sparsity_reactivations_rem_post = []
        sparsity_reactivations_nrem_post = []
        distance_pre = []
        distance_post = []
        distance_reactivations_rem_pre = []
        distance_reactivations_nrem_pre = []
        distance_reactivations_rem_post = []
        distance_reactivations_nrem_post = []

        for i, session in enumerate(self.session_list):
            sparsity_pre_, sparsity_post_, sparsity_reactivations_rem_pre_, sparsity_reactivations_nrem_pre_, \
                sparsity_reactivations_rem_post_, sparsity_reactivations_nrem_post_, distance_pre_, distance_post_, \
                distance_reactivations_rem_pre_, distance_reactivations_nrem_pre_, distance_reactivations_rem_post_, \
                distance_reactivations_nrem_post_ = \
                session.pre_long_sleep_post().sparsity_and_distance_of_modes(plotting=False)

            sparsity_pre.append(sparsity_pre_)
            sparsity_post.append(sparsity_post_)
            sparsity_reactivations_rem_pre.append(sparsity_reactivations_rem_pre_)
            sparsity_reactivations_nrem_pre.append(sparsity_reactivations_nrem_pre_)
            sparsity_reactivations_rem_post.append(sparsity_reactivations_rem_post_)
            sparsity_reactivations_nrem_post.append(sparsity_reactivations_nrem_post_)
            distance_pre.append(distance_pre_)
            distance_post.append(distance_post_)
            distance_reactivations_rem_pre.append(distance_reactivations_rem_pre_)
            distance_reactivations_nrem_pre.append(distance_reactivations_nrem_pre_)
            distance_reactivations_rem_post.append(distance_reactivations_rem_post_)
            distance_reactivations_nrem_post.append(distance_reactivations_nrem_post_)

        sparsity_pre = np.hstack(sparsity_pre)
        sparsity_post = np.hstack(sparsity_post)
        sparsity_reactivations_rem_pre = np.hstack(sparsity_reactivations_rem_pre)
        sparsity_reactivations_nrem_pre = np.hstack(sparsity_reactivations_nrem_pre)
        sparsity_reactivations_rem_post = np.hstack(sparsity_reactivations_rem_post)
        sparsity_reactivations_nrem_post = np.hstack(sparsity_reactivations_nrem_post)
        distance_pre = np.hstack(distance_pre)
        distance_post = np.hstack(distance_post)
        distance_reactivations_rem_pre = np.hstack(distance_reactivations_rem_pre)
        distance_reactivations_nrem_pre = np.hstack(distance_reactivations_nrem_pre)
        distance_reactivations_rem_post = np.hstack(distance_reactivations_rem_post)
        distance_reactivations_nrem_post = np.hstack(distance_reactivations_nrem_post)

        plt.figure(figsize=(12, 4))
        c = "white"
        res = [sparsity_pre, sparsity_reactivations_rem_pre, sparsity_reactivations_nrem_pre,
               sparsity_reactivations_rem_post, sparsity_reactivations_nrem_post, sparsity_post]
        bplot = plt.boxplot(res, positions=[1, 2, 3, 4, 5, 6], patch_artist=True,
                            labels=["PRE", "PRE_modes_REM", "PRE_modes_NREM", "POST_modes_REM",
                                    "POST_modes_NREM", "POST"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        y_min, y_max = plt.gca().get_ylim()
        y_span = y_max - y_min
        plt.ylabel("Sparsity")
        plt.ylim(y_min, y_min + 1.5 * y_span)
        plt.hlines(y_min + 1.4 * y_span, 1, 6)
        if mannwhitneyu(sparsity_pre, sparsity_post)[1] < 0.001:
            plt.text(3.5, y_min + 1.42 * y_span, "***")
        elif mannwhitneyu(sparsity_pre, sparsity_post)[1] < 0.01:
            plt.text(3.5, y_min + 1.42 * y_span, "**")
        elif mannwhitneyu(sparsity_pre, sparsity_post)[1] < 0.05:
            plt.text(3.5, y_min + 1.42 * y_span, "*")
        else:
            plt.text(3.5, y_min + 1.42 * y_span, "n.s.")
        plt.hlines(y_min + 1.3 * y_span, 1, 2)
        if mannwhitneyu(sparsity_pre, sparsity_reactivations_rem_pre)[1] < 0.001:
            plt.text(1.5, y_min + 1.32 * y_span, "***")
        elif mannwhitneyu(sparsity_pre, sparsity_reactivations_rem_pre)[1] < 0.01:
            plt.text(1.5, y_min + 1.32 * y_span, "**")
        elif mannwhitneyu(sparsity_pre, sparsity_reactivations_rem_pre)[1] < 0.05:
            plt.text(1.5, y_min + 1.32 * y_span, "*")
        else:
            plt.text(3.5, y_min + 1.32 * y_span, "n.s.")

        plt.hlines(y_min + 1.2 * y_span, 1, 3)
        if mannwhitneyu(sparsity_pre, sparsity_reactivations_nrem_pre)[1] < 0.001:
            plt.text(2, y_min + 1.22 * y_span, "***")
        elif mannwhitneyu(sparsity_pre, sparsity_reactivations_nrem_pre)[1] < 0.01:
            plt.text(2, y_min + 1.22 * y_span, "**")
        elif mannwhitneyu(sparsity_pre, sparsity_reactivations_nrem_pre)[1] < 0.05:
            plt.text(2, y_min + 1.22 * y_span, "*")
        else:
            plt.text(2, y_min + 1.22 * y_span, "n.s.")

        plt.hlines(y_min + 1.3 * y_span, 5, 6)
        if mannwhitneyu(sparsity_post, sparsity_reactivations_nrem_post)[1] < 0.001:
            plt.text(5.5, y_min + 1.32 * y_span, "***")
        elif mannwhitneyu(sparsity_post, sparsity_reactivations_nrem_post)[1] < 0.01:
            plt.text(5.5, y_min + 1.32 * y_span, "**")
        elif mannwhitneyu(sparsity_post, sparsity_reactivations_nrem_post)[1] < 0.05:
            plt.text(5.5, y_min + 1.32 * y_span, "*")
        else:
            plt.text(5.5, y_min + 1.32 * y_span, "n.s.")

        plt.hlines(y_min + 1.2 * y_span, 4, 6)
        if mannwhitneyu(sparsity_post, sparsity_reactivations_rem_post)[1] < 0.001:
            plt.text(5, y_min + 1.22 * y_span, "***")
        elif mannwhitneyu(sparsity_post, sparsity_reactivations_rem_post)[1] < 0.01:
            plt.text(5, y_min + 1.22 * y_span, "**")
        elif mannwhitneyu(sparsity_post, sparsity_reactivations_rem_post)[1] < 0.05:
            plt.text(5, y_min + 1.22 * y_span, "*")
        else:
            plt.text(5, y_min + 1.22 * y_span, "n.s.")
        plt.tight_layout()
        plt.grid(color="dimgrey", axis="y")
        plt.show()

        plt.figure(figsize=(12, 4))
        c = "white"
        res = [distance_pre, distance_reactivations_rem_pre, distance_reactivations_nrem_pre,
               distance_reactivations_rem_post, distance_reactivations_nrem_post, distance_post]
        bplot = plt.boxplot(res, positions=[1, 2, 3, 4, 5, 6], patch_artist=True,
                            labels=["PRE_modes", "PRE_modes_REM", "PRE_modes_NREM", "POST_modes_REM",
                                    "POST_modes_NREM", "POST_modes"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False, widths=0.2)
        plt.ylabel("Distance (cos)")
        plt.ylim(0, 2)
        plt.hlines(1.8, 1, 6)
        if mannwhitneyu(distance_pre, distance_post)[1] < 0.001:
            plt.text(3.5, 1.85, "***")
        elif mannwhitneyu(distance_pre, distance_post)[1] < 0.01:
            plt.text(3.5, 1.85, "**")
        elif mannwhitneyu(distance_pre, distance_post)[1] < 0.05:
            plt.text(3.5, 1.85, "*")
        else:
            plt.text(3.5, 1.85, "n.s.")
        plt.hlines(1.7, 1, 2)
        if mannwhitneyu(distance_pre, distance_reactivations_rem_pre)[1] < 0.001:
            plt.text(1.5, 1.71, "***")
        elif mannwhitneyu(distance_pre, distance_reactivations_rem_pre)[1] < 0.01:
            plt.text(1.5, 1.71, "**")
        elif mannwhitneyu(distance_pre, distance_reactivations_rem_pre)[1] < 0.05:
            plt.text(1.5, 1.71, "*")
        else:
            plt.text(3.5, 1.71, "n.s.")

        plt.hlines(1.6, 1, 3)
        if mannwhitneyu(distance_pre, distance_reactivations_nrem_pre)[1] < 0.001:
            plt.text(2, 1.6, "***")
        elif mannwhitneyu(distance_pre, distance_reactivations_nrem_pre)[1] < 0.01:
            plt.text(2, 1.61, "**")
        elif mannwhitneyu(distance_pre, distance_reactivations_nrem_pre)[1] < 0.05:
            plt.text(2, 1.61, "*")
        else:
            plt.text(2, 1.61, "n.s.")

        plt.hlines(1.7, 5, 6)
        if mannwhitneyu(distance_post, distance_reactivations_nrem_post)[1] < 0.001:
            plt.text(5.5, 1.71, "***")
        elif mannwhitneyu(distance_post, distance_reactivations_nrem_post)[1] < 0.01:
            plt.text(5.5, 1.71, "**")
        elif mannwhitneyu(distance_post, distance_reactivations_nrem_post)[1] < 0.05:
            plt.text(5.5, 1.71, "*")
        else:
            plt.text(5.5, 1.71, "n.s.")

        plt.hlines(1.6, 4, 6)
        if mannwhitneyu(distance_post, distance_reactivations_rem_post)[1] < 0.001:
            plt.text(5, 1.6, "***")
        elif mannwhitneyu(distance_post, distance_reactivations_rem_post)[1] < 0.01:
            plt.text(5, 1.61, "**")
        elif mannwhitneyu(distance_post, distance_reactivations_rem_post)[1] < 0.05:
            plt.text(5, 1.61, "*")
        else:
            plt.text(5, 1.61, "n.s.")

        plt.tight_layout()
        plt.grid(color="dimgrey", axis="y")
        plt.show()

    # </editor-fold>

    # <editor-fold desc="Long sleep">

    def long_sleep_sleep_stage_durations(self, save_fig=False):
        rem = []
        nrem = []

        for i, session in enumerate(self.session_list):
            r_, nr_ = session.long_sleep().sleep_stage_durations()
            rem.append(r_)
            nrem.append(nr_)

        dur_rem = np.hstack(rem)
        dur_nrem = np.hstack(nrem)

        # plt.hist(dur_nrem, density=True, color="blue", bins=50, label="NREM")
        # plt.hist(dur_rem, density=True, color="red", alpha=0.7, bins=50, label="REM")
        # plt.legend()
        # plt.xlabel("Duration (s)")
        # plt.ylabel("Density")
        # plt.xscale("log")
        # plt.show()
        if save_fig:
            plt.style.use('default')
        dur_rem_sorted = np.sort(dur_rem)
        p_dur_rem = 1. * np.arange(dur_rem_sorted.shape[0]) / (dur_rem_sorted.shape[0] - 1)
        dur_nrem_sorted = np.sort(dur_nrem)
        p_dur_nrem = 1. * np.arange(dur_nrem_sorted.shape[0]) / (dur_nrem_sorted.shape[0] - 1)
        plt.plot(dur_rem_sorted, p_dur_rem, color="red", label="REM")
        plt.plot(dur_nrem_sorted, p_dur_nrem, color="blue", label="NREM")
        plt.legend()
        plt.ylabel("CDF")
        plt.xlabel("Duration (s)")
        plt.xscale("log")
        plt.title("All sessions")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "rem_nrem_duration_all_sessions.svg"), transparent="True")
        else:
            plt.show()

    def long_sleep_memory_drift_short_time_scale_vs_long_time_scale(self, save_fig=False, smoothing=900,
                                                                    template_type="phmm", n_segments=10):

        delta_macro_drift = []
        delta_micro_drift = []
        sim_ratio = []
        for i, session in enumerate(self.session_list):
            print("GETTING DATA FROM " +session.session_name)
            res = session.long_sleep().memory_drift_plot_temporal_trend(template_type=template_type,
                                                                        n_moving_average_pop_vec=smoothing,
                                                                        plotting=False)
            delta_macro_drift.append(np.mean(res[-20:])-np.mean(res[:20]))
            delta_micro_drift.append(np.sum(np.abs(res)))
            sim_ratio.append(res)
            # ds_rem_cum_, ds_nrem_cum_, ds_rem_, ds_nrem_, _, _ = \
            #     session.long_sleep().memory_drift_rem_nrem_delta_score(template_type=template_type,
            #                                                            plotting=False,
            #                                                            n_moving_average_pop_vec=n_moving_average_pop_vec,
            #                                                            rem_pop_vec_threshold=rem_pop_vec_threshold)
            #
            # delta_micro_drift.append(np.sum(np.hstack((np.abs(ds_rem_),np.abs(ds_nrem_)))))

        delta_macro_drift = np.squeeze(np.vstack(delta_macro_drift))
        delta_micro_drift = np.hstack(delta_micro_drift)

        net_effect_seg = np.zeros((len(sim_ratio), n_segments))
        cum_effect_seg = np.zeros((len(sim_ratio), n_segments))
        # divide sleep into segments and check correlation between cumulative and net effect
        for sess_id, sr in enumerate(sim_ratio):
            seg_length = np.round(sr.shape[0]/n_segments).astype(int)
            for seg_id in range(n_segments):
                seg_data = sr[seg_id*seg_length:(seg_id+1)*seg_length]
                net_effect_seg[sess_id, seg_id] = np.abs(np.mean(seg_data[-20:])-np.mean(seg_data[:20]))
                cum_effect_seg[sess_id, seg_id] = np.sum(np.abs(seg_data))

        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(4,4))
        plt.scatter(zscore(net_effect_seg, axis=1).flatten(), zscore(cum_effect_seg, axis=1).flatten())
        plt.text(1, 0.7, "R="+str(np.round(pearsonr(zscore(net_effect_seg, axis=1).flatten(),
                                                  zscore(cum_effect_seg, axis=1).flatten())[0],4)))
        plt.text(1, 0.5, "p="+str(np.round(pearsonr(zscore(net_effect_seg, axis=1).flatten(),
                                                  zscore(cum_effect_seg, axis=1).flatten())[1], 2)))
        plt.xlabel("Net effect (z-scored)")
        plt.ylabel("Cum effect (z-scored)")
        plt.ylim(-1.75, 2.75)
        plt.xlim(-1.75, 2.75)
        plt.gca().set_aspect(1 / plt.gca().get_data_ratio())
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "correlation_cum_change_vs_net_change.svg"), transparent="True")
        else:
            plt.show()

        print(mannwhitneyu(delta_macro_drift, delta_micro_drift, alternative="less"))
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        y_dat = [delta_macro_drift, delta_micro_drift]
        plt.figure(figsize=(3, 5))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["Net effect", "Cumulative\n effect"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False,
                            widths=(0.6, 0.6))
        plt.ylabel("Delta sim_ratio")
        plt.yscale("log")
        plt.ylim(10e-2, 10e5)
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "delta_sim_ratio_timescales.svg"), transparent="True")
        else:
            plt.show()

    def long_sleep_memory_drift_temporal(self, save_fig=False, template_type="phmm", n_smoothing=20000):
        """
        plots memory drift for all sessions

        :param template_type: "phmm" or "ising"
        :type template_type: str
        :param smoothing: how much smoothing to use for ratio (default: 20000)
        :param smoothing: int
        :param save_fig: whether to save figure (True)
        :type save_fig: bool
        """
        data_t0 = []
        data_t_end = []

        if save_fig:
            plt.style.use('default')

        fig = plt.figure()
        ax = fig.add_subplot()
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        for i, session in enumerate(self.session_list):
            print("GETTING DATA FROM " +session.session_name)
            res = session.long_sleep().memory_drift_plot_temporal_trend(template_type=template_type,
                                                                        n_moving_average_pop_vec=0,
                                                                        plotting=False)
            data_t0.append(res[0])
            data_t_end.append(res[-1])
            print("... DONE")
            res_smooth = uniform_filter1d(res, n_smoothing)
            ax.plot(np.linspace(0, 1, res_smooth.shape[0]),res_smooth, linewidth=1, c=col[i])

        # stats
        p_value_t0 = ttest_1samp(data_t0, 0, alternative="less")[1]
        print("T-test for t=0, data < 0 --> p = "+str(p_value_t0))

        p_value_t_end = ttest_1samp(data_t_end, 0, alternative="greater")[1]
        print("T-test for t_end, data > 0 --> p = "+str(p_value_t_end))

        p_value_t_end_start = ttest_ind(data_t0, data_t_end, 0)[1]
        print("T-test for t_end vs t_start --> p = "+str(p_value_t_end_start))

        plt.grid(axis='y')
        plt.xlabel("Normalized duration")
        plt.xlim(-0.01, 1.01)
        # plt.ylim(-0.75, 0.25)
        plt.ylim(-1, 1)
        plt.yticks([-1,-0.5, 0, 0.5, 1], ["-1", "-0.5", "0", "0.5", "1"])
        plt.ylabel("sim_ratio")
        # plt.title("All sessions")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_all_sessions.svg"), transparent="True")
        else:
            plt.show()

    def long_sleep_memory_drift_compute_results(self, cells_to_use="all", template_type="ising"):
        """
        Computes results per session of drift vs. shuffle

        """

        for i, session in enumerate(self.session_list):
            session.long_sleep().memory_drift_compute_results(cells_to_use=cells_to_use, template_type=template_type)

    def long_sleep_memory_drift_data_vs_shuffle_compute_results(self):
        """
        Computes results per session of drift vs. shuffle

        """
        z_scored_init = []
        z_scored_end = []

        for i, session in enumerate(self.session_list):
            zsi, zse = session.long_sleep().memory_drift_entire_sleep_spike_shuffle_vs_data_compute_p_values()
            z_scored_init.append(zsi)
            z_scored_end.append(zse)

    def long_sleep_memory_time_course(self, save_fig=False, smoothing=20000, template_type="phmm"):
        """
        plots memory drift for all sessions

        :param template_type: "phmm" or "ising"
        :type template_type: str
        :param smoothing: how much smoothing to use for ratio (default: 20000)
        :param smoothing: int
        :param save_fig: whether to save figure (True)
        :type save_fig: bool
        """
        if save_fig:
            plt.style.use('default')

        r_delta_first_half = []
        r_delta_second_half = []

        for i, session in enumerate(self.session_list):

            r_d_f_h, r_d_s_h = session.long_sleep().memory_drift_time_course(template_type=template_type,
                                                                        n_moving_average_pop_vec=smoothing,
                                                                        )
            r_delta_first_half.append(r_d_f_h)
            r_delta_second_half.append(r_d_s_h)

        delta_ratio = np.array(r_delta_first_half)/np.array(r_delta_second_half)
        print(ttest_rel(r_delta_first_half, r_delta_second_half))
        print(ttest_1samp(delta_ratio, popmean=1))
        print(ttest_1samp(delta_ratio, popmean=1, alternative="greater"))

        plt.figure(figsize=(4,6))
        # plt.figure(figsize=(3,4))
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        for session_id, (first, second) in enumerate(zip(r_delta_first_half, r_delta_second_half)):
            plt.scatter([0.1, 0.2], [first,second], label=str(session_id), color=col[session_id], zorder=session_id)
            plt.plot([0.1, 0.2], [first,second], color=col[session_id], zorder=session_id)
            plt.xticks([0.1, 0.2], ["First half\nof sleep", "Second half\nof sleep"])
        plt.ylabel("Delta sim_ratio")
        plt.grid(axis="y", color="gray")
        plt.ylim(0,0.55)
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_first_vs_second_half.svg"), transparent="True")
        else:
            plt.show()

    def long_sleep_memory_drift_opposing_nrem_rem(self, save_fig=False, template_type="phmm", n_moving_average_pop_vec=20,
                                                                                                    rem_pop_vec_threshold=10):
        """
        compute cumulative effect on similarity ratio from NREM and REM

        :param template_type: "phmm" or "ising"
        :type template_type: str
        :param save_fig: whether to save the figure (True)
        :type save_fig: bool
        """

        # --------------------------------------------------------------------------------------------------------------
        # get results from all sessions
        # --------------------------------------------------------------------------------------------------------------
        ds_rem_cum = []
        ds_nrem_cum = []
        ds_rem = []
        ds_nrem = []
        rem_pos = []
        nrem_pos = []
        for i, session in enumerate(self.session_list):
            if template_type =="ising" and session.session_name == "mjc148R4R_0113":
                # this session shows some weird behavior
                ds_rem_cum_, ds_nrem_cum_ = session.long_sleep().memory_drift_plot_rem_nrem(template_type=template_type,
                                                                                                     plotting=False,
                                                                                            n_moving_average_pop_vec=60,
                                                                                            rem_pop_vec_threshold=100)
            else:
                ds_rem_cum_, ds_nrem_cum_ , ds_rem_, ds_nrem_, rem_pos_, nrem_pos_= \
                    session.long_sleep().memory_drift_plot_rem_nrem_delta_score(template_type=template_type,
                                                                                plotting=False,
                                                                                n_moving_average_pop_vec=n_moving_average_pop_vec,
                                                                                rem_pop_vec_threshold=rem_pop_vec_threshold)
            rem_pos.append(rem_pos_)
            nrem_pos.append(nrem_pos_)
            ds_rem_cum.append(ds_rem_cum_)
            ds_nrem_cum.append(ds_nrem_cum_)
            ds_rem.extend(ds_rem_)
            ds_nrem.extend(ds_nrem_)

        ds_rem_cum = np.array(ds_rem_cum)
        ds_nrem_cum = np.array(ds_nrem_cum)

        ds_rem = np.array(ds_rem)
        ds_nrem = np.array(ds_nrem)

        rem_pos = np.array(rem_pos)
        nrem_pos = np.array(nrem_pos)

        if save_fig:
            plt.style.use('default')
            e_c = "black"
            l_c = "white"
        else:
            e_c = "white"
            l_c = "black"
        rem_err = np.std(rem_pos)
        nrem_err = np.std(nrem_pos)

        print(mannwhitneyu(rem_pos, nrem_pos))

        plt.figure(figsize=(2, 4))
        plt.bar([0], np.mean(rem_pos), yerr = rem_err/np.sqrt(6), ecolor=e_c, color="red")
        plt.bar([1], np.mean(nrem_pos), yerr = nrem_err/np.sqrt(6), ecolor=e_c,color="blue")
        plt.bar([0], -1 * (1 - np.mean(rem_pos)), yerr = rem_err/np.sqrt(6), ecolor=e_c, color="red")
        plt.bar([1], -1 * (1 - np.mean(nrem_pos)), yerr = nrem_err/np.sqrt(6), ecolor=e_c,color="blue")
        plt.xticks([0, 1], ["REM", "NREM"])
        plt.ylim(-1, 1)
        plt.yticks([-1,-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], ["1.00","0.75", "0.50", "0.25", "0.00", "0.25", "0.50", "0.75", "1.00"])
        plt.hlines(0, -0.5, 1.5, color=l_c)
        plt.ylabel("Percentage of epochs \n with pos. or negative sign")

        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "opp_rem_nrem_sign.svg"), transparent="True")
        else:
            plt.show()


        c = "white"
        y_dat = [rem_pos, nrem_pos]
        plt.figure(figsize=(3,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["REM", "NREM"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),showfliers=False)
        plt.ylabel("%epochs with positive sign")
        plt.show()

        c = "white"
        y_dat = [1-rem_pos, 1-nrem_pos]
        plt.figure(figsize=(3,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["REM", "NREM"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),showfliers=False)
        plt.ylabel("%epochs with negative sign")
        plt.show()


        c = "white"
        y_dat = [ds_rem, ds_nrem]
        plt.figure(figsize=(3,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["REM", "NREM"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),showfliers=False)
        plt.ylabel("Delta score - all sessions")
        plt.show()


        print("REM vs. NREM delta score (MWU):")
        print(mannwhitneyu(ds_rem_cum, ds_nrem_cum))

        # --------------------------------------------------------------------------------------------------------------
        # plotting
        # --------------------------------------------------------------------------------------------------------------
        if save_fig:
            plt.style.use('default')
            e_c = "black"
        else:
            e_c = "white"
        plt.figure(figsize=(3,5))
        plt.bar(1, np.mean(ds_rem_cum), 0.6, color="red", label="Mean REM")
        plt.scatter(np.ones(ds_rem_cum.shape[0]),ds_rem_cum, zorder=1000, edgecolors=e_c, facecolor="none",
                    label="Session values")
        plt.hlines(0,0.5,2.5)
        plt.xlim(0.5,2.5)
        # plt.ylim(-1, 3)
        plt.bar(2, np.mean(ds_nrem_cum), 0.6, color="blue", label="Mean NREM")
        plt.scatter(np.ones(ds_nrem_cum.shape[0])*2,ds_nrem_cum, zorder=1000, edgecolors=e_c,facecolor="none")
        plt.xticks([1,2],["REM", "NREM"])
        plt.ylabel("CUMULATIVE DELTA SCORE")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.title("Opposing net effect for \nREM and NREM")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "opp_rem_nrem.svg"), transparent="True")
        else:
            plt.show()

    def long_sleep_memory_drift_neighbouring_epochs(self, save_fig=False, template_type="phmm", first_type="rem"):
        """
        computes delta scores of similarity ratio for neighboring and non-neighboring epochs from all sessions
        and scatter plots them

        :param first_type: with whicht sleep part to start ("rem" or "nrem")
        :type first_type:
        :param save_fig: whether to save figure (True) or not
        :type save_fig: bool
        :param template_type: "phmm" or "ising"
        :type template_type: str
        """
        if save_fig:
            plt.style.use('default')

        def make_square_axes(ax):
            """Make an axes square in screen units.

            Should be called after plotting.
            """
            ax.set_aspect(1 / ax.get_data_ratio())

        # --------------------------------------------------------------------------------------------------------------
        # non-neighboring epochs
        # --------------------------------------------------------------------------------------------------------------
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        ds_rem_nn = []
        ds_nrem_nn = []
        for i, session in enumerate(self.session_list):
            ds_rem_arr, ds_nrem_arr = session.long_sleep().memory_drift_neighbouring_epochs(template_type= template_type,
                                                                                            plotting=False,
                                                                                            first_type=first_type)
            ds_rem_arr_nn = ds_rem_arr[2:]
            ds_nrem_arr_nn = ds_nrem_arr[:-2]
            plt.scatter(ds_rem_arr_nn, ds_nrem_arr_nn, color=col[i], alpha=0.8, label=str(i))
            ds_rem_nn.append(ds_rem_arr_nn)
            ds_nrem_nn.append(ds_nrem_arr_nn)
        ds_rem_nn = np.hstack(ds_rem_nn)
        ds_nrem_nn = np.hstack(ds_nrem_nn)
        print("Non-neighbouring epochs, "+str(pearsonr(ds_rem_nn, ds_nrem_nn)))
        slope, intercept, r, p, stderr = scipy.stats.linregress(ds_rem_nn, ds_nrem_nn)

        if save_fig:
            line_col = "black"
        else:
            line_col = "white"

        plt.plot(ds_rem_nn, intercept + slope * ds_rem_nn, color=line_col,
                 label="R="+str(np.round(pearsonr(ds_rem_nn, ds_nrem_nn)[0], 2))+"\n p="+
                       str(pearsonr(ds_rem_nn, ds_nrem_nn)[1]))
        plt.xlabel("DELTA REM")
        plt.ylabel("DELTA NREM")
        plt.title("Non-Neighbouring Epochs")
        plt.legend()
        y_min, y_max = plt.gca().get_ylim()
        y_lim = np.max(np.abs(np.array([y_min, y_max])))
        plt.vlines(0,-y_lim, y_lim, zorder=-1000, color="gray", linewidth=1)
        plt.ylim(-y_lim, y_lim)
        x_min, x_max = plt.gca().get_xlim()
        x_lim = np.max(np.abs(np.array([x_min, x_max])))
        plt.hlines(0,-x_lim, x_lim, zorder=-1000, color="gray", linewidth=1)
        plt.xlim(-x_lim, x_lim)
        make_square_axes(plt.gca())
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "nn_epochs_"+template_type+"_"+first_type+"_"+".svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()
        # --------------------------------------------------------------------------------------------------------------
        # neighboring epochs
        # --------------------------------------------------------------------------------------------------------------
        ds_rem = []
        ds_nrem = []
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        for i, session in enumerate(self.session_list):
            ds_rem_arr, ds_nrem_arr = session.long_sleep().memory_drift_neighbouring_epochs(template_type=template_type,
                                                                                            plotting=False, first_type=first_type)
            plt.scatter(ds_rem_arr, ds_nrem_arr, color=col[i], alpha=0.8, label=str(i))
            ds_rem.append(ds_rem_arr)
            ds_nrem.append(ds_nrem_arr)
        ds_rem = np.hstack(ds_rem)
        ds_nrem = np.hstack(ds_nrem)
        print("Neighbouring epochs, "+str(pearsonr(ds_rem, ds_nrem)))
        slope, intercept, r, p, stderr = scipy.stats.linregress(ds_rem, ds_nrem)
        plt.plot(ds_rem, intercept + slope * ds_rem, color=line_col,
                 label="R="+str(np.round(pearsonr(ds_rem, ds_nrem)[0], 2))+"\n p="+
                       str(pearsonr(ds_rem, ds_nrem)[1]))
        plt.xlabel("DELTA REM")
        plt.ylabel("DELTA NREM")
        plt.title("Neighbouring Epochs")
        plt.legend()
        y_min, y_max = plt.gca().get_ylim()
        y_lim = np.max(np.abs(np.array([y_min, y_max])))
        plt.vlines(0,-y_lim, y_lim, zorder=-1000, color="gray", linewidth=1)
        plt.ylim(-y_lim, y_lim)
        x_min, x_max = plt.gca().get_xlim()
        x_lim = np.max(np.abs(np.array([x_min, x_max])))
        plt.hlines(0,-x_lim, x_lim, zorder=-1000, color="gray", linewidth=1)
        plt.xlim(-x_lim, x_lim)
        make_square_axes(plt.gca())
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "n_epochs_"+template_type+"_"+first_type+"_"+".svg"),
                        transparent="True")
        else:
            plt.show()

        return pearsonr(ds_rem, ds_nrem),  ds_rem_nn.shape[0], ds_nrem_nn.shape[0]

    def long_sleep_memory_drift_neighbouring_epochs_equalized(self, save_fig=False, template_type="phmm", first_type="nrem"):
        """
        computes delta scores of similarity ratio for neighboring and non-neighboring epochs from all sessions
        and scatter plots them

        :param first_type: with whicht sleep part to start ("rem" or "nrem")
        :type first_type:
        :param save_fig: whether to save figure (True) or not
        :type save_fig: bool
        :param template_type: "phmm" or "ising"
        :type template_type: str
        """
        if save_fig:
            plt.style.use('default')

        def make_square_axes(ax):
            """Make an axes square in screen units.

            Should be called after plotting.
            """
            ax.set_aspect(1 / ax.get_data_ratio())

        # --------------------------------------------------------------------------------------------------------------
        # non-neighboring epochs
        # --------------------------------------------------------------------------------------------------------------
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        ds_rem_nn = []
        ds_nrem_nn = []
        for i, session in enumerate(self.session_list):
            ds_rem_arr, ds_nrem_arr = session.long_sleep().memory_drift_rem_nrem_equalized_rem_nrem_pairs()
            ds_rem_arr_nn = ds_rem_arr[2:]
            ds_nrem_arr_nn = ds_nrem_arr[:-2]
            plt.scatter(ds_rem_arr_nn, ds_nrem_arr_nn, color=col[i], alpha=0.8, label=str(i))
            ds_rem_nn.append(ds_rem_arr_nn)
            ds_nrem_nn.append(ds_nrem_arr_nn)
        ds_rem_nn = np.hstack(ds_rem_nn)
        ds_nrem_nn = np.hstack(ds_nrem_nn)
        print("Non-neighbouring epochs, "+str(pearsonr(ds_rem_nn, ds_nrem_nn)))
        slope, intercept, r, p, stderr = scipy.stats.linregress(ds_rem_nn, ds_nrem_nn)

        if save_fig:
            line_col = "black"
        else:
            line_col = "white"

        plt.plot(ds_rem_nn, intercept + slope * ds_rem_nn, color=line_col,
                 label="R="+str(np.round(pearsonr(ds_rem_nn, ds_nrem_nn)[0], 2))+"\n p="+
                       str(pearsonr(ds_rem_nn, ds_nrem_nn)[1]))
        plt.xlabel("DELTA REM")
        plt.ylabel("DELTA NREM")
        plt.title("Non-Neighbouring Epochs")
        plt.legend()
        y_min, y_max = plt.gca().get_ylim()
        y_lim = np.max(np.abs(np.array([y_min, y_max])))
        plt.vlines(0,-y_lim, y_lim, zorder=-1000, color="gray", linewidth=1)
        plt.ylim(-y_lim, y_lim)
        x_min, x_max = plt.gca().get_xlim()
        x_lim = np.max(np.abs(np.array([x_min, x_max])))
        plt.hlines(0,-x_lim, x_lim, zorder=-1000, color="gray", linewidth=1)
        plt.xlim(-x_lim, x_lim)
        make_square_axes(plt.gca())
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "nn_epochs_"+template_type+"_"+first_type+"_equalized.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()
        # --------------------------------------------------------------------------------------------------------------
        # neighboring epochs
        # --------------------------------------------------------------------------------------------------------------
        ds_rem = []
        ds_nrem = []
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        for i, session in enumerate(self.session_list):
            ds_rem_arr, ds_nrem_arr = session.long_sleep().memory_drift_neighbouring_epochs(template_type=template_type,
                                                                                            plotting=False, first_type=first_type)
            plt.scatter(ds_rem_arr, ds_nrem_arr, color=col[i], alpha=0.8, label=str(i))
            ds_rem.append(ds_rem_arr)
            ds_nrem.append(ds_nrem_arr)
        ds_rem = np.hstack(ds_rem)
        ds_nrem = np.hstack(ds_nrem)
        print("Neighbouring epochs, "+str(pearsonr(ds_rem, ds_nrem)))
        slope, intercept, r, p, stderr = scipy.stats.linregress(ds_rem, ds_nrem)
        plt.plot(ds_rem, intercept + slope * ds_rem, color=line_col,
                 label="R="+str(np.round(pearsonr(ds_rem, ds_nrem)[0], 2))+"\n p="+
                       str(pearsonr(ds_rem, ds_nrem)[1]))
        plt.xlabel("DELTA REM")
        plt.ylabel("DELTA NREM")
        plt.title("Neighbouring Epochs")
        plt.legend()
        y_min, y_max = plt.gca().get_ylim()
        y_lim = np.max(np.abs(np.array([y_min, y_max])))
        plt.vlines(0,-y_lim, y_lim, zorder=-1000, color="gray", linewidth=1)
        plt.ylim(-y_lim, y_lim)
        x_min, x_max = plt.gca().get_xlim()
        x_lim = np.max(np.abs(np.array([x_min, x_max])))
        plt.hlines(0,-x_lim, x_lim, zorder=-1000, color="gray", linewidth=1)
        plt.xlim(-x_lim, x_lim)
        make_square_axes(plt.gca())
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "n_epochs_"+template_type+"_"+first_type+"_equalized.svg"),
                        transparent="True")
        else:
            plt.show()

    def long_sleep_memory_drift_neighbouring_epochs_stats(self):

        pearson_rem_nrem, n_rem, n_nrem = self.long_sleep_memory_drift_neighbouring_epochs(save_fig=False, first_type="rem")
        pearson_nrem_rem, n_rem_, n_nrem_ = self.long_sleep_memory_drift_neighbouring_epochs(save_fig=False,
                                                                                           first_type="nrem")

        # apply Fisher's z-transform to R-values
        r_rem_nrem_fish = 0.5*np.log((1+pearson_rem_nrem[0])/(1-pearson_rem_nrem[0]))
        r_nrem_rem_fish = 0.5*np.log((1+pearson_nrem_rem[0])/(1-pearson_nrem_rem[0]))

        std_ = np.sqrt((1/(n_rem-3))+(1/(n_nrem-3)))

        z_score = (r_rem_nrem_fish - r_nrem_rem_fish)/std_

        p_value_one_sided = scipy.stats.norm.sf(abs(z_score))
        p_value_two_sided = scipy.stats.norm.sf(abs(z_score)) * 2

    def long_sleep_memory_drift_neighbouring_epochs_same_sleep_phase(self, save_fig=False, template_type="phmm"):
        """
        computes delta scores of similarity ratio for neighboring and non-neighboring epochs from all sessions
        and scatter plots them

        :param save_fig: whether to save figure (True) or not
        :type save_fig: bool
        :param template_type: "phmm" or "ising"
        :type template_type: str
        """

        if save_fig:
            plt.style.use('default')
            line_col = "black"
        else:
            line_col = "white"

        def make_square_axes(ax):
            """Make an axes square in screen units.

            Should be called after plotting.
            """
            ax.set_aspect(1 / ax.get_data_ratio())

        # --------------------------------------------------------------------------------------------------------------
        # non-neighboring epochs
        # --------------------------------------------------------------------------------------------------------------
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        ds_rem = []
        ds_nrem = []
        for i, session in enumerate(self.session_list):
            ds_rem_arr, ds_nrem_arr = session.long_sleep().memory_drift_neighbouring_epochs(template_type=
                                                                                                 template_type,
                                                                                                 plotting=False)
            ds_rem.append(ds_rem_arr)
            ds_nrem.append(ds_nrem_arr)

        # REM results
        # --------------------------------------------------------------------------------------------------------------
        ds_rem_n = []
        ds_rem_n_plus_1 = []
        for d in ds_rem:
            ds_rem_n.append(d[:-1])
            ds_rem_n_plus_1.append(d[1:])

        ds_rem_n_arr = np.hstack(ds_rem_n)
        ds_rem_n_plus_1_arr = np.hstack(ds_rem_n_plus_1)

        # for neighbouring REM epochs
        print("Neighbouring REM epochs, " + str(pearsonr(ds_rem_n_arr, ds_rem_n_plus_1_arr)))
        for sess_id, (ds_rn, ds_rnp1) in enumerate(zip(ds_rem_n, ds_rem_n_plus_1)):
            plt.scatter(ds_rn, ds_rnp1, color=col[sess_id])
        slope, intercept, r, p, stderr = scipy.stats.linregress(ds_rem_n_arr, ds_rem_n_plus_1_arr)
        plt.plot(ds_rem_n_arr, intercept + slope * ds_rem_n_arr, color=line_col,
                 label="R="+str(np.round(pearsonr(ds_rem_n_arr, ds_rem_n_plus_1_arr)[0], 2))+"\n p="+str(pearsonr(ds_rem_n_arr, ds_rem_n_plus_1_arr)[1]))
        plt.xlabel("DELTA REM n")
        plt.ylabel("DELTA REM n+1")
        plt.legend()
        y_min, y_max = plt.gca().get_ylim()
        y_lim = np.max(np.abs(np.array([y_min, y_max])))
        plt.vlines(0,-y_lim, y_lim, zorder=-1000, color="gray", linewidth=1)
        plt.ylim(-y_lim, y_lim)
        x_min, x_max = plt.gca().get_xlim()
        x_lim = np.max(np.abs(np.array([x_min, x_max])))
        plt.hlines(0,-x_lim, x_lim, zorder=-1000, color="gray", linewidth=1)
        plt.xlim(-x_lim, x_lim)
        make_square_axes(plt.gca())
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "neighboring_rem_epochs.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # NREM results
        # --------------------------------------------------------------------------------------------------------------
        ds_nrem_n = []
        ds_nrem_n_plus_1 = []
        for d in ds_nrem:
            ds_nrem_n.append(d[:-1])
            ds_nrem_n_plus_1.append(d[1:])

        ds_nrem_n_arr = np.hstack(ds_nrem_n)
        ds_nrem_n_plus_1_arr = np.hstack(ds_nrem_n_plus_1)

        # for neighbouring REM epochs
        print("Neighbouring NREM epochs, " + str(pearsonr(ds_nrem_n_arr, ds_nrem_n_plus_1_arr)))
        for sess_id, (ds_nrn, ds_nrnp1) in enumerate(zip(ds_nrem_n, ds_nrem_n_plus_1)):
            plt.scatter(ds_nrn, ds_nrnp1, color=col[sess_id])
        slope, intercept, r, p, stderr = scipy.stats.linregress(ds_nrem_n_arr, ds_nrem_n_plus_1_arr)
        plt.plot(ds_nrem_n_arr, intercept + slope * ds_nrem_n_arr, color=line_col,
                 label="R="+str(np.round(pearsonr(ds_nrem_n_arr, ds_nrem_n_plus_1_arr)[0], 2))+
                       "\n p="+str(pearsonr(ds_nrem_n_arr, ds_nrem_n_plus_1_arr)[1]))
        plt.xlabel("DELTA NREM n")
        plt.ylabel("DELTA NREM n+1")
        plt.legend()
        y_min, y_max = plt.gca().get_ylim()
        y_lim = np.max(np.abs(np.array([y_min, y_max])))
        plt.vlines(0,-y_lim, y_lim, zorder=-1000, color="gray", linewidth=1)
        plt.ylim(-y_lim, y_lim)
        x_min, x_max = plt.gca().get_xlim()
        x_lim = np.max(np.abs(np.array([x_min, x_max])))
        plt.hlines(0,-x_lim, x_lim, zorder=-1000, color="gray", linewidth=1)
        plt.xlim(-x_lim, x_lim)
        make_square_axes(plt.gca())
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "neighboring_nrem_epochs.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def long_sleep_memory_drift_and_firing_prob(self, save_fig=False, use_only_non_stationary_periods=False,
                                                n_smoothing_firing_rates=200):

        ds_rem_clean = []
        ds_nrem_clean = []
        rem_stable_clean = []
        nrem_stable_clean = []
        rem_dec_clean = []
        nrem_dec_clean = []
        rem_inc_clean = []
        nrem_inc_clean = []
        r_rem_dec = []
        r_nrem_dec = []
        r_rem_inc = []
        r_nrem_inc = []
        r_rem_stable = []
        r_nrem_stable = []

        for i, session in enumerate(self.session_list):
            ds_rem_clean_, ds_nrem_clean_, rem_stable_clean_, nrem_stable_clean_, rem_dec_clean_, nrem_dec_clean_, \
            rem_inc_clean_, nrem_inc_clean_ = \
                session.long_sleep().memory_drift_and_firing_prob(plotting=False, template_type="phmm",
                                                                  use_only_non_stationary_periods=
                                                                  use_only_non_stationary_periods,
                                                                  n_smoothing_firing_rates=200)
            r_rem_dec.append(pearsonr(ds_rem_clean_, rem_dec_clean_)[0])
            r_nrem_dec.append(pearsonr(ds_nrem_clean_, nrem_dec_clean_)[0])
            r_rem_inc.append(pearsonr(ds_rem_clean_, rem_inc_clean_)[0])
            r_nrem_inc.append(pearsonr(ds_nrem_clean_, nrem_inc_clean_)[0])
            r_rem_stable.append(pearsonr(ds_rem_clean_, rem_stable_clean_)[0])
            r_nrem_stable.append(pearsonr(ds_nrem_clean_, nrem_stable_clean_)[0])
            ds_rem_clean.append(ds_rem_clean_)
            ds_nrem_clean.append(ds_nrem_clean_)
            rem_stable_clean.append(rem_stable_clean_)
            nrem_stable_clean.append(nrem_stable_clean_)
            rem_dec_clean.append(rem_dec_clean_)
            nrem_dec_clean.append(nrem_dec_clean_)
            rem_inc_clean.append(rem_inc_clean_)
            nrem_inc_clean.append(nrem_inc_clean_)

        ds_rem_clean = np.hstack(ds_rem_clean)
        ds_nrem_clean = np.hstack(ds_nrem_clean)
        rem_stable_clean = np.hstack(rem_stable_clean)
        nrem_stable_clean = np.hstack(nrem_stable_clean)
        rem_dec_clean = np.hstack(rem_dec_clean)
        nrem_dec_clean = np.hstack(nrem_dec_clean)
        rem_inc_clean = np.hstack(rem_inc_clean)
        nrem_inc_clean = np.hstack(nrem_inc_clean)

        ds_rem_min = - np.max([np.abs(np.min(ds_rem_clean)), np.max(ds_rem_clean)])
        ds_rem_max = np.max([np.abs(np.min(ds_rem_clean)), np.max(ds_rem_clean)])
        ds_nrem_min = - np.max([np.abs(np.min(ds_nrem_clean)), np.max(ds_nrem_clean)])
        ds_nrem_max = np.max([np.abs(np.min(ds_nrem_clean)), np.max(ds_nrem_clean)])

        inc_all = np.hstack((rem_inc_clean, nrem_inc_clean))
        dec_all = np.hstack((rem_dec_clean, nrem_dec_clean))
        stable_all = np.hstack((rem_stable_clean, nrem_stable_clean))
        ds_all = np.hstack((ds_rem_clean, ds_nrem_clean))

        # increasing cells: NREM
        # --------------------------------------------------------------------------------------------------------------
        # R values
        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(2.5, 2))
        plt.scatter(np.arange(len(r_nrem_inc)), r_nrem_inc, color="black")
        plt.ylim(-1,1)
        plt.xticks([0,6], [1,7])
        plt.yticks([-1, 0, 1])
        plt.ylabel("Pearson R")
        plt.xlabel("Session ID")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        plt.tight_layout()
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(os.path.join(save_path, "delta_score_prob_fir_inc_nrem_all_sessions_r_values.svg"),
                    transparent="True")
        plt.close()

        # scatter plot
        plt.figure(figsize=(4, 4))
        plt.grid(color="gray", zorder=-1000)
        plt.gca().set_axisbelow(True)
        plt.scatter(ds_nrem_clean, nrem_inc_clean, c="#F7931D", edgecolors="white", label="increasing cells")
        plt.xlabel("Delta_score NREM period")
        plt.ylabel("Change in firing probability")
        plt.text(-0.5, -0.2, "R=" + str(np.round(pearsonr(ds_nrem_clean, nrem_inc_clean)[0], 3)))
        plt.text(-0.5, -0.22, "p=" + str(pearsonr(ds_nrem_clean, nrem_inc_clean)[1]))
        plt.text(-0.5, 0.22, "NREM")
        plt.xlim(ds_nrem_min-0.25, ds_nrem_max+0.05)
        plt.ylim(-0.42, 0.32)
        plt.legend(loc=1)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "delta_score_prob_fir_inc_nrem_all_sessions.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # increasing cells: REM
        # --------------------------------------------------------------------------------------------------------------
        # R values
        plt.figure(figsize=(2.5, 2))
        plt.scatter(np.arange(len(r_rem_inc)), r_rem_inc, color="black")
        plt.ylim(-1,1)
        plt.xticks([0,6], [1,7])
        plt.yticks([-1,0,1])
        plt.ylabel("Pearson R")
        plt.xlabel("Session ID")
        plt.tight_layout()
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(os.path.join(save_path, "delta_score_prob_fir_inc_rem_all_sessions_r_values.svg"), transparent="True")
        plt.close()

        # scatter plot
        plt.figure(figsize=(4, 4))
        plt.scatter(ds_rem_clean, rem_inc_clean, c="#F7931D", edgecolors="white", label="increasing cells")
        plt.grid(color="gray", zorder=-1000)
        plt.gca().set_axisbelow(True)
        plt.xlabel("Delta_score REM period")
        plt.ylabel("Change in firing probability")
        plt.text(-0.5, -0.2, "R=" + str(np.round(pearsonr(ds_rem_clean, rem_inc_clean)[0], 3)))
        plt.text(-0.5, -0.22, "p=" + str(pearsonr(ds_rem_clean, rem_inc_clean)[1]))
        plt.text(-0.5, 0.22, "REM")
        plt.xlim(ds_rem_min-0.05, ds_rem_max+0.25)
        plt.ylim(-0.32, 0.42)
        plt.legend(loc=1)
        # plt.gca().set_aspect('equal')
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "delta_score_prob_fir_inc_rem_all_sessions.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # decreasing cells: NREM
        # --------------------------------------------------------------------------------------------------------------
        # R values
        plt.figure(figsize=(2.5, 2))
        plt.scatter(np.arange(len(r_nrem_dec)), r_nrem_dec, color="black")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        plt.ylim(-1,1)
        plt.xticks([0,6], [1,7])
        plt.yticks([-1,0,1])
        plt.ylabel("Pearson R")
        plt.xlabel("Session ID")
        plt.tight_layout()
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(os.path.join(save_path, "delta_score_prob_fir_dec_nrem_all_sessions_r_values.svg"), transparent="True")
        plt.close()

        # scatter plot
        plt.figure(figsize=(4, 4))
        plt.grid(color="gray", zorder=-1000)
        plt.gca().set_axisbelow(True)
        plt.scatter(ds_nrem_clean, nrem_dec_clean, c="#00A79D", edgecolors="white", label="decreasing cells")
        plt.xlabel("Delta_score NREM period")
        plt.ylabel("Change in firing probability")
        plt.text(-0.5, -0.2, "R=" + str(np.round(pearsonr(ds_nrem_clean, nrem_dec_clean)[0], 3)))
        plt.text(-0.5, -0.22, "p=" + str(pearsonr(ds_nrem_clean, nrem_dec_clean)[1]))
        plt.text(-0.5, 0.22, "NREM")
        plt.xlim(ds_nrem_min-0.25, ds_nrem_max+0.05)
        plt.ylim(-0.42, 0.32)
        plt.legend(loc=1)
        # plt.gca().set_aspect('equal')
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "delta_score_prob_fir_dec_nrem_all_sessions.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # decreasing cells: REM
        # --------------------------------------------------------------------------------------------------------------
        # R values
        plt.figure(figsize=(2.5, 2))
        plt.scatter(np.arange(len(r_rem_dec)), r_rem_dec, color="black")
        plt.ylim(-1,1)
        plt.xticks([0,6], [1,7])
        plt.yticks([-1,0,1])
        plt.ylabel("Pearson R")
        plt.xlabel("Session ID")
        plt.tight_layout()
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(os.path.join(save_path, "delta_score_prob_fir_dec_rem_all_sessions_r_values.svg"), transparent="True")
        plt.close()

        # scatter plot
        plt.figure(figsize=(4, 4))
        plt.grid(color="gray", zorder=-1000)
        plt.gca().set_axisbelow(True)
        plt.scatter(ds_rem_clean, rem_dec_clean, c="#00A79D", edgecolors="white", label="decreasing cells")
        plt.xlabel("Delta_score REM period")
        plt.ylabel("Change in firing probability")
        plt.text(-0.8, -0.2, "R=" + str(np.round(pearsonr(ds_rem_clean, rem_dec_clean)[0], 3)))
        plt.text(-0.8, -0.22, "p=" + str(pearsonr(ds_rem_clean, rem_dec_clean)[1]))
        plt.text(-0.8, 0.22, "REM")
        plt.xlim(ds_rem_min-0.05, ds_rem_max+0.25)
        plt.ylim(-0.32, 0.42)
        plt.legend(loc=1)
        # plt.gca().set_aspect('equal')
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "delta_score_prob_fir_dec_rem_all_sessions.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # stable cells: NREM
        # --------------------------------------------------------------------------------------------------------------
        # R values
        plt.figure(figsize=(2.5, 2))
        plt.scatter(np.arange(len(r_nrem_stable)), r_nrem_stable, color="black")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        plt.ylim(-1,1)
        plt.xticks([0,6], [1,7])
        plt.yticks([-1,0,1])
        plt.ylabel("Pearson R")
        plt.xlabel("Session ID")
        plt.tight_layout()
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(os.path.join(save_path, "delta_score_prob_fir_stable_nrem_all_sessions_r_values.svg"), transparent="True")
        plt.close()

        # scatter plot
        plt.figure(figsize=(4, 4))
        plt.grid(color="gray", zorder=-1000)
        plt.gca().set_axisbelow(True)
        plt.scatter(ds_nrem_clean, nrem_stable_clean, c="#91268F", edgecolors="white", label="persistent cells")
        plt.xlabel("Delta_score NREM period")
        plt.ylabel("Change in firing probability")
        plt.text(-0.6, -0.2, "R=" + str(np.round(pearsonr(ds_nrem_clean, nrem_stable_clean)[0], 3)))
        plt.text(-0.6, -0.22, "p=" + str(pearsonr(ds_nrem_clean, nrem_stable_clean)[1]))
        plt.xlim(ds_nrem_min-0.25, ds_nrem_max+0.05)
        plt.ylim(-0.42, 0.32)
        plt.legend(loc=1)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "delta_score_prob_fir_stable_nrem_all_sessions.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        # stable cells: REM
        # --------------------------------------------------------------------------------------------------------------
        # R values
        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(2.5, 2))
        plt.scatter(np.arange(len(r_rem_stable)), r_rem_stable, color="black")
        plt.ylim(-1,1)
        plt.xticks([0,6], [1,7])
        plt.yticks([-1,0,1])
        plt.ylabel("Pearson R")
        plt.xlabel("Session ID")
        plt.tight_layout()
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(os.path.join(save_path, "delta_score_prob_fir_stable_rem_all_sessions_r_values.svg"), transparent="True")
        plt.close()

        # scatter plot
        plt.figure(figsize=(4, 4))
        plt.grid(color="gray", zorder=-1000)
        plt.gca().set_axisbelow(True)
        plt.scatter(ds_rem_clean, rem_stable_clean, c="#91268F", edgecolors="white", label="persistent cells")
        plt.xlabel("Delta_score REM period")
        plt.ylabel("Change in firing probability")
        plt.text(-0.8, -0.2, "R=" + str(np.round(pearsonr(ds_rem_clean, rem_stable_clean)[0], 3)))
        plt.text(-0.8, -0.22, "p=" + str(pearsonr(ds_rem_clean, rem_stable_clean)[1]))
        plt.xlim(ds_rem_min-0.05, ds_rem_max+0.25)
        plt.ylim(-0.32, 0.42)
        plt.legend(loc=1)
        # plt.gca().set_aspect('equal')
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "delta_score_prob_fir_stable_rem_all_sessions.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # REM and NREM together
        # --------------------------------------------------------------------------------------------------------------
        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(5, 5))
        plt.scatter(ds_all, inc_all, c="#F7931D", edgecolors="white", label="increasing cells")
        plt.xlabel("Delta_score period")
        plt.ylabel("Change in firing probability")
        plt.text(-0.5, -0.25, "R=" + str(np.round(pearsonr(ds_all, inc_all)[0], 3)))
        plt.text(-0.5, -0.29, "p=" + str(pearsonr(ds_all, inc_all)[1]))
        plt.xlim(np.min([ds_nrem_min, ds_rem_min])-0.05, np.max([ds_nrem_max, ds_rem_max])+0.05)
        plt.ylim(-0.32, 0.32)
        plt.legend(loc=1)
        # plt.gca().set_aspect('equal')
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "delta_score_prob_fir_inc_nrem_all_sessions.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def long_sleep_memory_drift_delta_log_likelihood_and_firing_prob(self, save_fig=False, use_only_non_stationary_periods=False,
                                                n_smoothing_firing_rates=200):

        ds_pre_rem_clean = []
        ds_post_rem_clean = []
        ds_pre_nrem_clean = []
        ds_post_nrem_clean = []
        rem_dec_clean = []
        nrem_dec_clean = []
        rem_inc_clean = []
        nrem_inc_clean = []
        r_pre_rem_dec = []
        r_post_rem_dec = []
        r_pre_nrem_dec = []
        r_post_nrem_dec = []
        r_pre_rem_inc = []
        r_post_rem_inc = []
        r_pre_nrem_inc = []
        r_post_nrem_inc = []

        for i, session in enumerate(self.session_list):
            ds_pre_rem_clean_, ds_post_rem_clean_, ds_pre_nrem_clean_, ds_post_nrem_clean_, _, \
                _, rem_dec_clean_, nrem_dec_clean_, \
                rem_inc_clean_, nrem_inc_clean_ = \
                session.long_sleep().memory_drift_delta_log_likelihood_and_firing_prob(plotting=False, template_type="phmm",
                                                                  use_only_non_stationary_periods=
                                                                  use_only_non_stationary_periods,
                                                                  n_smoothing_firing_rates=200)
            r_pre_rem_dec.append(pearsonr(ds_pre_rem_clean_, rem_dec_clean_)[0])
            r_post_rem_dec.append(pearsonr(ds_post_rem_clean_, rem_dec_clean_)[0])
            r_pre_nrem_dec.append(pearsonr(ds_pre_nrem_clean_, nrem_dec_clean_)[0])
            r_post_nrem_dec.append(pearsonr(ds_post_nrem_clean_, nrem_dec_clean_)[0])
            r_pre_rem_inc.append(pearsonr(ds_pre_rem_clean_, rem_inc_clean_)[0])
            r_post_rem_inc.append(pearsonr(ds_post_rem_clean_, rem_inc_clean_)[0])
            r_pre_nrem_inc.append(pearsonr(ds_pre_nrem_clean_, nrem_inc_clean_)[0])
            r_post_nrem_inc.append(pearsonr(ds_post_nrem_clean_, nrem_inc_clean_)[0])
            ds_pre_rem_clean.append(ds_pre_rem_clean_)
            ds_post_rem_clean.append(ds_post_rem_clean_)
            ds_pre_nrem_clean.append(ds_pre_nrem_clean_)
            ds_post_nrem_clean.append(ds_post_nrem_clean_)
            rem_dec_clean.append(rem_dec_clean_)
            nrem_dec_clean.append(nrem_dec_clean_)
            rem_inc_clean.append(rem_inc_clean_)
            nrem_inc_clean.append(nrem_inc_clean_)

        ds_pre_rem_clean = np.hstack(ds_pre_rem_clean)
        ds_post_rem_clean = np.hstack(ds_post_rem_clean)
        ds_pre_nrem_clean = np.hstack(ds_pre_nrem_clean)
        ds_post_nrem_clean = np.hstack(ds_post_nrem_clean)
        rem_dec_clean = np.hstack(rem_dec_clean)
        nrem_dec_clean = np.hstack(nrem_dec_clean)
        rem_inc_clean = np.hstack(rem_inc_clean)
        nrem_inc_clean = np.hstack(nrem_inc_clean)

        ds_pre_rem_min = - np.max([np.abs(np.min(ds_pre_rem_clean)), np.max(ds_pre_rem_clean)])
        ds_post_rem_min = - np.max([np.abs(np.min(ds_post_rem_clean)), np.max(ds_post_rem_clean)])
        ds_pre_rem_max = np.max([np.abs(np.min(ds_pre_rem_clean)), np.max(ds_pre_rem_clean)])
        ds_post_rem_max = np.max([np.abs(np.min(ds_post_rem_clean)), np.max(ds_post_rem_clean)])
        ds_pre_nrem_min = - np.max([np.abs(np.min(ds_pre_nrem_clean)), np.max(ds_pre_nrem_clean)])
        ds_post_nrem_min = - np.max([np.abs(np.min(ds_post_nrem_clean)), np.max(ds_post_nrem_clean)])
        ds_pre_nrem_max = np.max([np.abs(np.min(ds_pre_nrem_clean)), np.max(ds_pre_nrem_clean)])
        ds_post_nrem_max = np.max([np.abs(np.min(ds_post_nrem_clean)), np.max(ds_post_nrem_clean)])

        inc_all = np.hstack((rem_inc_clean, nrem_inc_clean))
        dec_all = np.hstack((rem_dec_clean, nrem_dec_clean))
        ds_pre_all = np.hstack((ds_pre_rem_clean, ds_pre_nrem_clean))
        ds_post_all = np.hstack((ds_post_rem_clean, ds_post_nrem_clean))

        x_min = -12
        x_max = 12
        y_min = -0.3
        y_max = 0.3
        # increasing cells: NREM
        # --------------------------------------------------------------------------------------------------------------
        # R values

        plt.style.use('default')
        plt.figure(figsize=(2.5, 2))
        plt.scatter(np.arange(len(r_pre_nrem_inc)), r_pre_nrem_inc, color="black")
        plt.ylim(-1,1)
        plt.xticks([0,6], [1,7])
        plt.yticks([-1, 0, 1])
        plt.ylabel("Pearson R")
        plt.xlabel("Session ID")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "log_likeli_pre_fir_inc_nrem_all_sessions_r_values.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        plt.figure(figsize=(2.5, 2))
        plt.scatter(np.arange(len(r_post_nrem_inc)), r_post_nrem_inc, color="black")
        plt.ylim(-1,1)
        plt.xticks([0,6], [1,7])
        plt.yticks([-1, 0, 1])
        plt.ylabel("Pearson R")
        plt.xlabel("Session ID")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "log_likeli_post_fir_inc_nrem_all_sessions_r_values.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        # scatter plot
        plt.figure(figsize=(4, 4))
        # plt.grid(color="gray", zorder=-1000)
        plt.gca().set_axisbelow(True)
        plt.scatter(ds_pre_nrem_clean, nrem_inc_clean, c="#F7931D", edgecolors="white")
        slope, intercept, r, p, stderr = scipy.stats.linregress(ds_pre_nrem_clean, nrem_inc_clean)
        plt.plot(ds_pre_nrem_clean, intercept + slope * ds_pre_nrem_clean, color="#F7931D",
                 label="R="+str(np.round(pearsonr(ds_pre_nrem_clean, nrem_inc_clean)[0], 2))+"\n p="+str(pearsonr(ds_pre_nrem_clean, nrem_inc_clean)[1]))
        plt.xlabel("log-likelihood NREM")
        plt.ylabel("firing probability NREM")
        # plt.text(-0.5, 0.22, "NREM")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.hlines(0, x_min, x_max, color="lightgrey", zorder=-1000)
        plt.vlines(0, y_min, y_max, color="lightgrey", zorder=-1000)
        plt.legend(loc=1)
        plt.title("Acquisition: NREM")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "log_likeli_pre_fir_inc_nrem_all_sessions.svg"), transparent="True")
            plt.close()
        else:
            plt.show()


        # scatter plot
        plt.figure(figsize=(4, 4))
        # plt.grid(color="gray", zorder=-1000)
        plt.gca().set_axisbelow(True)
        plt.scatter(ds_post_nrem_clean, nrem_inc_clean, c="#F7931D", edgecolors="white")
        slope, intercept, r, p, stderr = scipy.stats.linregress(ds_post_nrem_clean, nrem_inc_clean)
        plt.plot(ds_post_nrem_clean, intercept + slope * ds_post_nrem_clean, color="#F7931D",
                 label="R="+str(np.round(pearsonr(ds_post_nrem_clean, nrem_inc_clean)[0], 2))+"\n p="+str(pearsonr(ds_post_nrem_clean, nrem_inc_clean)[1]))
        plt.xlabel("log-likelihood NREM")
        plt.ylabel("firing probability NREM")
        # plt.text(-0.5, 0.22, "NREM")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.hlines(0, x_min, x_max, color="lightgrey", zorder=-1000)
        plt.vlines(0, y_min, y_max, color="lightgrey", zorder=-1000)
        plt.legend(loc=1)
        plt.title("Recall: NREM")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "log_likeli_post_fir_inc_nrem_all_sessions.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # increasing cells: REM
        # --------------------------------------------------------------------------------------------------------------
        # R values

        plt.style.use('default')
        plt.figure(figsize=(2.5, 2))
        plt.scatter(np.arange(len(r_pre_rem_inc)), r_pre_rem_inc, color="black")
        plt.ylim(-1,1)
        plt.xticks([0,6], [1,7])
        plt.yticks([-1, 0, 1])
        plt.ylabel("Pearson R")
        plt.xlabel("Session ID")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "log_likeli_pre_fir_inc_rem_all_sessions_r_values.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        plt.figure(figsize=(2.5, 2))
        plt.scatter(np.arange(len(r_post_rem_inc)), r_post_rem_inc, color="black")
        plt.ylim(-1,1)
        plt.xticks([0,6], [1,7])
        plt.yticks([-1, 0, 1])
        plt.ylabel("Pearson R")
        plt.xlabel("Session ID")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "log_likeli_post_fir_inc_rem_all_sessions_r_values.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        # scatter plot
        plt.figure(figsize=(4, 4))
        # plt.grid(color="gray", zorder=-1000)
        plt.gca().set_axisbelow(True)
        plt.scatter(ds_pre_rem_clean, rem_inc_clean, c="#F7931D", edgecolors="white")
        slope, intercept, r, p, stderr = scipy.stats.linregress(ds_pre_rem_clean, rem_inc_clean)
        plt.plot(ds_pre_rem_clean, intercept + slope * ds_pre_rem_clean, color="#F7931D",
                 label="R="+str(np.round(pearsonr(ds_pre_rem_clean, rem_inc_clean)[0], 2))+"\n p="+str(pearsonr(ds_pre_rem_clean, rem_inc_clean)[1]))
        plt.xlabel("log-likelihood REM")
        plt.ylabel("firing probability REM")
        # plt.text(-0.5, 0.22, "NREM")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.hlines(0, x_min, x_max, color="lightgrey", zorder=-1000)
        plt.vlines(0, y_min, y_max, color="lightgrey", zorder=-1000)
        plt.legend(loc=1)
        plt.title("Acquisition: REM")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "log_likeli_pre_fir_inc_rem_all_sessions.svg"), transparent="True")
            plt.close()
        else:
            plt.show()


        # scatter plot
        plt.figure(figsize=(4, 4))
        # plt.grid(color="gray", zorder=-1000)
        plt.gca().set_axisbelow(True)
        plt.scatter(ds_post_rem_clean, rem_inc_clean, c="#F7931D", edgecolors="white")
        slope, intercept, r, p, stderr = scipy.stats.linregress(ds_post_rem_clean, rem_inc_clean)
        plt.plot(ds_post_rem_clean, intercept + slope * ds_post_rem_clean, color="#F7931D",
                 label="R="+str(np.round(pearsonr(ds_post_rem_clean, rem_inc_clean)[0], 2))+"\n p="+str(pearsonr(ds_post_rem_clean, rem_inc_clean)[1]))
        plt.xlabel("log-likelihood REM")
        plt.ylabel("firing probability REM")
        # plt.text(-0.5, 0.22, "NREM")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.hlines(0, x_min, x_max, color="lightgrey", zorder=-1000)
        plt.vlines(0, y_min, y_max, color="lightgrey", zorder=-1000)
        plt.legend(loc=1)
        plt.title("Recall: REM")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "log_likeli_post_fir_inc_rem_all_sessions.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # decreasing cells: NREM
        # --------------------------------------------------------------------------------------------------------------
        # R values

        plt.style.use('default')
        plt.figure(figsize=(2.5, 2))
        plt.scatter(np.arange(len(r_pre_nrem_dec)), r_pre_nrem_dec, color="black")
        plt.ylim(-1,1)
        plt.xticks([0,6], [1,7])
        plt.yticks([-1, 0, 1])
        plt.ylabel("Pearson R")
        plt.xlabel("Session ID")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "log_likeli_pre_fir_dec_nrem_all_sessions_r_values.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        plt.figure(figsize=(2.5, 2))
        plt.scatter(np.arange(len(r_post_nrem_dec)), r_post_nrem_dec, color="black")
        plt.ylim(-1,1)
        plt.xticks([0,6], [1,7])
        plt.yticks([-1, 0, 1])
        plt.ylabel("Pearson R")
        plt.xlabel("Session ID")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "log_likeli_post_fir_dec_nrem_all_sessions_r_values.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        # scatter plot
        plt.figure(figsize=(4, 4))
        # plt.grid(color="gray", zorder=-1000)
        plt.gca().set_axisbelow(True)
        plt.scatter(ds_pre_nrem_clean, nrem_dec_clean, c="#00A79D", edgecolors="white")
        slope, intercept, r, p, stderr = scipy.stats.linregress(ds_pre_nrem_clean, nrem_dec_clean)
        plt.plot(ds_pre_nrem_clean, intercept + slope * ds_pre_nrem_clean, color="#00A79D",
                 label="R="+str(np.round(pearsonr(ds_pre_nrem_clean, nrem_dec_clean)[0], 2))+"\n p="+str(pearsonr(ds_pre_nrem_clean, nrem_dec_clean)[1]))
        plt.xlabel("log-likelihood NREM")
        plt.ylabel("firing probability NREM")
        # plt.text(-0.5, 0.22, "NREM")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.hlines(0, x_min, x_max, color="lightgrey", zorder=-1000)
        plt.vlines(0, y_min, y_max, color="lightgrey", zorder=-1000)
        plt.legend(loc=1)
        plt.title("Acquisition: NREM")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "log_likeli_pre_fir_dec_nrem_all_sessions.svg"), transparent="True")
            plt.close()
        else:
            plt.show()


        # scatter plot
        plt.figure(figsize=(4, 4))
        # plt.grid(color="gray", zorder=-1000)
        plt.gca().set_axisbelow(True)
        plt.scatter(ds_post_nrem_clean, nrem_dec_clean, c="#00A79D", edgecolors="white")
        slope, intercept, r, p, stderr = scipy.stats.linregress(ds_post_nrem_clean, nrem_dec_clean)
        plt.plot(ds_post_nrem_clean, intercept + slope * ds_post_nrem_clean, color="#00A79D",
                 label="R="+str(np.round(pearsonr(ds_post_nrem_clean, nrem_dec_clean)[0], 2))+"\n p="+str(pearsonr(ds_post_nrem_clean, nrem_dec_clean)[1]))
        plt.xlabel("log-likelihood NREM")
        plt.ylabel("firing probability NREM")
        # plt.text(-0.5, 0.22, "NREM")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.hlines(0, x_min, x_max, color="lightgrey", zorder=-1000)
        plt.vlines(0, y_min, y_max, color="lightgrey", zorder=-1000)
        plt.legend(loc=1)
        plt.title("Recall: NREM")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "log_likeli_post_fir_dec_nrem_all_sessions.svg"), transparent="True")
            plt.close()
        else:
            plt.show()


        # decreasing cells: REM
        # --------------------------------------------------------------------------------------------------------------
        # R values

        plt.style.use('default')
        plt.figure(figsize=(2.5, 2))
        plt.scatter(np.arange(len(r_pre_rem_dec)), r_pre_rem_dec, color="black")
        plt.ylim(-1,1)
        plt.xticks([0,6], [1,7])
        plt.yticks([-1, 0, 1])
        plt.ylabel("Pearson R")
        plt.xlabel("Session ID")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "log_likeli_pre_fir_dec_rem_all_sessions_r_values.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        plt.figure(figsize=(2.5, 2))
        plt.scatter(np.arange(len(r_post_rem_dec)), r_post_rem_dec, color="black")
        plt.ylim(-1,1)
        plt.xticks([0,6], [1,7])
        plt.yticks([-1, 0, 1])
        plt.ylabel("Pearson R")
        plt.xlabel("Session ID")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "log_likeli_post_fir_dec_rem_all_sessions_r_values.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        # scatter plot
        plt.figure(figsize=(4, 4))
        # plt.grid(color="gray", zorder=-1000)
        plt.gca().set_axisbelow(True)
        plt.scatter(ds_pre_rem_clean, rem_dec_clean, c="#00A79D", edgecolors="white")
        slope, intercept, r, p, stderr = scipy.stats.linregress(ds_pre_rem_clean, rem_dec_clean)
        plt.plot(ds_pre_rem_clean, intercept + slope * ds_pre_rem_clean, color="#00A79D",
                 label="R="+str(np.round(pearsonr(ds_pre_rem_clean, rem_dec_clean)[0], 2))+"\n p="+str(pearsonr(ds_pre_rem_clean, rem_dec_clean)[1]))
        plt.xlabel("log-likelihood REM")
        plt.ylabel("firing probability REM")
        # plt.text(-0.5, 0.22, "NREM")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.hlines(0, x_min, x_max, color="lightgrey", zorder=-1000)
        plt.vlines(0, y_min, y_max, color="lightgrey", zorder=-1000)
        plt.legend(loc=1)
        plt.title("Acquisition: REM")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "log_likeli_pre_fir_dec_rem_all_sessions.svg"), transparent="True")
            plt.close()
        else:
            plt.show()


        # scatter plot
        plt.figure(figsize=(4, 4))
        # plt.grid(color="gray", zorder=-1000)
        plt.gca().set_axisbelow(True)
        plt.scatter(ds_post_rem_clean, rem_dec_clean, c="#00A79D", edgecolors="white")
        slope, intercept, r, p, stderr = scipy.stats.linregress(ds_post_rem_clean, rem_dec_clean)
        plt.plot(ds_post_rem_clean, intercept + slope * ds_post_rem_clean, color="#00A79D",
                 label="R="+str(np.round(pearsonr(ds_post_rem_clean, rem_dec_clean)[0], 2))+"\n p="+str(pearsonr(ds_post_rem_clean, rem_dec_clean)[1]))
        plt.xlabel("log-likelihood REM")
        plt.ylabel("firing probability REM")
        # plt.text(-0.5, 0.22, "NREM")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.hlines(0, x_min, x_max, color="lightgrey", zorder=-1000)
        plt.vlines(0, y_min, y_max, color="lightgrey", zorder=-1000)
        plt.legend(loc=1)
        plt.title("Recall: REM")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "log_likeli_post_fir_dec_rem_all_sessions.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def long_sleep_memory_drift_memory_drift_pre_post_mode_probability(self, save_fig=False):

        data_t0 = []
        data_t_end = []

        if save_fig:
            plt.style.use('default')

        plt.style.use('default')

        fig = plt.figure()
        ax = fig.add_subplot()
        col = ["#011D31", "#042E4C", "#0D3F63", "#1C547C", "#32688E", "#4C7EA1", "#7BA5C3"]
        for i, session in enumerate(self.session_list):
            pre, post = session.long_sleep().memory_drift_pre_post_mode_probability_raster(plotting=False)
            data_t0.append(pre[0])
            data_t_end.append(pre[-1])
            ax.plot(np.linspace(0, 1, pre.shape[0]),pre, linewidth=2, c=col[i], label="session "+str(i))
            ax.plot(np.linspace(0, 1, pre.shape[0]), post, linewidth=1, linestyle="--", color=col[i],
                    label="session " + str(i), zorder=-1000)

        # stats
        p_value_t0 = ttest_1samp(data_t0, 0, alternative="less")[1]
        print("T-test for t=0, data < 0 --> p = " + str(p_value_t0))

        p_value_t_end = ttest_1samp(data_t_end, 0, alternative="greater")[1]
        print("T-test for t_end, data > 0 --> p = " + str(p_value_t_end))

        p_value_t_start_end = ttest_ind(data_t0, data_t_end, 0)[1]
        print("T-test for t_start vs. t_end --> p = " + str(p_value_t_end))

        plt.grid(axis='y')
        plt.xlabel("Normalized duration")
        plt.xlim(-0.01, 1.01)
        # plt.ylim(-0.75, 0.25)
        plt.ylim(0, 1)
        plt.yticks([0, 0.5, 1], ["0", "0.5", "1"])
        plt.grid(axis="y", color="gray")
        plt.ylabel("Reactivation probability Acquisition modes")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "pre_post_mode_reactivation_prob_all_sessions.svg"), transparent="True")
        else:
            plt.show()

    def long_sleep_memory_drift_memory_drift_pre_post_likelhoods_temporal(self, save_fig=False, template_type="phmm"):

        pre_max = []
        post_max = []
        pre = []
        post = []
        sess_name = []

        for i, session in enumerate(self.session_list):
            pre_max_, post_max_, pre_, post_ = \
                session.long_sleep().memory_drift_plot_likelihoods_temporal(plotting=False, template_type=template_type)
            pre_max.append(pre_max_)
            post_max.append(post_max_)
            pre.append(pre_)
            post.append(post_)
            sess_name.append(session.session_name)

        # get min and max values for scaling
        min_max =np.zeros((len(sess_name), 2))

        for sess_id, (pre_dat, post_dat) in enumerate(zip(pre_max, post_max)):
            pre_dat_smooth = moving_average(pre_dat, int(pre_dat.shape[0] / 1.5))
            post_dat_smooth = moving_average(post_dat, int(pre_dat.shape[0] / 1.5))
            min_max[sess_id, :] = np.array([np.min([np.min(pre_dat_smooth), np.min(post_dat_smooth)]),
                                            np.max([np.max(pre_dat_smooth), np.max(post_dat_smooth)])])
        if save_fig:
            plt.style.use('default')
            c="black"
        else:
            c="white"

        pre_first = []
        pre_last = []
        pre_first_norm = []
        pre_last_norm = []

        # plot pre first
        fig = plt.figure(figsize=(5,3))
        gs = fig.add_gridspec(6, 10)
        ax = fig.add_subplot(gs[:, :7])
        # col_map = matplotlib.cm.Set1(np.linspace(0, 1, len(sess_name)))
        col_map = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        for i, (pre_dat, name) in enumerate(zip(pre_max, sess_name)):
            # z-score
            # pre_dat_z = zscore(pre_dat)
            # smooth curve
            pre_dat_smooth = moving_average(pre_dat, int(pre_dat.shape[0] / 1.5))
            pre_first.append(pre_dat_smooth[:100])
            pre_last.append(pre_dat_smooth[-100:])
            # pre_dat_smooth = moving_average(pre_dat, 1000)
            pre_dat_smooth = (pre_dat_smooth - np.min(pre_dat_smooth))/(np.max(pre_dat_smooth)-np.min(pre_dat_smooth))
            pre_first_norm.append(pre_dat_smooth[:100])
            pre_last_norm.append(pre_dat_smooth[-100:])
            # pre_dat_smooth = (pre_dat_smooth - min_max[i,0])/(min_max[i,1]-min_max[i,0])
            ax.plot(np.linspace(0, 1, pre_dat_smooth.shape[0]), pre_dat_smooth, linewidth=2, c=col_map[i],
                    label=str(i))
        ax.legend(ncol=2)
        ax.set_xlabel("Relative time")
        ax.set_ylabel("Max. acquisition \nlikelihood (normalized)")
        ax.set_ylim(-0.1, 1.2)
        ax2 = fig.add_subplot(gs[:, 7:])
        res = [np.hstack(pre_first_norm), np.hstack(pre_last_norm)]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["first n \n values", "last n \n values"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax2.set_yticks([])
        ax2.set_ylim(-0.1, 1.2)

        ax2.hlines(1.05, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.2, 1.05, "***")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "pre_likeli_temporal.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        post_first = []
        post_last = []
        post_first_norm = []
        post_last_norm = []
        # plot pre first
        fig = plt.figure(figsize=(5,3))
        gs = fig.add_gridspec(6, 10)
        ax = fig.add_subplot(gs[:, :7])
        for i, (pre_dat, name) in enumerate(zip(post_max, sess_name)):
            # if name == "mjc163R1L_0114":
            #     pre_dat_smooth = moving_average(pre_dat, 1000)
            # else:
            pre_dat_smooth = moving_average(pre_dat, int(pre_dat.shape[0] / 1.5))
            post_first.append(pre_dat_smooth[:100])
            post_last.append(pre_dat_smooth[-100:])
            # pre_dat_smooth = moving_average(pre_dat, 1000)
            pre_dat_smooth = (pre_dat_smooth - np.min(pre_dat_smooth))/(np.max(pre_dat_smooth)-np.min(pre_dat_smooth))
            # pre_dat_smooth = (pre_dat_smooth - min_max[i,0])/(min_max[i,1]-min_max[i,0])
            post_first_norm.append(pre_dat_smooth[:100])
            post_last_norm.append(pre_dat_smooth[-100:])
            ax.plot(np.linspace(0, 1, pre_dat_smooth.shape[0]), pre_dat_smooth, linewidth=2, c=col_map[i],
                    label=str(i))
        ax.legend(ncol=2)
        ax.set_xlabel("Relative time")
        ax.set_ylabel("Max. recall \nlikelihood (normalized)")
        ax.set_ylim(-0.1, 1.2)
        ax2 = fig.add_subplot(gs[:, 7:])
        res = [np.hstack(post_first_norm), np.hstack(post_last_norm)]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["first n \n values", "last n \n values"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax2.set_yticks([])
        ax2.set_ylim(-0.1, 1.2)
        ax2.hlines(1.05, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.2, 1.05, "***")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "post_likeli_temporal.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        res = [np.hstack(pre_first), np.hstack(post_first), np.hstack(pre_last), np.hstack(post_last)]
        c="white"
        bplot = plt.boxplot(res, positions=[1, 2, 3, 4], patch_artist=True,
                            labels=["pre", "post", "pre", "post"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        # plt.yscale("log")
        plt.show()

        print(mannwhitneyu(res[0], res[1]))
        print(mannwhitneyu(res[2], res[3]))
        print(mannwhitneyu(res[0], res[3]))
        print(mannwhitneyu(res[1], res[3]))

        if save_fig:
            plt.style.use('default')

        plt.style.use('default')

        fig = plt.figure()
        ax = fig.add_subplot()
        col = ["#011D31", "#042E4C", "#0D3F63", "#1C547C", "#32688E", "#4C7EA1", "#7BA5C3"]

        ax.plot(np.linspace(0, 1, pre.shape[0]),pre, linewidth=2, c=col[i], label="session "+str(i))
        ax.plot(np.linspace(0, 1, pre.shape[0]), post, linewidth=1, linestyle="--", color=col[i],
                label="session " + str(i), zorder=-1000)

        # stats
        p_value_t0 = ttest_1samp(data_t0, 0, alternative="less")[1]
        print("T-test for t=0, data < 0 --> p = " + str(p_value_t0))

        p_value_t_end = ttest_1samp(data_t_end, 0, alternative="greater")[1]
        print("T-test for t_end, data > 0 --> p = " + str(p_value_t_end))

        p_value_t_start_end = ttest_ind(data_t0, data_t_end, 0)[1]
        print("T-test for t_start vs. t_end --> p = " + str(p_value_t_end))

        plt.grid(axis='y')
        plt.xlabel("Normalized duration")
        plt.xlim(-0.01, 1.01)
        # plt.ylim(-0.75, 0.25)
        plt.ylim(0, 1)
        plt.yticks([0, 0.5, 1], ["0", "0.5", "1"])
        plt.grid(axis="y", color="gray")
        plt.ylabel("Reactivation probability Acquisition modes")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "pre_post_mode_reactivation_prob_all_sessions.svg"), transparent="True")
        else:
            plt.show()

    def long_sleep_memory_drift_decoded_vs_non_decoded_likelhoods(self, save_fig=False, template_type="phmm"):

        pre = []
        post = []
        pre_post_dec = []
        sess_name =[]
        pre_pre_decoded = []
        post_pre_decoded = []
        pre_post_decoded = []
        post_post_decoded = []
        
        for i, session in enumerate(self.session_list):
            pre_post_, pre_, post_, pre_pre_decoded_, \
                post_pre_decoded_, pre_post_decoded_, post_post_decoded_ = \
                session.long_sleep().memory_drift_combine_sleep_phases_decoded_likelihoods(plotting=False)
            pre_post_dec.append(pre_post_)
            post.append(post_)
            pre.append(pre_)
            pre_pre_decoded.append(pre_pre_decoded_)
            post_pre_decoded.append(post_pre_decoded_)
            pre_post_decoded.append(pre_post_decoded_)
            post_post_decoded.append(post_post_decoded_)
            sess_name.append(session.session_name)

        cmap = matplotlib.cm.get_cmap('Oranges')
        colors_to_plot_pre = cmap(np.linspace(0.1, 1, len(self.session_list)))

        if save_fig:
            plt.style.use('default')
            c="black"
        else:
            c="white"
        # plot likelihoods
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        f=[]
        l=[]
        for i, pp in enumerate(pre_pre_decoded):
            pp_s = uniform_filter1d(pp, int(pp.shape[0]/10))
            f.append(pp_s[:int(pp_s.shape[0]/2)])
            l.append(pp_s[-int(pp_s.shape[0]/2):])
            plt.plot(np.linspace(0,1,pp_s.shape[0]), pp_s, color=colors_to_plot_pre[i])
        plt.xlabel("Normalized duration")
        plt.title("Acquisition")
        plt.ylabel("Decoded state \n Log-likelihood")
        plt.ylim(-46, -29)
        plt.subplot(1,2,2)
        f_r=[]
        l_r=[]
        for i, pp in enumerate(post_post_decoded):
            pp_s = uniform_filter1d(pp, int(pp.shape[0]/10))
            f_r.append(pp_s[:int(pp_s.shape[0]/2)])
            l_r.append(pp_s[-int(pp_s.shape[0]/2):])
            plt.plot(np.linspace(0,1,pp_s.shape[0]), pp_s, color=colors_to_plot_pre[i])
        plt.xlabel("Normalized duration")
        plt.title("Recall")
        plt.gca().set_yticks([])
        plt.ylim(-46, -29)
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "pre_post_decoded_likelihoods.svg"), transparent="True")
            plt.close()
        else:
            plt.show()


        res = [np.hstack(f), np.hstack(l)]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["1st half", "2nd half"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        plt.title("Acquisition")
        plt.ylabel("Log-likelihood")
        plt.tight_layout()
        plt.show()

        res = [np.hstack(f_r), np.hstack(l_r)]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["1st half", "2nd half"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        plt.title("Recall")
        plt.ylabel("Log-likelihood")
        plt.tight_layout()
        plt.show()



        first_vals_pre_pre_decoded = []
        first_vals_post_pre_decoded = []
        first_vals_pre_post_decoded = []
        first_vals_post_post_decoded = []
        last_vals_pre_pre_decoded = []
        last_vals_post_pre_decoded = []
        last_vals_pre_post_decoded = []
        last_vals_post_post_decoded = []

        for pre_pre, post_pre, pre_post, post_post in zip(pre_pre_decoded, post_pre_decoded,
                                                          pre_post_decoded, post_post_decoded):
            first_vals_pre_pre_decoded.append(pre_pre[:int(pre_pre.shape[0]/2)])
            first_vals_post_pre_decoded.append(post_pre[:int(pre_pre.shape[0]/2)])
            first_vals_pre_post_decoded.append(pre_post[:int(pre_post.shape[0]/2)])
            first_vals_post_post_decoded.append(post_post[:int(pre_post.shape[0]/2)])
            last_vals_pre_pre_decoded.append(pre_pre[-int(pre_pre.shape[0]/2):])
            last_vals_post_pre_decoded.append(post_pre[-int(pre_pre.shape[0]/2):])
            last_vals_pre_post_decoded.append(pre_post[-int(pre_post.shape[0]/2):])
            last_vals_post_post_decoded.append(post_post[-int(pre_post.shape[0]/2):])

        # plot acquisition decoded first
        # --------------------------------------------------------------------------------------------------------
        pre_first = []
        pre_last = []

        fig = plt.figure(figsize=(4,3))
        gs = fig.add_gridspec(6, 21)
        ax = fig.add_subplot(gs[:, :15])
        for i, (pre_dat, name) in enumerate(zip(pre, sess_name)):
            pre_dat_s = uniform_filter1d(pre_dat, int(pre_dat.shape[0]/10))
            pre_first.append(pre_dat_s[:int(pre_dat_s.shape[0]/2)])
            pre_last.append(pre_dat_s[-int(pre_dat_s.shape[0]/2):])
            ax.plot(np.linspace(0, 1, pre_dat_s.shape[0]), pre_dat_s, linewidth=2, c=colors_to_plot_pre[i],
                    label=str(i))
        ax.legend(ncol=2)
        ax.set_xlabel("Normalized duration")
        ax.set_ylabel("Decoded state log-likelihood (Acquisition) -\n Max. log-likelihood from Recall")
        ax.set_ylim(0, 12)
        ax2 = fig.add_subplot(gs[:, 16:])
        res = [np.hstack(pre_first), np.hstack(pre_last)]
        bplot = ax2.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["1st half", "2nd half"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax2.set_yticks([])
        ax2.set_ylim(0, 12)
        ax2.hlines(11, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.05, 11, "***", color=c)

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "pre_decoded_likeli_non_decoded.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # do stats
        # --------------------------------------------------------------------------------------------------------------
        p_pre_first = 0
        p_pre_last = 0
        for pre_f, pre_l in zip(pre_first, pre_last):
            p_f = ttest_1samp(pre_f, 0, alternative="greater")[1]
            p_pre_first = np.max([p_f, p_pre_first])
            p_l = ttest_1samp(pre_l, 0, alternative="greater")[1]
            p_pre_last = np.max([p_l, p_pre_last])
        # correct for multiple comparisons using Bonferroni
        p_pre_first /= len(self.session_list)
        p_pre_last  /= len(self.session_list)

        print("Acquisition decoded, first half (> 0):")
        print(p_pre_first)
        print("Acquisition decoded, second half (> 0):")
        print(p_pre_last)


        # plot recall decoded
        # --------------------------------------------------------------------------------------------------------
        post_first = []
        post_last = []

        fig = plt.figure(figsize=(4,3))
        gs = fig.add_gridspec(6, 21)
        ax = fig.add_subplot(gs[:, :15])
        for i, (post_dat, name) in enumerate(zip(post, sess_name)):
            post_dat_s = uniform_filter1d(post_dat, int(post_dat.shape[0]/10))
            post_first.append(post_dat_s[:int(post_dat_s.shape[0]/2)])
            post_last.append(post_dat_s[-int(post_dat_s.shape[0]/2):])
            ax.plot(np.linspace(0, 1, post_dat_s.shape[0]), post_dat_s, linewidth=2, c=colors_to_plot_pre[i],
                    label=str(i))
        ax.legend(ncol=2)
        ax.set_xlabel("Relative time")
        ax.set_ylabel("Decoded state log-likelihood (Acquisition) -\n Max. log-likelihood from Recall")
        ax.set_ylim(0, 12)
        ax2 = fig.add_subplot(gs[:, 16:])
        res = [np.hstack(post_first), np.hstack(post_last)]
        bplot = ax2.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["1st half", "2nd half"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax2.set_yticks([])
        ax2.set_ylim(0, 12)
        ax2.hlines(11, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.05, 11, "***", color=c)

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "post_decoded_likeli_non_decoded.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # do stats
        # --------------------------------------------------------------------------------------------------------------
        p_post_first = 0
        p_post_last = 0
        for post_f, post_l in zip(post_first, post_last):
            p_f = ttest_1samp(post_f, 0, alternative="greater")[1]
        p_post_first = np.max([p_f, p_post_first])
        p_l = ttest_1samp(post_l, 0, alternative="greater")[1]
        p_post_last = np.max([p_l, p_post_last])
        # correct for multiple comparisons using Bonferroni
        p_post_first /= len(self.session_list)
        p_post_last  /= len(self.session_list)

        print("Recall decoded, first half (> 0):")
        print(p_post_first)
        print("Recall decoded, second half (> 0):")
        print(p_post_last)

        # plot pre and post
        # --------------------------------------------------------------------------------------------------------------
        # pre_post_first = []
        # pre_post_last = []
        # fig = plt.figure(figsize=(5,3))
        # gs = fig.add_gridspec(6, 10)
        # ax = fig.add_subplot(gs[:, :7])
        # for i, (pre_dat, name) in enumerate(zip(pre_post_dec, sess_name)):
        #     # if name == "mjc163R1L_0114":
        #     #     pre_dat_smooth = moving_average(pre_dat, 1000)
        #     # else:
        #     pre_post_first.append(pre_dat[:int(pre_dat.shape[0]/2)])
        #     pre_post_last.append(pre_dat[-int(pre_dat.shape[0]/2):])
        #     ax.plot(np.linspace(0, 1, pre_dat.shape[0]), pre_dat, linewidth=2, c=colors_to_plot_pre[i],
        #             label=str(i))
        # ax.legend(ncol=2)
        # ax.set_xlabel("Relative time")
        # ax.set_ylabel("Decoded state log-likelihood -\n Max. log-likelihood from other model")
        # ax.set_ylim(0, 12)
        # ax2 = fig.add_subplot(gs[:, 7:])
        # res = [np.hstack(pre_post_first), np.hstack(pre_post_last)]
        # bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
        #                     labels=["1st half", "2nd half"],
        #                     boxprops=dict(color=c),
        #                     capprops=dict(color=c),
        #                     whiskerprops=dict(color=c),
        #                     flierprops=dict(color=c, markeredgecolor=c),
        #                     medianprops=dict(color=c), showfliers=False
        #                     )
        # plt.xticks(rotation=45)
        # ax2.set_yticks([])
        # ax2.set_ylim(0, 12)
        # ax2.hlines(10, 1,2, color=c)
        # if mannwhitneyu(res[0], res[1])[1] < 0.001:
        #     plt.text(1.2, 10, "***")
        # plt.tight_layout()
        # if save_fig:
        #     plt.rcParams['svg.fonttype'] = 'none'
        #     plt.savefig(os.path.join(save_path, "pre_post_decoded_likeli_non_decoded.svg"), transparent="True")
        #     plt.close()
        # else:
        #     plt.show()

    def long_sleep_memory_drift_decoded_vs_non_decoded_likelhoods_shuffle(self, save_fig=False, load_from_temp=True):

        pre = []
        post = []

        for i, session in enumerate(self.session_list):
            if load_from_temp:
                infile = open("temp_data/shuffled_decoded_non_decoded/" + session.session_name, 'rb')
                results = pickle.load(infile)
                diff_pre_=np.hstack(results["pre_decoded_diff_z_scored"])
                diff_post_=np.hstack(results["post_decoded_diff_z_scored"])
                infile.close()
            else:
                diff_pre_, diff_post_ = \
                    session.long_sleep().memory_drift_combine_sleep_phases_decoded_non_decoded_likelihoods_shuffle(plotting=False)
            pre.append(diff_pre_)
            post.append(diff_post_)

        cmap = matplotlib.cm.get_cmap('Oranges')
        colors_to_plot_pre = cmap(np.linspace(0.1, 1, len(self.session_list)))

        if save_fig:
            plt.style.use('default')
            c="black"
        else:
            c="white"

        # plot acquisition decoded first
        # --------------------------------------------------------------------------------------------------------
        pre_first = []
        pre_last = []

        fig = plt.figure(figsize=(4,3))
        gs = fig.add_gridspec(6, 21)
        ax = fig.add_subplot(gs[:, :15])
        for i, (pre_dat) in enumerate(pre):
            pre_dat_s = uniform_filter1d(pre_dat, int(pre_dat.shape[0]/10))
            pre_first.append(pre_dat_s[:int(pre_dat_s.shape[0]/2)])
            pre_last.append(pre_dat_s[-int(pre_dat_s.shape[0]/2):])
            ax.plot(np.linspace(0, 1, pre_dat_s.shape[0]), pre_dat_s, linewidth=2, c=colors_to_plot_pre[i],
                    label=str(i))
        ax.legend(ncol=2)
        ax.set_xlabel("Normalized duration")
        ax.set_ylabel("Decoded state log-likelihood (Acquisition) -\n Max. log-likelihood from Recall \n(z-scored using shuffle)")
        ax.set_ylim(0, 1.5)
        ax2 = fig.add_subplot(gs[:, 16:])
        res = [np.hstack(pre_first), np.hstack(pre_last)]
        bplot = ax2.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["1st half", "2nd half"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax2.set_yticks([])
        ax2.set_ylim(0, 1.5)
        ax2.hlines(1.2, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.05, 1.2, "***", color=c)

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "pre_decoded_likeli_non_decoded_shuffle.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # do stats
        # --------------------------------------------------------------------------------------------------------------
        p_pre_first = 0
        p_pre_last = 0
        for pre_f, pre_l in zip(pre_first, pre_last):
            p_f = ttest_1samp(pre_f, 0, alternative="greater")[1]
            p_pre_first = np.max([p_f, p_pre_first])
            p_l = ttest_1samp(pre_l, 0, alternative="greater")[1]
            p_pre_last = np.max([p_l, p_pre_last])
        # correct for multiple comparisons using Bonferroni
        p_pre_first /= len(self.session_list)
        p_pre_last  /= len(self.session_list)

        print("Acquisition decoded, first half (> 0):")
        print(p_pre_first)
        print("Acquisition decoded, second half (> 0):")
        print(p_pre_last)


        # plot recall decoded
        # --------------------------------------------------------------------------------------------------------
        post_first = []
        post_last = []

        fig = plt.figure(figsize=(4,3))
        gs = fig.add_gridspec(6, 21)
        ax = fig.add_subplot(gs[:, :15])
        for i, (post_dat) in enumerate(post):
            post_dat_s = uniform_filter1d(post_dat, int(post_dat.shape[0]/10))
            post_first.append(post_dat_s[:int(post_dat_s.shape[0]/2)])
            post_last.append(post_dat_s[-int(post_dat_s.shape[0]/2):])
            ax.plot(np.linspace(0, 1, post_dat_s.shape[0]), post_dat_s, linewidth=2, c=colors_to_plot_pre[i],
                    label=str(i))
        ax.legend(ncol=2)
        ax.set_xlabel("Relative time")
        ax.set_ylabel("Decoded state log-likelihood (Acquisition) -\n Max. log-likelihood from Recall \n(z-scored using shuffle")
        ax.set_ylim(0, 1.5)
        ax2 = fig.add_subplot(gs[:, 16:])
        res = [np.hstack(post_first), np.hstack(post_last)]
        bplot = ax2.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["1st half", "2nd half"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax2.set_yticks([])
        ax2.set_ylim(0, 1.5)
        ax2.hlines(1.2, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.05, 1.2, "***", color=c)

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "post_decoded_likeli_non_decoded_shuffle.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # do stats
        # --------------------------------------------------------------------------------------------------------------
        p_post_first = 0
        p_post_last = 0
        for post_f, post_l in zip(post_first, post_last):
            p_f = ttest_1samp(post_f, 0, alternative="greater")[1]
        p_post_first = np.max([p_f, p_post_first])
        p_l = ttest_1samp(post_l, 0, alternative="greater")[1]
        p_post_last = np.max([p_l, p_post_last])
        # correct for multiple comparisons using Bonferroni
        p_post_first /= len(self.session_list)
        p_post_last  /= len(self.session_list)

        print("Recall decoded, first half (> 0):")
        print(p_post_first)
        print("Recall decoded, second half (> 0):")
        print(p_post_last)

    def long_sleep_memory_drift_decoded_vs_non_decoded_cosine_distance_shuffle(self, save_fig=False, load_from_temp=True):

        pre = []
        post = []


        for i, session in enumerate(self.session_list):
            if load_from_temp:
                infile = open("temp_data/shuffled_decoded_non_decoded_cosine/" + session.session_name, 'rb')
                results = pickle.load(infile)
                diff_pre_=np.hstack(results["pre_decoded_diff_z_scored"])
                diff_post_=np.hstack(results["post_decoded_diff_z_scored"])
                infile.close()
            else:
                diff_pre_, diff_post_ = \
                    session.long_sleep().memory_drift_combine_sleep_phases_decoded_non_decoded_cosine_distance_shuffle(plotting=False)
            pre.append(diff_pre_)
            post.append(diff_post_)

        cmap = matplotlib.cm.get_cmap('Oranges')
        colors_to_plot_pre = cmap(np.linspace(0.1, 1, len(self.session_list)))

        if save_fig:
            plt.style.use('default')
            c="black"
        else:
            c="white"

        # plot acquisition decoded first
        # --------------------------------------------------------------------------------------------------------
        pre_first = []
        pre_last = []

        fig = plt.figure(figsize=(4,3))
        gs = fig.add_gridspec(6, 21)
        ax = fig.add_subplot(gs[:, :15])
        for i, (pre_dat) in enumerate(pre):
            pre_dat = np.squeeze(pre_dat)
            pre_dat_s = uniform_filter1d(pre_dat, int(pre_dat.shape[0]/10))
            pre_first.append(pre_dat_s[:int(pre_dat_s.shape[0]/2)])
            pre_last.append(pre_dat_s[-int(pre_dat_s.shape[0]/2):])
            ax.plot(np.linspace(0, 1, pre_dat_s.shape[0]), pre_dat_s, linewidth=2, c=colors_to_plot_pre[i],
                    label=str(i))
        ax.legend(ncol=2)
        ax.set_xlabel("Normalized duration")
        ax.set_ylabel("Decoded state log-likelihood (Acquisition) -\n Max. log-likelihood from Recall \n(z-scored using shuffle)")
        ax.set_ylim(-1, 3)
        ax2 = fig.add_subplot(gs[:, 16:])
        res = [np.hstack(pre_first), np.hstack(pre_last)]
        bplot = ax2.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["1st half", "2nd half"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax2.set_yticks([])
        ax2.set_ylim(-1, 3)
        ax2.hlines(1.2, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.05, 1.2, "***", color=c)

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "pre_decoded_likeli_non_decoded_cosine_distance_shuffle.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # do stats
        # --------------------------------------------------------------------------------------------------------------
        p_pre_first = 0
        p_pre_last = 0
        for pre_f, pre_l in zip(pre_first, pre_last):
            p_f = ttest_1samp(pre_f, 0, alternative="greater")[1]
            p_pre_first = np.max([p_f, p_pre_first])
            p_l = ttest_1samp(pre_l, 0, alternative="greater")[1]
            p_pre_last = np.max([p_l, p_pre_last])
        # correct for multiple comparisons using Bonferroni
        p_pre_first /= len(self.session_list)
        p_pre_last  /= len(self.session_list)

        print("Acquisition decoded, first half (> 0):")
        print(p_pre_first)
        print("Acquisition decoded, second half (> 0):")
        print(p_pre_last)


        # plot recall decoded
        # --------------------------------------------------------------------------------------------------------
        post_first = []
        post_last = []

        fig = plt.figure(figsize=(4,3))
        gs = fig.add_gridspec(6, 21)
        ax = fig.add_subplot(gs[:, :15])
        for i, (post_dat) in enumerate(post):
            post_dat = np.squeeze(post_dat)
            post_dat_s = uniform_filter1d(post_dat, int(post_dat.shape[0]/10))
            post_first.append(post_dat_s[:int(post_dat_s.shape[0]/2)])
            post_last.append(post_dat_s[-int(post_dat_s.shape[0]/2):])
            ax.plot(np.linspace(0, 1, post_dat_s.shape[0]), post_dat_s, linewidth=2, c=colors_to_plot_pre[i],
                    label=str(i))
        ax.legend(ncol=2)
        ax.set_xlabel("Relative time")
        ax.set_ylabel("Decoded state log-likelihood (Acquisition) -\n Max. log-likelihood from Recall \n(z-scored using shuffle")
        ax.set_ylim(-1, 3)
        ax2 = fig.add_subplot(gs[:, 16:])
        res = [np.hstack(post_first), np.hstack(post_last)]
        bplot = ax2.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["1st half", "2nd half"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax2.set_yticks([])
        ax2.set_ylim(-1, 3)
        ax2.hlines(1.2, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.05, 1.2, "***", color=c)

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "post_decoded_likeli_non_decoded_cosine_distance_shuffle.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # do stats
        # --------------------------------------------------------------------------------------------------------------
        p_post_first = 0
        p_post_last = 0
        for post_f, post_l in zip(post_first, post_last):
            p_f = ttest_1samp(post_f, 0, alternative="greater")[1]
        p_post_first = np.max([p_f, p_post_first])
        p_l = ttest_1samp(post_l, 0, alternative="greater")[1]
        p_post_last = np.max([p_l, p_post_last])
        # correct for multiple comparisons using Bonferroni
        p_post_first /= len(self.session_list)
        p_post_last  /= len(self.session_list)

        print("Recall decoded, first half (> 0):")
        print(p_post_first)
        print("Recall decoded, second half (> 0):")
        print(p_post_last)

    def long_sleep_memory_drift_mean_firing(self, save_fig=False):
        delta_rem = []
        delta_nrem = []
        mean_firing_rem = []
        mean_firing_nrem = []

        for i, session in enumerate(self.session_list):
            d_nrem, mf_nrem, d_rem, mf_rem = session.long_sleep().memory_drift_and_mean_firing(template_type="phmm",
                                                                                               plotting=False)
            delta_rem.append(d_rem)
            delta_nrem.append(d_nrem)
            mean_firing_rem.append(mf_rem)
            mean_firing_nrem.append(mf_nrem)

        # z-score per session
        mean_firing_rem_z = []
        mean_firing_nrem_z = []

        for dr, dnr, mfr, mfnr in zip(delta_rem, delta_nrem, mean_firing_rem, mean_firing_nrem):
            delta_rem_z.append(zscore(dr))
            delta_nrem_z.append(zscore(dnr))
            mean_firing_rem_z.append(zscore(mfr))
            mean_firing_nrem_z.append(zscore(mfnr))

        mean_firing_rem_z = np.hstack(mean_firing_rem_z)
        mean_firing_nrem_z = np.hstack(mean_firing_nrem_z)

        delta_rem = np.hstack(delta_rem)
        delta_nrem = np.hstack(delta_nrem)

        plt.figure(figsize=(7,4))
        if save_fig:
            plt.style.use('default')
        plt.subplot(1,2,1)
        plt.scatter(delta_rem, mean_firing_rem_z, color="red", label="REM")
        plt.ylabel("Mean firing rate \n (z-scored)")
        plt.xlabel("Delta Drift score")
        plt.xlim(-0.7, 0.7)
        plt.text(-0.6, 4, "R="+str(np.round(pearsonr(delta_rem, mean_firing_rem_z)[0], 2)))
        plt.text(-0.6, 3, "p="+str(np.round(pearsonr(delta_rem, mean_firing_rem_z)[1], 2)))
        plt.title("REM")
        plt.subplot(1,2,2)
        plt.scatter(delta_nrem[mean_firing_nrem_z<6], mean_firing_nrem_z[mean_firing_nrem_z<6], color="blue", label="NREM")
        plt.text(-0.6, 2.5, "R="+str(np.round(pearsonr(delta_nrem[~np.isnan(mean_firing_nrem_z)],
                                                       mean_firing_nrem_z[~np.isnan(mean_firing_nrem_z)])[0], 2)))
        plt.text(-0.6, 2, "p="+str(np.round(pearsonr(delta_nrem[~np.isnan(mean_firing_nrem_z)],
                                                     mean_firing_nrem_z[~np.isnan(mean_firing_nrem_z)])[1], 2)))
        plt.xlabel("Delta Drift score")
        plt.xlim(-0.7, 0.7)
        plt.title("NREM")
        # plt.xscale("log")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "memory_drift_vs_mean_firing.svg"), transparent="True")
        else:
            plt.show()

    def long_sleep_memory_drift_nr_ripples(self, save_fig=False):

        delta_nrem = []
        nr_ripples_nrem = []

        for i, session in enumerate(self.session_list):
            d_nrem, mf_nrem = session.long_sleep(data_to_use="ext_eegh").memory_drift_and_ripples(template_type="phmm",
                                                                                                  plotting=False)

            delta_nrem.append(d_nrem)
            nr_ripples_nrem.append(mf_nrem)

        # z-score per session
        nr_ripples_nrem_z = []

        for mfnr in nr_ripples_nrem:
            nr_ripples_nrem_z.append(zscore(mfnr))

        nr_ripples_nrem_z = np.hstack(nr_ripples_nrem_z)

        delta_nrem = np.hstack(delta_nrem)

        plt.figure(figsize=(4,4))
        if save_fig:
            plt.style.use('default')
        plt.scatter(delta_nrem[nr_ripples_nrem_z<5],nr_ripples_nrem_z[nr_ripples_nrem_z<5], color="blue", label="NREM")
        plt.text(-0.6, 2.5, "R="+str(np.round(pearsonr(delta_nrem[~np.isnan(nr_ripples_nrem_z)],
                                                     nr_ripples_nrem_z[~np.isnan(nr_ripples_nrem_z)])[0], 2)))
        plt.text(-0.6, 1.5, "p="+str(np.round(pearsonr(delta_nrem[~np.isnan(nr_ripples_nrem_z)],
                                                     nr_ripples_nrem_z[~np.isnan(nr_ripples_nrem_z)])[1], 2)))
        plt.xlabel("Delta Drift score")
        plt.ylabel("Number of SWRs \n (z-scored)")
        plt.xlim(-0.7, 0.7)
        plt.ylim(-3,3)
        # plt.xscale("log")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "memory_drift_vs_number_of_ripples.svg"), transparent="True")
        else:
            plt.show()

    def long_sleep_memory_drift_memory_drift_pre_post_probability_stability_vs_similarity(self, save_fig=False,
                                                                                          distance_metric="correlation",
                                                                                          thresh_stab=0, parts_div_distance=3):

        min_distance_per_pre_mode = []
        coeff_mode_pre = []
        min_distance_per_post_mode = []
        coeff_mode_post = []

        for i, session in enumerate(self.session_list):
            md_pre, c_pre, md_post, c_post = \
                session.long_sleep().memory_drift_pre_post_mode_probability_stability_vs_similarity(plotting=False,
                                                                                                    distance_metric=
                                                                                                    distance_metric,
                                                                                                    thresh_stab=
                                                                                                    thresh_stab)

            min_distance_per_pre_mode.append(md_pre)
            coeff_mode_pre.append(c_pre)
            min_distance_per_post_mode.append(md_post)
            coeff_mode_post.append(c_post)

        min_distance_per_pre_mode = np.hstack(min_distance_per_pre_mode)
        coeff_mode_pre = np.hstack(coeff_mode_pre)
        min_distance_per_post_mode = np.hstack(min_distance_per_post_mode)
        coeff_mode_post = np.hstack(coeff_mode_post)

        plt.scatter(min_distance_per_pre_mode, coeff_mode_pre)
        plt.text(0.2, -0.07, pearsonr(min_distance_per_pre_mode, coeff_mode_pre))
        plt.xlabel("Min. distance to any POST mode " + distance_metric)
        plt.ylabel("Non-stationarity of reactivation probability")
        plt.show()

        plt.scatter(min_distance_per_post_mode, coeff_mode_post)
        plt.text(0.2, 0.07, pearsonr(min_distance_per_post_mode, coeff_mode_post))
        plt.xlabel("Min. distance to any PRE mode " + distance_metric)
        plt.ylabel("Non-stationarity of reactivation probability")
        plt.show()

        interval_size = (np.max(min_distance_per_pre_mode))/parts_div_distance

        coeff_interval_pre = []
        for interval_id in range(parts_div_distance):
            coeff_interval_pre.append(coeff_mode_pre[np.logical_and(interval_id*interval_size < min_distance_per_pre_mode,
                                                   interval_id*interval_size < min_distance_per_pre_mode)])
        c = "white"
        bplot = plt.boxplot(coeff_interval_pre, positions=np.arange(parts_div_distance), patch_artist=True,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xlabel("Interval of min. distance POST mode correlation")
        plt.title("PRE modes")
        plt.ylabel("Stability of reactivation")
        plt.show()
        print(mannwhitneyu(coeff_interval_pre[0], coeff_interval_pre[-1]))

        interval_size = (np.max(min_distance_per_post_mode))/parts_div_distance

        coeff_interval_post = []
        for interval_id in range(parts_div_distance):
            coeff_interval_post.append(coeff_mode_post[np.logical_and(interval_id*interval_size < min_distance_per_post_mode,
                                                   interval_id*interval_size < min_distance_per_post_mode)])
        c = "white"
        bplot = plt.boxplot(coeff_interval_post, positions=np.arange(parts_div_distance), patch_artist=True,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xlabel("Interval of min. distance PRE mode correlation")
        plt.title("POST modes")
        plt.ylabel("Stability of reactivation")
        plt.show()
        print(mannwhitneyu(coeff_interval_post[0], coeff_interval_post[-1]))

    def long_sleep_firing_rate_changes(self, stats_test="t_test_two_sided", lump_data_together=False,
                                       use_firing_prob=True, save_fig=False, use_only_non_stationary_periods=False,
                                       smoothing=200):
        """
        compares firing rate changes during REM and NREM epochs for decreasing and increasing cells. Applies stat.
        test to see if difference is significant

        :param save_fig: save as .svg
        :type save_fig: bool
        :param use_firing_prob: use firing probability or actualy firing rates
        :type use_firing_prob: bool
        :param lump_data_together: lumping data from different sessions together
        :type lump_data_together: bool
        :param stats_test: which statistical test to use ("t_test", "mwu", "ks", "mwu_two_sided", "anova",
                           "t_test_one_sided")
        :type stats_test: str
        """

        if lump_data_together:
            rem_dec_list = []
            cum_rem_dec = []
            cum_nrem_dec = []
            cum_rem_inc = []
            cum_nrem_inc = []
            nrem_dec_list = []
            rem_inc_list = []
            nrem_inc_list = []
            for i, session in enumerate(self.session_list):
                nrem_d, rem_d, rem_i, nrem_i, _, _, _ = \
                    session.long_sleep().firing_rate_changes(plotting=False, return_p_value=False,
                                                             use_firing_prob=use_firing_prob,
                                                             use_only_non_stationary_periods=
                                                             use_only_non_stationary_periods, smoothing=smoothing)
                rem_dec_list.append(rem_d)
                cum_rem_dec.append(np.sum(rem_d))
                cum_nrem_dec.append(np.sum(nrem_d))
                cum_rem_inc.append(np.mean(rem_i))
                cum_nrem_inc.append(np.mean(nrem_i))
                nrem_dec_list.append(nrem_d)
                rem_inc_list.append(rem_i)
                nrem_inc_list.append(nrem_i)

            rem_dec = np.hstack(rem_dec_list)
            nrem_dec = np.hstack(nrem_dec_list)
            rem_inc = np.hstack(rem_inc_list)
            nrem_inc = np.hstack(nrem_inc_list)

            rem_dec_sorted = np.sort(rem_dec)
            nrem_dec_sorted = np.sort(nrem_dec)
            rem_inc_sorted = np.sort(rem_inc)
            nrem_inc_sorted = np.sort(nrem_inc)

            p_rem_dec = 1. * np.arange(rem_dec_sorted.shape[0]) / (rem_dec_sorted.shape[0] - 1)
            p_nrem_dec = 1. * np.arange(nrem_dec_sorted.shape[0]) / (nrem_dec_sorted.shape[0] - 1)
            # plt.hlines(0.5, -0.02, 0.85, color="gray", linewidth=0.5)
            if save_fig:
                plt.style.use('default')
            plt.figure(figsize=(4, 2))
            plt.plot(rem_dec_sorted, p_rem_dec, color="red", label="REM")
            plt.plot(nrem_dec_sorted, p_nrem_dec, color="blue", label="NREM")
            plt.legend()
            plt.ylabel("cdf")
            plt.xlim(-0.22, 0.22)
            plt.grid(axis="x", color="gray")
            if use_firing_prob:
                plt.xlabel("Delta: Firing prob.")
            else:
                plt.xlabel("Delta: Mean #spikes")
            plt.tight_layout()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "firing_rate_changes_dec_all_sessions.svg"), transparent="True")
                plt.close()
            else:
                plt.title("Decreasing cells \n"+"p-val:"+str(mannwhitneyu(nrem_dec, rem_dec)[1]))
                plt.show()

            plt.figure(figsize=(4, 2))
            p_rem_inc = 1. * np.arange(rem_inc_sorted.shape[0]) / (rem_inc_sorted.shape[0] - 1)
            p_nrem_inc = 1. * np.arange(nrem_inc_sorted.shape[0]) / (nrem_inc_sorted.shape[0] - 1)
            # plt.hlines(0.5, -0.02, 0.85, color="gray", linewidth=0.5)
            plt.plot(rem_inc_sorted, p_rem_inc, color="red", label="REM")
            plt.plot(nrem_inc_sorted, p_nrem_inc, color="blue", label="NREM")
            plt.legend()
            plt.ylabel("cdf")
            plt.grid(axis="x", color="gray")
            if use_firing_prob:
                plt.xlabel("Delta: Firing prob.")
            else:
                plt.xlabel("Delta: Mean #spikes")
            plt.tight_layout()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "firing_rate_changes_inc_all_sessions.svg"), transparent="True")
                plt.close()
            else:
                plt.title(
                    "Increasing cells\n" + "p-val:" + str(mannwhitneyu(nrem_inc, rem_inc)[1]))
                plt.show()

        else:
            p_dec_list = []
            p_inc_list = []
            for i, session in enumerate(self.session_list):
                p_dec, p_inc = session.long_sleep().firing_rate_changes(plotting=False, stats_test=stats_test,
                                                                        return_p_value=True)
                p_dec_list.append(p_dec)
                p_inc_list.append(p_inc)

            # plotting
            # --------------------------------------------------------------------------------------------------------------

            for i, p in enumerate(p_dec_list):
                plt.scatter(i,p)
            plt.title("DECREASING")
            plt.yticks([0.05,0.01, 0.1,1])
            plt.ylabel("p-value "+stats_test)
            plt.grid(axis="y")
            plt.xlabel("SESSION")
            plt.show()

            for i, p in enumerate(p_inc_list):
                plt.scatter(i,p)
            plt.title("INCREASING")
            plt.yticks([0.05,0.01,0.1,1])
            plt.ylabel("p-value "+stats_test)
            plt.grid(axis="y")
            plt.xlabel("SESSION")
            plt.show()

    def long_sleep_firing_rate_changes_neighbouring_epochs(self, save_fig=False, first_type="random", plotting=False):
        """
        Firing rate changes in neighbouring epochs

        :param save_fig: save as .svg
        :type save_fig: bool
        """
        nrem_dec_smooth = []
        rem_dec_smooth = []
        nrem_inc_smooth = []
        rem_inc_smooth = []

        for i, session in enumerate(self.session_list):
            nrem_d, rem_d, nrem_i, rem_i = \
                session.long_sleep().firing_rate_changes_neighbouring_epochs(first_type=first_type)

            nrem_dec_smooth.append(nrem_d)
            rem_dec_smooth.append(rem_d)
            nrem_inc_smooth.append(nrem_i)
            rem_inc_smooth.append(rem_i)

        nrem_dec_smooth = np.hstack(nrem_dec_smooth)
        rem_dec_smooth = np.hstack(rem_dec_smooth)
        nrem_inc_smooth = np.hstack(nrem_inc_smooth)
        rem_inc_smooth = np.hstack(rem_inc_smooth)

        if plotting or save_fig:

            if save_fig:
                plt.style.use('default')
            print(pearsonr(nrem_dec_smooth, rem_dec_smooth))

            print(pearsonr(nrem_inc_smooth, rem_inc_smooth))

            print("Decreasing cells, neighbouring epochs, "+str(pearsonr(nrem_dec_smooth, rem_dec_smooth)))
            slope, intercept, r, p, stderr = scipy.stats.linregress(nrem_dec_smooth, rem_dec_smooth)
            plt.plot(rem_dec_smooth, intercept + slope * rem_dec_smooth,
                     color="turquoise", label="R="+ str(np.round(pearsonr(nrem_dec_smooth, rem_dec_smooth)[0], 2))+
                                              "\n p="+str(pearsonr(nrem_dec_smooth, rem_dec_smooth)[1]))
            plt.scatter(rem_dec_smooth, nrem_dec_smooth, color="lightcyan", edgecolors="paleturquoise")
            plt.xlabel("DELTA firing prob. REM")
            plt.ylabel("DELTA firing prob. NREM")
            if first_type == "nrem":
                plt.title("NREM --> REM: dec")
            elif first_type == "rem":
                plt.title("REM --> NREM: dec")
            else:
                plt.title("Neighbouring Epochs: dec")
            plt.legend()
            y_min, y_max = plt.gca().get_ylim()
            y_lim = np.max(np.abs(np.array([y_min, y_max])))
            plt.vlines(0,-y_lim, y_lim, zorder=-1000, color="gray", linewidth=1)
            plt.ylim(-y_lim, y_lim)
            x_min, x_max = plt.gca().get_xlim()
            x_lim = np.max(np.abs(np.array([x_min, x_max])))
            plt.hlines(0,-x_lim, x_lim, zorder=-1000, color="gray", linewidth=1)
            plt.xlim(-x_lim, x_lim)

            def make_square_axes(ax):
                """Make an axes square in screen units.

                Should be called after plotting.
                """
                ax.set_aspect(1 / ax.get_data_ratio())

            make_square_axes(plt.gca())
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "n_epochs_firing_prob_dec_first_epoch_"+first_type+".svg"), transparent="True")
                plt.close()
            else:
                plt.show()

            print("Increasing cells, neighbouring epochs, "+str(pearsonr(nrem_inc_smooth, rem_inc_smooth)))
            slope, intercept, r, p, stderr = scipy.stats.linregress(nrem_inc_smooth, rem_inc_smooth)
            plt.plot(rem_inc_smooth, intercept + slope * rem_inc_smooth,
                     color="orange", label="R="+str(np.round(pearsonr(nrem_inc_smooth, rem_inc_smooth)[0], 2))+
                                              "\n p="+str(pearsonr(nrem_inc_smooth, rem_inc_smooth)[1]))
            plt.scatter(rem_inc_smooth, nrem_inc_smooth, color="papayawhip", edgecolors="moccasin")
            plt.xlabel("DELTA firing prob. REM")
            plt.ylabel("DELTA firing prob. NREM")
            if first_type == "nrem":
                plt.title("NREM --> REM: inc")
            elif first_type == "rem":
                plt.title("REM --> NREM: inc")
            else:
                plt.title("Neighbouring Epochs: inc")
            plt.legend()
            y_min, y_max = plt.gca().get_ylim()
            y_lim = np.max(np.abs(np.array([y_min, y_max])))
            plt.vlines(0,-y_lim, y_lim, zorder=-1000, color="gray", linewidth=1)
            plt.ylim(-y_lim, y_lim)
            x_min, x_max = plt.gca().get_xlim()
            x_lim = np.max(np.abs(np.array([x_min, x_max])))
            plt.hlines(0,-x_lim, x_lim, zorder=-1000, color="gray", linewidth=1)
            plt.xlim(-x_lim, x_lim)

            make_square_axes(plt.gca())
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "n_epochs_firing_prob_inc_first_epoch_"+first_type+".svg"), transparent="True")
                plt.close()
            else:
                plt.show()

        return pearsonr(nrem_dec_smooth, rem_dec_smooth), pearsonr(nrem_inc_smooth, rem_inc_smooth), \
            nrem_dec_smooth.shape[0]

    def long_sleep_firing_rate_changes_neighbouring_epochs_stats(self):

        pearson_rem_nrem_dec, pearson_rem_nrem_inc, n = \
            self.long_sleep_firing_rate_changes_neighbouring_epochs(save_fig=False, first_type="rem")
        pearson_nrem_rem_dec, pearson_nrem_rem_inc, _\
            = self.long_sleep_firing_rate_changes_neighbouring_epochs(save_fig=False, first_type="nrem")

        # decreasing
        # --------------------------------------------------------------------------------------------------------------

        # apply Fisher's z-transform to R-values
        r_rem_nrem_fish_dec = 0.5*np.log((1+pearson_rem_nrem_dec[0])/(1-pearson_rem_nrem_dec[0]))
        r_nrem_rem_fish_dec = 0.5*np.log((1+pearson_nrem_rem_dec[0])/(1-pearson_nrem_rem_dec[0]))

        std_ = np.sqrt((1/(n-3))+(1/(n-3)))

        z_score_dec = (r_rem_nrem_fish_dec - r_nrem_rem_fish_dec)/std_

        p_value_one_sided_dec = scipy.stats.norm.sf(abs(z_score_dec))
        p_value_two_sided_dec = scipy.stats.norm.sf(abs(z_score_dec)) * 2

        # increasing
        # --------------------------------------------------------------------------------------------------------------
        # apply Fisher's z-transform to R-values
        r_rem_nrem_fish_inc = 0.5*np.log((1+pearson_rem_nrem_inc[0])/(1-pearson_rem_nrem_inc[0]))
        r_nrem_rem_fish_inc = 0.5*np.log((1+pearson_nrem_rem_inc[0])/(1-pearson_nrem_rem_inc[0]))

        std_ = np.sqrt((1/(n-3))+(1/(n-3)))

        z_score_inc = (r_rem_nrem_fish_inc - r_nrem_rem_fish_inc)/std_

        p_value_one_sided_inc = scipy.stats.norm.sf(abs(z_score_inc))
        p_value_two_sided_inc = scipy.stats.norm.sf(abs(z_score_inc)) * 2

    def long_sleep_firing_rate_changes_neighbouring_epochs_same_sleep_phase(self, save_fig=False):
        """
        Firing rate changes in neighbouring epochs

        :param save_fig: save as .svg
        :type save_fig: bool
        """
        nrem_dec_smooth = []
        rem_dec_smooth = []
        nrem_inc_smooth = []
        rem_inc_smooth = []

        for i, session in enumerate(self.session_list):
            nrem_d, rem_d, nrem_i, rem_i = \
                session.long_sleep().firing_rate_changes_neighbouring_epochs()

            nrem_dec_smooth.append(nrem_d)
            rem_dec_smooth.append(rem_d)
            nrem_inc_smooth.append(nrem_i)
            rem_inc_smooth.append(rem_i)

        # compute pairs of neighbouring epochs NREM
        nrem_dec_smooth_neigh_n = []
        nrem_dec_smooth_neigh_n_plus_one = []
        nrem_inc_smooth_neigh_n = []
        nrem_inc_smooth_neigh_n_plus_one = []
        for nrem_dec_smooth_, nrem_inc_smooth_ in zip(nrem_dec_smooth, nrem_inc_smooth):
            nrem_dec_smooth_neigh_n.append(nrem_dec_smooth_[:-1])
            nrem_dec_smooth_neigh_n_plus_one.append(nrem_dec_smooth_[1:])
            nrem_inc_smooth_neigh_n.append(nrem_inc_smooth_[:-1])
            nrem_inc_smooth_neigh_n_plus_one.append(nrem_inc_smooth_[1:])

        nrem_dec_smooth_neigh_n = np.hstack(nrem_dec_smooth_neigh_n)
        nrem_inc_smooth_neigh_n = np.hstack(nrem_inc_smooth_neigh_n)
        nrem_dec_smooth_neigh_n_plus_one = np.hstack(nrem_dec_smooth_neigh_n_plus_one)
        nrem_inc_smooth_neigh_n_plus_one = np.hstack(nrem_inc_smooth_neigh_n_plus_one)

        # compute pairs of neighbouring epochs REM
        rem_dec_smooth_neigh_n = []
        rem_dec_smooth_neigh_n_plus_one = []
        rem_inc_smooth_neigh_n = []
        rem_inc_smooth_neigh_n_plus_one = []
        for rem_dec_smooth_, rem_inc_smooth_ in zip(rem_dec_smooth, rem_inc_smooth):
            rem_dec_smooth_neigh_n.append(rem_dec_smooth_[:-1])
            rem_dec_smooth_neigh_n_plus_one.append(rem_dec_smooth_[1:])
            rem_inc_smooth_neigh_n.append(rem_inc_smooth_[:-1])
            rem_inc_smooth_neigh_n_plus_one.append(rem_inc_smooth_[1:])

        rem_dec_smooth_neigh_n = np.hstack(rem_dec_smooth_neigh_n)
        rem_inc_smooth_neigh_n = np.hstack(rem_inc_smooth_neigh_n)
        rem_dec_smooth_neigh_n_plus_one = np.hstack(rem_dec_smooth_neigh_n_plus_one)
        rem_inc_smooth_neigh_n_plus_one = np.hstack(rem_inc_smooth_neigh_n_plus_one)


        def make_square_axes(ax):
            """Make an axes square in screen units.

            Should be called after plotting.
            """
            ax.set_aspect(1 / ax.get_data_ratio())


        if save_fig:
            plt.style.use('default')

        print("Decreasing cells, neighbouring epochs NREM, "+str(pearsonr(nrem_dec_smooth_neigh_n,
                                                                     nrem_dec_smooth_neigh_n_plus_one)))
        slope, intercept, r, p, stderr = scipy.stats.linregress(nrem_dec_smooth_neigh_n,
                                                                     nrem_dec_smooth_neigh_n_plus_one)
        plt.plot(nrem_dec_smooth_neigh_n, intercept + slope * nrem_dec_smooth_neigh_n,
                 color="turquoise", label="R="+ str(np.round(pearsonr(nrem_dec_smooth_neigh_n,
                                                                      nrem_dec_smooth_neigh_n_plus_one)[0], 2))+"\n"+
                                          "p="+ str(np.round(pearsonr(nrem_dec_smooth_neigh_n,
                                                                      nrem_dec_smooth_neigh_n_plus_one)[1],5))
                 )
        plt.scatter(nrem_dec_smooth_neigh_n, nrem_dec_smooth_neigh_n_plus_one,
                    color="lightcyan", edgecolors="paleturquoise")
        plt.xlabel("DELTA firing prob. NREM n")
        plt.ylabel("DELTA firing prob. NREM n+1")
        plt.title("Decreasing cells (NREM)")
        plt.legend()
        y_min, y_max = plt.gca().get_ylim()
        y_lim = np.max(np.abs(np.array([y_min, y_max])))
        plt.vlines(0,-y_lim, y_lim, zorder=-1000, color="gray", linewidth=1)
        plt.ylim(-y_lim, y_lim)
        x_min, x_max = plt.gca().get_xlim()
        x_lim = np.max(np.abs(np.array([x_min, x_max])))
        plt.hlines(0,-x_lim, x_lim, zorder=-1000, color="gray", linewidth=1)
        plt.xlim(-x_lim, x_lim)

        make_square_axes(plt.gca())
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_prob_dec_n_n_plus_one_NREM.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        print("Increasing cells, neighbouring epochs NREM, "+str(pearsonr(nrem_inc_smooth_neigh_n,
                                                                     nrem_inc_smooth_neigh_n_plus_one)))
        slope, intercept, r, p, stderr = scipy.stats.linregress(nrem_inc_smooth_neigh_n,
                                                                     nrem_inc_smooth_neigh_n_plus_one)
        plt.plot(nrem_inc_smooth_neigh_n, intercept + slope * nrem_inc_smooth_neigh_n,
                 color="orange", label="R="+ str(np.round(pearsonr(nrem_inc_smooth_neigh_n,
                                                                      nrem_inc_smooth_neigh_n_plus_one)[0], 2))+"\n"+
                                          "p="+ str(np.round(pearsonr(nrem_inc_smooth_neigh_n,
                                                                      nrem_inc_smooth_neigh_n_plus_one)[1],5))
                 )
        plt.scatter(nrem_inc_smooth_neigh_n, nrem_inc_smooth_neigh_n_plus_one,
                    color="papayawhip", edgecolors="moccasin")
        plt.xlabel("DELTA firing prob. NREM n")
        plt.ylabel("DELTA firing prob. NREM n+1")
        plt.title("Increasing cells (NREM)")
        plt.legend()
        y_min, y_max = plt.gca().get_ylim()
        y_lim = np.max(np.abs(np.array([y_min, y_max])))
        plt.vlines(0,-y_lim, y_lim, zorder=-1000, color="gray", linewidth=1)
        plt.ylim(-y_lim, y_lim)
        x_min, x_max = plt.gca().get_xlim()
        x_lim = np.max(np.abs(np.array([x_min, x_max])))
        plt.hlines(0,-x_lim, x_lim, zorder=-1000, color="gray", linewidth=1)
        plt.xlim(-x_lim, x_lim)

        make_square_axes(plt.gca())
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_prob_inc_n_n_plus_one_NREM.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # REM
        # --------------------------------------------------------------------------------------------------------------

        print("Decreasing cells, neighbouring epochs REM, "+str(pearsonr(rem_dec_smooth_neigh_n,
                                                                     rem_dec_smooth_neigh_n_plus_one)))
        slope, intercept, r, p, stderr = scipy.stats.linregress(rem_dec_smooth_neigh_n,
                                                                     rem_dec_smooth_neigh_n_plus_one)
        plt.plot(rem_dec_smooth_neigh_n, intercept + slope * rem_dec_smooth_neigh_n,
                 color="turquoise", label="R="+ str(np.round(pearsonr(rem_dec_smooth_neigh_n,
                                                                      rem_dec_smooth_neigh_n_plus_one)[0], 2))+"\n"+
                                          "p="+ str(np.round(pearsonr(rem_dec_smooth_neigh_n,
                                                                      rem_dec_smooth_neigh_n_plus_one)[1],5))
                 )
        plt.scatter(rem_dec_smooth_neigh_n, rem_dec_smooth_neigh_n_plus_one,
                    color="lightcyan", edgecolors="paleturquoise")
        plt.xlabel("DELTA firing prob. REM n")
        plt.ylabel("DELTA firing prob. REM n+1")
        plt.title("Decreasing cells (REM)")
        plt.legend()
        y_min, y_max = plt.gca().get_ylim()
        y_lim = np.max(np.abs(np.array([y_min, y_max])))
        plt.vlines(0,-y_lim, y_lim, zorder=-1000, color="gray", linewidth=1)
        plt.ylim(-y_lim, y_lim)
        x_min, x_max = plt.gca().get_xlim()
        x_lim = np.max(np.abs(np.array([x_min, x_max])))
        plt.hlines(0,-x_lim, x_lim, zorder=-1000, color="gray", linewidth=1)
        plt.xlim(-x_lim, x_lim)

        make_square_axes(plt.gca())
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_prob_dec_n_n_plus_one_REM.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        print("Increasing cells, neighbouring epochs REM, "+str(pearsonr(rem_inc_smooth_neigh_n,
                                                                     rem_inc_smooth_neigh_n_plus_one)))
        slope, intercept, r, p, stderr = scipy.stats.linregress(rem_inc_smooth_neigh_n,
                                                                     rem_inc_smooth_neigh_n_plus_one)
        plt.plot(rem_inc_smooth_neigh_n, intercept + slope * rem_inc_smooth_neigh_n,
                 color="orange", label="R="+ str(np.round(pearsonr(rem_inc_smooth_neigh_n,
                                                                      rem_inc_smooth_neigh_n_plus_one)[0], 2))+"\n"+
                                          "p="+ str(np.round(pearsonr(rem_inc_smooth_neigh_n,
                                                                      rem_inc_smooth_neigh_n_plus_one)[1],5))
                 )
        plt.scatter(rem_inc_smooth_neigh_n, rem_inc_smooth_neigh_n_plus_one,
                    color="papayawhip", edgecolors="moccasin")
        plt.xlabel("DELTA firing prob. REM n")
        plt.ylabel("DELTA firing prob. REM n+1")
        plt.title("Increasing cells (REM)")
        plt.legend()
        y_min, y_max = plt.gca().get_ylim()
        y_lim = np.max(np.abs(np.array([y_min, y_max])))
        plt.vlines(0,-y_lim, y_lim, zorder=-1000, color="gray", linewidth=1)
        plt.ylim(-y_lim, y_lim)
        x_min, x_max = plt.gca().get_xlim()
        x_lim = np.max(np.abs(np.array([x_min, x_max])))
        plt.hlines(0,-x_lim, x_lim, zorder=-1000, color="gray", linewidth=1)
        plt.xlim(-x_lim, x_lim)

        make_square_axes(plt.gca())
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_prob_inc_n_n_plus_one_REM.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def long_sleep_firing_rates_all_cells(self, measure="mean", save_fig=False, chunks_in_min=2):
        """
        Compute firing rates during long sleep

        :param measure: "mean" or "max" firing rates
        :type measure: str
        :param save_fig: save as .svg
        :type save_fig: bool
        :param chunks_in_min: for which chunk size in minutes to compute firing rates
        :type chunks_in_min: float
        """

        firing_rem_stable = []
        firing_rem_dec = []
        firing_rem_inc = []
        firing_sleep_stable = []
        firing_sleep_dec = []
        firing_sleep_inc = []
        firing_nrem_stable = []
        firing_nrem_dec = []
        firing_nrem_inc = []

        for session in self.session_list:

            # get stable cell data
            # ----------------------------------------------------------------------------------------------------------

            sleep_z_stable = \
                session.long_sleep().firing_rate_distributions(plotting=False, measure=measure, cells_to_use="stable",
                                                               chunks_in_min=chunks_in_min, separate_sleep_phases=False)

            rem_z_stable, nrem_z_stable = \
                session.long_sleep().firing_rate_distributions(plotting=False, measure=measure, cells_to_use="stable",
                                                               chunks_in_min=chunks_in_min, separate_sleep_phases=True)

            firing_rem_stable.append(rem_z_stable)
            firing_sleep_stable.append(sleep_z_stable)
            firing_nrem_stable.append(nrem_z_stable)

            # get dec cell data
            # ----------------------------------------------------------------------------------------------------------

            sleep_z_dec = \
                session.long_sleep().firing_rate_distributions(plotting=False, measure=measure, cells_to_use="decreasing",
                                                               chunks_in_min=chunks_in_min, separate_sleep_phases=False)

            rem_z_dec, nrem_z_dec = \
                session.long_sleep().firing_rate_distributions(plotting=False, measure=measure, cells_to_use="decreasing",
                                                               chunks_in_min=chunks_in_min, separate_sleep_phases=True)

            firing_rem_dec.append(rem_z_dec)
            firing_sleep_dec.append(sleep_z_dec)
            firing_nrem_dec.append(nrem_z_dec)

            # get inc cell data
            # ----------------------------------------------------------------------------------------------------------

            sleep_z_inc = \
                session.long_sleep().firing_rate_distributions(plotting=False, measure=measure, cells_to_use="increasing",
                                                               chunks_in_min=chunks_in_min, separate_sleep_phases=False)

            rem_z_inc, nrem_z_inc = \
                session.long_sleep().firing_rate_distributions(plotting=False, measure=measure, cells_to_use="increasing",
                                                               chunks_in_min=chunks_in_min, separate_sleep_phases=True)

            firing_rem_inc.append(rem_z_inc)
            firing_sleep_inc.append(sleep_z_inc)
            firing_nrem_inc.append(nrem_z_inc)

        # combine session data
        firing_sleep_stable = np.hstack(firing_sleep_stable)
        firing_rem_stable = np.hstack(firing_rem_stable)
        firing_nrem_stable = np.hstack(firing_nrem_stable)

        # combine session data
        firing_sleep_dec = np.hstack(firing_sleep_dec)
        firing_rem_dec = np.hstack(firing_rem_dec)
        firing_nrem_dec = np.hstack(firing_nrem_dec)

        firing_sleep_inc = np.hstack(firing_sleep_inc)
        firing_rem_inc = np.hstack(firing_rem_inc)
        firing_nrem_inc = np.hstack(firing_nrem_inc)

        # combine dec and inc
        firing_sleep_unstable = np.hstack((firing_sleep_dec, firing_sleep_inc))
        firing_rem_unstable = np.hstack((firing_rem_dec, firing_rem_inc))
        firing_nrem_unstable = np.hstack((firing_nrem_dec, firing_nrem_inc))

        firing_rem_unstable = firing_rem_unstable[~np.isnan(firing_rem_unstable)]
        firing_nrem_unstable = firing_nrem_unstable[~np.isnan(firing_nrem_unstable)]

        # unstable vs. stable
        # --------------------------------------------------------------------------------------------------------------
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        print("Sleep")
        print(mannwhitneyu(firing_sleep_unstable, firing_sleep_stable))


        y_dat = [firing_sleep_unstable , firing_sleep_stable]
        plt.figure(figsize=(2,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["Unstable", "Stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), whis=[0.001, 99.999], showfliers=False)
        plt.title("Sleep")
        plt.yscale("symlog")
        if measure == "max":
            plt.ylim(-5, 200)
            plt.ylabel("Max firing rate (z-scored)")
        elif measure == "mean":
            plt.ylim(-4, 250)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_rate_" + measure + "_sleep_stable_vs_unstable.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        print("REM")
        print(mannwhitneyu(firing_rem_unstable, firing_rem_stable, alternative="less"))

        y_dat = [firing_rem_unstable , firing_rem_stable]
        plt.figure(figsize=(2,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["Unstable", "Stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), whis=[0.001, 99.999], showfliers=False)
        plt.title("REM")
        plt.yscale("symlog")
        if measure == "max":
            plt.ylim(-5, 200)
            plt.ylabel("Max firing rate (z-scored)")
        elif measure == "mean":
            plt.ylim(-4, 250)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_rate_" + measure + "_rem_stable_vs_unstable.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        print("NREM")
        print(mannwhitneyu(firing_nrem_unstable, firing_nrem_stable, alternative="greater"))

        y_dat = [firing_nrem_unstable , firing_nrem_stable]
        plt.figure(figsize=(2,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["Unstable", "Stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), whis=[0.001, 99.999], showfliers=False)
        plt.title("NREM")
        plt.yscale("symlog")
        if measure == "max":
            plt.ylim(-5, 200)
            plt.ylabel("Max firing rate (z-scored)")
        elif measure == "mean":
            plt.ylim(-4, 250)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_rate_" + measure + "_nrem_stable_vs_unstable.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        # stable vs. dec vs. inc
        # --------------------------------------------------------------------------------------------------------------

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        print("Sleep: inc. vs dec.")
        print(mannwhitneyu(firing_sleep_dec, firing_sleep_inc))
        print("Sleep: inc. vs stable")
        print(mannwhitneyu(firing_sleep_inc, firing_sleep_stable))
        print("Sleep: dec. vs stable")
        print(mannwhitneyu(firing_sleep_dec, firing_sleep_stable))

        y_dat = [firing_sleep_dec, firing_sleep_inc, firing_sleep_stable]
        plt.figure(figsize=(2, 4))
        bplot = plt.boxplot(y_dat, positions=[1, 2, 3], patch_artist=True,
                            labels=["Dec", "Inc", "Stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), whis=[0.001, 99.999], showfliers=False)
        plt.title("Sleep")
        plt.yscale("symlog")
        if measure == "max":
            plt.ylim(-5, 200)
            plt.ylabel("Max firing rate (z-scored)")
        elif measure == "mean":
            plt.ylim(-4, 250)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_rate_" + measure + "_sleep_stable_vs_unstable.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        print("REM")
        print("REM: inc. vs dec.")
        print(mannwhitneyu(firing_rem_dec, firing_rem_inc))
        print("REM: inc. vs stable")
        print(mannwhitneyu(firing_rem_inc, firing_rem_stable))
        print("REM: dec. vs stable")
        print(mannwhitneyu(firing_rem_dec, firing_rem_stable))

        y_dat = [firing_rem_dec, firing_rem_inc, firing_rem_stable]
        plt.figure(figsize=(2,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["Dec", "Inc", "Stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), whis=[0.001, 99.999], showfliers=False)
        plt.title("REM")
        plt.yscale("symlog")
        if measure == "max":
            plt.ylim(-5, 200)
            plt.ylabel("Max firing rate (z-scored)")
        elif measure == "mean":
            plt.ylim(-4, 250)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_rate_" + measure + "_rem_stable_vs_unstable.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

        print("NREM")
        print("NREM: inc. vs dec.")
        print(mannwhitneyu(firing_nrem_dec, firing_nrem_inc))
        print("NREM: inc. vs stable")
        print(mannwhitneyu(firing_nrem_inc, firing_nrem_stable))
        print("NREM: dec. vs stable")
        print(mannwhitneyu(firing_nrem_dec, firing_nrem_stable))

        y_dat = [firing_nrem_dec, firing_nrem_inc , firing_nrem_stable]
        plt.figure(figsize=(2,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["Dec", "Inc", "Stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), whis=[0.001, 99.999], showfliers=False)
        plt.title("NREM")
        plt.yscale("symlog")
        if measure == "max":
            plt.ylim(-5, 200)
            plt.ylabel("Max firing rate (z-scored)")
        elif measure == "mean":
            plt.ylim(-4, 250)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "firing_rate_" + measure + "_nrem_stable_vs_unstable.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

    def long_sleep_temporal_trend_stable_cells(self, save_fig=False):
        """
        Computes the slope of the memory drift (sim. ratio) using all or only stable cells and compares the two slopes.

        :param save_fig: whether to save (True) or show figure (False)
        :type save_fig: bool
        """
        slope_stable_list = []
        slope_all_list = []
        slope_dec_inc_list = []
        for i, session in enumerate(self.session_list):
            slope_all, slope_stable, slope_dec_inc = session.long_sleep().memory_drift_plot_temporal_trend_stable_cells(plotting=False)
            slope_stable_list.append(slope_stable)
            slope_all_list.append(slope_all)
            slope_dec_inc_list.append(slope_dec_inc)

        slope_stable_arr = np.array(slope_stable_list)
        slope_all_arr = np.array(slope_all_list)
        slope_dec_inc_arr = np.array(slope_dec_inc_list)

        # stats test
        print(mannwhitneyu(slope_stable_arr, slope_all_arr, alternative="less"))

        # plotting
        # --------------------------------------------------------------------------------------------------------------
        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(3,4))
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        for session_id, (st,all) in enumerate(zip(slope_stable_arr, slope_all_arr)):
            plt.scatter([1,2], [all,st], label=str(session_id), color=col[session_id], zorder=session_id)
            plt.plot([1,2], [all,st], color=col[session_id], zorder=session_id)
            plt.xticks([1,2], ["All cells", "Only stable cells"])
        plt.yticks([0.0, 0.5, 1])
        plt.xlim([0.9,2.1])
        plt.ylim([-0.3,1.4])
        plt.ylabel("Normalized slope")
        plt.grid(axis="y")
        plt.legend()
        # save or show figure
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "slope_stable.svg"), transparent="True")
        else:
            plt.show()

        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(3,4))
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        for session_id, (st,all) in enumerate(zip(slope_stable_arr, slope_dec_inc_arr)):
            plt.scatter([1,2], [all,st], label=str(session_id), color=col[session_id], zorder=session_id)
            plt.plot([1,2], [all,st], color=col[session_id], zorder=session_id)
            plt.xticks([1,2], ["Dec&Inc cells", "Only stable cells"])
        plt.yticks([0.0, 0.5, 1])
        plt.xlim([0.9,2.1])
        plt.ylim([-0.3,1.4])
        plt.ylabel("Normalized slope")
        plt.grid(axis="y")
        plt.legend()
        # save or show figure
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "slope_stable_dec_inc.svg"), transparent="True")
        else:
            plt.show()

    def long_sleep_temporal_trend_subsets(self, save_fig=False, n_moving_average_pop_vec=10000, nr_parts_to_split_data=1):
        """
        Computes the slope of the memory drift (sim. ratio) using all or only stable cells and compares the two slopes.

        :param save_fig: whether to save (True) or show figure (False)
        :type save_fig: bool
        """
        slope_stable_list = []
        slope_inc_list = []
        slope_dec_list = []
        slope_all_list = []
        for i, session in enumerate(self.session_list):
            slope_stable, slope_dec, slope_inc, slope_all = \
                session.long_sleep().memory_drift_temporal_subsets(plotting=False, n_moving_average_pop_vec=
                n_moving_average_pop_vec, nr_parts_to_split_data=nr_parts_to_split_data)
            slope_stable_list.append(slope_stable)
            slope_inc_list.append(slope_inc)
            slope_dec_list.append(slope_dec)
            slope_all_list.append(slope_all)

        slope_stable_arr = np.array(slope_stable_list)
        slope_inc_arr = np.array(slope_inc_list)
        slope_dec_arr = np.array(slope_dec_list)
        slope_all_arr = np.array(slope_all_list)

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        plt.figure(figsize=(4,3))
        # plt.figure(figsize=(2, 4))
        bplot = plt.boxplot([slope_all_arr, slope_dec_arr, slope_inc_arr, slope_stable_arr], positions=np.arange(4), patch_artist=True,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            labels=["all cells", "decreasing", "increasing", "persistent"],
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["grey", "turquoise", "orange", "#6B345C"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        # plt.grid(color="grey", axis="y")
        degrees = 45
        plt.xticks(rotation=degrees)
        # check significance
        plt.ylim(-0.2, 1.4)
        plt.yticks([0, 0.5, 1])
        plt.ylabel("Normalized slope of sim_ratio")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "slope_stable_dec_inc_boxplot.svg"), transparent="True")
        else:
            plt.show()

        # compute stats --> need to correct for multiple comparisons
        # --------------------------------------------------------------------------------------------------------------

        # all vs. rest
        print("All vs. Dec")
        print(mannwhitneyu(slope_all_arr, slope_dec_arr)[1]*6)

        print("All vs. Inc")
        print(mannwhitneyu(slope_all_arr, slope_inc_arr)[1]*6)

        print("All vs. Stable")
        print(mannwhitneyu(slope_all_arr, slope_stable_arr)[1]*6)

        print("Dec vs. Inc")
        print(mannwhitneyu(slope_dec_arr, slope_inc_arr)[1]*6)

        print("Dec vs. Stable")
        print(mannwhitneyu(slope_dec_arr, slope_stable_arr)[1]*6)

        print("Stable vs. Inc")
        print(mannwhitneyu(slope_inc_arr, slope_stable_arr)[1]*6)

        # stats test
        # print(mannwhitneyu(slope_stable_arr, slope_all_arr, alternative="less"))

        # plotting
        # --------------------------------------------------------------------------------------------------------------
        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(3,4))
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        for session_id, (st, dec, inc) in enumerate(zip(slope_stable_arr, slope_dec_arr, slope_inc_arr)):
            plt.scatter([1,2,3], [dec,st, inc], label=str(session_id), color=col[session_id], zorder=session_id)
            plt.plot([1,2,3], [dec,st, inc], color=col[session_id], zorder=session_id)
            plt.xticks([1,2,3], ["Dec", "Stable", "Inc"])
        plt.yticks([0.0, 0.5, 1])
        plt.xlim([0.9,3.1])
        plt.ylim([-0.3,1.4])
        plt.ylabel("Normalized slope")
        plt.grid(axis="y")
        plt.legend()
        plt.tight_layout()
        # save or show figure
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "slope_subsets.svg"), transparent="True")
        else:
            plt.show()

        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(3,4))
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        for session_id, (st,all) in enumerate(zip(slope_stable_arr, slope_dec_inc_arr)):
            plt.scatter([1,2], [all,st], label=str(session_id), color=col[session_id], zorder=session_id)
            plt.plot([1,2], [all,st], color=col[session_id], zorder=session_id)
            plt.xticks([1,2], ["Dec&Inc cells", "Only stable cells"])
        plt.yticks([0.0, 0.5, 1])
        plt.xlim([0.9,2.1])
        plt.ylim([-0.3,1.4])
        plt.ylabel("Normalized slope")
        plt.grid(axis="y")
        plt.legend()
        # save or show figure
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "slope_stable_dec_inc.svg"), transparent="True")
        else:
            plt.show()

    def long_sleep_nrem_rem_likelihoods(self, template_type="phmm", save_fig=False):
        """
        compares the likelihoods from decoding for REM and NREM

        :param template_type: which template to use "phmm" or "ising"
        :type template_type: str
        :param save_fig: save as .svg
        :type save_fig: bool
        """
        pre_prob_rem_max = []
        pre_prob_nrem_max = []
        pre_prob_rem_flat = []
        pre_prob_nrem_flat = []
        pre_max_posterior_rem = []
        pre_max_posterior_nrem = []
        for i, session in enumerate(self.session_list):
            rem_max, nrem_max, rem_flat, nrem_flat, nrem_max_posterior, rem_max_posterior = \
                session.long_sleep().memory_drift_rem_nrem_likelihoods(plotting=False, template_type=template_type)

            pre_max_posterior_rem.append(rem_max_posterior)
            pre_max_posterior_nrem.append(nrem_max_posterior)
            pre_prob_rem_max.append(rem_max)
            pre_prob_nrem_max.append(nrem_max)
            pre_prob_rem_flat.append(rem_flat)
            pre_prob_nrem_flat.append(nrem_flat)

        pre_prob_rem_max = np.hstack(pre_prob_rem_max)
        pre_prob_nrem_max = np.hstack(pre_prob_nrem_max)
        pre_prob_rem_flat = np.hstack(pre_prob_rem_flat)
        pre_prob_nrem_flat = np.hstack(pre_prob_nrem_flat)
        pre_max_posterior_rem = np.hstack(pre_max_posterior_rem)
        pre_max_posterior_nrem = np.hstack(pre_max_posterior_nrem)

        p_mwu = mannwhitneyu(pre_prob_rem_max, pre_prob_nrem_max, alternative="greater")
        print("Max. likelihoods, MWU-test: p-value = " + str(p_mwu))

        p_mwu = mannwhitneyu(pre_max_posterior_rem, pre_max_posterior_nrem)
        print("Max. posterior prob., MWU-test: p-value = " + str(p_mwu))

        pre_prob_rem_max_sorted = np.sort(pre_prob_rem_max)
        pre_prob_nrem_max_sorted = np.sort(pre_prob_nrem_max)

        p_rem = 1. * np.arange(pre_prob_rem_max.shape[0]) / (pre_prob_rem_max.shape[0] - 1)
        p_nrem = 1. * np.arange(pre_prob_nrem_max.shape[0]) / (pre_prob_nrem_max.shape[0] - 1)
        if save_fig:
            plt.style.use('default')
            plt.close()
        plt.plot(pre_prob_rem_max_sorted, p_rem, color="red", label="REM")
        plt.plot(pre_prob_nrem_max_sorted, p_nrem, color="blue", label="NREM")
        plt.gca().set_xscale("log")
        plt.xlabel("max. likelihood per PV")
        plt.ylabel("CDF")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "decoding_all_sessions_max_likelihoods.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        pre_prob_rem_flat_sorted = np.sort(pre_prob_rem_flat)
        pre_prob_nrem_flat_sorted = np.sort(pre_prob_nrem_flat)

        p_rem_flat = 1. * np.arange(pre_prob_rem_flat.shape[0]) / (pre_prob_rem_flat.shape[0] - 1)
        p_nrem_flat = 1. * np.arange(pre_prob_nrem_flat.shape[0]) / (pre_prob_nrem_flat.shape[0] - 1)
        plt.plot(pre_prob_rem_flat_sorted, p_rem_flat, color="red", label="REM")
        plt.plot(pre_prob_nrem_flat_sorted, p_nrem_flat, color="blue", label="NREM")
        plt.gca().set_xscale("log")
        plt.xlabel("Likelihoods per PV")
        plt.ylabel("CDF")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "decoding_all_sessions_likelihoods.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        pre_max_posterior_rem_sorted = np.sort(pre_max_posterior_rem)
        pre_max_posterior_nrem_sorted = np.sort(pre_max_posterior_nrem)

        p_rem_flat = 1. * np.arange(pre_max_posterior_rem.shape[0]) / (pre_max_posterior_rem.shape[0] - 1)
        p_nrem_flat = 1. * np.arange(pre_max_posterior_nrem.shape[0]) / (pre_max_posterior_nrem.shape[0] - 1)
        plt.plot(pre_max_posterior_rem_sorted, p_rem_flat, color="red", label="REM")
        plt.plot(pre_max_posterior_nrem_sorted, p_nrem_flat, color="blue", label="NREM")
        # plt.gca().set_xscale("log")
        plt.xlabel("Max. posterior probability per PV")
        plt.ylabel("CDF")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "decoding_all_sessions_max_posterior_prob.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def long_sleep_nrem_rem_decoding_similarity(self, template_type="phmm", save_fig=False, pre_or_post="pre"):
        """
        compares the likelihoods from decoding for REM and NREM

        :param template_type: which template to use "phmm" or "ising"
        :param pre_or_post: use "pre" or "post"
        :type pre_or_post: str
        :param save_fig: save as .svg
        :type save_fig: bool
        """
        pre_rem_mode_freq_norm = []
        pre_nrem_mode_freq_norm = []
        pre_nrem_mode_freq_norm_odd = []
        pre_nrem_mode_freq_norm_even = []
        pre_rem_mode_freq_norm_odd = []
        pre_rem_mode_freq_norm_even = []

        for i, session in enumerate(self.session_list):
            pre_rem, pre_nrem, pre_rem_odd, pre_rem_even, pre_nrem_odd, pre_nrem_even = \
                session.long_sleep().memory_drift_rem_nrem_decoding_similarity(plotting=False,
                                                                               template_type=template_type,
                                                                               pre_or_post=pre_or_post)

            pre_rem_mode_freq_norm.append(pre_rem)
            pre_nrem_mode_freq_norm.append(pre_nrem)
            pre_nrem_mode_freq_norm_odd.append(pre_rem_odd)
            pre_nrem_mode_freq_norm_even.append(pre_rem_even)
            pre_rem_mode_freq_norm_odd.append(pre_nrem_odd)
            pre_rem_mode_freq_norm_even.append(pre_nrem_even)

        pre_rem_mode_freq_norm = np.hstack(pre_rem_mode_freq_norm)
        pre_nrem_mode_freq_norm = np.hstack(pre_nrem_mode_freq_norm)
        # pre_nrem_mode_freq_norm_odd = np.hstack(pre_nrem_mode_freq_norm_odd)
        # pre_nrem_mode_freq_norm_even = np.hstack(pre_nrem_mode_freq_norm_even)
        # pre_rem_mode_freq_norm_odd = np.hstack(pre_rem_mode_freq_norm_odd)
        # pre_rem_mode_freq_norm_even = np.hstack(pre_rem_mode_freq_norm_even)

        diff = (pre_nrem_mode_freq_norm - pre_rem_mode_freq_norm)
        diff_sh = []
        nr_shuffle = 500
        for i in range(nr_shuffle):
            diff_sh.append((pre_nrem_mode_freq_norm - np.random.permutation(pre_rem_mode_freq_norm)))
        diff_sh = np.hstack(diff_sh)
        diff_sh = np.abs(diff_sh)
        diff = np.abs(diff)
        p_diff = 1. * np.arange(diff.shape[0]) / (diff.shape[0] - 1)
        p_diff_shuffle = 1. * np.arange(diff_sh.shape[0]) / (diff_sh.shape[0] - 1)

        def make_square_axes(ax):
            """Make an axes square in screen units.

            Should be called after plotting.
            """
            ax.set_aspect(1 / ax.get_data_ratio())
        if save_fig:
            plt.style.use('default')
        plt.scatter(pre_rem_mode_freq_norm, pre_nrem_mode_freq_norm, edgecolors="darkgray", facecolor="dimgrey",
                    linewidths=0.5)
        if template_type == "phmm":
            # plt.text(0, 0.6, "Spearman corr. = " + str(np.round(spearmanr(pre_rem_mode_freq_norm,
            #                                                              pre_nrem_mode_freq_norm)[0],4)))
            plt.text(0, 0.55, "R = " + str(np.round(pearsonr(pre_rem_mode_freq_norm,pre_nrem_mode_freq_norm)[0],4)))
            plt.text(0, 0.5, "p = " + str(pearsonr(pre_rem_mode_freq_norm,pre_nrem_mode_freq_norm)[1]))
        elif template_type == "ising":
            # plt.text(0, 0.07, "Spearman corr. = " + str(np.round(spearmanr(pre_rem_mode_freq_norm,
            #                                                               pre_nrem_mode_freq_norm)[0],4)))
            plt.text(0, 0.065, "R = " + str(np.round(pearsonr(pre_rem_mode_freq_norm,pre_nrem_mode_freq_norm)[0],4)))
            plt.text(0, 0.06, "p = " + str(pearsonr(pre_rem_mode_freq_norm,pre_nrem_mode_freq_norm)[1]))
        make_square_axes(plt.gca())
        if template_type == "phmm":
            plt.xlabel("mode decoding probability - REM")
            plt.ylabel("mode decoding probability - NREM")
        elif template_type == "ising":
            plt.xlabel("spatial bin decoding probability - REM")
            plt.ylabel("spatial bin decoding probability - NREM")
            plt.xlim(-0.005, 0.07)
            plt.ylim(-0.005, 0.08)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            if template_type == "phmm":
                plt.savefig(os.path.join(save_path, "mode_decoding_prob.svg"), transparent="True")
            elif template_type == "ising":
                plt.savefig(os.path.join(save_path, "spatial_bin_decoding_prob.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def long_sleep_nrem_rem_decoding_similarity_ising_and_phmm(self, save_fig=False):
        """
        compares the likelihoods from decoding for REM and NREM

        :param template_type: which template to use "phmm" or "ising"
        :param save_fig: save as .svg
        :type save_fig: bool
        """
        r_phmm = []
        r_ising = []

        for i, session in enumerate(self.session_list):
            pre_rem_phmm, pre_nrem_phmm, _, _, _, _ = \
                session.long_sleep().memory_drift_rem_nrem_decoding_similarity(plotting=False, template_type="phmm")

            r_phmm.append(pearsonr(pre_rem_phmm, pre_nrem_phmm)[0])

            pre_rem_ising, pre_nrem_ising, _, _, _, _  = \
                session.long_sleep().memory_drift_rem_nrem_decoding_similarity(plotting=False, template_type="ising")

            r_ising.append(pearsonr(pre_rem_ising, pre_nrem_ising)[0])

        r_phmm = np.array(r_phmm)
        r_ising = np.array(r_ising)

        y_dat = [r_phmm, r_ising]

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        plt.figure(figsize=(3,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["pHMM", "Bayesian"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),
                            )
        colors = ["darksalmon", 'bisque']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Pearson R")
        plt.ylim(0,1)
        plt.grid(color="grey", axis="y")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "r_nrem_rem_similarity.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def long_sleep_nrem_rem_decoding_cleanliness_per_mode(self, template_type="phmm", save_fig=False,
                                                          control_data="rem"):
        """
        compares the likelihoods from decoding for REM and NREM

        :param template_type: which template to use "phmm" or "ising"
        :param control_data: which data to use for control ("nrem" or "rem")
        :type control_data: str
        :param save_fig: save as .svg
        :type save_fig: bool
        """
        per_mode_ratio = []
        per_mode_ratio_sign = []
        mean_post_prob_rem_significant = []
        mean_post_prob_nrem_significant = []
        post_prob_rem_significant = []
        post_prob_nrem_significant = []
        above_one = []
        above_one_sign = []
        nr_sign = []
        nr_modes = []

        for i, session in enumerate(self.session_list):
            pmr_sign, pmr, m_rem, m_nrem, pp_rem, pp_nrem, nr_sign_, nr_modes_ = \
                session.long_sleep().memory_drift_rem_nrem_decoding_cleanliness_per_mode(plotting=False,
                                                                                          template_type=template_type,
                                                                                          control_data=control_data)
            post_prob_rem_significant.append(pp_rem)
            post_prob_nrem_significant.append(pp_nrem)
            per_mode_ratio_sign.append(pmr_sign)
            per_mode_ratio.append(pmr)
            above_one.append(np.count_nonzero(pmr>1)/pmr.shape[0])
            above_one_sign.append(np.count_nonzero(pmr_sign > 1) / pmr_sign.shape[0])
            mean_post_prob_rem_significant.append(m_rem)
            mean_post_prob_nrem_significant.append(m_nrem)
            nr_sign.append(nr_sign_)
            nr_modes.append(nr_modes_)

        per_mode_ratio = np.hstack(per_mode_ratio)
        per_mode_ratio_sign = np.hstack(per_mode_ratio_sign)

        # check if significant number of modes is differentially expresed
        for n_mode, n_sign in zip(nr_modes, nr_sign):
            print(binom_test(x=n_sign, n=n_mode, p=0.5))

        nr_all_modes = np.sum(nr_modes)
        nr_all_sign_modes = np.sum(nr_sign)
        print("All sessions together:")
        print(binom_test(x=nr_all_sign_modes, n=nr_all_modes, p=0.5))

        mean_post_prob_rem_significant = np.hstack(mean_post_prob_rem_significant)
        mean_post_prob_nrem_significant = np.hstack(mean_post_prob_nrem_significant)

        post_prob_rem_significant = np.hstack(post_prob_rem_significant)
        post_prob_nrem_significant = np.hstack(post_prob_nrem_significant)

        print("All sessions lumped, only sign. different ones:")
        print(ttest_1samp(per_mode_ratio_sign, popmean=1, alternative="greater"))
        print("All sessions lumped:")
        print(ttest_1samp(per_mode_ratio[~np.isnan(per_mode_ratio)], popmean=1, alternative="greater"))

        print("Ratios, only sign.")
        print(ttest_1samp(above_one_sign, popmean=0.5, alternative="greater"))
        print("Ratios")
        print(ttest_1samp(above_one, popmean=0.5, alternative="greater"))

        print("Raw mean post. probabilites")
        print(mannwhitneyu(mean_post_prob_rem_significant, mean_post_prob_nrem_significant))

        if save_fig:
            plt.style.use('default')
        sns.kdeplot(mean_post_prob_rem_significant, fill=True, color="red", label="REM")
        sns.kdeplot(mean_post_prob_nrem_significant, fill=True, color="blue", label="NREM")
        plt.xlabel("Mean post. probability per mode")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "decoding_cleanliness_per_mode.svg"), transparent="True")
            plt.close()
        else:
            plt.show()


        if save_fig:
            plt.style.use('default')
        sns.kdeplot(mean_post_prob_rem_significant, fill=True, color="red", label="REM")
        sns.kdeplot(mean_post_prob_nrem_significant, fill=True, color="blue", label="NREM")
        plt.xlabel("Mean post. probability per mode")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "decoding_cleanliness_per_mode.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        if save_fig:
            plt.style.use('default')
        g = sns.kdeplot(per_mode_ratio_sign, fill=True)
        max_y = g.viewLim.bounds[3]
        plt.vlines(1,0,max_y)
        plt.xlim(0.45, 1.55)
        plt.ylim(0, max_y)
        plt.xlabel("Post. prob. REM / Post. prob. NREM")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "decoding_cleanliness_per_mode_ratio.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def long_sleep_nrem_rem_decoding_cleanliness(self, template_type="phmm", save_fig=False):
        """
        compares the likelihoods from decoding for REM and NREM

        @param template_type: which template to use "phmm" or "ising"
        :param save_fig: save as .svg
        :type save_fig: bool
        """
        rem_first_second_max_ratio = []
        nrem_first_second_max_ratio = []
        max_prob_rem = []
        max_prob_nrem = []

        for i, session in enumerate(self.session_list):
            rfs, nfs, mr, mn = \
                session.long_sleep().memory_drift_rem_nrem_decoding_cleanliness(plotting=False, template_type=template_type)

            rem_first_second_max_ratio.append(rfs)
            nrem_first_second_max_ratio.append(nfs)
            max_prob_rem.append(mr)
            max_prob_nrem.append(mn)

        rem_first_second_max_ratio = np.hstack(rem_first_second_max_ratio)
        nrem_first_second_max_ratio = np.hstack(nrem_first_second_max_ratio)
        max_prob_rem = np.hstack(max_prob_rem)
        max_prob_nrem = np.hstack(max_prob_nrem)

        p_ratio_rem = 1. * np.arange(rem_first_second_max_ratio.shape[0]) / (rem_first_second_max_ratio.shape[0] - 1)
        p_ratio_nrem = 1. * np.arange(nrem_first_second_max_ratio.shape[0]) / (nrem_first_second_max_ratio.shape[0] - 1)

        p_prob_rem = 1. * np.arange(max_prob_rem.shape[0]) / (max_prob_rem.shape[0] - 1)
        p_prob_nrem = 1. * np.arange(max_prob_nrem.shape[0]) / (max_prob_nrem.shape[0] - 1)

        plt.plot(np.sort(rem_first_second_max_ratio), p_ratio_rem, label="REM")
        plt.plot(np.sort(nrem_first_second_max_ratio), p_ratio_nrem, label="NREM")
        plt.xscale("log")
        plt.legend()
        plt.xlabel("max. likeli / second largest likeli")
        plt.ylabel("cdf")
        plt.show()
        print(mannwhitneyu(rem_first_second_max_ratio, nrem_first_second_max_ratio, alternative="greater"))

        plt.plot(np.sort(max_prob_rem), p_prob_rem, label="REM")
        plt.plot(np.sort(max_prob_nrem), p_prob_nrem, label="NREM")
        plt.ylabel("cdf")
        plt.xlabel("max. prob.")
        plt.legend()
        plt.show()

        print(mannwhitneyu(max_prob_rem, max_prob_nrem, alternative="greater"))

    def long_sleep_nrem_rem_autocorrelation_temporal(self, template_type="phmm", save_fig=False):
        """
        compares the likelihoods from decoding for REM and NREM

        @param template_type: which template to use "phmm" or "ising"
        :param save_fig: save as .svg
        :type save_fig: bool
        """
        rem_exp = []
        nrem_exp = []

        for i, session in enumerate(self.session_list):
            re, nre = \
                session.long_sleep().memory_drift_rem_nrem_autocorrelation_temporal(plotting=False, template_type=template_type,
                                                                           duration_for_autocorrelation_nrem=1,
                                                                           duration_for_autocorrelation_rem=10)

            rem_exp.append(re)
            nrem_exp.append(nre)

        y_dat = np.vstack((np.array(rem_exp), np.array(nrem_exp)))

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        plt.figure(figsize=(3,4))
        bplot = plt.boxplot(y_dat.T, positions=[1, 2], patch_artist=True,
                            labels=["REM", "NREM"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),
                            )
        colors = ["red", 'blue']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Exponential coefficient k")
        plt.grid(color="grey", axis="y")
        plt.yscale("symlog")
        plt.yticks([np.median(rem_exp), -1, -10, np.median(nrem_exp)])

        plt.text(-0.001, np.median(rem_exp), str(np.round(np.median(rem_exp), 2)))
        plt.text(-0.001, np.median(nrem_exp), str(np.round(np.median(nrem_exp), 2)))
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "exponential_coeff.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def long_sleep_interval_nrem_rem(self):
        rem_nrem_interval = []
        nrem_rem_interval = []

        for i, session in enumerate(self.session_list):
            rem_nrem_interval_, nrem_rem_interval_ = \
                session.long_sleep().memory_drift_rem_nrem_merged_event_times_interval_between(plotting=False)

            rem_nrem_interval.append(rem_nrem_interval_)
            nrem_rem_interval.append(nrem_rem_interval_)

        rem_nrem_interval = np.hstack(rem_nrem_interval)
        nrem_rem_interval = np.hstack(nrem_rem_interval)
        print("HERE")
        plt.style.use('default')
        p_rem_nrem_interval = 1. * np.arange(rem_nrem_interval.shape[0]) / (rem_nrem_interval.shape[0] - 1)
        plt.plot(np.sort(rem_nrem_interval), p_rem_nrem_interval, label="rem->nrem interval")
        p_nrem_rem_interval = 1. * np.arange(nrem_rem_interval.shape[0]) / (nrem_rem_interval.shape[0] - 1)
        plt.plot(np.sort(nrem_rem_interval), p_nrem_rem_interval, label="nrem->rem interval")
        plt.legend(loc=4)
        plt.xscale("log")
        plt.text(10, 0.5, "n.s.")
        plt.xlabel("Temporal interval (s)")
        plt.ylabel("CDF")
        plt.show()
        print(mannwhitneyu(rem_nrem_interval, nrem_rem_interval))

        plt.hist(rem_nrem_interval, label="rem->nrem interval", bins=10000)
        plt.hist(nrem_rem_interval, label="nrem->rem interval", bins=10000)
        plt.xscale("log")
        plt.legend()
        plt.show()

    def long_sleep_constant_spike_bin_length(self, save_fig=False):
        """
        compares the spike bin length in rem and nrem

        :param save_fig: save as .svg
        :type save_fig: bool
        """
        nrem_bin_length = []
        rem_bin_length = []

        for i, session in enumerate(self.session_list):
            n_l, r_l = \
                session.long_sleep().get_constant_spike_bin_length(plotting=False)

            nrem_bin_length.append(n_l)
            rem_bin_length.append(r_l)

        nrem_bin_length = np.hstack(nrem_bin_length)
        rem_bin_length = np.hstack(rem_bin_length)

        if save_fig:
            plt.style.use('default')

        p_rem = 1. * np.arange(rem_bin_length.shape[0]) / (rem_bin_length.shape[0] - 1)
        p_nrem = 1. * np.arange(nrem_bin_length.shape[0]) / (nrem_bin_length.shape[0] - 1)
        plt.plot(np.sort(nrem_bin_length), p_nrem, color="blue", label="NREM")
        plt.plot(np.sort(rem_bin_length), p_rem, color="red", label="REM")
        plt.legend()
        plt.xscale("log")
        plt.ylabel("cdf")
        plt.xlabel("12-spike-bin duration (s)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "constant_spike_bin_length.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def long_sleep_memory_drift_nr_swrs(self):

        delta = []
        nr_swrs = []
        nr_swrs_per_s = []
        merged_events_nrem_length_s = []
        delta_wo_outlier = []
        nr_swrs_wo_outlier = []

        for session in self.session_list:
            d, nr, nr_s, dur_s, d_wo_out, nr_s_wo_out = \
                session.long_sleep().memory_drift_and_nr_swrs(template_type="phmm", plotting=False)

            delta.append(d)
            nr_swrs.append(nr)
            nr_swrs_per_s.append(nr_s)
            merged_events_nrem_length_s.append(dur_s)
            delta_wo_outlier.append(d_wo_out)
            nr_swrs_wo_outlier.append(nr_s_wo_out)

        delta_wo_outlier = np.hstack(delta_wo_outlier)
        nr_swrs_wo_outlier = np.hstack(nr_swrs_wo_outlier)
        delta = np.hstack(delta)
        nr_swrs = np.hstack(nr_swrs)
        nr_swrs_per_s = np.hstack(nr_swrs_per_s)
        merged_events_nrem_length_s = np.hstack(merged_events_nrem_length_s)

        plt.scatter(delta_wo_outlier, nr_swrs_wo_outlier, c="gray")
        plt.xlabel("delta_score")
        plt.ylabel("#SWRs/s")
        plt.text(0, 0.4, "R=" + str(np.round(pearsonr(delta_wo_outlier, nr_swrs_wo_outlier)[0], 2)), color="red")
        print(pearsonr(delta_wo_outlier, nr_swrs_wo_outlier))
        plt.title("NREM: All sessions")
        plt.show()


        plt.scatter(merged_events_nrem_length_s, nr_swrs)
        plt.xlabel("Duration epoch (s)")
        plt.ylabel("#SWRs")
        plt.text(0, 1000, "R=" + str(np.round(pearsonr(merged_events_nrem_length_s, nr_swrs)[0], 2)), color="red")
        print(pearsonr(merged_events_nrem_length_s, nr_swrs))
        plt.title("NREM: All sessions")
        plt.show()

        plt.scatter(merged_events_nrem_length_s, delta)
        plt.xlabel("Duration epoch (s)")
        plt.ylabel("Delta score")
        plt.text(1000, 0.5, "R=" + str(np.round(pearsonr(merged_events_nrem_length_s, delta)[0], 2)), color="red")
        print(pearsonr(merged_events_nrem_length_s, delta))
        plt.title("NREM: All sessions")
        plt.show()

        plt.scatter(delta, nr_swrs, c="gray")
        plt.xlabel("delta_score")
        plt.ylabel("#SWRs")
        plt.text(0, 1000, "R=" + str(np.round(pearsonr(delta, nr_swrs)[0], 2)), color="red")
        print(pearsonr(delta, nr_swrs))
        plt.title("NREM: All sessions")
        plt.show()

        plt.scatter(delta, nr_swrs_per_s, c="gray")
        plt.xlabel("delta_score")
        plt.ylabel("#SWRs/s")
        plt.text(0, 2, "R=" + str(np.round(pearsonr(delta, nr_swrs_per_s)[0], 2)), color="red")
        print(pearsonr(delta, nr_swrs_per_s))
        plt.title("NREM: All sessions")
        plt.show()

    def long_sleep_bayesian_decoding_jump_distance(self, save_fig=False):

        rem_distance = []
        nrem_distance = []

        for session in self.session_list:
            rem_dist, nrem_dist = \
                session.long_sleep().bayesian_decoding_jump_size(plotting=False)
            rem_distance.append(rem_dist)
            nrem_distance.append(nrem_dist)

        rem_distance = np.hstack(rem_distance)
        nrem_distance = np.hstack(nrem_distance)
        if save_fig:
            plt.style.use('default')
        plt.subplot(2, 1, 1)
        plt.hist(rem_distance, density=True, bins=20, label="REM", color="red")
        plt.legend()
        plt.xticks([])
        plt.xlim(0, 120)
        plt.ylabel("Density")
        plt.subplot(2, 1, 2)
        plt.hist(nrem_distance, density=True, bins=20, label="NREM", color="blue")
        plt.xlabel("cm")
        plt.xlim(0,120)
        plt.ylabel("Density")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "jump_distance_rem_nrem.svg"), transparent="True")
        else:
            plt.show()

    def memory_drift_and_firing_prob_firing_rate(self):

        mean_firing_rate_rem_dec = []
        rem_dec_clean = []
        mean_firing_rate_nrem_dec = []
        nrem_dec_clean = []

        for session in self.session_list:
            mean_firing_rate_rem_dec_, rem_dec_clean_, mean_firing_rate_nrem_dec_, nrem_dec_clean_ = \
                session.long_sleep().memory_drift_and_firing_prob_firing_rate(template_type="phmm", plotting=False)

            mean_firing_rate_rem_dec.append(mean_firing_rate_rem_dec_)
            rem_dec_clean.append(rem_dec_clean_)
            mean_firing_rate_nrem_dec.append(mean_firing_rate_nrem_dec_)
            nrem_dec_clean.append(nrem_dec_clean_)

        mean_firing_rate_rem_dec = np.hstack(mean_firing_rate_rem_dec)
        rem_dec_clean = np.hstack(rem_dec_clean)
        mean_firing_rate_nrem_dec = np.hstack(mean_firing_rate_nrem_dec)
        nrem_dec_clean = np.hstack(nrem_dec_clean)

        plt.scatter(mean_firing_rate_rem_dec, rem_dec_clean)
        plt.xlabel("Mean firing rate REM (z-scored)")
        plt.ylabel("Change in firing prob REM (z-scored)")
        plt.title("Decreasing cells")
        plt.text(0.75, 0.5, "R=" + str(np.round(pearsonr(mean_firing_rate_rem_dec, rem_dec_clean)[0], 2)),
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=plt.gca().transAxes)
        plt.show()

        plt.scatter(mean_firing_rate_nrem_dec, nrem_dec_clean)
        plt.xlabel("Mean firing rate NREM (z-scored)")
        plt.ylabel("Change in firing prob NREM (z-scored)")
        plt.title("Decreasing cells")
        plt.text(0.45, 0.75, "R=" + str(np.round(pearsonr(mean_firing_rate_nrem_dec, nrem_dec_clean)[0], 2)),
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=plt.gca().transAxes)
        plt.show()

    def long_sleep_memory_drift_similarity_decoded_modes(self, save_fig=False):

        sign_pre_only_dec = []
        sign_post_only_dec = []
        sign_pre_post_only_dec = []
        fraction_pre_on_mode_vec_only_decoded = []
        fraction_post_on_mode_vec_only_decoded = []
        proj_new_pre_only_decoded = []
        proj_new_post_only_decoded = []
        distance_activity_decoded_mode_pre_only_decoded = []
        distance_activity_decoded_mode_post_only_decoded = []

        for session in self.session_list:
            sign_pre_only_dec_, sign_post_only_dec_, sign_pre_post_only_dec_, \
                fraction_pre_on_mode_vec_only_decoded_, fraction_post_on_mode_vec_only_decoded_,\
                proj_new_pre_only_decoded_, proj_new_post_only_decoded_,\
                    distance_activity_decoded_mode_pre_only_decoded_, distance_activity_decoded_mode_post_only_decoded_ = \
                session.long_sleep().memory_drift_similarity_decoded_modes(plotting=False)

            sign_pre_only_dec.append(sign_pre_only_dec_)
            sign_post_only_dec.append(sign_post_only_dec_)
            sign_pre_post_only_dec.append(sign_pre_post_only_dec_)
            fraction_pre_on_mode_vec_only_decoded.append(fraction_pre_on_mode_vec_only_decoded_)
            fraction_post_on_mode_vec_only_decoded.append(fraction_post_on_mode_vec_only_decoded_)
            proj_new_pre_only_decoded.append(proj_new_pre_only_decoded_)
            proj_new_post_only_decoded.append(proj_new_post_only_decoded_)
            distance_activity_decoded_mode_pre_only_decoded.append(distance_activity_decoded_mode_pre_only_decoded_)
            distance_activity_decoded_mode_post_only_decoded.append(distance_activity_decoded_mode_pre_only_decoded_)

        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(4,4))
        cmap = plt.cm.get_cmap("Oranges")
        beg_res = []
        end_res = []
        # need to interpolate results per session to compute mean across sessions
        result_inter = []
        x_for_inter = np.linspace(0, 1, 50000)

        for sess_id, res in enumerate(fraction_post_on_mode_vec_only_decoded):
            smooth_res = moving_average(res, 500)
            end_res.append(np.mean(smooth_res[-10:]))
            beg_res.append(np.mean(smooth_res[:10]))
            # plt.plot(np.linspace(0,1,  smooth_res.shape[0]),
            #          smooth_res, color=cmap(sess_id/len(fraction_post_on_mode_vec_only_decoded)), label=sess_id)
            plt.plot(np.linspace(0,1,  smooth_res.shape[0]),
                     smooth_res, color=cmap(0.1), label=sess_id)
            result_inter.append(np.interp(x_for_inter, np.linspace(0,1,  smooth_res.shape[0]), smooth_res))

            # plt.title(sess_id)
            # plt.ylim(0,1)
            # plt.show()
        result_inter = np.vstack(result_inter)
        result_inter_mean =np.mean(result_inter, axis=0)
        plt.plot(x_for_inter,  result_inter_mean, color="orange")
        plt.xlabel("Normalized duration")
        plt.ylabel("Relative projected distance \n to decoded Recall state")
        plt.ylim(0,1)
        plt.legend()
        # need one sided test --> for POST: gets closer
        plt.text(0, 0.8, ttest_ind(end_res, beg_res, alternative="less")[1])
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "relative_projected_distance_recall_color_invert.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(4,4))
        cmap = plt.cm.get_cmap("Blues")
        beg_res = []
        end_res = []
        result_inter_acq = []
        for sess_id, res in enumerate(fraction_pre_on_mode_vec_only_decoded):
            smooth_res = moving_average(res, 500)
            end_res.append(np.mean(smooth_res[-10:]))
            beg_res.append(np.mean(smooth_res[:10]))
            # plt.plot(np.linspace(0,1,  smooth_res.shape[0]),
            #          smooth_res, color=cmap(sess_id/len(fraction_pre_on_mode_vec_only_decoded)), label=sess_id)
            plt.plot(np.linspace(0,1,  smooth_res.shape[0]),
                     smooth_res, color=cmap(0.1), label=sess_id)
            result_inter_acq.append(np.interp(x_for_inter, np.linspace(0,1,  smooth_res.shape[0]), smooth_res))

        result_inter_acq = np.vstack(result_inter_acq)
        result_inter_acq_mean =np.mean(result_inter_acq, axis=0)
        plt.plot(x_for_inter,  result_inter_acq_mean, color="blue")
        plt.xlabel("Normalized duration")
        plt.ylabel("Relative projected distance \n to decoded Acquisition state")
        plt.legend()
        plt.ylim(0,1)
        # need one sided test --> for POST: gets closer
        plt.text(0, 0.8, ttest_ind(end_res, beg_res, alternative="greater")[1])
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "relative_projected_distance_acquisition_color_invert.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        cmap = plt.cm.get_cmap("Greys")
        plt.figure(figsize=(5,4))
        beg_res = []
        end_res = []
        for sess_id, res in enumerate(sign_pre_post_only_dec):
            smooth_res = moving_average(res, 30)
            end_res.append(np.mean(smooth_res[-10:]))
            beg_res.append(np.mean(smooth_res[:10]))
            plt.plot(np.linspace(0, 1, smooth_res.shape[0]),
                     smooth_res, color=cmap(sess_id / len(sign_pre_post_only_dec)+0.1), label=sess_id)
        plt.xlabel("Normalized duration")
        plt.ylabel("Fraction significant \n reactivations")
        plt.ylim(0, 1.1)
        # need one sided test --> for POST: gets closer
        plt.text(0, 0.8, ttest_ind(end_res, beg_res, alternative="greater")[1])
        plt.legend()
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "fraction_significant_reactivations.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        for res in sign_pre_only_dec:
            plt.plot(res)
        plt.ylabel("Fraction significant")
        plt.xlabel("Window ID (time)")
        plt.title("PRE: only when PRE decoded")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

        for res in sign_post_only_dec:
            plt.plot(res)
        plt.ylabel("Fraction significant")
        plt.xlabel("Window ID (time)")
        plt.title("POST: only when POST decoded")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

        for res in sign_pre_post_only_dec:
            plt.plot(res)
        plt.ylabel("Fraction significant")
        plt.xlabel("Window ID (time)")
        plt.title("PRE&POST: only when \n PRE/POST decoded")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

        for res in distance_activity_decoded_mode_pre_only_decoded:
            plt.plot(moving_average(res,500))
        plt.title("PRE - only when decoded")
        plt.xlabel("Window ID (Time)")
        plt.ylabel("Distance to PRE mode")
        plt.ylim(0,1)
        plt.tight_layout()
        plt.show()

    def long_sleep_memory_drift_nr_significant_nrem_reactivations(self, save_fig=False, window_size_min = 10):

        sign_reactivations = []

        for session in self.session_list:
            s_r = \
                session.long_sleep().memory_drift_nrem_significant_reactivations(plotting=False)

            sign_reactivations.append(s_r)

        sign_reac_initial = []
        sign_reac_end = []

        for reactivations in sign_reactivations:
            sign_reac_initial.append(reactivations[0])
            sign_reac_end.append(reactivations[-1])

        ttest_ind(sign_reac_initial, sign_reac_end)

        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(4, 4))
        cmap = plt.cm.get_cmap("Greys")
        for sess_id, sess_dat in enumerate(sign_reactivations):
            plt.plot(np.linspace(0,1,sess_dat.shape[0]), sess_dat, color=cmap(sess_id/len(sign_reactivations)))
        plt.ylim(0,150)
        plt.xlabel("Normalized duration")
        plt.ylabel("#Sign. SWR reactivations \n / 10 min")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "sign_swr_reactivations.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def long_sleep_memory_drift_nr_swr_per_window(self, save_fig=False, window_size_min = 2):

        swr_per_window = []

        for session in self.session_list:
            s_r = \
                session.long_sleep().swr_frequency_nrem(window_size_min=window_size_min)

            swr_per_window.append(s_r)

        sign_reac_initial = []
        sign_reac_end = []

        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(4, 4))
        cmap = plt.cm.get_cmap("Greys")
        for sess_id, sess_dat in enumerate(swr_per_window):
            sess_dat = moving_average(sess_dat, int(0.1*sess_dat.shape[0]))
            sign_reac_initial.append(sess_dat[0])
            sign_reac_end.append(sess_dat[-1])
            plt.plot(np.linspace(0,1,sess_dat.shape[0]), sess_dat, color=cmap(sess_id/len(swr_per_window)))
        plt.ylim(0, 60)
        plt.xlabel("Normalized duration")
        plt.ylabel("#SWR \n / 2 min")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "swr_per_window_"+str(window_size_min)+"_min.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        print(ttest_ind(sign_reac_initial, sign_reac_end))

    def long_sleep_memory_drift_decoding_quality_across_time(self, save_fig=False):
        sim_z_scored_pre_only_decoded = []
        sim_z_scored_post_only_decoded = []
        sim_z_scored_pre_and_post_only_decoded = []
        diff_pre_decoded = []
        diff_post_decoded = []

        for i, session in enumerate(self.session_list):
            sim_z_scored_pre_only_decoded_, sim_z_scored_post_only_decoded_, sim_z_scored_pre_and_post_only_decoded_, \
                diff_pre, diff_post = session.long_sleep().memory_drift_decoding_quality_across_time()
            sim_z_scored_pre_only_decoded.append(sim_z_scored_pre_only_decoded_)
            sim_z_scored_post_only_decoded.append(sim_z_scored_post_only_decoded_)
            sim_z_scored_pre_and_post_only_decoded.append(sim_z_scored_pre_and_post_only_decoded_)
            diff_pre_decoded.append(diff_pre)
            diff_post_decoded.append(diff_post)

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        cmap = matplotlib.cm.get_cmap('Oranges')
        colors_to_plot_pre = cmap(np.linspace(0.1, 1, len(self.session_list)))

        # plot acquisition decoded first
        # --------------------------------------------------------------------------------------------------------
        pre_first = []
        pre_last = []

        fig = plt.figure(figsize=(4,3))
        gs = fig.add_gridspec(6, 21)
        ax = fig.add_subplot(gs[:, :15])
        for i, dat in enumerate(diff_pre_decoded):
            d_s = uniform_filter1d(dat, int(dat.shape[0]/10))
            pre_first.append(d_s[:int(d_s.shape[0]/2)])
            pre_last.append(d_s[-int(d_s.shape[0]/2):])
            plt.plot(np.linspace(0,1, d_s.shape[0]), d_s, c=colors_to_plot_pre[i])
        ax.set_ylim(0,0.25)
        ax.set_xlabel("Normalized duration")
        ax.set_ylabel("cosine sim decoded state \n - cosine sim non-dec")
        ax2 = fig.add_subplot(gs[:, 16:])
        res = [np.hstack(pre_first), np.hstack(pre_last)]
        bplot = ax2.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["1st half", "2nd half"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax2.set_yticks([])
        ax2.set_ylim(0,0.25)
        ax2.hlines(0.22, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.05, 0.22, "***", color=c)

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "pre_cosine_distance_decoded_non_decoded.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # do stats
        # --------------------------------------------------------------------------------------------------------------
        p_pre_first = 0
        p_pre_last = 0
        for pre_f, pre_l in zip(pre_first, pre_last):
            p_f = ttest_1samp(pre_f, 0, alternative="greater")[1]
            p_pre_first = np.max([p_f, p_pre_first])
            p_l = ttest_1samp(pre_l, 0, alternative="greater")[1]
            p_pre_last = np.max([p_l, p_pre_last])
        # correct for multiple comparisons using Bonferroni
        p_pre_first /= len(self.session_list)
        p_pre_last  /= len(self.session_list)

        print("Acquisition decoded, first half (> 0):")
        print(p_pre_first)
        print("Acquisition decoded, second half (> 0):")
        print(p_pre_last)


        # plot recall decoded
        # --------------------------------------------------------------------------------------------------------
        post_first = []
        post_last = []

        fig = plt.figure(figsize=(4,3))
        gs = fig.add_gridspec(6, 21)
        ax = fig.add_subplot(gs[:, :15])
        for i, dat in enumerate(diff_post_decoded):
            post_dat_s = uniform_filter1d(dat, int(dat.shape[0]/10))
            post_first.append(post_dat_s[:int(post_dat_s.shape[0]/2)])
            post_last.append(post_dat_s[-int(post_dat_s.shape[0]/2):])
            ax.plot(np.linspace(0, 1, post_dat_s.shape[0]), post_dat_s, linewidth=2, c=colors_to_plot_pre[i],
                    label=str(i))
        ax.set_xlabel("Normalized duration")
        ax.set_ylabel("cosine sim decoded state \n - cosine sim non-dec")
        ax.set_ylim(0, 0.25)
        ax2 = fig.add_subplot(gs[:, 16:])
        res = [np.hstack(post_first), np.hstack(post_last)]
        bplot = ax2.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["1st half", "2nd half"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax2.set_yticks([])
        ax2.set_ylim(0, 0.25)
        ax2.hlines(0.22, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.05, 0.22, "***", color=c)

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "post_cosine_distance_decoded_non_decoded.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # do stats
        # --------------------------------------------------------------------------------------------------------------
        p_post_first = 0
        p_post_last = 0
        for post_f, post_l in zip(post_first, post_last):
            p_f = ttest_1samp(post_f, 0, alternative="greater")[1]
        p_post_first = np.max([p_f, p_post_first])
        p_l = ttest_1samp(post_l, 0, alternative="greater")[1]
        p_post_last = np.max([p_l, p_post_last])
        # correct for multiple comparisons using Bonferroni
        p_post_first /= len(self.session_list)
        p_post_last  /= len(self.session_list)

        print("Recall decoded, first half (> 0):")
        print(p_post_first)
        print("Recall decoded, second half (> 0):")
        print(p_post_last)

        ################################################################################################################
        # Cosine similarity using shuffle
        ################################################################################################################
        # plot acquisition decoded first
        # --------------------------------------------------------------------------------------------------------
        pre_first = []
        pre_last = []

        fig = plt.figure(figsize=(4,3))
        gs = fig.add_gridspec(6, 21)
        ax = fig.add_subplot(gs[:, :15])
        for i, dat in enumerate(sim_z_scored_pre_only_decoded):
            d_s = uniform_filter1d(dat, int(dat.shape[0]/10))
            pre_first.append(d_s[:int(d_s.shape[0]/2)])
            pre_last.append(d_s[-int(d_s.shape[0]/2):])
            plt.plot(np.linspace(0,1, d_s.shape[0]), d_s, c=colors_to_plot_pre[i])
        ax.set_ylim(0,6)
        # ax.legend(ncol=2)
        ax.set_xlabel("Normalized duration")
        ax.set_ylabel("Cosine similarity \n with decoded state \n (z-scored using shuffle)")
        ax2 = fig.add_subplot(gs[:, 16:])
        res = [np.hstack(pre_first), np.hstack(pre_last)]
        bplot = ax2.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["1st half", "2nd half"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax2.set_yticks([])
        ax2.set_ylim(0, 6)
        if ttest_1samp(res[0], 2, alternative="greater")[1] < 0.001:
            plt.text(.7, 5, "***", color=c)
        if ttest_1samp(res[1], 2, alternative="greater")[1] < 0.001:
            plt.text(1.8, 5, "***", color=c)

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "pre_cosine_distance_shuffle.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # do stats
        # --------------------------------------------------------------------------------------------------------------
        p_pre_first = 0
        p_pre_last = 0
        for pre_f, pre_l in zip(pre_first, pre_last):
            p_f = ttest_1samp(pre_f, 2, alternative="greater")[1]
            p_pre_first = np.max([p_f, p_pre_first])
            p_l = ttest_1samp(pre_l, 2, alternative="greater")[1]
            p_pre_last = np.max([p_l, p_pre_last])
        # correct for multiple comparisons using Bonferroni
        p_pre_first /= len(self.session_list)
        p_pre_last  /= len(self.session_list)

        print("Acquisition decoded, first half (> 2):")
        print(p_pre_first)
        print("Acquisition decoded, second half (> 2):")
        print(p_pre_last)


        # plot recall decoded
        # --------------------------------------------------------------------------------------------------------
        post_first = []
        post_last = []

        fig = plt.figure(figsize=(4,3))
        gs = fig.add_gridspec(6, 21)
        ax = fig.add_subplot(gs[:, :15])
        for i, dat in enumerate(sim_z_scored_post_only_decoded):
            post_dat_s = uniform_filter1d(dat, int(dat.shape[0]/10))
            post_first.append(post_dat_s[:int(post_dat_s.shape[0]/2)])
            post_last.append(post_dat_s[-int(post_dat_s.shape[0]/2):])
            ax.plot(np.linspace(0, 1, post_dat_s.shape[0]), post_dat_s, linewidth=2, c=colors_to_plot_pre[i],
                    label=str(i))

        ax.set_xlabel("Normalized duration")
        ax.set_ylabel("Cosine similarity \n with decoded state \n (z-scored using shuffle)")
        ax.set_ylim(0, 6)
        ax2 = fig.add_subplot(gs[:, 16:])
        res = [np.hstack(post_first), np.hstack(post_last)]
        bplot = ax2.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["1st half", "2nd half"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax2.set_yticks([])
        ax2.set_ylim(0, 6)
        if ttest_1samp(res[0], 2, alternative="greater")[1] < 0.001:
            plt.text(0.8, 5, "***", color=c)
        if ttest_1samp(res[1], 2, alternative="greater")[1] < 0.001:
            plt.text(1.7, 5, "***", color=c)

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "post_cosine_distance_shuffle.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # do stats
        # --------------------------------------------------------------------------------------------------------------
        p_post_first = 0
        p_post_last = 0
        for post_f, post_l in zip(post_first, post_last):
            p_f = ttest_1samp(post_f, 0, alternative="greater")[1]
        p_post_first = np.max([p_f, p_post_first])
        p_l = ttest_1samp(post_l, 0, alternative="greater")[1]
        p_post_last = np.max([p_l, p_post_last])
        # correct for multiple comparisons using Bonferroni
        p_post_first /= len(self.session_list)
        p_post_last  /= len(self.session_list)

        print("Recall decoded, first half (> 0):")
        print(p_post_first)
        print("Recall decoded, second half (> 0):")
        print(p_post_last)

    def long_sleep_memory_drift_decoding_quality_across_time_likelihoods(self, save_fig=False, load_from_temp=True):
        sim_z_scored_pre_only_decoded = []
        sim_z_scored_post_only_decoded = []

        for i, session in enumerate(self.session_list):
            if load_from_temp:
                infile = open("temp_data/decoding_quality_likelihoods/" + session.session_name, 'rb')
                results = pickle.load(infile)
                sim_z_scored_pre_only_decoded.append(np.hstack(results["pre_decoded_z_scored"]))
                sim_z_scored_post_only_decoded.append(np.hstack(results["post_decoded_z_scored"]))
                infile.close()
            else:
                sim_z_scored_pre_only_decoded_, sim_z_scored_post_only_decoded_= \
                    session.long_sleep().memory_drift_decoding_quality_across_time_likelihoods()
                sim_z_scored_pre_only_decoded.append(sim_z_scored_pre_only_decoded_)
                sim_z_scored_post_only_decoded.append(sim_z_scored_post_only_decoded_)

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        colors_to_plot = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]

        ################################################################################################################
        # Cosine similarity using shuffle
        ################################################################################################################
        # plot acquisition decoded first
        # --------------------------------------------------------------------------------------------------------
        pre_first = []
        pre_last = []

        fig = plt.figure(figsize=(4,3))
        gs = fig.add_gridspec(6, 21)
        ax = fig.add_subplot(gs[:, :15])
        for i, dat in enumerate(sim_z_scored_pre_only_decoded):
            d_s = uniform_filter1d(dat, int(dat.shape[0]/10))
            pre_first.append(d_s[:int(d_s.shape[0]/2)])
            pre_last.append(d_s[-int(d_s.shape[0]/2):])
            plt.plot(np.linspace(0,1, d_s.shape[0]), d_s, c=colors_to_plot[i])
        ax.set_ylim(0,3)
        # ax.legend(ncol=2)
        ax.set_xlabel("Normalized duration")
        ax.set_ylabel("Log-likelihood decoded state \n (z-scored using shuffle)")
        ax2 = fig.add_subplot(gs[:, 16:])
        res = [np.hstack(pre_first), np.hstack(pre_last)]
        bplot = ax2.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["1st half", "2nd half"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax2.set_yticks([])
        ax2.set_ylim(0, 3)
        if ttest_1samp(res[0], 2.5, alternative="greater")[1] < 0.001:
            plt.text(.7, 2, "***", color=c)
        if ttest_1samp(res[1], 2.5, alternative="greater")[1] < 0.001:
            plt.text(1.8, 2, "***", color=c)

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "pre_decoding_quality_likelihoods_shuffle.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # do stats
        # --------------------------------------------------------------------------------------------------------------
        p_pre_first = 0
        p_pre_last = 0
        for pre_f, pre_l in zip(pre_first, pre_last):
            p_f = ttest_1samp(pre_f, 2, alternative="greater")[1]
            p_pre_first = np.max([p_f, p_pre_first])
            p_l = ttest_1samp(pre_l, 2, alternative="greater")[1]
            p_pre_last = np.max([p_l, p_pre_last])
        # correct for multiple comparisons using Bonferroni
        p_pre_first /= len(self.session_list)
        p_pre_last  /= len(self.session_list)

        print("Acquisition decoded, first half (> 2):")
        print(p_pre_first)
        print("Acquisition decoded, second half (> 2):")
        print(p_pre_last)


        # plot recall decoded
        # --------------------------------------------------------------------------------------------------------
        post_first = []
        post_last = []

        fig = plt.figure(figsize=(4,3))
        gs = fig.add_gridspec(6, 21)
        ax = fig.add_subplot(gs[:, :15])
        for i, dat in enumerate(sim_z_scored_post_only_decoded):
            post_dat_s = uniform_filter1d(dat, int(dat.shape[0]/10))
            post_first.append(post_dat_s[:int(post_dat_s.shape[0]/2)])
            post_last.append(post_dat_s[-int(post_dat_s.shape[0]/2):])
            ax.plot(np.linspace(0, 1, post_dat_s.shape[0]), post_dat_s, linewidth=2, c=colors_to_plot[i],
                    label=str(i))

        ax.set_xlabel("Normalized duration")
        ax.set_ylabel("Log-likelihood decoded state \n (z-scored using shuffle)")
        ax.set_ylim(0, 3)
        ax2 = fig.add_subplot(gs[:, 16:])
        res = [np.hstack(post_first), np.hstack(post_last)]
        bplot = ax2.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["1st half", "2nd half"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax2.set_yticks([])
        ax2.set_ylim(0, 3)
        if ttest_1samp(res[0], 2.5, alternative="greater")[1] < 0.001:
            plt.text(0.8, 5, "***", color=c)
        if ttest_1samp(res[1], 2.5, alternative="greater")[1] < 0.001:
            plt.text(1.7, 5, "***", color=c)

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "post_decoding_quality_shuffle.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # do stats
        # --------------------------------------------------------------------------------------------------------------
        p_post_first = 0
        p_post_last = 0
        for post_f, post_l in zip(post_first, post_last):
            p_f = ttest_1samp(post_f, 0, alternative="greater")[1]
        p_post_first = np.max([p_f, p_post_first])
        p_l = ttest_1samp(post_l, 0, alternative="greater")[1]
        p_post_last = np.max([p_l, p_post_last])
        # correct for multiple comparisons using Bonferroni
        p_post_first /= len(self.session_list)
        p_post_last  /= len(self.session_list)

        print("Recall decoded, first half (> 0):")
        print(p_post_first)
        print("Recall decoded, second half (> 0):")
        print(p_post_last)

    def long_sleep_memory_drift_spike_jittering(self, save_fig=False, load_from_temp=True, pre_model="pre",
                                                post_model="post"):

        sim_ratio = []
        sim_ratio_j = []
        diff_pre = []
        diff_post = []
        max_pre = []
        max_pre_jit = []
        max_post = []
        max_post_jit = []

        for i, session in enumerate(self.session_list):
            if load_from_temp:
                if pre_model == "pre" and post_model == "post":
                    infile = open("temp_data/jittered/" + session.session_name, 'rb')
                elif pre_model == "pre_familiar" and post_model == "post_familiar":
                    infile = open("temp_data/jittered/familiar_exploration/" + session.session_name, 'rb')
                results = pickle.load(infile)
                sim_ratio_ = results["sim_ratio"]
                sim_ratio_j_ = results["sim_ratio_j"]
                diff_pre_ = results["diff_pre"]
                diff_post_ = results["diff_post"]
                max_pre_ = results["max_pre"]
                max_pre_jit_ =results["max_pre_jit"]
                max_post_ = results["max_post"]
                max_post_jit_ = results["max_post_jit"]
                infile.close()
            else:
                sim_ratio_, sim_ratio_j_, diff_pre_, diff_post_, max_pre_, max_pre_jit_, max_post_, max_post_jit_ = \
                    session.long_sleep().memory_drift_control_jitter_combine_sleep_phases(plotting=False,
                                                                    pre_model=pre_model, post_model=post_model)
            sim_ratio.append(sim_ratio_)
            sim_ratio_j.append(sim_ratio_j_)
            diff_pre.append(diff_pre_)
            diff_post.append(diff_post_)
            max_pre.append(max_pre_)
            max_pre_jit.append(max_pre_jit_)
            max_post.append(max_post_)
            max_post_jit.append(max_post_jit_)

        # compute slopes of drift score for original and jittered case
        slopes_original = []
        slopes_jittered = []

        for drift_score, drift_score_j in zip(sim_ratio, sim_ratio_j):
            coef_original = np.polyfit(np.linspace(0,1,drift_score.shape[0]), drift_score, 1)
            slopes_original.append(coef_original[0])
            coef_j = np.polyfit(np.linspace(0,1,drift_score_j.shape[0]), drift_score_j, 1)
            slopes_jittered.append(coef_j[0])

        # need to interpolate all differences from different session --> they need to be of the same length to compute
        # mean and std
        # max_bins = np.max([x.shape[0] for x in diff_pre])
        #
        # diff_pre_interp = []
        # diff_post_interp = []
        # diff_pre_interp_init = []
        # diff_pre_interp_last = []
        # diff_post_interp_init = []
        # diff_post_interp_last = []
        #
        # for pre, post in zip(diff_pre, diff_post):
        #     if pre.shape[0] < max_bins:
        #         diff_pre_interp.append(np.interp(np.linspace(0, 1, max_bins), np.linspace(0, 1, pre.shape[0]), pre))
        #         diff_post_interp.append(np.interp(np.linspace(0, 1, max_bins), np.linspace(0, 1, post.shape[0]), post))
        #     else:
        #         diff_pre_interp.append(pre)
        #         diff_post_interp.append(post)
        #
        #     diff_pre_interp_init.append(diff_pre_interp[-1][:100])
        #     diff_pre_interp_last.append(diff_pre_interp[-1][-100:])
        #     diff_post_interp_init.append(diff_post_interp[-1][:100])
        #     diff_post_interp_last.append(diff_post_interp[-1][-100:])
        #
        # diff_pre_interp = np.vstack(diff_pre_interp)
        # diff_post_interp = np.vstack(diff_post_interp)

        # # compute mean and std for plotting
        # diff_pre_interp_mean = np.mean(diff_pre_interp, axis=0)
        # diff_pre_interp_std = np.std(diff_pre_interp, axis=0)
        # diff_post_interp_mean = np.mean(diff_post_interp, axis=0)
        # diff_post_interp_std = np.std(diff_post_interp, axis=0)
        #
        # if save_fig:
        #     plt.style.use('default')
        #     c = "black"
        # else:
        #     c = "white"
        #
        # fig = plt.figure(figsize=(8, 7))
        # gs = fig.add_gridspec(9, 10)
        # ax1 = fig.add_subplot(gs[:4, :7])
        # ax2 = fig.add_subplot(gs[:4, 8:])
        # ax3 = fig.add_subplot(gs[5:, :7])
        # ax4 = fig.add_subplot(gs[5:, 8:])
        #
        # y_pre_max = np.max(diff_pre_interp_mean+diff_pre_interp_std)+0.5
        # y_post_max = np.max(diff_post_interp_mean+diff_post_interp_std)+0.5
        #
        # # ax1.errorbar(np.linspace(0, 1, diff_pre_interp_mean.shape[0]), diff_pre_interp_mean, yerr=diff_pre_interp_std, color="moccasin", zorder=-1000)
        # ax1.plot(np.linspace(0, 1, diff_pre_interp_mean.shape[0]), diff_pre_interp_mean-diff_pre_interp_std, color="moccasin", linewidth=1,
        #          linestyle="--")
        # ax1.plot(np.linspace(0, 1, diff_pre_interp_mean.shape[0]), diff_pre_interp_mean+diff_pre_interp_std, color="moccasin", linewidth=1,
        #          linestyle="--")
        # ax1.plot(np.linspace(0, 1, diff_pre_interp_mean.shape[0]), diff_pre_interp_mean, color="orange", linewidth=2,
        #          label="Acquisition states")
        # # ax1.fill(np.append(np.linspace(0, 1, diff_pre_interp_mean.shape[0]), np.linspace(0, 1, diff_pre_interp_mean.shape[0])[::-1]),
        # #          np.append(diff_pre_interp_mean+diff_pre_interp_std, (diff_pre_interp_mean-diff_pre_interp_std)[::-1]), 'moccasin')
        # ax1.set_ylabel("Log-likelihood_original - \n Log-likelihood_jittered")
        # ax1.legend()
        # ax1.set_ylim(0, y_pre_max)
        #
        # # ax3.errorbar(np.linspace(0, 1, diff_post_interp_mean.shape[0]), diff_post_interp_mean, yerr=diff_post_interp_std, color="lightcyan", zorder=-1000)
        # ax3.plot(np.linspace(0, 1, diff_post_interp_mean.shape[0]), diff_post_interp_mean, color="lightblue", linewidth=2, label="Recall states")
        # ax3.plot(np.linspace(0, 1, diff_post_interp_mean.shape[0]), diff_post_interp_mean-diff_post_interp_std, color="lightcyan",
        #          linewidth=1, linestyle="--")
        # ax3.plot(np.linspace(0, 1, diff_post_interp_mean.shape[0]), diff_post_interp_mean+diff_post_interp_std, color="lightcyan",
        #          linewidth=1, linestyle="--")
        # # ax3.fill(np.append(np.linspace(0, 1, diff_post_interp_mean.shape[0]), np.linspace(0, 1, diff_post_interp_mean.shape[0])[::-1]),
        # #          np.append(diff_post_interp_mean+diff_post_interp_std, (diff_post_interp_mean-diff_post_interp_std)[::-1]), 'lightcyan')
        #
        # ax3.legend()
        # ax3.set_xlabel("Normalized duration")
        # ax3.set_ylabel("Log-likelihood_original - \n Log-likelihood_jittered")
        # ax3.set_ylim(0, y_post_max)
        # res = [np.hstack(diff_pre_interp_init), np.hstack(diff_pre_interp_last)]
        # bplot=ax2.boxplot(res, positions=[1, 2], patch_artist=True,
        #                     labels=["first n\n values", "last n\n values"],
        #                     boxprops=dict(color=c),
        #                     capprops=dict(color=c),
        #                     whiskerprops=dict(color=c),
        #                     flierprops=dict(color=c, markeredgecolor=c),
        #                     medianprops=dict(color=c), showfliers=False)
        # ax2.set_ylim(0, y_pre_max)
        # ax2.tick_params(axis='x', labelrotation=45)
        # ax2.set_yticks([])
        # y_base = y_pre_max-0.2
        # ax2.hlines(y_base, 1, 2)
        # if mannwhitneyu(res[0], res[1])[1] > 0.05:
        #     ax2.text(1.4, y_base, "n.s.")
        # elif mannwhitneyu(res[0], res[1])[1] < 0.001:
        #     ax2.text(1.4, y_base, "***")
        # elif mannwhitneyu(res[0], res[1])[1] < 0.01:
        #     ax2.text(1.4, y_base, "**")
        # elif mannwhitneyu(res[0], res[1])[1] < 0.05:
        #     ax2.text(1.4, y_base, "*")
        #
        # colors = ["moccasin", "moccasin"]
        # for patch, color in zip(bplot['boxes'], colors):
        #     patch.set_facecolor(color)
        #
        # res = [np.hstack(diff_post_interp_init), np.hstack(diff_post_interp_last)]
        # bplot=ax4.boxplot(res, positions=[1, 2], patch_artist=True,
        #                     labels=["first n\n values", "last n\n values"],
        #                     boxprops=dict(color=c),
        #                     capprops=dict(color=c),
        #                     whiskerprops=dict(color=c),
        #                     flierprops=dict(color=c, markeredgecolor=c),
        #                     medianprops=dict(color=c), showfliers=False)
        # ax4.set_ylim(0, y_post_max)
        # ax4.tick_params(axis='x', labelrotation=45)
        # ax4.set_yticks([])
        # y_base = y_post_max - 0.2
        # ax4.hlines(y_base, 1, 2)
        # if mannwhitneyu(res[0], res[1])[1] > 0.05:
        #     ax4.text(1.4, y_base, "n.s.")
        # elif mannwhitneyu(res[0], res[1])[1] < 0.001:
        #     ax4.text(1.4, y_base, "***")
        # elif mannwhitneyu(res[0], res[1])[1] < 0.01:
        #     ax4.text(1.4, y_base, "**")
        # elif mannwhitneyu(res[0], res[1])[1] < 0.05:
        #     ax4.text(1.4, y_base, "*")
        # colors = ["lightcyan", "lightcyan"]
        # for patch, color in zip(bplot['boxes'], colors):
        #     patch.set_facecolor(color)
        # plt.tight_layout()
        # if save_fig:
        #     plt.rcParams['svg.fonttype'] = 'none'
        #     plt.savefig(os.path.join(save_path, "jittering_log_likeli_diff.svg"), transparent="True")
        # else:
        #     plt.show()
        #
        # print("HERE")
        # # do t-test to check whether values are above zero
        # print("POST initial values > 0:")
        # print(ttest_1samp(np.hstack(diff_post_interp_init), 0, alternative="greater"))
        # print("POST last values > 0:")
        # print(ttest_1samp(np.hstack(diff_post_interp_last), 0, alternative="greater"))
        #
        # print("PRE initial values > 0:")
        # print(ttest_1samp(np.hstack(diff_pre_interp_init), 0, alternative="greater"))
        # print("PRE last values > 0:")
        # print(ttest_1samp(np.hstack(diff_pre_interp_last), 0, alternative="greater"))

        # get min and max values for scaling
        cmap = matplotlib.cm.get_cmap('Oranges')
        colors_to_plot_pre = cmap(np.linspace(0, 1, len(self.session_list)))
        cmap = matplotlib.cm.get_cmap('Purples')
        colors_to_plot_post = cmap(np.linspace(0, 1, len(self.session_list)))

        if save_fig:
            plt.style.use('default')
            c="black"
        else:
            c="white"

        # plot acquisition decoded first
        # --------------------------------------------------------------------------------------------------------
        pre_first = []
        pre_last = []
        max_pre_first = []
        max_pre_last = []
        max_pre_jit_first = []
        max_pre_jit_last = []

        fig = plt.figure(figsize=(12,4))
        gs = fig.add_gridspec(6, 21)
        ax = fig.add_subplot(gs[:, :10])
        for i, (pre_dat, max_pre_dat, max_post_dat, max_pre_jit_dat) in enumerate(zip(diff_pre, max_pre,
                                                                                      max_post, max_pre_jit)):
            # only use data when PRE was decoded
            pre_dat = pre_dat[max_pre_dat > max_post_dat]
            max_pre_jit_dat = max_pre_jit_dat[max_pre_dat > max_post_dat]
            max_pre_dat = max_pre_dat[max_pre_dat > max_post_dat]

            max_pre_first.append(max_pre_dat[:int(max_pre_dat.shape[0]/2)])
            max_pre_last.append(max_pre_dat[-int(max_pre_dat.shape[0]/2):])
            max_pre_jit_first.append(max_pre_jit_dat[:int(max_pre_jit_dat.shape[0]/2)])
            max_pre_jit_last.append(max_pre_jit_dat[-int(max_pre_jit_dat.shape[0]/2):])

            pre_dat_smooth = uniform_filter1d(pre_dat, int(pre_dat.shape[0]/5))
            pre_first.append(pre_dat_smooth[:int(pre_dat_smooth.shape[0]/2)])
            pre_last.append(pre_dat_smooth[-int(pre_dat_smooth.shape[0]/2):])


            ax.plot(np.linspace(0, 1, pre_dat_smooth.shape[0]), pre_dat_smooth, linewidth=2, c=colors_to_plot_pre[i],
                    label=str(i))
        ax.legend(ncol=2)
        ax.set_xlabel("Relative time")
        ax.set_ylabel("Log-likelihood_original - \n Log-likelihood_jittered")
        ax.set_ylim(-0.1, 10)
        ax2 = fig.add_subplot(gs[:, 10:12])
        res = [np.hstack(pre_first), np.hstack(pre_last)]
        bplot = ax2.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["1st half", "2nd half"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax2.set_yticks([])
        ax2.set_ylim(-0.1, 10)
        ax2.hlines(4, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.05, 4, "***", color=c)

        ax3 = fig.add_subplot(gs[:, 15:17])
        res = [np.hstack(max_pre_first), np.hstack(max_pre_jit_first)]
        bplot = ax3.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["Original", "Jittered"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax3.set_ylabel("Log-likelihood")
        ax3.set_title("1st half")
        ax3.hlines(-19, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.05, -19, "***", color=c)
        ax3.set_ylim(-62, -10)

        ax4 = fig.add_subplot(gs[:, 19:])
        res = [np.hstack(max_pre_last), np.hstack(max_pre_jit_last)]
        bplot = ax4.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["Original", "Jittered"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax4.set_yticks([])
        ax4.set_title("2nd half")
        ax4.hlines(-19, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.05, -19, "***", color=c)
        ax4.set_ylim(-62, -10)

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "pre_decoded_original_vs_jitter.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # do stats per session
        print("STATS: Acquisition")
        print("################## All sessions:")
        print(ttest_1samp(np.hstack(pre_first), 0, alternative="greater"))
        print(ttest_1samp(np.hstack(pre_last), 0, alternative="greater"))
        print("##################")
        print("Difference > 0:")
        # difference above zero: first and second half
        max_p_diff_first_half = 0
        max_p_diff_second_half = 0
        for pf, pl in zip(pre_first, pre_last):
            max_p_diff_first_half = np.max([ttest_1samp(pf, 0, alternative="greater")[1], max_p_diff_first_half])
            max_p_diff_second_half = np.max([ttest_1samp(pl, 0, alternative="greater")[1], max_p_diff_second_half])
        print("max p-value (first half): "+str(max_p_diff_first_half/7))
        print("max p-value (second half): "+str(max_p_diff_second_half/7))

        print("original > jittered:")
        # difference above zero: first and second half
        max_p_diff_first_half = 0
        max_p_diff_second_half = 0
        for mf, ml, mjf, mjl in zip(max_pre_first, max_pre_last, max_pre_jit_first, max_pre_jit_last):
            max_p_diff_first_half = np.max([mannwhitneyu(mf, mjf, alternative="greater")[1], max_p_diff_first_half])
            max_p_diff_second_half = np.max([mannwhitneyu(ml, mjl, alternative="greater")[1], max_p_diff_second_half])
        print("max p-value (first half): "+str(max_p_diff_first_half/7))
        print("max p-value (second half): "+str(max_p_diff_second_half/7))

        # plot recall decoded
        # --------------------------------------------------------------------------------------------------------
        post_first = []
        post_last = []
        max_post_first = []
        max_post_last = []
        max_post_jit_first = []
        max_post_jit_last = []

        fig = plt.figure(figsize=(12,4))
        gs = fig.add_gridspec(6, 21)
        ax = fig.add_subplot(gs[:, :10])
        for i, (post_dat, max_pre_dat, max_post_dat, max_post_jit_dat) in enumerate(zip(diff_post, max_pre,
                                                                                      max_post, max_post_jit)):
            # only use data when POST was decoded
            post_dat = post_dat[max_pre_dat < max_post_dat]
            max_post_jit_dat = max_post_jit_dat[max_pre_dat < max_post_dat]
            max_post_dat = max_post_dat[max_pre_dat < max_post_dat]

            max_post_first.append(max_post_dat[:int(max_post_dat.shape[0]/2)])
            max_post_last.append(max_post_dat[-int(max_post_dat.shape[0]/2):])
            max_post_jit_first.append(max_post_jit_dat[:int(max_post_jit_dat.shape[0]/2)])
            max_post_jit_last.append(max_post_jit_dat[-int(max_post_jit_dat.shape[0]/2):])

            post_dat_smooth = uniform_filter1d(post_dat, int(post_dat.shape[0]/5))
            post_first.append(post_dat_smooth[:int(post_dat_smooth.shape[0]/2)])
            post_last.append(post_dat_smooth[-int(post_dat_smooth.shape[0]/2):])


            ax.plot(np.linspace(0, 1, post_dat_smooth.shape[0]), post_dat_smooth, linewidth=2, c=colors_to_plot_pre[i],
                    label=str(i))
        ax.legend(ncol=2)
        ax.set_xlabel("Relative time")
        ax.set_ylabel("Log-likelihood_original - \n Log-likelihood_jittered")
        ax.set_ylim(-0.1, 10)
        ax2 = fig.add_subplot(gs[:, 10:12])
        res = [np.hstack(post_first), np.hstack(post_last)]
        bplot = ax2.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["1st half", "2nd half"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax2.set_yticks([])
        ax2.set_ylim(-0.1, 10)
        ax2.hlines(8, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.05, 8, "***", color=c)

        ax3 = fig.add_subplot(gs[:, 15:17])
        res = [np.hstack(max_post_first), np.hstack(max_post_jit_first)]
        bplot = ax3.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["Original", "Jittered"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax3.set_ylabel("Log-likelihood")
        ax3.set_title("1st half")
        ax3.hlines(-16, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.05, -16, "***", color=c)
        ax3.set_ylim(-65, -10)

        ax4 = fig.add_subplot(gs[:, 19:])
        res = [np.hstack(max_post_last), np.hstack(max_post_jit_last)]
        bplot = ax4.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["Original", "Jittered"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax4.set_yticks([])
        ax4.set_title("2nd half")
        ax4.hlines(-16, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.05, -16, "***", color=c)
        ax4.set_ylim(-65, -10)

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "post_decoded_original_vs_jitter.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # do stats per session
        print("STATS: Recall")
        print("################## All sessions:")
        print(ttest_1samp(np.hstack(post_first), 0, alternative="greater"))
        print(ttest_1samp(np.hstack(post_last), 0, alternative="greater"))
        print("##################")
        print("Difference > 0:")
        # difference above zero: first and second half
        min_p_diff_first_half = 1
        min_p_diff_second_half = 1
        for pf, pl in zip(post_first, post_last):
            min_p_diff_first_half = np.min([ttest_1samp(pf, 0, alternative="greater")[1], min_p_diff_first_half])
            min_p_diff_second_half = np.min([ttest_1samp(pl, 0, alternative="greater")[1], min_p_diff_second_half])
        print("min p-value (first half): "+str(min_p_diff_first_half))
        print("min p-value (second half): "+str(min_p_diff_second_half))

        print("original > jittered:")
        # difference above zero: first and second half
        min_p_diff_first_half = 1
        min_p_diff_second_half = 1
        for mf, ml, mjf, mjl in zip(max_post_first, max_post_last, max_post_jit_first, max_post_jit_last):
            min_p_diff_first_half = np.min([mannwhitneyu(mf, mjf, alternative="greater")[1], min_p_diff_first_half])
            min_p_diff_second_half = np.min([mannwhitneyu(ml, mjl, alternative="greater")[1], min_p_diff_second_half])
        print("min p-value (first half): "+str(min_p_diff_first_half))
        print("min p-value (second half): "+str(min_p_diff_second_half))




        plt.figure(figsize=(2, 3))

        res = [slopes_original, slopes_jittered]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["Original", "Jittered"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        _, y_max = plt.gca().get_ylim()
        plt.ylim(0, y_max+0.2)
        plt.ylabel("Slope of drift score")
        plt.xticks(rotation=45)
        y_base = y_max+0.1
        plt.hlines(y_base, 1, 2)
        if mannwhitneyu(res[0], res[1])[1] > 0.05:
            plt.text(1.4, y_base, "n.s.")
        elif mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.4, y_base, "***")
        elif mannwhitneyu(res[0], res[1])[1] < 0.01:
            plt.text(1.4, y_base, "**")
        elif mannwhitneyu(res[0], res[1])[1] < 0.05:
            plt.text(1.4, y_base, "*")

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "jittering_slope_drift_score.svg"), transparent="True")
        else:
            plt.show()


    def long_sleep_memory_drift_null_model(self, save_fig=False, load_from_temp=True):

        sim_ratio = []
        sim_ratio_j = []
        diff_pre = []
        diff_post = []
        max_pre = []
        max_pre_jit = []
        max_post = []
        max_post_jit = []

        for i, session in enumerate(self.session_list):
            if load_from_temp:
                infile = open("temp_data/null_model/" + session.session_name, 'rb')
                results = pickle.load(infile)
                sim_ratio_ = results["sim_ratio"]
                sim_ratio_j_ = results["sim_ratio_j"]
                diff_pre_ = results["diff_pre"]
                diff_post_ = results["diff_post"]
                max_pre_ = results["max_pre"]
                max_pre_jit_ =results["max_pre_jit"]
                max_post_ = results["max_post"]
                max_post_jit_ = results["max_post_jit"]
                infile.close()
            else:
                sim_ratio_, sim_ratio_j_, diff_pre_, diff_post_ = \
                    session.long_sleep().memory_drift_control_jitter_combine_sleep_phases(plotting=False)
            sim_ratio.append(sim_ratio_)
            sim_ratio_j.append(sim_ratio_j_)
            diff_pre.append(diff_pre_)
            diff_post.append(diff_post_)
            max_pre.append(max_pre_)
            max_pre_jit.append(max_pre_jit_)
            max_post.append(max_post_)
            max_post_jit.append(max_post_jit_)

        # compute slopes of drift score for original and jittered case
        slopes_original = []
        slopes_jittered = []

        for drift_score, drift_score_j in zip(sim_ratio, sim_ratio_j):
            coef_original = np.polyfit(np.linspace(0,1,drift_score.shape[0]), drift_score, 1)
            slopes_original.append(coef_original[0])
            coef_j = np.polyfit(np.linspace(0,1,drift_score_j.shape[0]), drift_score_j, 1)
            slopes_jittered.append(coef_j[0])

        # need to interpolate all differences from different session --> they need to be of the same length to compute
        # mean and std
        max_bins = np.max([x.shape[0] for x in diff_pre])

        diff_pre_interp = []
        diff_post_interp = []
        diff_pre_interp_init = []
        diff_pre_interp_last = []
        diff_post_interp_init = []
        diff_post_interp_last = []

        for pre, post in zip(diff_pre, diff_post):
            if pre.shape[0] < max_bins:
                diff_pre_interp.append(np.interp(np.linspace(0, 1, max_bins), np.linspace(0, 1, pre.shape[0]), pre))
                diff_post_interp.append(np.interp(np.linspace(0, 1, max_bins), np.linspace(0, 1, post.shape[0]), post))
            else:
                diff_pre_interp.append(pre)
                diff_post_interp.append(post)

            diff_pre_interp_init.append(diff_pre_interp[-1][:100])
            diff_pre_interp_last.append(diff_pre_interp[-1][-100:])
            diff_post_interp_init.append(diff_post_interp[-1][:100])
            diff_post_interp_last.append(diff_post_interp[-1][-100:])

        diff_pre_interp = np.vstack(diff_pre_interp)
        diff_post_interp = np.vstack(diff_post_interp)

        print("End of sleep: diff pre vs. diff post")
        print(mannwhitneyu(np.hstack(diff_pre_interp_last), np.hstack(diff_post_interp_last), alternative="less"))
        print(np.mean(np.hstack(diff_pre_interp_last)), np.mean(np.hstack(diff_post_interp_last)))

        # compute mean and std for plotting
        diff_pre_interp_mean = np.mean(diff_pre_interp, axis=0)
        diff_pre_interp_std = np.std(diff_pre_interp, axis=0)
        diff_post_interp_mean = np.mean(diff_post_interp, axis=0)
        diff_post_interp_std = np.std(diff_post_interp, axis=0)
        #
        # if save_fig:
        #     plt.style.use('default')
        #     c = "black"
        # else:
        #     c = "white"
        #
        # fig = plt.figure(figsize=(8, 7))
        # gs = fig.add_gridspec(9, 10)
        # ax1 = fig.add_subplot(gs[:4, :7])
        # ax2 = fig.add_subplot(gs[:4, 8:])
        # ax3 = fig.add_subplot(gs[5:, :7])
        # ax4 = fig.add_subplot(gs[5:, 8:])
        #
        # y_pre_max = np.max(diff_pre_interp_mean+diff_pre_interp_std)+1.5
        # y_post_max = np.max(diff_post_interp_mean+diff_post_interp_std)+1.5
        #
        # # ax1.errorbar(np.linspace(0, 1, diff_pre_interp_mean.shape[0]), diff_pre_interp_mean, yerr=diff_pre_interp_std, color="moccasin", zorder=-1000)
        # ax1.plot(np.linspace(0, 1, diff_pre_interp_mean.shape[0]), diff_pre_interp_mean-diff_pre_interp_std, color="moccasin", linewidth=1,
        #          linestyle="--")
        # ax1.plot(np.linspace(0, 1, diff_pre_interp_mean.shape[0]), diff_pre_interp_mean+diff_pre_interp_std, color="moccasin", linewidth=1,
        #          linestyle="--")
        # ax1.plot(np.linspace(0, 1, diff_pre_interp_mean.shape[0]), diff_pre_interp_mean, color="orange", linewidth=2,
        #          label="Acquisition states")
        # # ax1.fill(np.append(np.linspace(0, 1, diff_pre_interp_mean.shape[0]), np.linspace(0, 1, diff_pre_interp_mean.shape[0])[::-1]),
        # #          np.append(diff_pre_interp_mean+diff_pre_interp_std, (diff_pre_interp_mean-diff_pre_interp_std)[::-1]), 'moccasin')
        # ax1.set_ylabel("Log-likelihood_original - \n Log-likelihood_null_model")
        # ax1.legend()
        # ax1.set_ylim(-0.7, y_pre_max)
        #
        # # ax3.errorbar(np.linspace(0, 1, diff_post_interp_mean.shape[0]), diff_post_interp_mean, yerr=diff_post_interp_std, color="lightcyan", zorder=-1000)
        # ax3.plot(np.linspace(0, 1, diff_post_interp_mean.shape[0]), diff_post_interp_mean, color="lightblue", linewidth=2, label="Recall states")
        # ax3.plot(np.linspace(0, 1, diff_post_interp_mean.shape[0]), diff_post_interp_mean-diff_post_interp_std, color="lightcyan",
        #          linewidth=1, linestyle="--")
        # ax3.plot(np.linspace(0, 1, diff_post_interp_mean.shape[0]), diff_post_interp_mean+diff_post_interp_std, color="lightcyan",
        #          linewidth=1, linestyle="--")
        # # ax3.fill(np.append(np.linspace(0, 1, diff_post_interp_mean.shape[0]), np.linspace(0, 1, diff_post_interp_mean.shape[0])[::-1]),
        # #          np.append(diff_post_interp_mean+diff_post_interp_std, (diff_post_interp_mean-diff_post_interp_std)[::-1]), 'lightcyan')
        #
        # ax3.legend()
        # ax3.set_xlabel("Normalized duration")
        # ax3.set_ylabel("Log-likelihood_original - \n Log-likelihood_null_model")
        # ax3.set_ylim(-0.7, y_post_max)
        # res = [np.hstack(diff_pre_interp_init), np.hstack(diff_pre_interp_last)]
        # bplot=ax2.boxplot(res, positions=[1, 2], patch_artist=True,
        #                     labels=["first n\n values", "last n\n values"],
        #                     boxprops=dict(color=c),
        #                     capprops=dict(color=c),
        #                     whiskerprops=dict(color=c),
        #                     flierprops=dict(color=c, markeredgecolor=c),
        #                     medianprops=dict(color=c), showfliers=False)
        # ax2.set_ylim(-0.7, y_pre_max)
        # ax2.tick_params(axis='x', labelrotation=45)
        # ax2.set_yticks([])
        # y_base = y_pre_max-0.2
        # ax2.hlines(y_base, 1, 2)
        # if mannwhitneyu(res[0], res[1])[1] > 0.05:
        #     ax2.text(1.4, y_base, "n.s.")
        # elif mannwhitneyu(res[0], res[1])[1] < 0.001:
        #     ax2.text(1.4, y_base, "***")
        # elif mannwhitneyu(res[0], res[1])[1] < 0.01:
        #     ax2.text(1.4, y_base, "**")
        # elif mannwhitneyu(res[0], res[1])[1] < 0.05:
        #     ax2.text(1.4, y_base, "*")
        #
        # colors = ["moccasin", "moccasin"]
        # for patch, color in zip(bplot['boxes'], colors):
        #     patch.set_facecolor(color)
        #
        # res = [np.hstack(diff_post_interp_init), np.hstack(diff_post_interp_last)]
        # bplot=ax4.boxplot(res, positions=[1, 2], patch_artist=True,
        #                     labels=["first n\n values", "last n\n values"],
        #                     boxprops=dict(color=c),
        #                     capprops=dict(color=c),
        #                     whiskerprops=dict(color=c),
        #                     flierprops=dict(color=c, markeredgecolor=c),
        #                     medianprops=dict(color=c), showfliers=False)
        # ax4.set_ylim(-0.7, y_post_max)
        # ax4.tick_params(axis='x', labelrotation=45)
        # ax4.set_yticks([])
        # y_base = y_post_max - 0.2
        # ax4.hlines(y_base, 1, 2)
        # if mannwhitneyu(res[0], res[1])[1] > 0.05:
        #     ax4.text(1.4, y_base, "n.s.")
        # elif mannwhitneyu(res[0], res[1])[1] < 0.001:
        #     ax4.text(1.4, y_base, "***")
        # elif mannwhitneyu(res[0], res[1])[1] < 0.01:
        #     ax4.text(1.4, y_base, "**")
        # elif mannwhitneyu(res[0], res[1])[1] < 0.05:
        #     ax4.text(1.4, y_base, "*")
        # colors = ["lightcyan", "lightcyan"]
        # for patch, color in zip(bplot['boxes'], colors):
        #     patch.set_facecolor(color)
        # plt.tight_layout()
        # if save_fig:
        #     plt.rcParams['svg.fonttype'] = 'none'
        #     plt.savefig(os.path.join(save_path, "null_model_log_likeli_diff.svg"), transparent="True")
        # else:
        #     plt.show()
        #
        #
        # # comparing slopes of drift score for original and jittered
        # # --------------------------------------------------------------------------------------------------------------
        # plt.figure(figsize=(2, 3))
        # res = [slopes_original, slopes_jittered]
        # bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
        #                     labels=["Original", "Null model"],
        #                     boxprops=dict(color=c),
        #                     capprops=dict(color=c),
        #                     whiskerprops=dict(color=c),
        #                     flierprops=dict(color=c, markeredgecolor=c),
        #                     medianprops=dict(color=c), showfliers=False)
        # _, y_max = plt.gca().get_ylim()
        # plt.ylim(0, y_max+0.2)
        # plt.ylabel("Slope of drift score")
        # plt.xticks(rotation=45)
        # y_base = y_max+0.1
        # plt.hlines(y_base, 1, 2)
        # if mannwhitneyu(res[0], res[1])[1] > 0.05:
        #     plt.text(1.4, y_base, "n.s.")
        # elif mannwhitneyu(res[0], res[1])[1] < 0.001:
        #     plt.text(1.4, y_base, "***")
        # elif mannwhitneyu(res[0], res[1])[1] < 0.01:
        #     plt.text(1.4, y_base, "**")
        # elif mannwhitneyu(res[0], res[1])[1] < 0.05:
        #     plt.text(1.4, y_base, "*")
        # plt.tight_layout()
        # if save_fig:
        #     plt.rcParams['svg.fonttype'] = 'none'
        #     plt.savefig(os.path.join(save_path, "null_model_slope_drift_score.svg"), transparent="True")
        # else:
        #     plt.show()
        #
        # # comparing slopes of drift score for original and jittered
        # # --------------------------------------------------------------------------------------------------------------
        # plt.figure(figsize=(2, 3))
        # res = [np.hstack(diff_pre_interp_last), np.hstack(diff_post_interp_last)]
        # bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
        #                     labels=["last 100\n values Acq.", "last 100\n values Recall"],
        #                     boxprops=dict(color=c),
        #                     capprops=dict(color=c),
        #                     whiskerprops=dict(color=c),
        #                     flierprops=dict(color=c, markeredgecolor=c),
        #                     medianprops=dict(color=c), showfliers=False)
        # _, y_max = plt.gca().get_ylim()
        # plt.ylim(-0.6, y_max+0.2)
        # plt.ylabel("Likelihood_original -\n Likelihood_null_model")
        # plt.xticks(rotation=45)
        # y_base = y_max+0.1
        # plt.hlines(y_base, 1, 2)
        # if mannwhitneyu(res[0], res[1])[1] > 0.05:
        #     plt.text(1.4, y_base, "n.s.")
        # elif mannwhitneyu(res[0], res[1])[1] < 0.001:
        #     plt.text(1.4, y_base, "***")
        # elif mannwhitneyu(res[0], res[1])[1] < 0.01:
        #     plt.text(1.4, y_base, "**")
        # elif mannwhitneyu(res[0], res[1])[1] < 0.05:
        #     plt.text(1.4, y_base, "*")
        # plt.tight_layout()
        # if save_fig:
        #     plt.rcParams['svg.fonttype'] = 'none'
        #     plt.savefig(os.path.join(save_path, "null_model_last_pre_vs_last_post.svg"), transparent="True")
        # else:
        #     plt.show()
        #
        # # stats test
        # # --------------------------------------------------------------------------------------------------------------
        #
        # # do t-test to check whether values are above zero
        # print("POST initial values > 0:")
        # print(ttest_1samp(np.hstack(diff_post_interp_init), 0, alternative="greater"))
        # print("POST last values > 0:")
        # print(ttest_1samp(np.hstack(diff_post_interp_last), 0, alternative="greater"))
        #
        # print("PRE initial values > 0:")
        # print(ttest_1samp(np.hstack(diff_pre_interp_init), 0, alternative="greater"))
        # print("PRE last values > 0:")
        # print(ttest_1samp(np.hstack(diff_pre_interp_last), 0, alternative="greater"))

        # get min and max values for scaling
        cmap = matplotlib.cm.get_cmap('Oranges')
        colors_to_plot_pre = cmap(np.linspace(0, 1, len(self.session_list)))
        cmap = matplotlib.cm.get_cmap('Purples')
        colors_to_plot_post = cmap(np.linspace(0, 1, len(self.session_list)))

        if save_fig:
            plt.style.use('default')
            c="black"
        else:
            c="white"

        # plot acquisition decoded first
        # --------------------------------------------------------------------------------------------------------
        pre_first = []
        pre_last = []
        max_pre_first = []
        max_pre_last = []
        max_pre_jit_first = []
        max_pre_jit_last = []

        fig = plt.figure(figsize=(8,3))
        gs = fig.add_gridspec(6, 21)
        ax = fig.add_subplot(gs[:, :7])
        for i, (pre_dat, max_pre_dat, max_post_dat, max_pre_jit_dat) in enumerate(zip(diff_pre, max_pre,
                                                                                      max_post, max_pre_jit)):
            # only use data when PRE was decoded
            pre_dat = pre_dat[max_pre_dat > max_post_dat]
            max_pre_jit_dat = max_pre_jit_dat[max_pre_dat > max_post_dat]
            max_pre_dat = max_pre_dat[max_pre_dat > max_post_dat]

            pre_first.append(pre_dat[:100])
            pre_last.append(pre_dat[-100:])
            max_pre_first.append(max_pre_dat[:100])
            max_pre_last.append(max_pre_dat[-100:])
            max_pre_jit_first.append(max_pre_jit_dat[:100])
            max_pre_jit_last.append(max_pre_jit_dat[-100:])

            ax.plot(np.linspace(0, 1, pre_dat.shape[0]), pre_dat, linewidth=2, c=colors_to_plot_pre[i],
                    label=str(i))
        ax.legend(ncol=2)
        ax.set_xlabel("Relative time")
        ax.set_ylabel("Log-likelihood_original - \n Log-likelihood_null_model")
        ax.set_ylim(-0.5,2.2)
        ax2 = fig.add_subplot(gs[:, 7:10])
        res = [np.hstack(pre_first), np.hstack(pre_last)]
        bplot = ax2.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["first n \n values", "last n \n values"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax2.set_yticks([])
        ax2.set_ylim(-0.5, 2.2)
        ax2.hlines(1.7, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.05, 1.7, "***", color=c)

        ax3 = fig.add_subplot(gs[:, 12:15])
        res = [np.hstack(max_pre_first), np.hstack(max_pre_jit_first)]
        bplot = ax3.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["Original", "Jittered"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax3.set_ylabel("Log-likelihood")
        ax3.set_title("First 100 \n values")
        ax3.hlines(-34, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.05, -34, "***", color=c)
        ax3.set_ylim(-48, -32)

        ax4 = fig.add_subplot(gs[:, 18:])
        res = [np.hstack(max_pre_last), np.hstack(max_pre_jit_last)]
        bplot = ax4.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["Original", "Null model"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax4.set_yticks([])
        ax4.set_title("Last 100 \n values")
        ax4.hlines(-34, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.05, -34, "***", color=c)
        ax4.set_ylim(-48, -32)

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "pre_decoded_original_vs_null_model.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # plot recall decoded
        # --------------------------------------------------------------------------------------------------------
        post_first = []
        post_last = []
        max_post_first = []
        max_post_last = []
        max_post_jit_first = []
        max_post_jit_last = []


        fig = plt.figure(figsize=(8,3))
        gs = fig.add_gridspec(6, 21)
        ax = fig.add_subplot(gs[:, :7])
        for i, (post_dat, max_pre_dat, max_post_dat, max_post_jit_dat) in enumerate(zip(diff_post, max_pre,
                                                                                        max_post, max_post_jit)):
            # only use data when POST was decoded
            post_dat = post_dat[max_pre_dat < max_post_dat]
            max_post_jit_dat = max_post_jit_dat[max_pre_dat < max_post_dat]
            max_post_dat = max_post_dat[max_pre_dat < max_post_dat]

            post_first.append(post_dat[:100])
            post_last.append(post_dat[-100:])
            max_post_first.append(max_post_dat[:100])
            max_post_last.append(max_post_dat[-100:])
            max_post_jit_first.append(max_post_jit_dat[:100])
            max_post_jit_last.append(max_post_jit_dat[-100:])

            ax.plot(np.linspace(0, 1, post_dat.shape[0]), post_dat, linewidth=2, c=colors_to_plot_post[i],
                    label=str(i))
        ax.legend(ncol=2)
        ax.set_xlabel("Relative time")
        ax.set_ylabel("Log-likelihood_original - \n Log-likelihood_null_model")
        ax.set_ylim(-0.5,2.2)
        ax2 = fig.add_subplot(gs[:, 7:10])
        res = [np.hstack(post_first), np.hstack(post_last)]
        bplot = ax2.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["first n \n values", "last n \n values"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax2.set_yticks([])
        ax2.set_ylim(-0.5,2.2)
        ax2.hlines(1.7, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.05, 1.7, "***", color=c)

        ax3 = fig.add_subplot(gs[:, 12:15])
        res = [np.hstack(max_post_first), np.hstack(max_post_jit_first)]
        bplot = ax3.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["Original", "Null model"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax3.set_ylabel("Log-likelihood")
        ax3.set_title("First 100 \n values")
        ax3.hlines(-33, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.05, -33, "***", color=c)
        ax3.set_ylim(-48, -31)

        ax4 = fig.add_subplot(gs[:, 18:])
        res = [np.hstack(max_post_last), np.hstack(max_post_jit_last)]
        bplot = ax4.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["Original", "Null model"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        ax4.set_yticks([])
        ax4.set_title("Last 100 \n values")
        ax4.hlines(-33, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.05, -33, "***", color=c)
        ax4.set_ylim(-48, -31)

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "post_decoded_original_vs_null_model.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        plt.figure(figsize=(2, 3))

        res = [slopes_original, slopes_jittered]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["Original", "Null model"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        _, y_max = plt.gca().get_ylim()
        plt.ylim(0, y_max+0.2)
        plt.ylabel("Slope of drift score")
        plt.xticks(rotation=45)
        y_base = y_max+0.1
        plt.hlines(y_base, 1, 2)
        if mannwhitneyu(res[0], res[1])[1] > 0.05:
            plt.text(1.4, y_base, "n.s.")
        elif mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.4, y_base, "***")
        elif mannwhitneyu(res[0], res[1])[1] < 0.01:
            plt.text(1.4, y_base, "**")
        elif mannwhitneyu(res[0], res[1])[1] < 0.05:
            plt.text(1.4, y_base, "*")

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "null_model_slope_drift_score.svg"), transparent="True")
        else:
            plt.show()


    def long_sleep_memory_drift_equalized(self, save_fig=False, load_from_temp=False, pre_model="pre",
                                          post_model="post"):

        sim_ratio = []
        sim_ratio_j = []

        for i, session in enumerate(self.session_list):
            if load_from_temp:
                infile = open("temp_data/equalized/" + session.session_name, 'rb')
                results = pickle.load(infile)
                sim_ratio_ = results["sim_ratio"]
                sim_ratio_j_ = results["sim_ratio_j"]
                infile.close()
            else:
                sim_ratio_, sim_ratio_j_ = \
                        session.long_sleep().memory_drift_control_equalize_combine_sleep_phases(plotting=False,
                                                        pre_model=pre_model, post_model=post_model)
            sim_ratio.append(sim_ratio_)
            sim_ratio_j.append(sim_ratio_j_)


        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        plt.figure(figsize=(4, 3))
        # plot sim ratios for jittered
        cmap = matplotlib.cm.get_cmap('Oranges')
        colors_to_plot_pre = cmap(np.linspace(0.1, 1, len(self.session_list)))
        for i, sr in enumerate(sim_ratio_j):
            sr_s = uniform_filter1d(sr, int(sr.shape[0]/10))
            plt.plot(np.linspace(0,1,sr_s.shape[0]), sr_s, c=colors_to_plot_pre[i])
        plt.ylim(-1,1)
        plt.xlabel("Normalized duration")
        plt.ylabel("Drift score")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_scores_per_session_equalized_all_sessions.svg"), transparent="True")
        else:
            plt.show()

        # compute slopes of drift score for original and jittered case
        slopes_original = []
        slopes_jittered = []

        for drift_score, drift_score_j in zip(sim_ratio, sim_ratio_j):
            coef_original = np.polyfit(np.linspace(0,1,drift_score.shape[0]), drift_score, 1)
            slopes_original.append(coef_original[0])
            coef_j = np.polyfit(np.linspace(0,1,drift_score_j.shape[0]), drift_score_j, 1)
            slopes_jittered.append(coef_j[0])

        plt.figure(figsize=(2, 3))

        res = [slopes_original, slopes_jittered]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["Original", "Equalized"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        _, y_max = plt.gca().get_ylim()
        plt.ylim(0, y_max+0.2)
        plt.ylabel("Slope of drift score")
        plt.xticks(rotation=45)
        y_base = y_max+0.1
        plt.hlines(y_base, 1, 2)
        if mannwhitneyu(res[0], res[1])[1] > 0.05:
            plt.text(1.4, y_base, "n.s.")
        elif mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.4, y_base, "***")
        elif mannwhitneyu(res[0], res[1])[1] < 0.01:
            plt.text(1.4, y_base, "**")
        elif mannwhitneyu(res[0], res[1])[1] < 0.05:
            plt.text(1.4, y_base, "*")

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_score_equalized_all_sessions.svg"), transparent="True")
        else:
            plt.show()

        print("HERE")

    def long_sleep_memory_drift_subsets(self, save_fig=False, load_from_temp=False, pre_model="pre",
                                              post_model="post"):

        sim_ratio = []
        sim_ratio_dec = []
        sim_ratio_inc = []
        sim_ratio_stable = []

        for i, session in enumerate(self.session_list):
            sim_ratio_, sim_ratio_dec_, sim_ratio_inc_, sim_ratio_stable_  = \
                session.long_sleep().memory_drift_combine_sleep_phases_subsets(plotting=False)
            sim_ratio.append(sim_ratio_)
            sim_ratio_dec.append(sim_ratio_dec_)
            sim_ratio_inc.append(sim_ratio_inc_)
            sim_ratio_stable.append(sim_ratio_stable_)


        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"


        # compute slopes of drift score for original and jittered case
        slopes_all = []
        slopes_dec = []
        slopes_inc = []
        slopes_stable = []

        for ds_all, ds_dec, ds_inc, ds_stable in zip(sim_ratio, sim_ratio_dec, sim_ratio_inc, sim_ratio_stable):
            slopes_all.append(np.polyfit(np.linspace(0,1,ds_all.shape[0]), ds_all, 1)[0])
            slopes_dec.append(np.polyfit(np.linspace(0,1,ds_dec.shape[0]), ds_dec, 1)[0])
            slopes_inc.append(np.polyfit(np.linspace(0,1,ds_inc.shape[0]), ds_inc, 1)[0])
            slopes_stable.append(np.polyfit(np.linspace(0,1,ds_stable.shape[0]), ds_stable, 1)[0])

        res = [np.hstack(slopes_all), np.hstack(slopes_dec), np.hstack(slopes_inc), np.hstack(slopes_stable)]

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        plt.figure(figsize=(4,3))
        bplot = plt.boxplot(res, positions=np.arange(4), patch_artist=True,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            labels=["all cells", "decreasing", "increasing", "persistent"],
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["grey", "turquoise", "orange", "#6B345C"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        # plt.grid(color="grey", axis="y")
        degrees = 45
        plt.xticks(rotation=degrees)
        # check significance
        plt.ylim(-0.2, 1.4)
        plt.yticks([0, 0.5, 1])
        plt.ylabel("Slope of sim_ratio")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "slope_stable_dec_inc_boxplot.svg"), transparent="True")
        else:
            plt.show()

        # compute stats --> need to correct for multiple comparisons
        # --------------------------------------------------------------------------------------------------------------

        # all vs. rest
        print("All vs. Dec")
        print(mannwhitneyu(res[0], res[1])[1]*6)

        print("All vs. Inc")
        print(mannwhitneyu(res[0], res[2])[1]*6)

        print("All vs. Stable")
        print(mannwhitneyu(res[0], res[3])[1]*6)

        print("Dec vs. Inc")
        print(mannwhitneyu(res[1], res[2])[1]*6)

        print("Dec vs. Stable")
        print(mannwhitneyu(res[1], res[3])[1]*6)

        print("Stable vs. Inc")
        print(mannwhitneyu(res[2], res[3])[1]*6)

    def long_sleep_memory_drift_pre_post_vs_pre_fam_post_fam_subset(self, save_fig=False):

        sim_ratio = []
        sim_ratio_j = []

        for i, session in enumerate(self.session_list):

            sim_ratio_, sim_ratio_j_ = \
                session.long_sleep().memory_drift_subsets_pre_post_vs_fam(plotting=False)
            sim_ratio.append(sim_ratio_)
            sim_ratio_j.append(sim_ratio_j_)

        # compute slopes of drift score for original and jittered case
        slopes_original = []
        slopes_jittered = []

        for drift_score, drift_score_j in zip(sim_ratio, sim_ratio_j):
            coef_original = np.polyfit(np.linspace(0,1,drift_score.shape[0]), drift_score, 1)
            slopes_original.append(coef_original[0])
            coef_j = np.polyfit(np.linspace(0,1,drift_score_j.shape[0]), drift_score_j, 1)
            slopes_jittered.append(coef_j[0])

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        plt.figure(figsize=(3, 4))

        res = [slopes_original, slopes_jittered]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["Acquisition-Recall", "Pre_exploration - Post_exploration"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        _, y_max = plt.gca().get_ylim()
        plt.ylim(-0.1, y_max+0.2)
        plt.ylabel("Slope of drift score")
        plt.xticks(rotation=45)
        y_base = y_max+0.1
        plt.hlines(y_base, 1, 2)
        if wilcoxon(res[0], res[1], alternative="less")[1] > 0.05:
            plt.text(1.4, y_base, "n.s.")
        elif wilcoxon(res[0], res[1], alternative="less")[1] < 0.001:
            plt.text(1.4, y_base, "***")
        elif wilcoxon(res[0], res[1], alternative="less")[1] < 0.01:
            plt.text(1.4, y_base, "**")
        elif wilcoxon(res[0], res[1], alternative="less")[1] < 0.05:
            plt.text(1.4, y_base, "*")

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_score_subset_pre_post_vs_pre_fam_post_fam_all_sessions.svg"),
                        transparent="True")
        else:
            plt.show()

    def long_sleep_memory_drift_pre_post_vs_pre_fam_post_fam(self, save_fig=False, z_score=False):

        pre_nrem = []
        pre_excluded = []
        post_nrem = []
        post_excluded = []
        sim_ratio = []
        sim_ratio_excluded = []
        frac_first_half = []
        frac_second_half = []

        for i, session in enumerate(self.session_list):
            pre_nrem_, pre_excluded_, post_nrem_, post_excluded_, sim_ratio_, sim_ratio_j_, frac_first_half_, \
                frac_second_half_ = \
                session.long_sleep().memory_drift_fam_exploration(plotting=False, z_score=z_score)
            pre_nrem.append(pre_nrem_)
            pre_excluded.append(pre_excluded_)
            post_nrem.append(post_nrem_)
            post_excluded.append(post_excluded_)
            sim_ratio.append(sim_ratio_)
            sim_ratio_excluded.append(sim_ratio_j_)
            frac_first_half.append(frac_first_half_)
            frac_second_half.append(frac_second_half_)

        slopes_long = []
        slopes_excl = []

        sim_ratio_s = []
        sim_ratio_excl_s = []

        # print drift score for long_sleep and fam_sleep
        for i, (sim_ratio_sess, sim_ratio_fam_sess) in enumerate(zip(sim_ratio, sim_ratio_excluded)):
            sim_ratio_sess_s = moving_average(sim_ratio_sess, int(sim_ratio_sess.shape[0]/10))
            sim_ratio_sess_fam_s = moving_average(sim_ratio_fam_sess, int(sim_ratio_fam_sess.shape[0]/10))
            coef_original = np.polyfit(np.linspace(0, 1, sim_ratio_sess_s .shape[0]), sim_ratio_sess_s , 1)
            slopes_long.append(coef_original[0])
            coef_fam = np.polyfit(np.linspace(0, 1, sim_ratio_sess_fam_s .shape[0]), sim_ratio_sess_fam_s , 1)
            slopes_excl.append(coef_fam[0])

            sim_ratio_s.append(sim_ratio_sess_s)
            sim_ratio_excl_s.append(sim_ratio_sess_fam_s)


        # using the whole rest:
        pre_nrem_all = np.hstack(pre_nrem)
        pre_excluded_all = np.hstack(pre_excluded)
        post_nrem_all = np.hstack(post_nrem)
        post_excluded_all = np.hstack(post_excluded)

        pre_nrem_first = []
        pre_nrem_second = []
        pre_excluded_first = []
        pre_excluded_second = []
        post_nrem_first = []
        post_nrem_second = []
        post_excluded_first = []
        post_excluded_second = []

        for pre_n, pre_e, post_n, post_e in zip(pre_nrem, pre_excluded, post_nrem, post_excluded):
            pre_nrem_first.append(pre_n[:int(pre_n.shape[0]/2)])
            pre_nrem_second.append(pre_n[int(pre_n.shape[0]/2):])
            pre_excluded_first.append(pre_e[:int(pre_e.shape[0]/2)])
            pre_excluded_second.append(pre_e[int(pre_e.shape[0]/2):])
            post_nrem_first.append(post_n[:int(post_n.shape[0]/2)])
            post_nrem_second.append(post_n[int(post_n.shape[0]/2):])
            post_excluded_first.append(post_e[:int(post_e.shape[0]/2)])
            post_excluded_second.append(post_e[int(post_e.shape[0]/2):])


        max_len_long_sleep = np.max([x.shape[0] for x in sim_ratio_s])
        max_len_fam_sleep = np.max([x.shape[0] for x in sim_ratio_excl_s])

        # compute interpolated data
        long_sleep_inter = np.zeros((max_len_long_sleep, len(self.session_list)))
        fam_sleep_inter = np.zeros((max_len_fam_sleep, len(self.session_list)))

        for i, (sim_ratio_sess, sim_ratio_fam_sess) in enumerate(zip(sim_ratio_s, sim_ratio_excl_s)):
            long_sleep_inter[:, i] = np.interp(np.linspace(0, 1, max_len_long_sleep), np.linspace(0, 1,
                                                                  sim_ratio_sess.shape[0]), sim_ratio_sess)
            fam_sleep_inter[:, i] = np.interp(np.linspace(0, 1, max_len_fam_sleep), np.linspace(0, 1,
                                                    sim_ratio_fam_sess.shape[0]), sim_ratio_fam_sess)

        long_sleep_mean = np.mean(long_sleep_inter, axis=1)
        long_sleep_std = np.std(long_sleep_inter, axis=1)
        fam_sleep_mean = np.mean(fam_sleep_inter, axis=1)
        fam_sleep_std = np.std(fam_sleep_inter, axis=1)

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        fig = plt.figure(figsize=(8, 5))
        gs = fig.add_gridspec(9, 10)
        ax1 = fig.add_subplot(gs[:, :7])
        ax2 = fig.add_subplot(gs[:, 8:])

        y_long_max = np.max(long_sleep_mean+long_sleep_std)+0.5
        y_fam_max = np.max(fam_sleep_mean+fam_sleep_std)+0.5

        # ax1.errorbar(np.linspace(0, 1, diff_pre_interp_mean.shape[0]), diff_pre_interp_mean, yerr=diff_pre_interp_std, color="moccasin", zorder=-1000)
        ax1.plot(np.linspace(0, 1, long_sleep_mean.shape[0]), long_sleep_mean-long_sleep_std, color="grey", linewidth=1,
                 linestyle="--")
        ax1.plot(np.linspace(0, 1, long_sleep_mean.shape[0]), long_sleep_mean+long_sleep_std, color="grey", linewidth=1,
                 linestyle="--")
        ax1.plot(np.linspace(0, 1, long_sleep_mean.shape[0]), long_sleep_mean, color="black", linewidth=2,
                 label="Long rest using \n Acquisition and Recall")

        ax1.plot(np.linspace(0, 1, fam_sleep_mean.shape[0]), fam_sleep_mean-fam_sleep_std, color="lightblue", linewidth=1,
                 linestyle="--")
        ax1.plot(np.linspace(0, 1, fam_sleep_mean.shape[0]), fam_sleep_mean+fam_sleep_std, color="lightblue", linewidth=1,
                 linestyle="--")
        ax1.plot(np.linspace(0, 1, fam_sleep_mean.shape[0]), fam_sleep_mean, color="blue", linewidth=2,
                 label="Long rest using \n Familiar exploration (pre and post)")

        ax1.set_ylabel("Drift score")
        ax1.set_xlabel("Normalized duration")
        ax1.legend()
        # ax1.set_ylim(0, y_pre_max)
        res = [slopes_long, slopes_excl]
        bplot = ax2.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["Acquisition \n and recall", "Familiar exploration"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["gray", 'gray']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.grid(color="grey", axis="y")
        print(mannwhitneyu(res[0], res[1]))
        plt.ylim(0, 0.8)
        plt.xticks(rotation=45)
        plt.hlines(0.7, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05:
            plt.text(1.3, 0.7, "n.s.")
            plt.ylabel("Slope of drift score")
            plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_score_acquisition_recall_vs_familiar_exploration.svg"), transparent="True")
            plt.close()
        else:
            plt.show()



        plt.figure(figsize=(4,4))
        # first and second half for acquisition
        res = [np.hstack(pre_nrem_first), np.hstack(post_nrem_first), np.hstack(pre_nrem_second),
               np.hstack(post_nrem_second)]

        plt.boxplot(res, positions=[1, 2, 3, 4], patch_artist=True,
                          labels=["Acquisition", "Recall",
                                  "Acquisition", "Recall"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        plt.ylim(-65, 0)
        y_base=-16
        plt.hlines(y_base, 1, 2, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/4:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/4:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/4:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/4:
            plt.text(1.4, y_base, "*", color=c)
        plt.hlines(y_base, 3, 4, color=c)
        if mannwhitneyu(res[2], res[3])[1] > 0.05/4:
            plt.text(3.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.001/4:
            plt.text(3.4, y_base, "***", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.01/4:
            plt.text(3.4, y_base, "**", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.05/4:
            plt.text(3.4, y_base, "*", color=c)
        # first vs. second half NREM
        y_base=-10
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05/4:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001/4:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01/4:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05/4:
            plt.text(2, y_base, "*", color=c)

        # first vs. second half excluded
        y_base=-4
        plt.hlines(y_base, 2, 4, color=c)
        if mannwhitneyu(res[1], res[3])[1] > 0.05/4:
            plt.text(3, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.001/4:
            plt.text(3, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.01/4:
            plt.text(3, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.05/4:
            plt.text(3, y_base, "*", color=c)
        plt.ylabel("Log-likelihood")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("likelihood_4_models_pre_post.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        # first and second half for recall
        plt.figure(figsize=(4,4))
        res = [np.hstack(pre_excluded_first), np.hstack(post_excluded_first), np.hstack(pre_excluded_second),
                          np.hstack(post_excluded_second)]

        plt.boxplot(res, positions=[1, 2, 3, 4], patch_artist=True,
                          labels=["Familiar pre", "Familiar post",
                                  "Familiar pre", "Familiar post"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        plt.ylim(-65, 0)
        y_base=-16
        plt.hlines(y_base, 1, 2, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/4:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/4:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/4:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/4:
            plt.text(1.4, y_base, "*", color=c)
        plt.hlines(y_base, 3, 4, color=c)
        if mannwhitneyu(res[2], res[3])[1] > 0.05/4:
            plt.text(3.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.001/4:
            plt.text(3.4, y_base, "***", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.01/4:
            plt.text(3.4, y_base, "**", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.05/4:
            plt.text(3.4, y_base, "*", color=c)
        # first vs. second half NREM
        y_base=-10
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05/4:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001/4:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01/4:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05/4:
            plt.text(2, y_base, "*", color=c)

        # first vs. second half excluded
        y_base=-4
        plt.hlines(y_base, 2, 4, color=c)
        if mannwhitneyu(res[1], res[3])[1] > 0.05/4:
            plt.text(3, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.001/4:
            plt.text(3, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.01/4:
            plt.text(3, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.05/4:
            plt.text(3, y_base, "*", color=c)
        plt.ylabel("Log-likelihood")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("likelihood_4_models_fam_pre_post.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        frac_first_half_ = np.vstack(frac_first_half)

        res = frac_first_half_
        plt.figure(figsize=(4,3))
        bplot=plt.boxplot(res, positions=[1, 2, 3, 4], patch_artist=True,
                            labels=["Acquisition", "Familiar expl. PRE", "Recall", "Familiar expl. POST"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        plt.ylim(0,0.8)
        y_base=0.5
        plt.hlines(y_base, 1, 2, color=c)
        if mannwhitneyu(res[:, 0], res[:, 1])[1] > 0.05/2:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[:,0], res[:,1])[1] < 0.001/2:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[:,0], res[:,1])[1] < 0.01/2:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[:,0], res[:,1])[1] < 0.05/2:
            plt.text(1.4, y_base, "*", color=c)
        plt.hlines(y_base, 3, 4, color=c)
        if mannwhitneyu(res[:,2], res[:,3])[1] > 0.05/2:
            plt.text(3.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[:,2], res[:,3])[1] < 0.001/2:
            plt.text(3.4, y_base, "***", color=c)
        elif mannwhitneyu(res[:,2], res[:,3])[1] < 0.01/2:
            plt.text(3.4, y_base, "**", color=c)
        elif mannwhitneyu(res[:,2], res[:,3])[1] < 0.05/2:
            plt.text(3.4, y_base, "*", color=c)
        plt.ylabel("Proportion decoded")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("fraction_4_models_first_half.svg", transparent="True")
            plt.close()
        else:
            plt.show()



        frac_second_half_ = np.vstack(frac_second_half)

        res = frac_second_half_
        plt.figure(figsize=(4,3))
        bplot=plt.boxplot(res, positions=[1, 2, 3, 4], patch_artist=True,
                          labels=["Acquisition", "Familiar expl. PRE", "Recall", "Familiar expl. POST"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        plt.ylim(0,0.8)
        y_base=0.5
        plt.hlines(y_base, 1, 2, color=c)
        if mannwhitneyu(res[:, 0], res[:, 1])[1] > 0.05/2:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[:,0], res[:,1])[1] < 0.001/2:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[:,0], res[:,1])[1] < 0.01/2:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[:,0], res[:,1])[1] < 0.05/2:
            plt.text(1.4, y_base, "*", color=c)
        plt.hlines(y_base, 3, 4, color=c)
        if mannwhitneyu(res[:,2], res[:,3])[1] > 0.05/2:
            plt.text(3.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[:,2], res[:,3])[1] < 0.001/2:
            plt.text(3.4, y_base, "***", color=c)
        elif mannwhitneyu(res[:,2], res[:,3])[1] < 0.01/2:
            plt.text(3.4, y_base, "**", color=c)
        elif mannwhitneyu(res[:,2], res[:,3])[1] < 0.05/2:
            plt.text(3.4, y_base, "*", color=c)
        plt.ylabel("Proportion decoded")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("fraction_4_models_second_half.svg", transparent="True")
            plt.close()
        else:
            plt.show()


    def long_sleep_memory_drift_vs_duration_excluded_periods(self, save_fig=False, window_size_min=5):

        sim_ratio = []
        frac_excluded = []

        for i, session in enumerate(self.session_list):
            sim_ratio_, frac_excluded_ = \
                    session.long_sleep().memory_drift_vs_duration_excluded_periods(window_size_min=window_size_min)
            sim_ratio.append(sim_ratio_)
            frac_excluded.append(frac_excluded_)

        # compute slopes of drift score for original and jittered case
        slopes_original = []

        for drift_score in sim_ratio:
            coef_original = np.polyfit(np.linspace(0,1,drift_score.shape[0]), drift_score, 1)
            slopes_original.append(coef_original[0])

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        plt.figure(figsize=(3, 3))

        plt.scatter(frac_excluded, slopes_original)
        plt.xlabel("Fraction Awake")
        plt.ylabel("Slope of Drift score")
        plt.text(0.4, 0.12, "R="+str(np.round(pearsonr(frac_excluded, slopes_original)[0], 2)))
        plt.text(0.4, 0.05, "p="+str(np.round(pearsonr(frac_excluded, slopes_original)[1], 4)))
        plt.ylim(0,1)
        plt.xlim(0, 0.8)
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_score_vs_dur_excluded_periods.svg"), transparent="True")
        else:
            plt.show()

        print("HERE")

    def long_sleep_memory_drift_spike_bin_size(self, save_fig=False, nr_spikes_per_bin=6):

        sim_ratio = []
        sim_ratio_j = []

        for i, session in enumerate(self.session_list):
            sim_ratio_, sim_ratio_j_ =  session.long_sleep().memory_drift_spike_bin_size(plotting=False, nr_spikes_per_bin=nr_spikes_per_bin)
            sim_ratio.append(sim_ratio_)
            sim_ratio_j.append(sim_ratio_j_)

        # compute slopes of drift score for original and jittered case
        slopes_original = []
        slopes_jittered = []

        for drift_score, drift_score_j in zip(sim_ratio, sim_ratio_j):
            coef_original = np.polyfit(np.linspace(0,1,drift_score.shape[0]), drift_score, 1)
            slopes_original.append(coef_original[0])
            coef_j = np.polyfit(np.linspace(0,1,drift_score_j.shape[0]), drift_score_j, 1)
            slopes_jittered.append(coef_j[0])

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        cmap = matplotlib.cm.get_cmap('Oranges')
        colors_to_plot = cmap(np.linspace(0.1, 1, len(self.session_list)))
        plt.figure(figsize=(4, 3))
        for i, d in enumerate(sim_ratio_j):
            d_s = uniform_filter1d(d, int(d.shape[0]/5))
            plt.plot(np.linspace(0,1,d_s.shape[0]), d_s, color=colors_to_plot[i])
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("drift_nr_spikes_"+str(nr_spikes_per_bin)+"_all_sessions.svg", transparent="True")
        else:
            plt.show()


        plt.figure(figsize=(2, 3))

        res = [slopes_original, slopes_jittered]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["12 spikes",str(nr_spikes_per_bin)+" spikes"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        _, y_max = plt.gca().get_ylim()
        plt.ylim(0, y_max+0.2)
        plt.ylabel("Slope of drift score")
        plt.xticks(rotation=45)
        y_base = y_max+0.1
        print(mannwhitneyu(res[0], res[1]))
        plt.hlines(y_base, 1, 2)
        if mannwhitneyu(res[0], res[1])[1] > 0.05:
            plt.text(1.4, y_base, "n.s.")
        elif mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.4, y_base, "***")
        elif mannwhitneyu(res[0], res[1])[1] < 0.01:
            plt.text(1.4, y_base, "**")
        elif mannwhitneyu(res[0], res[1])[1] < 0.05:
            plt.text(1.4, y_base, "*")

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_score_nr_spikes_"+str(nr_spikes_per_bin)+"_all_sessions.svg"),
                        transparent="True")
        else:
            plt.show()

    def long_sleep_memory_drift_nrem_swr_vs_excluded_swr(self, save_fig=False, load_from_temp=True):

        pre_nrem = []
        pre_excluded = []
        post_nrem = []
        post_excluded = []
        sim_ratio = []
        sim_ratio_excluded = []

        for i, session in enumerate(self.session_list):
            pre_nrem_, pre_excluded_, post_nrem_, post_excluded_, sim_ratio_, sim_ratio_j_ = \
                session.long_sleep().memory_drift_nrem_swr_vs_swr_excluded_periods(plotting=False)
            pre_nrem.append(pre_nrem_)
            pre_excluded.append(pre_excluded_)
            post_nrem.append(post_nrem_)
            post_excluded.append(post_excluded_)
            sim_ratio.append(sim_ratio_)
            sim_ratio_excluded.append(sim_ratio_j_)

        slopes_long = []
        slopes_excl = []

        sim_ratio_s = []
        sim_ratio_excl_s = []

        # print drift score for long_sleep and fam_sleep
        for i, (sim_ratio_sess, sim_ratio_fam_sess) in enumerate(zip(sim_ratio, sim_ratio_excluded)):
            sim_ratio_sess_s = moving_average(sim_ratio_sess, int(sim_ratio_sess.shape[0]/10))
            sim_ratio_sess_fam_s = moving_average(sim_ratio_fam_sess, int(sim_ratio_fam_sess.shape[0]/10))
            coef_original = np.polyfit(np.linspace(0, 1, sim_ratio_sess_s .shape[0]), sim_ratio_sess_s , 1)
            slopes_long.append(coef_original[0])
            coef_fam = np.polyfit(np.linspace(0, 1, sim_ratio_sess_fam_s .shape[0]), sim_ratio_sess_fam_s , 1)
            slopes_excl.append(coef_fam[0])

            sim_ratio_s.append(sim_ratio_sess_s)
            sim_ratio_excl_s.append(sim_ratio_sess_fam_s)


        # using the whole rest:
        pre_nrem_all = np.hstack(pre_nrem)
        pre_excluded_all = np.hstack(pre_excluded)
        post_nrem_all = np.hstack(post_nrem)
        post_excluded_all = np.hstack(post_excluded)

        pre_nrem_first = []
        pre_nrem_second = []
        pre_excluded_first = []
        pre_excluded_second = []
        post_nrem_first = []
        post_nrem_second = []
        post_excluded_first = []
        post_excluded_second = []

        for pre_n, pre_e, post_n, post_e in zip(pre_nrem, pre_excluded, post_nrem, post_excluded):
            # apply a bit of smoothing
            pre_n_s = uniform_filter1d(pre_n, int(pre_n.shape[0]/10))
            pre_e_s = uniform_filter1d(pre_e, int(pre_e.shape[0]/10))
            post_n_s = uniform_filter1d(post_n, int(post_n.shape[0]/10))
            post_e_s = uniform_filter1d(post_e, int(post_e.shape[0]/10))

            pre_nrem_first.append(pre_n_s[:int(pre_n_s.shape[0]/2)])
            pre_nrem_second.append(pre_n_s[-int(pre_n_s.shape[0]/2):])
            pre_excluded_first.append(pre_e_s[:int(pre_e_s.shape[0]/2)])
            pre_excluded_second.append(pre_e_s[-int(pre_e_s.shape[0]/2):])
            post_nrem_first.append(post_n_s[:int(post_n_s.shape[0]/2)])
            post_nrem_second.append(post_n_s[-int(post_n_s.shape[0]/2):])
            post_excluded_first.append(post_e_s[:int(post_e_s.shape[0]/2)])
            post_excluded_second.append(post_e_s[-int(post_e_s.shape[0]/2):])

        max_len_long_sleep = np.max([x.shape[0] for x in sim_ratio_s])
        max_len_fam_sleep = np.max([x.shape[0] for x in sim_ratio_excl_s])

        # compute interpolated data
        long_sleep_inter = np.zeros((max_len_long_sleep, len(self.session_list)))
        fam_sleep_inter = np.zeros((max_len_fam_sleep, len(self.session_list)))

        for i, (sim_ratio_sess, sim_ratio_fam_sess) in enumerate(zip(sim_ratio_s, sim_ratio_excl_s)):
            long_sleep_inter[:, i] = np.interp(np.linspace(0, 1, max_len_long_sleep), np.linspace(0, 1,
                                                                  sim_ratio_sess.shape[0]), sim_ratio_sess)
            fam_sleep_inter[:, i] = np.interp(np.linspace(0, 1, max_len_fam_sleep), np.linspace(0, 1,
                                                    sim_ratio_fam_sess.shape[0]), sim_ratio_fam_sess)

        long_sleep_mean = np.mean(long_sleep_inter, axis=1)
        long_sleep_std = np.std(long_sleep_inter, axis=1)
        fam_sleep_mean = np.mean(fam_sleep_inter, axis=1)
        fam_sleep_std = np.std(fam_sleep_inter, axis=1)

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        fig = plt.figure(figsize=(8, 5))
        gs = fig.add_gridspec(9, 10)
        ax1 = fig.add_subplot(gs[:, :7])
        ax2 = fig.add_subplot(gs[:, 8:])

        y_long_max = np.max(long_sleep_mean+long_sleep_std)+0.5
        y_fam_max = np.max(fam_sleep_mean+fam_sleep_std)+0.5

        # ax1.errorbar(np.linspace(0, 1, diff_pre_interp_mean.shape[0]), diff_pre_interp_mean, yerr=diff_pre_interp_std, color="moccasin", zorder=-1000)
        ax1.plot(np.linspace(0, 1, long_sleep_mean.shape[0]), long_sleep_mean-long_sleep_std, color="grey", linewidth=1,
                 linestyle="--")
        ax1.plot(np.linspace(0, 1, long_sleep_mean.shape[0]), long_sleep_mean+long_sleep_std, color="grey", linewidth=1,
                 linestyle="--")
        ax1.plot(np.linspace(0, 1, long_sleep_mean.shape[0]), long_sleep_mean, color="black", linewidth=2,
                 label="NREM SWRs")

        ax1.plot(np.linspace(0, 1, fam_sleep_mean.shape[0]), fam_sleep_mean-fam_sleep_std, color="lightblue", linewidth=1,
                 linestyle="--")
        ax1.plot(np.linspace(0, 1, fam_sleep_mean.shape[0]), fam_sleep_mean+fam_sleep_std, color="lightblue", linewidth=1,
                 linestyle="--")
        ax1.plot(np.linspace(0, 1, fam_sleep_mean.shape[0]), fam_sleep_mean, color="blue", linewidth=2,
                 label="awake SWRs")

        ax1.set_ylabel("Drift score")
        ax1.set_xlabel("Normalized duration")
        ax1.legend()
        # ax1.set_ylim(0, y_pre_max)
        res = [slopes_long, slopes_excl]
        bplot = ax2.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["NREM SWR", "awake SWR"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["gray", 'gray']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.grid(color="grey", axis="y")
        print(mannwhitneyu(res[0], res[1]))
        plt.ylim(-0.2, 1.2)
        plt.xticks(rotation=45)
        plt.hlines(1, 1,2, color=c)
        if mannwhitneyu(res[0], res[1])[1] < 0.01:
            plt.text(1.3, 1, "**")
        plt.ylabel("Slope of drift score")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_score_awake_nrem_swr.svg"), transparent="True")
            plt.close()
        else:
            plt.show()



        plt.figure(figsize=(4,4))
        # first and second half for acquisition
        res = [np.hstack(pre_nrem_first), np.hstack(pre_excluded_first), np.hstack(pre_nrem_second),
               np.hstack(pre_excluded_second)]

        bplot=plt.boxplot(res, positions=[1, 2, 3, 4], patch_artist=True,
                          labels=["NREM SWR", "awake SWR",
                                  "NREM SWR", "awake SWR"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        plt.ylim(-50,-10)
        y_base=-28
        plt.hlines(y_base, 1, 2, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/4:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/4:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/4:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/4:
            plt.text(1.4, y_base, "*", color=c)
        plt.hlines(y_base, 3, 4, color=c)
        if mannwhitneyu(res[2], res[3])[1] > 0.05/4:
            plt.text(3.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.001/4:
            plt.text(3.4, y_base, "***", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.01/4:
            plt.text(3.4, y_base, "**", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.05/4:
            plt.text(3.4, y_base, "*", color=c)
        # first vs. second half NREM
        y_base=-22
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05/4:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001/4:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01/4:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05/4:
            plt.text(2, y_base, "*", color=c)

        # first vs. second half excluded
        y_base=-16
        plt.hlines(y_base, 2, 4, color=c)
        if mannwhitneyu(res[1], res[3])[1] > 0.05/4:
            plt.text(3, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.001/4:
            plt.text(3, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.01/4:
            plt.text(3, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.05/4:
            plt.text(3, y_base, "*", color=c)
        plt.ylabel("Log-likelihood")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("likelihood_pre_nrem_swr_excluded_swr_first_second_half.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        # first and second half for recall
        plt.figure(figsize=(4,4))
        res = [np.hstack(post_nrem_first), np.hstack(post_excluded_first), np.hstack(post_nrem_second),
                          np.hstack(post_excluded_second)]

        bplot=plt.boxplot(res, positions=[1, 2, 3, 4], patch_artist=True,
                          labels=["NREM SWR", "awake SWR",
                                  "NREM SWR", "awake SWR"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        plt.ylim(-50,-10)
        y_base=-28
        plt.hlines(y_base, 1, 2, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/4:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/4:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/4:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/4:
            plt.text(1.4, y_base, "*", color=c)
        plt.hlines(y_base, 3, 4, color=c)
        if mannwhitneyu(res[2], res[3])[1] > 0.05/4:
            plt.text(3.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.001/4:
            plt.text(3.4, y_base, "***", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.01/4:
            plt.text(3.4, y_base, "**", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.05/4:
            plt.text(3.4, y_base, "*", color=c)
        # first vs. second half NREM
        y_base=-22
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05/4:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001/4:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01/4:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05/4:
            plt.text(2, y_base, "*", color=c)

        # first vs. second half excluded
        y_base=-16
        plt.hlines(y_base, 2, 4, color=c)
        if mannwhitneyu(res[1], res[3])[1] > 0.05/4:
            plt.text(3, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.001/4:
            plt.text(3, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.01/4:
            plt.text(3, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.05/4:
            plt.text(3, y_base, "*", color=c)
        plt.ylabel("Log-likelihood")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("likelihood_post_nrem_swr_excluded_swr_first_second_half.svg", transparent="True")
            plt.close()
        else:
            plt.show()



        res = [pre_nrem_all, pre_excluded_all, post_nrem_all, post_excluded_all]
        bplot=plt.boxplot(res, positions=[1, 2, 3, 4], patch_artist=True,
                            labels=["Acq. NREM SWR", "Acq. EXCL. SWR", "Rec. NREM SWR", "Rec. EXCL. SWR"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        plt.ylim(-3,5)
        y_base=4
        plt.hlines(y_base, 1, 2, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/2:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/2:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/2:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/2:
            plt.text(1.4, y_base, "*", color=c)
        plt.hlines(y_base, 3, 4, color=c)
        if mannwhitneyu(res[2], res[3])[1] > 0.05/2:
            plt.text(3.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.001/2:
            plt.text(3.4, y_base, "***", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.01/2:
            plt.text(3.4, y_base, "**", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.05/2:
            plt.text(3.4, y_base, "*", color=c)
        plt.ylabel("Log-likelihood \n (z-scored per session and model)")
        plt.tight_layout()
        plt.show()
        print(mannwhitneyu(pre_nrem_all, pre_excluded_all))
        print(mannwhitneyu(post_nrem_all, post_excluded_all))

    def long_sleep_memory_drift_nrem_outside_swr(self, outside_swr_min_bin_size=0.5, save_fig=False):

        pre_nrem = []
        pre_excluded = []
        post_nrem = []
        post_excluded = []

        for i, session in enumerate(self.session_list):
            pre_nrem_, pre_excluded_, post_nrem_, post_excluded_ = \
                session.long_sleep().memory_drift_nrem_outside_swr(plotting=False,
                                                                   outside_swr_min_bin_size=outside_swr_min_bin_size)
            pre_nrem.append(pre_nrem_)
            pre_excluded.append(pre_excluded_)
            post_nrem.append(post_nrem_)
            post_excluded.append(post_excluded_)

        # using the whole rest:
        pre_nrem_all = np.hstack(pre_nrem)
        pre_excluded_all = np.hstack(pre_excluded)
        post_nrem_all = np.hstack(post_nrem)
        post_excluded_all = np.hstack(post_excluded)

        pre_nrem_first = []
        pre_nrem_second = []
        pre_excluded_first = []
        pre_excluded_second = []
        post_nrem_first = []
        post_nrem_second = []
        post_excluded_first = []
        post_excluded_second = []

        for pre_n, pre_e, post_n, post_e in zip(pre_nrem, pre_excluded, post_nrem, post_excluded):
            pre_nrem_first.append(pre_n[:int(pre_n.shape[0]/2)])
            pre_nrem_second.append(pre_n[int(pre_n.shape[0]/2):])
            pre_excluded_first.append(pre_e[:int(pre_e.shape[0]/2)])
            pre_excluded_second.append(pre_e[int(pre_e.shape[0]/2):])
            post_nrem_first.append(post_n[:int(post_n.shape[0]/2)])
            post_nrem_second.append(post_n[int(post_n.shape[0]/2):])
            post_excluded_first.append(post_e[:int(post_e.shape[0]/2)])
            post_excluded_second.append(post_e[int(post_e.shape[0]/2):])

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        plt.figure(figsize=(4,4))
        # first and second half for acquisition
        res = [np.hstack(pre_nrem_first), np.hstack(pre_excluded_first), np.hstack(pre_nrem_second),
               np.hstack(pre_excluded_second)]

        bplot=plt.boxplot(res, positions=[1, 2, 3, 4], patch_artist=True,
                          labels=["Acquisition\n NREM SWR", "Acquisition\n NREM outside SWR",
                                  "Acquisition\n NREM SWR", "Acquisition\n NREM outside SWR"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        plt.ylim(-2,6)
        y_base=3.2
        plt.hlines(y_base, 1, 2, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/4:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/4:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/4:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/4:
            plt.text(1.4, y_base, "*", color=c)
        plt.hlines(y_base, 3, 4, color=c)
        if mannwhitneyu(res[2], res[3])[1] > 0.05/4:
            plt.text(3.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.001/4:
            plt.text(3.4, y_base, "***", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.01/4:
            plt.text(3.4, y_base, "**", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.05/4:
            plt.text(3.4, y_base, "*", color=c)
        # first vs. second half NREM
        y_base=4
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05/4:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001/4:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01/4:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05/4:
            plt.text(2, y_base, "*", color=c)

        # first vs. second half excluded
        y_base=4.8
        plt.hlines(y_base, 2, 4, color=c)
        if mannwhitneyu(res[1], res[3])[1] > 0.05/4:
            plt.text(3, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.001/4:
            plt.text(3, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.01/4:
            plt.text(3, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.05/4:
            plt.text(3, y_base, "*", color=c)
        plt.ylabel("Log-likelihood \n (z-scored per model for each session)")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("likelihood_pre_nrem_swr_outside_swr_first_second_half.svg", transparent="True")
        else:
            plt.show()

        # first and second half for recall
        plt.figure(figsize=(4,4))
        res = [np.hstack(post_nrem_first), np.hstack(post_excluded_first), np.hstack(post_nrem_second),
               np.hstack(post_excluded_second)]

        bplot=plt.boxplot(res, positions=[1, 2, 3, 4], patch_artist=True,
                          labels=["Recall\n NREM SWR", "Recall\n NREM outside SWR",
                                  "Recall\n NREM SWR", "Recall\n NREM outside SWR"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        plt.ylim(-2.5,6)
        y_base=3.2
        plt.hlines(y_base, 1, 2, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/4:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/4:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/4:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/4:
            plt.text(1.4, y_base, "*", color=c)
        plt.hlines(y_base, 3, 4, color=c)
        if mannwhitneyu(res[2], res[3])[1] > 0.05/4:
            plt.text(3.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.001/4:
            plt.text(3.4, y_base, "***", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.01/4:
            plt.text(3.4, y_base, "**", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.05/4:
            plt.text(3.4, y_base, "*", color=c)
        # first vs. second half NREM
        y_base=4
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05/4:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001/4:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01/4:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05/4:
            plt.text(2, y_base, "*", color=c)

        # first vs. second half excluded
        y_base=4.8
        plt.hlines(y_base, 2, 4, color=c)
        if mannwhitneyu(res[1], res[3])[1] > 0.05/4:
            plt.text(3, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.001/4:
            plt.text(3, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.01/4:
            plt.text(3, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[3])[1] < 0.05/4:
            plt.text(3, y_base, "*", color=c)
        plt.ylabel("Log-likelihood \n (z-scored per model for each session)")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("likelihood_post_nrem_swr_outside_swr_first_second_half.svg", transparent="True")
        else:
            plt.show()



        res = [pre_nrem_all, pre_excluded_all, post_nrem_all, post_excluded_all]
        bplot=plt.boxplot(res, positions=[1, 2, 3, 4], patch_artist=True,
                          labels=["Acq. NREM SWR", "Acq. EXCL. SWR", "Rec. NREM SWR", "Rec. EXCL. SWR"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        plt.ylim(-3,5)
        y_base=4
        plt.hlines(y_base, 1, 2, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/2:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/2:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/2:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/2:
            plt.text(1.4, y_base, "*", color=c)
        plt.hlines(y_base, 3, 4, color=c)
        if mannwhitneyu(res[2], res[3])[1] > 0.05/2:
            plt.text(3.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.001/2:
            plt.text(3.4, y_base, "***", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.01/2:
            plt.text(3.4, y_base, "**", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.05/2:
            plt.text(3.4, y_base, "*", color=c)
        plt.ylabel("Log-likelihood \n (z-scored per session and model)")
        plt.tight_layout()
        plt.show()
        print(mannwhitneyu(pre_nrem_all, pre_excluded_all))
        print(mannwhitneyu(post_nrem_all, post_excluded_all))

    def long_sleep_memory_drift_nrem_outside_swr_vs_jitter(self, save_fig=False):

        pre_nrem = []
        pre_nrem_jitter = []
        pre_excluded = []
        post_nrem = []
        post_nrem_jitter = []
        post_excluded = []

        for i, session in enumerate(self.session_list):
            pre_nrem_, pre_excluded_, pre_nrem_jitter_, post_nrem_, post_excluded_, post_nrem_jitter_ = \
                session.long_sleep().memory_drift_nrem_outside_swr_vs_jitter(plotting=False)
            pre_nrem.append(pre_nrem_)
            pre_nrem_jitter.append(pre_nrem_jitter_)
            pre_excluded.append(pre_excluded_)
            post_nrem.append(post_nrem_)
            post_nrem_jitter.append(post_nrem_jitter_)
            post_excluded.append(post_excluded_)

        # using the whole rest:
        pre_nrem_all = np.hstack(pre_nrem)
        pre_excluded_all = np.hstack(pre_excluded)
        post_nrem_all = np.hstack(post_nrem)
        post_excluded_all = np.hstack(post_excluded)

        pre_nrem_first = []
        pre_nrem_second = []
        pre_nrem_jitter_first = []
        pre_nrem_jitter_second = []
        pre_excluded_first = []
        pre_excluded_second = []
        post_nrem_first = []
        post_nrem_second = []
        post_nrem_jitter_first = []
        post_nrem_jitter_second = []
        post_excluded_first = []
        post_excluded_second = []

        for pre_n, pre_n_j, pre_e, post_n, post_n_j, post_e in zip(pre_nrem, pre_nrem_jitter, pre_excluded, post_nrem,
                                                                   post_nrem_jitter, post_excluded):
            pre_nrem_first.append(pre_n[:int(pre_n.shape[0]/2)])
            pre_nrem_second.append(pre_n[int(pre_n.shape[0]/2):])
            pre_nrem_jitter_first.append(pre_n_j[:int(pre_n_j.shape[0]/2)])
            pre_nrem_jitter_second.append(pre_n_j[int(pre_n_j.shape[0]/2):])
            pre_excluded_first.append(pre_e[:int(pre_e.shape[0]/2)])
            pre_excluded_second.append(pre_e[int(pre_e.shape[0]/2):])
            post_nrem_first.append(post_n[:int(post_n.shape[0]/2)])
            post_nrem_second.append(post_n[int(post_n.shape[0]/2):])
            post_nrem_jitter_first.append(post_n_j[:int(post_n_j.shape[0]/2)])
            post_nrem_jitter_second.append(post_n_j[int(post_n_j.shape[0]/2):])
            post_excluded_first.append(post_e[:int(post_e.shape[0]/2)])
            post_excluded_second.append(post_e[int(post_e.shape[0]/2):])

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        plt.figure(figsize=(6,4))
        # first and second half for acquisition
        res = [np.hstack(pre_nrem_jitter_first), np.hstack(pre_nrem_first), np.hstack(pre_excluded_first),
               np.hstack(pre_nrem_jitter_second), np.hstack(pre_nrem_second),np.hstack(pre_excluded_second)]

        bplot=plt.boxplot(res, positions=[1, 2, 3, 4, 5, 6], patch_artist=True,
                          labels=["Acquisition\n NREM SWR jit", "Acquisition\n NREM SWR", "Acquisition\n NREM outside SWR",
                                  "Acquisition\n NREM SWR jit", "Acquisition\n NREM SWR", "Acquisition\n NREM outside SWR"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        plt.ylim(-2,5)
        y_base=3.2
        plt.hlines(y_base, 1, 2, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/4:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/4:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/4:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/4:
            plt.text(1.4, y_base, "*", color=c)
        # first vs. second half NREM
        y_base=4
        plt.hlines(y_base, 2, 3, color=c)
        if mannwhitneyu(res[1], res[2])[1] > 0.05/4:
            plt.text(2.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.001/4:
            plt.text(2.4, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.01/4:
            plt.text(2.4, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.05/4:
            plt.text(2.4, y_base, "*", color=c)

        # first vs. second half excluded
        y_base=3
        plt.hlines(y_base, 4, 5, color=c)
        if mannwhitneyu(res[4], res[3])[1] > 0.05/4:
            plt.text(4.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[4], res[3])[1] < 0.001/4:
            plt.text(4.4, y_base, "***", color=c)
        elif mannwhitneyu(res[4], res[3])[1] < 0.01/4:
            plt.text(4.4, y_base, "**", color=c)
        elif mannwhitneyu(res[4], res[3])[1] < 0.05/4:
            plt.text(4.4, y_base, "*", color=c)

            # first vs. second half excluded
        y_base=4
        plt.hlines(y_base, 5, 6, color=c)
        if mannwhitneyu(res[4], res[5])[1] > 0.05/4:
            plt.text(5.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[4], res[5])[1] < 0.001/4:
            plt.text(5.4, y_base, "***", color=c)
        elif mannwhitneyu(res[4], res[5])[1] < 0.01/4:
            plt.text(5.4, y_base, "**", color=c)
        elif mannwhitneyu(res[4], res[5])[1] < 0.05/4:
            plt.text(5.4, y_base, "*", color=c)
        plt.ylabel("Log-likelihood \n (z-scored per model for each session)")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("likelihood_pre_nrem_swr_outside_swr_vs_jit_first_second_half.svg", transparent="True")
        else:
            plt.show()

        # first and second half for recall
        plt.figure(figsize=(6,4))
        res = [np.hstack(post_nrem_jitter_first), np.hstack(post_nrem_first), np.hstack(post_excluded_first),
               np.hstack(post_nrem_jitter_second), np.hstack(post_nrem_second),
               np.hstack(post_excluded_second)]


        bplot=plt.boxplot(res, positions=[1, 2, 3, 4, 5, 6], patch_artist=True,
                          labels=["Recall\n NREM SWR jit", "Recall\n NREM SWR", "Recall\n NREM outside SWR",
                                  "Recall\n NREM SWR jit", "Recall\n NREM SWR", "Recall\n NREM outside SWR"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        plt.ylim(-2,5)
        y_base=3.2
        plt.hlines(y_base, 1, 2, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/4:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/4:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/4:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/4:
            plt.text(1.4, y_base, "*", color=c)
        # first vs. second half NREM
        y_base=4
        plt.hlines(y_base, 2, 3, color=c)
        if mannwhitneyu(res[1], res[2])[1] > 0.05/4:
            plt.text(2.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.001/4:
            plt.text(2.4, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.01/4:
            plt.text(2.4, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.05/4:
            plt.text(2.4, y_base, "*", color=c)

        # first vs. second half excluded
        y_base=3
        plt.hlines(y_base, 4, 5, color=c)
        if mannwhitneyu(res[4], res[3])[1] > 0.05/4:
            plt.text(4.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[4], res[3])[1] < 0.001/4:
            plt.text(4.4, y_base, "***", color=c)
        elif mannwhitneyu(res[4], res[3])[1] < 0.01/4:
            plt.text(4.4, y_base, "**", color=c)
        elif mannwhitneyu(res[4], res[3])[1] < 0.05/4:
            plt.text(4.4, y_base, "*", color=c)

            # first vs. second half excluded
        y_base=4
        plt.hlines(y_base, 5, 6, color=c)
        if mannwhitneyu(res[4], res[5])[1] > 0.05/4:
            plt.text(5.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[4], res[5])[1] < 0.001/4:
            plt.text(5.4, y_base, "***", color=c)
        elif mannwhitneyu(res[4], res[5])[1] < 0.01/4:
            plt.text(5.4, y_base, "**", color=c)
        elif mannwhitneyu(res[4], res[5])[1] < 0.05/4:
            plt.text(5.4, y_base, "*", color=c)
        plt.ylabel("Log-likelihood \n (z-scored per model for each session)")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("likelihood_post_nrem_swr_outside_swr_vs_jit_first_second_half.svg", transparent="True")
        else:
            plt.show()



        res = [pre_nrem_all, pre_excluded_all, post_nrem_all, post_excluded_all]
        bplot=plt.boxplot(res, positions=[1, 2, 3, 4], patch_artist=True,
                          labels=["Acq. NREM SWR", "Acq. EXCL. SWR", "Rec. NREM SWR", "Rec. EXCL. SWR"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        plt.ylim(-3,5)
        y_base=4
        plt.hlines(y_base, 1, 2, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/2:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/2:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/2:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/2:
            plt.text(1.4, y_base, "*", color=c)
        plt.hlines(y_base, 3, 4, color=c)
        if mannwhitneyu(res[2], res[3])[1] > 0.05/2:
            plt.text(3.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.001/2:
            plt.text(3.4, y_base, "***", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.01/2:
            plt.text(3.4, y_base, "**", color=c)
        elif mannwhitneyu(res[2], res[3])[1] < 0.05/2:
            plt.text(3.4, y_base, "*", color=c)
        plt.ylabel("Log-likelihood \n (z-scored per session and model)")
        plt.tight_layout()
        plt.show()
        print(mannwhitneyu(pre_nrem_all, pre_excluded_all))
        print(mannwhitneyu(post_nrem_all, post_excluded_all))

    def long_sleep_memory_drift_nrem_outside_swr_rem_vs_jitter(self, save_fig=False, load_from_temp=False):

        pre_nrem = []
        pre_nrem_jitter = []
        pre_excluded = []
        pre_excluded_jitter = []
        pre_rem = []
        pre_rem_jitter = []
        post_nrem = []
        post_nrem_jitter = []
        post_excluded = []
        post_excluded_jitter = []
        post_rem = []
        post_rem_jitter = []
        modes_original_pre = []
        modes_excluded_pre = []
        modes_rem_pre = []
        modes_original_post = []
        modes_excluded_post = []
        modes_rem_post = []

        for i, session in enumerate(self.session_list):
            if load_from_temp:
                infile = open("temp_data/outside_rem_swr/" + session.session_name, 'rb')
                results_file = pickle.load(infile)
                results = results_file["results"]
                infile.close()
            else:
                results = session.long_sleep().memory_drift_nrem_outside_swr_vs_jitter_and_rem(plotting=False)
            pre_nrem.append(results[0])
            pre_nrem_jitter.append(results[1])
            pre_excluded.append(results[2])
            pre_excluded_jitter.append(results[3])
            pre_rem.append(results[4])
            pre_rem_jitter.append(results[5])
            post_nrem.append(results[6])
            post_nrem_jitter.append(results[7])
            post_excluded.append(results[8])
            post_excluded_jitter.append(results[9])
            post_rem.append(results[10])
            post_rem_jitter.append(results[11])

            modes_original_pre.append(results[12])
            modes_excluded_pre.append(results[13])
            modes_rem_pre.append(results[14])
            modes_original_post.append(results[15])
            modes_excluded_post.append(results[16])
            modes_rem_post.append(results[17])

        # plot fraction of decoding state in NREM SWR, outside SWR, REM
        # --------------------------------------------------------------------------------------------------------------
        # if save_fig:
        #     plt.style.use('default')
        #     c = "black"
        # else:
        #     c = "white"
        #
        # fig = plt.figure(figsize=(7, 6))
        # gs = fig.add_gridspec(23, 23)
        # ax1 = fig.add_subplot(gs[:10, 1:11])
        # ax1.scatter(modes_original_pre[2], modes_excluded_pre[2])
        # ax1.set_xlim(-0.01, 0.2)
        # ax1.set_ylim(-0.01, 0.2)
        # ax1.set_ylabel("Fraction state \n decoded outside NREM SWR")
        # ax1.set_title("Acquisition")
        # ax1.text(0, 0.18, "R = "+str(np.round(pearsonr(modes_original_pre[2], modes_excluded_pre[2])[0], 2)))
        # ax1.text(0, 0.16, "p = "+str(pearsonr(modes_original_pre[2], modes_excluded_pre[2])[1]))
        # ax1.set_xticks([])
        # ax2 = fig.add_subplot(gs[:10, 13:])
        # ax2.scatter(modes_original_post[2], modes_excluded_post[2])
        # ax2.set_xlim(-0.01, 0.2)
        # ax2.set_ylim(-0.01, 0.2)
        # ax2.set_yticks([])
        # ax2.set_xticks([])
        # ax2.set_title("Recall")
        # ax2.text(0, 0.18, "R = "+str(np.round(pearsonr(modes_original_post[2], modes_excluded_post[2])[0], 2)))
        # ax2.text(0, 0.16, "p = "+str(pearsonr(modes_original_post[2], modes_excluded_post[2])[1]))
        # ax3 = fig.add_subplot(gs[11:21, 1:11])
        # ax3.scatter(modes_original_pre[2], modes_rem_pre[2])
        # ax3.set_xlim(-0.01, 0.2)
        # ax3.set_ylim(-0.01, 0.2)
        # ax3.set_ylabel("Fraction state \n decoded REM")
        # ax3.set_xlabel("Fraction state \n decoded NREM SWR")
        # ax3.text(0, 0.18, "R = "+str(np.round(pearsonr(modes_original_pre[2], modes_rem_pre[2])[0], 2)))
        # ax3.text(0, 0.16, "p = "+str(pearsonr(modes_original_pre[2], modes_rem_pre[2])[1]))
        # ax4 = fig.add_subplot(gs[11:21, 13:])
        # ax4.scatter(modes_original_post[2], modes_rem_post[2])
        # ax4.text(0, 0.18, "R = "+str(np.round(pearsonr(modes_original_post[2], modes_rem_post[2])[0], 2)))
        # ax4.text(0, 0.16, "p = "+str(pearsonr(modes_original_post[2], modes_rem_post[2])[1]))
        # ax4.set_xlim(-0.01, 0.2)
        # ax4.set_ylim(-0.01, 0.2)
        # ax4.set_yticks([])
        # ax4.set_xlabel("Fraction state \n decoded NREM SWR")
        # plt.tight_layout()
        # if save_fig:
        #     plt.rcParams['svg.fonttype'] = 'none'
        #     plt.savefig("frac_decoded_pre_post_nrem_swr_outside_rem_example.svg", transparent="True")
        # else:
        #     plt.show()
        #
        # fig = plt.figure(figsize=(3, 2))
        # gs = fig.add_gridspec(23, 23)
        # ax1 = fig.add_subplot(gs[:10, 1:11])
        # for sess_id, (mo_pre, me_pre) in enumerate(zip(modes_original_pre, modes_excluded_pre)):
        #     plt.scatter(sess_id+1, pearsonr(mo_pre, me_pre)[0], color="grey")
        # ax1.set_xlim(0, 8)
        # ax1.set_ylim(0, 1.1)
        # ax1.set_ylabel("R")
        # ax1.set_xlabel("Session ID")
        # ax1.set_xticks([1, 7], [1, 7])
        # ax2 = fig.add_subplot(gs[:10, 13:])
        # for sess_id, (mo_pre, me_pre) in enumerate(zip(modes_original_post, modes_excluded_post)):
        #     plt.scatter(sess_id+1, pearsonr(mo_pre, me_pre)[0], color="grey")
        # ax2.set_xlim(0, 8)
        # ax2.set_ylim(0, 1.1)
        # ax2.set_ylabel("R")
        # ax2.set_xlabel("Session ID")
        # ax2.set_xticks([1, 7], [1, 7])
        # ax3 = fig.add_subplot(gs[11:21, 1:11])
        # for sess_id, (mo_pre, me_pre) in enumerate(zip(modes_original_pre, modes_rem_pre)):
        #     plt.scatter(sess_id+1, pearsonr(mo_pre, me_pre)[0], color="grey")
        # ax3.set_xlim(0, 8)
        # ax3.set_ylim(0, 1.1)
        # ax3.set_ylabel("R")
        # ax3.set_xlabel("Session ID")
        # ax3.set_xticks([1, 7], [1, 7])
        # ax4 = fig.add_subplot(gs[11:21, 13:])
        # for sess_id, (mo_pre, me_pre) in enumerate(zip(modes_original_post, modes_rem_post)):
        #     plt.scatter(sess_id+1, pearsonr(mo_pre, me_pre)[0], color="grey")
        # ax4.set_xlim(0, 8)
        # ax4.set_ylim(0, 1.1)
        # ax4.set_ylabel("R")
        # ax4.set_xlabel("Session ID")
        # ax4.set_xticks([1, 7], [1, 7])
        # plt.tight_layout()
        # if save_fig:
        #     plt.rcParams['svg.fonttype'] = 'none'
        #     plt.savefig("frac_decoded_pre_post_nrem_swr_outside_rem_r_all_sessions.svg", transparent="True")
        # else:
        #     plt.show()

        # get colors
        # --------------------------------------------------------------------------------------------------------

        # swr and swr jitter --> pre
        # --------------------------------------------------------------------------------------------------------
        pre_first_swr = []
        pre_last_swr = []

        for i, (pre_, pre_j_, post_) in enumerate(zip(pre_nrem, pre_nrem_jitter, post_nrem)):
            # only use data when PRE was decoded
            pre_dec = pre_ > post_
            pre_ = pre_[pre_dec]
            pre_j_ = pre_j_[pre_dec]

            diff_pre = pre_ - pre_j_

            diff_pre_smooth = uniform_filter1d(diff_pre, int(diff_pre.shape[0]/5))
            pre_first_swr.append(diff_pre_smooth[:int(diff_pre_smooth.shape[0]/2)])
            pre_last_swr.append(diff_pre_smooth[-int(diff_pre_smooth.shape[0]/2):])

        # swr and swr jitter --> post
        # --------------------------------------------------------------------------------------------------------
        post_first_swr = []
        post_last_swr = []

        for i, (pre_, post_j_, post_) in enumerate(zip(pre_nrem, post_nrem_jitter, post_nrem)):
            # only use data when PRE was decoded
            pre_dec = pre_ > post_
            post_ = post_[~pre_dec]
            post_j_ = post_j_[~pre_dec]

            diff_post = post_ - post_j_

            diff_post_smooth = uniform_filter1d(diff_post, int(diff_post.shape[0]/5))
            post_first_swr.append(diff_post_smooth[:int(diff_post_smooth.shape[0]/2)])
            post_last_swr.append(diff_post_smooth[-int(diff_post_smooth.shape[0]/2):])

        # outside swr and outside swr jitter --> pre
        # --------------------------------------------------------------------------------------------------------
        pre_first_outside_swr = []
        pre_last_outside_swr = []

        for i, (pre_, pre_j_, post_) in enumerate(zip(pre_excluded, pre_excluded_jitter, post_excluded)):
            # only use data when PRE was decoded
            pre_dec = pre_ > post_
            pre_ = pre_[pre_dec]
            pre_j_ = pre_j_[pre_dec]

            diff_pre = pre_ - pre_j_

            diff_pre_smooth = uniform_filter1d(diff_pre, int(diff_pre.shape[0]/2))
            pre_first_outside_swr.append(diff_pre_smooth[:int(diff_pre_smooth.shape[0]/2)])
            pre_last_outside_swr.append(diff_pre_smooth[-int(diff_pre_smooth.shape[0]/2):])

        # outside swr and outside swr jitter --> post
        # --------------------------------------------------------------------------------------------------------
        post_first_outside_swr = []
        post_last_outside_swr = []

        for i, (pre_, post_j_, post_) in enumerate(zip(pre_excluded, post_excluded_jitter, post_excluded)):
            # only use data when PRE was decoded
            pre_dec = pre_ > post_
            post_ = post_[~pre_dec]
            post_j_ = post_j_[~pre_dec]

            diff_post = post_ - post_j_

            diff_post_smooth = uniform_filter1d(diff_post, int(diff_post.shape[0]/2))
            post_first_outside_swr.append(diff_post_smooth[:int(diff_post_smooth.shape[0]/2)])
            post_last_outside_swr.append(diff_post_smooth[-int(diff_post_smooth.shape[0]/2):])

        # rem and rem jitter --> pre
        # --------------------------------------------------------------------------------------------------------
        pre_first_rem = []
        pre_last_rem = []

        for i, (pre_, pre_j_, post_) in enumerate(zip(pre_rem, pre_rem_jitter, post_rem)):
            # only use data when PRE was decoded
            pre_dec = pre_ > post_
            pre_ = pre_[pre_dec]
            pre_j_ = pre_j_[pre_dec]

            diff_pre = pre_ - pre_j_

            diff_pre_smooth = uniform_filter1d(diff_pre, int(diff_pre.shape[0] / 5))
            pre_first_rem.append(diff_pre_smooth[:int(diff_pre_smooth.shape[0] / 2)])
            pre_last_rem.append(diff_pre_smooth[-int(diff_pre_smooth.shape[0] / 2):])

        # rem and rem jitter --> post
        # --------------------------------------------------------------------------------------------------------
        post_first_rem = []
        post_last_rem = []

        for i, (pre_, post_j_, post_) in enumerate(zip(pre_rem, post_rem_jitter, post_rem)):
            # only use data when PRE was decoded
            pre_dec = pre_ > post_
            post_ = post_[~pre_dec]
            post_j_ = post_j_[~pre_dec]

            diff_post = post_ - post_j_

            diff_post_smooth = uniform_filter1d(diff_post, int(diff_post.shape[0] / 5))
            post_first_rem.append(diff_post_smooth[:int(diff_post_smooth.shape[0] / 2)])
            post_last_rem.append(diff_post_smooth[-int(diff_post_smooth.shape[0] / 2):])

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        # plot first half --> Acquisition
        plt.figure(figsize=(3,4))
        res = [np.hstack(pre_first_swr), np.hstack(pre_first_outside_swr), np.hstack(pre_first_rem)]
        bplot = plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                            labels=["NREM SWRs", "NREM \noutside SWRs", "REM"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        plt.ylim(0, 12)
        plt.title("Acquisition: 1st half")
        plt.ylabel("Log-likelihood original \n - Log_likelihood jittered")
        if ttest_1samp(res[0], 0, alternative="greater")[1] < 0.001:
            plt.text(0.8, 4, "***", color=c)
        if ttest_1samp(res[1], 0, alternative="greater")[1] < 0.001:
            plt.text(1.8, 4, "***", color=c)
        if ttest_1samp(res[2], 0, alternative="greater")[1] < 0.001:
            plt.text(2.8, 4, "***", color=c)
        y_base = 5
        plt.hlines(y_base, 1, 1.9, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/3:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/3:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/3:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/3:
            plt.text(1.4, y_base, "*", color=c)
        # first vs. second half NREM
        plt.hlines(y_base, 2.1, 2.9, color=c)
        if mannwhitneyu(res[1], res[2])[1] > 0.05/3:
            plt.text(2.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.001/3:
            plt.text(2.4, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.01/3:
            plt.text(2.4, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.05/3:
            plt.text(2.4, y_base, "*", color=c)
        y_base = 6
        # first vs. second half excluded
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05/3:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001/3:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01/3:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05/3:
            plt.text(2, y_base, "*", color=c)

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "pre_outside_swr_first_half.svg"), transparent="True")
            plt.close()
        else:
            plt.show()


        # plot first half --> Acquisition
        plt.figure(figsize=(3,4))
        res = [np.hstack(pre_last_swr), np.hstack(pre_last_outside_swr), np.hstack(pre_last_rem)]
        bplot = plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                            labels=["NREM SWRs", "NREM \noutside SWRs", "REM"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        plt.ylim(0, 12)
        plt.title("Acquisition: 2nd half")
        plt.ylabel("Log-likelihood original \n - Log_likelihood jittered")
        if ttest_1samp(res[0], 0, alternative="greater")[1] < 0.001:
            plt.text(0.8, 4, "***", color=c)
        if ttest_1samp(res[1], 0, alternative="greater")[1] < 0.001:
            plt.text(1.8, 4, "***", color=c)
        if ttest_1samp(res[2], 0, alternative="greater")[1] < 0.001:
            plt.text(2.8, 4, "***", color=c)
        y_base = 5
        plt.hlines(y_base, 1, 1.9, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/3:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/3:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/3:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/3:
            plt.text(1.4, y_base, "*", color=c)
        # first vs. second half NREM
        plt.hlines(y_base, 2.1, 2.9, color=c)
        if mannwhitneyu(res[1], res[2])[1] > 0.05/3:
            plt.text(2.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.001/3:
            plt.text(2.4, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.01/3:
            plt.text(2.4, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.05/3:
            plt.text(2.4, y_base, "*", color=c)
        y_base = 6
        # first vs. second half excluded
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05/3:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001/3:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01/3:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05/3:
            plt.text(2, y_base, "*", color=c)

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "pre_outside_swr_second_half.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # plot first half --> Acquisition
        plt.figure(figsize=(3,4))
        res = [np.hstack(post_first_swr), np.hstack(post_first_outside_swr), np.hstack(post_first_rem)]
        bplot = plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                            labels=["NREM SWRs", "NREM \noutside SWRs", "REM"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        plt.ylim(0, 12)
        plt.title("Recall: 1st half")
        plt.ylabel("Log-likelihood original \n - Log_likelihood jittered")
        if ttest_1samp(res[0], 0, alternative="greater")[1] < 0.001:
            plt.text(0.8, 4, "***", color=c)
        if ttest_1samp(res[1], 0, alternative="greater")[1] < 0.001:
            plt.text(1.8, 4, "***", color=c)
        if ttest_1samp(res[2], 0, alternative="greater")[1] < 0.001:
            plt.text(2.8, 4, "***", color=c)
        y_base = 8
        plt.hlines(y_base, 1, 1.9, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/3:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/3:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/3:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/3:
            plt.text(1.4, y_base, "*", color=c)
        # first vs. second half NREM
        plt.hlines(y_base, 2.1, 2.9, color=c)
        if mannwhitneyu(res[1], res[2])[1] > 0.05/3:
            plt.text(2.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.001/3:
            plt.text(2.4, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.01/3:
            plt.text(2.4, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.05/3:
            plt.text(2.4, y_base, "*", color=c)
        y_base = 10
        # first vs. second half excluded
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05/3:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001/3:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01/3:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05/3:
            plt.text(2, y_base, "*", color=c)

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "post_outside_swr_first_half.svg"), transparent="True")
            plt.close()
        else:
            plt.show()


        # plot first half --> Acquisition
        plt.figure(figsize=(3,4))
        res = [np.hstack(post_last_swr), np.hstack(post_last_outside_swr), np.hstack(post_last_rem)]
        bplot = plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                            labels=["NREM SWRs", "NREM \noutside SWRs", "REM"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        plt.ylim(0, 12)
        plt.title("Recall: 2nd half")
        plt.ylabel("Log-likelihood original \n - Log_likelihood jittered")
        if ttest_1samp(res[0], 0, alternative="greater")[1] < 0.001:
            plt.text(0.8, 4, "***", color=c)
        if ttest_1samp(res[1], 0, alternative="greater")[1] < 0.001:
            plt.text(1.8, 4, "***", color=c)
        if ttest_1samp(res[2], 0, alternative="greater")[1] < 0.001:
            plt.text(2.8, 4, "***", color=c)
        y_base = 8
        plt.hlines(y_base, 1, 1.9, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/3:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/3:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/3:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/3:
            plt.text(1.4, y_base, "*", color=c)
        # first vs. second half NREM
        plt.hlines(y_base, 2.1, 2.9, color=c)
        if mannwhitneyu(res[1], res[2])[1] > 0.05/3:
            plt.text(2.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.001/3:
            plt.text(2.4, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.01/3:
            plt.text(2.4, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.05/3:
            plt.text(2.4, y_base, "*", color=c)
        y_base = 10
        # first vs. second half excluded
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05/3:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001/3:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01/3:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05/3:
            plt.text(2, y_base, "*", color=c)

        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "post_outside_swr_second_half.svg"), transparent="True")
            plt.close()
        else:
            plt.show()



    def long_sleep_memory_drift_nrem_outside_swr_rem_vs_jitter_spatial_and_goal_coding(self, save_fig=False):

        pre_states_decoded_median_distance_weighted = []
        post_states_decoded_median_distance_weighted = []
        pre_states_decoded_out_median_distance_weighted = []
        post_states_decoded_out_median_distance_weighted = []
        pre_states_decoded_rem_median_distance_weighted = []
        post_states_decoded_rem_median_distance_weighted = []
        pre_states_decoded_goal_coding_weighted = []
        post_states_decoded_goal_coding_weighted = []
        pre_states_decoded_out_goal_coding_weighted = []
        post_states_decoded_out_goal_coding_weighted = []
        pre_states_decoded_rem_goal_coding_weighted = []
        post_states_decoded_rem_goal_coding_weighted = []

        for i, session in enumerate(self.session_list):
            results = session.pre_long_sleep_post(data_to_use_long_sleep="ext_eegh").nrem_outside_swr_rem_states_spatial_and_goal_coding(plotting=False)
            pre_states_decoded_median_distance_weighted.append(results[0])
            post_states_decoded_median_distance_weighted.append(results[1])
            pre_states_decoded_out_median_distance_weighted.append(results[2])
            post_states_decoded_out_median_distance_weighted.append(results[3])
            pre_states_decoded_rem_median_distance_weighted.append(results[4])
            post_states_decoded_rem_median_distance_weighted.append(results[5])
            pre_states_decoded_goal_coding_weighted.append(results[6])
            post_states_decoded_goal_coding_weighted.append(results[7])
            pre_states_decoded_out_goal_coding_weighted.append(results[8])
            post_states_decoded_out_goal_coding_weighted.append(results[9])
            pre_states_decoded_rem_goal_coding_weighted.append(results[10])
            post_states_decoded_rem_goal_coding_weighted.append(results[11])
            # delete session to save memory?
            del session

        print("HERE")
        # using the whole rest:
        pre_states_decoded_median_distance_weighted_all = np.hstack(pre_states_decoded_median_distance_weighted)
        post_states_decoded_median_distance_weighted_all = np.hstack(post_states_decoded_median_distance_weighted)
        pre_states_decoded_out_median_distance_weighted_all = np.hstack(pre_states_decoded_out_median_distance_weighted)
        post_states_decoded_out_median_distance_weighted_all = np.hstack(post_states_decoded_out_median_distance_weighted)
        pre_states_decoded_rem_median_distance_weighted_all = np.hstack(pre_states_decoded_rem_median_distance_weighted)
        post_states_decoded_rem_median_distance_weighted_all = np.hstack(post_states_decoded_rem_median_distance_weighted)
        pre_states_decoded_goal_coding_weighted_all = np.hstack(pre_states_decoded_goal_coding_weighted)
        post_states_decoded_goal_coding_weighted_all = np.hstack(post_states_decoded_goal_coding_weighted)
        pre_states_decoded_out_goal_coding_weighted_all = np.hstack(pre_states_decoded_out_goal_coding_weighted)
        post_states_decoded_out_goal_coding_weighted_all = np.hstack(post_states_decoded_out_goal_coding_weighted)
        pre_states_decoded_rem_goal_coding_weighted_all = np.hstack(pre_states_decoded_rem_goal_coding_weighted)
        post_states_decoded_rem_goal_coding_weighted_all = np.hstack(post_states_decoded_rem_goal_coding_weighted)

        # filter out nans
        pre_states_decoded_median_distance_weighted_all = \
            pre_states_decoded_median_distance_weighted_all[~np.isnan(pre_states_decoded_median_distance_weighted_all)]
        post_states_decoded_median_distance_weighted_all = \
            post_states_decoded_median_distance_weighted_all[~np.isnan(post_states_decoded_median_distance_weighted_all)]
        pre_states_decoded_out_median_distance_weighted_all =\
            pre_states_decoded_out_median_distance_weighted_all[~np.isnan(pre_states_decoded_out_median_distance_weighted_all)]
        post_states_decoded_out_median_distance_weighted_all = \
            post_states_decoded_out_median_distance_weighted_all[~np.isnan(post_states_decoded_out_median_distance_weighted_all)]
        pre_states_decoded_rem_median_distance_weighted_all = \
            pre_states_decoded_rem_median_distance_weighted_all[~np.isnan(pre_states_decoded_rem_median_distance_weighted_all)]
        post_states_decoded_rem_median_distance_weighted_all = \
            post_states_decoded_rem_median_distance_weighted_all[~np.isnan(post_states_decoded_rem_median_distance_weighted_all)]
        pre_states_decoded_goal_coding_weighted_all = \
            pre_states_decoded_goal_coding_weighted_all[~np.isnan(pre_states_decoded_goal_coding_weighted_all)]
        post_states_decoded_goal_coding_weighted_all = \
            post_states_decoded_goal_coding_weighted_all[~np.isnan(post_states_decoded_goal_coding_weighted_all)]
        pre_states_decoded_out_goal_coding_weighted_all = \
            pre_states_decoded_out_goal_coding_weighted_all[~np.isnan(pre_states_decoded_out_goal_coding_weighted_all)]
        post_states_decoded_out_goal_coding_weighted_all = \
            post_states_decoded_out_goal_coding_weighted_all[~np.isnan(post_states_decoded_out_goal_coding_weighted_all)]
        pre_states_decoded_rem_goal_coding_weighted_all = \
            pre_states_decoded_rem_goal_coding_weighted_all[~np.isnan(pre_states_decoded_rem_goal_coding_weighted_all)]
        post_states_decoded_rem_goal_coding_weighted_all = \
            post_states_decoded_rem_goal_coding_weighted_all[~np.isnan(post_states_decoded_rem_goal_coding_weighted_all)]


        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        plt.figure(figsize=(6,6))
        # first and second half for acquisition
        res = [pre_states_decoded_median_distance_weighted_all, pre_states_decoded_out_median_distance_weighted_all,
               pre_states_decoded_rem_median_distance_weighted_all]

        bplot=plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                          labels=["NREM SWRs", "NREM \n outside SWRs", "REM"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        y_base = 1.3
        plt.hlines(y_base, 1, 1.9, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/3:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/3:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/3:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/3:
            plt.text(1.4, y_base, "*", color=c)
        # first vs. second half NREM
        plt.hlines(y_base, 2.1, 2.9, color=c)
        if mannwhitneyu(res[1], res[2])[1] > 0.05/3:
            plt.text(2.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.001/3:
            plt.text(2.4, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.01/3:
            plt.text(2.4, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.05/3:
            plt.text(2.4, y_base, "*", color=c)
        y_base = 1.5
        # first vs. second half excluded
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05/3:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001/3:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01/3:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05/3:
            plt.text(2, y_base, "*", color=c)
        plt.ylim(-0.01, 1.6)
        plt.ylabel("Weighted median \n distance of activations")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("decoded_modes_spatial_selectivity_pre_nrem_swr_outside_swr_rem.svg", transparent="True")
        else:
            plt.show()

        plt.figure(figsize=(6, 6))
        # first and second half for acquisition
        res = [post_states_decoded_median_distance_weighted_all, post_states_decoded_out_median_distance_weighted_all,
               post_states_decoded_rem_median_distance_weighted_all]

        bplot = plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                            labels=["NREM SWRs", "NREM \n outside SWRs", "REM"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        y_base = 1.3
        plt.hlines(y_base, 1, 1.9, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05 / 3:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001 / 3:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01 / 3:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05 / 3:
            plt.text(1.4, y_base, "*", color=c)
        # first vs. second half NREM
        plt.hlines(y_base, 2.1, 2.9, color=c)
        if mannwhitneyu(res[1], res[2])[1] > 0.05 / 3:
            plt.text(2.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.001 / 3:
            plt.text(2.4, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.01 / 3:
            plt.text(2.4, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.05 / 3:
            plt.text(2.4, y_base, "*", color=c)
        y_base = 1.5
        # first vs. second half excluded
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05 / 3:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001 / 3:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01 / 3:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05 / 3:
            plt.text(2, y_base, "*", color=c)
        plt.ylim(-0.01, 1.6)
        plt.ylabel("Weighted median \n distance of activations")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("decoded_modes_spatial_selectivity_post_nrem_swr_outside_swr_rem.svg", transparent="True")
        else:
            plt.show()

        # goal coding
        # --------------------------------------------------------------------------------------------------------------

        plt.figure(figsize=(6, 6))
        # first and second half for acquisition
        res = [pre_states_decoded_goal_coding_weighted_all, pre_states_decoded_out_goal_coding_weighted_all,
               pre_states_decoded_rem_goal_coding_weighted_all]

        bplot = plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                            labels=["NREM SWRs", "NREM \n outside SWRs", "REM"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        y_base = 0.03
        plt.hlines(y_base, 1, 1.9, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05 / 3:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001 / 3:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01 / 3:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05 / 3:
            plt.text(1.4, y_base, "*", color=c)
        # first vs. second half NREM
        plt.hlines(y_base, 2.1, 2.9, color=c)
        if mannwhitneyu(res[1], res[2])[1] > 0.05 / 3:
            plt.text(2.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.001 / 3:
            plt.text(2.4, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.01 / 3:
            plt.text(2.4, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.05 / 3:
            plt.text(2.4, y_base, "*", color=c)
        y_base = 0.035
        # first vs. second half excluded
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05 / 3:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001 / 3:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01 / 3:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05 / 3:
            plt.text(2, y_base, "*", color=c)
        plt.ylim(-0.001, 0.04)
        plt.ylabel("Weighted goal_coding")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("decoded_modes_goal_coding_pre_nrem_swr_outside_swr_rem.svg", transparent="True")
        else:
            plt.show()

        plt.figure(figsize=(6, 6))
        # first and second half for acquisition
        res = [post_states_decoded_goal_coding_weighted_all, post_states_decoded_out_goal_coding_weighted_all,
               post_states_decoded_rem_goal_coding_weighted_all]

        bplot = plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                            labels=["NREM SWRs", "NREM \n outside SWRs", "REM"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        y_base = 0.03
        plt.hlines(y_base, 1, 1.9, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05 / 3:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001 / 3:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01 / 3:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05 / 3:
            plt.text(1.4, y_base, "*", color=c)
        # first vs. second half NREM
        plt.hlines(y_base, 2.1, 2.9, color=c)
        if mannwhitneyu(res[1], res[2])[1] > 0.05 / 3:
            plt.text(2.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.001 / 3:
            plt.text(2.4, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.01 / 3:
            plt.text(2.4, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.05 / 3:
            plt.text(2.4, y_base, "*", color=c)
        y_base = 0.035
        # first vs. second half excluded
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05 / 3:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001 / 3:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01 / 3:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05 / 3:
            plt.text(2, y_base, "*", color=c)
        plt.ylim(-0.001, 0.04)
        plt.ylabel("Weighted goal_coding")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("decoded_modes_sgoal_coding_post_nrem_swr_outside_swr_rem.svg", transparent="True")
        else:
            plt.show()

    def long_sleep_memory_drift_nrem_outside_swr_rem_goal_coding(self, save_fig=False, only_decoded=False):
        pre = []
        post = []
        pre_out = []
        post_out = []
        pre_rem = []
        post_rem = []

        for i, session in enumerate(self.session_list):
            pre_, post_, pre_out_, post_out_, pre_rem_, post_rem_ = \
                session.pre_long_sleep_post(data_to_use_long_sleep="ext_eegh").nrem_outside_swr_rem_ising_goal_coding(only_decoded=only_decoded)

            pre.append(pre_)
            post.append(post_)
            pre_out.append(pre_out_)
            post_out.append(post_out_)
            pre_rem.append(pre_rem_)
            post_rem.append(post_rem_)



        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        plt.figure(figsize=(4,4))
        # first and second half for acquisition
        res = [pre, pre_out, pre_rem]

        bplot=plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                          labels=["NREM SWRs", "NREM \n outside SWRs", "REM"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        y_base = 0.6
        plt.hlines(y_base, 1, 1.9, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/3:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/3:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/3:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/3:
            plt.text(1.4, y_base, "*", color=c)
        # first vs. second half NREM
        plt.hlines(y_base, 2.1, 2.9, color=c)
        if mannwhitneyu(res[1], res[2])[1] > 0.05/3:
            plt.text(2.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.001/3:
            plt.text(2.4, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.01/3:
            plt.text(2.4, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.05/3:
            plt.text(2.4, y_base, "*", color=c)
        y_base = 0.7
        # first vs. second half excluded
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05/3:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001/3:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01/3:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05/3:
            plt.text(2, y_base, "*", color=c)
        plt.ylim(-0.01, 0.8)
        plt.ylabel("Probability of \n decoding goals")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            if only_decoded:
                plt.savefig("decoded_modes_goal_coding_pre_nrem_swr_outside_swr_rem_ising_only_decoded.svg", transparent="True")
            else:
                plt.savefig("decoded_modes_goal_coding_pre_nrem_swr_outside_swr_rem_ising.svg", transparent="True")
        else:
            plt.show()

        plt.figure(figsize=(4,4))
        # first and second half for acquisition
        res = [post, post_out, post_rem]

        bplot=plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                          labels=["NREM SWRs", "NREM \n outside SWRs", "REM"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        y_base = 0.6
        plt.hlines(y_base, 1, 1.9, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/3:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/3:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/3:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/3:
            plt.text(1.4, y_base, "*", color=c)
        # first vs. second half NREM
        plt.hlines(y_base, 2.1, 2.9, color=c)
        if mannwhitneyu(res[1], res[2])[1] > 0.05/3:
            plt.text(2.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.001/3:
            plt.text(2.4, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.01/3:
            plt.text(2.4, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.05/3:
            plt.text(2.4, y_base, "*", color=c)
        y_base =0.7
        # first vs. second half excluded
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05/3:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001/3:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01/3:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05/3:
            plt.text(2, y_base, "*", color=c)
        plt.ylim(-0.01, 0.8)
        plt.ylabel("Probability of \n decoding goals")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            if only_decoded:
                plt.savefig("decoded_modes_goal_coding_post_nrem_swr_outside_swr_rem_ising_only_decoded.svg", transparent="True")
            else:
                plt.savefig("decoded_modes_goal_coding_post_nrem_swr_outside_swr_rem_ising.svg", transparent="True")
        else:
            plt.show()

    def long_sleep_memory_drift_fam_sleep_long_sleep_swr(self, save_fig=False, only_decoded=True):

        pre_long = []
        pre_fam = []
        post_long = []
        post_fam = []
        sim_ratio = []
        sim_ratio_fam = []
        frac_pre_long_sleep = []
        frac_pre_fam_sleep = []

        for i, session in enumerate(self.session_list):
            pre_long_, pre_fam_, post_long_, post_fam_, sim_ratio_, sim_ratio_fam_, f_ls, f_fs = \
                session.sleep_long_sleep(data_to_use="ext_eegh",
                                         sleep_experiment_phase=["sleep_familiar"]).memory_drift_sleep_long_sleep(plotting=False, only_decoded=only_decoded)
            pre_long.append(pre_long_)
            pre_fam.append(pre_fam_)
            post_long.append(post_long_)
            post_fam.append(post_fam_)
            sim_ratio.append(sim_ratio_)
            sim_ratio_fam.append(sim_ratio_fam_)
            frac_pre_long_sleep.append(f_ls)
            frac_pre_fam_sleep.append(f_fs)

        cmap = matplotlib.cm.get_cmap('Greys')
        colors_to_plot_grey = cmap(np.linspace(0, 1, len(self.session_list)))
        cmap = matplotlib.cm.get_cmap('Reds')
        colors_to_plot_red = cmap(np.linspace(0, 1, len(self.session_list)))

        slopes_long = []
        slopes_fam = []

        sim_ratio_s = []
        sim_ratio_fam_s = []

        # print drift score for long_sleep and fam_sleep
        for i, (sim_ratio_sess, sim_ratio_fam_sess) in enumerate(zip(sim_ratio, sim_ratio_fam)):
            sim_ratio_sess_s = moving_average(sim_ratio_sess, int(sim_ratio_sess.shape[0]/10))
            sim_ratio_sess_fam_s = moving_average(sim_ratio_fam_sess, int(sim_ratio_fam_sess.shape[0]/10))
            coef_original = np.polyfit(np.linspace(0, 1, sim_ratio_sess_s .shape[0]), sim_ratio_sess_s , 1)
            slopes_long.append(coef_original[0])
            coef_fam = np.polyfit(np.linspace(0, 1, sim_ratio_sess_fam_s .shape[0]), sim_ratio_sess_fam_s , 1)
            slopes_fam.append(coef_fam[0])

            sim_ratio_s.append(sim_ratio_sess_s)
            sim_ratio_fam_s.append(sim_ratio_sess_fam_s)

            plt.plot(np.linspace(0,1,sim_ratio_sess_s.shape[0]),
                     sim_ratio_sess_s, color=colors_to_plot_grey[i])
            plt.plot(np.linspace(0,1,sim_ratio_sess_fam_s.shape[0]),
                     sim_ratio_sess_fam_s, color=colors_to_plot_red[i])
        plt.xlabel("Normalized duration")
        plt.ylabel("Drift score")
        plt.tight_layout()
        plt.show()

        max_len_long_sleep = np.max([x.shape[0] for x in sim_ratio_s])
        max_len_fam_sleep = np.max([x.shape[0] for x in sim_ratio_fam_s])

        # compute interpolated data
        long_sleep_inter = np.zeros((max_len_long_sleep, len(self.session_list)))
        fam_sleep_inter = np.zeros((max_len_fam_sleep, len(self.session_list)))

        for i, (sim_ratio_sess, sim_ratio_fam_sess) in enumerate(zip(sim_ratio_s, sim_ratio_fam_s)):
            long_sleep_inter[:, i] = np.interp(np.linspace(0, 1, max_len_long_sleep), np.linspace(0, 1,
                                        sim_ratio_sess.shape[0]), sim_ratio_sess)
            fam_sleep_inter[:, i] = np.interp(np.linspace(0, 1, max_len_fam_sleep), np.linspace(0, 1,
                                                            sim_ratio_fam_sess.shape[0]), sim_ratio_fam_sess)

        long_sleep_mean = np.mean(long_sleep_inter, axis=1)
        long_sleep_std = np.std(long_sleep_inter, axis=1)
        fam_sleep_mean = np.mean(fam_sleep_inter, axis=1)
        fam_sleep_std = np.std(fam_sleep_inter, axis=1)

        if save_fig:
                plt.style.use('default')
                c = "black"
        else:
            c = "white"

        fig = plt.figure(figsize=(2, 5))
        res = [frac_pre_long_sleep, frac_pre_fam_sleep]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["Long rest", "Familiar rest"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["gray", 'gray']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.grid(color="grey", axis="y")
        print(mannwhitneyu(res[0], res[1]))
        plt.ylim(0, 1.09)
        plt.xticks(rotation=45)
        plt.hlines(1, 1,2, color=c)
        plt.text(1.3, 1, "**")
        plt.ylabel("Fraction Acquisition \n decoded")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "familiar_sleep_long_sleep_frac_pre_decoded.svg"), transparent="True")
        else:
            plt.show()


        fig = plt.figure(figsize=(8, 5))
        gs = fig.add_gridspec(9, 10)
        ax1 = fig.add_subplot(gs[:, :7])
        ax2 = fig.add_subplot(gs[:, 8:])

        y_long_max = np.max(long_sleep_mean+long_sleep_std)+0.5
        y_fam_max = np.max(fam_sleep_mean+fam_sleep_std)+0.5

        # ax1.errorbar(np.linspace(0, 1, diff_pre_interp_mean.shape[0]), diff_pre_interp_mean, yerr=diff_pre_interp_std, color="moccasin", zorder=-1000)
        ax1.plot(np.linspace(0, 1, long_sleep_mean.shape[0]), long_sleep_mean-long_sleep_std, color="grey", linewidth=1,
                 linestyle="--")
        ax1.plot(np.linspace(0, 1, long_sleep_mean.shape[0]), long_sleep_mean+long_sleep_std, color="grey", linewidth=1,
                 linestyle="--")
        ax1.plot(np.linspace(0, 1, long_sleep_mean.shape[0]), long_sleep_mean, color="black", linewidth=2,
                 label="Long rest")

        ax1.plot(np.linspace(0, 1, fam_sleep_mean.shape[0]), fam_sleep_mean-fam_sleep_std, color="salmon", linewidth=1,
                 linestyle="--")
        ax1.plot(np.linspace(0, 1, fam_sleep_mean.shape[0]), fam_sleep_mean+fam_sleep_std, color="salmon", linewidth=1,
                 linestyle="--")
        ax1.plot(np.linspace(0, 1, fam_sleep_mean.shape[0]), fam_sleep_mean, color="red", linewidth=2,
                 label="Familiar rest")

        ax1.set_ylabel("Drift score")
        ax1.set_xlabel("Normalized duration")
        ax1.legend()
        # ax1.set_ylim(0, y_pre_max)
        res = [slopes_long, slopes_fam]
        bplot = ax2.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["Long rest", "Familiar rest"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["gray", 'gray']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.grid(color="grey", axis="y")
        print(mannwhitneyu(res[0], res[1]))
        plt.ylim(-0.2, 1)
        plt.xticks(rotation=45)
        plt.hlines(0.8, 1,2, color=c)
        plt.text(1.3, 0.7, "**")
        plt.ylabel("Slope of drift score")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "drift_score_familiar_sleep.svg"), transparent="True")
        else:
            plt.show()

        pre_long_first = []
        pre_long_second = []
        post_long_first = []
        post_long_second = []
        pre_fam_smooth = []
        post_fam_smooth = []

        for pre_l, post_l, pre_f, post_f in zip(pre_long, post_long, pre_fam, post_fam):

            pre_l_s = uniform_filter1d(pre_l, int(pre_l.shape[0]/10))
            post_l_s = uniform_filter1d(post_l, int(post_l.shape[0]/10))
            pre_fam_smooth.append(uniform_filter1d(pre_f, int(pre_f.shape[0]/10)))
            if post_f.shape[0] > 10:
                post_fam_smooth.append(uniform_filter1d(post_f, int(post_f.shape[0] / 10)))

            pre_long_first.append(pre_l_s[:int(pre_l_s.shape[0]/2)])
            pre_long_second.append(pre_l_s[-int(pre_l_s.shape[0]/2):])
            post_long_first.append(post_l_s[:int(post_l_s.shape[0]/2)])
            post_long_second.append(post_l_s[-int(post_l_s.shape[0]/2):])

        # Acquisition: fam_rest and long_sleep first/second half
        plt.figure(figsize=(4,4))
        res = [np.hstack(pre_fam_smooth), np.hstack(pre_long_first), np.hstack(pre_long_second)]

        bplot=plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                          labels=["SWR fam_rest", "SWR 1st half",
                                  "SWR 2nd half"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        plt.ylim(-50,-25)
        y_base=-32
        plt.hlines(y_base, 1, 1.9, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/3:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/3:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/3:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/3:
            plt.text(1.4, y_base, "*", color=c)
        plt.hlines(y_base, 2.1, 3, color=c)
        if mannwhitneyu(res[1], res[2])[1] > 0.05/3:
            plt.text(2.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.001/3:
            plt.text(2.4, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.01/3:
            plt.text(2.4, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.05/3:
            plt.text(2.4, y_base, "*", color=c)
        # first vs. second half NREM
        y_base=-28
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05/3:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001/3:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01/3:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05/3:
            plt.text(2, y_base, "*", color=c)
        plt.ylabel("Log-likelihood decoded state")
        plt.title("Acquisition")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("pre_likelihood_fam_rest_long_sleep_first_second_half.svg", transparent="True")
        else:
            plt.show()


        # Recall: fam_rest and long_sleep first/second half
        plt.figure(figsize=(4,4))
        res = [np.hstack(post_fam_smooth), np.hstack(post_long_first), np.hstack(post_long_second)]

        bplot=plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                          labels=["SWR fam_rest", "SWR 1st half",
                                  "SWR 2nd half"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        plt.ylim(-50,-25)
        y_base=-30
        plt.hlines(y_base, 1, 1.9, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/3:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/3:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/3:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/3:
            plt.text(1.4, y_base, "*", color=c)
        plt.hlines(y_base, 2.1, 3, color=c)
        if mannwhitneyu(res[1], res[2])[1] > 0.05/3:
            plt.text(2.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.001/3:
            plt.text(2.4, y_base, "***", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.01/3:
            plt.text(2.4, y_base, "**", color=c)
        elif mannwhitneyu(res[1], res[2])[1] < 0.05/3:
            plt.text(2.4, y_base, "*", color=c)
        # first vs. second half NREM
        y_base=-28
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05/3:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001/3:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01/3:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05/3:
            plt.text(2, y_base, "*", color=c)
        plt.ylabel("Log-likelihood decoded state")
        plt.title("Recall")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("post_likelihood_fam_rest_long_sleep_first_second_half.svg", transparent="True")
        else:
            plt.show()

    def long_sleep_memory_decoded_likelihoods(self, sleep_phase="rem"):

        pre = []
        post = []

        for i, session in enumerate(self.session_list):
            pre_, post_ = \
                session.long_sleep().memory_drift_sleep_phases_decoded_likelihoods(sleep_phase=sleep_phase)
            pre.append(pre_)
            post.append(post_)


        f=[]
        l=[]
        for i, pp in enumerate(pre):
            pp_s = uniform_filter1d(pp, int(pp.shape[0]/10))
            f.append(pp_s[:int(pp_s.shape[0]/2)])
            l.append(pp_s[-int(pp_s.shape[0]/2):])


        f_r=[]
        l_r=[]
        for i, pp in enumerate(post):
            pp_s = uniform_filter1d(pp, int(pp.shape[0]/10))
            f_r.append(pp_s[:int(pp_s.shape[0]/2)])
            l_r.append(pp_s[-int(pp_s.shape[0]/2):])

        c="white"
        res = [np.hstack(f), np.hstack(l)]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["1st half", "2nd half"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        plt.title("Acquisition: "+sleep_phase)
        plt.ylabel("Log-likelihood")
        plt.tight_layout()
        plt.show()

        res = [np.hstack(f_r), np.hstack(l_r)]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["1st half", "2nd half"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.xticks(rotation=45)
        plt.title("Recall: "+sleep_phase)
        plt.ylabel("Log-likelihood")
        plt.tight_layout()
        plt.show()

    def long_sleep_swr_profile_subsets(self, save_fig=False):

        stable = []
        dec = []
        inc = []

        for i, session in enumerate(self.session_list):
            s, d, i = \
                session.long_sleep(data_to_use="ext_eegh").swr_profile(plotting=False)
            stable.append(s)
            dec.append(d)
            inc.append(i)

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        plt.figure(figsize=(4,4))
        # first and second half for acquisition
        res = [np.hstack(stable), np.hstack(dec), np.hstack(inc)]

        bplot=plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                          labels=["persistent", "decreasing", "increasing"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        for patch, color in zip(bplot['boxes'], ["violet", "turquoise", "orange"]):
            patch.set_facecolor(color)
        plt.ylim(-3000,3000)
        y_base=2100
        plt.hlines(y_base, 1, 1.9, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/3:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/3:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/3:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/3:
            plt.text(1.4, y_base, "*", color=c)
        plt.hlines(y_base, 2.1, 2.9, color=c)
        if mannwhitneyu(res[2], res[1])[1] > 0.05/3:
            plt.text(2.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[2], res[1])[1] < 0.001/3:
            plt.text(2.4, y_base, "***", color=c)
        elif mannwhitneyu(res[2], res[1])[1] < 0.01/3:
            plt.text(2.4, y_base, "**", color=c)
        elif mannwhitneyu(res[2], res[1])[1] < 0.05/3:
            plt.text(2.4, y_base, "*", color=c)
        y_base=2400
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05/3:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001/3:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01/3:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05/3:
            plt.text(2, y_base, "*", color=c)
        plt.ylabel("SWR peak value (a.u.)")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("swr_profile_subsets.svg", transparent="True")
        else:
            plt.show()

    def long_sleep_duration(self):
        duration =[]
        for i, session in enumerate(self.session_list):
            dur_ =session.long_sleep().sleep_duration_h()
        duration.append(dur_)

    # </editor-fold>

    # <editor-fold desc="Pre_probe, PRE, POST, post_probe">
    # def pre_probe_pre_post_post_probe_rate_map_stability(self, cells_to_use="all", spatial_resolution=5,
    #                                                      nr_of_splits=3, save_fig=False, z_score=False,
    #                                                      normalize=False, plotting=False):
    #     """
    #     Computes rate map stability between pre_probe, pre, post, post_probe
    #
    #     :param cells_to_use: which cells to use ("all", "stable", "increasing", "decreasing")
    #     :type cells_to_use: str
    #     :param spatial_resolution: for rate maps in cm2
    #     :type spatial_resolution: int
    #     :param nr_of_splits: in how many chunks to split each experimental phase
    #     :type nr_of_splits: int
    #     :param save_fig: save as .svg
    #     :type save_fig: bool
    #     :param z_score: z-score correlations
    #     :type z_score: bool
    #     :param normalize: normalize all values outside the diagonal to lie between 0 and 1
    #     :type normalize: bool
    #     :param plotting: plot results
    #     :type plotting: bool
    #     :return: ratio pre_similarity_with_post/pre_similarity_with_pre
    #     :rtype: float
    #     """
    #     sim_matrices = []
    #     for session in self.session_list:
    #         pre_probe, pre = \
    #             session.cheeseboard_pre_prob_pre().phmm_modes_likelihoods(plotting=False, cells_to_use=cells_to_use)
    #         pre_probe_median_log_likeli.append(pre_probe)
    #         pre_median_log_likeli.append(pre)
    #
    #     if save_fig:
    #         plt.style.use('default')
    #     plt.figure(figsize=(4,6))
    #     # plt.figure(figsize=(3,4))
    #     col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
    #     for session_id, (first, second) in enumerate(zip(pre_probe_median_log_likeli, pre_median_log_likeli)):
    #         plt.scatter([0.1, 0.2], [first,second], label=str(session_id), color=col[session_id], zorder=session_id)
    #         plt.plot([0.1, 0.2], [first,second], color=col[session_id], zorder=session_id)
    #         plt.xticks([0.1, 0.2], ["Pre_probe", "PRE"])
    #     plt.ylabel("Median Log-likelihood of PRE states")
    #     plt.grid(axis="y", color="gray")
    #     plt.title(cells_to_use)
    #     plt.ylim(-17, -7)
    #     plt.legend()
    #     if save_fig:
    #         plt.rcParams['svg.fonttype'] = 'none'
    #         plt.savefig("pre_probe_pre_likeli_"+cells_to_use+".svg", transparent="True")
    #     else:
    #         plt.show()

    def compare_spatial_measures_and_firing_rates(self, spatial_resolution=2):

        firing_pre_stable, firing_pre_dec, firing_pre_inc, firing_sleep_stable, firing_sleep_dec, \
        firing_sleep_inc, firing_post_stable, firing_post_dec, firing_post_inc = \
            self.pre_long_sleep_post_firing_rates_all_cells(plotting=False, measure="mean")

        pre_stable_sk_s, post_stable_sk_s, pre_dec_sk_s, post_inc_sk_s, post_dec_sk_s, pre_inc_sk_s = \
            self.pre_post_cheeseboard_spatial_information(plotting=False, info_measure="skaggs_second",
                                                          remove_nan=False, spatial_resolution=spatial_resolution)
        pre_stable_spar, post_stable_spar, pre_dec_spar, post_inc_spar, post_dec_spar, pre_inc_spar = \
            self.pre_post_cheeseboard_spatial_information(plotting=False, info_measure="sparsity",
                                                          remove_nan=False, spatial_resolution=spatial_resolution)

        plt.scatter(pre_stable_sk_s, pre_stable_spar)
        plt.xlabel("Skaggs per second")
        plt.ylabel("Sparsity")
        plt.title("Stable (PRE)")
        plt.show()

        plt.scatter(firing_pre_stable, pre_stable_spar)
        plt.xlabel("Mean firing rate")
        plt.ylabel("Sparsity")
        plt.title("Stable (PRE)")
        plt.show()

        plt.scatter(firing_pre_stable, pre_stable_sk_s)
        plt.xlabel("Mean firing rate")
        plt.ylabel("Skaggs per second")
        plt.title("Stable (PRE)")
        plt.show()

        plt.scatter(pre_dec_sk_s, pre_dec_spar)
        plt.xlabel("Skaggs per second")
        plt.ylabel("Sparsity")
        plt.title("Decreasing (PRE)")
        plt.show()

        plt.scatter(firing_pre_dec, pre_dec_spar)
        plt.xlabel("Mean firing rate")
        plt.ylabel("Sparsity")
        plt.title("Decreasing (PRE)")
        plt.show()

        plt.scatter(firing_pre_dec, pre_dec_sk_s)
        plt.xlabel("Mean firing rate")
        plt.ylabel("Skaggs per second")
        plt.title("Decreasing (PRE)")
        plt.show()

    def post_cheeseboard_occupancy_around_goals(self, save_fig=False, first_half=True):

        around_goals = []
        wo_goals = []

        for session in self.session_list:
            occ_around_goals_per_cm2, occ_wo_goals_per_cm2 = \
                session.cheese_board(experiment_phase=["learning_cheeseboard_2"]).occupancy_around_goals(first_half=
                                                                                                         first_half)
            around_goals.append(occ_around_goals_per_cm2)
            wo_goals.append(occ_wo_goals_per_cm2)

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        plt.figure(figsize=(4, 5))
        plt.tight_layout()
        res = [around_goals, wo_goals]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["Around goals", "Away from goals"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["gray", 'gray']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Occupancy per spatial bin (s/m2)")
        plt.grid(color="grey", axis="y")
        print(mannwhitneyu(around_goals, wo_goals))
        plt.ylim(0, 0.6)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "occupancy_post_all_sessions.svg"), transparent="True")
        else:
            plt.show()

    def rate_map_stability_pre_probe_pre_post_post_probe(self, cells_to_use="all", spatial_resolution=5,
                                                         nr_of_splits=3, save_fig=False, z_score=False,
                                                         normalize=True, plotting=False, use_max=False):

        sim_matrices = []
        for session in self.session_list:
            # s_m = session.cheeseboard_pre_prob_pre_post_post_prob().rate_map_stability(cells_to_use=cells_to_use,
            s_m = session.pre_prob_pre_post_post_prob().rate_map_stability(cells_to_use=cells_to_use,
                                                                                    spatial_resolution=spatial_resolution,
                                                                                    plotting=False, nr_of_splits=nr_of_splits,
                                                                                       z_score=z_score, use_max=use_max)
            sim_matrices.append(s_m)

        map_similarity=np.nanmean(np.array(sim_matrices), axis=0)

        if normalize:
            #def norm_matrix(matrix):
            #   matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
            #    return matrix
            #map_similarity = squareform(norm_matrix(squareform(map_similarity-np.eye(map_similarity.shape[0]))))

            np.fill_diagonal(map_similarity, np.nan)
            map_similarity = (map_similarity - np.nanmin(map_similarity.flatten()))/(np.nanmax(map_similarity.flatten())- np.nanmin(map_similarity.flatten()))

        ratio_per_session = []
        # compute per session to have error bars
        for sim_matrix_session in sim_matrices:
            # plt.imshow(sim_matrix_session)
            # plt.xlim(-0.5, sim_matrix_session.shape[0] - 0.5)
            # plt.ylim(-0.5, sim_matrix_session.shape[0] - 0.5)
            # plt.colorbar()
            # plt.show()
            # normalize
            np.fill_diagonal(sim_matrix_session, np.nan)
            sim_normalized = (sim_matrix_session - np.nanmin(sim_matrix_session.flatten()))/(np.nanmax(sim_matrix_session.flatten())- np.nanmin(sim_matrix_session.flatten()))
            # normalize
            # sim_normalized = squareform(norm_matrix(squareform(sim_matrix_session-np.eye(sim_matrix_session.shape[0]))))
            # plt.imshow(sim_normalized)
            # plt.xlim(-0.5, sim_normalized.shape[0] - 0.5)
            # plt.ylim(-0.5, sim_normalized.shape[0] - 0.5)
            # plt.colorbar()
            # plt.show()
            pre_post = sim_normalized[
                       int((sim_normalized.shape[0] / 2)):int((sim_normalized.shape[0] / 2)) + (nr_of_splits),
                       int((sim_normalized.shape[0] / 2)) - (nr_of_splits):int((sim_normalized.shape[0] / 2))]
            # plt.imshow(pre_post, vmin=0, vmax=1)
            # plt.show()
            pre_similarity_with_post = np.mean(pre_post, axis=0)
            pre_pre_prob = sim_normalized[
                           int((sim_normalized.shape[0] / 2)) - (nr_of_splits):int((sim_normalized.shape[0] / 2)),
                           0:int(nr_of_splits)]
            # plt.imshow(pre_pre_prob, vmin=0, vmax=1)
            # plt.show()
            pre_similarity_with_pre = np.mean(pre_pre_prob, axis=1)

            ratio_per_session.append(pre_similarity_with_post/pre_similarity_with_pre)

        labels = []
        phases = np.array(["pre-probe", "learn-PRE", "POST", "post-probe"])
        for phase in phases:
            for subdiv in range(nr_of_splits):
                labels.append(phase + "_" + str(subdiv))

        pre_post = map_similarity[int((map_similarity.shape[0]/2)):int((map_similarity.shape[0]/2))+nr_of_splits,
                          int((map_similarity.shape[0]/2))-nr_of_splits:int((map_similarity.shape[0]/2))]

        pre_similarity_with_post = np.mean(pre_post, axis=0)

        pre_prob_pre = map_similarity[int((map_similarity.shape[0]/2)):int((map_similarity.shape[0]/2))+nr_of_splits,
                          int((map_similarity.shape[0]/2))-nr_of_splits:int((map_similarity.shape[0]/2))]


        probe_stable = map_similarity[int((map_similarity.shape[0]/2))-nr_of_splits:int((map_similarity.shape[0]/2)),
                       0:int(nr_of_splits)]

        pre_similarity_with_pre = np.mean(probe_stable, axis=1)

        if plotting or save_fig:
            if save_fig:
                plt.style.use('default')
            # plt.figure(figsize=(6,5))
            plt.imshow(map_similarity)
            plt.yticks(np.arange(map_similarity.shape[0]), labels)
            plt.xticks(np.arange(map_similarity.shape[0]), labels, rotation='vertical')
            plt.xlim(-0.5, map_similarity.shape[0] - 0.5)
            plt.ylim(-0.5, map_similarity.shape[0] - 0.5)
            # plt.ylim(0, map_similarity.shape[0])
            a = plt.colorbar()
            if z_score:
                a.set_label("Mean population vector correlation (z-scored)")
            elif normalize:
                a.set_label("Mean population vector correlation (normalized)")
            else:
                a.set_label("Mean population vector correlation")
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "rate_map_stability_"+cells_to_use+".svg"), transparent="True")
                plt.close()
            else:
                plt.show()

            plt.plot(pre_similarity_with_post, label="similarity with post")
            plt.plot(pre_similarity_with_pre, label="similarity with pre-probe")
            xlabels = ["pre_"+str(i) for i in range(nr_of_splits)]
            xlabels_pos = [i for i in range(nr_of_splits)]
            plt.xticks(xlabels_pos, xlabels)
            if normalize:
                plt.ylabel("Mean population vector correlation (normalized)")
            else:
                plt.ylabel("Mean population vector correlation (normalized)")
            plt.legend()
            plt.show()
        else:
            return pre_similarity_with_post/pre_similarity_with_pre, ratio_per_session

    def pre_probe_pre_post_post_probe_rate_map_stability_cell_comparison(self, spatial_resolution=5, nr_of_splits=3,
                                                                         save_fig=False, z_score=False, use_max=False):
        """
        Computes rate map stability for decreasing, stable and increasing cells and compares results

        :param spatial_resolution: spatial resolution in cm2
        :type spatial_resolution: int
        :param nr_of_splits: in how many chunks to split exp. phase
        :type nr_of_splits: int
        :param save_fig: save as .svg
        :type save_fig: bool
        :param z_score: z-score correlations
        :type z_score: bool
        """
        sim_matrices_stable = []
        sim_matrices_dec = []
        sim_matrices_inc = []
        for session in self.session_list:
            s_m = session.pre_prob_pre_post_post_prob().rate_map_stability(cells_to_use="stable",
                                                                                    spatial_resolution=spatial_resolution,
                                                                                    plotting=False, nr_of_splits=nr_of_splits,
                                                                           z_score=z_score,use_max=use_max)
            sim_matrices_stable.append(s_m)
            s_i = session.pre_prob_pre_post_post_prob().rate_map_stability(cells_to_use="increasing",
                                                                                    spatial_resolution=spatial_resolution,
                                                                                    plotting=False, nr_of_splits=nr_of_splits,
                                                                           z_score=z_score, use_max=use_max)
            sim_matrices_inc.append(s_i)
            s_d = session.pre_prob_pre_post_post_prob().rate_map_stability(cells_to_use="decreasing",
                                                                                    spatial_resolution=spatial_resolution,
                                                                                    plotting=False, nr_of_splits=nr_of_splits,
                                                                           z_score=z_score, use_max=use_max)
            sim_matrices_dec.append(s_d)

        map_similarity_stable=np.array(sim_matrices_stable).mean(axis=0)
        map_similarity_inc=np.array(sim_matrices_inc).mean(axis=0)
        map_similarity_dec=np.array(sim_matrices_dec).mean(axis=0)

        # compare PRE and POST
        # --------------------------------------------------------------------------------------------------------------
        pre_post_stable = map_similarity_stable[int(map_similarity_stable.shape[0]/2):
                                                (int(map_similarity_stable.shape[0]/2)+nr_of_splits),
                                                (int(map_similarity_stable.shape[0]/2) - nr_of_splits):
                                                int(map_similarity_stable.shape[0]/2)]

        pre_post_inc = map_similarity_inc[int((map_similarity_stable.shape[0]/2)):int((
                map_similarity_stable.shape[0]/2))+nr_of_splits,
                          int((map_similarity_stable.shape[0]/2))-nr_of_splits:int((map_similarity_stable.shape[0]/2))]

        pre_post_dec = map_similarity_dec[int((map_similarity_stable.shape[0]/2)):int((
                map_similarity_stable.shape[0]/2))+nr_of_splits, int((map_similarity_stable.shape[0]/2)) -
                                                                 nr_of_splits:int((map_similarity_stable.shape[0]/2))]

        # get all values (not mean) for stats test & boxplot
        # --------------------------------------------------------------------------------------------------------------
        sim_matrices_stable_arr = np.array(sim_matrices_stable)
        sim_matrices_dec_arr = np.array(sim_matrices_dec)
        sim_matrices_inc_arr = np.array(sim_matrices_inc)
        pre_post_stable_all_val = sim_matrices_stable_arr[:,int((map_similarity_stable.shape[0]/2)):int((
                map_similarity_stable.shape[0]/2))+nr_of_splits,int((map_similarity_stable.shape[0]/2)) -
                                                    nr_of_splits:int((map_similarity_stable.shape[0]/2))].flatten()

        pre_post_inc_all_val = sim_matrices_inc_arr[:,int((map_similarity_stable.shape[0]/2)):int((
                map_similarity_stable.shape[0]/2))+nr_of_splits, int((map_similarity_stable.shape[0]/2)) -
                                                       nr_of_splits:int((map_similarity_stable.shape[0]/2))].flatten()

        pre_post_dec_all_val = sim_matrices_dec_arr[:,int((map_similarity_stable.shape[0]/2)):int((
                map_similarity_stable.shape[0]/2))+nr_of_splits, int((map_similarity_stable.shape[0]/2)) -
                                                        nr_of_splits:int((map_similarity_stable.shape[0]/2))].flatten()

        print("stable vs. dec")
        print(mannwhitneyu(pre_post_dec_all_val, pre_post_stable_all_val))
        print("stable vs. inc")
        print(mannwhitneyu(pre_post_inc_all_val, pre_post_stable_all_val))
        print("dec vs. inc")
        print(mannwhitneyu(pre_post_inc_all_val, pre_post_inc_all_val))

        y_dat = np.vstack((pre_post_dec_all_val, pre_post_inc_all_val, pre_post_stable_all_val))
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        plt.figure(figsize=(3,4))
        bplot = plt.boxplot(y_dat.T, positions=[1, 2, 3], patch_artist=True,
                            labels=["dec", "inc", "stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),
                            )
        colors = ["blue", 'red', "yellow"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        if z_score:
            plt.ylabel("Mean population vector correlation (z-scored)")
        else:
            plt.ylabel("Mean population vector correlation")
        plt.grid(color="grey", axis="y")
        plt.ylim(0,1)
        # plt.yscale("symlog")
        # plt.yticks([np.median(rem_exp), 0.1, -1, np.median(nrem_exp), np.median(awake_exp)])
        # plt.text(-0.001, np.median(rem_exp), np.str(np.round(np.median(rem_exp), 2)))
        # plt.text(-0.001, np.median(nrem_exp), np.str(np.round(np.median(nrem_exp), 2)))
        # plt.text(-0.001, np.median(awake_exp), np.str(np.round(np.median(awake_exp), 2)))
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "rate_map_stability_all_cells_comparison.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # only stable and decreasing cells
        # --------------------------------------------------------------------------------------------------------------
        y_dat = np.vstack((pre_post_dec_all_val, pre_post_stable_all_val))
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        plt.figure(figsize=(3,5))
        bplot = plt.boxplot(y_dat.T, positions=[1, 2], patch_artist=True,
                            labels=["dec", "stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),
                            )
        colors = ["blue", "yellow"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        if z_score:
            plt.ylabel("Mean population vector correlation (z-scored)")
        else:
            plt.ylabel("Mean population vector correlation")
        plt.grid(color="grey", axis="y")
        plt.ylim(0, 1)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "rate_map_stability_dec_stable_comparison.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        probe_stable = map_similarity_stable[int(map_similarity_stable.shape[0]-nr_of_splits):int(
            map_similarity_stable.shape[0]+1), 0:int(nr_of_splits)]

        probe_inc = map_similarity_inc[int(map_similarity_stable.shape[0]-nr_of_splits):int(
            map_similarity_stable.shape[0]+1), 0:int(nr_of_splits)]

        probe_dec = map_similarity_dec[int(map_similarity_stable.shape[0]-nr_of_splits):int(
            map_similarity_stable.shape[0]+1),0:int(nr_of_splits)]

        y_dat = np.vstack((probe_dec.flatten(), probe_inc.flatten(), probe_stable.flatten()))
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        plt.figure(figsize=(3,4))
        bplot = plt.boxplot(y_dat.T, positions=[1, 2, 3], patch_artist=True,
                            labels=["dec", "inc", "stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),
                            )
        colors = ["blue", 'red', "yellow"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Mean population vector correlation")
        plt.grid(color="grey", axis="y")
        plt.title("PRE_PROBE - POST_PROBE")
        # plt.yscale("symlog")
        # plt.yticks([np.median(rem_exp), 0.1, -1, np.median(nrem_exp), np.median(awake_exp)])
        # plt.text(-0.001, np.median(rem_exp), np.str(np.round(np.median(rem_exp), 2)))
        # plt.text(-0.001, np.median(nrem_exp), np.str(np.round(np.median(nrem_exp), 2)))
        # plt.text(-0.001, np.median(awake_exp), np.str(np.round(np.median(awake_exp), 2)))
        # if save_fig:
        #     plt.rcParams['svg.fonttype'] = 'none'
        #     plt.savefig("exponential_coeff_likelihood_vec.svg", transparent="True")
        #     plt.close()
        # else:
        plt.show()

        # check significance of difference between single entries
        # --------------------------------------------------------------------------------------------------------------

        sim_matrices_stable = np.array(sim_matrices_stable)
        sim_matrices_dec = np.array(sim_matrices_dec)
        sim_matrices_inc = np.array(sim_matrices_inc)

        # stable vs. decreasing
        stable_vs_dec = np.zeros(sim_matrices_stable.shape[1:])

        for iy, ix in np.ndindex(sim_matrices_stable.shape[1:]):
            if not iy==ix:
                stable_vs_dec[iy , ix] = mannwhitneyu(sim_matrices_stable[:,iy,ix],sim_matrices_dec[:,iy,ix])[1]
            else:
                stable_vs_dec[iy , ix] = np.nan

        labels = []
        phases = np.array(["pre-probe", "learn-PRE", "POST", "post-probe"])
        for phase in phases:
            for subdiv in range(nr_of_splits):
                labels.append(phase + "_" + str(subdiv))
        # plt.figure(figsize=(6,5))
        plt.imshow(stable_vs_dec)
        plt.yticks(np.arange(stable_vs_dec.shape[0]), labels)
        plt.xticks(np.arange(stable_vs_dec.shape[0]), labels, rotation='vertical')
        plt.xlim(-0.5, stable_vs_dec.shape[0] - 0.5)
        plt.ylim(-0.5, stable_vs_dec.shape[0] - 0.5)
        # plt.ylim(0, map_similarity.shape[0])
        a = plt.colorbar()
        a.set_label("p-value")
        plt.title("Stable vs. decreasing")
        plt.show()

        # stable vs. increasing
        stable_vs_inc = np.zeros(sim_matrices_stable.shape[1:])

        for iy, ix in np.ndindex(sim_matrices_stable.shape[1:]):
            if not iy==ix:
                stable_vs_inc[iy , ix] = mannwhitneyu(sim_matrices_stable[:,iy,ix],sim_matrices_inc[:,iy,ix])[1]
            else:
                stable_vs_inc[iy , ix] = np.nan

        labels = []
        phases = np.array(["pre-probe", "learn-PRE", "POST", "post-probe"])
        for phase in phases:
            for subdiv in range(nr_of_splits):
                labels.append(phase + "_" + str(subdiv))
        # plt.figure(figsize=(6,5))
        plt.imshow(stable_vs_inc)
        plt.yticks(np.arange(stable_vs_inc.shape[0]), labels)
        plt.xticks(np.arange(stable_vs_inc.shape[0]), labels, rotation='vertical')
        plt.xlim(-0.5, stable_vs_dec.shape[0] - 0.5)
        plt.ylim(-0.5, stable_vs_dec.shape[0] - 0.5)
        # plt.ylim(0, map_similarity.shape[0])
        a = plt.colorbar()
        a.set_label("p-value")
        plt.title("Stable vs. increasing")
        plt.show()

    def pre_prob_pre_rate_map(self, nr_of_splits=4, save_fig=True, all_cell_types=False,  use_max=False):

        stable, stable_per_sess = self.rate_map_stability_pre_probe_pre_post_post_probe(normalize=True, cells_to_use="stable",
                                                                       nr_of_splits=nr_of_splits, use_max=use_max)
        dec, dec_per_sess = self.rate_map_stability_pre_probe_pre_post_post_probe(normalize=True, cells_to_use="decreasing",
                                                                    nr_of_splits=nr_of_splits, use_max=use_max)
        if all_cell_types:
            inc, inc_per_sess = self.rate_map_stability_pre_probe_pre_post_post_probe(normalize=True, cells_to_use="increasing",
                                                                        nr_of_splits=nr_of_splits)

        ratio_stable_per_ses = np.vstack(stable_per_sess)
        mean_ratio_stable_per_ses = np.mean(ratio_stable_per_ses, axis=0)
        std_ratio_stable_per_ses = np.std(ratio_stable_per_ses, axis=0)/np.sqrt(7)
        ratio_dec_per_ses = np.vstack(dec_per_sess)
        mean_ratio_dec_per_ses = np.mean(ratio_dec_per_ses, axis=0)
        std_ratio_dec_per_ses = np.std(ratio_dec_per_ses, axis=0)/np.sqrt(7)
        if save_fig:
            plt.style.use('default')
        if all_cell_types:
            ratio_inc_per_ses = np.vstack(inc_per_sess)
            mean_ratio_inc_per_ses = np.mean(ratio_inc_per_ses, axis=0)
            std_ratio_inc_per_ses = np.std(ratio_inc_per_ses, axis=0) / np.sqrt(7)
            plt.errorbar(np.arange(4+0.06), mean_ratio_stable_per_ses, std_ratio_stable_per_ses, label="Increasing",
                         color="orange")
        plt.errorbar(np.arange(4), mean_ratio_stable_per_ses, std_ratio_stable_per_ses, label="Persistent",  color="violet")
        plt.errorbar(np.arange(4)+0.03, mean_ratio_dec_per_ses, std_ratio_dec_per_ses, label="Decreasing", color="green")
        plt.legend()
        plt.xlabel("Subdivision of acquisition")
        plt.ylabel("PV corr with recall / \n PV corr with Pre-probe (mean +- sem)")
        plt.ylim(0.4, 1.6)
        plt.xticks([0,1,2,3], ["1", "2", "3", "4"])
        plt.hlines(1, -0.2, 3.2,linestyles="--", color="gray", zorder=-10000)
        plt.xlim(-0.2,3.2)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            if all_cell_types:
                plt.savefig(os.path.join(save_path, "rate_map_stability_all_cells_per_session.svg"), transparent="True")
            else:
                plt.savefig(os.path.join(save_path, "rate_map_stability_stable_dec_per_session.svg"),
                            transparent="True")
            plt.close()
        else:
            plt.show()

        print(ttest_1samp(ratio_stable_per_ses[:,-1], 1, alternative="greater"))
        print(ttest_1samp(ratio_dec_per_ses[:, -1], 1, alternative="greater"))
        diff_stable = ratio_stable_per_ses[:,-1]-ratio_stable_per_ses[:,0]
        diff_dec = ratio_dec_per_ses[:, -1] - ratio_dec_per_ses[:, 0]
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        bplot = plt.boxplot([diff_stable, diff_dec], positions=[1, 2], patch_artist=True,
                            labels=["stable", "dec"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.show()
        print(ttest_ind(diff_stable, diff_dec, alternative="greater"))

        labels = ["pre_"+str(i) for i in range(nr_of_splits)]
        label_pos = [i for i in range(nr_of_splits)]
        plt.plot(stable, label="stable", color="violet")
        plt.plot(dec, label="dec", color="green")
        if all_cell_types:
            plt.plot(inc, label="increasing", color="orange")
        plt.ylabel("PV correlation with POST /\n PV correlation with PRE_PROBE")
        plt.legend()
        plt.xticks(label_pos, labels)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            if all_cell_types:
                plt.savefig(os.path.join(save_path, "rate_map_stability_all_cells.svg"), transparent="True")
            else:
                plt.savefig(os.path.join(save_path, "rate_map_stability_stable_dec.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def distinguish_contexts(self, cells_to_use ="stable", min_nr_spikes_per_pv=2, nr_splits=2, save_fig=False):
        sim_matrices = []
        for session in self.session_list:
            # s_m = session.cheeseboard_pre_prob_pre_post_post_prob().rate_map_stability(cells_to_use=cells_to_use,
            s_m = session.exp_fam_pre_post_exp_fam().distinguish_contexts_subsets(nr_splits=nr_splits, nr_cells_in_subset=None, nr_subsets=10,
                                     cells_to_use=cells_to_use, min_nr_spikes_per_pv=min_nr_spikes_per_pv, plotting=False)
            sim_matrices.append(s_m)

        for sm in sim_matrices:
            plt.imshow(sm)
            plt.xticks([0, 1, 2, 3], ["exp_fam_1", "pre", "post", "exp_fam_2"])
            plt.yticks([0, 1, 2, 3], ["exp_fam_1", "pre", "post", "exp_fam_2"])
            a = plt.colorbar()
            a.set_label("Mean accuracy")
            plt.xlim(-0.5, 3.5)
            plt.show()

        sim_matrices = np.array(sim_matrices)
        min_val, max_val = 0.2, 0.7
        n = 10
        orig_cmap = plt.cm.Greys
        colors = orig_cmap(np.linspace(min_val, max_val, n))
        cmap_custom = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors)

        mean_all_sessions = np.mean(sim_matrices, axis=0)
        if save_fig:
            plt.style.use('default')
        plt.imshow(mean_all_sessions, cmap=cmap_custom)
        plt.xticks([0, 1, 2, 3], ["Pre-probe", "Acquisition", "Recall", "Post-probe"])
        plt.yticks([0, 1, 2, 3], ["Pre-probe", "Acquisition", "Recall", "Post-probe"])
        plt.xlim(-0.5, 3.5)
        plt.ylim(3.5, -0.5)
        a = plt.colorbar()
        a.set_label("SVM accuracy")
        plt.xticks(rotation=45)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "svm_decoding_episodes.svg"), transparent="True")
        else:
            plt.show()

        pre_probe_pre = sim_matrices[:, 0, 1]
        pre_probe_post = sim_matrices[:, 0, 2]
        pre_probe_post_probe = sim_matrices[:, 0, 3]
        pre_post = sim_matrices[:, 1, 2]
        pre_post_probe = sim_matrices[:, 1, 3]
        post_post_probe = sim_matrices[:, 2, 3]

        res = [pre_post, pre_post_probe, pre_probe_post, pre_probe_post_probe, post_post_probe]
        if save_fig:
            c = "black"
        else:
            c = "white"
        labels = ["Acquisition \nvs\n Recall", "Acquisition \nvs\n Post-probe", "Pre_probe \nvs\n Recall",
                  "Pre_probe \nvs\n Post-probe", "Recall \nvs\n post_probe"]
        bplot = plt.boxplot(res, positions=[1, 2, 3, 4, 5], patch_artist=True,
                            labels=labels,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        # colors = ["magenta", 'magenta', "blue", "blue"]
        # for patch, color in zip(bplot['boxes'], colors):
        #     patch.set_facecolor(color)
        plt.ylabel("Decoding accuracy (SVM)")
        plt.grid(color="grey", axis="y")
        plt.gca().set_xticklabels(labels)
        plt.xticks(rotation=45)
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.ylim(0, 1.5)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "svm_decoding_episodes_boxplot.svg"), transparent="True")
        else:
            plt.show()

        print(mannwhitneyu(pre_post, pre_post_probe))
        print(mannwhitneyu(pre_post, pre_probe_post))
        print(mannwhitneyu(pre_post, pre_probe_post_probe))
        print(mannwhitneyu(pre_post, post_post_probe))

    def excess_path_per_goal(self, nth_trial=2):

        ex_path_start_learning = []
        ex_path_end_learning = []
        ex_path_nth_learning = []
        ex_path_start_post = []
        ex_path_end_post = []
        ex_path_nth_post = []

        for session in self.session_list:
            # s_m = session.cheeseboard_pre_prob_pre_post_post_prob().rate_map_stability(cells_to_use=cells_to_use,
            s_learn, eight_learn, e_learn, start_post, eight_post, end_post = \
                session.pre_prob_pre_post_post_prob().excess_path_per_goal(plotting=False, nth_trial=nth_trial)
            ex_path_start_learning.append(s_learn)
            ex_path_nth_learning.append(eight_learn)
            ex_path_end_learning.append(e_learn)
            ex_path_start_post.append(start_post)
            ex_path_nth_post.append(eight_post)
            ex_path_end_post.append(end_post)

        ex_path_start_learning = np.hstack(ex_path_start_learning)
        ex_path_nth_learning = np.hstack(ex_path_nth_learning)
        ex_path_end_learning = np.hstack(ex_path_end_learning)
        ex_path_start_post = np.hstack(ex_path_start_post)
        ex_path_nth_post = np.hstack(ex_path_nth_post)
        ex_path_end_post = np.hstack(ex_path_end_post)


        plt.figure(figsize=(5,6))
        c = "white"
        res = [ex_path_start_learning, ex_path_nth_learning, ex_path_end_learning, ex_path_start_post, ex_path_nth_post,
               ex_path_end_post]
        bplot = plt.boxplot(res, positions=[1, 2, 3, 4, 5, 6], patch_artist=True,
                            labels=["1st PRE", str(nth_trial)+"th PRE", "last PRE", "1st POST",
                                    str(nth_trial)+"th POST", "last POST"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["gray", 'gray', "gray", "black", "black", "black"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Excess path (%)")
        plt.grid(color="grey", axis="y")
        plt.xticks(rotation=45)
        plt.show()
        print("HERE")

    def excess_path_per_goal_all_trials(self, save_fig=False, n_comparisons=9):

        ex_path_pre = []
        ex_path_post = []

        for session in self.session_list:
            # s_m = session.cheeseboard_pre_prob_pre_post_post_prob().rate_map_stability(cells_to_use=cells_to_use,
            ex_p_pre, ex_p_post = \
                session.pre_prob_pre_post_post_prob().excess_path_per_goal_all_trials(plotting=False)
            ex_path_pre.append(ex_p_pre)
            ex_path_post.append(ex_p_post)

        # need to combine results from PRE
        min_nr_trials = np.min(np.array([x.shape[0] for x in ex_path_pre]))

        first_post = np.hstack([x[0] for x in ex_path_post])

        ex_path_new = [x[:min_nr_trials] for x in ex_path_pre]
        ex_path_new = np.hstack(ex_path_new)

        ex_path_new = ex_path_new[:10, :]

        stats_ = []
        # compute statistics
        for ex_path_trial in ex_path_new:
            stats_.append(mannwhitneyu(ex_path_trial, first_post, alternative="greater")[1])

        res = np.vstack((ex_path_new, first_post))
        labels = [str(i) for i in range(ex_path_new.shape[0])]
        labels.append("POST")
        plt.figure(figsize=(7,5))
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        bplot = plt.boxplot(res.T, positions=np.arange(ex_path_new.shape[0]+1), patch_artist=True,
                            labels=labels,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["gray"] * ex_path_new.shape[0]
        colors.append("black")
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Excess path (multiple of optimal path)")
        plt.grid(color="grey", axis="y")

        _, corrected_p_values = fdrcorrection(np.array(stats_))
        # plt.xticks(rotation=45)
        for i, stats_res in enumerate(corrected_p_values):
            if stats_res < 0.001:
                plt.text(i, 22, "***")
            elif stats_res < 0.01:
                plt.text(i, 22, "**")
            elif stats_res < 0.05:
                plt.text(i, 22, "*")
            else:
                plt.text(i, 22, "n.s.")
        plt.ylim(0,28)
        plt.yticks([1, 5, 10, 15, 20])
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "Excess_path.svg"), transparent="True")
        else:
            plt.show()

    # </editor-fold>

    # <editor-fold desc="Before_sleep, sleep">

    def before_sleep_sleep_compute_likelihoods(self, cells_to_use="all"):
        """
        Computes likelihoods in sleep before and sleep_1

        :param cells_to_use: "all", "stable", "increasing", "decreasing"
        :type cells_to_use: str
        """
        for session in self.session_list:
            session.sleep_before_sleep(data_to_use="ext").compute_likelihoods(cells_to_use=cells_to_use)

    def before_sleep_sleep_compare_likelihoods(self, use_max=True, save_fig=False, cells_to_use="all", split_sleep=True,
                                               z_score=False):
        """
        Compare likelihoods from before_sleep and sleep_1

        :param use_max: use max (otherwise overall likelihoods)
        :type use_max: bool
        :param save_fig: save as .svg
        :type save_fig: bool
        :param cells_to_use: "all", "stable", ...
        :type cells_to_use: str
        :param split_sleep: splits sleep into rem and nrem
        :type split_sleep: bool
        :param z_score: z-score results
        :type z_score: bool
        """
        if split_sleep:
            likelihood_sleep_before = []
            likelihood_sleep_1 = []
            likelihood_sleep_2 = []
        else:
            likelihood_sleep_before = []
            likelihood_sleep = []

        for session in self.session_list:
            if split_sleep:
                likeli_s_b, likeli_s1, likeli_s2 = \
                    session.sleep_before_sleep(data_to_use="std").compare_likelihoods(plotting=False, use_max=use_max,
                                                                                      cells_to_use=cells_to_use,
                                                                                      split_sleep=True)
                likelihood_sleep_before.append(likeli_s_b)
                likelihood_sleep_1.append(likeli_s1)
                likelihood_sleep_2.append(likeli_s2)
            else:
                likeli_s_b, likeli_s = \
                    session.sleep_before_sleep(data_to_use="std").compare_likelihoods(plotting=False,use_max=use_max,
                                                                                      cells_to_use=cells_to_use,
                                                                                      split_sleep=False)
                likelihood_sleep_before.append(likeli_s_b)
                likelihood_sleep.append(likeli_s)

        if split_sleep:

            likelihood_sleep_1 = np.hstack(likelihood_sleep_1)
            likelihood_sleep_2 = np.hstack(likelihood_sleep_2)
            likelihood_sleep_before = np.hstack(likelihood_sleep_before)

            print("Sleep before vs. sleep 1")
            print(mannwhitneyu(likelihood_sleep_before, likelihood_sleep_1))
            print("Sleep before vs. sleep 2")
            print(mannwhitneyu(likelihood_sleep_before, likelihood_sleep_2))

            p_sleep_before = 1. * np.arange(likelihood_sleep_before.shape[0]) / (likelihood_sleep_before.shape[0] - 1)
            p_sleep_1 = 1. * np.arange(likelihood_sleep_1.shape[0]) / (likelihood_sleep_1.shape[0] - 1)
            p_sleep_2 = 1. * np.arange(likelihood_sleep_2.shape[0]) / (likelihood_sleep_2.shape[0] - 1)
            if save_fig:
                plt.style.use('default')
                plt.close()
            plt.plot(np.sort(likelihood_sleep_before), p_sleep_before, color="yellow", label="Sleep before")
            plt.plot(np.sort(likelihood_sleep_1), p_sleep_1, color="lightgreen", label="Sleep_1")
            plt.plot(np.sort(likelihood_sleep_2), p_sleep_2, color="limegreen", label="Sleep_2")
            if not z_score:
                plt.gca().set_xscale("log")
            if use_max:
                if z_score:
                    plt.xlabel("Max. likelihood per PV (z-scored)")
                else:
                    plt.xlabel("Max. likelihood per PV")
            else:
                if z_score:
                    plt.xlabel("Likelihood per PV (z-scored)")
                else:
                    plt.xlabel("Likelihood per PV")
            plt.ylabel("CDF")
            plt.legend()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "decoding_sleep_before_sleep_example_" + cells_to_use + ".svg"),
                            transparent="True")
                plt.close()
            else:
                plt.show()
        else:


            likelihood_sleep = np.hstack(likelihood_sleep)
            likelihood_sleep_before = np.hstack(likelihood_sleep_before)

            print("All sessions:\n")
            print(mannwhitneyu(likelihood_sleep_before, likelihood_sleep))

            p_sleep_before = 1. * np.arange(likelihood_sleep_before.shape[0]) / (likelihood_sleep_before.shape[0] - 1)
            p_sleep = 1. * np.arange(likelihood_sleep.shape[0]) / (likelihood_sleep.shape[0] - 1)
            if save_fig:
                plt.style.use('default')
                plt.close()
            plt.plot(np.sort(likelihood_sleep_before), p_sleep_before, color="greenyellow", label="Sleep before")
            plt.plot(np.sort(likelihood_sleep), p_sleep, color="limegreen", label="Sleep")
            plt.gca().set_xscale("log")
            if use_max:
                if z_score:
                    plt.xlabel("Max. likelihood per PV (z-scored)")
                else:
                    plt.xlabel("Max. likelihood per PV")
            else:
                if z_score:
                    plt.xlabel("Likelihood per PV (z-scored)")
                else:
                    plt.xlabel("Likelihood per PV")
            plt.ylabel("CDF")
            plt.legend()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "decoding_sleep_before_sleep_example_"+cells_to_use+".svg"),
                            transparent="True")
                plt.close()
            else:
                plt.show()

    def before_sleep_sleep_compare_max_post_probabilities(self, save_fig=False, cells_to_use="all"):
        """
        Compares max. posterior probabilites in sleep_before and sleep

        :param save_fig: save as .svg
        :type save_fig: bool
        :param cells_to_use: "all", "stable" ,...
        :type cells_to_use: str
        """
        max_posterior_prob_sleep_before = []
        max_posterior_prob_sleep_1 = []
        max_posterior_prob_sleep_2 = []

        for session in self.session_list:
            maxp_s_b, maxp_s1, maxp_s2 = session.sleep_before_sleep(data_to_use="std").compare_max_post_probabilities(plotting=False,
                                                                                                                      cells_to_use=cells_to_use)
            max_posterior_prob_sleep_before.append(maxp_s_b)
            max_posterior_prob_sleep_1.append(maxp_s1)
            max_posterior_prob_sleep_2.append(maxp_s2)

        max_posterior_prob_sleep_before = np.hstack(max_posterior_prob_sleep_before)
        max_posterior_prob_sleep_1 = np.hstack(max_posterior_prob_sleep_1)
        max_posterior_prob_sleep_2 = np.hstack(max_posterior_prob_sleep_2)

        p_sleep_before = 1. * np.arange(max_posterior_prob_sleep_before.shape[0]) / (
                    max_posterior_prob_sleep_before.shape[0] - 1)
        p_sleep_1 = 1. * np.arange(max_posterior_prob_sleep_1.shape[0]) / (
                    max_posterior_prob_sleep_1.shape[0] - 1)
        p_sleep_2 = 1. * np.arange(max_posterior_prob_sleep_2.shape[0]) / (
                    max_posterior_prob_sleep_2.shape[0] - 1)
        if save_fig:
            plt.style.use('default')
            plt.close()
        plt.plot(np.sort(max_posterior_prob_sleep_before), p_sleep_before, color="greenyellow",
                 label="Sleep before")
        plt.plot(np.sort(max_posterior_prob_sleep_1), p_sleep_1, color="lightgreen", label="Sleep_1")
        plt.plot(np.sort(max_posterior_prob_sleep_2), p_sleep_2, color="limegreen", label="Sleep_2")
        plt.gca().set_xscale("log")

        plt.xlabel("Max. post. probability per PV")
        plt.ylabel("CDF")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "decoding_sleep_before_sleep_example_" + cells_to_use + ".svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

    def before_sleep_sleep_compute_likelihoods_subsets(self, cells_to_use="stable"):
        """
        Compute likelihoods for subsets of cells (models are also fit to subsets only)

        :param cells_to_use: "stable", "decreasing", "increasing"
        :type cells_to_use: str
        """
        for session in self.session_list:
            session.sleep_before_sleep(data_to_use="ext").compute_likelihoods_subsets(cells_to_use=cells_to_use)

    def before_sleep_sleep_compare_likelihoods_subsets(self, use_max=True, save_fig=False, cells_to_use="stable",
                                                       split_sleep=True, z_score=False):
        """
        Compare likelihoods in sleep_before and sleep for different subsets

        :param use_max: use max (otherwise overall likelihoods)
        :type use_max: bool
        :param save_fig: save as .svg
        :type save_fig: bool
        :param cells_to_use: "all", "stable", ...
        :type cells_to_use: str
        :param split_sleep: splits sleep into rem and nrem
        :type split_sleep: bool
        :param z_score: z-score results
        :type z_score: bool
        """
        if split_sleep:
            likelihood_sleep_before = []
            likelihood_sleep_1 = []
            likelihood_sleep_2 = []
        else:
            likelihood_sleep_before = []
            likelihood_sleep = []

        for session in self.session_list:
            if split_sleep:
                likeli_s_b, likeli_s1, likeli_s2 = session.sleep_before_sleep(data_to_use="std").compare_likelihoods_subsets(plotting=False,
                                                                                                         use_max=use_max,
                                                                                                         cells_to_use=cells_to_use,
                                                                                                         split_sleep=True, z_score=z_score)
                likelihood_sleep_before.append(likeli_s_b)
                likelihood_sleep_1.append(likeli_s1)
                likelihood_sleep_2.append(likeli_s2)
            else:
                likeli_s_b, likeli_s = session.sleep_before_sleep(data_to_use="std").compare_likelihoods_subsets(plotting=False,
                                                                                                         use_max=use_max,
                                                                                                         cells_to_use=cells_to_use,
                                                                                                         split_sleep=False, z_score=z_score)
                likelihood_sleep_before.append(likeli_s_b)
                likelihood_sleep.append(likeli_s)


        if split_sleep:

            likelihood_sleep_1 = np.hstack(likelihood_sleep_1)
            likelihood_sleep_2 = np.hstack(likelihood_sleep_2)
            likelihood_sleep_before = np.hstack(likelihood_sleep_before)

            print("\n\n\n For all sessions:\n")
            print("Sleep before vs. sleep 1")
            print(mannwhitneyu(likelihood_sleep_before, likelihood_sleep_1))
            print("Sleep before vs. sleep 2")
            print(mannwhitneyu(likelihood_sleep_before, likelihood_sleep_2))

            p_sleep_before = 1. * np.arange(likelihood_sleep_before.shape[0]) / ( likelihood_sleep_before.shape[0] - 1)
            p_sleep_1 = 1. * np.arange(likelihood_sleep_1.shape[0]) / (likelihood_sleep_1.shape[0] - 1)
            p_sleep_2 = 1. * np.arange(likelihood_sleep_2.shape[0]) / (likelihood_sleep_2.shape[0] - 1)
            if save_fig:
                plt.style.use('default')
                plt.close()
            plt.plot(np.sort(likelihood_sleep_before), p_sleep_before, color="yellow", label="Sleep before")
            plt.plot(np.sort(likelihood_sleep_1), p_sleep_1, color="lightgreen", label="Sleep_1")
            plt.plot(np.sort(likelihood_sleep_2), p_sleep_2, color="limegreen", label="Sleep_2")
            if not z_score:
                plt.gca().set_xscale("log")
            if use_max:
                if z_score:
                    plt.xlabel("Max. likelihood per PV (z-scored)")
                else:
                    plt.xlabel("Max. likelihood per PV")
            else:
                if z_score:
                    plt.xlabel("Likelihood per PV (z-scored)")
                else:
                    plt.xlabel("Likelihood per PV")
            plt.ylabel("CDF")
            plt.legend()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "decoding_sleep_before_sleep_example_" + cells_to_use + ".svg"),
                            transparent="True")
                plt.close()
            else:
                plt.show()
        else:


            likelihood_sleep = np.hstack(likelihood_sleep)
            likelihood_sleep_before = np.hstack(likelihood_sleep_before)

            print("All sessions:\n")
            print(mannwhitneyu(likelihood_sleep_before, likelihood_sleep))

            p_sleep_before = 1. * np.arange(likelihood_sleep_before.shape[0]) / (likelihood_sleep_before.shape[0] - 1)
            p_sleep = 1. * np.arange(likelihood_sleep.shape[0]) / (likelihood_sleep.shape[0] - 1)
            if save_fig:
                plt.style.use('default')
                plt.close()
            plt.plot(np.sort(likelihood_sleep_before), p_sleep_before, color="greenyellow", label="Sleep before")
            plt.plot(np.sort(likelihood_sleep), p_sleep, color="limegreen", label="Sleep")
            plt.gca().set_xscale("log")
            if use_max:
                if z_score:
                    plt.xlabel("Max. likelihood per PV (z-scored)")
                else:
                    plt.xlabel("Max. likelihood per PV")
            else:
                if z_score:
                    plt.xlabel("Likelihood per PV (z-scored)")
                else:
                    plt.xlabel("Likelihood per PV")
            plt.ylabel("CDF")
            plt.legend()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "decoding_sleep_before_sleep_example_"+cells_to_use+".svg"),
                            transparent="True")
                plt.close()
            else:
                plt.show()

    def before_sleep_sleep_diff_likelihoods_subsets(self, save_fig=False, use_norm_diff=True, use_decoded_modes=True,
                                                    use_abs=True):
        """
        Computes difference in likelihoods between sleep_before and sleep for different subsets

        :param save_fig: save as .svg
        :type save_fig: bool
        :param use_norm_diff: normalize difference in likelihoods between sleep_before and sleep
        :type use_norm_diff: bool
        :param use_decoded_modes: use only decoded modes (otherwise: likelihoods from all modes are used)
        :type use_decoded_modes: bool
        :param use_abs: use abs. values of likelihood difference
        :type use_abs: bool
        """
        diff_sleep_stable = []
        diff_sleep_dec = []
        diff_sleep_inc = []


        for session in self.session_list:

            diff_s = session.sleep_before_sleep(data_to_use="std").likelihood_difference_subsets(cells_to_use="stable",
                                                                                                 use_norm_diff=use_norm_diff,
                                                                                                 use_decoded_modes=use_decoded_modes)
            diff_d = session.sleep_before_sleep(data_to_use="std").likelihood_difference_subsets(cells_to_use="decreasing",
                                                                                                 use_norm_diff=use_norm_diff,
                                                                                                 use_decoded_modes=use_decoded_modes)
            diff_i = session.sleep_before_sleep(data_to_use="std").likelihood_difference_subsets(cells_to_use="increasing",
                                                                                                 use_norm_diff=use_norm_diff,
                                                                                                 use_decoded_modes=use_decoded_modes)
            diff_sleep_stable.append(diff_s)
            diff_sleep_dec.append(diff_d)
            diff_sleep_inc.append(diff_i)

        diff_sleep_sleep_stable = np.hstack(diff_sleep_stable)
        diff_sleep_sleep_dec = np.hstack(diff_sleep_dec)
        diff_sleep_sleep_inc = np.hstack(diff_sleep_inc)

        if use_norm_diff:
            cutoff = 0.5
            # compute ratio of modes within -0.5/+0.5 and outside per session
            dec_outside_bound_per_session = []
            stable_outside_bound_per_session = []
            stable_below_neg_0_5 = []
            stable_above_pos_0_5 = []
            dec_below_neg_0_5 = []
            dec_above_pos_0_5 = []

            for diff_stable_per_session, diff_dec_per_session in zip(diff_sleep_stable, diff_sleep_dec):
                # compute ratio of modes within -0.5/+0.5 and outside
                stable_outside_bound = np.round(diff_stable_per_session[np.logical_or((-cutoff > diff_stable_per_session),
                                                           (diff_stable_per_session > cutoff))].shape[0] / diff_stable_per_session.shape[0], 4)

                dec_outside_bound = np.round(diff_dec_per_session[np.logical_or((-cutoff > diff_dec_per_session),
                                                                                (diff_dec_per_session > cutoff))].shape[0] / diff_dec_per_session.shape[0], 4)
                stable_below_neg_0_5.append(np.round(diff_stable_per_session[(-cutoff > diff_stable_per_session)].shape[0] / diff_stable_per_session.shape[0], 4))
                stable_above_pos_0_5.append(np.round(diff_stable_per_session[(cutoff < diff_stable_per_session)].shape[0] / diff_stable_per_session.shape[0], 4))
                dec_below_neg_0_5.append(np.round(diff_dec_per_session[(-cutoff > diff_dec_per_session)].shape[0] / diff_dec_per_session.shape[0], 4))
                dec_above_pos_0_5.append(np.round(diff_dec_per_session[(cutoff < diff_dec_per_session)].shape[0] / diff_dec_per_session.shape[0], 4))

                dec_outside_bound_per_session.append(dec_outside_bound)
                stable_outside_bound_per_session.append(stable_outside_bound)

            if save_fig:
                plt.style.use('default')
                c = "black"
            else:
                c = "white"
            plt.figure(figsize=(3,5))

            labels = ["Stable", "Decreasing"]
            stable_outside_bound_per_session = np.vstack(stable_outside_bound_per_session).flatten()*100
            dec_outside_bound_per_session = np.vstack(dec_outside_bound_per_session).flatten()*100
            res = [stable_outside_bound_per_session, dec_outside_bound_per_session]
            bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                                labels=labels,
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            # colors = ["magenta", 'magenta', "blue", "blue"]
            # for patch, color in zip(bplot['boxes'], colors):
            #     patch.set_facecolor(color)
            plt.ylabel("% Pre-& Replay-modes")
            plt.grid(color="grey", axis="y")
            plt.gca().set_xticklabels(labels)
            plt.ylim(0,100)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "replay_preplay_modes_fraction.svg"), transparent="True")
                plt.close()
            else:
                plt.show()
            print(mannwhitneyu(stable_outside_bound_per_session, dec_outside_bound_per_session, alternative="greater"))

            plt.figure(figsize=(3,5))
            labels = ["Stable", "Decreasing"]
            stable_outside_bound_per_session = np.vstack(stable_above_pos_0_5).flatten()*100
            dec_outside_bound_per_session = np.vstack(dec_above_pos_0_5).flatten()*100
            res = [stable_outside_bound_per_session, dec_outside_bound_per_session]
            bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                                labels=labels,
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            # colors = ["magenta", 'magenta', "blue", "blue"]
            # for patch, color in zip(bplot['boxes'], colors):
            #     patch.set_facecolor(color)
            plt.ylabel("% Preplay-modes")
            plt.grid(color="grey", axis="y")
            plt.gca().set_xticklabels(labels)
            plt.ylim(0,100)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "preplay_modes_fraction.svg"), transparent="True")
                plt.close()
            else:
                plt.show()
            print(mannwhitneyu(stable_outside_bound_per_session, dec_outside_bound_per_session, alternative="greater"))

            plt.figure(figsize=(3,5))
            labels = ["Stable", "Decreasing"]
            stable_outside_bound_per_session = np.vstack(stable_below_neg_0_5).flatten()*100
            dec_outside_bound_per_session = np.vstack(dec_below_neg_0_5).flatten()*100
            res = [stable_outside_bound_per_session, dec_outside_bound_per_session]
            bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                                labels=labels,
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            # colors = ["magenta", 'magenta', "blue", "blue"]
            # for patch, color in zip(bplot['boxes'], colors):
            #     patch.set_facecolor(color)
            plt.ylabel("% Replay-modes")
            plt.grid(color="grey", axis="y")
            plt.gca().set_xticklabels(labels)
            plt.ylim(0,100)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "replay_modes_fraction.svg"), transparent="True")
                plt.close()
            else:
                plt.show()
            print(mannwhitneyu(stable_outside_bound_per_session, dec_outside_bound_per_session, alternative="greater"))

            # compute ratio of modes within -0.5/+0.5 and outside
            stable_within_bound = np.round(diff_sleep_sleep_stable[np.logical_and((-cutoff<diff_sleep_sleep_stable),
                                                                         (diff_sleep_sleep_stable < cutoff))].shape[0]/\
                                  diff_sleep_sleep_stable.shape[0], 4)
            stable_outside_bound = np.round(1-stable_within_bound, 4)

            dec_within_bound = np.round(diff_sleep_sleep_dec[np.logical_and((-cutoff<diff_sleep_sleep_dec),
                                                                         (diff_sleep_sleep_dec < cutoff))].shape[0]/\
                                  diff_sleep_sleep_dec.shape[0], 4)
            dec_outside_bound = np.round(1-dec_within_bound, 4)

            outside = [stable_outside_bound*100, dec_outside_bound*100]
            within = [stable_within_bound*100, dec_within_bound*100]
            labels = ["stable", "dec"]
            width = 0.80

            plt.figure(figsize=(3,5))
            plt.bar(labels, outside, width, label="abs(diff)>0.5", color="dimgrey")
            plt.bar(labels, within, width, bottom=outside, color="lightgray",
                   label="abs(diff)<0.5")
            plt.ylabel("% of modes")
            plt.legend(loc="lower left")
            plt.show()

            if use_abs:
                diff_sleep_sleep_stable = np.abs(diff_sleep_sleep_stable)
                diff_sleep_sleep_dec = np.abs(diff_sleep_sleep_dec)
                diff_sleep_sleep_inc = np.abs(diff_sleep_sleep_inc)

            plt.hist(diff_sleep_sleep_dec, color="turquoise", label="Decreasing", bins=20, density=True)
            plt.hist(diff_sleep_sleep_stable, color="violet", label="Stable", bins=20, alpha=0.7, density=True)
            if use_abs:
                plt.xlabel("abs((sleep_before-sleep)\n/(sleep_before+sleep))")
            else:
                plt.xlabel("(sleep_after-sleep_before)\n/(sleep_before+sleep)")
            plt.legend()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "replay_preplay_modes_hist_abs_"+str(use_abs)+".svg"),
                            transparent="True")
                plt.close()
            else:
                plt.show()

            plt.hist(diff_sleep_sleep_inc, color="orange", label="Increasing", bins=20, density=True)
            plt.hist(diff_sleep_sleep_stable, color="violet", label="Stable", bins=20, alpha=0.7, density=True)
            if use_abs:
                plt.xlabel("abs((sleep_after-sleep_before)/(sleep_before+sleep))")
            else:
                plt.xlabel("(sleep_after-sleep_before)/(sleep_before+sleep)")
            plt.legend()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "replay_preplay_modes_inc_stable_hist_abs_"+str(use_abs)+".svg"),
                            transparent="True")
                plt.close()
            else:
                plt.show()

            print("\n\n\n For all sessions:\n")
            print("Sleep before vs. sleep (dec vs. stable)")
            print(mannwhitneyu(diff_sleep_sleep_stable, diff_sleep_sleep_dec))

            p_s_stable = 1. * np.arange(diff_sleep_sleep_stable.shape[0]) / (diff_sleep_sleep_stable.shape[0] - 1)
            p_s_dec = 1. * np.arange(diff_sleep_sleep_dec.shape[0]) / (diff_sleep_sleep_dec.shape[0] - 1)
            p_s_inc = 1. * np.arange(diff_sleep_sleep_inc.shape[0]) / (diff_sleep_sleep_inc.shape[0] - 1)

            plt.plot(np.sort(diff_sleep_sleep_inc), p_s_inc, color="orange", label="Increasing")
            plt.plot(np.sort(diff_sleep_sleep_stable), p_s_stable, color="violet", label="Stable")
            plt.plot(np.sort(diff_sleep_sleep_dec), p_s_dec, color="turquoise", label="Decreasing")
            if use_abs:
                plt.xlabel("abs((sleep_before-sleep)/(sleep_before+sleep))")
            else:
                plt.xlabel("(sleep_before-sleep)/(sleep_before+sleep)")
            plt.ylabel("CDF")
            plt.legend()
            plt.title("Sleep before - sleep after")
            # plt.xscale("symlog")
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "likeli_sleep_before_sleep_abs"+str(use_abs)+".svg"),
                            transparent="True")
                plt.close()
            else:
                plt.show()

        else:

            diff_sleep_sleep_stable = np.log(diff_sleep_sleep_stable)
            diff_sleep_sleep_dec = np.log(diff_sleep_sleep_dec)

            plt.hist(diff_sleep_sleep_dec, color="turquoise", label="Decreasing", bins=800, density=True)
            plt.hist(diff_sleep_sleep_stable, color="violet", label="Stable", bins=800, alpha=0.8, density=True)
            plt.xscale("symlog")
            plt.xlabel("per mode: mean likeli sleep / mean likeli sleep_before")
            plt.legend()
            plt.show()

            print("\n\n\n For all sessions:\n")
            print("Sleep before vs. sleep")
            print(mannwhitneyu(diff_sleep_sleep_stable, diff_sleep_sleep_dec))

            p_s_stable = 1. * np.arange(diff_sleep_sleep_stable.shape[0]) / (diff_sleep_sleep_stable.shape[0] - 1)
            p_s_dec = 1. * np.arange(diff_sleep_sleep_dec.shape[0]) / (diff_sleep_sleep_dec.shape[0] - 1)

            plt.plot(np.sort(diff_sleep_sleep_stable), p_s_stable, color="violet", label="Stable")
            plt.plot(np.sort(diff_sleep_sleep_dec), p_s_dec, color="turquoise", label="Decreasing")
            # plt.xlabel("Per mode: mean likelihood sleep before/mean likelihood sleep after")
            plt.xlabel("per mode: mean likeli sleep / mean likeli sleep_before")
            plt.ylabel("CDF")
            plt.legend()
            plt.title("Sleep before - sleep after")
            plt.xscale("symlog")
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "mean_likeli_sleep_before.svg"), transparent="True")
                plt.close()
            else:
                plt.show()

    def ratio_modes_active_sleep_before_or_sleep(self, use_norm_diff=True, use_decoded_modes=True):
        """
        Computes ratio of modes that are active in sleep_before and in sleep to assess pre-play

        :param use_norm_diff: normalized difference or use absolute difference
        :type use_norm_diff: bool
        :param use_decoded_modes: only use decoded modes
        :type use_decoded_modes: bool
        """
        ratio_stable = []
        ratio_dec = []

        for session in self.session_list:

            diff_s = session.sleep_before_sleep(data_to_use="std").ratio_modes_active_before_sleep_or_sleep(cells_to_use="stable",
                                                                                                 use_norm_diff=use_norm_diff,
                                                                                                 use_decoded_modes=use_decoded_modes)
            diff_d = session.sleep_before_sleep(data_to_use="std").ratio_modes_active_before_sleep_or_sleep(cells_to_use="decreasing",
                                                                                                 use_norm_diff=use_norm_diff,
                                                                                                 use_decoded_modes=use_decoded_modes)
            ratio_stable.append(diff_s)
            ratio_dec.append(diff_d)

        ratio_stable = np.hstack(ratio_stable)
        ratio_dec = np.hstack(ratio_dec)

        plt.figure(figsize=(3, 5))
        c = "white"
        labels = ["Stable", "Decreasing"]
        res = [ratio_stable, ratio_dec]
        plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=labels,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        # colors = ["magenta", 'magenta', "blue", "blue"]
        # for patch, color in zip(bplot['boxes'], colors):
        #     patch.set_facecolor(color)
        plt.ylabel("% modes active during sleep before or after")
        plt.grid(color="grey", axis="y")
        plt.gca().set_xticklabels(labels)
        plt.show()
        print(mannwhitneyu(ratio_stable, ratio_dec, alternative="greater"))

    def before_sleep_pre_sleep_diff_likelihoods_subsets(self, only_decoded_modes=False, use_norm_diff=True,
                                                        use_abs=True):
        """
        Computes likelihoods for sleep_before, PRE and sleep

        :param only_decoded_modes: to only use decoded modes
        :type only_decoded_modes: bool
        :param use_norm_diff: using normalized difference
        :type use_norm_diff: bool
        :param use_abs: use abs. value of difference between sleep before and sleep
        :type use_abs: bool
        """
        pre_modulation_stable = []
        likeli_ratio_sleep_stable = []
        pre_modulation_dec = []
        likeli_ratio_sleep_dec = []

        for session in self.session_list:
            pre_mod, sleep_mod = session.sleep_before_pre_sleep(data_to_use="std").pre_play_learning_phmm_modes(plotting=False,
                                                                                                    cells_to_use="stable",
                                                                                                    only_decoded_modes=only_decoded_modes,
                                                                                                    use_norm_diff=use_norm_diff)
            pre_modulation_stable.append(pre_mod)
            likeli_ratio_sleep_stable.append(sleep_mod)

            pre_mod, sleep_mod = session.sleep_before_pre_sleep(data_to_use="std").pre_play_learning_phmm_modes(plotting=False,
                                                                                            cells_to_use="decreasing",
                                                                                            only_decoded_modes=only_decoded_modes,
                                                                                            use_norm_diff=use_norm_diff)
            pre_modulation_dec.append(pre_mod)
            likeli_ratio_sleep_dec.append(sleep_mod)

        pre_modulation_stable = np.hstack(pre_modulation_stable)
        likeli_ratio_sleep_stable = np.hstack(likeli_ratio_sleep_stable)
        pre_modulation_dec = np.hstack(pre_modulation_dec)
        likeli_ratio_sleep_dec = np.hstack(likeli_ratio_sleep_dec)

        if use_abs:
            likeli_ratio_sleep_stable = np.abs(likeli_ratio_sleep_stable)
            likeli_ratio_sleep_dec = np.abs(likeli_ratio_sleep_dec)
            pre_modulation_stable = np.abs(pre_modulation_stable)
            pre_modulation_dec = np.abs(pre_modulation_dec)

        # # sort
        # stable_sort = np.argsort(likeli_ratio_sleep_stable)
        # likeli_ratio_sleep_stable_sorted = likeli_ratio_sleep_stable[stable_sort]
        # pre_modulation_stable_sorted = pre_modulation_stable[stable_sort]
        #
        # pre_play_modes_stable_likeli = likeli_ratio_sleep_stable_sorted[int(0.75*likeli_ratio_sleep_stable_sorted.shape[0]):]
        # pre_play_modes_stable_pre_modul = pre_modulation_stable_sorted[int(0.75*pre_modulation_stable_sorted.shape[0]):]
        # pre_play_modes_stable_pre_modul = pre_play_modes_stable_pre_modul[pre_play_modes_stable_likeli<20000]
        # pre_play_modes_stable_likeli = pre_play_modes_stable_likeli[pre_play_modes_stable_likeli<20000]
        #
        #
        # plt.scatter(pre_play_modes_stable_likeli, pre_play_modes_stable_pre_modul)
        # print(pearsonr(pre_play_modes_stable_likeli, pre_play_modes_stable_pre_modul)[0])
        # plt.xscale("symlog")
        # plt.show()
        #
        # re_play_modes_stable_likeli = likeli_ratio_sleep_stable_sorted[:int(0.25*likeli_ratio_sleep_stable_sorted.shape[0])]
        # re_play_modes_stable_pre_modul = pre_modulation_stable_sorted[:int(0.25*pre_modulation_stable_sorted.shape[0])]
        # re_play_modes_stable_pre_modul = pre_play_modes_stable_pre_modul[pre_play_modes_stable_likeli<20000]
        # re_play_modes_stable_likeli = pre_play_modes_stable_likeli[pre_play_modes_stable_likeli<20000]


        # plt.scatter(re_play_modes_stable_likeli, re_play_modes_stable_pre_modul)
        # print(pearsonr(re_play_modes_stable_likeli, re_play_modes_stable_pre_modul)[0])
        # plt.xscale("symlog")
        # plt.show()
        #
        #
        # # delete modes with too large entries
        # pre_modulation_stable = pre_modulation_stable[likeli_ratio_sleep_stable< 10e18]
        # likeli_ratio_sleep_stable = likeli_ratio_sleep_stable[likeli_ratio_sleep_stable< 10e18]

        print("Stable cells:")
        print(pearsonr(pre_modulation_stable, likeli_ratio_sleep_stable)[0])
        plt.scatter(likeli_ratio_sleep_stable, pre_modulation_stable, edgecolor="blue", facecolor="lightblue", alpha=0.7)
        # plt.ylabel("m (regression post. prob. learning)")
        plt.ylabel("m (frequency of mode occurence)")
        plt.xlabel("abs(sleep_before - sleep_after/(sleep_before+sleep_after)")
        # plt.xlabel("likeli_sleep_before/likeli_sleep_after")
        # plt.yscale("symlog")
        # plt.xscale("symlog")
        # plt.xlim(-0.1,1.1)
        # plt.ylim(-0.1,1.1)
        # plt.hlines(0,0,1.1, color="white", zorder=-100)
        # plt.vlines(1,-1,1, color="white", zorder=-100)
        plt.title("Stable cells")
        plt.text(1.1,1, "r="+str(np.round(pearsonr(pre_modulation_stable, likeli_ratio_sleep_stable)[0],2)))
        # plt.gca().set_aspect('equal', 'box')

        plt.show()

        # delete modes with too large entries
        pre_modulation_dec = pre_modulation_dec[likeli_ratio_sleep_dec< 10e18]
        likeli_ratio_sleep_dec = likeli_ratio_sleep_dec[likeli_ratio_sleep_dec< 10e18]

        print("Decreasing cells:")
        print(pearsonr(pre_modulation_dec, likeli_ratio_sleep_dec)[0])
        plt.scatter(likeli_ratio_sleep_dec, pre_modulation_dec, edgecolor="blue", facecolor="lightblue", alpha=0.7)
        # plt.ylabel("m (regression post. prob. learning)")
        plt.ylabel("m (frequency of mode occurence)")
        # plt.xlabel("likeli_sleep_before/likeli_sleep_after")
        plt.xlabel("abs(sleep_before - sleep_after/(sleep_before+sleep_after)")
        # plt.yscale("symlog")
        # plt.xscale("symlog")
        # plt.xlim(-0.1,1.1)
        # plt.ylim(-0.1,1.1)
        # plt.hlines(0,0,1.1, color="white", zorder=-100)
        # plt.vlines(1,-1,1, color="white", zorder=-100)
        plt.title("Decreasing cells")
        plt.text(1.1,1, "r="+str(np.round(pearsonr(pre_modulation_dec, likeli_ratio_sleep_dec)[0],2)))
        plt.show()

    def before_sleep_pre_play_replay_mode_expression(self, only_decoded_modes=True):
        """
        Computes fraction of replay and pre-play modes

        :param only_decoded_modes: using only decoded modes
        :type only_decoded_modes: bool
        """
        within_exp_stable = []
        outside_exp_stable = []
        within_exp_dec = []
        outside_exp_dec = []

        for session in self.session_list:
            ws, os = session.sleep_before_pre_sleep(data_to_use="std").pre_play_replay_mode_expression(cells_to_use="stable",
                                                                                                    only_decoded_modes=only_decoded_modes)

            wd, od = session.sleep_before_pre_sleep(data_to_use="std").pre_play_replay_mode_expression(cells_to_use="decreasing",
                                                                                                    only_decoded_modes=only_decoded_modes)
            within_exp_stable.append(ws)
            outside_exp_stable.append(os)
            within_exp_dec.append(wd)
            outside_exp_dec.append(od)

        outside_exp_stable_sum = np.zeros(len(outside_exp_stable))
        for i, per_sess in enumerate(outside_exp_stable):
            outside_exp_stable_sum[i] = np.sum(per_sess)

        outside_exp_dec_sum = np.zeros(len(outside_exp_dec))
        for i, per_sess in enumerate(outside_exp_dec):
            outside_exp_dec_sum[i] = np.sum(per_sess)

        labels = ["Stable", "Decreasing"]
        c = "white"
        res = [outside_exp_stable_sum, outside_exp_dec_sum]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=labels,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        # colors = ["magenta", 'magenta', "blue", "blue"]
        # for patch, color in zip(bplot['boxes'], colors):
        #     patch.set_facecolor(color)
        plt.ylabel("%active time during pre (sum of pre-/replay modes)")
        plt.grid(color="grey", axis="y")
        plt.gca().set_xticklabels(labels, rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.show()


        within_exp_stable = np.hstack(within_exp_stable)
        outside_exp_stable = np.hstack(outside_exp_stable)
        within_exp_dec = np.hstack(within_exp_dec)
        outside_exp_dec = np.hstack(outside_exp_dec)

        all_exp_stable = np.hstack((within_exp_stable, outside_exp_stable))
        all_exp_dec = np.hstack((within_exp_dec, outside_exp_dec))


        p_stable = 1. * np.arange(all_exp_stable.shape[0]) / (all_exp_stable.shape[0] - 1)
        p_dec = 1. * np.arange(all_exp_dec.shape[0]) / (all_exp_dec.shape[0] - 1)

        plt.plot(np.sort(all_exp_stable), p_stable, label="stable")
        plt.plot(np.sort(all_exp_dec), p_dec, label="dec")
        plt.title("All modes that are either expressed in sleep before/after")
        plt.legend()
        plt.xlabel("%time expressed during PRE")
        plt.show()

        print(mannwhitneyu(all_exp_stable, all_exp_dec))




        p_within_stable = 1. * np.arange(within_exp_stable.shape[0]) / (within_exp_stable.shape[0] - 1)
        p_within_dec = 1. * np.arange(within_exp_dec.shape[0]) / (within_exp_dec.shape[0] - 1)

        plt.plot(np.sort(within_exp_stable), p_within_stable, label="stable")
        plt.plot(np.sort(within_exp_dec), p_within_dec, label="dec")
        plt.title("Equally expressed in sleep before/after")
        plt.legend()
        plt.xlabel("%time expressed during PRE")
        plt.show()

        print(mannwhitneyu(within_exp_stable, within_exp_dec))


        p_outside_stable = 1. * np.arange(outside_exp_stable.shape[0]) / (outside_exp_stable.shape[0] - 1)
        p_outside_dec = 1. * np.arange(outside_exp_dec.shape[0]) / (outside_exp_dec.shape[0] - 1)

        plt.plot(np.sort(outside_exp_stable), p_outside_stable, label="stable")
        plt.plot(np.sort(outside_exp_dec), p_outside_dec, label="dec")
        plt.title("More expressed in either sleep before or sleep after")
        plt.legend()
        plt.xlabel("%time expressed during PRE")
        plt.show()

        print(mannwhitneyu(outside_exp_stable, outside_exp_dec))

    def learning_phmm_modes_frequency_and_likelihoods(self, only_decoded_modes=True, cells_to_use="stable",
                                                      subset_of_sleep=None, z_score_pre_modulation=False):
        percent_active = []
        likeli_ratio = []
        for session in self.session_list:
            perc_act, likeli_rat = session.sleep_before_pre_sleep(data_to_use="std").learning_phmm_modes_frequency_and_likelihoods(
                cells_to_use=cells_to_use, subset_of_sleep=subset_of_sleep,
                only_decoded_modes=only_decoded_modes, plotting=False, z_score_pre_modulation=z_score_pre_modulation)
            percent_active.append(perc_act)
            likeli_ratio.append(likeli_rat)

        percent_active = np.hstack(percent_active)
        likeli_ratio = np.hstack(likeli_ratio)

        plt.scatter(likeli_ratio, percent_active)
        if z_score_pre_modulation:
            plt.ylabel("%times active in PRE (z-scored)")
        else:
            plt.ylabel("%times active in PRE")
        # plt.xlabel("likeli_sleep_before/likeli_sleep_after")
        plt.xlabel("(likelihood_sleep_before - likelihood_sleep) / (likelihood_sleep_before + likelihood_sleep)")
        plt.title(cells_to_use)
        y_min, y_max = plt.gca().get_ylim()
        plt.text(0, y_min+(y_max-y_min)/2, "R="+str(np.round(pearsonr(likeli_ratio, percent_active)[0],2)), color="red")
        plt.text(0, y_min+(y_max-y_min)/2-(y_max-y_min)*0.05, "p="+str(pearsonr(likeli_ratio, percent_active)[1]), color="red")
        plt.show()

    def learning_phmm_modes_frequency_and_likelihoods_plot_single(self, only_decoded_modes=True, cells_to_use="stable"):

        for session in self.session_list:
            session.sleep_before_pre_sleep(data_to_use="std").learning_phmm_modes_frequency_and_likelihoods(
                cells_to_use=cells_to_use,
                only_decoded_modes=only_decoded_modes, plotting=True)

    def learning_phmm_modes_temporal_trend(self, cells_to_use="stable"):
        for session in self.session_list:
            session.sleep_before_pre_sleep(data_to_use="std").pre_play_learning_phmm_modes(cells_to_use=cells_to_use,
                                                                                           plotting=True)

    def learning_phmm_modes_decoded_before_after(self, save_fig=False):

        before_dec = []
        after_dec = []
        common_dec = []
        never_dec = []
        before_stable = []
        after_stable = []
        common_stable = []
        never_stable = []

        for session in self.session_list:
            b,a,c,n = session.sleep_before_sleep(data_to_use="std").decoded_modes_before_after(cells_to_use="decreasing")
            before_dec.append(b)
            after_dec.append(a)
            common_dec.append(c)
            never_dec.append(n)

        for session in self.session_list:
            b,a,c,n = session.sleep_before_sleep(data_to_use="std").decoded_modes_before_after(cells_to_use="stable")
            before_stable.append(b)
            after_stable.append(a)
            common_stable.append(c)
            never_stable.append(n)


        c = "white"
        labels = ["Only before stable", "Only before dec", "Only after stable", "Only after dec", "Common stable", "Common dec"]
        res = [before_stable, before_dec, after_stable, after_dec, common_stable, common_dec]
        bplot = plt.boxplot(res, positions=[1,2,3,4,5,6], patch_artist=True,
                            labels=labels,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),showfliers=False
                            )
        # colors = ["magenta", 'magenta', "blue", "blue"]
        # for patch, color in zip(bplot['boxes'], colors):
        #     patch.set_facecolor(color)
        plt.ylabel("%modes")
        plt.grid(color="grey", axis="y")
        plt.gca().set_xticklabels(labels, rotation = 45, ha="right")
        plt.title("Decoded modes")
        plt.show()

        plt.figure(figsize=(3,5))
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        labels = ["Persistent",
                  "Decreasing"]
        res = [np.array(never_stable)*100, np.array(never_dec)*100]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=labels,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        # colors = ["magenta", 'magenta', "blue", "blue"]
        # for patch, color in zip(bplot['boxes'], colors):
        #     patch.set_facecolor(color)
        plt.ylabel("% of states never decoded during sleep")
        plt.grid(color="grey", axis="y")
        plt.ylim(0,80)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "Never_decoded_modes.svg"), transparent="True")
        else:
            plt.show()
        print(mannwhitneyu(never_stable, never_dec))

    def learning_phmm_modes_decoded_before_after_freq_during_learning(self):

        before_dec = []
        after_dec = []
        common_dec = []
        never_dec = []
        before_stable = []
        after_stable = []
        common_stable = []
        never_stable = []

        for session in self.session_list:
            b,a,c,n = session.sleep_before_pre_sleep(data_to_use="std").decoded_modes_before_after_freq_during_learning(cells_to_use="decreasing")
            before_dec.append(b)
            after_dec.append(a)
            common_dec.append(c)
            never_dec.append(n)

        for session in self.session_list:
            b,a,c,n = session.sleep_before_pre_sleep(data_to_use="std").decoded_modes_before_after_freq_during_learning(cells_to_use="stable")
            before_stable.append(b)
            after_stable.append(a)
            common_stable.append(c)
            never_stable.append(n)


        c = "white"
        labels = ["Only before stable", "Only before dec", "Only after stable", "Only after dec", "Common stable", "Common dec"]
        res = [before_stable, before_dec, after_stable, after_dec, common_stable, common_dec]
        bplot = plt.boxplot(res, positions=[1,2,3,4,5,6], patch_artist=True,
                            labels=labels,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),showfliers=False
                            )
        # colors = ["magenta", 'magenta', "blue", "blue"]
        # for patch, color in zip(bplot['boxes'], colors):
        #     patch.set_facecolor(color)
        plt.ylabel("%time active during learning")
        plt.grid(color="grey", axis="y")
        plt.gca().set_xticklabels(labels, rotation = 45, ha="right")
        plt.title("Decoded modes")
        plt.show()

        c = "white"
        labels = ["Never decoded stable",
                  "Never decoded dec"]
        res = [never_stable, never_dec]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=labels,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        # colors = ["magenta", 'magenta', "blue", "blue"]
        # for patch, color in zip(bplot['boxes'], colors):
        #     patch.set_facecolor(color)
        plt.ylabel("%time active during learning")
        plt.grid(color="grey", axis="y")
        plt.gca().set_xticklabels(labels, rotation=45, ha="right")
        plt.title("Never decoded modes")
        plt.show()

    def before_sleep_pre_sleep_correlations(self):
        """
        Compares correlation structure between sleep_before, PRE and sleep

        """
        corr_matrix_stable = []
        corr_matrix_dec = []
        corr_matrix_inc = []

        for session in self.session_list:
            c_s, c_d, c_i = session.sleep_before_pre_sleep(data_to_use="std").correlation_structure_similarity(plotting=False)
            corr_matrix_stable.append(c_s)
            corr_matrix_dec.append(c_d)
            corr_matrix_inc.append(c_i)

        corr_matrix_stable = np.array(corr_matrix_stable)
        corr_matrix_dec = np.array(corr_matrix_dec)
        corr_matrix_inc = np.array(corr_matrix_inc)

        corr_matrix_stable_mean = np.mean(corr_matrix_stable, axis=0)
        corr_matrix_dec_mean = np.mean(corr_matrix_dec, axis=0)
        corr_matrix_inc_mean = np.mean(corr_matrix_inc, axis=0)

        np.fill_diagonal(corr_matrix_inc_mean, np.nan)
        plt.imshow(corr_matrix_inc_mean, cmap="Greys")
        a = plt.colorbar()
        a.set_label("Mean Pearson R")
        labels = np.array(["sleep_before", "PRE_1", "PRE_2", "PRE_3", "Sleep"])
        plt.yticks(np.arange(corr_matrix_inc_mean.shape[0]), labels)
        plt.xticks(np.arange(corr_matrix_inc_mean.shape[0]), labels, rotation='vertical')
        plt.xlim(-0.5, corr_matrix_inc_mean.shape[0] - 0.5)
        plt.ylim(-0.5, corr_matrix_inc_mean.shape[0] - 0.5)
        plt.title("Increasing")
        plt.show()

        np.fill_diagonal(corr_matrix_dec_mean, np.nan)
        plt.imshow(corr_matrix_dec_mean, cmap="Greys")
        a = plt.colorbar()
        a.set_label("Pearson R")
        labels = np.array(["sleep_before", "PRE_1", "PRE_2", "PRE_3", "Sleep"])
        plt.yticks(np.arange(corr_matrix_dec_mean.shape[0]), labels)
        plt.xticks(np.arange(corr_matrix_dec_mean.shape[0]), labels, rotation='vertical')
        plt.xlim(-0.5, corr_matrix_dec_mean.shape[0] - 0.5)
        plt.ylim(-0.5, corr_matrix_dec_mean.shape[0] - 0.5)
        plt.title("Decreasing")
        plt.show()

        np.fill_diagonal(corr_matrix_stable_mean, np.nan)
        plt.imshow(corr_matrix_stable_mean, cmap="Greys")
        a = plt.colorbar()
        a.set_label("Pearson R")
        labels = np.array(["sleep_before", "PRE_1", "PRE_2", "PRE_3", "Sleep"])
        plt.yticks(np.arange(corr_matrix_stable_mean.shape[0]), labels)
        plt.xticks(np.arange(corr_matrix_stable_mean.shape[0]), labels, rotation='vertical')
        plt.xlim(-0.5, corr_matrix_stable_mean.shape[0] - 0.5)
        plt.ylim(-0.5, corr_matrix_stable_mean.shape[0] - 0.5)
        plt.title("Stable")
        plt.show()

        pre_play_stable = np.vstack((corr_matrix_stable[:,0,1], corr_matrix_stable[:,0,2], corr_matrix_stable[:,0,3])).flatten()
        re_play_stable = np.vstack((corr_matrix_stable[:,4,1], corr_matrix_stable[:,4,2], corr_matrix_stable[:,4,3])).flatten()
        pre_play_dec = np.vstack((corr_matrix_dec[:,0,1], corr_matrix_dec[:,0,2], corr_matrix_dec[:,0,3])).flatten()
        re_play_dec = np.vstack((corr_matrix_dec[:,4,1], corr_matrix_dec[:,4,2], corr_matrix_dec[:,4,3])).flatten()
        pre_play_inc = np.vstack((corr_matrix_inc[:,0,1], corr_matrix_inc[:,0,2], corr_matrix_inc[:,0,3])).flatten()
        re_play_inc = np.vstack((corr_matrix_inc[:,4,1], corr_matrix_inc[:,4,2], corr_matrix_inc[:,4,3])).flatten()

        # compute max correlation (from corr(sleep,pre1),corr(sleep,pre2),corr(sleep,pre3))
        pre_play_stable_max = np.max(corr_matrix_stable[:, 0, 1:4], axis=1)
        re_play_stable_max = np.max(corr_matrix_stable[:, 4, 1:4], axis=1)
        pre_play_dec_max = np.max(corr_matrix_dec[:, 0, 1:4], axis=1)
        re_play_dec_max = np.max(corr_matrix_dec[:, 4, 1:4], axis=1)
        pre_play_inc_max = np.max(corr_matrix_inc[:, 0, 1:4], axis=1)
        re_play_inc_max = np.max(corr_matrix_inc[:, 4, 1:4], axis=1)

        print(ttest_ind(pre_play_stable, re_play_stable))
        print(ttest_ind(pre_play_dec, re_play_dec))
        print(ttest_ind(pre_play_inc, re_play_inc))
        c = "white"
        labels = ["Pre-play stable", "Re-play stable", "Pre-play dec", "Re-play dec", "Pre-play inc", "Re-play inc"]
        res = [pre_play_stable, re_play_stable, pre_play_dec, re_play_dec, pre_play_inc, re_play_inc]
        bplot = plt.boxplot(res, positions=[1,2,3,4,5,6], patch_artist=True,
                            labels=labels,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),showfliers=False
                            )
        # colors = ["magenta", 'magenta', "blue", "blue"]
        # for patch, color in zip(bplot['boxes'], colors):
        #     patch.set_facecolor(color)
        plt.ylabel("Pearson")
        plt.grid(color="grey", axis="y")
        plt.gca().set_xticklabels(labels, rotation = 45, ha="right")
        plt.show()

        print(ttest_ind(pre_play_stable_max, re_play_stable_max))
        print(ttest_ind(pre_play_dec_max, re_play_dec_max))
        print(ttest_ind(pre_play_inc_max, re_play_inc_max))
        c = "white"
        labels = ["Pre-play stable", "Re-play stable", "Pre-play dec", "Re-play dec", "Pre-play inc", "Re-play inc"]
        res = [pre_play_stable_max, re_play_stable_max, pre_play_dec_max, re_play_dec_max, pre_play_inc_max, re_play_inc_max]
        bplot = plt.boxplot(res, positions=[1,2,3,4,5,6], patch_artist=True,
                            labels=labels,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),showfliers=False
                            )
        # colors = ["magenta", 'magenta', "blue", "blue"]
        # for patch, color in zip(bplot['boxes'], colors):
        #     patch.set_facecolor(color)
        plt.ylabel("Pearson")
        plt.grid(color="grey", axis="y")
        plt.gca().set_xticklabels(labels, rotation = 45, ha="right")
        plt.show()

    # </editor-fold>

    # <editor-fold desc="Pre_probe, PRE">

    def pre_prob_pre_likelihoods(self, cells_to_use="stable", save_fig=False):
        """
        uses PRE phmm to decode activity in PRE and pre_probe to compare likelihoods

        :param cells_to_use: "all", "stable", "decreasing", "increasing"
        :type cells_to_use: str
        :param save_fig: save as .svg
        :type save_fig: bool
        """
        pre_probe_median_log_likeli = []
        pre_median_log_likeli = []

        for session in self.session_list:
            pre_probe, pre = \
                session.pre_prob_pre().phmm_modes_likelihoods(plotting=False, cells_to_use=cells_to_use)
            pre_probe_median_log_likeli.append(pre_probe)
            pre_median_log_likeli.append(pre)

        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(4, 6))
        # plt.figure(figsize=(3,4))
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        for session_id, (first, second) in enumerate(zip(pre_probe_median_log_likeli, pre_median_log_likeli)):
            plt.scatter([0.1, 0.2], [first, second], label=str(session_id), color=col[session_id], zorder=session_id)
            plt.plot([0.1, 0.2], [first, second], color=col[session_id], zorder=session_id)
            plt.xticks([0.1, 0.2], ["Pre_probe", "PRE"])
        plt.ylabel("Median Log-likelihood of PRE states")
        plt.grid(axis="y", color="gray")
        plt.title(cells_to_use)
        plt.ylim(-17, -7)
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "pre_probe_pre_likeli_" + cells_to_use + ".svg"), transparent="True")
        else:
            plt.show()

    # def pre_prob_pre_rate_map(self, nr_of_splits=4, save_fig=True, all_cell_types=True):
    #     """
    #     Compares rate map similarity in pre_probe and pre
    #
    #     :param nr_of_splits: in how many chunks to split experimental phase
    #     :type nr_of_splits: int
    #     :param save_fig: save as .svg
    #     :type save_fig: bool
    #     :param all_cell_types: whether to use all cell types (True: stable, increasing, decreasing) or only
    #                            decrasing and stable cells (False)
    #     :type all_cell_types:
    #     """
    #     stable = self.pre_probe_pre_post_post_probe_rate_map_stability(normalize=True, cells_to_use="stable",
    #                                                                    nr_of_splits=nr_of_splits)
    #     dec = self.pre_probe_pre_post_post_probe_rate_map_stability(normalize=True, cells_to_use="decreasing",
    #                                                                 nr_of_splits=nr_of_splits)
    #     if all_cell_types:
    #         inc = self.pre_probe_pre_post_post_probe_rate_map_stability(normalize=True, cells_to_use="increasing",
    #                                                                     nr_of_splits=nr_of_splits)
    #
    #     labels = ["pre_" + str(i) for i in range(nr_of_splits)]
    #     label_pos = [i for i in range(nr_of_splits)]
    #     if save_fig:
    #         plt.style.use('default')
    #     plt.plot(stable, label="stable", color="violet")
    #     plt.plot(dec, label="dec", color="green")
    #     if all_cell_types:
    #         plt.plot(inc, label="increasing", color="orange")
    #     plt.ylabel("PV correlation with POST /\n PV correlation with PRE_PROBE")
    #     plt.legend()
    #     plt.xticks(label_pos, labels)
    #     if save_fig:
    #         plt.rcParams['svg.fonttype'] = 'none'
    #         if all_cell_types:
    #             plt.savefig("rate_map_stability_all_cells.svg", transparent="True")
    #         else:
    #             plt.savefig("rate_map_stability_stable_dec.svg", transparent="True")
    #         plt.close()
    #     else:
    #         plt.show()

    # </editor-fold>

    # <editor-fold desc="Others">

    def plot_classified_cell_distribution(self, save_fig=False):
        """
        plots boxplot of nr. stable/decreasing/increasing cells

        @param save_fig: whether to save figure as svg
        @type save_fig: bool
        """
        stable = []
        decreasing = []
        increasing = []

        for session in self.session_list:
            cell_ids_stable, cell_ids_decreasing, cell_ids_increasing = session.cheese_board(
                experiment_phase=["learning_cheeseboard_1"]).get_cell_classification_labels()
            stable.append(cell_ids_stable.shape[0])
            decreasing.append(cell_ids_decreasing.shape[0])
            increasing.append(cell_ids_increasing.shape[0])

        print("stable: " + str(stable) + "\n")
        print("decreasing: " + str(decreasing) + "\n")
        print("increasing: " + str(increasing) + "\n")

        if save_fig:
            plt.style.use('default')

        plt.bar([0, 1, 2], [np.sum(np.array(stable)), np.sum(np.array(decreasing)), np.sum(np.array(increasing))],
                width=0.5, color=["violet", "green", "yellow"])
        plt.xticks([0, 1, 2], ["stable", "decreasing", "increasing"])
        plt.yticks([100, 200, 300, 400])
        plt.ylabel("Nr. cells")
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(os.path.join(save_path, "cell_classification_numbers.svg"), transparent="True")
        plt.show()

    def compare_spatial_measures_and_firing_rates(self, spatial_resolution=2):
        """
        compares spatial measures and firing rates for stable, decreasing and increasing cells

        :param spatial_resolution: in cm2
        :type spatial_resolution: int
        """
        firing_pre_stable, firing_pre_dec, firing_pre_inc, firing_sleep_stable, firing_sleep_dec, \
        firing_sleep_inc, firing_post_stable, firing_post_dec, firing_post_inc = \
            self.pre_long_sleep_post_firing_rates_all_cells(plotting=False, measure="mean")

        pre_stable_sk_s, post_stable_sk_s, pre_dec_sk_s, post_inc_sk_s, post_dec_sk_s, pre_inc_sk_s = \
            self.pre_post_cheeseboard_spatial_information(plotting=False, info_measure="skaggs_second",
                                                          remove_nan=False, spatial_resolution=spatial_resolution)
        pre_stable_spar, post_stable_spar, pre_dec_spar, post_inc_spar, post_dec_spar, pre_inc_spar = \
            self.pre_post_cheeseboard_spatial_information(plotting=False, info_measure="sparsity",
                                                          remove_nan=False, spatial_resolution=spatial_resolution)

        plt.scatter(pre_stable_sk_s, pre_stable_spar)
        plt.xlabel("Skaggs per second")
        plt.ylabel("Sparsity")
        plt.title("Stable (PRE)")
        plt.show()

        plt.scatter(firing_pre_stable, pre_stable_spar)
        plt.xlabel("Mean firing rate")
        plt.ylabel("Sparsity")
        plt.title("Stable (PRE)")
        plt.show()

        plt.scatter(firing_pre_stable, pre_stable_sk_s)
        plt.xlabel("Mean firing rate")
        plt.ylabel("Skaggs per second")
        plt.title("Stable (PRE)")
        plt.show()

        plt.scatter(pre_dec_sk_s, pre_dec_spar)
        plt.xlabel("Skaggs per second")
        plt.ylabel("Sparsity")
        plt.title("Decreasing (PRE)")
        plt.show()

        plt.scatter(firing_pre_dec, pre_dec_spar)
        plt.xlabel("Mean firing rate")
        plt.ylabel("Sparsity")
        plt.title("Decreasing (PRE)")
        plt.show()

        plt.scatter(firing_pre_dec, pre_dec_sk_s)
        plt.xlabel("Mean firing rate")
        plt.ylabel("Skaggs per second")
        plt.title("Decreasing (PRE)")
        plt.show()

    def assess_feature_stability_subsets(self, save_fig=False):
        """
        Compares stability of recording for different subsets of cells

        :param save_fig: save as .svg
        :type save_fig: bool
        """
        stable_cell_z = []
        dec_cell_z = []
        inc_cell_z = []

        for session in self.session_list:
            s_z, d_z, i_z = session.all_data().assess_feature_stability_subsets(plotting=False)
            stable_cell_z.append(s_z)
            dec_cell_z.append(d_z)
            inc_cell_z.append(s_z)

        all_stable_z = np.hstack(stable_cell_z)
        all_dec_z = np.hstack(dec_cell_z)
        all_inc_z = np.hstack(inc_cell_z)
        all_unstable_z = np.hstack((all_dec_z, all_inc_z))

        print("Stable vs. dec:")
        print(mannwhitneyu(all_stable_z, all_dec_z))

        print("Stable vs. inc:")
        print(mannwhitneyu(all_stable_z, all_inc_z))

        print("Stable vs. unstable:")
        print(mannwhitneyu(all_stable_z, all_unstable_z))

        p_stable = 1. * np.arange(all_stable_z.shape[0]) / (all_stable_z.shape[0] - 1)
        p_inc = 1. * np.arange(all_inc_z.shape[0]) / (all_inc_z.shape[0] - 1)
        p_dec = 1. * np.arange(all_dec_z.shape[0]) / (all_dec_z.shape[0] - 1)
        if save_fig:
            plt.style.use('default')
        plt.plot(np.sort(all_stable_z), p_stable, color="violet", label="stable")
        plt.plot(np.sort(all_dec_z), p_dec, color="turquoise", label="dec")
        plt.plot(np.sort(all_inc_z), p_inc, color="orange", label="inc")
        plt.ylabel("cdf")
        plt.xlabel("(mean_last_10%-mean_first_10%)/std_first_10% \n all features")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "clustering_stability.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        plt.plot(np.sort(np.abs(all_stable_z)), p_stable, color="violet", label="stable")
        plt.plot(np.sort(np.abs(all_dec_z)), p_dec, color="turquoise", label="dec")
        plt.plot(np.sort(np.abs(all_inc_z)), p_inc, color="orange", label="inc")
        plt.ylabel("cdf")
        plt.xlabel("abs((mean_last_10%-mean_first_10%)/std_first_10%) \n all features")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "clustering_stability_abs.svg"), transparent="True")
        else:
            plt.show()

    def assess_feature_stability(self, save_fig=False, hours_to_compute=None):
        """
        Computes stability based on clustering features

        :param save_fig: save as .svg
        :type save_fig: bool
        :param hours_to_compute: for which hours to compute stability
        :type hours_to_compute: list
        """
        if hours_to_compute is None:
            hours_to_compute = [5, 10, 15]
        # labels for plotting
        labels = [str(i) for i in hours_to_compute]

        res = []
        per_cell_within = []
        per_cell_across = []

        for session in self.session_list:
            r, pc_w, pc_a = session.all_data().assess_feature_stability(plotting=False, hours_to_compute=hours_to_compute)
            res.append(r)
            per_cell_within.append(pc_w)
            per_cell_across.append(pc_a)

        per_cell_within = np.hstack(per_cell_within)
        per_cell_across = np.hstack(per_cell_across)
        # combine results from all sessions
        all_sess_res = []
        for hour in range(len(res[0])):
            hour_res = []
            for sess_res in res:
                hour_res.extend(sess_res[hour])
            a = np.array(hour_res)
            all_sess_res.append(np.ma.masked_invalid(a))

        cmap = matplotlib.cm.get_cmap('viridis')
        colors_to_plot = cmap(np.linspace(0, 1, 4))
        plt.figure(figsize=(3, 4))
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        bplot = plt.boxplot(all_sess_res, positions=[1, 2, 3], patch_artist=True,
                            labels=labels,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.ylim(-1.5, 2.0)
        for patch, color in zip(bplot['boxes'], colors_to_plot[1:]):
            patch.set_facecolor(color)
        plt.ylabel("z-scored mean of each feature \n using mean and std from first hour")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "feature_drift.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

        # check if distribution at the end of experiment is different from m=0
        print(ttest_1samp(all_sess_res[-1], 0))
        print(ttest_ind(all_sess_res[0], all_sess_res[1]))
        print(ttest_ind(all_sess_res[1], all_sess_res[2]))
        print(ttest_ind(all_sess_res[0], all_sess_res[2]))

        print(mannwhitneyu(np.ma.masked_invalid(per_cell_within), np.ma.masked_invalid(per_cell_across)))
        if save_fig:
            plt.style.use('default')
        p_within = 1. * np.arange(per_cell_within.shape[0]) / (per_cell_within.shape[0] - 1)
        p_across = 1. * np.arange(per_cell_across.shape[0]) / (per_cell_across.shape[0] - 1)
        plt.plot(np.sort(per_cell_within), p_within, label="within")
        plt.plot(np.sort(per_cell_across), p_across, label="across")
        plt.ylabel("cdf")
        plt.xlabel("Abs. z-scored means of features last hour \n using mean/std of first hour")
        plt.legend()

        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "feature_over_time.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def cluster_quality_over_time(self, save_fig=False, hours_to_compute=None):
        """
        Computes stability based on clustering features

        :param save_fig: save as .svg
        :type save_fig: bool
        :param hours_to_compute: for which hours to compute stability
        :type hours_to_compute: list
        """
        p_own_spikes_own_mean_vs_own_spikes_other_means = []
        p_own_spikes_own_mean_vs_other_spikes_own_mean = []
        p_own_mean_own_mean_vs_other_means_own_mean = []
        p_own_mean_own_mean_vs_own_mean_other_means = []
        p_other_cell_spikes_own_mean_first_hour_vs_own_spikes_hour_own_mean_last_hour = []

        for session in self.session_list:
            p_own_spikes_own_mean_vs_own_spikes_other_means_, p_own_spikes_own_mean_vs_other_spikes_own_mean_, \
                p_own_mean_own_mean_vs_other_means_own_mean_, p_own_mean_own_mean_vs_own_mean_other_means_, \
                p_other_cell_spikes_own_mean_first_hour_vs_own_spikes_hour_own_mean_last_hour_ = \
                session.all_data().cluster_quality_over_time(plotting=False, hours_to_compute=hours_to_compute)

            p_own_spikes_own_mean_vs_own_spikes_other_means.append(p_own_spikes_own_mean_vs_own_spikes_other_means_)
            p_own_spikes_own_mean_vs_other_spikes_own_mean.append(p_own_spikes_own_mean_vs_other_spikes_own_mean_)
            p_own_mean_own_mean_vs_other_means_own_mean.append(p_own_mean_own_mean_vs_other_means_own_mean_)
            p_own_mean_own_mean_vs_own_mean_other_means.append(p_own_mean_own_mean_vs_own_mean_other_means_)
            p_other_cell_spikes_own_mean_first_hour_vs_own_spikes_hour_own_mean_last_hour.append(
                p_other_cell_spikes_own_mean_first_hour_vs_own_spikes_hour_own_mean_last_hour_)

        # make arrays
        p_own_spikes_own_mean_vs_own_spikes_other_means = np.hstack(p_own_spikes_own_mean_vs_own_spikes_other_means)
        print("Min. p-value --> Own spikes own mean vs. own spikes other means:")
        print(np.min(p_own_spikes_own_mean_vs_own_spikes_other_means))
        p_own_spikes_own_mean_vs_other_spikes_own_mean = np.hstack(p_own_spikes_own_mean_vs_other_spikes_own_mean)
        print("Min. p-value --> Own spikes own mean vs. other spikes own means:")
        print(np.min(p_own_spikes_own_mean_vs_other_spikes_own_mean))
        p_other_cell_spikes_own_mean_first_hour_vs_own_spikes_hour_own_mean_last_hour = \
            np.hstack(p_other_cell_spikes_own_mean_first_hour_vs_own_spikes_hour_own_mean_last_hour)
        print("Max. p-value --> across first hour vs. within last hour")
        print(np.max(p_other_cell_spikes_own_mean_first_hour_vs_own_spikes_hour_own_mean_last_hour))

    def cluster_quality_over_time_l_ratio(self, save_fig=False, hours_to_compute=[0, 20]):
        """
        Computes stability based on clustering features

        :param save_fig: save as .svg
        :type save_fig: bool
        :param hours_to_compute: for which hours to compute stability
        :type hours_to_compute: list
        """
        l_ratio = []
        sim_ratio = []

        for session in self.session_list:
            l_ratio_ = \
                session.all_data().assess_feature_stability_per_tetrode_l_ratio_per_hour(hours_to_compute=hours_to_compute)
            sim_ratio_, _ = \
                session.long_sleep().memory_drift_vs_duration_excluded_periods()

            l_ratio.append(l_ratio_)
            sim_ratio.append(sim_ratio_)

        # compute slopes of drift score for original and jittered case
        slopes_original = []

        for drift_score in sim_ratio:
            coef_original = np.polyfit(np.linspace(0, 1, drift_score.shape[0]), drift_score, 1)
            slopes_original.append(coef_original[0])

        if save_fig:
            plt.style.use('default')
        # plot difference in l_ratio per session vs. drift
        mean_per_sess = np.zeros(len(l_ratio))
        plt.figure(figsize=(7,10))
        for i_sess, l_ratio_sess in enumerate(l_ratio):
            # use values from first hour to z-score last values
            mean_fh = np.mean(l_ratio_sess[:, 0])
            std_fh = np.std(l_ratio_sess[:, 0])
            z_scored_lh = (l_ratio_sess[:,1]-mean_fh)/std_fh
            mean_per_sess[i_sess] = np.median(z_scored_lh)
            plt.subplot(3,3,i_sess+1)
            plt.boxplot(z_scored_lh, showfliers=False)
            abs_max = np.max([plt.gca().get_ylim()[1], np.abs(plt.gca().get_ylim()[0])])
            plt.ylim(-abs_max, abs_max)
            plt.text(0.55, 0, "Session "+str(i_sess+1))
            plt.xticks([])
            plt.hlines(0, 0.5, 1.5, color="grey", zorder=-1000, linestyle="--")
            if i_sess == 3:
                plt.ylabel("L_ratio (z-scored using L_ratio values from first hour)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "L_ratio_z_scored_all_sessions.svg"), transparent="True")
            plt.close()
        else:
            plt.show()
        # plot z-scored l_ratio of the last hour using l_ratio values from the first hour

        plt.figure(figsize=(4,4))
        plt.scatter(slopes_original, mean_per_sess)
        print(pearsonr(slopes_original, mean_per_sess))
        plt.xlabel("Slope of drift score")
        plt.ylabel("Median z-scored L_ratio")
        plt.text(0.4, -0.3, "R="+str(np.round(pearsonr(slopes_original, mean_per_sess)[0], 4)))
        plt.text(0.4, -0.32, "p="+str(np.round(pearsonr(slopes_original, mean_per_sess)[1], 4)))
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "L_ratio_z_scored_vs_drift_slope_all_sessions.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def l_ratio_vs_drift(self, save_fig=False):
        """
        Computes stability based on clustering features

        :param save_fig: save as .svg
        :type save_fig: bool
        :param hours_to_compute: for which hours to compute stability
        :type hours_to_compute: list
        """
        l_ratio = []
        sim_ratio = []

        for session in self.session_list:
            l_ratio_ = \
                session.all_data().assess_feature_stability_per_tetrode_l_ratio()
            sim_ratio_, _ = \
                session.long_sleep().memory_drift_vs_duration_excluded_periods()

            l_ratio.append(l_ratio_)
            sim_ratio.append(sim_ratio_)

        # compute slopes of drift score for original and jittered case
        slopes_original = []

        for drift_score in sim_ratio:
            coef_original = np.polyfit(np.linspace(0, 1, drift_score.shape[0]), drift_score, 1)
            slopes_original.append(coef_original[0])

        # get mean l_ratio per session
        mean_per_sess = [np.mean(np.hstack(x)) for x in l_ratio]
        mean_per_sess_z = zscore(mean_per_sess)

        # plot z-scored l_ratio of the last hour using l_ratio values from the first hour
        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(4,4))
        plt.scatter(slopes_original, mean_per_sess_z)
        plt.xlabel("Slope of drift score")
        plt.ylabel("L_ratio (z-scored)")
        plt.text(0.4, -0.3, "R="+str(np.round(pearsonr(slopes_original, mean_per_sess_z)[0], 4)))
        plt.text(0.4, -0.6, "p="+str(np.round(pearsonr(slopes_original, mean_per_sess_z)[1], 4)))
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "L_ratio_vs_drift_slope_all_sessions.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def waveform_stability(self, save_fig=False):
        """
        Compute waveform stability by dividing within similarity by across cell similarity

        :param save_fig: save as .svg
        :type save_fig: bool
        """
        ratios = []

        for session in self.session_list:
            r = session.all_data().waveform_stability(plotting=False)
            ratios.append(r)

        p_5_10 = []
        p_10_15 = []
        p_5_15 = []
        # go through all sessions and test significance
        for sess_id, sess_ratios in enumerate(ratios):
            sess_ratios = sess_ratios[1:]
            p_5_10.append(mannwhitneyu(sess_ratios[0], sess_ratios[1])[1])
            p_10_15.append(mannwhitneyu(sess_ratios[1], sess_ratios[2])[1])
            p_5_15.append(mannwhitneyu(sess_ratios[0], sess_ratios[2])[1])
            c = "white"
            # don't use ratios from first hour
            plt.figure(figsize=(4, 5))
            bplot = plt.boxplot(sess_ratios, positions=[1, 2, 3], patch_artist=True,
                                labels=["Hour 5", "Hour 10", "Hour 15"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            plt.ylabel("Within_corr/across_corr")
            plt.xlabel("Time")
            plt.title(self.session_list[sess_id].session_name)
            plt.show()

        plt.hist(p_5_10, bins=10, density=True)
        plt.xscale("log")
        plt.xlabel("p-value t-test, one-sided")
        plt.ylabel("Density")
        plt.title("Hour 5 vs. Hour 10")
        plt.xlim(0, 1)
        plt.show()
        plt.hist(p_10_15, bins=10, density=True)
        plt.xscale("log")
        plt.xlabel("p-value t-test, one-sided")
        plt.ylabel("Density")
        plt.title("Hour 10 vs. Hour 15")
        plt.xlim(0, 1)
        plt.show()
        plt.hist(p_5_15, bins=10, density=True)
        plt.xscale("log")
        plt.xlabel("p-value t-test, one-sided")
        plt.ylabel("Density")
        plt.title("Hour 5 vs. Hour 15")
        plt.xlim(0, 1)
        plt.show()

    def waveform_stability_per_tetrode(self, save_fig=False, hours_to_compute=None):
        """
        Computes waveform stabilty per tetrode

        :param save_fig: save as .svg
        :type save_fig: bool
        :param hours_to_compute: for which hours to compute stability
        :type hours_to_compute: list
        """
        if hours_to_compute is None:
            hours_to_compute = [0, 7, 14, 21]

        labels = [str(i) for i in hours_to_compute[1:]]

        ratios = []

        for session in self.session_list:
            r = session.all_data().waveform_stability_per_tetrode(plotting=False, hours_to_compute=hours_to_compute)
            ratios.append(r)

        ratios = np.hstack(ratios)

        print(ttest_ind(ratios[0],
                        ratios[1], alternative="greater"))
        print(ttest_ind(ratios[1],
                        ratios[2], alternative="greater"))
        print(ttest_ind(ratios[0],
                        ratios[2], alternative="greater"))
        print(mannwhitneyu(ratios[0],
                           ratios[1], alternative="greater"))
        print(mannwhitneyu(ratios[1],
                           ratios[2], alternative="greater"))
        print(mannwhitneyu(ratios[0],
                           ratios[2], alternative="greater"))

        plt.figure(figsize=(3, 5))
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        bplot = plt.boxplot(ratios.T, positions=[1, 2, 3],
                            patch_artist=True,
                            labels=labels,
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.ylabel("similarity within using mean / similarity across using mean")
        plt.ylim(0.5, 1.5)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "waveform_within_across_using_mean_all_sessions.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

    def waveform_stability_all_means_from_first_hour(self):
        """
        Computes stability using all means from first hour

        """
        p_ttest = []
        p_mwu = []

        for session in self.session_list:
            p_t, p_m = session.all_data().waveform_stability_per_tetrode(plotting=False)
            p_ttest.append(p_t)
            p_mwu.append(p_m)

        plt.hist(p_mwu, density=True, bins=20)
        plt.xlim(0,1)
        plt.vlines(0.05, 0, 1, color="red")
        plt.xlabel("p-value (Hour 5 vs. Hour 15)")
        plt.ylabel("CDF")
        plt.title("MWU")
        plt.show()

        plt.hist(p_ttest, density=True, bins=20)
        plt.xlim(0,1)
        plt.vlines(0.05, 0, 1, color="red")
        plt.xlabel("p-value (Hour 5 vs. Hour 15)")
        plt.ylabel("CDF")
        plt.title("t-test")
        plt.show()
        print("HERE")

    def feature_stability_subsets(self, save_fig=False):
        """
        Looks at clustering feature stability for different subsets of cells

        :param save_fig: save as .svg
        :type save_fig: bool
        """
        stable = []
        dec = []
        inc = []

        for session in self.session_list:
            s,d,i = session.all_data().feature_stability_per_tetrode_subsets(plotting=False)
            stable.append(s)
            dec.append(d)
            inc.append(i)

        stable = np.hstack(stable)
        dec = np.hstack(dec)
        inc = np.hstack(inc)
        print("ALL SESSIONS")
        print(ttest_ind(stable, dec))
        print(ttest_ind(stable, inc))
        print(ttest_ind(dec, inc))

        # combine data across sessions
        cmap = matplotlib.cm.get_cmap('viridis')
        colors_to_plot = cmap(np.linspace(0, 1, 3))
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        bplot = plt.boxplot([stable, dec, inc], positions=[1, 2, 3], patch_artist=True,
                            labels=["Stable", "Dec", "Inc"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )

        for patch, color in zip(bplot['boxes'], colors_to_plot):
            patch.set_facecolor(color)
        plt.ylabel("similarity within using mean / similarity with other mean")
        plt.ylim(0.4, 1.6)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "feature_stability_subsets.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def cluster_quality_over_time_subsets(self, hours_to_compute=[15, 20], save_fig=False):

        to_own_mean_vs_to_other_means_stable = []
        to_own_mean_vs_to_other_means_dec = []
        to_own_mean_vs_to_other_means_inc = []

        for session in self.session_list:
            s, d, i = session.all_data().cluster_quality_over_time_subsets(hours_to_compute=hours_to_compute,
                                                                           plotting=False)
            to_own_mean_vs_to_other_means_stable.append(s)
            to_own_mean_vs_to_other_means_dec.append(d)
            to_own_mean_vs_to_other_means_inc.append(i)

        to_own_mean_vs_to_other_means_stable = np.hstack(to_own_mean_vs_to_other_means_stable)
        to_own_mean_vs_to_other_means_dec = np.hstack(to_own_mean_vs_to_other_means_dec)
        to_own_mean_vs_to_other_means_inc = np.hstack(to_own_mean_vs_to_other_means_inc)

        for hour_id, hour in enumerate(hours_to_compute):

            plot_data = [to_own_mean_vs_to_other_means_stable[hour_id,:], to_own_mean_vs_to_other_means_dec[hour_id,:],
                         to_own_mean_vs_to_other_means_inc[hour_id,:]]
            print("##############################################")
            print("HOUR: "+str(hour))
            print("##############################################")
            print("persistent vs. decreasing")
            print(mannwhitneyu(plot_data[0], plot_data[1]))
            print("persistent vs. increasing")
            print(mannwhitneyu(plot_data[0], plot_data[2]))
            print("increasing vs. decreasing")
            print(mannwhitneyu(plot_data[2], plot_data[1]))
            plt.figure(figsize=(2, 3))
            cmap = matplotlib.cm.get_cmap('viridis')
            if save_fig:
                plt.style.use('default')
                c = "black"
            else:
                c = "white"
            bplot = plt.boxplot(plot_data, positions=[1, 2, 3],
                                patch_artist=True,
                                labels=["persistent", "decreasing", "increasing"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            plt.ylabel("All spikes: distance to own mean from first hour / \n "
                       "distance to means of other cells from first hour")
            plt.ylim(-0.01, 1.5)
            plt.title(str(hour) + "th hour")
            plt.xticks(rotation=45)
            plt.tight_layout()
            colors = ["violet", 'turquoise', "orange"]
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path,
                                         "spikes_own_other_mean_vs_other_means_subsets_example_hour_"+str(hour)+".svg"),
                            transparent="True")
                plt.close()
            else:
                plt.show()

    def pre_model_decoding_expl_fam_expl_cb_learning(self, save_fig=False, load_from_temp=True):

        pre = []
        fam = []
        exp = []

        for i, session in enumerate(self.session_list):
            pre_, fam_, exp_ = \
                session.exploration_exploration_cheeseboard().pre_model_decoding(plotting=False)
            pre.append(pre_)
            fam.append(fam_)
            exp.append(exp_)

        pre = np.hstack(pre)
        fam = np.hstack(fam)
        exp = np.hstack(exp)


        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        plt.figure(figsize=(3,4))
        # first and second half for acquisition
        res = [np.hstack(fam), np.hstack(pre)]

        bplot=plt.boxplot(res, positions=[1, 2], patch_artist=True,
                          labels=["Exploration \n familiar",
                                  "Acquisition"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        plt.ylim(-60,0)
        y_base=-10
        plt.hlines(y_base, 1, 2, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05:
            plt.text(1.4, y_base, "*", color=c)
        plt.ylabel("Log-likelihood")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("pre_likelihood_fam_exp_acquisition.svg", transparent="True")
        else:
            plt.show()

    def pre_long_sleep_theta_phase(self, save_fig=False, load_from_temp=True):
        print("Starting theta analysis ...")
        stable = []
        dec = []
        inc = []

        for session in self.session_list:
            if load_from_temp:
                infile = open("temp_data/theta_phase/" + session.session_name, 'rb')
                results = pickle.load(infile)
                s = results["diff_stable"]
                d = results["diff_dec"]
                i = results["diff_inc"]
            else:
                s,d,i = session.cheese_board_long_sleep(data_to_use="ext_eegh").theta_phase_preference_subsets(plotting=
                                                                                                           False)
            stable.append(s)
            dec.append(d)
            inc.append(i)

        diff_stable = []
        diff_dec = []
        diff_inc = []

        for s_, d_, i_ in zip(stable, dec, inc):
            # make all angles positive
            s_[s_ < 0] = 2*np.pi+s_[s_ < 0]
            d_[d_ < 0] = 2*np.pi+d_[d_ < 0]
            i_[i_ < 0] = 2*np.pi+i_[i_ < 0]

            diff_stable.append(s_)
            diff_dec.append(d_)
            diff_inc.append(i_)

        # concatenate data
        stable = np.hstack(stable)
        dec = np.hstack(dec)
        inc = np.hstack(inc)
        # plotting

        if save_fig:
            plt.style.use('default')
            c="black"
        else:
            c="white"
        # persistent cells
        # --------------------------------------------------------------------------------------------------------------
        bins_number = 10  # the [0, 360) interval will be subdivided into this
        # number of equal bins
        bins = np.linspace(0.0, 2 * np.pi, bins_number + 1)
        plt.clf()
        width = 2 * np.pi / bins_number
        ax = plt.subplot(1, 1, 1, projection='polar')
        n, _, _ = plt.hist(stable, bins, density=True, alpha=0)
        bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
        for bar in bars:
            # bar.set_alpha(0.5)
            bar.set_color("violet")
        ax.set_title("persistent cells \n phase_REM - phase_awake")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("theta_rem_awake_all_sessions_persistent.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        # decreasing cells
        # --------------------------------------------------------------------------------------------------------------
        ax = plt.subplot(1, 1, 1, projection='polar')
        n, _, _ = plt.hist(dec, bins, density=True, alpha=0)
        bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
        for bar in bars:
            # bar.set_alpha(0.5)
            bar.set_color("turquoise")
        ax.set_title("decreasing cells \n phase_REM - phase_awake")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("theta_rem_awake_all_sessions_dec.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        # increasing cells
        # --------------------------------------------------------------------------------------------------------------
        ax = plt.subplot(1, 1, 1, projection='polar')
        n, _, _ = plt.hist(inc, bins, density=True, alpha=0)
        bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
        for bar in bars:
            # bar.set_alpha(0.5)
            bar.set_color("orange")
        ax.set_title("increasing cells \n phase_REM - phase_awake")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("theta_rem_awake_all_sessions_inc.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        # all cells with stats
        # --------------------------------------------------------------------------------------------------------------
        plt.figure()
        ax = plt.subplot(1, 1, 1, projection='polar')
        ax.vlines(np.mean(inc), 0, 1, color="orange")
        ax.vlines(np.mean(dec), 0, 1, color="turquoise")
        ax.vlines(np.mean(stable), 0, 1, color="violet")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("theta_rem_awake_all_sessions.svg", transparent="True")
        else:
            plt.show()

        # save data as csv --> stats in R
        np.savetxt("stable.csv", stable, delimiter=",")
        np.savetxt("dec.csv", dec, delimiter=",")
        np.savetxt("inc.csv", inc, delimiter=",")

    def pre_long_sleep_theta_phase_separate(self, save_fig=False):
        print("Starting theta analysis ...")
        stable = []
        dec = []
        inc = []
        ls_stable = []
        ls_dec = []
        ls_inc = []

        for session in self.session_list:


            s,d,i, ls_s,ls_d,ls_i = \
                session.cheese_board_long_sleep(data_to_use="ext_eegh").theta_phase_preference_subsets(plotting=False, return_diff=False)
            stable.append(s)
            dec.append(d)
            inc.append(i)
            ls_stable.append(ls_s)
            ls_dec.append(ls_d)
            ls_inc.append(ls_i)

        # concatenate data
        stable = np.hstack(stable)
        dec = np.hstack(dec)
        inc = np.hstack(inc)
        ls_stable = np.hstack(ls_stable)
        ls_dec = np.hstack(ls_dec)
        ls_inc = np.hstack(ls_inc)
        # plotting

        if save_fig:
            plt.style.use('default')
            c="black"
        else:
            c="white"
        # persistent cells awake
        # --------------------------------------------------------------------------------------------------------------
        bins_number = 10  # the [0, 360) interval will be subdivided into this
        # number of equal bins
        bins = np.linspace(0.0, 2 * np.pi, bins_number + 1)
        plt.clf()
        width = 2 * np.pi / bins_number
        ax = plt.subplot(1, 1, 1, projection='polar')
        n, _, _ = plt.hist(stable, bins, density=True, alpha=0)
        bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
        for bar in bars:
            # bar.set_alpha(0.5)
            bar.set_color("violet")
        ax.set_title("persistent cells \n phase_awake")
        plt.legend(loc=4)
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("theta_awake_all_sessions_persistent.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        # persistent cells rem
        # --------------------------------------------------------------------------------------------------------------
        bins_number = 10  # the [0, 360) interval will be subdivided into this
        # number of equal bins
        bins = np.linspace(0.0, 2 * np.pi, bins_number + 1)
        plt.clf()
        width = 2 * np.pi / bins_number
        ax = plt.subplot(1, 1, 1, projection='polar')
        n, _, _ = plt.hist(ls_stable, bins, density=True, alpha=0)
        bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
        for bar in bars:
            # bar.set_alpha(0.5)
            bar.set_color("violet")
        ax.set_title("persistent cells \n phase_REM")
        plt.legend(loc=4)
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("theta_rem_all_sessions_persistent.svg", transparent="True")
            plt.close()
        else:
            plt.show()


        # decreasing cells: awake
        # --------------------------------------------------------------------------------------------------------------
        ax = plt.subplot(1, 1, 1, projection='polar')
        n, _, _ = plt.hist(dec, bins, density=True, alpha=0)
        bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
        for bar in bars:
            # bar.set_alpha(0.5)
            bar.set_color("turquoise")
        ax.set_title("decreasing cells \n phase_awake")
        plt.legend(loc=4)
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("theta_awake_all_sessions_dec.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        # decreasing cells: rem
        # --------------------------------------------------------------------------------------------------------------
        ax = plt.subplot(1, 1, 1, projection='polar')
        n, _, _ = plt.hist(ls_dec, bins, density=True, alpha=0)
        bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
        for bar in bars:
            # bar.set_alpha(0.5)
            bar.set_color("turquoise")
        ax.set_title("decreasing cells \n phase_REM")
        plt.legend(loc=4)
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("theta_rem_all_sessions_dec.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        # increasing cells
        # --------------------------------------------------------------------------------------------------------------
        ax = plt.subplot(1, 1, 1, projection='polar')
        n, _, _ = plt.hist(inc, bins, density=True, alpha=0)
        bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
        for bar in bars:
            # bar.set_alpha(0.5)
            bar.set_color("orange")
        ax.set_title("increasing cells \n phase_awake")
        plt.legend(loc=4)
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("theta_awake_all_sessions_inc.svg", transparent="True")
            plt.close()
        else:
            plt.show()


        # increasing cells: sleep
        # --------------------------------------------------------------------------------------------------------------
        ax = plt.subplot(1, 1, 1, projection='polar')
        n, _, _ = plt.hist(ls_inc, bins, density=True, alpha=0)
        bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
        for bar in bars:
            # bar.set_alpha(0.5)
            bar.set_color("orange")
        ax.set_title("increasing cells \n phase_rem")
        plt.legend(loc=4)
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("theta_rem_all_sessions_inc.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        # boxplot with stats
        # --------------------------------------------------------------------------------------------------------------
        res =[stable, dec, inc]
        plt.figure(figsize=(4,5))
        bplot=plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                          labels=["persistent", "decreasing", "increasing"],
                          boxprops=dict(color=c),
                          capprops=dict(color=c),
                          whiskerprops=dict(color=c),
                          flierprops=dict(color=c, markeredgecolor=c),
                          medianprops=dict(color=c), showfliers=False)
        plt.tick_params(axis='x', labelrotation=45)
        plt.ylim(0,9)
        y_base=7
        n_comp =3
        plt.hlines(y_base, 1, 1.9, color=c)
        if mannwhitneyu(res[0], res[1])[1] > 0.05/n_comp:
            plt.text(1.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.001/n_comp:
            plt.text(1.4, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.01/n_comp:
            plt.text(1.4, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[1])[1] < 0.05/n_comp:
            plt.text(1.4, y_base, "*", color=c)
        plt.hlines(y_base, 2.1, 3, color=c)
        if mannwhitneyu(res[2], res[1])[1] > 0.05/n_comp:
            plt.text(2.4, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[2], res[1])[1] < 0.001/n_comp:
            plt.text(2.4, y_base, "***", color=c)
        elif mannwhitneyu(res[2], res[1])[1] < 0.01/n_comp:
            plt.text(2.4, y_base, "**", color=c)
        elif mannwhitneyu(res[2], res[1])[1] < 0.05/n_comp:
            plt.text(2.4, y_base, "*", color=c)
        # first vs. second half NREM
        y_base=8
        plt.hlines(y_base, 1, 3, color=c)
        if mannwhitneyu(res[0], res[2])[1] > 0.05/n_comp:
            plt.text(2, y_base, "n.s.", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.001/n_comp:
            plt.text(2, y_base, "***", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.01/n_comp:
            plt.text(2, y_base, "**", color=c)
        elif mannwhitneyu(res[0], res[2])[1] < 0.05/n_comp:
            plt.text(2, y_base, "*", color=c)
        for patch, color in zip(bplot['boxes'], ["violet", "turquoise", "orange"]):
            patch.set_facecolor(color)

        plt.ylabel("Theta_REM_phase - \n Theta_awake_phase (rad)")
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("theta_rem_awake_all_sessions.svg", transparent="True")
        else:
            plt.show()

    # </editor-fold>
