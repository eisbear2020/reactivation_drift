########################################################################################################################
#
#
#   MAIN FOR NEURAL DATA ANALYSIS: REACTIVATION DRIFT
#
#
#   Description:
#
#                   1)  takes args as input or runs debug by default
#                   2)  classes with analysis methods can either be called directly or in analysis scripts in the
#                       analysis_scripts directory
#
#
#   Author: Lars Bollmann, IST Austria
#
#   First created: 28/01/2020
#
########################################################################################################################
import argparse
import importlib
from function_files.sessions import SingleSession, MultipleSessions, TwoPopSingleSession, TwoPopMultipleSessions
from parameter_files.standard_analysis_parameters import Parameters

if __name__ == '__main__':

    """#################################################################################################################
    #   Get parser arguments
    #################################################################################################################"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_step', default="neuron_main_figures", type=str)
    parser.add_argument('--cell_type', nargs="+", default=["p1"], help="CELL TYPE WITHIN """)
    parser.add_argument('--binning_method', default="temporal_spike", type=str,
                        help="[temporal, temporal_spike, temporal_binary, spike_binning]")
    parser.add_argument('--time_bin_size', default=1, type=float, help="TIME BIN SIZE IN SECONDS (FLOAT)")
    parser.add_argument("--data_to_use", default="ext", type=str,
                        help="use standard data [std] or extended (incl. lfp) data [ext]")
    parser.add_argument('--session_name', nargs="+", default=["mjc169R1R_0114"], type=str)
    parser.add_argument("--experiment_phase", nargs="+", default=["sleep_long_1"],
                        help="EXPERIMENT PHASE """)
    args = parser.parse_args()

    """#################################################################################################################
    #   LOAD STANDARD ANALYSIS PARAMETERS
    #################################################################################################################"""

    params = Parameters()

    """#################################################################################################################
    #   PRE-PROCESS ARGS
    #################################################################################################################"""

    if len(args.session_name) == 1:
        # if one phase is provided
        session_name = args.session_name[0]

    else:
        # if two experiment phases are supposed to be analyzed
        session_name = args.session_name

    if len(args.cell_type) == 1:
        # if only one cell type is provided --> analyzing one region
        cell_type = args.cell_type[0]

    else:
        # if two cell types are provided --> analyze both regions
        cell_type = args.cell_type

    """#################################################################################################################
    #   SAVING DIRECTORY FOR (TEMPORARY) RESULTS AND DATA
    #################################################################################################################"""

    # assign directory for pre-processed data
    # ------------------------------------------------------------------------------------------------------------------
    params.pre_proc_dir = None

    if params.pre_proc_dir is None:
        raise Exception("Need to define directory for pre-processed data")

    """#################################################################################################################
        #   MAIN ANALYSIS PART
    #################################################################################################################"""

    if len(args.session_name) == 1:
        """#############################################################################################################
        # SINGLE SESSION
        #############################################################################################################"""
        if args.compute_step == "debug":
            # example for an analysis of a single session: how to call/run a function
            single_ses = SingleSession(session_name=session_name, params=params,
                                       cell_type=cell_type).long_sleep()
            single_ses.memory_drift_decoding_quality_across_time_likelihoods(save_result=True)
        else:
            # run existing analysis scripts (from analysis_scripts directory)
            routine = importlib.import_module("analysis_scripts."+args.compute_step)
            routine.execute(params=params)

        """#############################################################################################################
        # FOR MULTIPLE SESSIONS
        #############################################################################################################"""
    else:
        if args.compute_step == "debug":
            # example for an analysis of multiple sessions: how to call/run a function
            tpms = TwoPopMultipleSessions(session_names=session_name, params=params, cell_type_1="p1", cell_type_2="b1")
            tpms.memory_drift_delta_score_interneuron_firing(use_abs_delta_score=True, invert_rem_sign=False,
                                                             control_mul_comp=False)
        else:
            # run existing analysis scripts (from analysis_scripts directory)
            routine = importlib.import_module("analysis_scripts."+args.compute_step)
            routine.execute(params=params)

