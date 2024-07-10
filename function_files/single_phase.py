########################################################################################################################
#
#   SINGLE PHASE CLASSES
#
#   Description: contains classes for different SINGLE experimental phases
#
#   Author: Lars Bollmann
#
#   Last modified: 19/04/2022
#
#   Structure:
#
#               (1) class BaseMethods: methods to analyze sleep and awake data for ONE POPULATION
#
#                   (a) class Sleep: derived from BaseMethods --> used to analyze sleep data
#
#                   (b) class Exploration: derived from BaseMethods --> used to analyze exploration data
#
#               (2) class Cheeseboard: methods to analyze cheeseboard task data
#
########################################################################################################################

import numpy as np
import math
from scipy import signal
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import collections as matcoll
from matplotlib.patches import Circle
import numpy as matlib
from scipy.spatial.distance import pdist
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import median_abs_deviation
import itertools
import copy
from scipy.special import factorial, gammaln
import random
from functools import partial
import multiprocessing as mp
import os
import matplotlib
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from numpy.linalg import norm
from math import floor
from scipy import io
from scipy.spatial import distance
from scipy.stats import pearsonr, entropy, mannwhitneyu, multivariate_normal, \
    zscore, ttest_ind_from_stats
import scipy.ndimage as nd
from scipy.io import loadmat
from sklearn.metrics import pairwise_distances
from scipy import optimize
from collections import OrderedDict
import pywt

from .support_functions import upper_tri_without_diag, independent_shuffle,\
    simple_gaussian, find_hse, \
    butterworth_bandpass, butterworth_lowpass, moving_average, evaluate_clustering_fit, decode_using_ising_map, \
    decode_using_phmm_modes, compute_values_from_likelihoods, \
    generate_colormap, correlations_from_raster, bayes_likelihood, collective_goal_coding, \
    distance_peak_firing_to_closest_goal, make_square_axes, compute_spatial_information, cross_correlate_matrices, \
    down_sample_array_sum, read_integers

from .pre_processing import PreProcessSleep, PreProcessAwake
from .plotting_functions import plot_pop_clusters
from .ml_methods import MlMethodsOnePopulation, PoissonHMM

# define default location to save plots
import os
save_path = os.path.dirname(os.path.realpath(__file__)) + "/../plots"


class BaseMethods:
    """Base class for general electro-physiological data analysis"""

    def __init__(self, data_dic, cell_type, params, session_params, experiment_phase):
        # --------------------------------------------------------------------------------------------------------------
        # args: - data_dic, dictionary with standard data
        #       - cell_type, string: cell type to do analysis with
        #       - params, python class: contains all parameters
        # --------------------------------------------------------------------------------------------------------------

        # get parameters
        self.params = copy.deepcopy(params)
        self.session_params = session_params
        self.session_name = self.session_params.session_name
        # get spatial factor: cm per .whl arbitrary unit
        self.spatial_factor = self.session_params.data_params_dictionary["spatial_factor"]

        # --------------------------------------------------------------------------------------------------------------
        # get phase specific info (ID & description)
        # --------------------------------------------------------------------------------------------------------------

        # check if list is passed:
        if isinstance(experiment_phase, list):
            experiment_phase = experiment_phase[0]

        self.experiment_phase = experiment_phase
        self.experiment_phase_id = session_params.data_description_dictionary[self.experiment_phase]

        # --------------------------------------------------------------------------------------------------------------
        # LFP PARAMETERS
        # --------------------------------------------------------------------------------------------------------------
        # if set to an integer --> only this tetrode is used to detect SWR etc., if None --> all tetrodes are used
        # check what tetrodes are assigned to different populations/hemispheres
        if hasattr(self.session_params, "tetrodes_p1_l") and hasattr(self.session_params, "tetrodes_p1_r"):
            if cell_type == "p1_l":
                self.session_params.lfp_tetrodes = self.session_params.tetrodes_p1_l
            elif cell_type == "p1_r":
                self.session_params.lfp_tetrodes = self.session_params.tetrodes_p1_r
        else:
            self.session_params.lfp_tetrodes = None

        # get data dictionary
        # check if list or dictionary is passed:
        if isinstance(data_dic, list):
            data_dic = data_dic[0]
        else:
            data_dic = data_dic

        # get all spike times
        self.firing_times = data_dic["spike_times"][cell_type]

        # get last recorded spike
        if "last_spike" in data_dic.keys():
            self.last_spike = data_dic["last_spike"]
        else:
            self.last_spike = None

        # get location data
        self.whl = data_dic["whl"]

        # check if extended data dictionary is provided (contains lfp)
        if "eeg" in data_dic.keys():
            self.eeg = data_dic["eeg"]
        if "eegh" in data_dic.keys():
            self.eegh = data_dic["eegh"]

        # which cell type to be analyzed
        self.cell_type = cell_type

        # initialize raster, loc and vel as None
        self.raster = None
        self.loc = None
        self.speed = None

        # initialize dimensionality reduction results as None
        self.result_dr = None

    # <editor-fold desc="Plotting & getter functions">

    # Plotting & getter functions
    # ------------------------------------------------------------------------------------------------------------------

    def view_raster(self):
        """
        plot raster data
        """
        raster = self.raster[:, 0:100]
        plt.imshow(raster.T)
        plt.xlabel("Time bins")
        plt.ylabel("Cells")

    def get_raster(self):
        """
         return raster
        """
        return self.raster

    def get_nr_cells(self):
        return len(self.firing_times)

    def save_raster(self, file_format=None, file_name=None):
        """
        save raster as file

        @param file_format: ["mat"] or None: file format e.g. for MATLAB
        @type file_format: str
        @param file_name: name of file
        @type file_name: str
        """

        if file_name is None:
            # if file name is not provided derive one
            file_name = self.cell_type + "_" + self.params.binning_method + "_" + \
                        str(self.params.time_bin_size)+"s"

        if file_format is None:
            np.save(file_name, self.raster, fix_imports=True)
        elif file_format == "mat":
            # export data as .mat file to use in matlab
            io.savemat(file_name+".mat", {"raster": self.raster})

    def get_raster_loc_vel(self):
        # --------------------------------------------------------------------------------------------------------------
        # return raster, location and speed
        # --------------------------------------------------------------------------------------------------------------

        return self.raster, self.loc, self.speed

    def get_location(self):
        # --------------------------------------------------------------------------------------------------------------
        # return transformed location data
        # --------------------------------------------------------------------------------------------------------------
        return self.loc

    def plot_location_data(self):
        plt.scatter(self.loc[:, 0], self.loc[:, 1])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def plot_speed(self):
        speed = self.speed
        plt.plot(speed)
        plt.show()

    def get_speed(self):
        speed = self.speed()
        return speed

    @staticmethod
    def get_cell_id(data_dir, session_name, cell_type):
        # --------------------------------------------------------------------------------------------------------------
        # returns cell IDs from .des file for the selected cell type
        #
        # args:         - data_dir, str: directory where data folders are stored (directory containig all mjc.. folders)
        #               - session_name, str: "e.g. mjc163_2_0104"
        #               - cell type, str: can be on of the following (updated: 16.11.2020)
        #
        #                   - u1: unidentified?
        #                   - p1: pyramidal cells of the HPC
        #                   - p2 - p3: pyramidal cells of the PFC
        #                   - b1: interneurons of HPC
        #                   - b2 - b3: interneurons of HPC
        #                   - pe: pyramidal cells MEC
        #
        # returns:      - cell_ids, list: list containing cell ids
        # --------------------------------------------------------------------------------------------------------------

        with open(data_dir + "/" + session_name + "/" + session_name + ".des") as f:
            des = f.read()
        des = des.splitlines()
        # offset by 2 entries
        cell_ids = [i + 2 for i in range(len(des)) if des[i] == cell_type]

        return cell_ids

    # </editor-fold>

    # <editor-fold desc="LFP analysis">

    # LFP analysis
    # ------------------------------------------------------------------------------------------------------------------

    def detect_swr_michele(self, thr=4):
        """
        detects sharp wave ripples (SWR) and returns start, peak and end timings at params.time_bin_size resolution

        @param thr: nr. std. to use for detection
        @type thr: int
        @return: SWR
        @rtype:
        """

        print("NEEDS TO BE IMPLEMENTED!")
        # ["eeg"] --> all channels downsampled 16 times (1.25kHz)
        # ["eegh"] --> one channel per tetrode downsampled 4 times (5kHz) --> time bin size = 0.0002s
        # swr_pow = compute_power_stft(input_data=self.eegh, input_data_time_res=0.0002,
        #                         output_data_time_res=self.params.time_bin_size)
        #
        # plt.plot(swr_pow[:1000])
        # plt.show()

        # upper and lower bound in Hz for SWR
        freq_lo_bound = 140
        freq_hi_bound = 240

        # load data
        # ["eeg"] --> all channels downsampled 16 times (1.25kHz)
        # ["eegh"] --> one channel per tetrode downsampled 4 times (5kHz) --> time bin size = 0.0002s

        # only select one tetrode
        data = self.eegh[:, 11]
        freq = 5000
        low_pass_cut_off_freq = 30

        # nyquist theorem --> need half the frequency
        sig_bandpass = butterworth_bandpass(input_data=data, freq_nyquist=freq/2, freq_lo_bound=freq_lo_bound,
                                            freq_hi_bound=freq_hi_bound)
        # compute rectified signal
        sig_abs = np.abs(sig_bandpass)
        # low pass filter signal
        sig_lo_pass = butterworth_lowpass(input_data=sig_abs, freq_nyquist=freq/2, cut_off_freq=low_pass_cut_off_freq)
        # z-score
        sig_z_scored = zscore(sig_lo_pass)

        swr = []  # return [beginning, end, peak] of each event, in whl time indexes

        # add additional zero at the end
        swr_p = np.array(sig_z_scored, copy=True)
        swr_p = np.hstack([swr_p, 0])
        # check when it is above / below threshold
        der = np.array(swr_p[1:] > thr, dtype=int) - np.array(swr_p[:-1] > thr, dtype=int)
        # beginnings are where  der > 0
        begs = np.where(der > 0)[0] + 1
        last = 0
        for beg in begs:  # check each swr
            if beg > last:  # not to overlap
                # include 50 ms before: usually a lot of spiking happens before high SWR power
                first = max(beg - np.round(50/(1/freq)).astype(int), 0)
                # just a sanity check - something is wrong if this is not satisfied - probably threshold too low!!
                if np.min(swr_p[beg:beg + np.round(1000/self.params.time_bin_size).astype(int)]) < 0.8 * thr:
                    # end SWR where power is less 80% threshold
                    last = beg + np.where(swr_p[beg:beg + np.round(1000/(1/freq)).astype(int)]
                                          < 0.8 * thr)[0][0]
                    # peak power
                    peak = first + np.argmax(swr_p[first:last])
                    # check length: must be between 75 and 750 ms, else something is off
                    if (np.round(75/(1/freq)).astype(int) < last - first <
                            np.round(750/(1/freq)).astype(int)):
                        swr.append([first, last, peak])
        return swr

    def analyze_lfp(self):

        # upper and lower bound in Hz for SWR
        freq_lo_bound = 140
        freq_hi_bound = 240

        # load data
        # ["eeg"] --> all channels downsampled 16 times (1.25kHz)
        # ["eegh"] --> one channel per tetrode downsampled 4 times (5kHz) --> time bin size = 0.0002s

        combined = []

        for tet in range(15):

            # only select one tetrode --> TODO: maybe select multiple, compute SWR times and take overlap
            data = self.eegh[:, tet]
            # freq of the input signal (eegh --> 5kHz --> freq=5000)
            freq = 5000
            # for low pass filtering of the signal before z-scoring (20-30Hz is good)
            low_pass_cut_off_freq = 30
            # min time gap between swr in seconds
            # min_gap_between_events = 0.3

            # nyquist theorem --> need half the frequency
            sig_bandpass = butterworth_bandpass(input_data=data, freq_nyquist=freq/2, freq_lo_bound=freq_lo_bound,
                                                freq_hi_bound=freq_hi_bound)

            # compute rectified signal
            sig_abs = np.abs(sig_bandpass)

            # if only peak position is supposed to be returned

            # low pass filter signal
            sig_lo_pass = butterworth_lowpass(input_data=sig_abs, freq_nyquist=freq/2,
                                              cut_off_freq=low_pass_cut_off_freq)

            combined.append(sig_lo_pass)

        combined = np.array(combined)
        sig_lo_pass = np.mean(combined, axis=0)
        # z-score
        sig_z_scored = zscore(sig_lo_pass)

        plt.plot(sig_z_scored[:100000])
        plt.show()

    def detect_swr(self, thr=4, plot_for_control=False):
        """
        detects swr in lfp and returns start, peak and end timings at params.time_bin_size resolution
        ripple frequency: 140-240 Hz

        @param thr: nr. std. above average to detect ripple event (usually: 4-6)
        @type thr: int
        @param plot_for_control: True to plot intermediate results
        @type plot_for_control: bool
        @return: start, end, peak of each swr in seconds
        @rtype: int, int, int
        """
        file_name = \
            self.session_name + "_" + self.experiment_phase_id + "_swr_" + self.cell_type + "_tet_" + str(
                self.session_params.lfp_tetrodes) + ".npy"
        # check if results exist already
        if not os.path.isfile(self.params.pre_proc_dir+"swr_periods/" + file_name):

            # check if results exist already --> if not

            # upper and lower bound in Hz for SWR
            freq_lo_bound = 140
            freq_hi_bound = 240

            # load data
            # ["eeg"] --> all channels downsampled 16 times (1.25kHz)
            # ["eegh"] --> one channel per tetrode downsampled 4 times (5kHz) --> time bin size = 0.0002s

            # check if one tetrode or all tetrodes to use
            if self.session_params.lfp_tetrodes is None:
                print(" - DETECTING SWR USING ALL TETRODES ...\n")
                data = self.eegh[:, :]
            else:
                print(" - DETECTING SWR USING TETRODE(S) "+str(self.session_params.lfp_tetrodes) + " ...\n")
                data = self.eegh[:, self.session_params.lfp_tetrodes]
            # freq of the input signal (eegh --> 5kHz --> freq=5000)
            freq = 5000
            # for low pass filtering of the signal before z-scoring (20-30Hz is good)
            low_pass_cut_off_freq = 30
            # minimum gap in seconds between events. If two events have
            # a gap < min_gap_between_events --> events are joint and become one event
            min_gap_between_events = 0.1

            # if data is too large --> need to chunk it up

            if data.shape[0] > 10000000:
                start_times = np.zeros(0)
                peak_times = np.zeros(0)
                end_times = np.zeros(0)
                size_chunk = 10000000
                for nr_chunk in range(np.ceil(data.shape[0]/size_chunk).astype(int)):
                    chunk_data = data[nr_chunk*size_chunk:min(data.shape[0], (nr_chunk+1)*size_chunk)]

                    # compute offset in seconds for current chunk
                    offset_sec = nr_chunk * size_chunk * 1/freq

                    start_times_chunk, end_times_chunk, peak_times_chunk \
                        = self.detect_lfp_events(data=chunk_data, freq=freq, thr=thr, freq_lo_bound=freq_lo_bound,
                                                 freq_hi_bound=freq_hi_bound,
                                                 low_pass_cut_off_freq=low_pass_cut_off_freq,
                                                 min_gap_between_events=min_gap_between_events,
                                                 plot_for_control=plot_for_control)

                    # check if event was detected
                    if start_times_chunk is not None:
                        start_times = np.hstack((start_times, (start_times_chunk + offset_sec)))
                        end_times = np.hstack((end_times, (end_times_chunk + offset_sec)))
                        peak_times = np.hstack((peak_times, (peak_times_chunk + offset_sec)))

            else:
                # times in seconds
                start_times, end_times, peak_times = \
                    self.detect_lfp_events(data=data, freq=freq, thr=thr, freq_lo_bound=freq_lo_bound,
                                           freq_hi_bound=freq_hi_bound, low_pass_cut_off_freq=low_pass_cut_off_freq,
                                           min_gap_between_events=min_gap_between_events,
                                           plot_for_control=plot_for_control)

            result_dic = {
                "start_times": start_times,
                "end_times": end_times,
                "peak_times": peak_times
            }

            outfile = open(self.params.pre_proc_dir+"swr_periods/"+file_name, 'wb')
            pickle.dump(result_dic, outfile)
            outfile.close()

        # load results from file
        infile = open(self.params.pre_proc_dir+"swr_periods/" + file_name, 'rb')
        result_dic = pickle.load(infile)
        infile.close()

        start_times = result_dic["start_times"]
        end_times = result_dic["end_times"]
        peak_times = result_dic["peak_times"]

        print(" - " + str(start_times.shape[0]) + " SWRs FOUND\n")

        return start_times, end_times, peak_times

    @staticmethod
    def detect_lfp_events(data, freq, thr, freq_lo_bound, freq_hi_bound, low_pass_cut_off_freq,
                          min_gap_between_events, plot_for_control=False):
        """
        detects events in lfp and returns start, peak and end timings at second resolution

        @param data: input data (either from one or many tetrodes)
        @type data: array [nxm]
        @param freq: sampling frequency of input data in Hz
        @type freq: int
        @param thr: nr. std. above average to detect ripple event
        @type thr: int
        @param freq_lo_bound: lower bound for frequency band in Hz
        @type freq_lo_bound: int
        @param freq_hi_bound: upper bound for frequency band in Hz
        @type freq_hi_bound: int
        @param low_pass_cut_off_freq: cut off frequency for envelope in Hz
        @type low_pass_cut_off_freq: int
        @param min_gap_between_events: minimum gap in seconds between events. If two events have
         a gap < min_gap_between_events --> events are joint and become one event
        @type min_gap_between_events: float
        @param plot_for_control: plot some examples to double check detection
        @type plot_for_control: bool
        @return: start_times, end_times, peak_times of each event in seconds --> are all set to None if no event was
        detected
        @rtype: array, array, array
        """

        # check if data from one or multiple tetrodes was provided
        if len(data.shape) == 1:
            # only one tetrode
            # nyquist theorem --> need half the frequency
            sig_bandpass = butterworth_bandpass(input_data=data, freq_nyquist=freq / 2, freq_lo_bound=freq_lo_bound,
                                                freq_hi_bound=freq_hi_bound)

            # compute rectified signal
            sig_abs = np.abs(sig_bandpass)

            # if only peak position is supposed to be returned

            # low pass filter signal
            sig_lo_pass = butterworth_lowpass(input_data=sig_abs, freq_nyquist=freq / 2,
                                              cut_off_freq=low_pass_cut_off_freq)
            # z-score
            sig_z_scored = zscore(sig_lo_pass)

        else:
            # multiple tetrodes
            combined_lo_pass = []
            # go trough all tetrodes
            for tet_data in data.T:
                # nyquist theorem --> need half the frequency
                sig_bandpass = butterworth_bandpass(input_data=tet_data, freq_nyquist=freq/2,
                                                    freq_lo_bound=freq_lo_bound, freq_hi_bound=freq_hi_bound)

                # compute rectified signal
                sig_abs = np.abs(sig_bandpass)

                # if only peak position is supposed to be returned

                # low pass filter signal
                sig_lo_pass = butterworth_lowpass(input_data=sig_abs, freq_nyquist=freq/2,
                                                  cut_off_freq=low_pass_cut_off_freq)

                combined_lo_pass.append(sig_lo_pass)

            combined_lo_pass = np.array(combined_lo_pass)
            avg_lo_pass = np.mean(combined_lo_pass, axis=0)

            # z-score
            sig_z_scored = zscore(avg_lo_pass)

        # find entries above the threshold
        bool_above_thresh = sig_z_scored > thr
        sig = bool_above_thresh.astype(int) * sig_z_scored

        # find event start / end
        diff = np.diff(sig)
        start = np.argwhere(diff > 0.8 * thr)
        end = np.argwhere(diff < -0.8 * thr)

        # check that first element is actually the start (not that event started before this chunk and we only
        # observe the end of the event)
        if end[0] < start[0]:
            # if first end is before first start --> need to delete first end
            print("  --> CURRENT CHUNK: FIRST END BEFORE FIRST START --> DELETED FIRST END ELEMENT ")
            end = end[1:]

        if end[-1] < start[-1]:
            # there is another start after the last end --> need to delete last start
            print("  --> CURRENT CHUNK: LAST START AFTER LAST END --> DELETED LAST START ELEMENT ")
            start = start[:-1]

        # join events if there are less than min_gap_between_events seconds apart --> this is then one event!
        # compute differences between start time of n+1th event with end time of nth --> if < gap --> delete both
        # entries
        gap = np.squeeze((start[1:] - end[:-1]) * 1 / freq)
        to_delete = np.argwhere(gap < min_gap_between_events)
        end = np.delete(end, to_delete)
        start = np.delete(start, to_delete + 1)

        # add 25ms to the beginning of event (many spikes occur in that window)
        pad_infront = np.round(0.025/(1/freq)).astype(int)
        start -= pad_infront
        # don't want negative values (in case event happens within the 50ms of the recording)
        start[start < 0] = 0

        # # add 20ms to the end of event
        # pad_end = np.round(0.02/(1/freq)).astype(int)
        # end += pad_end
        # # don't want to extend beyond the recording
        # end[end > sig.shape[0]] = sig.shape[0]

        # check length of events --> shouldn't be shorter than 95 ms or larger than 750 ms
        len_events = (end - start) * 1 / freq
        #
        # plt.hist(len_events, bins=50)
        # plt.show()
        # exit()

        to_delete_len = np.argwhere((0.75 < len_events) | (len_events < 0.05))

        start = np.delete(start, to_delete_len)
        end = np.delete(end, to_delete_len)

        peaks = []
        for s, e in zip(start, end):
            peaks.append(s+np.argmax(sig[s:e]))

        peaks = np.array(peaks)

        # check if there were any events detected --> if not: None
        if not peaks.size == 0:
            # get peak times in s
            time_bins = np.arange(data.shape[0]) * 1 / freq
            peak_times = time_bins[peaks]
            start_times = time_bins[start]
            end_times = time_bins[end]
        else:
            peak_times = None
            start_times = None
            end_times = None

        # plot some events with start, peak and end for control
        if plot_for_control:
            a = np.random.randint(0, start.shape[0], 5)
            # a = range(start.shape[0])
            for i in a:
                plt.plot(sig_z_scored, label="z-scored signal")
                plt.vlines(start[i], 0, 15, colors="r", label="start")
                plt.vlines(peaks[i], 0, 15, colors="y", label="peak")
                plt.vlines(end[i], 0, 15, colors="g", label="end")
                plt.xlim((start[i] - 5000), (end[i] + 5000))
                plt.ylabel("LFP FILTERED (140-240Hz) - Z-SCORED")
                plt.xlabel("TIME BINS / "+str(1/freq) + " s")
                plt.legend()
                plt.title("EVENT DETECTION, EVENT ID "+str(i))
                plt.show()

        return start_times, end_times, peak_times

    def phase_preference_per_cell_subset(self, angle_20k, cell_ids):
        """
        Phase preference analysis for subsets of cells

        :param angle_20k: oscillation angle at 20kHz
        :type angle_20k: numpy.array
        :param cell_ids: cell ids of cells to be used
        :type cell_ids: numpy.array
        :return: preferred angle per cell
        :rtype: numpy.array
        """
        # spike times at 20kHz
        spike_times = self.firing_times

        # get keys from dictionary and get correct order
        cell_names = []
        for key in spike_times.keys():
            cell_names.append(key[4:])
        cell_names = np.array(cell_names).astype(int)
        cell_names.sort()

        pref_angle = []

        for cell_id in cell_names[cell_ids]:
            all_cell_spikes = spike_times["cell" + str(cell_id)]
            # remove spikes that like outside array
            all_cell_spikes = all_cell_spikes[all_cell_spikes < angle_20k.shape[0]]
            # make array
            spk_ang = angle_20k[all_cell_spikes]
            pref_angle.append(np.angle(np.sum(np.exp(-1j * spk_ang))))

        return np.array(pref_angle)

    def phase_preference_analysis(self, oscillation="theta", tetrode=1, plot_for_control=False, plotting=True):
        """
        LFP phase preference analysis for stable, inc, dec subsets

        :param oscillation: which oscillation to use ("theta", "slow_gamma", "medium_gamma")
        :type oscillation: str
        :param tetrode: which tetrode to use
        :type tetrode: int
        :param plot_for_control: if intermediate results are supposed to be plotted
        :type plot_for_control: bool
        :param plotting: plot results
        :type plotting: bool
        :return: angles (all positive) for stable, increasing, decreasing subsets
        :rtype: numpy.array
        """
        # get lfp data
        # ["eeg"] --> all channels downsampled 16 times (1.25kHz)
        # ["eegh"] --> one channel per tetrode downsampled 4 times (5kHz) --> time bin size = 0.0002s
        lfp = self.eegh[:, tetrode]

        # downsample to dt = 0.001 --> 1kHz --> take every 5th value
        lfp = lfp[::5]

        # Say you have an LFP signal LFP_Data and some spikes from a cell spk_t
        # First we extract the angle from the signal in a specific frequency band
        # Frequency Range to Extract, you can also select it AFTER running the wavelet on the entire frequency spectrum,
        # by using the variable frequency to select the desired ones
        if oscillation == "theta":
            frq_limits = [8, 12]
        elif oscillation == "slow_gamma":
            frq_limits = [20, 50]
        elif oscillation == "medium_gamma":
            frq_limits = [60, 90]
        else:
            raise Exception("Oscillation not defined!")
        # [8,12] Theta
        # [20,50] Slow Gamma
        # [60,90] Medium Gamma
        # LFP time bin duration in seconds
        # dt = 1/5e3
        dt = 0.001
        # ‘morl’ wavelet
        wavelet = "cmor1.5-1.0"
        scales = np.arange(1, 128)
        s2f = pywt.scale2frequency(wavelet, scales) / dt
        # This block is just to setup the wavelet analysis
        scales = scales[(s2f >= frq_limits[0]) * (s2f < frq_limits[1])]
        # scales = scales[np.logical_and(s2f >= Frq_Limits[0], s2f < Frq_Limits[1])]
        print(" - started wavelet decomposition ...")
        # Wavelet decomposition
        [cfs, frequencies] = pywt.cwt(data=lfp, scales=scales, wavelet=wavelet, sampling_period=dt)
        print(" - done!")
        # This is the angle
        angl = np.angle(np.sum(cfs, axis=0))

        # plot for control
        if plot_for_control:
            plt.plot(lfp[:200])
            plt.xlabel("Time")
            plt.ylabel("LFP")
            plt.show()

            for i in range(frequencies.shape[0]):
                plt.plot(cfs[i, :200])
            plt.xlabel("Time")
            plt.ylabel("Coeff")
            plt.show()

            plt.plot(np.sum(cfs[:, :200], axis=0), label="coeff_sum")
            plt.plot(angl[:200]/np.max(angl[:200]), label="angle")
            plt.xlabel("Time")
            plt.ylabel("Angle (norm) / Coeff_sum (norm)")
            plt.legend()
            plt.show()

        # interpolate results to match 20k
        # --------------------------------------------------------------------------------------------------------------
        x_1k = np.arange(lfp.shape[0])*dt
        x_20k = np.arange(lfp.shape[0]*20)*1/20e3
        angle_20k = np.interp(x_20k, x_1k, angl, left=np.nan, right=np.nan)

        if plot_for_control:
            plt.plot(angle_20k[:4000])
            plt.ylabel("Angle")
            plt.xlabel("Time bin (20kHz)")
            plt.show()

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        stable = class_dic["stable_cell_ids"]
        dec = class_dic["decrease_cell_ids"]
        inc = class_dic["increase_cell_ids"]

        pref_angle_stable = self.phase_preference_per_cell_subset(angle_20k=angle_20k, cell_ids=stable)
        pref_angle_dec = self.phase_preference_per_cell_subset(angle_20k=angle_20k, cell_ids=dec)
        pref_angle_inc = self.phase_preference_per_cell_subset(angle_20k=angle_20k, cell_ids=inc)

        pref_angle_stable_deg = pref_angle_stable * 180 / np.pi
        pref_angle_dec_deg = pref_angle_dec * 180 / np.pi
        pref_angle_inc_deg = pref_angle_inc * 180 / np.pi

        if plotting:
            plt.hist(pref_angle_stable_deg, density=True, label="stable")
            plt.hist(pref_angle_dec_deg, density=True, label="dec")
            plt.hist(pref_angle_inc_deg, density=True, label="inc")
            plt.show()

        all_positive_angles_stable = np.copy(pref_angle_stable)
        all_positive_angles_stable[all_positive_angles_stable < 0] = \
            2*np.pi+all_positive_angles_stable[all_positive_angles_stable < 0]

        all_positive_angles_dec = np.copy(pref_angle_dec)
        all_positive_angles_dec[all_positive_angles_dec < 0] = 2 * np.pi + all_positive_angles_dec[
            all_positive_angles_dec < 0]

        all_positive_angles_inc = np.copy(pref_angle_inc)
        all_positive_angles_inc[all_positive_angles_inc < 0] = 2 * np.pi + all_positive_angles_inc[
            all_positive_angles_inc < 0]

        if plotting:

            bins_number = 10  # the [0, 360) interval will be subdivided into this
            # number of equal bins
            bins = np.linspace(0.0, 2 * np.pi, bins_number + 1)
            angles = all_positive_angles_stable
            n, _, _ = plt.hist(angles, bins, density=True)

            plt.clf()
            width = 2 * np.pi / bins_number
            ax = plt.subplot(projection='polar')
            bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
            for bar in bars:
                bar.set_alpha(0.5)
            ax.set_title("stable cells")
            plt.show()
            angles = all_positive_angles_dec
            n, _, _ = plt.hist(angles, bins, density=True)

            plt.clf()
            width = 2 * np.pi / bins_number
            ax = plt.subplot(projection='polar')
            bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
            for bar in bars:
                bar.set_alpha(0.5)
            ax.set_title("dec. cells")
            plt.show()

            angles = all_positive_angles_inc
            n, _, _ = plt.hist(angles, bins, density=True)

            plt.clf()
            width = 2 * np.pi / bins_number
            ax = plt.subplot(projection='polar')
            bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
            for bar in bars:
                bar.set_alpha(0.5)
            ax.set_title("inc. cells")
            plt.show()

        else:
            return all_positive_angles_stable, all_positive_angles_dec, all_positive_angles_inc

    def swr_profile(self, thr=4, plot_for_control=False):
        """
        detects swr in lfp and returns start, peak and end timings at params.time_bin_size resolution
        ripple frequency: 140-240 Hz

        @param thr: nr. std. above average to detect ripple event (usually: 4-6)
        @type thr: int
        @param plot_for_control: True to plot intermediate results
        @type plot_for_control: bool
        @return: start, end, peak of each swr in seconds
        @rtype: int, int, int
        """
        data_dir = self.session_params.data_params_dictionary["data_dir"]
        # upper and lower bound in Hz for SWR
        freq_lo_bound = 140
        freq_hi_bound = 240
        # freq of the input signal (eegh --> 5kHz --> freq=5000)
        freq = 5000
        mean_shape_all_tetrodes = []
        nr_tetrodes = self.eegh.shape[1]

        cells_per_tetrode = []
        # go through all tetrodes and compute mean swr
        for tetrode in range(0, nr_tetrodes):
            print(" - Loading data from tetrode " + str(tetrode+1))
            # check if tetrode data exists
            if not os.path.isfile(data_dir + "/" + self.session_name + "/" + self.session_name[1:] + ".clu." +
                                  str(tetrode+1)):
                print(" --> .clu file not found")
                continue

            # load cluster IDs (from .clu) and times of spikes (from .res)
            clu = read_integers(data_dir + "/" + self.session_name + "/" + self.session_name[1:] + ".clu." + str(tetrode+1))

            curr_tet_clust_ids = np.unique(clu)
            # only select cells that are of the appropriate type
            # curr_tet_clust_ids = curr_tet_clust_ids[np.isin(curr_tet_clust_ids, p1_cell_ids)]

            curr_tet_clust_ids  = curr_tet_clust_ids[curr_tet_clust_ids>1]

            if curr_tet_clust_ids.shape[0] > 0:
                cells_per_tetrode.append(np.ones(curr_tet_clust_ids.shape[0])*tetrode)

            data = self.eegh[:, tetrode]
            file_name = \
                self.session_name + "_" + self.experiment_phase_id + "_swr_" + self.cell_type + "_tet_" + str(
                    self.session_params.lfp_tetrodes) + ".npy"
            # check if results exist already
            if not os.path.isfile(self.params.pre_proc_dir+"swr_periods/" + file_name):

                # check if results exist already --> if not
                print(" - DETECTING SWR USING TETRODE(S) "+str(self.session_params.lfp_tetrodes) + " ...\n")

                # for low pass filtering of the signal before z-scoring (20-30Hz is good)
                low_pass_cut_off_freq = 30
                # minimum gap in seconds between events. If two events have
                # a gap < min_gap_between_events --> events are joint and become one event
                min_gap_between_events = 0.1

                # if data is too large --> need to chunk it up

                if data.shape[0] > 10000000:
                    start_times = np.zeros(0)
                    peak_times = np.zeros(0)
                    end_times = np.zeros(0)
                    size_chunk = 10000000
                    for nr_chunk in range(np.ceil(data.shape[0]/size_chunk).astype(int)):
                        chunk_data = data[nr_chunk*size_chunk:min(data.shape[0], (nr_chunk+1)*size_chunk)]

                        # compute offset in seconds for current chunk
                        offset_sec = nr_chunk * size_chunk * 1/freq

                        start_times_chunk, end_times_chunk, peak_times_chunk \
                            = self.detect_lfp_events(data=chunk_data, freq=freq, thr=thr, freq_lo_bound=freq_lo_bound,
                                                     freq_hi_bound=freq_hi_bound,
                                                     low_pass_cut_off_freq=low_pass_cut_off_freq,
                                                     min_gap_between_events=min_gap_between_events,
                                                     plot_for_control=plot_for_control)

                        # check if event was detected
                        if start_times_chunk is not None:
                            start_times = np.hstack((start_times, (start_times_chunk + offset_sec)))
                            end_times = np.hstack((end_times, (end_times_chunk + offset_sec)))
                            peak_times = np.hstack((peak_times, (peak_times_chunk + offset_sec)))

                else:
                    # times in seconds
                    start_times, end_times, peak_times = \
                        self.detect_lfp_events(data=data, freq=freq, thr=thr, freq_lo_bound=freq_lo_bound,
                                               freq_hi_bound=freq_hi_bound, low_pass_cut_off_freq=low_pass_cut_off_freq,
                                               min_gap_between_events=min_gap_between_events,
                                               plot_for_control=plot_for_control)

                result_dic = {
                    "start_times": start_times,
                    "end_times": end_times,
                    "peak_times": peak_times
                }

                outfile = open(self.params.pre_proc_dir+"swr_periods/"+file_name, 'wb')
                pickle.dump(result_dic, outfile)
                outfile.close()

            # load results from file
            infile = open(self.params.pre_proc_dir+"swr_periods/" + file_name, 'rb')
            result_dic = pickle.load(infile)
            infile.close()

            # times in seconds
            peak_times = result_dic["peak_times"]

            # need to convert to 5kHz to select LFP data (.eggh file)
            peak_times = peak_times * 5e3

            # plus minus from peak
            p_m_s = 0.1
            p_m = int(p_m_s * 5e3)

            swr_shapes = np.zeros((peak_times.shape[0], p_m*2))

            # go through all peak
            for i, peak in enumerate(peak_times):
                swr_dat = data[int(peak-p_m):int(peak+p_m)]
                if swr_dat.shape[0] == swr_shapes.shape[1]:
                    swr_shapes[i, :] = data[int(peak-p_m):int(peak+p_m)]
                else:
                    swr_shapes[i, :] = data[int(peak - p_m):int(peak + p_m)][:swr_shapes.shape[1]]

            mean_shape = np.mean(swr_shapes, axis=0)
            mean_shape_all_tetrodes.append(mean_shape)
            if plot_for_control:
                plt.plot(mean_shape)
                plt.show()

        cells_per_tetrode = np.hstack(cells_per_tetrode)

        # need to only consider p1 neurons
        p1_cell_ids = self.get_cell_id(data_dir=data_dir, session_name=self.session_name, cell_type=self.cell_type)
        p1_cell_ids = np.hstack(p1_cell_ids)

        # need to offset by 2 again, because noise & artefact clusters were removed
        p1_cells_per_tetrode = cells_per_tetrode[p1_cell_ids-2]

        return mean_shape_all_tetrodes, p1_cells_per_tetrode

    # </editor-fold>

    # <editor-fold desc="pHMM">

    # pHMM
    # ------------------------------------------------------------------------------------------------------------------

    def cross_val_poisson_hmm(self, cl_ar=np.arange(1, 40, 5), sel_range=None):
        """
        cross validation of poisson hmm fits to data

        :param cl_ar: #clusters to fit to data
        :type cl_ar: range
        :param sel_range: subset of data to be used
        :type sel_range: range
        """

        print(" - CROSS-VALIDATING POISSON HMM --> #modes ...\n")

        if sel_range is None:
            X = self.raster
        else:
            X = self.raster[:,sel_range]

        nr_cores = 12

        test_range_per_fold = None

        if self.params.cross_val_splits == "custom_splits":

            # number of folds
            nr_folds = 10
            # how many times to fit for each split
            max_nr_fitting = 5
            # how many chunks (for pre-computed splits)
            nr_chunks = 10

            unobserved_lo_array = pickle.load(
                open("temp_data/unobserved_lo_cv" + str(nr_folds) + "_" + str(nr_chunks) + "_chunks", "rb"))
            unobserved_hi_array = pickle.load(
                open("temp_data/unobserved_hi_cv" + str(nr_folds) + "_" + str(nr_chunks) + "_chunks", "rb"))

            # set number of time bins
            bin_num = X.shape[1]
            bins = np.arange(bin_num + 1)

            # length of one chunk
            n_chunks = int(bin_num / nr_chunks)
            test_range_per_fold = []
            for fold in range(nr_folds):

                # unobserved_lo: start bins (in spike data resolution) for all test data chunks
                unobserved_lo = []
                unobserved_hi = []
                for lo, hi in zip(unobserved_lo_array[fold], unobserved_hi_array[fold]):
                    unobserved_lo.append(bins[lo * n_chunks])
                    unobserved_hi.append(bins[hi * n_chunks])

                unobserved_lo = np.array(unobserved_lo)
                unobserved_hi = np.array(unobserved_hi)

                test_range = []
                for lo, hi in zip(unobserved_lo, unobserved_hi):
                    test_range += (list(range(lo, hi)))
                test_range_per_fold.append(np.array(test_range))


        folder_name = self.params.session_name +"_"+self.experiment_phase_id+"_"+self.cell_type

        new_ml = MlMethodsOnePopulation(params=self.params)
        new_ml.parallelize_cross_val_model(nr_cluster_array=cl_ar, nr_cores=nr_cores, model_type="POISSON_HMM",
                                           raster_data=X, folder_name=folder_name, splits=test_range_per_fold)
        new_ml.cross_val_view_results(folder_name=folder_name)

    def view_cross_val_results(self, range_to_plot=None):

        folder_name = self.params.session_name + "_" + str(
            int(self.params.experiment_phase_id[0])) + "_" + self.cell_type
        new_ml = MlMethodsOnePopulation(params=self.params)
        new_ml.cross_val_view_results(folder_name=folder_name, range_to_plot=range_to_plot)

    def fit_poisson_hmm(self, nr_modes):
        # --------------------------------------------------------------------------------------------------------------
        # fits poisson hmm to data
        #
        # args:     - nr_modes, int: #clusters to fit to data
        #           - file_identifier, string: string that is added at the end of file for identification
        # --------------------------------------------------------------------------------------------------------------

        print(" - FITTING POISSON HMM WITH "+str(nr_modes)+" MODES ...")

        X = self.raster

        model = PoissonHMM(n_components=nr_modes)
        model.fit(X.T)

        file_name = self.params.session_name + "_" + str(
            int(self.params.experiment_phase_id[0])) + "_" + self.cell_type +"_"+str(nr_modes)+"_modes"

        with open(self.params.pre_proc_dir+"phmm/"+file_name+".pkl", "wb") as file: pickle.dump(model, file)

    def decode_poisson_hmm(self, nr_modes=None, file_name=None, plotting=False):
        # --------------------------------------------------------------------------------------------------------------
        # loads model from file and decodes data
        #
        # args:     - nr_modes, int: #clusters to fit to data --> used to identify file that fits the data
        #           - file_name, string:    is used if model from a different experiment phase is supposed to be used
        #                                   (e.g. model from awake is supposed to be fit to sleep data)
        # --------------------------------------------------------------------------------------------------------------

        X = self.raster

        if (nr_modes is None) & (file_name is None):
            raise Exception("PROVIDE NR. MODES OR FILE NAME")

        if file_name is None:
            file_name = self.params.session_name + "_" + str(
                int(self.params.experiment_phase_id[0])) + "_" + self.cell_type +"_"+str(nr_modes)+"_modes"
        else:
            file_name =file_name

        with open(self.params.pre_proc_dir+"phmm/"+file_name+".pkl", "rb") as file:
            model = pickle.load(file)

        nr_modes_ = model.means_.shape[0]

        # compute most likely sequence
        sequence = model.predict(X.T)
        if plotting:
            plot_pop_clusters(map=X, labels=sequence, params=self.params,
                              nr_clusters=nr_modes, sel_range=range(100, 200))

        return sequence, nr_modes_

    def load_poisson_hmm(self, nr_modes=None, file_name=None):
        # --------------------------------------------------------------------------------------------------------------
        # loads model from file and returns model
        #
        # args:     - nr_modes, int: #clusters to fit to data --> used to identify file that fits the data
        #           - file_name, string:    is used if model from a different experiment phase is supposed to be used
        #                                   (e.g. model from awake is supposed to be fit to sleep data)
        # --------------------------------------------------------------------------------------------------------------

        if (nr_modes is None) & (file_name is None):
            raise Exception("PROVIDE NR. MODES OR FILE NAME")

        if file_name is None:
            file_name = self.params.session_name + "_" + str(
                int(self.params.experiment_phase_id[0])) + "_" + self.cell_type +"_"+str(nr_modes)+"_modes"
        else:
            file_name =file_name

        with open(self.params.pre_proc_dir+"phmm/"+file_name+".pkl", "rb") as file:
            model = pickle.load(file)

        return model

    def plot_poisson_hmm_info(self, nr_modes):
        model = self.load_poisson_hmm(nr_modes=nr_modes)
        plt.imshow(model.means_.T, interpolation='nearest', aspect='auto', cmap="jet")
        plt.ylabel("CELL ID")
        plt.xlabel("MODE ID")
        a = plt.colorbar()
        a.set_label(r'$\lambda$'+ " (#SPIKES/100ms WINDOW)")
        plt.title("STATE NEURAL PATTERNS")
        plt.show()

    def evaluate_poisson_hmm(self, nr_modes):
        # --------------------------------------------------------------------------------------------------------------
        # fits poisson hmm to data and evaluates the goodness of the model by comparing basic statistics (avg. firing
        # rate, correlation values, k-statistics) between real data and data sampled from the model
        #
        # args:     - nr_modes, int: #clusters to fit to data
        #           - load_from_file, bool: whether to load model from file or to fit model again
        # --------------------------------------------------------------------------------------------------------------

        print(" - EVALUATING POISSON HMM FIT (BASIC STATISTICS) ...")

        X = self.raster
        nr_time_bins = X.shape[1]
        # X = X[:, :1000]

        file_name = self.params.session_name + "_" + str(
            int(self.params.experiment_phase_id[0])) + "_" + self.cell_type +"_"+str(nr_modes)+"_modes"

        # check if model file exists already --> otherwise fit model again
        if os.path.isfile(self.params.pre_proc_dir+"phmm/" + file_name + ".pkl"):
            print("- LOADING PHMM MODEL FROM FILE\n")
            with open(self.params.pre_proc_dir+"phmm/" + file_name + ".pkl", "rb") as file:
                model = pickle.load(file)
        else:
            print("- PHMM MODEL FILE NOT FOUND --> FITTING PHMM TO DATA\n")
            model = PoissonHMM(n_components=nr_modes)
            model.fit(X.T)

        samples, sequence = model.sample(nr_time_bins)
        samples = samples.T

        evaluate_clustering_fit(real_data=X, samples=samples, binning="TEMPORAL_SPIKE",
                                   time_bin_size=0.1, plotting=True)

    def poisson_hmm_fit_discreteness(self, nr_modes=80, analysis_method="LIN_SEP"):
        # --------------------------------------------------------------------------------------------------------------
        # checks how discrete identified modes are
        #
        # args:     - nr_modes, int: defines which number of modes were fit to data (for file identification)
        #           - analysis_method, string:  - "CORR" --> correlations between lambda vectors
        #                                       - "LIN_SEP" --> check linear separability of population vectors that
        #                                                       were assigned to different modes
        #                                       - "SAMPLING" --> samples from lambda vector using Poisson emissions and
        #                                                        plots samples color coded using MDS
        # --------------------------------------------------------------------------------------------------------------

        print(" - ASSESSING DISCRETENESS OF MODES ...")

        # load data
        X = self.raster

        with open(self.params.pre_proc_dir+"ML/poisson_hmm_" +
                  str(nr_modes) + "_modes_" + self.cell_type + ".pkl", "rb") as file:
            model = pickle.load(file)

        seq = model.predict(X.T)

        lambda_per_mode = model.means_.T

        nr_modes = lambda_per_mode.shape[1]

        if analysis_method == "CORR":
            plt.imshow(lambda_per_mode, interpolation='nearest', aspect='auto', cmap="jet")
            a = plt.colorbar()
            a.set_label("LAMBDA (SPIKES PER TIME BIN)")
            plt.ylabel("CELL ID")
            plt.xlabel("MODE ID")
            plt.title("LAMBDA VECTOR PER MODE")
            plt.show()

            # control --> shuffle each lambda vector
            nr_shuffles = 10000
            correlation_shuffle = []
            for i in range(nr_shuffles):
                shuffled_data = independent_shuffle(lambda_per_mode)
                correlation_shuffle.append(upper_tri_without_diag(np.corrcoef(shuffled_data.T)))

            corr_shuffle = np.array(correlation_shuffle).flatten()
            corr_shuffle_95_perc = np.percentile(corr_shuffle, 95)

            plt.hist(corr_shuffle, density=True, label="SHUFFLE")

            corr_data = np.corrcoef(lambda_per_mode.T)
            # get off diagonal elements
            corr_data_vals = upper_tri_without_diag(corr_data)
            plt.hist(corr_data_vals, density=True, alpha=0.5, label="DATA")
            plt.xlabel("PEARSON CORRELATION")
            plt.ylabel("COUNTS")
            plt.title("CORR. BETWEEN LAMBDA VECTORS FOR EACH MODE")
            y_max = plt.ylim()
            plt.vlines(corr_shuffle_95_perc, 0, 5, colors="r", label="95 percentile shuffle")
            plt.legend()
            plt.show()

            plt.imshow(corr_data, interpolation="nearest", aspect="auto", cmap="jet")
            a = plt.colorbar()
            a.set_label("PEARSON CORRELATION")
            plt.title("CORRELATION BETWEEN LAMBDA VECTORS PER MODE")
            plt.xlabel("MODE ID")
            plt.ylabel("MODE ID")
            plt.show()

        elif analysis_method == "LIN_SEP":
            # find pop vectors with mode id and try to fit svm

            D = np.zeros((nr_modes, nr_modes))
            for template_mode_id in np.arange(nr_modes):
                others = np.delete(np.arange(nr_modes), template_mode_id)
                for compare_mode_id in others:
                    mode_id_1 = template_mode_id
                    mode_id_2 = compare_mode_id

                    mode_1 = X[:, seq == mode_id_1]
                    mode_2 = X[:, seq == mode_id_2]
                    mode_id_1_label = mode_id_1 * np.ones(mode_1.shape[1])
                    mode_id_2_label = mode_id_2 * np.ones(mode_2.shape[1])

                    # plt.imshow(mode_1, interpolation='nearest', aspect='auto', cmap="jet")
                    # plt.show()
                    # plt.imshow(mode_2, interpolation='nearest', aspect='auto', cmap="jet")
                    # plt.show()

                    nr_mode_0 = mode_1.shape[1]

                    data = np.hstack((mode_1, mode_2))
                    labels = np.hstack((mode_id_1_label, mode_id_2_label))

                    D[template_mode_id, compare_mode_id] = MlMethodsOnePopulation(params=self.params).linear_separability(input_data=data, input_labels=labels)

            D[D == 0] = np.nan
            plt.imshow(D, interpolation='nearest', aspect='auto', cmap="jet")
            a = plt.colorbar()
            a.set_label("ACCURACY (SVM)")
            plt.title("LINEAR SEPARABILTY OF POP. VEC PER MODE")
            plt.xlabel("MODE ID")
            plt.ylabel("MODE ID")
            plt.show()

            return D

        elif analysis_method == "SAMPLING":

            lambda_per_mode = model.means_.T
            mode_id = np.arange(0, lambda_per_mode.shape[1])
            poisson_firing = np.empty((lambda_per_mode.shape[0],0))
            nr_samples = 50
            # sample from modes
            for i in range(nr_samples):
                poisson_firing = np.hstack((poisson_firing, np.random.poisson(lambda_per_mode)))

            rep_mode_id = np.tile(mode_id, nr_samples)

            reduced_dim = MlMethodsOnePopulation(params=self.params).reduce_dimension(input_data=poisson_firing)

            plt.scatter(reduced_dim[:, 0], reduced_dim[:, 1], c="gray")
            plt.scatter(reduced_dim[rep_mode_id == 0, 0], reduced_dim[rep_mode_id == 0, 1], c="r", label="MODE 0")
            plt.scatter(reduced_dim[rep_mode_id == 18, 0], reduced_dim[rep_mode_id == 18, 1], c="g", label="MODE 18")
            plt.scatter(reduced_dim[rep_mode_id == 50, 0], reduced_dim[rep_mode_id == 50, 1], c="b", label="MODE 50")
            plt.legend()
            plt.title("MODE SAMPLE SIMILARITY")
            plt.show()

    def poisson_hmm_mode_progression(self, nr_modes, window_size=10):
        """
        checks how discrete identified modes are

        @param nr_modes: #modes of model (to find according file containing the model)
        @type nr_modes: int
        @param window_size: length of window size in seconds to compute frequency of modes
        @type window_size: float
        """
        print(" - ASSESSING PROGRESSION OF MODES ...")

        file_name = self.params.session_name + "_" + str(
            int(self.params.experiment_phase_id[0])) + "_" + self.cell_type +"_"+str(nr_modes)+"_modes"

        X = self.raster
        with open(self.params.pre_proc_dir+"phmm/" + file_name +".pkl", "rb") as file: model = pickle.load(file)
        seq = model.predict(X.T)
        nr_modes = model.means_.shape[0]

        print(X.shape)
        log_prob, post = model.score_samples(X.T)
        print(post.shape)
        plt.imshow(post.T, interpolation='nearest', aspect='auto')
        a = plt.colorbar()
        a.set_label("POST. PROBABILITY")
        plt.ylabel("MODE ID")
        plt.xlabel("TIME BINS")
        plt.title("MODE PROBABILITIES")
        plt.show()
        print(np.sum(post.T, axis=0))


        bins_per_window = int(window_size / self.params.time_bin_size)

        nr_windows = int(X.shape[1]/bins_per_window)

        mode_prob = np.zeros((nr_modes, nr_windows))
        for i in range(nr_windows):
            a,b = np.unique(seq[i*bins_per_window:(i+1)*bins_per_window], return_counts=True)
            mode_prob[a, i] = b/bins_per_window

        max_mode_prob = np.max(mode_prob, axis=1)
        mode_prob_norm = mode_prob / mode_prob.max(axis=1)[:, None]

        plt.imshow(mode_prob_norm, interpolation='nearest', aspect='auto')
        plt.title("MODE FREQUENCY - NORMALIZED - "+str(window_size)+"s WINDOW")
        plt.ylabel("MODE ID")
        plt.xlabel("WINDOW ID")
        a = plt.colorbar()
        a.set_label("MODE FREQUENCY - NORMALIZED")

        # compute weighted average
        windows = np.tile(np.arange(nr_windows).T, nr_modes).reshape((nr_modes, nr_windows))
        weighted_av = np.average(windows, axis=1, weights=mode_prob)

        a = (windows - weighted_av[:, None]) ** 2

        weighted_std = np.sqrt(np.average(a, axis=1, weights=mode_prob))

        plt.scatter(weighted_av, np.arange(nr_modes), c="r", s=1, label="WEIGHTED AVERAGE")
        plt.legend()
        plt.show()

        plt.scatter(weighted_av, weighted_std)
        plt.title("MODE OCCURENCE: WEIGHTED AVERAGE & WEIGHTED STD")
        plt.ylabel("WEIGHTED STD")
        plt.xlabel("WEIGHTED AVERAGE")
        plt.show()

    def poisson_hmm_transitions(self, nr_modes=80):
        # --------------------------------------------------------------------------------------------------------------
        # views and analyzes transition matrix
        #
        # args:   - nr_modes, int: defines which number of modes were fit to data (for file identification)
        #
        # --------------------------------------------------------------------------------------------------------------
        print(" - ASSESSING TRANSITIONS OF MODES ...")

        with open(self.params.pre_proc_dir+"ML/poisson_hmm_" +
                  str(nr_modes) + "_modes_" + self.cell_type + ".pkl", "rb") as file:
            model = pickle.load(file)

        trans_mat = model.transmat_

        plt.imshow(trans_mat, interpolation='nearest', aspect='auto')

        plt.ylabel("MODE ID")
        plt.xlabel("MODE ID")
        plt.title("TRANSITION PROBABILITY")
        a = plt.colorbar()
        a.set_label("TRANSITION PROBABILITY")
        plt.show()

        np.fill_diagonal(trans_mat, np.nan)

        plt.imshow(trans_mat, interpolation='nearest', aspect='auto')

        plt.ylabel("MODE ID")
        plt.xlabel("MODE ID")
        plt.title("TRANSITION PROBABILITY WO DIAGONAL")
        a = plt.colorbar()
        a.set_label("TRANSITION PROBABILITY")
        plt.show()

    def poisson_hmm_mode_drift(self, file_name):
        # --------------------------------------------------------------------------------------------------------------
        # checks assemblies assigned to different modes drift
        # --------------------------------------------------------------------------------------------------------------

        print(" - ASSESSING DRIFT OF ASSEMBLIES WITHIN MODES ...")

        X = self.raster
        with open(self.params.pre_proc_dir+"ML/" + file_name , "rb") as file: model = pickle.load(file)
        seq = model.predict(X.T)
        nr_modes = model.means_.shape[0]
        change = []

        for mode_id in range(nr_modes):
            X_mode = X[:, seq == mode_id]
            X_mode = (X_mode - np.min(X_mode, axis=1, keepdims=True)) / \
                (np.max(X_mode, axis=1, keepdims=True) - np.min(X_mode, axis=1, keepdims=True))
            X_mode = np.nan_to_num(X_mode)

            # plt.imshow(X_mode, interpolation='nearest', aspect='auto')
            # plt.colorbar()
            # plt.show()
            res = []
            for cell in X_mode:
                try:
                    m, b = np.polyfit(range(X_mode.shape[1]), cell, 1)
                    res.append(m)
                except:
                    res.append(0)

            res = np.array(res)
            mean_change_cell = np.mean(np.abs(res))
            change.append(mean_change_cell)

        change = np.array(change)
        a = np.flip(np.argsort(change))
        print(a)
        plt.plot(change)
        plt.show()

        X_mode = X[:, seq == 90]

        plt.imshow(X_mode, interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.show()

        exit()

        for mode_id in range(nr_modes):
            X_mode = X[:, seq==mode_id]



            exit()


            X_temp = X_mode[:, :int(X_mode.shape[0]*0.1)]
            X_compare = X_mode[:, int(X_mode.shape[0]*0.1):]

            res = np.zeros(X_compare.shape[1])
            for i, x in enumerate(X_compare.T):
                all_comp = []
                for temp in X_temp.T:
                    all_comp.append(1-distance.euclidean(x, temp))
                res[i] = np.max(np.array(all_comp))
            try:
                m, b = np.polyfit(range(X_compare.shape[1]), res, 1)
                change.append(m)
            except:
                continue


        plt.plot(change)
        plt.show()
        exit()




        plt.scatter(range(X_mode.shape[1]), X_mode[102,:])
        # plt.plot(X_mode[102, :])
        plt.show()

    # </editor-fold>


class Sleep(BaseMethods):
    """Base class for sleep analysis"""

    def __init__(self, data_dic, cell_type, params, session_params, experiment_phase=None):
        # --------------------------------------------------------------------------------------------------------------
        # args: - data_dic, dictionary with standard data
        #       - cell_type, string: cell type to do analysis with
        #       - params, python class: contains all parameters
        # --------------------------------------------------------------------------------------------------------------

        # get attributes from parent class
        BaseMethods.__init__(self, data_dic, cell_type, params, session_params, experiment_phase)

        # import analysis parameters that are specific for the current session
        # --------------------------------------------------------------------------------------------------------------
        # rem/nrem phases with speeds above this threshold are discarded
        self.params.sleep_phase_speed_threshold = self.session_params.sleep_phase_speed_threshold
        self.session_name = self.session_params.session_name

        # compression factor:
        #
        # compression factor used for sleep decoding --> e.g when we use constant #spike bins with 12 spikes
        # we need to check how many spikes we have in e.g. 100ms windows if this was used for awake encoding
        # if we have a mean of 30 spikes for awake --> compression factor = 12/30 --> 0.4
        # is used to scale awake activity to fit sleep activity
        # --------------------------------------------------------------------------------------------------------------

        if cell_type == "p1_l":
            self.session_params.sleep_compression_factor_12spikes_100ms = \
                self.session_params.sleep_compression_factor_12spikes_100ms_p1_l
        elif cell_type == "p1_r":
            self.session_params.sleep_compression_factor_12spikes_100ms = \
                self.session_params.sleep_compression_factor_12spikes_100ms_p1_r
        else:
            self.session_params.sleep_compression_factor_12spikes_100ms = \
                self.session_params.sleep_compression_factor_12spikes_100ms

        # default models for behavioral data
        # --------------------------------------------------------------------------------------------------------------

        if cell_type == "p1_l":
            self.session_params.default_pre_phmm_model = self.session_params.default_pre_phmm_model_p1_l
            self.session_params.default_post_phmm_model = self.session_params.default_post_phmm_model_p1_l
            self.session_params.default_pre_ising_model = self.session_params.default_pre_ising_model_p1_l
            self.session_params.default_post_ising_model = self.session_params.default_post_ising_model_p1_l
        elif cell_type == "p1_r":
            self.session_params.default_pre_phmm_model = self.session_params.default_pre_phmm_model_p1_r
            self.session_params.default_post_phmm_model = self.session_params.default_post_phmm_model_p1_r
            self.session_params.default_pre_ising_model = self.session_params.default_pre_ising_model_p1_r
            self.session_params.default_post_ising_model = self.session_params.default_post_ising_model_p1_r
        else:
            self.session_params.default_pre_phmm_model = self.session_params.default_pre_phmm_model
            self.session_params.default_post_phmm_model = self.session_params.default_post_phmm_model
            self.session_params.default_pre_ising_model = self.session_params.default_pre_ising_model
            self.session_params.default_post_ising_model = self.session_params.default_post_ising_model

        # get sleep type (e.g. nREM, SW)
        self.sleep_type = params.sleep_type

        # get data dictionary
        # check if list or dictionary is passed:
        if isinstance(data_dic, list):
            data_dic = data_dic[0]
        else:
            data_dic = data_dic
        # get time stamps for sleep type
        # time stamps for sleep (.nrem, .rem, .sw) are at 20 kHz --> just like .res data
        self.time_stamps = data_dic["timestamps"]

        # TODO: do pre processing!
        # should pre-process sleep to exclude moving periods --> is especially important for long sleep
        # PreProcessSleep(firing_times=self.firing_times, params=self.params, time_stamps=self.time_stamps,
        #                 whl=self.whl).speed_filter_raw_data(eegh=self.eegh)

        # initialize all as None, so that we do not have to go through the pre-processing everytime we generate
        # an instance of the class --> if this data is needed, call self.compute_raster_speed_loc
        # --------------------------------------------------------------------------------------------------------------
        self.raster = np.empty(0)
        self.speed = np.empty(0)
        self.loc = np.empty(0)

    # <editor-fold desc="Basic computations">

    # basic computations
    # ------------------------------------------------------------------------------------------------------------------
    def compute_raster_speed_loc(self, time_bin_size=None):
        """
        Computes raster, speed and location
        """
        # check first if data exists already
        if self.raster.shape[0] == 0:
            pre_prop_sleep = PreProcessSleep(firing_times=self.firing_times, params=self.params, time_stamps=None,
                                             last_spike=self.last_spike, whl=self.whl)
            raster = pre_prop_sleep.get_raster(time_bin_size=time_bin_size)
            speed = pre_prop_sleep.get_speed(time_bin_size=time_bin_size)
            loc = pre_prop_sleep.get_loc(time_bin_size=time_bin_size)

            # trim to same length
            common_length = np.min([raster.shape[1], speed.shape[0], loc.shape[0]])

            self.raster = raster[:, :common_length]
            self.speed = speed[:common_length]
            self.loc = loc[:, :common_length]

    def compute_speed(self, time_bin_size = None):
        """
        Computes speed
        """
        # check if speed exists already
        if self.speed.shape[0] == 0:

            pre_prop_sleep = PreProcessSleep(firing_times=self.firing_times, params=self.params, time_stamps=None,
                                             last_spike=self.last_spike, whl=self.whl)
            self.speed = pre_prop_sleep.get_speed(time_bin_size=time_bin_size)

        return self.speed
    # </editor-fold>

    # <editor-fold desc="Getter methods">

    # getter methods
    # ------------------------------------------------------------------------------------------------------------------

    def get_duration_sec(self):
        """
        Duration of sleep in seconds
        :return: dur of sleep
        :rtype: float
        """
        return self.whl.shape[0] * 0.0256

    def get_raster(self, speed_threshold=False, time_bin_size=None):
        """
        Computes and returns raster
        :param speed_threshold: speed threshold in cm/s (usually 5)
        :type speed_threshold: float
        :return: raster
        :rtype: numpy.array
        """
        self.compute_raster_speed_loc(time_bin_size=time_bin_size)
        if speed_threshold:
            speed_threshold = self.session_params.sleep_phase_speed_threshold
            raster = self.raster[:, self.speed < speed_threshold]
        else:
            raster = self.raster
        return raster

    def get_speed(self):
        """
        Computes and returns speed of the animal
        :return: speed
        :rtype: numpy.array
        """
        self.compute_raster_speed_loc()
        return self.speed

    def get_swr_frequency_nrem(self, window_size_min=2, speed_threshold=None, plot_for_control=False):

        start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

        # convert to one array for event_spike_binning
        event_times = np.vstack((start_times, end_times, peak_times)).T

        # only select SWR during nrem phases
        # get nrem phases in seconds
        n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                 classification_method="std")
        swr_in_n_rem = np.zeros(event_times.shape[0])
        swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
        for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
            n_rem_start = n_rem_phase[0]
            n_rem_end = n_rem_phase[1]
            for i, e_t in enumerate(event_times):
                event_start = e_t[0]
                event_end = e_t[1]
                if (n_rem_start < event_start) and (event_end < n_rem_end):
                    swr_in_n_rem[i] += 1
                    swr_in_which_n_rem[n_rem_phase_id, i] = 1

        swr_times_nrem = event_times[swr_in_n_rem == 1]
        swr_times_nrem_start = swr_times_nrem[:, 0]

        swr_per_window = []
        # go through each nrem phase and apply window
        for nrem_phase in n_rem_time_stamps:
            phase_dur = nrem_phase[1] - nrem_phase[0]
            window_size_sec = window_size_min * 60
            n_windows = np.round(phase_dur / window_size_sec).astype(int)
            for i_window in range(n_windows):
                # need to add start of nrem phase to every window
                swr_per_window.append(np.count_nonzero(np.logical_and((nrem_phase[0]+i_window*window_size_sec)<swr_times_nrem_start,
                                                                swr_times_nrem_start < (nrem_phase[0]+(i_window+1)*window_size_sec))))

        return np.hstack(swr_per_window)

    def get_duration_excluded_periods(self, speed_threshold=None,
                                      use_pre_computed_time_stamp=True):
        """
        return duration of excluded periods
        """

        if not "nslp" in self.time_stamps:
            use_pre_computed_time_stamp = False
            print("Computing awake periods because .nslp file was not in data dictionary.")

        if use_pre_computed_time_stamp:

            # speed_threshold = 1e3 --> don't use speed_threshold
            excluded_time_periods_above_threshold = self.get_sleep_phase(sleep_phase="nslp",
                                                                         speed_threshold=1e3,
                                                    classification_method="std")

            dur_excluded = np.sum(excluded_time_periods_above_threshold[:,1]-excluded_time_periods_above_threshold[:,0])

            # speed_threshold = 1e3 --> don't use speed_threshold
            # included_time_periods_above_threshold = self.get_sleep_phase(sleep_phase="slp",
            #                                                              speed_threshold=1e3,
            #                                         classification_method="std")
            #
            # dur_included = np.sum(included_time_periods_above_threshold[:,1]-included_time_periods_above_threshold[:,0])
            #
            # ratio = dur_excluded/(dur_excluded+dur_included)

        else:
            # get nrem phases in seconds
            event_times_nrem = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                     classification_method="std")

            print(" - "+str(event_times_nrem.shape[0])+" NREM phases found (speed thr.: "+str(speed_threshold)+")\n")

            duration_nrem = np.sum(event_times_nrem[:,1] - event_times_nrem[:,0])

            # get rem intervals in seconds
            event_times_rem = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold,
                                                   classification_method="std")

            print(" - "+str(event_times_rem.shape[0])+" REM phases found (speed thr.: "+str(speed_threshold)+")\n")

            duration_rem = np.sum(event_times_rem[:,1] - event_times_rem[:,0])

            nrem_rem_duration = duration_rem + duration_nrem

            duration = self.get_duration_sec()

            dur_excluded = duration - nrem_rem_duration

        return dur_excluded

    def get_duration_excluded_periods_s(self, speed_threshold=None, window_size_min=2,
                                        smoothing_window_s=5, plot_for_control=False):
        """
        return duration of excluded periods
        """

        if speed_threshold is None:
            speed_threshold = self.params.sleep_phase_speed_threshold

        speed = self.compute_speed(time_bin_size=1)
        # apply smoothing --> compute median value and set whole smoothing window to the median
        nr_smoothing_windows = int(np.round(self.speed.shape[0]/smoothing_window_s))

        speed_smooth = np.zeros(speed.shape[0])
        for s_w in range(nr_smoothing_windows):
            speed_smooth[s_w*smoothing_window_s:(s_w+1)*smoothing_window_s].fill(np.median(speed[s_w*smoothing_window_s:(s_w+1)*smoothing_window_s]))

        if plot_for_control:
            plt.plot(speed, color="grey")
            plt.plot(speed_smooth, color="red", label="smoothed")
            plt.legend()
            plt.show()

        window_size_sec = window_size_min * 60
        nr_windows = int(np.round(self.speed.shape[0]/window_size_sec))

        movement_in_window = np.zeros(nr_windows)
        # go trough all windows and check if speed was above threshold
        for i_window in range(nr_windows):
            if any(speed_smooth[i_window*window_size_sec:(i_window+1)*window_size_sec] >
                   speed_threshold):
                movement_in_window[i_window] = 1

        return np.count_nonzero(movement_in_window)*window_size_sec

    def get_correlation_matrices(self, bins_per_corr_matrix, cell_selection=None, only_upper_triangle=False):
        """
        Computes and returns correlation between cell firing

        :param bins_per_corr_matrix: how many temporal bins to use to compute correlations between cells
        :type bins_per_corr_matrix: int
        :param cell_selection: if a subset of cells is supposed to be used
        :type cell_selection: np.array
        :param only_upper_triangle: only return upper triangle (corr. matrix is symmetric and diagonal is one)
        :type only_upper_triangle: bool
        :return: correlation matrices
        :rtype: np.array
        """
        if cell_selection is None:
            file_name = self.params.session_name +"_"+self.experiment_phase+"_"+str(bins_per_corr_matrix)+\
                        "_bin_p_m_"+"excl_diag_"+\
                        str(only_upper_triangle)+ "_all"
        else:
            file_name = self.params.session_name +"_"+self.params.experiment_phase+"_"+str(bins_per_corr_matrix)+"_bin_p_m_"+"excl_diag_"+\
                        str(only_upper_triangle)+ "_stable"

        # check first if correlation matrices have been computed and saved
        if not os.path.isfile(self.params.pre_proc_dir + "correlation_matrices/" + file_name+".npy"):
            self.compute_raster_speed_loc()
            raster = self.get_raster(speed_threshold=True)
            if cell_selection is not None:
                raster = raster[cell_selection, :]
            else:
                raster = raster
            corr_matrices = correlations_from_raster(raster=raster, bins_per_corr_matrix=bins_per_corr_matrix,
                                        only_upper_triangle=only_upper_triangle)
            np.save(self.params.pre_proc_dir + "correlation_matrices/" + file_name, corr_matrices)

        else:
            print("  - LOADING EXISTING CORRELATION MATRICES")
            # load data
            corr_matrices = np.load(self.params.pre_proc_dir + "correlation_matrices/" + file_name+".npy")

        return corr_matrices

    def get_spike_binned_raster(self, nr_spikes_per_bin=None, return_estimated_times=False):
        """
        Computes and returns spike binned raster (constant number of spikes per bin)

        :param nr_spikes_per_bin: how many spikes per bin to use (usually 12 or so)
        :type nr_spikes_per_bin: int
        :param return_estimated_times: return estimated time per constant spike bin
        :type return_estimated_times: bool
        :return: raster (and estimated time if set to True)
        :rtype: np.array
        """
        if nr_spikes_per_bin is None:
            nr_spikes_per_bin = self.params.spikes_per_bin

        return PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                whl=self.whl, last_spike=self.last_spike).spike_binning(
            spikes_per_bin=nr_spikes_per_bin, return_estimated_times=return_estimated_times)

    def get_spike_binned_raster_combined_sleep_phases_jittered(self, spikes_per_bin=None,
                                                         plot_for_control=False,
                                                    speed_threshold=None, nr_spikes_per_jitter_window=200):
        """
        Computes and returns spike binned raster (constant number of spikes per bin)

        :param nr_spikes_per_bin: how many spikes per bin to use (usually 12 or so)
        :type nr_spikes_per_bin: int
        :param return_estimated_times: return estimated time per constant spike bin
        :type return_estimated_times: bool
        :return: raster (and estimated time if set to True)
        :rtype: np.array
        """

        # get SWR timings (in sec) & compute spike rasters (constant #spikes)
        # ------------------------------------------------------------------------------------------------------
        start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

        # convert to one array for event_spike_binning
        event_times = np.vstack((start_times, end_times, peak_times)).T

        # only select SWR during nrem phases
        # get nrem phases in seconds
        n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                 classification_method="std")
        swr_in_n_rem = np.zeros(event_times.shape[0])
        swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
        for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
            n_rem_start = n_rem_phase[0]
            n_rem_end = n_rem_phase[1]
            for i, e_t in enumerate(event_times):
                event_start = e_t[0]
                event_end = e_t[1]
                if (n_rem_start < event_start) and (event_end < n_rem_end):
                    swr_in_n_rem[i] += 1
                    swr_in_which_n_rem[n_rem_phase_id, i] = 1

        event_times_nrem = event_times[swr_in_n_rem == 1]
        print(" - "+str(event_times_nrem.shape[0])+" WERE ASSIGNED TO NREM PHASES\n")

        # get rem intervals in seconds
        event_times_rem = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold,
                                           classification_method="std")

        print(" - "+str(event_times_rem.shape[0])+" REM phases found (speed thr.: "+str(speed_threshold)+")\n")

        # combine REM and NREM event times (order them first)
        start_times = np.hstack((event_times_rem[:,0], event_times_nrem[:,0]))
        end_times = np.hstack((event_times_rem[:,1], event_times_nrem[:,2]))
        labels = np.zeros(event_times_rem.shape[0]+event_times_nrem.shape[0])
        labels[:event_times_rem.shape[0]] = 1
        order = np.argsort(start_times)

        all_event_times_ordered = np.vstack((start_times[order], end_times[order])).T
        print("Duration of data: " +str(np.sum(all_event_times_ordered[:,1]-all_event_times_ordered[:,0]))+"s")

        labels_ordered = labels[order]

        # for debugging:
        # all_event_times_ordered = all_event_times_ordered[:50,:]
        # labels_ordered = labels_ordered[:50]

        # compute #spike binning for each event
        # --------------------------------------------------------------------------------------------------------------
        event_spike_rasters = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning_fast_jittered(event_times=all_event_times_ordered,
                                                                                event_time_freq=1,
                                                                       spikes_per_bin=spikes_per_bin,
                                                                            nr_spikes_per_jitter_window=nr_spikes_per_jitter_window)

        return event_times, event_spike_rasters

    def get_spike_binned_raster_combined_sleep_phases_equalized(self, spikes_per_bin=None, plot_for_control=False,
                                                                speed_threshold=None, nr_chunks=4):
        """
        Computes and returns spike binned raster (constant number of spikes per bin)

        :param nr_spikes_per_bin: how many spikes per bin to use (usually 12 or so)
        :type nr_spikes_per_bin: int
        :param return_estimated_times: return estimated time per constant spike bin
        :type return_estimated_times: bool
        :return: raster (and estimated time if set to True)
        :rtype: np.array
        """

        # get SWR timings (in sec) & compute spike rasters (constant #spikes)
        # ------------------------------------------------------------------------------------------------------
        start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

        # convert to one array for event_spike_binning
        event_times = np.vstack((start_times, end_times, peak_times)).T

        # only select SWR during nrem phases
        # get nrem phases in seconds
        n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                 classification_method="std")
        swr_in_n_rem = np.zeros(event_times.shape[0])
        swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
        for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
            n_rem_start = n_rem_phase[0]
            n_rem_end = n_rem_phase[1]
            for i, e_t in enumerate(event_times):
                event_start = e_t[0]
                event_end = e_t[1]
                if (n_rem_start < event_start) and (event_end < n_rem_end):
                    swr_in_n_rem[i] += 1
                    swr_in_which_n_rem[n_rem_phase_id, i] = 1

        event_times_nrem = event_times[swr_in_n_rem == 1]
        print(" - "+str(event_times_nrem.shape[0])+" WERE ASSIGNED TO NREM PHASES\n")

        # get rem intervals in seconds
        event_times_rem = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold,
                                           classification_method="std")

        print(" - "+str(event_times_rem.shape[0])+" REM phases found (speed thr.: "+str(speed_threshold)+")\n")

        # combine REM and NREM event times (order them first)
        start_times = np.hstack((event_times_rem[:,0], event_times_nrem[:,0]))
        end_times = np.hstack((event_times_rem[:,1], event_times_nrem[:,2]))
        labels = np.zeros(event_times_rem.shape[0]+event_times_nrem.shape[0])
        labels[:event_times_rem.shape[0]] = 1
        order = np.argsort(start_times)

        all_event_times_ordered = np.vstack((start_times[order], end_times[order])).T
        labels_ordered = labels[order]

        # for debugging:
        # all_event_times_ordered = all_event_times_ordered[:50,:]
        # labels_ordered = labels_ordered[:50]

        # compute #spike binning for each event
        # --------------------------------------------------------------------------------------------------------------
        event_spike_rasters = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning_fast_equalized(event_times=all_event_times_ordered,
                                                                                event_time_freq=1,
                                                                       spikes_per_bin=spikes_per_bin, nr_chunks=nr_chunks)

        return event_times, event_spike_rasters

    def get_spike_binned_raster_equalized_between_two_epochs(self, event_times_epoch_1, event_times_epoch_2,
                                                             spikes_per_bin=None, plot_for_control=False,
                                                                    speed_threshold=None):
        """
        Computes and returns spike binned raster (constant number of spikes per bin)

        :param nr_spikes_per_bin: how many spikes per bin to use (usually 12 or so)
        :type nr_spikes_per_bin: int
        :param return_estimated_times: return estimated time per constant spike bin
        :type return_estimated_times: bool
        :return: raster (and estimated time if set to True)
        :rtype: np.array
        """

        # compute duration
        dur_epoch_1 = np.sum(event_times_epoch_1[1,:]-event_times_epoch_1[0,:])
        dur_epoch_2 = np.sum(event_times_epoch_2[1,:]-event_times_epoch_2[0,:])

        temporal_factor_epoch_1_epoch_2 = dur_epoch_2/dur_epoch_1
        # compute #spike binning for each event
        # --------------------------------------------------------------------------------------------------------------
        event_spike_rasters_epoch_1, event_spike_rasters_epoch_2 = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning_fast_equalized_two_epochs(event_times_epoch_1=event_times_epoch_1.T,
                                                                                        event_times_epoch_2=event_times_epoch_2.T,
                                                                                        event_time_freq=1, temporal_factor=temporal_factor_epoch_1_epoch_2,
                                                                                        spikes_per_bin=spikes_per_bin)

        return event_spike_rasters_epoch_1, event_spike_rasters_epoch_2

    def get_spike_binned_raster_combined_sleep_phases_jitter_subset(self, cell_ids, spikes_per_bin=None,
                                                               plot_for_control=False,
                                                               speed_threshold=None,
                                                                    time_interval_jitter_s=20):
        """
        Computes and returns spike binned raster (constant number of spikes per bin)

        :param nr_spikes_per_bin: how many spikes per bin to use (usually 12 or so)
        :type nr_spikes_per_bin: int
        :param return_estimated_times: return estimated time per constant spike bin
        :type return_estimated_times: bool
        :return: raster (and estimated time if set to True)
        :rtype: np.array
        """

        # get SWR timings (in sec) & compute spike rasters (constant #spikes)
        # ------------------------------------------------------------------------------------------------------
        start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

        # convert to one array for event_spike_binning
        event_times = np.vstack((start_times, end_times, peak_times)).T

        # only select SWR during nrem phases
        # get nrem phases in seconds
        n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                 classification_method="std")
        swr_in_n_rem = np.zeros(event_times.shape[0])
        swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
        for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
            n_rem_start = n_rem_phase[0]
            n_rem_end = n_rem_phase[1]
            for i, e_t in enumerate(event_times):
                event_start = e_t[0]
                event_end = e_t[1]
                if (n_rem_start < event_start) and (event_end < n_rem_end):
                    swr_in_n_rem[i] += 1
                    swr_in_which_n_rem[n_rem_phase_id, i] = 1

        event_times_nrem = event_times[swr_in_n_rem == 1]
        print(" - "+str(event_times_nrem.shape[0])+" WERE ASSIGNED TO NREM PHASES\n")

        # get rem intervals in seconds
        event_times_rem = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold,
                                               classification_method="std")

        print(" - "+str(event_times_rem.shape[0])+" REM phases found (speed thr.: "+str(speed_threshold)+")\n")

        # combine REM and NREM event times (order them first)
        start_times = np.hstack((event_times_rem[:,0], event_times_nrem[:,0]))
        end_times = np.hstack((event_times_rem[:,1], event_times_nrem[:,2]))
        labels = np.zeros(event_times_rem.shape[0]+event_times_nrem.shape[0])
        labels[:event_times_rem.shape[0]] = 1
        order = np.argsort(start_times)

        all_event_times_ordered = np.vstack((start_times[order], end_times[order])).T
        labels_ordered = labels[order]

        # for debugging:
        # all_event_times_ordered = all_event_times_ordered[:50,:]
        # labels_ordered = labels_ordered[:50]

        # compute #spike binning for each event
        # --------------------------------------------------------------------------------------------------------------
        event_spike_rasters = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning_fast_jitter_subset(event_times=all_event_times_ordered,
                                                                            event_time_freq=1,
                                                                            spikes_per_bin=spikes_per_bin,
                                                                                 time_interval_jitter_s=time_interval_jitter_s,
                                                                                 cell_ids=cell_ids)
        # delete event with empty bins
        events_to_delete_bin = np.array([x.shape[1] == 0 for x in event_spike_rasters])
        events_to_delete = np.argwhere(events_to_delete_bin).flatten()
        # start from the last entry and delete events from list
        for ev in np.flip(events_to_delete):
            del event_spike_rasters[ev]

        all_event_times_ordered = all_event_times_ordered[~events_to_delete_bin, :]
        return event_times, event_spike_rasters
    def get_spike_binned_raster_combined_sleep_phases(self, spikes_per_bin=None,
                                                    plot_for_control=False,
                                                    speed_threshold=None):
        """
        Computes and returns spike binned raster (constant number of spikes per bin)

        :param nr_spikes_per_bin: how many spikes per bin to use (usually 12 or so)
        :type nr_spikes_per_bin: int
        :param return_estimated_times: return estimated time per constant spike bin
        :type return_estimated_times: bool
        :return: raster (and estimated time if set to True)
        :rtype: np.array
        """

        # get SWR timings (in sec) & compute spike rasters (constant #spikes)
        # ------------------------------------------------------------------------------------------------------
        start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

        # convert to one array for event_spike_binning
        event_times = np.vstack((start_times, end_times, peak_times)).T

        # only select SWR during nrem phases
        # get nrem phases in seconds
        n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                 classification_method="std")
        swr_in_n_rem = np.zeros(event_times.shape[0])
        swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
        for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
            n_rem_start = n_rem_phase[0]
            n_rem_end = n_rem_phase[1]
            for i, e_t in enumerate(event_times):
                event_start = e_t[0]
                event_end = e_t[1]
                if (n_rem_start < event_start) and (event_end < n_rem_end):
                    swr_in_n_rem[i] += 1
                    swr_in_which_n_rem[n_rem_phase_id, i] = 1

        event_times_nrem = event_times[swr_in_n_rem == 1]
        print(" - "+str(event_times_nrem.shape[0])+" WERE ASSIGNED TO NREM PHASES\n")

        # get rem intervals in seconds
        event_times_rem = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold,
                                               classification_method="std")

        print(" - "+str(event_times_rem.shape[0])+" REM phases found (speed thr.: "+str(speed_threshold)+")\n")

        # combine REM and NREM event times (order them first)
        start_times = np.hstack((event_times_rem[:,0], event_times_nrem[:,0]))
        end_times = np.hstack((event_times_rem[:,1], event_times_nrem[:,2]))
        labels = np.zeros(event_times_rem.shape[0]+event_times_nrem.shape[0])
        labels[:event_times_rem.shape[0]] = 1
        order = np.argsort(start_times)

        all_event_times_ordered = np.vstack((start_times[order], end_times[order])).T
        labels_ordered = labels[order]

        # for debugging:
        # all_event_times_ordered = all_event_times_ordered[:50,:]
        # labels_ordered = labels_ordered[:50]

        # compute #spike binning for each event
        # --------------------------------------------------------------------------------------------------------------
        event_spike_rasters = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning_fast(event_times=all_event_times_ordered,
                                                                                event_time_freq=1,
                                                                                spikes_per_bin=spikes_per_bin)
        # delete event with empty bins
        events_to_delete_bin = np.array([x.shape[1] == 0 for x in event_spike_rasters])
        events_to_delete = np.argwhere(events_to_delete_bin).flatten()
        # start from the last entry and delete events from list
        for ev in np.flip(events_to_delete):
            del event_spike_rasters[ev]

        all_event_times_ordered = all_event_times_ordered[~events_to_delete_bin, :]

        return all_event_times_ordered, event_spike_rasters

    def get_spike_binned_raster_slow_combined_sleep_phases(self, spikes_per_bin=None,
                                                      plot_for_control=False,
                                                      speed_threshold=None):
        """
        Computes and returns spike binned raster (constant number of spikes per bin)

        :param nr_spikes_per_bin: how many spikes per bin to use (usually 12 or so)
        :type nr_spikes_per_bin: int
        :param return_estimated_times: return estimated time per constant spike bin
        :type return_estimated_times: bool
        :return: raster (and estimated time if set to True)
        :rtype: np.array
        """

        # get SWR timings (in sec) & compute spike rasters (constant #spikes)
        # ------------------------------------------------------------------------------------------------------
        start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

        # convert to one array for event_spike_binning
        event_times = np.vstack((start_times, end_times, peak_times)).T

        # only select SWR during nrem phases
        # get nrem phases in seconds
        n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                 classification_method="std")
        swr_in_n_rem = np.zeros(event_times.shape[0])
        swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
        for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
            n_rem_start = n_rem_phase[0]
            n_rem_end = n_rem_phase[1]
            for i, e_t in enumerate(event_times):
                event_start = e_t[0]
                event_end = e_t[1]
                if (n_rem_start < event_start) and (event_end < n_rem_end):
                    swr_in_n_rem[i] += 1
                    swr_in_which_n_rem[n_rem_phase_id, i] = 1

        event_times_nrem = event_times[swr_in_n_rem == 1]
        print(" - "+str(event_times_nrem.shape[0])+" WERE ASSIGNED TO NREM PHASES\n")

        # get rem intervals in seconds
        event_times_rem = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold,
                                               classification_method="std")

        print(" - "+str(event_times_rem.shape[0])+" REM phases found (speed thr.: "+str(speed_threshold)+")\n")

        # combine REM and NREM event times (order them first)
        start_times = np.hstack((event_times_rem[:,0], event_times_nrem[:,0]))
        end_times = np.hstack((event_times_rem[:,1], event_times_nrem[:,2]))
        labels = np.zeros(event_times_rem.shape[0]+event_times_nrem.shape[0])
        labels[:event_times_rem.shape[0]] = 1
        order = np.argsort(start_times)

        all_event_times_ordered = np.vstack((start_times[order], end_times[order])).T
        labels_ordered = labels[order]

        # for debugging:
        # all_event_times_ordered = all_event_times_ordered[:50,:]
        # labels_ordered = labels_ordered[:50]

        # compute #spike binning for each event
        # --------------------------------------------------------------------------------------------------------------
        event_spike_rasters, _ = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning(event_times=all_event_times_ordered,
                                                                   event_time_freq=1,
                                                                   spikes_per_bin=spikes_per_bin)
        # delete event with empty bins
        events_to_delete_bin = np.array([x.shape[1] == 0 for x in event_spike_rasters])
        events_to_delete = np.argwhere(events_to_delete_bin).flatten()
        # start from the last entry and delete events from list
        for ev in np.flip(events_to_delete):
            del event_spike_rasters[ev]

        all_event_times_ordered = all_event_times_ordered[~events_to_delete_bin, :]

        return all_event_times_ordered, event_spike_rasters

    def get_spike_binned_raster_sleep_phase(self, spikes_per_bin=None, sleep_phase="nrem",
                                                         plot_for_control=False, speed_threshold=None):
        """
        Computes and returns spike binned raster (constant number of spikes per bin)

        :param nr_spikes_per_bin: how many spikes per bin to use (usually 12 or so)
        :type nr_spikes_per_bin: int
        :param return_estimated_times: return estimated time per constant spike bin
        :type return_estimated_times: bool
        :return: raster (and estimated time if set to True)
        :rtype: np.array
        """

        # get sleep phase intervals
        # --------------------------------------------------------------------------------------------------------------
        if sleep_phase == "nrem":

            # get SWR timings (in sec) & compute spike rasters (constant #spikes)
            # ------------------------------------------------------------------------------------------------------
            start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

            # convert to one array for event_spike_binning
            event_times = np.vstack((start_times, end_times, peak_times)).T

            # only select SWR during nrem phases
            # get nrem phases in seconds
            n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                     classification_method="std")
            swr_in_n_rem = np.zeros(event_times.shape[0])
            swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
            for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
                n_rem_start = n_rem_phase[0]
                n_rem_end = n_rem_phase[1]
                for i, e_t in enumerate(event_times):
                    event_start = e_t[0]
                    event_end = e_t[1]
                    if (n_rem_start < event_start) and (event_end < n_rem_end):
                        swr_in_n_rem[i] += 1
                        swr_in_which_n_rem[n_rem_phase_id, i] = 1

            if np.count_nonzero((swr_in_n_rem)) > 0:
                event_times = event_times[swr_in_n_rem == 1]
            else:
                print("No SWR in NREM --> using all SWRs")

            print(" - "+str(event_times.shape[0])+" WERE ASSIGNED TO NREM PHASES\n")

        elif sleep_phase == "rem":
            # get rem intervals in seconds
            event_times = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold,
                                               classification_method="std")

            print(" - "+str(event_times.shape[0])+" REM phases found (speed thr.: "+str(speed_threshold)+")\n")

        # compute #spike binning for each event
        # --------------------------------------------------------------------------------------------------------------
        # event_spike_rasters, event_spike_window_lenghts = \
        #     PreProcessSleep(firing_times=self.firing_times, params=self.params,
        #                     whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1,
        #                                                                spikes_per_bin=spikes_per_bin)

        event_spike_rasters = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning_fast(event_times=event_times, event_time_freq=1,
                                                                       spikes_per_bin=spikes_per_bin)

        # plt.subplot(2,1,1)
        # plt.imshow(event_spike_rasters[0])
        # plt.subplot(2,1,2)
        # plt.imshow(event_spike_rasters_new[0])
        # plt.show()
        # diff = np.sum(event_spike_rasters[0] - event_spike_rasters_new[0], axis=0)
        #


        # delete event with empty bins
        events_to_delete_bin = np.array([x.shape[1] == 0 for x in event_spike_rasters])
        events_to_delete = np.argwhere(events_to_delete_bin).flatten()
        # start from the last entry and delete events from list
        for ev in np.flip(events_to_delete):
            del event_spike_rasters[ev]

        event_times = event_times[~events_to_delete_bin, :]

        return event_times, event_spike_rasters

    def get_spike_binned_raster_around_swr(self, spikes_per_bin=None, return_bin_duration=False,
                                                plot_for_control=False, speed_threshold=None):
        """
        Computes and returns spike binned raster (constant number of spikes per bin)

        :param nr_spikes_per_bin: how many spikes per bin to use (usually 12 or so)
        :type nr_spikes_per_bin: int
        :param return_estimated_times: return estimated time per constant spike bin
        :type return_estimated_times: bool
        :return: raster (and estimated time if set to True)
        :rtype: np.array
        """

        # get SWR timings (in sec) & compute spike rasters (constant #spikes)
        # ------------------------------------------------------------------------------------------------------
        start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

        # convert to one array for event_spike_binning
        event_times_orig = np.vstack((start_times, end_times)).T

        # only select SWR during nrem phases
        # get nrem phases in seconds
        n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                 classification_method="std")
        swr_in_n_rem = np.zeros(event_times_orig.shape[0])
        swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times_orig.shape[0]))
        for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
            n_rem_start = n_rem_phase[0]
            n_rem_end = n_rem_phase[1]
            for i, e_t in enumerate(event_times_orig):
                event_start = e_t[0]
                event_end = e_t[1]
                if (n_rem_start < event_start) and (event_end < n_rem_end):
                    swr_in_n_rem[i] += 1
                    swr_in_which_n_rem[n_rem_phase_id, i] = 1

        event_times_orig = event_times_orig[swr_in_n_rem == 1]

        print(" - "+str(event_times_orig.shape[0])+" WERE ASSIGNED TO NREM PHASES\n")

        # take 1 second before and 1 second after

        new_before = event_times_orig[:, 0] - 1
        new_after = event_times_orig[:, 1] + 1

        nrem_time_stamps = np.hstack((np.vstack((new_before, event_times_orig[:, 0])),
                                  np.vstack((event_times_orig[:, 1], new_after)))).T
        if plot_for_control:
            for et_o in event_times_orig:
                plt.hlines(1, et_o[0], et_o[1], color="red")
            plt.ylim(0,1.1)
            plt.xlim(2835, 2840)
            plt.show()

        # need to delete periods that overlap with SWRs
        new_nrem_time_stamps = []

        # go through all nrem time stamps
        for nrem_ts in nrem_time_stamps:
            swr_start_within = start_times[np.logical_and(nrem_ts[0]<start_times, start_times<nrem_ts[1])]
            swr_end_within = end_times[np.logical_and(nrem_ts[0]<start_times, start_times<nrem_ts[1])]
            # new intervals run from end of swr to start of next swr --> first interval: start of nrem until first swr
            # --> last iterval: end of last swr until end of nrem
            swr_end_within = np.insert(swr_end_within, 0, nrem_ts[0])
            swr_start_within = np.append(swr_start_within, nrem_ts[-1])
            new_nrem_time_stamps.append(np.vstack((swr_end_within, swr_start_within)))

        event_times = np.hstack(new_nrem_time_stamps).T

        if plot_for_control:
            for et_o in event_times_orig:
                plt.hlines(1, et_o[0], et_o[1], color="red")
            for et in event_times:
                plt.hlines(0.9, et[0], et[1], color="blue")
            plt.ylim(0,1.1)
            plt.xlim(2835, 2840)
            plt.show()

        diff = event_times[:, 1]-event_times[:, 0]

        event_times = event_times[diff > 0, :]

        if return_bin_duration:
            event_spike_rasters, bin_dur = \
                PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1,
                                                                       spikes_per_bin=spikes_per_bin)
        else:
            event_spike_rasters = \
                PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                whl=self.whl).event_spike_binning_fast(event_times=event_times, event_time_freq=1,
                                                                       spikes_per_bin=spikes_per_bin)

        # delete event with empty bins
        events_to_delete_bin = np.array([x.shape[1] == 0 for x in event_spike_rasters])
        events_to_delete = np.argwhere(events_to_delete_bin).flatten()
        # start from the last entry and delete events from list
        for ev in np.flip(events_to_delete):
            del event_spike_rasters[ev]
            if return_bin_duration:
                del bin_dur[ev]

        event_times = event_times[~events_to_delete_bin, :]

        if plot_for_control:
            a = np.hstack(bin_dur)
            plt.hist(a,bins=100, density=True)
            plt.xlim(0, 0.5)
            plt.xlabel("Duration (s)")
            plt.title("Duration of 12spike bin")
            plt.ylabel("Density")
            plt.tight_layout()
            plt.show()

        if return_bin_duration:
            return event_times, event_spike_rasters, bin_dur
        else:
            return event_times, event_spike_rasters

    def get_spike_binned_raster_around_swr_jittered(self, spikes_per_bin=None, nr_spikes_per_jitter_window=200,
                                                plot_for_control=False, speed_threshold=None):
        """
        Computes and returns spike binned raster (constant number of spikes per bin)

        :param nr_spikes_per_bin: how many spikes per bin to use (usually 12 or so)
        :type nr_spikes_per_bin: int
        :param return_estimated_times: return estimated time per constant spike bin
        :type return_estimated_times: bool
        :return: raster (and estimated time if set to True)
        :rtype: np.array
        """

        # get SWR timings (in sec) & compute spike rasters (constant #spikes)
        # ------------------------------------------------------------------------------------------------------
        start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

        # convert to one array for event_spike_binning
        event_times_orig = np.vstack((start_times, end_times)).T

        # only select SWR during nrem phases
        # get nrem phases in seconds
        n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                 classification_method="std")
        swr_in_n_rem = np.zeros(event_times_orig.shape[0])
        swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times_orig.shape[0]))
        for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
            n_rem_start = n_rem_phase[0]
            n_rem_end = n_rem_phase[1]
            for i, e_t in enumerate(event_times_orig):
                event_start = e_t[0]
                event_end = e_t[1]
                if (n_rem_start < event_start) and (event_end < n_rem_end):
                    swr_in_n_rem[i] += 1
                    swr_in_which_n_rem[n_rem_phase_id, i] = 1

        event_times_orig = event_times_orig[swr_in_n_rem == 1]

        print(" - "+str(event_times_orig.shape[0])+" WERE ASSIGNED TO NREM PHASES\n")

        # take 1 second before and 1 second after

        new_before = event_times_orig[:, 0] - 1
        new_after = event_times_orig[:, 1] + 1

        nrem_time_stamps = np.hstack((np.vstack((new_before, event_times_orig[:, 0])),
                                  np.vstack((event_times_orig[:, 1], new_after)))).T
        if plot_for_control:
            for et_o in event_times_orig:
                plt.hlines(1, et_o[0], et_o[1], color="red")
            plt.ylim(0,1.1)
            plt.xlim(2835, 2840)
            plt.show()

        # need to delete periods that overlap with SWRs
        new_nrem_time_stamps = []

        # go through all nrem time stamps
        for nrem_ts in nrem_time_stamps:
            swr_start_within = start_times[np.logical_and(nrem_ts[0]<start_times, start_times<nrem_ts[1])]
            swr_end_within = end_times[np.logical_and(nrem_ts[0]<start_times, start_times<nrem_ts[1])]
            # new intervals run from end of swr to start of next swr --> first interval: start of nrem until first swr
            # --> last iterval: end of last swr until end of nrem
            swr_end_within = np.insert(swr_end_within, 0, nrem_ts[0])
            swr_start_within = np.append(swr_start_within, nrem_ts[-1])
            new_nrem_time_stamps.append(np.vstack((swr_end_within, swr_start_within)))

        event_times = np.hstack(new_nrem_time_stamps).T

        if plot_for_control:
            for et_o in event_times_orig:
                plt.hlines(1, et_o[0], et_o[1], color="red")
            for et in event_times:
                plt.hlines(0.9, et[0], et[1], color="blue")
            plt.ylim(0,1.1)
            plt.xlim(2835, 2840)
            plt.show()

        diff = event_times[:, 1]-event_times[:, 0]

        event_times = event_times[diff > 0, :]

        # compute #spike binning for each event
        # --------------------------------------------------------------------------------------------------------------
        event_spike_rasters = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning_fast_jittered(event_times=event_times, event_time_freq=1,
                                                              spikes_per_bin=spikes_per_bin,
                                                            nr_spikes_per_jitter_window=nr_spikes_per_jitter_window)

        return event_times, event_spike_rasters


    def get_spike_binned_raster_nrem_outside_swr(self, spikes_per_bin=None,
                                                plot_for_control=False, speed_threshold=None):
        """
        Computes and returns spike binned raster (constant number of spikes per bin)

        :param nr_spikes_per_bin: how many spikes per bin to use (usually 12 or so)
        :type nr_spikes_per_bin: int
        :param return_estimated_times: return estimated time per constant spike bin
        :type return_estimated_times: bool
        :return: raster (and estimated time if set to True)
        :rtype: np.array
        """

        # get SWR timings (in sec) to exclude
        # ------------------------------------------------------------------------------------------------------
        start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

        # only select SWR during nrem phases
        # get nrem phases in seconds
        nrem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                 classification_method="std")

        # need to generate new time stamps excluding SWR periods
        new_nrem_time_stamps = []

        # go through all nrem time stamps
        for nrem_ts in nrem_time_stamps:
            swr_start_within = start_times[np.logical_and(nrem_ts[0]<start_times, start_times<nrem_ts[1])]
            swr_end_within = end_times[np.logical_and(nrem_ts[0]<start_times, start_times<nrem_ts[1])]
            # new intervals run from end of swr to start of next swr --> first interval: start of nrem until first swr
            # --> last iterval: end of last swr until end of nrem
            swr_end_within = np.insert(swr_end_within, 0, nrem_ts[0])
            swr_start_within = np.append(swr_start_within, nrem_ts[-1])
            new_nrem_time_stamps.append(np.vstack((swr_end_within, swr_start_within)))

        new_nrem_time_stamps = np.hstack(new_nrem_time_stamps).T

        if plot_for_control:
            for nrem_ts in nrem_time_stamps:
                plt.hlines(1, nrem_ts[0], nrem_ts[1], color="blue", label="NREM")
            for swr_s, swr_e in zip(start_times, end_times):
                plt.hlines(2, swr_s, swr_e, color="yellow", label="SWR")
            for nrem_ts in new_nrem_time_stamps:
                plt.hlines(1.5, nrem_ts[0], nrem_ts[1], color="red", label="NEW NREM")
            plt.ylim(0,2.2)
            plt.xlim(40,50)
            # plt.legend()
            plt.show()

        event_spike_rasters = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning_fast(event_times=new_nrem_time_stamps, event_time_freq=1,
                                                                   spikes_per_bin=spikes_per_bin)

          # delete event with empty bins
        events_to_delete_bin = np.array([x.shape[1] == 0 for x in event_spike_rasters])
        events_to_delete = np.argwhere(events_to_delete_bin).flatten()
        # start from the last entry and delete events from list
        for ev in np.flip(events_to_delete):
            del event_spike_rasters[ev]

        event_times = new_nrem_time_stamps[~events_to_delete_bin, :]

        return event_times, event_spike_rasters

    def get_spike_binned_raster_excluded_periods(self, spikes_per_bin=None,
                                                          plot_for_control=False,
                                                          speed_threshold=None, use_pre_computed_time_stamp=True):
        """
        Computes and returns spike binned raster (constant number of spikes per bin)

        :param nr_spikes_per_bin: how many spikes per bin to use (usually 12 or so)
        :type nr_spikes_per_bin: int
        :param return_estimated_times: return estimated time per constant spike bin
        :type return_estimated_times: bool
        :return: raster (and estimated time if set to True)
        :rtype: np.array
        """

        if not "nslp" in self.time_stamps:
            use_pre_computed_time_stamp = False
            print("Computing awake periods because .nslp file was not in data dictionary.")

        if use_pre_computed_time_stamp:

            # speed_threshold = 1e3 --> don't use speed_threshold
            excluded_time_periods_above_threshold = self.get_sleep_phase(sleep_phase="nslp",
                                                                         speed_threshold=1e3,
                                                    classification_method="std")
        else:
            # get nrem phases in seconds
            event_times_nrem = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                     classification_method="std")

            print(" - "+str(event_times_nrem.shape[0])+" NREM phases found (speed thr.: "+str(speed_threshold)+")\n")

            # get rem intervals in seconds
            event_times_rem = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold,
                                                   classification_method="std")

            print(" - "+str(event_times_rem.shape[0])+" REM phases found (speed thr.: "+str(speed_threshold)+")\n")

            # combine REM and NREM event times (order them first)
            start_times = np.hstack((event_times_rem[:,0], event_times_nrem[:,0]))
            end_times = np.hstack((event_times_rem[:,1], event_times_nrem[:,1]))
            labels = np.zeros(event_times_rem.shape[0]+event_times_nrem.shape[0])
            labels[:event_times_rem.shape[0]] = 1
            order = np.argsort(start_times)

            # combine all events (REM & NREM)
            all_event_times_ordered = np.vstack((start_times[order], end_times[order])).T

            # get duration of whole data file (last spike resolution: 20 kHz--> in seconds)
            dur_data = self.last_spike/20e3
            # end times of excluded data == start times of NREM/REM periods
            end_times = all_event_times_ordered[:, 0]
            # --> need to insert end of data file: 5 second buffer
            end_times = np.append(end_times, dur_data-5)
            # start times of excluded data == end times of NREM/REM periods
            start_times = all_event_times_ordered[:, 1]
            # --> need to insert zero at the beginning for first period
            start_times = np.insert(start_times, 0, 0)
            excluded_time_periods = np.vstack((start_times, end_times)).T

            # make sure that intervals have a certain length
            dur_period = end_times - start_times
            excluded_time_periods = excluded_time_periods[dur_period>1, :]

            if plot_for_control:
                # plot NREM/REM and excluded periods
                for events in all_event_times_ordered:
                    plt.hlines(1, events[0], events[1], color="blue", label="NREM/REM")
                for events in excluded_time_periods:
                    plt.hlines(0.9, events[0], events[1], color="red", label="NREM/REM")
                plt.ylim(0.0,1.2)
                plt.xlabel("TIME")
                plt.yticks([0.9, 1], ["EXCL.", "NREM/REM"])
                plt.tight_layout()
                plt.show()

            # only use periods where the average speed is above threshold
            self.compute_speed()
            speed = self.speed
            # convert excluded_periods_into (1s resolution) speed resolution (time_bin_size resolution)
            res_matched_excluded_time_periods = excluded_time_periods * (1/self.params.time_bin_size)
            if plot_for_control:
                plt.plot(speed)
                plt.title("Excluded periods")
                for events in res_matched_excluded_time_periods:
                    plt.hlines(0.9, events[0], events[1], color="red")
                # plt.ylim(0.0,1.2)
                plt.xlabel("TIME")
                plt.ylabel("Speed (a.u.)")
                plt.hlines(self.session_params.sleep_phase_speed_threshold, 0, speed.shape[0], label="threshold")
                plt.ylim(0, 100)
                plt.legend()
                plt.tight_layout()
                plt.show()

            # compute speed in 5 second windows and only include periods above this threshold
            excluded_time_periods_above_threshold = []

            window_size_matched_res = 5 * (1/self.params.time_bin_size)

            # go through excluded periods and split into windows
            for tp in res_matched_excluded_time_periods:
                nr_windows_per_period = int(np.floor((tp[1]-tp[0])/window_size_matched_res))
                for i_window in range(nr_windows_per_period):
                    start_w = int(np.floor(tp[0]+i_window*window_size_matched_res))
                    end_w = int(np.floor(tp[0]+(i_window+1)*window_size_matched_res))
                    if np.mean(speed[start_w:end_w]) > self.session_params.sleep_phase_speed_threshold:
                        excluded_time_periods_above_threshold.append([tp[0]+i_window*window_size_matched_res,
                                                                      tp[0]+(i_window+1)*window_size_matched_res])

            if plot_for_control:
                plt.plot(speed)
                plt.title("Excluded periods")
                for events in excluded_time_periods_above_threshold:
                    plt.hlines(0.9, events[0], events[1], color="red")
                # plt.ylim(0.0,1.2)
                plt.xlabel("TIME")
                plt.ylabel("Speed (a.u.)")
                plt.hlines(self.session_params.sleep_phase_speed_threshold, 0, speed.shape[0], label="threshold")
                plt.ylim(0, 100)
                plt.legend()
                plt.tight_layout()
                plt.show()

            # convert to seconds again
            excluded_time_periods_above_threshold = np.vstack(excluded_time_periods_above_threshold) * self.params.time_bin_size

            print(" - "+str(excluded_time_periods_above_threshold.shape[0])+" excluded periods used\n")

            if excluded_time_periods_above_threshold.shape[0] == 0:
                return None, None

        # compute #spike binning for each event
        # --------------------------------------------------------------------------------------------------------------
        event_spike_rasters = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning_fast(event_times=excluded_time_periods_above_threshold,
                                                                   event_time_freq=1,
                                                                   spikes_per_bin=spikes_per_bin)
        # delete event with empty bins
        events_to_delete_bin = np.array([x.shape[1] == 0 for x in event_spike_rasters])
        events_to_delete = np.argwhere(events_to_delete_bin).flatten()
        # start from the last entry and delete events from list
        for ev in np.flip(events_to_delete):
            del event_spike_rasters[ev]

        excluded_time_periods_above_threshold = excluded_time_periods_above_threshold[~events_to_delete_bin, :]

        return excluded_time_periods_above_threshold, event_spike_rasters

    def get_spike_binned_raster_swr_during_excluded_periods(self, spikes_per_bin=None,
                                                          plot_for_control=False, window_size_min = 5,smoothing_window_s=5,
                                                          speed_threshold=None, use_pre_computed_time_stamp=False):
        """
        Computes and returns spike binned raster (constant number of spikes per bin)

        :param nr_spikes_per_bin: how many spikes per bin to use (usually 12 or so)
        :type nr_spikes_per_bin: int
        :param return_estimated_times: return estimated time per constant spike bin
        :type return_estimated_times: bool
        :return: raster (and estimated time if set to True)
        :rtype: np.array
        """

        if not "nslp" in self.time_stamps:
            use_pre_computed_time_stamp = False
            print("Computing awake periods because .nslp file was not in data dictionary.")

        if use_pre_computed_time_stamp:

            # speed_threshold = 1e3 --> don't use speed_threshold
            excluded_time_periods = self.get_sleep_phase(sleep_phase="nslp",
                                                                         speed_threshold=1e3,
                                                    classification_method="std")
        else:
            if speed_threshold is None:
                speed_threshold = self.params.sleep_phase_speed_threshold


            # take 5 min windows and check if animal moved -->if yes: assign window to excluded period
            # ----------------------------------------------------------------------------------------------------------
            # compute speed at 1s resolution
            speed = self.compute_speed(time_bin_size=1)

            # apply smoothing --> compute median value and set whole smoothing window to the median
            nr_smoothing_windows = int(np.round(self.speed.shape[0] / smoothing_window_s))

            speed_smooth = np.zeros(speed.shape[0])
            for s_w in range(nr_smoothing_windows):
                speed_smooth[s_w * smoothing_window_s:(s_w + 1) * smoothing_window_s].fill(
                    np.median(speed[s_w * smoothing_window_s:(s_w + 1) * smoothing_window_s]))

            if plot_for_control:
                plt.plot(speed, color="grey")
                plt.plot(speed_smooth, color="red", label="smoothed")
                plt.legend()
                plt.show()


            window_size_sec = window_size_min * 60
            nr_windows = int(np.round(speed_smooth.shape[0]/window_size_sec))

            sec_bins_movement = np.zeros(speed_smooth.shape[0])
            # go trough all windows and check if speed was above threshold
            for i_window in range(nr_windows):
                if any(speed_smooth[i_window*window_size_sec:(i_window+1)*window_size_sec] >
                       speed_threshold):
                    sec_bins_movement[i_window*window_size_sec:(i_window+1)*window_size_sec] = 1

            transitions = np.diff(sec_bins_movement)

            start = []
            end = []

            if sec_bins_movement[0] == 1:
                # first data point during movemet
                start.append(0)

            for bin_nr, tran in enumerate(transitions):
                if tran == -1:
                    end.append(bin_nr)
                if tran == 1:
                    start.append(bin_nr+1)

            if sec_bins_movement[-1] == 1:
                # last data point during movement
                end.append(sec_bins_movement.shape[0])

            start = np.array(start)
            end = np.array(end)
            excluded_time_periods = np.vstack((start, end))

            if plot_for_control:
                plt.plot(self.speed)
                plt.plot(sec_bins_movement*4)
                for s, e in zip(start, end):
                    plt.vlines(s, 0, 4, color="blue")
                    plt.vlines(e, 0, 4, color="red")
                plt.show()

        # check if there any excluded periods --> if not: return empty raster
        if excluded_time_periods.shape[0] == 0:
            print("no excluded periods")
            return None, None

        # get SWR timings (in sec) & compute spike rasters (constant #spikes)
        # ------------------------------------------------------------------------------------------------------
        start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

        # convert to one array for event_spike_binning
        event_times = np.vstack((start_times, end_times, peak_times)).T

        swr_awake = np.zeros(event_times.shape[0])
        for n_rem_phase_id, excl_phase in enumerate(excluded_time_periods.T):
            awake_start = excl_phase[0]
            awake_end = excl_phase[1]
            for i, e_t in enumerate(event_times):
                event_start = e_t[0]
                event_end = e_t[1]
                if (awake_start < event_start) and (event_end < awake_end):
                    swr_awake[i] += 1

        event_times = event_times[swr_awake == 1]

        print(" - " + str(event_times.shape[0]) + " WERE ASSIGNED TO AWAKE\n")

        # compute #spike binning for each event
        # --------------------------------------------------------------------------------------------------------------
        event_spike_rasters = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning_fast(event_times=event_times,
                                                                   event_time_freq=1,
                                                                   spikes_per_bin=spikes_per_bin)
        # delete event with empty bins
        events_to_delete_bin = np.array([x.shape[1] == 0 for x in event_spike_rasters])
        events_to_delete = np.argwhere(events_to_delete_bin).flatten()
        # start from the last entry and delete events from list
        for ev in np.flip(events_to_delete):
            del event_spike_rasters[ev]

        event_times = event_times[~events_to_delete_bin, :]

        return event_times, event_spike_rasters

    def get_spike_binned_raster_sleep_phase_jittered(self, spikes_per_bin=None, sleep_phase="nrem",
                                                     plot_for_control=False, speed_threshold=None,
                                                     nr_spikes_per_jitter_window=200):
        """
        Computes and returns spike binned raster (constant number of spikes per bin)

        :param nr_spikes_per_bin: how many spikes per bin to use (usually 12 or so)
        :type nr_spikes_per_bin: int
        :param return_estimated_times: return estimated time per constant spike bin
        :type return_estimated_times: bool
        :return: raster (and estimated time if set to True)
        :rtype: np.array
        """

        # get sleep phase intervals
        # --------------------------------------------------------------------------------------------------------------
        if sleep_phase == "nrem":

            # get SWR timings (in sec) & compute spike rasters (constant #spikes)
            # ------------------------------------------------------------------------------------------------------
            start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

            # convert to one array for event_spike_binning
            event_times = np.vstack((start_times, end_times, peak_times)).T

            # only select SWR during nrem phases
            # get nrem phases in seconds
            n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                     classification_method="std")
            swr_in_n_rem = np.zeros(event_times.shape[0])
            swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
            for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
                n_rem_start = n_rem_phase[0]
                n_rem_end = n_rem_phase[1]
                for i, e_t in enumerate(event_times):
                    event_start = e_t[0]
                    event_end = e_t[1]
                    if (n_rem_start < event_start) and (event_end < n_rem_end):
                        swr_in_n_rem[i] += 1
                        swr_in_which_n_rem[n_rem_phase_id, i] = 1

            event_times = event_times[swr_in_n_rem == 1]
            # assignment: which SWR belongs to which nrem phase (for plotting)
            swr_to_nrem = swr_in_which_n_rem[:, swr_in_n_rem == 1]

            print(" - "+str(event_times.shape[0])+" WERE ASSIGNED TO NREM PHASES\n")

            start_times = event_times[:, 0]
            end_times = event_times[:, 1]

        elif sleep_phase == "rem":
            # get rem intervals in seconds
            event_times = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold,
                                               classification_method="std")

            print(" - "+str(event_times.shape[0])+" REM phases found (speed thr.: "+str(speed_threshold)+")\n")

            start_times = event_times[:, 0]
            end_times = event_times[:, 1]

        # compute #spike binning for each event
        # --------------------------------------------------------------------------------------------------------------
        event_spike_rasters = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning_fast_jittered(event_times=event_times, event_time_freq=1,
                                                              spikes_per_bin=spikes_per_bin,
                                                            nr_spikes_per_jitter_window=nr_spikes_per_jitter_window)

        return event_times, event_spike_rasters

    def get_sleep_phase(self, sleep_phase, speed_threshold=None, classification_method="std"):
        """
        returns start & end (+peak for sharp waves) times for defined sleep phase in seconds

        - timestamps are at 20kHz resolution for sleep data when loaded from .rem, .nrem, .sw

        @param sleep_phase: start/end times for which sleep phase ("rem", "nrem", "sw")
        @type sleep_phase: str
        @param speed_threshold: used to filter periods of movements (only used when type="std"), when None -->
        parameter is loaded from params
        @type speed_threshold: float
        @param classification_method: which sleep classification to use --> "std": Jozsef's standard algorithm,
        "k_means": Juan's sleep
        classification
        @type classification_method: str
        """
        if classification_method == "std":
            # load timestamps at 20kHz resolution
            time_stamps_orig = self.time_stamps[sleep_phase]

        elif classification_method == "k_means":
            time_stamps_orig = np.loadtxt(self.params.pre_proc_dir+"sleep_states_k_means/"+self.session_name+"/" +
                                          self.session_name+"_" +\
                                          self.experiment_phase_id+"."+sleep_phase).astype(int)

        else:
            raise Exception("Sleep classification method not found!")

        if speed_threshold is None:
            speed_threshold = self.session_params.sleep_phase_speed_threshold

        # convert time stamps to same resolution like speed
        temp_time_stamps = (time_stamps_orig * 0.00005)

        # get speed at 1 second resolution
        speed = self.compute_speed(time_bin_size=1)

        # good periods
        good_per = np.ones(time_stamps_orig.shape[0])

        # go through all time stamps
        for i, t_s in enumerate(temp_time_stamps):
            speed_during_per = speed[np.round(t_s[0]).astype(int):np.round(t_s[1]).astype(int)]
            len_per = speed_during_per.shape[0]
            above = np.count_nonzero(np.nan_to_num(speed_during_per) > speed_threshold)
            # if there are periods longer than 1 sec above threshold or more than 50% above threshold,
            # and period should not be shorter than 1 sec
            # --> need to be deleted
            if above * self.params.time_bin_size > 1 or above > 0.05 * len_per or \
                    len_per * self.params.time_bin_size < 3:
                # make it a bad period
                good_per[i] = 0

        time_stamps_orig = time_stamps_orig[good_per.astype(bool), :]

        # get time stamps in seconds
        time_stamps = time_stamps_orig * 0.00005

        # make sure there is no interval that is smaller than one second
        dur = time_stamps[:,1] - time_stamps[:,0]
        to_delete = np.where(dur<1)[0]
        time_stamps = np.delete(time_stamps, to_delete, axis=0)

        return time_stamps

    def get_event_spike_rasters(self, part_to_analyze, speed_threshold=None, plot_for_control=False,
                                return_event_times=False, pop_vec_threshold=None, sleep_classification_method="std"):
        """
        Computes constant spike rasters for specific parts of sleep (e.g. SWR, rem, nrem)

        :param part_to_analyze: which part of sleep ("all_swr", "rem", "nrem")
        :type part_to_analyze: str
        :param speed_threshold: additional speed threshold for sleep (to exclude waking periods)
        :type speed_threshold: float
        :param plot_for_control: plot intermediate results
        :type plot_for_control: bool
        :param return_event_times: if set to True --> times of events (e.g. SWR) are returned as well
        :type return_event_times: bool
        :param pop_vec_threshold: minimum number of population vectors (with constant number of spikes) per event
        :type pop_vec_threshold: int
        :return: event_spike rasters, window_length (+start_times, +end_times)
        :rtype: np.array
        """

        file_name = self.session_name + "_" + self.experiment_phase_id
        result_dir = self.params.pre_proc_dir + "sleep_spike_rasters/" + part_to_analyze

        # check if results exist
        if os.path.isfile(result_dir + "/" + file_name):

            res_dic = pickle.load(open(result_dir + "/" + file_name, "rb"))

            event_spike_rasters = res_dic["event_spike_rasters"]
            event_spike_window_lengths = res_dic["event_spike_window_lengths"]
            start_times = res_dic["start_times"]
            end_times = res_dic["end_times"]

        else:

            # load speed threshold for REM/NREM from parameter file if not provided
            if speed_threshold is None:
                speed_threshold = self.session_params.sleep_phase_speed_threshold

            # check if SWR or REM phases are supposed to be analyzed
            if part_to_analyze == "all_swr":

                # get SWR timings (in sec) & compute spike rasters (constant #spikes)
                # ------------------------------------------------------------------------------------------------------
                start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

                # convert to one array for event_spike_binning
                event_times = np.vstack((start_times, end_times, peak_times)).T

                # compute #spike binning for each event --> TODO: implement sliding window!
                event_spike_rasters, event_spike_window_lengths = \
                    PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)
                # do not need this info here (all SWR)
                swr_to_nrem = None

            elif part_to_analyze == "nrem":

                # get SWR timings (in sec) & compute spike rasters (constant #spikes)
                # ------------------------------------------------------------------------------------------------------
                start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

                # convert to one array for event_spike_binning
                event_times = np.vstack((start_times, end_times, peak_times)).T

                # only select SWR during nrem phases
                # get nrem phases in seconds
                n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold)
                swr_in_n_rem = np.zeros(event_times.shape[0])
                swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
                for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
                    n_rem_start = n_rem_phase[0]
                    n_rem_end = n_rem_phase[1]
                    for i, e_t in enumerate(event_times):
                        event_start = e_t[0]
                        event_end = e_t[1]
                        if (n_rem_start < event_start) and (event_end < n_rem_end):
                            swr_in_n_rem[i] += 1
                            swr_in_which_n_rem[n_rem_phase_id, i] = 1

                event_times = event_times[swr_in_n_rem == 1]
                # assignment: which SWR belongs to which nrem phase (for plotting)
                swr_to_nrem = swr_in_which_n_rem[:, swr_in_n_rem == 1]

                print(" - " + str(event_times.shape[0]) + " WERE ASSIGNED TO NREM PHASES\n")

                # do not need this info here
                swr_to_nrem = None

                start_times = event_times[:, 0]
                end_times = event_times[:,
                            1]  # compute #spike binning for each event --> TODO: implement sliding window!
                event_spike_rasters, event_spike_window_lengths = \
                    PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

            elif part_to_analyze == "rem":
                # get rem intervals in seconds
                event_times = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold,
                                                   classification_method=sleep_classification_method)

                print(" - " + str(event_times.shape[0]) + " REM phases found (speed thr.: " + str(
                    speed_threshold) + ")\n")

                # compute #spike binning for each event --> TODO: implement sliding window!
                event_spike_rasters, event_spike_window_lengths = \
                    PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

                # do not need this info here
                swr_to_nrem = None

                start_times = event_times[:, 0]
                end_times = event_times[:, 1]

            else:
                raise Exception("Sleep phase was not one of [all_swr, nrem, rem]")

            result_post = {
                "event_spike_rasters": event_spike_rasters,
                "event_spike_window_lengths": event_spike_window_lengths,
                "start_times": start_times,
                "end_times": end_times
            }
            outfile = open(result_dir + "/" + file_name, 'wb')
            pickle.dump(result_post, outfile)
            print("  - SAVED NEW RESULTS!\n")

        if pop_vec_threshold is not None:
            # filter epochs that are too short
            a = [x.shape[1] for x in event_spike_rasters]
            to_delete = np.argwhere(np.array(a) < pop_vec_threshold).flatten()
            # remove results
            for i in reversed(to_delete):
                del event_spike_rasters[i]
                del event_spike_window_lengths[i]

            start_times = np.delete(start_times, to_delete)
            end_times = np.delete(end_times, to_delete)

        if return_event_times:

            return event_spike_rasters, event_spike_window_lengths, start_times, end_times

        else:

            return event_spike_rasters, event_spike_window_lengths

    def get_event_spike_rasters_and_times(self, part_to_analyze, speed_threshold=None, plot_for_control=False,
                                          pop_vec_threshold=None, sleep_classification_method="std"):
        """
        Computes constant spike rasters for specific parts of sleep (e.g. SWR, rem, nrem)

        :param part_to_analyze: which part of sleep ("all_swr", "rem", "nrem")
        :type part_to_analyze: str
        :param speed_threshold: additional speed threshold for sleep (to exclude waking periods)
        :type speed_threshold: float
        :param plot_for_control: plot intermediate results
        :type plot_for_control: bool
        :param return_event_times: if set to True --> times of events (e.g. SWR) are returned as well
        :type return_event_times: bool
        :param pop_vec_threshold: minimum number of population vectors (with constant number of spikes) per event
        :type pop_vec_threshold: int
        :return: event_spike rasters, window_length (+start_times, +end_times)
        :rtype: np.array
        """

        # load speed threshold for REM/NREM from parameter file if not provided
        if speed_threshold is None:
            speed_threshold = self.session_params.sleep_phase_speed_threshold

        # check if SWR or REM phases are supposed to be analyzed
        if part_to_analyze == "all_swr":

            # get SWR timings (in sec) & compute spike rasters (constant #spikes)
            # ------------------------------------------------------------------------------------------------------
            start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

            # convert to one array for event_spike_binning
            event_times = np.vstack((start_times, end_times, peak_times)).T

            # compute #spike binning for each event --> TODO: implement sliding window!
            event_spike_rasters, event_spike_window_lengths = \
                PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)
            # do not need this info here (all SWR)
            swr_to_nrem = None

        elif part_to_analyze == "nrem":

            # get SWR timings (in sec) & compute spike rasters (constant #spikes)
            # ------------------------------------------------------------------------------------------------------
            start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

            # convert to one array for event_spike_binning
            event_times = np.vstack((start_times, end_times, peak_times)).T

            # only select SWR during nrem phases
            # get nrem phases in seconds
            n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold)
            swr_in_n_rem = np.zeros(event_times.shape[0])
            swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
            for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
                n_rem_start = n_rem_phase[0]
                n_rem_end = n_rem_phase[1]
                for i, e_t in enumerate(event_times):
                    event_start = e_t[0]
                    event_end = e_t[1]
                    if (n_rem_start < event_start) and (event_end < n_rem_end):
                        swr_in_n_rem[i] += 1
                        swr_in_which_n_rem[n_rem_phase_id, i] = 1

            event_times = event_times[swr_in_n_rem == 1]
            # assignment: which SWR belongs to which nrem phase (for plotting)
            swr_to_nrem = swr_in_which_n_rem[:, swr_in_n_rem == 1]

            print(" - " + str(event_times.shape[0]) + " WERE ASSIGNED TO NREM PHASES\n")

            # do not need this info here
            swr_to_nrem = None

            start_times = event_times[:, 0]
            end_times = event_times[:,
                        1]  # compute #spike binning for each event --> TODO: implement sliding window!
            event_spike_rasters, event_spike_window_lengths, spike_bin_times = \
                PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1,
                                                                  return_bin_times=True)

        elif part_to_analyze == "rem":
            # get rem intervals in seconds
            event_times = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold,
                                               classification_method=sleep_classification_method)

            print(" - " + str(event_times.shape[0]) + " REM phases found (speed thr.: " + str(
                speed_threshold) + ")\n")

            # compute #spike binning for each event --> TODO: implement sliding window!
            event_spike_rasters, event_spike_window_lengths, spike_bin_times = \
                PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1,
                                                                  return_bin_times=True)

            # do not need this info here
            swr_to_nrem = None

            start_times = event_times[:, 0]
            end_times = event_times[:, 1]

        else:
            raise Exception("Sleep phase was not one of [all_swr, nrem, rem]")

        if pop_vec_threshold is not None:
            # filter epochs that are too short
            a = [x.shape[1] for x in event_spike_rasters]
            to_delete = np.argwhere(np.array(a) < pop_vec_threshold).flatten()
            # remove results
            for i in reversed(to_delete):
                del event_spike_rasters[i]
                del event_spike_window_lengths[i]

        return event_spike_rasters, event_spike_window_lengths, spike_bin_times

    def get_event_time_bin_rasters(self, sleep_phase, time_bin_size=None, speed_threshold=None,
                                   plot_for_control=False):
        """
        Computes temporal binning rasters for sleep parts (e.g. rem, nrem, all_swr)

        :param sleep_phase: which part of sleep ("nrem", "rem", "all_swr")
        :type sleep_phase: str
        :param time_bin_size: which time bin size (in sec) to use
        :type time_bin_size: float
        :param speed_threshold: additional speed threshold for sleep
        :type speed_threshold: float
        :param plot_for_control: plot intermediate results
        :type plot_for_control: bool
        :return: rasters and start_times, end_times
        :rtype: np.array
        """
        # load speed threshold for REM/NREM from parameter file if not provided
        if speed_threshold is None:
            speed_threshold = self.session_params.sleep_phase_speed_threshold

        # check if SWR or REM phases are supposed to be analyzed
        if sleep_phase == "all_swr":

            # get SWR timings (in sec) & compute spike rasters (constant #spikes)
            # ------------------------------------------------------------------------------------------------------
            start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

            # convert to one array for event_spike_binning
            event_times = np.vstack((start_times, end_times, peak_times)).T

            # compute #spike binning for each event --> TODO: implement sliding window!
            event_time_bin_rasters = \
                PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                whl=self.whl).event_temporal_binning(event_times=event_times, event_time_freq=1,
                                                                     time_bin_size=time_bin_size)

        elif sleep_phase == "nrem":

            # get SWR timings (in sec) & compute spike rasters (constant #spikes)
            # ------------------------------------------------------------------------------------------------------
            start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

            # convert to one array for event_spike_binning
            event_times = np.vstack((start_times, end_times, peak_times)).T

            # only select SWR during nrem phases
            # get nrem phases in seconds
            n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold)
            swr_in_n_rem = np.zeros(event_times.shape[0])
            swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
            for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
                n_rem_start = n_rem_phase[0]
                n_rem_end = n_rem_phase[1]
                for i, e_t in enumerate(event_times):
                    event_start = e_t[0]
                    event_end = e_t[1]
                    if (n_rem_start < event_start) and (event_end < n_rem_end):
                        swr_in_n_rem[i] += 1
                        swr_in_which_n_rem[n_rem_phase_id, i] = 1

            event_times = event_times[swr_in_n_rem == 1]
            # assignment: which SWR belongs to which nrem phase (for plotting)
            swr_to_nrem = swr_in_which_n_rem[:, swr_in_n_rem == 1]

            print(" - " + str(event_times.shape[0]) + " WERE ASSIGNED TO NREM PHASES\n")

            # do not need this info here
            swr_to_nrem = None

            start_times = event_times[:, 0]
            end_times = event_times[:,
                        1]  # compute #spike binning for each event --> TODO: implement sliding window!
            event_time_bin_rasters = \
                PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                whl=self.whl).event_temporal_binning(event_times=event_times, event_time_freq=1,
                                                                     time_bin_size=time_bin_size)

        elif sleep_phase == "rem":
            # get rem intervals in seconds
            event_times = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold)

            print(" - " + str(event_times.shape[0]) + " REM phases found (speed thr.: " + str(
                speed_threshold) + ")\n")

            # compute #spike binning for each event --> TODO: implement sliding window!
            event_time_bin_rasters = \
                PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                whl=self.whl).event_temporal_binning(event_times=event_times, event_time_freq=1,
                                                                     time_bin_size=time_bin_size)

            # do not need this info here
            swr_to_nrem = None

            start_times = event_times[:, 0]
            end_times = event_times[:, 1]

        else:
            raise Exception("Sleep phase was not one of [all_swr, nrem, rem]")

        return event_time_bin_rasters, start_times, end_times

    def get_event_spike_rasters_artificial_combined_sleep_phases(self, spikes_per_bin=None,
                                                          plot_for_control=False, window_in_min=30,
                                                          speed_threshold=None, save_rasters=False, save_id=None):
        """
        Computes and returns spike binned raster (constant number of spikes per bin)

        :param nr_spikes_per_bin: how many spikes per bin to use (usually 12 or so)
        :type nr_spikes_per_bin: int
        :param return_estimated_times: return estimated time per constant spike bin
        :type return_estimated_times: bool
        :return: raster (and estimated time if set to True)
        :rtype: np.array
        """

        # get SWR timings (in sec) & compute spike rasters (constant #spikes)
        # ------------------------------------------------------------------------------------------------------
        start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

        # convert to one array for event_spike_binning
        event_times = np.vstack((start_times, end_times, peak_times)).T

        # only select SWR during nrem phases
        # get nrem phases in seconds
        n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                 classification_method="std")
        swr_in_n_rem = np.zeros(event_times.shape[0])
        swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
        for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
            n_rem_start = n_rem_phase[0]
            n_rem_end = n_rem_phase[1]
            for i, e_t in enumerate(event_times):
                event_start = e_t[0]
                event_end = e_t[1]
                if (n_rem_start < event_start) and (event_end < n_rem_end):
                    swr_in_n_rem[i] += 1
                    swr_in_which_n_rem[n_rem_phase_id, i] = 1

        event_times_nrem = event_times[swr_in_n_rem == 1]
        print(" - "+str(event_times_nrem.shape[0])+" WERE ASSIGNED TO NREM PHASES\n")

        # get rem intervals in seconds
        event_times_rem = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold,
                                               classification_method="std")

        print(" - "+str(event_times_rem.shape[0])+" REM phases found (speed thr.: "+str(speed_threshold)+")\n")

        # combine REM and NREM event times (order them first)
        start_times = np.hstack((event_times_rem[:,0], event_times_nrem[:,0]))
        end_times = np.hstack((event_times_rem[:,1], event_times_nrem[:,2]))
        labels = np.zeros(event_times_rem.shape[0]+event_times_nrem.shape[0])
        labels[:event_times_rem.shape[0]] = 1
        order = np.argsort(start_times)

        all_event_times_ordered = np.vstack((start_times[order], end_times[order])).T

        # for debugging:
        # all_event_times_ordered = all_event_times_ordered[:50,:]
        # labels_ordered = labels_ordered[:50]

        # compute #spike binning for each event --> TODO: implement sliding window!
        # --------------------------------------------------------------------------------------------------------------
        event_time_bin_rasters, _ = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning_artificial_data(event_times=all_event_times_ordered,
                                                                              event_time_freq=1,
                                                                              spikes_per_bin=spikes_per_bin,
                                                                              window_in_min=window_in_min)

        return event_times, event_time_bin_rasters

    def get_pre_post_templates(self):
        """
        Returns models from PRE (pHMM and Ising) and models from POST
        :return:
        :rtype:
        """
        return self.session_params.default_pre_phmm_model, self.session_params.default_post_phmm_model, \
               self.session_params.default_pre_ising_model, self.session_params.default_post_ising_model

    def get_eegh(self):
        return self.eegh

    # </editor-fold>

    # <editor-fold desc="LFP analysis">

    def mean_lfp_for_interval(self, intervals_s):
        # eegh --> 5 kHz
        lfp = self.eegh

        # convert interval in seconds to interval in 5kHz
        intervals = intervals_s * 5e3

        mean_vals = np.zeros(intervals_s.shape[0])
        for i, iv in enumerate(intervals):
            mean_vals[i] = np.mean(np.mean(lfp[int(iv[0]):int(iv[1]), :] ,axis=1))

        return mean_vals

    def nr_swr_for_interval(self, intervals_s):

        _, _, peak_times = self.detect_swr(plot_for_control=False)

        nr_swr_per_interal = np.zeros(intervals_s.shape[0])
        for i, iv in enumerate(intervals_s):
            nr_swr_per_interal[i] = np.sum(np.count_nonzero(np.logical_and(iv[0]>peak_times, peak_times<iv[1])))

        return nr_swr_per_interal

    def phase_preference_per_cell_subset(self, angle_20k, cell_ids, sleep_phase="rem"):
        """
        Phase preference analysis for subsets of cells

        :param sleep_phase: which sleep phase to use ("all", "rem")
        :type sleep_phase: str
        :param angle_20k: oscillation angle at 20kHz
        :type angle_20k: numpy.array
        :param cell_ids: cell ids of cells to be used
        :type cell_ids: numpy.array
        :return: preferred angle per cell
        :rtype: numpy.array
        """
        # spike times at 20kHz
        spike_times = self.firing_times

        # get keys from dictionary and get correct order
        cell_names = []
        for key in spike_times.keys():
            cell_names.append(key[4:])
        cell_names = np.array(cell_names).astype(int)
        cell_names.sort()

        pref_angle = []

        if sleep_phase == "all":
            for cell_id in cell_names[cell_ids]:
                all_cell_spikes = spike_times["cell" + str(cell_id)]
                # remove spikes that like outside array
                all_cell_spikes = all_cell_spikes[all_cell_spikes<angle_20k.shape[0]]
                # make array
                spk_ang = angle_20k[all_cell_spikes]
                pref_angle.append(np.angle(np.sum(np.exp(-1j * spk_ang))))
        elif sleep_phase == "rem":
            # get rem time stamps
            rem_time_stamps_s = self.get_sleep_phase(sleep_phase="rem")
            # convert into 20kHz to match spikes
            rem_time_stamps_20k = rem_time_stamps_s / (1 / 20e3)

            for cell_id in cell_names[cell_ids]:
                cell_spike_times = spike_times["cell" + str(cell_id)]
                # concatenate trial data
                all_cell_spikes = []
                for rem_phase in rem_time_stamps_20k:
                    all_cell_spikes.extend(
                        cell_spike_times[np.logical_and(rem_phase[0] < cell_spike_times,
                                                        cell_spike_times < rem_phase[1])])

                # make array
                spk_ang = angle_20k[all_cell_spikes]
                pref_angle.append(np.angle(np.sum(np.exp(-1j * spk_ang))))

            return np.array(pref_angle)

        return np.array(pref_angle)

    def phase_preference_analysis(self, oscillation="theta", tetrode=1, plot_for_control=False, plotting=True,
                                  sleep_phase="rem"):
        """
        LFP phase preference analysis for stable, inc, dec subsets

        :param oscillation: which oscillation to use ("theta", "slow_gamma", "medium_gamma")
        :type oscillation: str
        :param tetrode: which tetrode to use
        :type tetrode: int
        :param plot_for_control: if intermediate results are supposed to be plotted
        :type plot_for_control: bool
        :param plotting: plot results
        :type plotting: bool
        :param sleep_phase: which sleep phase to use ("all", "rem")
        :type sleep_phase: str
        :return: angles (all positive) for stable, increasing, decreasing subsets
        :rtype: numpy.array
        """

        # get lfp data
        # ["eeg"] --> all channels downsampled 16 times (1.25kHz)
        # ["eegh"] --> one channel per tetrode downsampled 4 times (5kHz) --> time bin size = 0.0002s
        lfp = self.eegh[:, tetrode]

        # downsample to dt = 0.001 --> 1kHz --> take every 5th value
        lfp = lfp[::5]

        # Say you have an LFP signal LFP_Data and some spikes from a cell spk_t
        # First we extract the angle from the signal in a specific frequency band
        # Frequency Range to Extract, you can also select it AFTER running the wavelet on the entire frequency spectrum,
        # by using the variable frequency to select the desired ones
        if oscillation == "theta":
            frq_Limits = [8, 12]
        elif oscillation == "slow_gamma":
            frq_Limits = [20, 50]
        elif oscillation == "medium_gamma":
            frq_Limits = [60, 90]
        else:
            raise Exception("Oscillation not defined!")
        # [8,12] Theta
        # [20,50] Slow Gamma
        # [60,90] Medium Gamma
        # LFP time bin duration in seconds
        # dt = 1/5e3
        dt=0.001
        # ‘morl’ wavelet
        wavelet = "cmor1.5-1.0" # 'cmor1.5-1.0'
        scales = np.arange(1,128)
        s2f = pywt.scale2frequency(wavelet, scales) / dt
        # This block is just to setup the wavelet analysis
        scales = scales[(s2f >= frq_Limits[0]) * (s2f < frq_Limits[1])]
        # scales = scales[np.logical_and(s2f >= Frq_Limits[0], s2f < Frq_Limits[1])]
        print(" - started wavelet decomposition ...")
        # Wavelet decomposition
        [cfs, frequencies] = pywt.cwt(data=lfp, scales=scales, wavelet=wavelet, sampling_period=dt)
        print(" - done!")
        # This is the angle
        angl = np.angle(np.sum(cfs, axis=0))

        # plot for control
        if plot_for_control:
            plt.plot(lfp[:200])
            plt.xlabel("Time")
            plt.ylabel("LFP")
            plt.show()

            for i in range(frequencies.shape[0]):
                plt.plot(cfs[i, :200])
            plt.xlabel("Time")
            plt.ylabel("Coeff")
            plt.show()

            plt.plot(np.sum(cfs[:, :200], axis=0), label="coeff_sum")
            plt.plot(angl[:200]/np.max(angl[:200]), label="angle")
            plt.xlabel("Time")
            plt.ylabel("Angle (norm) / Coeff_sum (norm)")
            plt.legend()
            plt.show()

        # interpolate results to match 20k
        # --------------------------------------------------------------------------------------------------------------
        x_1k = np.arange(lfp.shape[0])*dt
        x_20k = np.arange(lfp.shape[0]*20)*1/20e3
        angle_20k = np.interp(x_20k, x_1k, angl, left=np.nan, right=np.nan)

        if plot_for_control:
            plt.plot(angle_20k[:4000])
            plt.ylabel("Angle")
            plt.xlabel("Time bin (20kHz)")
            plt.show()

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        stable = class_dic["stable_cell_ids"]
        dec = class_dic["decrease_cell_ids"]
        inc = class_dic["increase_cell_ids"]

        pref_angle_stable = self.phase_preference_per_cell_subset(angle_20k=angle_20k, cell_ids=stable,
                                                                  sleep_phase=sleep_phase)
        pref_angle_dec = self.phase_preference_per_cell_subset(angle_20k=angle_20k, cell_ids=dec,
                                                               sleep_phase=sleep_phase)
        pref_angle_inc = self.phase_preference_per_cell_subset(angle_20k=angle_20k, cell_ids=inc,
                                                               sleep_phase=sleep_phase)

        pref_angle_stable_deg = pref_angle_stable *180/np.pi
        pref_angle_dec_deg = pref_angle_dec * 180 / np.pi
        pref_angle_inc_deg = pref_angle_inc * 180 / np.pi

        if plotting:
            plt.hist(pref_angle_stable_deg, density=True, label="stable")
            plt.hist(pref_angle_dec_deg, density=True, label="dec")
            plt.hist(pref_angle_inc_deg, density=True, label="inc")
            plt.show()

        all_positive_angles_stable = np.copy(pref_angle_stable)
        all_positive_angles_stable[all_positive_angles_stable < 0] = \
            2*np.pi+all_positive_angles_stable[all_positive_angles_stable < 0]

        all_positive_angles_dec = np.copy(pref_angle_dec)
        all_positive_angles_dec[all_positive_angles_dec < 0] = 2 * np.pi + all_positive_angles_dec[
            all_positive_angles_dec < 0]

        all_positive_angles_inc = np.copy(pref_angle_inc)
        all_positive_angles_inc[all_positive_angles_inc < 0] = 2 * np.pi + all_positive_angles_inc[
            all_positive_angles_inc < 0]

        if plotting:

            bins_number = 10  # the [0, 360) interval will be subdivided into this
            # number of equal bins
            bins = np.linspace(0.0, 2 * np.pi, bins_number + 1)
            angles = all_positive_angles_stable
            n, _, _ = plt.hist(angles, bins, density=True)

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

            plt.clf()
            width = 2 * np.pi / bins_number
            ax = plt.subplot(1, 1, 1, projection='polar')
            bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
            for bar in bars:
                bar.set_alpha(0.5)
            ax.set_title("inc. cells")
            plt.show()

        else:
            return all_positive_angles_stable, all_positive_angles_dec, all_positive_angles_inc

    def power_spectrum_analysis(self, tetrode=1, analysis_method="wavelet", time_interval_s=None, debug=False,
                                scales_first_half=False, scales_second_half=False, use_lower_spectrum=False):
        """
        LFP power spectrum analysis

        :param tetrode: which tetrode to use
        :type tetrode: int
        :param plot_for_control: if intermediate results are supposed to be plotted
        :type plot_for_control: bool
        :return: frequency, time (s), power
        :rtype: numpy.array
        """

        # get lfp data
        # ["eeg"] --> all channels downsampled 16 times (1.25kHz)
        # ["eegh"] --> one channel per tetrode downsampled 4 times (5kHz) --> time bin size = 0.0002s
        lfp = self.eegh[:, tetrode]
        # downsample to 1kHz
        lfp = lfp[::5]
        dt = 0.001

        # check if only certain time window is supposed to be analyzed
        if time_interval_s is not None:
            lfp = lfp[np.round(time_interval_s[0]/dt).astype(int):np.round(time_interval_s[1]/dt).astype(int)]

        if analysis_method == "wavelet":
            if use_lower_spectrum:
                frq_Limits = [0, 70]

                # ‘morl’ wavelet
                wavelet = "cmor1.5-1.0"  # 'cmor1.5-1.0'
                # for debugging use less scales to make computation faster

                scales = np.arange(10, 300, 5)
                s2f = pywt.scale2frequency(wavelet, scales) / dt
                # This block is just to setup the wavelet analysis
                scales = scales[(s2f >= frq_Limits[0]) * (s2f < frq_Limits[1])]
            else:
                frq_Limits = [0, 300]

                # ‘morl’ wavelet
                wavelet = "cmor1.5-1.0"  # 'cmor1.5-1.0'
                # for debugging use less scales to make computation faster
                if debug:
                    scales = np.arange(1, 10)
                else:
                    scales = np.arange(1, 128)
                s2f = pywt.scale2frequency(wavelet, scales) / dt
                # This block is just to setup the wavelet analysis
                scales = scales[(s2f >= frq_Limits[0]) * (s2f < frq_Limits[1])]
                if scales_first_half:
                    scales = scales[:np.round(scales.shape[0]/2).astype(int)]
                elif scales_second_half:
                    scales = scales[np.round(scales.shape[0] / 2).astype(int):]
            # scales = scales[np.logical_and(s2f >= Frq_Limits[0], s2f < Frq_Limits[1])]
            # print(" - started wavelet decomposition ("+str(lfp.shape)+") ...")
            # todo: need split LFP interval if too large --> otherwise out of memory error

            # Wavelet decomposition
            [cfs, frequencies] = pywt.cwt(data=lfp, scales=scales, wavelet=wavelet, sampling_period=dt)
            power = np.abs(cfs)
            if time_interval_s is None:
                time = np.arange(power.shape[1])*dt
            else:
                time = np.arange(time_interval_s[0],time_interval_s[0]+power.shape[1]) * dt
            # plt.imshow(np.abs(cfs), interpolation='nearest', aspect='auto' )
            # plt.yticks(np.arange(frequencies.shape[0])[::10], np.round(frequencies)[::10])
            # plt.xticks(np.arange(power.shape[1])[::1000], time[::1000])
            # plt.xlabel("Time [s]")
            # plt.ylabel("Freq [Hz]")
            # plt.colorbar()
            # plt.show()

        elif analysis_method == "fft":
            # lfp_part = lfp[:10000]
            #
            # plt.plot(lfp_part)
            # plt.xlabel("Time")
            # plt.ylabel("Amplitude")
            # plt.show()

            frequencies, time, power = signal.spectrogram(x=lfp, fs=1000, scaling="spectrum", nperseg=125)
            # Sxx_dB = 20.0 * np.log10(Sxx)
            #S = 20 * np.log10(Sxx / np.max(Sxx))
            # plt.pcolormesh(t, f, Sxx, shading='gouraud')
            # plt.ylim(0,1250)
            # plt.ylabel('Frequency [Hz]')
            # plt.xlabel('Time [sec]')
            # plt.show()

        return frequencies, time, power

    def power_spectrum_analysis_all_tetrodes(self, analysis_method="wavelet", time_interval_s=None, debug=False,
                                             use_lower_spectrum=False):
        """
        LFP power spectrum analysis

        :param tetrode: which tetrode to use
        :type tetrode: int
        :param plot_for_control: if intermediate results are supposed to be plotted
        :type plot_for_control: bool
        :return: frequency, time (s), power
        :rtype: numpy.array
        """
        print("Started power spectral analysis ...")
        power_list = []
        for tetrode_id in range(self.eegh.shape[1]):
            # get lfp data
            # ["eeg"] --> all channels downsampled 16 times (1.25kHz)
            # ["eegh"] --> one channel per tetrode downsampled 4 times (5kHz) --> time bin size = 0.0002s
            lfp = self.eegh[:, tetrode_id]
            # downsample to 1kHz
            lfp = lfp[::5]
            dt = 0.001

            # check if only certain time window is supposed to be analyzed
            if time_interval_s is not None:
                lfp = lfp[np.round(time_interval_s[0]/dt).astype(int):np.round(time_interval_s[1]/dt).astype(int)]

            if analysis_method == "wavelet":
                if use_lower_spectrum:
                    frq_Limits = [0, 70]

                    # ‘morl’ wavelet
                    wavelet = "cmor1.5-1.0"  # 'cmor1.5-1.0'
                    # for debugging use less scales to make computation faster

                    scales = np.arange(10, 300, 5)
                    s2f = pywt.scale2frequency(wavelet, scales) / dt
                    # This block is just to setup the wavelet analysis
                    scales = scales[(s2f >= frq_Limits[0]) * (s2f < frq_Limits[1])]
                else:
                    frq_Limits = [0, 300]

                    # ‘morl’ wavelet
                    wavelet = "cmor1.5-1.0"  # 'cmor1.5-1.0'
                    # for debugging use less scales to make computation faster
                    if debug:
                        scales = np.arange(1, 10)
                    else:
                        scales = np.arange(1, 128)
                    s2f = pywt.scale2frequency(wavelet, scales) / dt
                    # This block is just to setup the wavelet analysis
                    scales = scales[(s2f >= frq_Limits[0]) * (s2f < frq_Limits[1])]
                    scales = scales[::2]
                # scales = scales[np.logical_and(s2f >= Frq_Limits[0], s2f < Frq_Limits[1])]
                # Wavelet decomposition
                print("Starting wavelet analysis (dat_len="+str(lfp.shape)+")...")
                [cfs, frequencies] = pywt.cwt(data=lfp, scales=scales, wavelet=wavelet, sampling_period=dt)
                print("Done with wavelet analysis\n")
                power = np.abs(cfs)
                if time_interval_s is None:
                    time = np.arange(power.shape[1])*dt
                else:
                    time = np.arange(time_interval_s[0],time_interval_s[0]+power.shape[1]) * dt
                # plt.imshow(np.abs(cfs), interpolation='nearest', aspect='auto' )
                # plt.yticks(np.arange(frequencies.shape[0])[::10], np.round(frequencies)[::10])
                # plt.xticks(np.arange(power.shape[1])[::1000], time[::1000])
                # plt.xlabel("Time [s]")
                # plt.ylabel("Freq [Hz]")
                # plt.colorbar()
                # plt.show()

            elif analysis_method == "fft":
                # lfp_part = lfp[:10000]
                #
                # plt.plot(lfp_part)
                # plt.xlabel("Time")
                # plt.ylabel("Amplitude")
                # plt.show()

                frequencies, time, power = signal.spectrogram(x=lfp, fs=1000, scaling="spectrum", nperseg=125)
                # Sxx_dB = 20.0 * np.log10(Sxx)
                #S = 20 * np.log10(Sxx / np.max(Sxx))
                # plt.pcolormesh(t, f, Sxx, shading='gouraud')
                # plt.ylim(0,1250)
                # plt.ylabel('Frequency [Hz]')
                # plt.xlabel('Time [sec]')
                # plt.show()
            power_list.append(power)

        power = np.array(power_list)
        print("Done ... \n")

        return frequencies, time, power

    # </editor-fold>

    # <editor-fold desc="Plotting & saving">

    def view_raster(self):
        """
        plot raster data
        """
        self.compute_raster_speed_loc()
        plt.imshow(self.raster.T)
        plt.show()

    def save_spike_binned_raster(self):

        raster = self.get_event_spike_rasters(part_to_analyze="rem")
        raster = raster[0][0]
        plt.style.use('default')
        max_val = np.max(raster[:,:20])+1
        cmap = plt.get_cmap('viridis', max_val)
        plt.imshow(raster[:,:20], interpolation='nearest', aspect='auto', cmap=cmap)
        a = plt.colorbar()
        tick_locs = (np.arange(max_val) + 0.5) * (max_val - 1) / max_val
        a.set_ticks(tick_locs)
        a.set_ticklabels(np.arange(max_val).astype(int))
        plt.rcParams['svg.fonttype'] = 'none'
        #plt.show()
        plt.savefig(os.path.join(save_path, "spike_bin_raster_2.svg"), transparent="True")

    # saving
    # ------------------------------------------------------------------------------------------------------------------

    def save_spike_times(self, save_dir):
        # --------------------------------------------------------------------------------------------------------------
        # determines spike times of each cell and saves them as a list of list (each list: firing times of one cell)
        # --> used for TreeHMM
        #
        # args:   - save_dir, str
        # --------------------------------------------------------------------------------------------------------------

        spike_times = PreProcessSleep(self.firing_times, self.params, self.time_stamps).spike_times()
        filename = save_dir + "/" + self.params.session_name+"_"+self.sleep_type+"_"+self.cell_type
        # pickle in using python2 protocol
        with open(filename, "wb") as f:
            pickle.dump(spike_times, f, protocol=2)

    # </editor-fold>

    # <editor-fold desc="Sleep classification">

    def analyze_sleep_phase(self, speed_threshold=None):
        """
        Plots different sleep stages and speed

        :param speed_threshold: which speed threshold to use for sleep classification in cm/s
        :type speed_threshold: float
        """
        if speed_threshold is None:
            speed_threshold = self.session_params.sleep_phase_speed_threshold

        self.compute_speed()
        time_stamps_rem = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold)
        time_stamps_nrem = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold)
        # compute pre processed data
        plt.plot(np.arange(self.speed.shape[0])*self.params.time_bin_size, self.speed)
        plt.ylabel("SPEED / cm/s")
        plt.xlabel("TIME / s")
        plt.ylim(0,100)
        # plt.title("SLEEP: "+ sleep_phase+" PHASES")

        rem_per = np.zeros(int(self.speed.shape[0]*self.params.time_bin_size))
        for t_s in time_stamps_rem:
            rem_per[int(t_s[0]):int(t_s[1])] = 10

        nrem_per = np.zeros(int(self.speed.shape[0]*self.params.time_bin_size))
        for t_s in time_stamps_nrem:
            nrem_per[int(t_s[0]):int(t_s[1])] = 10

        plt.plot(rem_per, color="r", label="REM")
        plt.plot(nrem_per, color="b", alpha=0.5, label="NREM")
        if speed_threshold is not None:
            axes = plt.gca()
            plt.hlines(speed_threshold, 0, self.speed.shape[0]*self.params.time_bin_size, color="yellow", zorder=1000,
                       label="THRESH.")
        plt.ylim(0,100)
        plt.legend()

        # for t_s in time_stamps:
        #     # time stamps are in seconds --> need in time bin size
        #     plt.hlines(1, t_s[0], t_s[1], colors="r", zorder=1000)

        plt.show()

    def compare_sleep_classification(self, speed_threshold=5):
        """
        Compares sleep classification results from different approaches (k-means and theta/delta ratio)

        :param speed_threshold: which speed threshold to use for sleep classification
        :type speed_threshold: float
        """
        time_stamps_rem_new = self.get_sleep_phase(sleep_phase="rem", classification_method="k_means")

        time_stamps_nrem_new = self.get_sleep_phase(sleep_phase="nrem", classification_method="k_means")

        self.compute_speed()
        time_stamps_rem = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold)
        time_stamps_nrem = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold)

        # compute pre processed data
        plt.plot(np.arange(self.speed.shape[0])*self.params.time_bin_size, self.speed)
        plt.ylabel("SPEED / cm/s")
        plt.xlabel("TIME / s")
        # plt.title("SLEEP: "+ sleep_phase+" PHASES")

        rem_per = np.zeros(int(self.speed.shape[0]*self.params.time_bin_size))
        for t_s in time_stamps_rem:
            rem_per[int(t_s[0]):int(t_s[1])] = 10

        rem_per_new = np.zeros(int(self.speed.shape[0]*self.params.time_bin_size))
        for t_s in time_stamps_rem_new:
            rem_per_new[int(t_s[0]):int(t_s[1])] = 10

        nrem_per_new = np.zeros(int(self.speed.shape[0]*self.params.time_bin_size))
        for t_s in time_stamps_nrem_new:
            nrem_per_new[int(t_s[0]):int(t_s[1])] = 10

        # plt.plot(rem_per, color="r", label="REM")
        plt.plot(rem_per*-1, color="orangered", alpha=0.5, label="REM")
        plt.plot(rem_per_new, color="red", alpha=0.5, label="REM NEW")
        plt.plot(nrem_per_new, color="blue", alpha=0.5, label="NREM NEW")
        if speed_threshold is not None:
            axes = plt.gca()
            plt.hlines(speed_threshold, 0, self.speed.shape[0]*self.params.time_bin_size, color="yellow", zorder=1000,
                       label="THRESH.")
        plt.legend()

        # for t_s in time_stamps:
        #     # time stamps are in seconds --> need in time bin size
        #     plt.hlines(1, t_s[0], t_s[1], colors="r", zorder=1000)

        plt.show()

    # </editor-fold>

    # <editor-fold desc="Sleep decoding analysis: PRE and POST model">
    
    @staticmethod
    def compute_values_from_likelihoods(pre_lh_list, post_lh_list, pre_lh_z_list, post_lh_z_list):
        """
        Computes values such as the normalized ratio between pre and post likelihoods given the raw data
        
        :param pre_lh_list: list with pre likelihoods per event
        :type pre_lh_list: list
        :param post_lh_list: list with post likelihoods per event
        :type post_lh_list: list
        :param pre_lh_z_list: list with z-scored pre likelihoods per event
        :type pre_lh_z_list: list
        :param post_lh_z_list: list with z-scored post likelihoods per event
        :type post_lh_z_list: list
        :return: multiple measures such as normalized pre-post ratio for likelihoods
        :rtype: np.array
        """
        
        # per event results
        event_pre_post_ratio = []
        event_pre_post_ratio_z = []
        event_pre_prob = []
        event_post_prob = []
        event_len_seq = []

        # per population vector results
        pop_vec_pre_post_ratio = []
        pre_seq_list = []
        pre_seq_list_z = []
        post_seq_list = []
        pre_seq_list_prob = []
        post_seq_list_prob = []
        pop_vec_post_prob = []
        pop_vec_pre_prob = []

        # go trough all events
        for pre_array, post_array, pre_array_z, post_array_z in zip(pre_lh_list, post_lh_list, pre_lh_z_list,
                                                                    post_lh_z_list):
            # make sure that there is any data for the current SWR
            if pre_array.shape[0] > 0:
                pre_sequence = np.argmax(pre_array, axis=1)
                pre_sequence_z = np.argmax(pre_array_z, axis=1)
                pre_sequence_prob = np.max(pre_array, axis=1)
                post_sequence = np.argmax(post_array, axis=1)
                post_sequence_prob = np.max(post_array, axis=1)
                pre_seq_list_z.extend(pre_sequence_z)
                pre_seq_list.extend(pre_sequence)
                post_seq_list.extend(post_sequence)
                pre_seq_list_prob.extend(pre_sequence_prob)
                post_seq_list_prob.extend(post_sequence_prob)

                # check how likely observed sequence is considering transitions from model (awake behavior)
                event_len_seq.append(pre_sequence.shape[0])

                # per SWR computations
                # ----------------------------------------------------------------------------------------------
                # arrays: [nr_pop_vecs_per_SWR, nr_time_spatial_time_bins]
                # get maximum value per population vector and take average across the SWR
                if pre_array.shape[0] > 0:
                    # save pre and post probabilities
                    event_pre_prob.append(np.mean(np.max(pre_array, axis=1)))
                    event_post_prob.append(np.mean(np.max(post_array, axis=1)))
                    # compute ratio by picking "winner" mode by first comparing z scored probabilities
                    # then the probability of the most over expressed mode (highest z-score) is used
                    pre_sequence_z = np.argmax(pre_array_z, axis=1)
                    prob_pre_z = np.mean(pre_array[:, pre_sequence_z])
                    post_sequence_z = np.argmax(post_array_z, axis=1)
                    prob_post_z = np.mean(post_array[:, post_sequence_z])
                    event_pre_post_ratio_z.append((prob_post_z - prob_pre_z) / (prob_post_z + prob_pre_z))

                    # compute ratio using probabilites
                    prob_pre = np.mean(np.max(pre_array, axis=1))
                    prob_post = np.mean(np.max(post_array, axis=1))
                    event_pre_post_ratio.append((prob_post - prob_pre) / (prob_post + prob_pre))
                else:
                    event_pre_prob.append(np.nan)
                    event_post_prob.append(np.nan)
                    event_pre_post_ratio.append(np.nan)

                # per population vector computations
                # ----------------------------------------------------------------------------------------------
                # compute per population vector similarity score
                prob_post = np.max(post_array, axis=1)
                prob_pre = np.max(pre_array, axis=1)
                pop_vec_pre_post_ratio.extend((prob_post - prob_pre) / (prob_post + prob_pre))

                if pre_array.shape[0] > 0:
                    pop_vec_pre_prob.extend(np.max(pre_array, axis=1))
                    pop_vec_post_prob.extend(np.max(post_array, axis=1))
                else:
                    pop_vec_pre_prob.extend([np.nan])
                    pop_vec_post_prob.extend([np.nan])

        pop_vec_pre_prob = np.array(pop_vec_pre_prob)
        pop_vec_post_prob = np.array(pop_vec_post_prob)
        pop_vec_pre_post_ratio = np.array(pop_vec_pre_post_ratio)
        pre_seq_list = np.array(pre_seq_list)

        return event_pre_post_ratio, event_pre_post_ratio_z, event_pre_prob, event_post_prob, event_len_seq,\
               pop_vec_pre_post_ratio, pre_seq_list, pre_seq_list_z, post_seq_list, pre_seq_list_prob, \
               post_seq_list_prob, pop_vec_post_prob, pop_vec_pre_prob

    # content based memory drift (decoding using ising map, phmm modes from PRE and POST)
    # ------------------------------------------------------------------------------------------------------------------

    def decode_activity_using_pre_post(self, template_type, part_to_analyze, pre_file_name=None, post_file_name=None,
                                       compression_factor=None, speed_threshold=None, plot_for_control=False,
                                       return_results=True, sleep_classification_method="std", cells_to_use="all",
                                       shuffling=False):
        """
        decodes sleep activity using pHMM modes/spatial bins from ising model from before and after awake activity and
        computes similarity measure

        @param template_type: whether to use "phmm" or "ising"
        @type template_type: str
        @param pre_file_name: name of file containing the template from PRE behavioral data --> if = None: file
        defined in session parameter file
        @type pre_file_name: str
        @param post_file_name: name of file containing the template from AFTER/POST awake/behavioral data --> if = None:
        file defined in session parameter file
        @type post_file_name: str
        @param part_to_analyze: which data to analyse ("all_swr", "rem", "nrem")
        @type part_to_analyze: str
        @param compression_factor: scaling factor between sleep activity and awake activity --> if awake
        model was build with constant time bin size (e.g. 100ms): compute how many spikes on average happen during this
        window size (e.g. mean = 30), check how many #spikes you used for constant #spike binning (e.g. 12) for the
        sleep data --> awake_activity_scaling_factor = 12/30 --> = 0.4
        if None --> is loaded from session specific parameter file
        @type compression_factor: float
        @param speed_threshold: filter for rem/nrem phases --> all phases with speeds above this threshold are
        neglected, if you don't want to use a speed_threshold set to None.
        @type speed_threshold: int
        @param plot_for_control: plots intermediate results if True
        @type plot_for_control: bool
        @param sleep_classification_method: which sleep classification to use --> std: Jozsef's script, k_mean: Juan's
        script
        @type sleep_classification_method: str
        @param cells_to_use: which cells to use ("all", "stable", "inc", "dec")
        @type cells_to_use: str
        @param return_results: whether to return results or not
        @type return_results: bool
        @return: pre_prob (probabilities of PRE), post_prob (probabilities of ), event_times (in sec),
        swr_in_which_n_rem (assignement SWR - nrem phase)
        @rtype: list, list, list, numpy.array
        @param shuffling: whether to perform spike shuffling or not (swapping spikes n times)
        @type shuffling: bool
        """

        # load speed threshold for REM/NREM from parameter file if not provided
        if speed_threshold is None:
            speed_threshold = self.session_params.sleep_phase_speed_threshold

        if template_type == "phmm":
            # get template file name from parameter file of session if not provided
            if pre_file_name is None:
                pre_file_name = self.session_params.default_pre_phmm_model
            if post_file_name is None:
                post_file_name = self.session_params.default_post_phmm_model
            if sleep_classification_method == "std":
                print(" - Decoding "+self.experiment_phase+" ("+part_to_analyze + ") using PHMM modes & std. "
                                                                                  "sleep classification...\n")
                if cells_to_use == "stable":
                    if shuffling:
                        result_dir = "phmm_decoding/stable_cells_shuffled_"+self.params.stable_cell_method
                    else:
                        result_dir = "phmm_decoding/stable_cells_"+self.params.stable_cell_method
                elif cells_to_use == "increasing":
                    result_dir = "phmm_decoding/inc_cells_"+self.params.stable_cell_method
                elif cells_to_use == "decreasing":
                    result_dir = "phmm_decoding/dec_cells_"+self.params.stable_cell_method
                elif cells_to_use == "dec_inc":
                    result_dir = "phmm_decoding/dec_inc_cells_"+self.params.stable_cell_method
                elif cells_to_use == "all":
                    if shuffling:
                        result_dir = "phmm_decoding/spike_shuffled"
                    else:
                        result_dir = "phmm_decoding"

                else:
                    raise Exception("Not defined which cells to use [all, decreasing, increasing, stable]")
            elif sleep_classification_method == "k_means":
                print(" - Decoding "+self.experiment_phase+" ("+part_to_analyze + ") using PHMM modes & k-means "
                                                                                  "sleep classification...\n")
                result_dir = "phmm_decoding/k_means_sleep_classification"

        elif template_type == "ising":
            print(" - Decoding "+self.experiment_phase+" ("+part_to_analyze + ") using Ising model ...\n")
            # get template file name from parameter file of session if not provided
            if pre_file_name is None:
                pre_file_name = self.session_params.default_pre_ising_model
            if post_file_name is None:
                post_file_name = self.session_params.default_post_ising_model

            if cells_to_use == "stable":
                result_dir = "ising_glm_decoding/stable_cells_"+self.params.stable_cell_method
            else:
                result_dir = "ising_glm_decoding"

            if not self.params.spikes_per_bin == 12:
                result_dir = result_dir +"/"+str(self.params.spikes_per_bin)+"_spikes"
                if not os.path.exists(self.params.pre_proc_dir +result_dir):
                    os.makedirs(self.params.pre_proc_dir +result_dir)
        else:
            raise Exception("Template needs to be defined [phmm, ising]")

        if pre_file_name is None or post_file_name is None:
            raise Exception("AT LEAST ONE TEMPLATE FILE WAS NEITHER PROVIDED\n NOR IN SESSION PARAMETER FILE DEFINED")

        file_name_pre = self.session_name + "_" +self.experiment_phase_id + "_" + part_to_analyze + "_" + \
                        self.cell_type+"_PRE.npy"
        file_name_post = self.session_name + "_" + self.experiment_phase_id + "_" + part_to_analyze + "_" + \
                         self.cell_type+"_POST.npy"
        # To compute results again (not using the saved ones)
        # file_name_pre = self.session_name + "_" +self.experiment_phase_id + "_" + part_to_analyze + "_" + \
        #                 self.cell_type+"new_PRE.npy"
        # file_name_post = self.session_name + "_" + self.experiment_phase_id + "_" + part_to_analyze + "_" + \
        #                  self.cell_type+"new_POST.npy"

        # check if PRE and POST result exists already
        # --------------------------------------------------------------------------------------------------------------
        if os.path.isfile(self.params.pre_proc_dir + result_dir + "/" + file_name_pre) and \
            os.path.isfile(self.params.pre_proc_dir + result_dir + "/" + file_name_post):
            print(" - PRE and POST results exist already -- using existing results\n")
        else:
            # if PRE and/or POST do not exist yet --> compute results

            # check if SWR or REM phases are supposed to be analyzed

            if part_to_analyze == "all_swr":

                # get SWR timings (in sec) & compute spike rasters (constant #spikes)
                # ------------------------------------------------------------------------------------------------------
                start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

                # convert to one array for event_spike_binning
                event_times = np.vstack((start_times, end_times, peak_times)).T

                # compute #spike binning for each event --> TODO: implement sliding window!
                event_spike_rasters, event_spike_window_lenghts = \
                    PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)
                # do not need this info here (all SWR)
                swr_to_nrem = None

            elif part_to_analyze == "nrem":

                # get SWR timings (in sec) & compute spike rasters (constant #spikes)
                # ------------------------------------------------------------------------------------------------------
                start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

                # convert to one array for event_spike_binning
                event_times = np.vstack((start_times, end_times, peak_times)).T

                # only select SWR during nrem phases
                # get nrem phases in seconds
                n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                         classification_method=sleep_classification_method)
                swr_in_n_rem = np.zeros(event_times.shape[0])
                swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
                for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
                    n_rem_start = n_rem_phase[0]
                    n_rem_end = n_rem_phase[1]
                    for i, e_t in enumerate(event_times):
                        event_start = e_t[0]
                        event_end = e_t[1]
                        if (n_rem_start < event_start) and (event_end < n_rem_end):
                            swr_in_n_rem[i] += 1
                            swr_in_which_n_rem[n_rem_phase_id, i] = 1

                event_times = event_times[swr_in_n_rem == 1]
                # assignment: which SWR belongs to which nrem phase (for plotting)
                swr_to_nrem = swr_in_which_n_rem[:, swr_in_n_rem == 1]

                print(" - "+str(event_times.shape[0])+" WERE ASSIGNED TO NREM PHASES\n")

                # do not need this info here
                swr_to_nrem = None

                start_times = event_times[:, 0]
                end_times = event_times[:, 1]
                # compute #spike binning for each event
                event_spike_rasters, event_spike_window_lenghts = \
                    PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

            elif part_to_analyze == "rem":
                # get rem intervals in seconds
                event_times = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold,
                                                   classification_method=sleep_classification_method)

                print(" - "+str(event_times.shape[0])+" REM phases found (speed thr.: "+str(speed_threshold)+")\n")
                # compute #spike binning for each event
                event_spike_rasters, event_spike_window_lenghts = \
                    PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

                # do not need this info here
                swr_to_nrem = None

                start_times = event_times[:, 0]
                end_times = event_times[:, 1]

            else:
                raise Exception("Provided part_to_analyze is not defined!")

            # plot raster and detected SWR, example spike rasters from SWRs
            if plot_for_control:
                # ------------------------------------------------------------------------------------------------------
                # plot detected events
                # ------------------------------------------------------------------------------------------------------
                # compute pre processed data
                self.compute_raster_speed_loc()
                to_plot = np.random.randint(0, start_times.shape[0], 5)
                for i in to_plot:
                    plt.imshow(self.raster, interpolation='nearest', aspect='auto')
                    plt.vlines(start_times[i] / self.params.time_bin_size, 0, self.raster.shape[0] - 1,
                               colors="r",
                               linewidth=0.5,
                               label="START")
                    plt.vlines(end_times[i] / self.params.time_bin_size, 0, self.raster.shape[0] - 1,
                               colors="g",
                               linewidth=0.5, label="END")
                    plt.xlim(start_times[i] / self.params.time_bin_size - 10,
                             end_times[i] / self.params.time_bin_size + 10)
                    plt.ylabel("CELL IDS")
                    plt.xlabel("TIME BINS / "+str(self.params.time_bin_size)+"s")
                    plt.legend()
                    a = plt.colorbar()
                    a.set_label("SPIKES PER BIN (" + str(self.params.time_bin_size) + " s)")
                    plt.title("EVENT ID " + str(i))
                    plt.show()

                    plt.imshow(event_spike_rasters[i], interpolation='nearest', aspect='auto')
                    a = plt.colorbar()
                    a.set_label("SPIKES PER CONST. #SPIKES BIN")
                    plt.xlabel("CONST. #SPIKE POP. VEC. ID")
                    plt.ylabel("CELL ID")
                    plt.title("BINNED EVENT ID " + str(i) + "\n#SPIKES PER BIN: " + str(
                        self.params.spikes_per_bin))
                    plt.show()

                # ------------------------------------------------------------------------------------------------------
                # compute length of constant #spike windows
                # ------------------------------------------------------------------------------------------------------
                event_spike_window_lengths_avg = np.mean(np.concatenate(event_spike_window_lenghts, axis=0))
                event_spike_window_lengths_median = np.median(np.concatenate(event_spike_window_lenghts, axis=0))

                y, x, _ = plt.hist(np.concatenate(event_spike_window_lenghts, axis=0), bins=30)
                plt.xlabel("LENGTH CONST #SPIKES POP. VEC / s")
                plt.ylabel("COUNTS")
                plt.title("POPULATION VECTOR LENGTH")
                plt.vlines(event_spike_window_lengths_avg, 0, y.max(), colors="r", label="MEAN: "+
                                        str(np.round(event_spike_window_lengths_avg,2)))
                plt.vlines(event_spike_window_lengths_median, 0, y.max(), colors="b", label="MEDIAN: "+
                                                str(np.round(event_spike_window_lengths_median,2)))
                plt.legend()
                plt.show()

            event_spike_rasters_orig = copy.deepcopy(event_spike_rasters)

            for result_file_name, template_file_name in zip([file_name_pre, file_name_post],
                                                            [pre_file_name, post_file_name]):
                if os.path.isfile(self.params.pre_proc_dir + result_dir + "/" + result_file_name):
                    print(" - RESULT EXISTS ALREADY (" + result_file_name + ")\n")
                    continue
                else:
                    print(" - DECODING SLEEP ACTIVITY USING " + template_file_name + " ...\n")

                    if template_type == "phmm":
                        # load pHMM model
                        with open(self.params.pre_proc_dir + "phmm/" + template_file_name + '.pkl', 'rb') as f:
                            model_dic = pickle.load(f)
                        # get means of model (lambdas) for decoding
                        mode_means = model_dic.means_

                        time_bin_size_encoding = model_dic.time_bin_size

                        # check if const. #spike bins are correct for the loaded compression factor
                        if not self.params.spikes_per_bin == 12:
                            raise Exception("TRYING TO LOAD COMPRESSION FACTOR FOR 12 SPIKES PER BIN, "
                                            "BUT CURRENT #SPIKES PER BIN != 12")

                        # load correct compression factor (as defined in parameter file of the session)
                        if time_bin_size_encoding == 0.01:
                            compression_factor = \
                                np.round(self.session_params.sleep_compression_factor_12spikes_100ms * 10, 3)
                        elif time_bin_size_encoding == 0.1:
                            compression_factor = self.session_params.sleep_compression_factor_12spikes_100ms
                        else:
                            raise Exception("COMPRESSION FACTOR NEITHER PROVIDED NOR FOUND IN PARAMETER FILE")

                        # if you want to use different compression factors for PRE/POST
                        # if "PRE" in result_file_name:
                        #     compression_factor = 0.4
                        # elif "POST" in result_file_name:
                        #     compression_factor = 0.6
                        if cells_to_use == "stable":
                            cell_selection = "all"
                            # load cell ids of stable cells
                            # get stable, decreasing, increasing cells
                            with open(self.params.pre_proc_dir + "cell_classification/" +
                                      self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                                      "rb") as f:
                                class_dic = pickle.load(f)

                            cells_ids = class_dic["stable_cell_ids"]

                        elif cells_to_use == "increasing":
                            cell_selection = "all"
                            # load cell ids of stable cells

                            with open(self.params.pre_proc_dir + "cell_classification/" +
                                      self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                                      "rb") as f:
                                class_dic = pickle.load(f)

                            cells_ids = class_dic["increase_cell_ids"]

                        elif cells_to_use == "decreasing":
                            cell_selection = "all"
                            # load cell ids of stable cells

                            with open(self.params.pre_proc_dir + "cell_classification/" +
                                      self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                                      "rb") as f:
                                class_dic = pickle.load(f)

                            cells_ids = class_dic["decrease_cell_ids"]


                        elif cells_to_use == "dec_inc":
                            cell_selection = "all"
                            # load cell ids of stable cells

                            with open(self.params.pre_proc_dir + "cell_classification/" +
                                      self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                                      "rb") as f:
                                class_dic = pickle.load(f)

                            cells_ids = np.hstack((class_dic["decrease_cell_ids"], class_dic["increase_cell_ids"]))

                        elif cells_to_use == "all":
                            cell_selection = "all"

                        else:
                            raise Exception("Cells_to_use needs to be defined [stable, increasing, decreasing, all]")

                        if cells_to_use in ["stable", "increasing", "decreasing", "dec_inc"]:
                            # need to select event_spike_rasters only for subsets, the same for lambdas
                            # to only select cells that are wanted
                            event_spike_rasters_modified = []
                            for ev_r in event_spike_rasters_orig:
                                if ev_r.shape[1] > 0:
                                    event_spike_rasters_modified.append(ev_r[cells_ids, :])

                            event_spike_rasters = event_spike_rasters_modified

                            mode_means = mode_means[:, cells_ids]

                        if shuffling:
                            print(" -- STARTED SWAPPING PROCEDURE ...")
                            # merge all events
                            conc_data = np.hstack(event_spike_rasters)
                            nr_swaps = conc_data.shape[1]*10
                            for shuffle_id in range(nr_swaps):
                                # select two random time bins
                                t1 = 1
                                t2 = 1
                                while(t1 == t2):
                                    t1 = np.random.randint(conc_data.shape[1])
                                    t2 = np.random.randint(conc_data.shape[1])
                                # check in both time bins which cells are active
                                act_cells_t1 = np.argwhere(conc_data[:, t1].flatten()>0).flatten()
                                act_cells_t2 = np.argwhere(conc_data[:, t2].flatten()>0).flatten()
                                # find intersect (same cells need to be firing in t1 and t2 in order to exchange
                                # spikes
                                # original code
                                # --------------------------------------------------------------------------------------
                                # cells_firing_in_both = np.intersect1d(act_cells_t1, act_cells_t2)
                                # if cells_firing_in_both.shape[0] > 1:
                                #     # select first cell to swap
                                #     cell_1 = 1
                                #     cell_2 = 1
                                #     while (cell_1 == cell_2):
                                #         cell_1 = np.random.choice(cells_firing_in_both)
                                #         cell_2 = np.random.choice(cells_firing_in_both)
                                #     # do the actual swapping
                                #     conc_data[cell_1, t1] += 1
                                #     conc_data[cell_1, t2] -= 1
                                #     conc_data[cell_2, t1] -= 1
                                #     conc_data[cell_2, t2] += 1

                                if act_cells_t1.shape[0] > 1 and act_cells_t2.shape[0] > 1:
                                    # select first cell to swap
                                    cell_1 = 1
                                    cell_2 = 1
                                    while (cell_1 == cell_2):
                                        cell_1 = np.random.choice(act_cells_t2)
                                        cell_2 = np.random.choice(act_cells_t1)
                                    # do the actual swapping
                                    conc_data[cell_1, t1] += 1
                                    conc_data[cell_1, t2] -= 1
                                    conc_data[cell_2, t1] -= 1
                                    conc_data[cell_2, t2] += 1

                            print(" -- ... DONE!")
                            # split data again into list
                            event_lengths = [x.shape[1] for x in event_spike_rasters]

                            event_spike_rasters_shuffled = []
                            start = 0
                            for el in event_lengths:
                                event_spike_rasters_shuffled.append(conc_data[:,start:start+el])
                                start = el

                            event_spike_rasters = event_spike_rasters_shuffled
                        # start with actual decoding
                        # ----------------------------------------------------------------------------------------------

                        print(" - DECODING USING "+ cells_to_use + " CELLS")

                        results_list = decode_using_phmm_modes(mode_means=mode_means,
                                                               event_spike_rasters=event_spike_rasters,
                                                               compression_factor=compression_factor,
                                                               cell_selection=cell_selection)

                    elif template_type == "ising":
                        # load ising template
                        with open(self.params.pre_proc_dir + 'awake_ising_maps/' + template_file_name + '.pkl',
                                  'rb') as f:
                            model_dic = pickle.load(f)

                        # if compression_factor is not provided --> load from parameter file
                        if compression_factor is None:
                            # get time_bin_size of encoding
                            time_bin_size_encoding = model_dic["time_bin_size"]

                            # load correct compression factor (as defined in parameter file of the session)
                            if time_bin_size_encoding == 0.01:
                                compression_factor = \
                                    np.round(self.session_params.sleep_compression_factor_12spikes_100ms * 10, 3)
                            elif time_bin_size_encoding == 0.1:
                                compression_factor = self.session_params.sleep_compression_factor_12spikes_100ms
                            else:
                                raise Exception("COMPRESSION FACTOR NEITHER PROVIDED NOR FOUND IN PARAMETER FILE")

                            # check if const. #spike bins are correct for the loaded compression factor
                            if not self.params.spikes_per_bin == 12:
                                # compression factor is for 12 spike windows --> if results need to be computed
                                # for other #spikes per bin: need to rescale
                                compression_factor = (self.params.spikes_per_bin/12) * compression_factor

                        # check if only stable subset is supposed to be used
                        if cells_to_use == "stable":
                            cell_selection = "all"
                            # load cell ids of stable cells
                            # get stable, decreasing, increasing cells
                            with open(self.params.pre_proc_dir + "cell_classification/" +
                                      self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                                      "rb") as f:
                                class_dic = pickle.load(f)

                            cells_ids = class_dic["stable_cell_ids"]

                        elif cells_to_use == "increasing":
                            cell_selection = "all"
                            # load cell ids of stable cells

                            if self.params.stable_cell_method == "k_means":
                                # load only stable cells
                                with open(self.params.pre_proc_dir + "cell_classification/" +
                                        self.params.session_name + "_k_means.pickle", "rb") as f:
                                    class_dic = pickle.load(f)
                                cells_ids = class_dic["increase_cell_ids"]

                            elif self.params.stable_cell_method == "mean_firing_awake":
                                # load only stable cells
                                with open(self.params.pre_proc_dir + "cell_classification/" +
                                        self.params.session_name + "_mean_firing_awake.pickle", "rb") as f:
                                    class_dic = pickle.load(f)

                                cells_ids = class_dic["increase_cell_ids"]

                            else:
                                raise Exception("NOT IMPLEMENTED YET!!!")

                        elif cells_to_use == "decreasing":
                            cell_selection = "all"
                            # load cell ids of stable cells

                            if self.params.stable_cell_method == "k_means":
                                # load only stable cells
                                with open(self.params.pre_proc_dir + "cell_classification/" +
                                        self.params.session_name + "_k_means.pickle", "rb") as f:
                                    class_dic = pickle.load(f)
                                cells_ids = class_dic["decrease_cell_ids"]

                            elif self.params.stable_cell_method == "mean_firing_awake":
                                # load only stable cells
                                with open(self.params.pre_proc_dir + "cell_classification/" +
                                        self.params.session_name + "_mean_firing_awake.pickle", "rb") as f:
                                    class_dic = pickle.load(f)

                                cells_ids = class_dic["decrease_cell_ids"]

                            else:
                                raise Exception("NOT IMPLEMENTED YET!!!")

                        elif cells_to_use == "all":
                            cell_selection = "all"

                        else:
                            raise Exception("Cells_to_use needs to be defined [stable, increasing, decreasing, all]")

                        # get template map
                        template_map = model_dic["res_map"]

                        # check if we need only subset for decoding

                        if cells_to_use in ["stable", "increasing", "decreasing"]:
                            # need to select event_spike_rasters only for subsets, the same for lambdas
                            # to only select cells that are wanted
                            event_spike_rasters_modified = []
                            for ev_r in event_spike_rasters_orig:
                                if ev_r.shape[1] > 0:
                                    event_spike_rasters_modified.append(ev_r[cells_ids, :])

                            event_spike_rasters = event_spike_rasters_modified

                            template_map = template_map[cells_ids,:,:]

                        print(" - DECODING USING "+ cells_to_use + " CELLS (#spikes per bin = "+
                              str(self.params.spikes_per_bin)+")")

                        results_list = decode_using_ising_map(template_map=template_map,
                                                                       event_spike_rasters=event_spike_rasters,
                                                                       compression_factor=compression_factor,
                                                                       cell_selection="all")

                    # plot maps of some SWR for control
                    if plot_for_control:
                        swr_to_plot = np.random.randint(0, len(results_list), 10)
                        for swr in swr_to_plot:
                            res = results_list[swr]
                            plt.imshow(res.T, interpolation='nearest', aspect='auto')
                            plt.xlabel("POP.VEC. ID")
                            plt.ylabel("MODE ID")
                            a = plt.colorbar()
                            a.set_label("PROBABILITY")
                            plt.title("PROBABILITY MAP, SWR ID: " + str(swr))
                            plt.show()

                    # saving results
                    # --------------------------------------------------------------------------------------------------
                    # create dictionary with results
                    result_post = {
                        "results_list": results_list,
                        "event_times": event_times,
                        "swr_to_nrem": swr_to_nrem
                    }
                    outfile = open(self.params.pre_proc_dir + result_dir +"/" + result_file_name, 'wb')
                    pickle.dump(result_post, outfile)
                    outfile.close()
                    print("  - SAVED NEW RESULTS!\n")

        if return_results:

            while True:
                # load decoded maps
                try:
                    result_pre = pickle.load(open(self.params.pre_proc_dir+result_dir + "/" + file_name_pre,"rb"))
                    result_post = pickle.load(open(self.params.pre_proc_dir+result_dir + "/" + file_name_post, "rb"))
                    break
                except:
                    print("Trying to load data ...")
                    continue

            pre_prob = result_pre["results_list"]
            post_prob = result_post["results_list"]
            event_times = result_pre["event_times"]
            swr_to_nrem = result_pre["swr_to_nrem"]

            return pre_prob, post_prob, event_times, swr_to_nrem

    def decode_activity_control(self, pre_or_post="pre", compression_factor=None, speed_threshold=None,
                                plot_for_control=False, return_results=True, sleep_classification_method="std",
                                cells_to_use="all", shuffling=False):
        """
        decodes sleep activity using pHMM modes/spatial bins from ising model from before and after awake activity and
        computes similarity measure

        @param template_type: whether to use "phmm" or "ising"
        @type template_type: str
        @param pre_file_name: name of file containing the template from PRE behavioral data --> if = None: file
        defined in session parameter file
        @type pre_file_name: str
        @param post_file_name: name of file containing the template from AFTER/POST awake/behavioral data --> if = None:
        file defined in session parameter file
        @type post_file_name: str
        @param part_to_analyze: which data to analyse ("all_swr", "rem", "nrem")
        @type part_to_analyze: str
        @param compression_factor: scaling factor between sleep activity and awake activity --> if awake
        model was build with constant time bin size (e.g. 100ms): compute how many spikes on average happen during this
        window size (e.g. mean = 30), check how many #spikes you used for constant #spike binning (e.g. 12) for the
        sleep data --> awake_activity_scaling_factor = 12/30 --> = 0.4
        if None --> is loaded from session specific parameter file
        @type compression_factor: float
        @param speed_threshold: filter for rem/nrem phases --> all phases with speeds above this threshold are
        neglected, if you don't want to use a speed_threshold set to None.
        @type speed_threshold: int
        @param plot_for_control: plots intermediate results if True
        @type plot_for_control: bool
        @param sleep_classification_method: which sleep classification to use --> std: Jozsef's script, k_mean: Juan's
        script
        @type sleep_classification_method: str
        @param cells_to_use: which cells to use ("all", "stable", "inc", "dec")
        @type cells_to_use: str
        @param return_results: whether to return results or not
        @type return_results: bool
        @return: pre_prob (probabilities of PRE), post_prob (probabilities of ), event_times (in sec),
        swr_in_which_n_rem (assignement SWR - nrem phase)
        @rtype: list, list, list, numpy.array
        @param shuffling: whether to perform spike shuffling or not (swapping spikes n times)
        @type shuffling: bool
        """

        # get template file name from parameter file of session if not provided
        if pre_or_post == "pre":
            file_name = self.session_params.default_pre_phmm_model
        elif pre_or_post == "post":
            file_name = self.session_params.default_post_phmm_model

        # load pHMM model
        with open(self.params.pre_proc_dir + "phmm/" + file_name + '.pkl', 'rb') as f:
            model_dic = pickle.load(f)
        # get means of model (lambdas) for decoding
        mode_means = model_dic.means_

        time_bin_size_encoding = model_dic.time_bin_size

        art_data = np.zeros((mode_means.shape[1], 1000))
        rng = np.random.default_rng()
        # generate artificial data set by sampling from different modes
        for time_bin in np.arange(art_data.shape[1]):
            mode_per_cell = np.random.randint(0, mode_means.shape[0], mode_means.shape[1])
            art_data[:, time_bin] = rng.poisson(lam=mode_means[mode_per_cell, np.arange(mode_means.shape[1])],
                                                size=mode_means.shape[1])

        # compute number of spikes for compression factor
        compression_factor = 12/np.mean(np.sum(art_data, axis=0))

        # make 12 spike bins
        # select randomly 12 spikes by deleting random spikes

        art_data_12_spike = np.zeros((art_data.shape[0], art_data.shape[1]))

        for pop_vec_id in range(art_data.shape[1]):
            pop_vec_art = np.copy(art_data[:,pop_vec_id])
            nr_spikes_to_delete = np.sum(pop_vec_art)-12
            while np.sum(pop_vec_art)> 12:
                cells_with_spikes = np.argwhere(pop_vec_art > 0).flatten()
                cell_to_subtract_spike = np.random.choice(cells_with_spikes, 1)
                pop_vec_art[cell_to_subtract_spike] -= 1
            art_data_12_spike[:, pop_vec_id] = pop_vec_art




        # for mo in range(mode_means.shape[1]):
        #     l_i = mode_means[:, mo]
        #     sp_t_all = np.zeros((0,))
        #     sp_i_all = np.zeros((0,))
        #     sp_m = np.zeros(l_i.shape[0])
        #     for nn in range(l_i.shape[0]):
        #         sp_t = np.cumsum(np.random.exponential(1 / l_i[nn], (10000, 1)))
        #         sp_m[nn] = np.max(sp_t)
        #         sp_t_all = np.concatenate((sp_t_all, sp_t))
        #         sp_i = np.ones((10000,)) * nn
        #         sp_i_all = np.concatenate((sp_i_all, sp_i))
        #         # Take the earlier last spike from any cell
        #     thr = np.min(sp_m)
        #     sp_i_all = sp_i_all[sp_t_all < thr]
        #     sp_t_all = sp_t_all[sp_t_all < thr]
        #
        #     # Rearrange spike in time
        #     aa = np.argsort(sp_t_all)
        #     sp_i_all = sp_i_all[aa]
        #
        #     # Build avarage spike occurrence
        #     n_samp = int(np.floor(sp_t_all.shape[0] / 12))
        #     raster_mod = np.zeros((l_i.shape[0], n_samp))
        #     for ss in range(n_samp):
        #         take_sp = sp_i_all[ss * 12:ss * 12 + 12].astype(int)
        #         for sp in range(len(take_sp)):
        #             raster_mod[take_sp[sp], ss] += 1
        #     mode_means[:, mo] = np.mean(raster_mod, axis=1)

        results_per_event = np.zeros((art_data_12_spike.shape[1], mode_means.shape[0]))

        for pop_vec in art_data_12_spike.T:
            for mode_id, mean_vec in enumerate(mode_means):
                    # pythonic way
                    rates = compression_factor * mean_vec
                    prob_f = np.sum(np.log(np.power(rates, pop_vec) * np.exp(-1 * rates) /
                                           factorial(pop_vec)))
                    results_per_event[pop_vec_id, mode_id] = np.exp(prob_f)

        max_likeli = np.max(results_per_event, axis=1)
        return max_likeli

    def decode_phmm_pre_post_using_event_times(self, event_times, pre_file_name=None, post_file_name=None,
                                               speed_threshold=None, plot_for_control=False):
        """
        decodes sleep activity using pHMM modes/spatial bins from ising model from before and after awake activity and
        computes similarity measure

        @param template_type: whether to use "phmm" or "ising"
        @type template_type: str
        @param pre_file_name: name of file containing the template from PRE behavioral data --> if = None: file
        defined in session parameter file
        @type pre_file_name: str
        @param post_file_name: name of file containing the template from AFTER/POST awake/behavioral data --> if = None:
        file defined in session parameter file
        @type post_file_name: str
        @param part_to_analyze: which data to analyse ("all_swr", "rem", "nrem")
        @type part_to_analyze: str
        @param compression_factor: scaling factor between sleep activity and awake activity --> if awake
        model was build with constant time bin size (e.g. 100ms): compute how many spikes on average happen during this
        window size (e.g. mean = 30), check how many #spikes you used for constant #spike binning (e.g. 12) for the
        sleep data --> awake_activity_scaling_factor = 12/30 --> = 0.4
        if None --> is loaded from session specific parameter file
        @type compression_factor: float
        @param speed_threshold: filter for rem/nrem phases --> all phases with speeds above this threshold are
        neglected, if you don't want to use a speed_threshold set to None.
        @type speed_threshold: int
        @param plot_for_control: plots intermediate results if True
        @type plot_for_control: bool
        @param sleep_classification_method: which sleep classification to use --> std: Jozsef's script, k_mean: Juan's
        script
        @type sleep_classification_method: str
        @param cells_to_use: which cells to use ("all", "stable", "inc", "dec")
        @type cells_to_use: str
        @param return_results: whether to return results or not
        @type return_results: bool
        @return: pre_prob (probabilities of PRE), post_prob (probabilities of ), event_times (in sec),
        swr_in_which_n_rem (assignement SWR - nrem phase)
        @rtype: list, list, list, numpy.array
        @param shuffling: whether to perform spike shuffling or not (swapping spikes n times)
        @type shuffling: bool
        """

        # load speed threshold for REM/NREM from parameter file if not provided
        if speed_threshold is None:
            speed_threshold = self.session_params.sleep_phase_speed_threshold

        # get template file name from parameter file of session if not provided
        if pre_file_name is None:
            pre_file_name = self.session_params.default_pre_phmm_model
        if post_file_name is None:
            post_file_name = self.session_params.default_post_phmm_model

        # compute #spike binning for each event
        event_spike_rasters, event_spike_window_lenghts = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

        start_times = event_times[:, 0]
        end_times = event_times[:, 1]


        # plot raster and detected SWR, example spike rasters from SWRs
        if plot_for_control:
            # ------------------------------------------------------------------------------------------------------
            # plot detected events
            # ------------------------------------------------------------------------------------------------------
            # compute pre processed data
            self.compute_raster_speed_loc()
            to_plot = np.random.randint(0, start_times.shape[0], 5)
            for i in to_plot:
                plt.imshow(self.raster, interpolation='nearest', aspect='auto')
                plt.vlines(start_times[i] / self.params.time_bin_size, 0, self.raster.shape[0] - 1,
                           colors="r",
                           linewidth=0.5,
                           label="START")
                plt.vlines(end_times[i] / self.params.time_bin_size, 0, self.raster.shape[0] - 1,
                           colors="g",
                           linewidth=0.5, label="END")
                plt.xlim(start_times[i] / self.params.time_bin_size - 10,
                         end_times[i] / self.params.time_bin_size + 10)
                plt.ylabel("CELL IDS")
                plt.xlabel("TIME BINS / "+str(self.params.time_bin_size)+"s")
                plt.legend()
                a = plt.colorbar()
                a.set_label("SPIKES PER BIN (" + str(self.params.time_bin_size) + " s)")
                plt.title("EVENT ID " + str(i))
                plt.show()

                plt.imshow(event_spike_rasters[i], interpolation='nearest', aspect='auto')
                a = plt.colorbar()
                a.set_label("SPIKES PER CONST. #SPIKES BIN")
                plt.xlabel("CONST. #SPIKE POP. VEC. ID")
                plt.ylabel("CELL ID")
                plt.title("BINNED EVENT ID " + str(i) + "\n#SPIKES PER BIN: " + str(
                    self.params.spikes_per_bin))
                plt.show()

            # ------------------------------------------------------------------------------------------------------
            # compute length of constant #spike windows
            # ------------------------------------------------------------------------------------------------------
            event_spike_window_lengths_avg = np.mean(np.concatenate(event_spike_window_lenghts, axis=0))
            event_spike_window_lengths_median = np.median(np.concatenate(event_spike_window_lenghts, axis=0))

            y, x, _ = plt.hist(np.concatenate(event_spike_window_lenghts, axis=0), bins=30)
            plt.xlabel("LENGTH CONST #SPIKES POP. VEC / s")
            plt.ylabel("COUNTS")
            plt.title("POPULATION VECTOR LENGTH")
            plt.vlines(event_spike_window_lengths_avg, 0, y.max(), colors="r", label="MEAN: "+
                                    str(np.round(event_spike_window_lengths_avg,2)))
            plt.vlines(event_spike_window_lengths_median, 0, y.max(), colors="b", label="MEDIAN: "+
                                            str(np.round(event_spike_window_lengths_median,2)))
            plt.legend()
            plt.show()

        event_spike_rasters_orig = copy.deepcopy(event_spike_rasters)

        for i, template_file_name in enumerate([pre_file_name, post_file_name]):

                # load pHMM model
                with open(self.params.pre_proc_dir + "phmm/" + template_file_name + '.pkl', 'rb') as f:
                    model_dic = pickle.load(f)
                # get means of model (lambdas) for decoding
                mode_means = model_dic.means_

                time_bin_size_encoding = model_dic.time_bin_size

                # check if const. #spike bins are correct for the loaded compression factor
                if not self.params.spikes_per_bin == 12:
                    raise Exception("TRYING TO LOAD COMPRESSION FACTOR FOR 12 SPIKES PER BIN, "
                                    "BUT CURRENT #SPIKES PER BIN != 12")

                # load correct compression factor (as defined in parameter file of the session)
                if time_bin_size_encoding == 0.01:
                    compression_factor = \
                        np.round(self.session_params.sleep_compression_factor_12spikes_100ms * 10, 3)
                elif time_bin_size_encoding == 0.1:
                    compression_factor = self.session_params.sleep_compression_factor_12spikes_100ms
                else:
                    raise Exception("COMPRESSION FACTOR NEITHER PROVIDED NOR FOUND IN PARAMETER FILE")

                cell_selection = "all"
                results_list = decode_using_phmm_modes(mode_means=mode_means,
                                                       event_spike_rasters=event_spike_rasters,
                                                       compression_factor=compression_factor,
                                                       cell_selection=cell_selection)
                # plot maps of some SWR for control
                if plot_for_control:
                    swr_to_plot = np.random.randint(0, len(results_list), 10)
                    for swr in swr_to_plot:
                        res = results_list[swr]
                        plt.imshow(res.T, interpolation='nearest', aspect='auto')
                        plt.xlabel("POP.VEC. ID")
                        plt.ylabel("MODE ID")
                        a = plt.colorbar()
                        a.set_label("PROBABILITY")
                        plt.title("PROBABILITY MAP, SWR ID: " + str(swr))
                        plt.show()

                if i==0:
                    pre_res = results_list
                elif i==1:
                    post_res = results_list

        return pre_res, post_res, event_spike_window_lenghts

    def decode_activity_using_pre_post_plot_results(self, template_type, part_to_analyze, pre_file_name=None,
                                                    post_file_name=None, plot_for_control=False,
                                                    n_moving_average_events=10, n_moving_average_pop_vec=40,
                                                    only_stable_cells=False):
        """
        plots results of memory drift analysis (if results don't exist --> they are computed)

        @param template_type: whether to use "phmm" or "ising"
        @type template_type: str
        @param part_to_analyze: which data to analyse ("all_swr", "rem", "nrem", "nrem_rem")
        @type part_to_analyze: str
        @param pre_file_name: name of file containing the template from PRE behavioral data --> if = None: file
        defined in session parameter file
        @type pre_file_name: str
        @param post_file_name: name of file containing the template from AFTER/POST awake/behavioral data --> if = None:
        file defined in session parameter file
        @type post_file_name: str
        @param plot_for_control: plots intermediate results if True
        @type plot_for_control: bool
        @param n_moving_average_events: n for moving average across events
        @type n_moving_average_events: int
        @param n_moving_average_pop_vec: n for moving average across population vectors
        @type n_moving_average_pop_vec: int
        @param only_stable_cells: whether to only use cells with stable firing
        @type only_stable_cells: bool
        """
        if part_to_analyze == "nrem_rem":
            # plot nrem data first
            # ----------------------------------------------------------------------------------------------------------

            pre_lh_list, post_lh_list, event_times, swr_to_nrem = self.decode_activity_using_pre_post(
                template_type=template_type, pre_file_name=pre_file_name, post_file_name=post_file_name,
                plot_for_control=plot_for_control, part_to_analyze="nrem", only_stable_cells=only_stable_cells)

            pre_prob_arr = np.vstack(pre_lh_list)
            post_prob_arr = np.vstack(post_lh_list)

            # z-scoring of probabilites
            pre_prob_arr_z = zscore(pre_prob_arr + 1e-50, axis=0)
            post_prob_arr_z = zscore(post_prob_arr + 1e-50, axis=0)

            # assign array chunks to events again (either SWR or rem phases)
            event_lengths = [x.shape[0] for x in pre_lh_list]
            pre_prob_z = []
            post_prob_z = []
            first = 0
            for swr_id in range(len(event_lengths)):
                pre_prob_z.append(pre_prob_arr_z[first:first + event_lengths[swr_id], :])
                post_prob_z.append(post_prob_arr_z[first:first + event_lengths[swr_id], :])
                first += event_lengths[swr_id]

            event_pre_post_ratio, event_pre_post_ratio_z, event_pre_prob, event_post_prob, event_len_seq, \
            pop_vec_pre_post_ratio, pre_seq_list, pre_seq_list_z, post_seq_list, pre_seq_list_prob, \
            post_seq_list_prob, pop_vec_post_prob, pop_vec_pre_prob = compute_values_from_likelihoods(
                pre_lh_list=pre_lh_list, post_lh_list=post_lh_list, post_lh_z_list=post_prob_z,
            pre_lh_z_list=pre_prob_z)

            # smoothen
            # ------------------------------------------------------------------------------------------------------
            event_pre_post_ratio_smooth = moving_average(a=np.array(event_pre_post_ratio), n=n_moving_average_events)
            event_pre_post_ratio_smooth_z = moving_average(a=np.array(event_pre_post_ratio_z), n=n_moving_average_events)
            pop_vec_pre_post_ratio = np.array(pop_vec_pre_post_ratio)
            # compute moving average to smooth signal
            pop_vec_pre_post_ratio_smooth = moving_average(a=pop_vec_pre_post_ratio, n=50)
            event_len_seq_smooth = moving_average(a=np.array(event_len_seq), n=n_moving_average_events)

            swr_to_nrem = swr_to_nrem[:,:event_pre_post_ratio_smooth.shape[0]]
            event_times = event_times[:event_pre_post_ratio_smooth.shape[0],:]
            # # plot per nrem phase
            fig = plt.figure()
            ax = fig.add_subplot()
            # for nrem_id in range(swr_to_nrem.shape[0]):
            #     ax.plot(event_times[swr_to_nrem[nrem_id,:]==1,1],
            #             event_pre_post_ratio_smooth[swr_to_nrem[nrem_id,:]==1],c="blue", label="NREM")

            # plot per nrem phase
            start = 0
            for rem_length, rem_time in zip(event_lengths, event_times):
                if start + rem_length > pop_vec_pre_post_ratio_smooth.shape[0]:
                    continue
                ax.plot(np.linspace(rem_time[0], rem_time[1], rem_length),
                        pop_vec_pre_post_ratio_smooth[start:start+rem_length],c="blue", label="NREM")
                start += rem_length


            # plot rem data
            # ----------------------------------------------------------------------------------------------------------
            pre_lh_list, post_lh_list, event_times, swr_to_nrem = self.decode_activity_using_pre_post(
                template_type=template_type, pre_file_name=pre_file_name, post_file_name=post_file_name,
                plot_for_control=plot_for_control,
                part_to_analyze="rem")

            pre_prob_arr = np.vstack(pre_lh_list)
            post_prob_arr = np.vstack(post_lh_list)

            # z-scoring of probabilites
            pre_prob_arr_z = zscore(pre_prob_arr + 1e-50, axis=0)
            post_prob_arr_z = zscore(post_prob_arr + 1e-50, axis=0)

            # assign array chunks to events again (either SWR or rem phases)
            event_lengths = [x.shape[0] for x in pre_lh_list]
            pre_prob_z = []
            post_prob_z = []
            first = 0
            for swr_id in range(len(event_lengths)):
                pre_prob_z.append(pre_prob_arr_z[first:first + event_lengths[swr_id], :])
                post_prob_z.append(post_prob_arr_z[first:first + event_lengths[swr_id], :])
                first += event_lengths[swr_id]

            event_pre_post_ratio, event_pre_post_ratio_z, event_pre_prob, event_post_prob, event_len_seq, \
            pop_vec_pre_post_ratio, pre_seq_list, pre_seq_list_z, post_seq_list, pre_seq_list_prob, \
            post_seq_list_prob, pop_vec_post_prob, pop_vec_pre_prob = self.compute_values_from_likelihoods(
                pre_lh_list=pre_lh_list, post_lh_list=post_lh_list, post_lh_z_list=post_prob_z,
            pre_lh_z_list=pre_prob_z)

            # smoothen
            # ------------------------------------------------------------------------------------------------------
            event_pre_post_ratio_smooth = moving_average(a=np.array(event_pre_post_ratio), n=n_moving_average_events)
            event_pre_post_ratio_smooth_z = moving_average(a=np.array(event_pre_post_ratio_z), n=n_moving_average_events)
            pop_vec_pre_post_ratio = np.array(pop_vec_pre_post_ratio)
            # compute moving average to smooth signal
            pop_vec_pre_post_ratio_smooth = moving_average(a=pop_vec_pre_post_ratio, n=n_moving_average_pop_vec)
            event_len_seq_smooth = moving_average(a=np.array(event_len_seq), n=n_moving_average_events)

            # plot per nrem phase
            start = 0
            for rem_length, rem_time in zip(event_lengths, event_times):
                if start + rem_length > pop_vec_pre_post_ratio_smooth.shape[0]:
                    continue
                ax.plot(np.linspace(rem_time[0], rem_time[1], rem_length),
                        pop_vec_pre_post_ratio_smooth[start:start+rem_length],c="red", label="REM", alpha=0.5)
                start += rem_length

            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.xlabel("TIME / s")
            plt.ylabel("PRE_POST SIMILARITY")
            plt.title("PRE_POST SIMILARITY USING "+template_type)
            plt.show()

        else:
            pre_lh_list, post_lh_list, event_times, swr_to_nrem = self.decode_activity_using_pre_post(
                template_type=template_type, pre_file_name=pre_file_name, post_file_name=post_file_name,
                plot_for_control=plot_for_control,
                part_to_analyze=part_to_analyze, only_stable_cells=only_stable_cells)

            pre_prob_arr = np.vstack(pre_lh_list)
            post_prob_arr = np.vstack(post_lh_list)

            # z-scoring of probabilites: add very small value in case entries are all zero --> would get an error
            # for z-scoring otherwise
            pre_prob_arr_z = zscore(pre_prob_arr+1e-50, axis=0)
            post_prob_arr_z = zscore(post_prob_arr+1e-50, axis=0)

            # assign array chunks to events again (either SWR or rem phases)
            event_lengths = [x.shape[0] for x in pre_lh_list]
            pre_prob_z = []
            post_prob_z = []
            first = 0
            for swr_id in range(len(event_lengths)):
                pre_prob_z.append(pre_prob_arr_z[first:first + event_lengths[swr_id], :])
                post_prob_z.append(post_prob_arr_z[first:first + event_lengths[swr_id], :])
                first += event_lengths[swr_id]

            nr_modes_pre = pre_lh_list[0].shape[1]
            nr_modes_post = post_lh_list[0].shape[1]

            # per event results
            event_pre_post_ratio = []
            event_pre_post_ratio_z = []
            event_pre_prob = []
            event_post_prob = []
            event_len_seq = []

            # per population vector results
            pop_vec_pre_post_ratio = []
            pre_seq_list = []
            pre_seq_list_z = []
            post_seq_list = []
            pre_seq_list_prob = []
            post_seq_list_prob = []
            pop_vec_post_prob = []
            pop_vec_pre_prob = []

            # go trough all events
            for pre_array, post_array, pre_array_z, post_array_z in zip(pre_lh_list, post_lh_list, pre_prob_z,
                                                                        post_prob_z):
                # make sure that there is any data for the current SWR
                if pre_array.shape[0] > 0:
                    pre_sequence = np.argmax(pre_array, axis=1)
                    pre_sequence_z = np.argmax(pre_array_z, axis=1)
                    pre_sequence_prob = np.max(pre_array, axis=1)
                    post_sequence = np.argmax(post_array, axis=1)
                    post_sequence_prob = np.max(post_array, axis=1)
                    pre_seq_list_z.extend(pre_sequence_z)
                    pre_seq_list.extend(pre_sequence)
                    post_seq_list.extend(post_sequence)
                    pre_seq_list_prob.extend(pre_sequence_prob)
                    post_seq_list_prob.extend(post_sequence_prob)

                    # check how likely observed sequence is considering transitions from model (awake behavior)
                    mode_before = pre_sequence[:-1]
                    mode_after = pre_sequence[1:]
                    event_len_seq.append(pre_sequence.shape[0])

                    # per SWR computations
                    # ----------------------------------------------------------------------------------------------
                    # arrays: [nr_pop_vecs_per_SWR, nr_time_spatial_time_bins]
                    # get maximum value per population vector and take average across the SWR
                    if pre_array.shape[0] > 0:
                        # save pre and post probabilities
                        event_pre_prob.append(np.mean(np.max(pre_array, axis=1)))
                        event_post_prob.append(np.mean(np.max(post_array, axis=1)))
                        # compute ratio by picking "winner" mode by first comparing z scored probabilities
                        # then the probability of the most over expressed mode (highest z-score) is used
                        pre_sequence_z = np.argmax(pre_array_z, axis=1)
                        prob_pre_z = np.mean(pre_array[:, pre_sequence_z])
                        post_sequence_z = np.argmax(post_array_z, axis=1)
                        prob_post_z = np.mean(post_array[:, post_sequence_z])
                        event_pre_post_ratio_z.append((prob_post_z - prob_pre_z) / (prob_post_z + prob_pre_z))

                        # compute ratio using probabilites
                        prob_pre = np.mean(np.max(pre_array, axis=1))
                        prob_post = np.mean(np.max(post_array, axis=1))
                        event_pre_post_ratio.append((prob_post - prob_pre) / (prob_post + prob_pre))
                    else:
                        event_pre_prob.append(np.nan)
                        event_post_prob.append(np.nan)
                        event_pre_post_ratio.append(np.nan)

                    # per population vector computations
                    # ----------------------------------------------------------------------------------------------
                    # compute per population vector similarity score
                    prob_post = np.max(post_array, axis=1)
                    prob_pre = np.max(pre_array, axis=1)
                    pop_vec_pre_post_ratio.extend((prob_post - prob_pre) / (prob_post + prob_pre))

                    if pre_array.shape[0] > 0:
                        pop_vec_pre_prob.extend(np.max(pre_array, axis=1))
                        pop_vec_post_prob.extend(np.max(post_array, axis=1))
                    else:
                        pop_vec_pre_prob.extend([np.nan])
                        pop_vec_post_prob.extend([np.nan])

            pop_vec_pre_prob = np.array(pop_vec_pre_prob)
            pop_vec_post_prob = np.array(pop_vec_post_prob)
            pop_vec_pre_post_ratio = np.array(pop_vec_pre_post_ratio)
            pre_seq_list = np.array(pre_seq_list)

            # to plot a detailed description of one phase of sleep

            # r_to_plot = range(0, 200)
            # plt.figure(figsize=(10, 15))
            # plt.subplot(3, 1, 1)
            # plt.plot(pop_vec_pre_prob[r_to_plot], label="MAX. PROB. PRE")
            # plt.plot(pop_vec_post_prob[r_to_plot], label="MAX. PROB. POST")
            # # plt.plot(pre_SWR_prob_arr[r_to_plot, 10], c="r", label="PROB. MODE 60")
            # plt.legend()
            # plt.ylabel("PROB")
            # plt.grid()
            # plt.yscale("log")
            # plt.title("RESULTS FOR PHMM TEMPLATE" if (template_type=="phmm") else "RESULTS FOR ISING TEMPLATE")
            # plt.subplot(3, 1, 2)
            # plt.scatter(r_to_plot, pop_vec_pre_post_ratio[r_to_plot], c="magenta")
            # plt.ylabel("PRE_POST RATIO")
            # plt.grid()
            # plt.subplot(3, 1, 3)
            # plt.scatter(r_to_plot, pre_seq_list[r_to_plot], c="y")
            # plt.ylabel("PRE MODE ID")
            # plt.grid()
            # plt.xlabel("POP. VEC. ID")
            # plt.show()

            # smoothen
            # ------------------------------------------------------------------------------------------------------
            event_pre_post_ratio_smooth = moving_average(a=np.array(event_pre_post_ratio), n=n_moving_average_events)
            event_pre_post_ratio_smooth_z = moving_average(a=np.array(event_pre_post_ratio_z), n=n_moving_average_events)
            pop_vec_pre_post_ratio = np.array(pop_vec_pre_post_ratio)
            # compute moving average to smooth signal
            pop_vec_pre_post_ratio_smooth = moving_average(a=pop_vec_pre_post_ratio, n=n_moving_average_pop_vec)
            event_len_seq_smooth = moving_average(a=np.array(event_len_seq), n=n_moving_average_events)

            # compute per mode info
            # ------------------------------------------------------------------------------------------------------
            pre_seq = np.array(pre_seq_list)
            post_seq = np.array(post_seq_list)
            mode_score_mean_pre = np.zeros(pre_lh_list[0].shape[1])
            mode_score_std_pre = np.zeros(pre_lh_list[0].shape[1])
            mode_score_mean_post = np.zeros(post_lh_list[0].shape[1])
            mode_score_std_post = np.zeros(post_lh_list[0].shape[1])

            # go through all pre modes and check the average score
            for i in range(pre_lh_list[0].shape[1]):
                ind_sel = np.where(pre_seq == i)[0]
                if ind_sel.size == 0 or ind_sel.size == 1:
                    mode_score_mean_pre[i] = np.nan
                    mode_score_std_pre[i] = np.nan
                else:
                    # delete all indices that are too large (becaue of moving average)
                    ind_sel = ind_sel[ind_sel < pop_vec_pre_post_ratio.shape[0]]
                    mode_score_mean_pre[i] = np.mean(pop_vec_pre_post_ratio[ind_sel])
                    mode_score_std_pre[i] = np.std(pop_vec_pre_post_ratio[ind_sel])

            # go through all post modes and check the average score
            for i in range(post_lh_list[0].shape[1]):
                ind_sel = np.where(post_seq == i)[0]
                if ind_sel.size == 0 or ind_sel.size == 1:
                    mode_score_mean_post[i] = np.nan
                    mode_score_std_post[i] = np.nan
                else:
                    # delete all indices that are too large (becaue of moving average)
                    ind_sel = ind_sel[ind_sel < pop_vec_pre_post_ratio.shape[0]]
                    mode_score_mean_post[i] = np.mean(pop_vec_pre_post_ratio[ind_sel])
                    mode_score_std_post[i] = np.std(pop_vec_pre_post_ratio[ind_sel])

            low_score_modes = np.argsort(mode_score_mean_pre)
            # need to skip nans
            nr_nans = np.count_nonzero(np.isnan(mode_score_mean_pre))
            high_score_modes = np.flip(low_score_modes)[nr_nans:]

            # check if modes get more often/less often reactivated over time
            pre_seq_list = np.array(pre_seq_list)
            nr_pop_vec = 20
            nr_windows = int(pre_seq_list.shape[0] / nr_pop_vec)
            occurence_modes_pre = np.zeros((nr_modes_pre, nr_windows))
            for i in range(nr_windows):
                seq = pre_seq_list[i * nr_pop_vec:(i + 1) * nr_pop_vec]
                mode, counts = np.unique(seq, return_counts=True)
                occurence_modes_pre[mode, i] = counts

            # check if modes get more often/less often reactivated over time
            post_seq_list = np.array(post_seq_list)
            nr_pop_vec = 20
            nr_windows = int(post_seq_list.shape[0] / nr_pop_vec)
            occurence_modes_post = np.zeros((nr_modes_pre, nr_windows))
            for i in range(nr_windows):
                seq = post_seq_list[i * nr_pop_vec:(i + 1) * nr_pop_vec]
                mode, counts = np.unique(seq, return_counts=True)
                occurence_modes_post[mode, i] = counts

            # per event (SWR/rem phase) data
            # ----------------------------------------------------------------------------------------------------------

            # check if plotting smoothed data makes sense
            if len(event_lengths) > 3 * n_moving_average_events:
                # per event pre-post ratio smoothed
                plt.plot(event_pre_post_ratio_smooth, c="r", label="n_mov_avg = " + str(n_moving_average_events))
                plt.title("PRE-POST RATIO FOR EACH EVENT: PHMM" if (template_type=="phmm") else
                          "PRE-POST RATIO FOR EACH EVENT: ISING")
                plt.xlabel("SWR ID" if part_to_analyze=="nrem" else "REM PHASE ID")
                plt.ylabel("PRE-POST SIMILARITY")
                plt.ylim(-1, 1)
                plt.grid()
                plt.legend()
                plt.show()

                # per event pre-post ratio after z-scoring smoothed
                plt.plot(event_pre_post_ratio_smooth_z, c="r", label="n_mov_avg = " + str(n_moving_average_events))
                plt.title(
                    "PRE-POST RATIO FOR EACH EVENT: PHMM\n Z-SCORED TO SELECT WINNER" if (template_type == "phmm") else
                    "PRE-POST RATIO FOR EACH EVENT: ISING\n Z-SCORED TO SELECT WINNER")
                plt.xlabel("SWR ID" if part_to_analyze == "nrem" else "REM PHASE ID")
                plt.ylabel("PRE-POST SIMILARITY")
                plt.ylim(-1, 1)
                plt.grid()
                plt.legend()
                plt.show()

                # event length smoothed
                plt.plot(event_len_seq_smooth, label="n_mov_avg = " + str(n_moving_average_events))
                plt.title("EVENT LENGTH")
                plt.xlabel("SWR ID" if part_to_analyze == "nrem" else "REM PHASE ID")
                plt.ylabel("#POP.VEC. PER SWR")
                plt.grid()
                plt.legend()
                plt.show()
            else:
                # per event pre-post ratio not smoothed
                plt.plot(event_pre_post_ratio, c="r", label="not smoothed")
                plt.title("PRE-POST RATIO FOR EACH EVENT: PHMM" if (template_type=="phmm") else
                          "PRE-POST RATIO FOR EACH EVENT: ISING")
                plt.xlabel("SWR ID" if part_to_analyze=="nrem" else "REM PHASE ID")
                plt.ylabel("PRE-POST SIMILARITY")
                plt.ylim(-1, 1)
                plt.grid()
                plt.legend()
                plt.show()

                # per event pre-post ratio after z-scoring not smoothed
                plt.plot(event_pre_post_ratio_z, c="r", label="not smoothed")
                plt.title(
                    "PRE-POST RATIO FOR EACH EVENT: PHMM\n Z-SCORED TO SELECT WINNER" if (template_type == "phmm") else
                    "PRE-POST RATIO FOR EACH EVENT: ISING\n Z-SCORED TO SELECT WINNER")
                plt.xlabel("SWR ID" if part_to_analyze == "nrem" else "REM PHASE ID")
                plt.ylabel("PRE-POST SIMILARITY")
                plt.ylim(-1, 1)
                plt.grid()
                plt.legend()
                plt.show()

                # event length not smoothed
                plt.plot(event_len_seq, label="not smoothed")
                plt.title("EVENT LENGTH")
                plt.xlabel("SWR ID" if part_to_analyze == "nrem" else "REM PHASE ID")
                plt.ylabel("#POP.VEC. PER SWR")
                plt.grid()
                plt.legend()
                plt.show()

            # per population vector data
            # ----------------------------------------------------------------------------------------------------------

            # per population vector ratio smoothed
            plt.plot(pop_vec_pre_post_ratio_smooth, label="n_mov_avg = " + str(n_moving_average_pop_vec))
            plt.title("PRE-POST RATIO FOR EACH POP. VECTOR: PHMM" if (template_type=="phmm") else
                      "PRE-POST RATIO FOR EACH POP. VECTOR: ISING")
            plt.xlabel("POP.VEC. ID")
            plt.ylabel("PRE-POST SIMILARITY")
            plt.ylim(-1, 1)
            plt.grid()
            plt.legend()
            plt.show()

            # per mode / per spatial bin data
            # ----------------------------------------------------------------------------------------------------------

            # occurrence of modes/spatial bins of PRE
            plt.imshow(occurence_modes_pre, interpolation='nearest', aspect='auto')
            plt.ylabel("MODE ID" if (template_type=="phmm") else "SPATIAL BIN")
            plt.xlabel("WINDOW ID")
            a = plt.colorbar()
            a.set_label("#WINS/" + str(nr_pop_vec) + " POP. VEC. WINDOW")
            plt.title("PRE: OCCURENCE (#WINS) OF MODES IN \nWINDOWS OF FIXED LENGTH" if (template_type=="phmm") else
                      "PRE: OCCURENCE (#WINS) OF SPATIAL BINS IN \nWINDOWS OF FIXED LENGTH")
            plt.show()

            # occurrence of modes/spatial bins of PRE
            plt.imshow(occurence_modes_post, interpolation='nearest', aspect='auto')
            plt.ylabel("MODE ID" if (template_type=="phmm") else "SPATIAL BIN")
            plt.xlabel("WINDOW ID")
            a = plt.colorbar()
            a.set_label("#WINS/" + str(nr_pop_vec) + " POP. VEC. WINDOW")
            plt.title("POST: OCCURENCE (#WINS) OF MODES IN \nWINDOWS OF FIXED LENGTH" if (template_type=="phmm") else
                      "POST: OCCURENCE (#WINS) OF SPATIAL BINS IN \nWINDOWS OF FIXED LENGTH")
            plt.show()

            # mode/spatial bin - score assignment: PRE
            plt.errorbar(range(pre_lh_list[0].shape[1]), mode_score_mean_pre, yerr=mode_score_std_pre,
                         linestyle="")
            plt.scatter(range(pre_lh_list[0].shape[1]), mode_score_mean_pre)
            plt.title("PRE-POST SCORE PER MODE: PRE" if (template_type=="phmm") else "PRE-POST SCORE PER SPATIAL BIN: PRE")
            plt.xlabel("MODE ID" if (template_type=="phmm") else "SPATIAL BIN ID")
            plt.ylabel("PRE-POST SCORE: MEAN AND STD")
            plt.show()

            # mode/spatial bin - score assignment: POST
            plt.errorbar(range(post_lh_list[0].shape[1]), mode_score_mean_post, yerr=mode_score_std_post,
                         linestyle="")
            plt.scatter(range(post_lh_list[0].shape[1]), mode_score_mean_post)
            plt.title("PRE-POST SCORE PER MODE: POST" if (template_type=="phmm") else "PRE-POST SCORE PER SPATIAL BIN: POST")
            plt.xlabel("MODE ID" if (template_type=="phmm") else "SPATIAL BIN ID")
            plt.ylabel("PRE-POST SCORE: MEAN AND STD")
            plt.show()

    def plot_likelihoods(self, template_type="phmm", part_to_analyze="rem"):
        """
        Plots likelihoods from PRE/POST model during sleep

        :param template_type: "phmm" or "ising"
        :type template_type: str
        :param part_to_analyze: which part of sleep to analyze ("nrem", "rem")
        :type part_to_analyze: str
        """

        pre_prob, post_prob, event_times, swr_to_nrem = \
            self.decode_activity_using_pre_post(template_type=template_type, part_to_analyze=part_to_analyze)

        # use only second time bin to plot likelihoods
        pre_like = pre_prob[0][1, :]
        post_like = post_prob[0][1, :]

        n_col = 8
        scaler = 1
        plt.style.use('default')

        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(6, n_col)

        max_val = pre_like[:9].max()
        min_val = pre_like[:9].min()

        for i in range(n_col-1):
            ax1 = fig.add_subplot(gs[:, i])
            a = ax1.imshow(np.expand_dims(np.expand_dims(pre_like[i], 1),0), vmin=min_val, vmax=max_val, cmap="YlOrRd")
            ax1.set_xticks([])
            ax1.set_yticks([], [])
            ax1.set_xlabel(str(i))
            ax1.set_aspect(scaler)

        plt.tight_layout()
        ax1 = fig.add_subplot(gs[:, n_col - 1])
        fig.colorbar(a, cax=ax1)
        plt.rcParams['svg.fonttype'] = 'none'
        plt.show()

        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(6, n_col)

        max_val = post_like[:7].max()
        min_val = post_like[:7].min()

        for i in range(n_col-1):
            ax1 = fig.add_subplot(gs[:, i])
            a = ax1.imshow(np.expand_dims(np.expand_dims(post_like[i], 1),0), vmin=min_val, vmax=max_val, cmap="YlOrRd")
            ax1.set_xticks([])
            ax1.set_yticks([], [])
            ax1.set_xlabel(str(i))
            ax1.set_aspect(scaler)

        plt.tight_layout()
        ax1 = fig.add_subplot(gs[:, n_col - 1])
        fig.colorbar(a, cax=ax1)
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(os.path.join(save_path, "post_likeli.svg"), transparent="True")
        #plt.show()

    # </editor-fold>

    # <editor-fold desc="Sleep decoding analysis: one model">

    def decode_phmm_one_model(self, part_to_analyze, template_file_name, template_type ="phmm", speed_threshold=None,
                              plot_for_control=False, return_results=True, sleep_classification_method="std",
                              cells_to_use="all", shuffling=False, compression_factor=None, cell_selection="all",
                              compute_spike_bins_with_subsets=True):
        """
        decodes sleep activity using pHMM modes/spatial bins from ising model from before or after awake activity and
        computes similarity measure

        @param part_to_analyze: which data to analyse ("all_swr", "rem", "nrem")
        @type part_to_analyze: str
        @param speed_threshold: filter for rem/nrem phases --> all phases with speeds above this threshold are
        neglected, if you don't want to use a speed_threshold set to None.
        @type speed_threshold: int
        @param compression_factor: scaling factor between sleep activity and awake activity --> if awake
        model was build with constant time bin size (e.g. 100ms): compute how many spikes on average happen during this
        window size (e.g. mean = 30), check how many #spikes you used for constant #spike binning (e.g. 12) for the
        sleep data --> awake_activity_scaling_factor = 12/30 --> = 0.4
        if None --> is loaded from session specific parameter file
        @type compression_factor: float
        @param plot_for_control: plots intermediate results if True
        @type plot_for_control: bool
        @param sleep_classification_method: which sleep classification to use --> std: Jozsef's script, k_mean: Juan's script
        @type sleep_classification_method: str
        @param cells_to_use: which cells to use ("all", "stable", "inc", "dec")
        @type cells_to_use: str
        @param return_results: whether to return results or not
        @type return_results: bool
        @return: pre_prob (probabilities of PRE), post_prob (probabilities of ), event_times (in sec),
        swr_in_which_n_rem (assignement SWR - nrem phase)
        @rtype: list, list, list, numpy.array
        """

        # load speed threshold for REM/NREM from parameter file if not provided
        if speed_threshold is None:
            speed_threshold = self.session_params.sleep_phase_speed_threshold

        if template_type == "phmm":
            if sleep_classification_method == "std":
                print(" - Decoding "+self.experiment_phase+" ("+part_to_analyze + ") using PHMM modes & std. "
                                                                                  "sleep classification...\n")
                if cells_to_use == "stable":
                    if shuffling:
                        result_dir = "phmm_decoding/stable_cells_shuffled_"+self.params.stable_cell_method
                    else:
                        result_dir = "phmm_decoding/stable_cells_"+self.params.stable_cell_method
                if cells_to_use == "increasing":
                    result_dir = "phmm_decoding/inc_cells_"+self.params.stable_cell_method
                if cells_to_use == "decreasing":
                    result_dir = "phmm_decoding/dec_cells_"+self.params.stable_cell_method
                elif cells_to_use == "all":
                    if shuffling:
                        result_dir = "phmm_decoding/spike_shuffled"
                    else:
                        result_dir = "phmm_decoding"
            elif sleep_classification_method == "k_means":
                print(" - Decoding "+self.experiment_phase+" ("+part_to_analyze + ") using PHMM modes & k-means "
                                                                                  "sleep classification...\n")
                result_dir = "phmm_decoding/k_means_sleep_classification"

        elif template_type == "ising":
            print(" - Decoding "+self.experiment_phase+" ("+part_to_analyze + ") using Ising model ...\n")
            result_dir = "ising_glm_decoding"

        result_file_name = self.session_name +"_"+self.experiment_phase_id + "_"+\
                        part_to_analyze+"_"+ self.cell_type+"_one_model.npy"

        # Check if results exist already
        # ----------------------------------------------------------------------------------------------------------
        if os.path.isfile(self.params.pre_proc_dir + result_dir + "/" + result_file_name):
            print(" - Result exists already (" + result_file_name + ")\n")

        else:
            print(" - Decoding sleep ("+ part_to_analyze +") "+ template_file_name + ", using "+cells_to_use
                  + " cells ...\n")

            if cells_to_use == "stable":
                # load cell ids of stable cells
                # get stable, decreasing, increasing cells
                with open(self.params.pre_proc_dir + "cell_classification/" +
                          self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                          "rb") as f:
                    class_dic = pickle.load(f)

                cells_ids = class_dic["stable_cell_ids"]

            elif cells_to_use == "increasing":
                # load cell ids of stable cells

                with open(self.params.pre_proc_dir + "cell_classification/" +
                          self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                          "rb") as f:
                    class_dic = pickle.load(f)

                cells_ids = class_dic["increase_cell_ids"]

            elif cells_to_use == "decreasing":

                # load cell ids of stable cells

                with open(self.params.pre_proc_dir + "cell_classification/" +
                          self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                          "rb") as f:
                    class_dic = pickle.load(f)

                cells_ids = class_dic["decrease_cell_ids"]

            elif cells_to_use == "all":

                cells_ids = None

            # check if SWR or REM phases are supposed to be analyzed

            if part_to_analyze == "all_swr":

                # get SWR timings (in sec) & compute spike rasters (constant #spikes)
                # ------------------------------------------------------------------------------------------------------
                start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

                # convert to one array for event_spike_binning
                event_times = np.vstack((start_times, end_times, peak_times)).T

                # compute #spike binning for each event --> TODO: implement sliding window!
                if compute_spike_bins_with_subsets:
                    event_spike_rasters, event_spike_window_lenghts = \
                        PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                        whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1,
                                                                          cell_ids=cells_ids)
                else:
                    event_spike_rasters, event_spike_window_lenghts = \
                        PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                        whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)
                # do not need this info here (all SWR)
                swr_to_nrem = None

            elif part_to_analyze == "nrem":

                # get SWR timings (in sec) & compute spike rasters (constant #spikes)
                # ------------------------------------------------------------------------------------------------------
                start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

                # convert to one array for event_spike_binning
                event_times = np.vstack((start_times, end_times, peak_times)).T

                # only select SWR during nrem phases
                # get nrem phases in seconds
                n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                         classification_method=sleep_classification_method)
                swr_in_n_rem = np.zeros(event_times.shape[0])
                swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
                for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
                    n_rem_start = n_rem_phase[0]
                    n_rem_end = n_rem_phase[1]
                    for i, e_t in enumerate(event_times):
                        event_start = e_t[0]
                        event_end = e_t[1]
                        if (n_rem_start < event_start) and (event_end < n_rem_end):
                            swr_in_n_rem[i] += 1
                            swr_in_which_n_rem[n_rem_phase_id, i] = 1

                event_times = event_times[swr_in_n_rem == 1]
                # assignment: which SWR belongs to which nrem phase (for plotting)
                swr_to_nrem = swr_in_which_n_rem[:, swr_in_n_rem == 1]

                print(" - " + str(event_times.shape[0]) + " WERE ASSIGNED TO NREM PHASES\n")

                # do not need this info here
                swr_to_nrem = None

                start_times = event_times[:, 0]
                end_times = event_times[:,1]
                # compute #spike binning for each event --> TODO: implement sliding window!
                if compute_spike_bins_with_subsets:
                    event_spike_rasters, event_spike_window_lenghts = \
                        PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                        whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1,
                                                                          cell_ids=cells_ids)
                else:
                    event_spike_rasters, event_spike_window_lenghts = \
                        PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                        whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

            elif part_to_analyze == "rem":
                # get rem intervals in seconds
                event_times = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold,
                                                   classification_method=sleep_classification_method)

                print(" - " + str(event_times.shape[0]) + " REM phases found (speed thr.: " + str(
                    speed_threshold) + ")\n")

                # compute #spike binning for each event --> TODO: implement sliding window!
                if compute_spike_bins_with_subsets:
                    event_spike_rasters, event_spike_window_lenghts = \
                        PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                        whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1,
                                                                          cell_ids=cells_ids)
                else:
                    event_spike_rasters, event_spike_window_lenghts = \
                        PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                        whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

                # do not need this info here
                swr_to_nrem = None

                start_times = event_times[:, 0]
                end_times = event_times[:, 1]

            else:
                raise Exception("Provided part_to_analyze is not defined!")

            # plot raster and detected SWR, example spike rasters from SWRs
            if plot_for_control:
                # ------------------------------------------------------------------------------------------------------
                # plot detected events
                # ------------------------------------------------------------------------------------------------------
                # compute pre processed data
                self.compute_raster_speed_loc()
                to_plot = np.random.randint(0, start_times.shape[0], 5)
                for i in to_plot:
                    if cells_to_use == "all":
                        raster_for_plotting = self.raster
                    else:
                        raster_for_plotting =self.raster[cells_ids, :]
                    plt.imshow(raster_for_plotting, interpolation='nearest', aspect='auto')
                    plt.vlines(start_times[i] / self.params.time_bin_size, 0, raster_for_plotting.shape[0] - 1,
                               colors="r",
                               linewidth=0.5,
                               label="START")
                    plt.vlines(end_times[i] / self.params.time_bin_size, 0, raster_for_plotting.shape[0] - 1,
                               colors="g",
                               linewidth=0.5, label="END")
                    plt.xlim(start_times[i] / self.params.time_bin_size - 10,
                             end_times[i] / self.params.time_bin_size + 10)
                    plt.ylabel("CELL IDS")
                    plt.xlabel("TIME BINS / " + str(self.params.time_bin_size) + "s")
                    plt.legend()
                    a = plt.colorbar()
                    a.set_label("SPIKES PER BIN (" + str(self.params.time_bin_size) + " s)")
                    plt.title("EVENT ID " + str(i))
                    plt.show()

                    plt.imshow(event_spike_rasters[i], interpolation='nearest', aspect='auto')
                    a = plt.colorbar()
                    a.set_label("SPIKES PER CONST. #SPIKES BIN")
                    plt.xlabel("CONST. #SPIKE POP. VEC. ID")
                    plt.ylabel("CELL ID")
                    plt.title("BINNED EVENT ID " + str(i) + "\n#SPIKES PER BIN: " + str(
                        self.params.spikes_per_bin))
                    plt.show()

                # ------------------------------------------------------------------------------------------------------
                # compute length of constant #spike windows
                # ------------------------------------------------------------------------------------------------------
                event_spike_window_lengths_avg = np.mean(np.concatenate(event_spike_window_lenghts, axis=0))
                event_spike_window_lengths_median = np.median(np.concatenate(event_spike_window_lenghts, axis=0))

                y, x, _ = plt.hist(np.concatenate(event_spike_window_lenghts, axis=0), bins=30)
                plt.xlabel("LENGTH CONST #SPIKES POP. VEC / s")
                plt.ylabel("COUNTS")
                plt.title("POPULATION VECTOR LENGTH")
                plt.vlines(event_spike_window_lengths_avg, 0, y.max(), colors="r", label="MEAN: " +
                                                                                         str(np.round(
                                                                                             event_spike_window_lengths_avg,
                                                                                             2)))
                plt.vlines(event_spike_window_lengths_median, 0, y.max(), colors="b", label="MEDIAN: " +
                                                                                            str(np.round(
                                                                                                event_spike_window_lengths_median,
                                                                                                2)))
                plt.legend()
                plt.show()

            if template_type == "phmm":
                # load pHMM model
                with open(self.params.pre_proc_dir + "phmm/" + template_file_name + '.pkl', 'rb') as f:
                    model_dic = pickle.load(f)
                # get means of model (lambdas) for decoding
                mode_means = model_dic.means_

                # only select lambdas of cells to be used
                if cells_to_use in ["decreasing", "stable", "increasing"]:
                    mode_means = mode_means[:, cells_ids]

                # get time bin size at time of decoding
                time_bin_size_encoding = model_dic.time_bin_size

                # check if const. #spike bins are correct for the loaded compression factor
                if not self.params.spikes_per_bin == 12:
                    raise Exception("TRYING TO LOAD COMPRESSION FACTOR FOR 12 SPIKES PER BIN, "
                                    "BUT CURRENT #SPIKES PER BIN != 12")

                # load correct compression factor (as defined in parameter file of the session)
                if time_bin_size_encoding == 0.01:
                    compression_factor = np.round(self.session_params.sleep_compression_factor_12spikes_100ms * 10, 3)
                elif time_bin_size_encoding == 0.1:
                    compression_factor = self.session_params.sleep_compression_factor_12spikes_100ms
                else:
                    raise Exception("COMPRESSION FACTOR NEITHER PROVIDED NOR FOUND IN PARAMETER FILE")

                # if you want to use different compression factors for PRE/POST
                # if "PRE" in result_file_name:
                #     compression_factor = 0.4
                # elif "POST" in result_file_name:
                #     compression_factor = 0.6

                if shuffling:
                    print(" -- STARTED SWAPPING PROCEDURE ...")
                    # merge all events
                    conc_data = np.hstack(event_spike_rasters)
                    nr_swaps = conc_data.shape[1]*10
                    for shuffle_id in range(nr_swaps):
                        # select two random time bins
                        t1 = 1
                        t2 = 1
                        while(t1 == t2):
                            t1 = np.random.randint(conc_data.shape[1])
                            t2 = np.random.randint(conc_data.shape[1])
                        # check in both time bins which cells are active
                        act_cells_t1 = np.argwhere(conc_data[:, t1].flatten()>0).flatten()
                        act_cells_t2 = np.argwhere(conc_data[:, t2].flatten()>0).flatten()
                        # find intersect (same cells need to be firing in t1 and t2 in order to exchange
                        # spikes
                        # original code
                        # --------------------------------------------------------------------------------------
                        # cells_firing_in_both = np.intersect1d(act_cells_t1, act_cells_t2)
                        # if cells_firing_in_both.shape[0] > 1:
                        #     # select first cell to swap
                        #     cell_1 = 1
                        #     cell_2 = 1
                        #     while (cell_1 == cell_2):
                        #         cell_1 = np.random.choice(cells_firing_in_both)
                        #         cell_2 = np.random.choice(cells_firing_in_both)
                        #     # do the actual swapping
                        #     conc_data[cell_1, t1] += 1
                        #     conc_data[cell_1, t2] -= 1
                        #     conc_data[cell_2, t1] -= 1
                        #     conc_data[cell_2, t2] += 1

                        if act_cells_t1.shape[0] > 1 and act_cells_t2.shape[0] > 1:
                            # select first cell to swap
                            cell_1 = 1
                            cell_2 = 1
                            while (cell_1 == cell_2):
                                cell_1 = np.random.choice(act_cells_t2)
                                cell_2 = np.random.choice(act_cells_t1)
                            # do the actual swapping
                            conc_data[cell_1, t1] += 1
                            conc_data[cell_1, t2] -= 1
                            conc_data[cell_2, t1] -= 1
                            conc_data[cell_2, t2] += 1

                    print(" -- ... DONE!")
                    # split data again into list
                    event_lengths = [x.shape[1] for x in event_spike_rasters]

                    event_spike_rasters_shuffled = []
                    start = 0
                    for el in event_lengths:
                        event_spike_rasters_shuffled.append(conc_data[:,start:start+el])
                        start = el

                    event_spike_rasters = event_spike_rasters_shuffled
                # start with actual decoding
                # ----------------------------------------------------------------------------------------------

                print(" - DECODING USING "+ cells_to_use + " CELLS")

                if not compute_spike_bins_with_subsets:
                    # event_spike_raster size and mode_means size don't match --> need to go through event_spike_rasters
                    # to only select cells that are wanted
                    event_spike_rasters_modified = []
                    for ev_r in event_spike_rasters:
                        event_spike_rasters_modified.append(ev_r[cells_ids, :])

                    event_spike_rasters = event_spike_rasters_modified

                results_list = decode_using_phmm_modes(mode_means=mode_means,
                                                       event_spike_rasters=event_spike_rasters,
                                                       compression_factor=compression_factor,
                                                       cell_selection=cell_selection)
            elif template_type == "ising":
                # load ising template
                with open(self.params.pre_proc_dir + 'awake_ising_maps/' + template_file_name + '.pkl',
                          'rb') as f:
                    model_dic = pickle.load(f)

                # if compression_factor is not provided --> load from parameter file
                if compression_factor is None:
                    # get time_bin_size of encoding
                    time_bin_size_encoding = model_dic["time_bin_size"]

                    # check if const. #spike bins are correct for the loaded compression factor
                    if not self.params.spikes_per_bin == 12:
                        raise Exception("TRYING TO LOAD COMPRESSION FACTOR FOR 12 SPIKES PER BIN, "
                                        "BUT CURRENT #SPIKES PER BIN != 12")

                    # load correct compression factor (as defined in parameter file of the session)
                    if time_bin_size_encoding == 0.01:
                        compression_factor = \
                            np.round(self.session_params.sleep_compression_factor_12spikes_100ms * 10, 3)
                    elif time_bin_size_encoding == 0.1:
                        compression_factor = self.session_params.sleep_compression_factor_12spikes_100ms
                    else:
                        raise Exception("COMPRESSION FACTOR NEITHER PROVIDED NOR FOUND IN PARAMETER FILE")

                # get template map
                template_map = model_dic["res_map"]

                results_list = decode_using_ising_map(template_map=template_map, event_spike_rasters=event_spike_rasters,
                                                               compression_factor=compression_factor,
                                                               cell_selection="all")

            # plot maps of some SWR for control
            if plot_for_control:
                swr_to_plot = []
                n_swr = 0
                while (len(swr_to_plot) < 10):
                    if results_list[n_swr].shape[0]>0:
                        swr_to_plot.append(n_swr)
                    n_swr += 1

                for swr in swr_to_plot:
                    res = results_list[swr]
                    plt.imshow(res.T, interpolation='nearest', aspect='auto')
                    plt.xlabel("POP.VEC. ID")
                    plt.ylabel("MODE ID")
                    a = plt.colorbar()
                    a.set_label("PROBABILITY")
                    plt.title("PROBABILITY MAP, SWR ID: " + str(swr))
                    plt.show()

            # saving results
            # --------------------------------------------------------------------------------------------------
            # create dictionary with results
            result_post = {
                "results_list": results_list,
                "event_times": event_times,
                "swr_to_nrem": swr_to_nrem
            }
            outfile = open(self.params.pre_proc_dir + result_dir +"/" + result_file_name, 'wb')
            print("  - saving new results ...")
            pickle.dump(result_post, outfile)
            outfile.close()
            print("  - ... done!\n")
        if return_results:

            while True:
                # load decoded maps
                try:
                    result = pickle.load(open(self.params.pre_proc_dir+result_dir + "/" + result_file_name,"rb"))
                    break
                except:
                    continue

            likelihood = result["results_list"]
            event_times = result["event_times"]
            swr_to_nrem = result["swr_to_nrem"]
            try:
                most_likely_state_sequence = result["most_likely_state_sequence"]
            except:
                most_likely_state_sequence = []

            return likelihood, event_times, swr_to_nrem, most_likely_state_sequence

    def decode_phmm_one_model_cell_subset(self, part_to_analyze, template_file_name, speed_threshold=None,
                              plot_for_control=False, return_results=True, file_extension="PRE",
                              cells_to_use="stable", shuffling=False, cell_selection="all",
                              compute_spike_bins_with_subsets=True, sleep_classification_method="std"):
        """
        decodes sleep activity using pHMM modes/spatial bins from ising model from before and after awake activity and
        computes similarity measure

        @param part_to_analyze: which data to analyse ("all_swr", "rem", "nrem")
        @type part_to_analyze: str
        @param speed_threshold: filter for rem/nrem phases --> all phases with speeds above this threshold are
        neglected, if you don't want to use a speed_threshold set to None.
        @type speed_threshold: int
        @param compression_factor: scaling factor between sleep activity and awake activity --> if awake
        model was build with constant time bin size (e.g. 100ms): compute how many spikes on average happen during this
        window size (e.g. mean = 30), check how many #spikes you used for constant #spike binning (e.g. 12) for the
        sleep data --> awake_activity_scaling_factor = 12/30 --> = 0.4
        if None --> is loaded from session specific parameter file
        @type compression_factor: float
        @param plot_for_control: plots intermediate results if True
        @type plot_for_control: bool
        @param sleep_classification_method: which sleep classification to use --> std: Jozsef's script, k_mean: Juan's script
        @type sleep_classification_method: str
        @param cells_to_use: which cells to use ("all", "stable", "inc", "dec")
        @type cells_to_use: str
        @param return_results: whether to return results or not
        @type return_results: bool
        @return: pre_prob (probabilities of PRE), post_prob (probabilities of ), event_times (in sec),
        swr_in_which_n_rem (assignement SWR - nrem phase)
        @rtype: list, list, list, numpy.array
        """

        # load speed threshold for REM/NREM from parameter file if not provided
        if speed_threshold is None:
            speed_threshold = self.session_params.sleep_phase_speed_threshold

        print(" - Decoding "+self.experiment_phase+" ("+part_to_analyze + ") using PHMM modes & std. "
                                                                          "sleep classification...\n")
        if cells_to_use == "stable":
            result_dir = "phmm_decoding/stable_subset/"
        if cells_to_use == "increasing":
            result_dir = "phmm_decoding/increasing_subset/"
        if cells_to_use == "decreasing":
            result_dir = "phmm_decoding/decreasing_subset/"

        result_file_name = self.session_name +"_"+self.experiment_phase_id + "_"+\
                        part_to_analyze+"_"+ self.cell_type+"_"+file_extension

        # Check if results exist already
        # ----------------------------------------------------------------------------------------------------------
        if os.path.isfile(self.params.pre_proc_dir + result_dir + "/" + result_file_name):
            print(" - Result exists already (" + result_file_name + ")\n")

        else:
            print(" - Decoding sleep ("+ part_to_analyze +") "+ template_file_name + ", using "+cells_to_use
                  + " cells ...\n")

            if cells_to_use == "stable":
                # load cell ids of stable cells
                # get stable, decreasing, increasing cells
                with open(self.params.pre_proc_dir + "cell_classification/" +
                          self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                          "rb") as f:
                    class_dic = pickle.load(f)

                cells_ids = class_dic["stable_cell_ids"]

            elif cells_to_use == "increasing":
                # load cell ids of stable cells

                with open(self.params.pre_proc_dir + "cell_classification/" +
                          self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                          "rb") as f:
                    class_dic = pickle.load(f)

                cells_ids = class_dic["increase_cell_ids"]

            elif cells_to_use == "decreasing":

                # load cell ids of stable cells

                with open(self.params.pre_proc_dir + "cell_classification/" +
                          self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                          "rb") as f:
                    class_dic = pickle.load(f)

                cells_ids = class_dic["decrease_cell_ids"]

            # check if SWR or REM phases are supposed to be analyzed

            if part_to_analyze == "all_swr":

                # get SWR timings (in sec) & compute spike rasters (constant #spikes)
                # ------------------------------------------------------------------------------------------------------
                start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

                # convert to one array for event_spike_binning
                event_times = np.vstack((start_times, end_times, peak_times)).T

                # compute #spike binning for each event --> TODO: implement sliding window!
                if compute_spike_bins_with_subsets:
                    event_spike_rasters, event_spike_window_lenghts = \
                        PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                        whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1,
                                                                          cell_ids=cells_ids)
                else:
                    event_spike_rasters, event_spike_window_lenghts = \
                        PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                        whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)
                # do not need this info here (all SWR)
                swr_to_nrem = None

            elif part_to_analyze == "nrem":

                # get SWR timings (in sec) & compute spike rasters (constant #spikes)
                # ------------------------------------------------------------------------------------------------------
                start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

                # convert to one array for event_spike_binning
                event_times = np.vstack((start_times, end_times, peak_times)).T

                # only select SWR during nrem phases
                # get nrem phases in seconds
                n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                         classification_method=sleep_classification_method)
                swr_in_n_rem = np.zeros(event_times.shape[0])
                swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
                for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
                    n_rem_start = n_rem_phase[0]
                    n_rem_end = n_rem_phase[1]
                    for i, e_t in enumerate(event_times):
                        event_start = e_t[0]
                        event_end = e_t[1]
                        if (n_rem_start < event_start) and (event_end < n_rem_end):
                            swr_in_n_rem[i] += 1
                            swr_in_which_n_rem[n_rem_phase_id, i] = 1

                event_times = event_times[swr_in_n_rem == 1]
                # assignment: which SWR belongs to which nrem phase (for plotting)
                swr_to_nrem = swr_in_which_n_rem[:, swr_in_n_rem == 1]

                print(" - " + str(event_times.shape[0]) + " WERE ASSIGNED TO NREM PHASES\n")

                # do not need this info here
                swr_to_nrem = None

                start_times = event_times[:, 0]
                end_times = event_times[:,1]
                # compute #spike binning for each event --> TODO: implement sliding window!

                event_spike_rasters, event_spike_window_lenghts = \
                        PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                        whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1,
                                                                          cell_ids=cells_ids)

            elif part_to_analyze == "rem":
                # get rem intervals in seconds
                event_times = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold,
                                                   classification_method=sleep_classification_method)

                print(" - " + str(event_times.shape[0]) + " REM phases found (speed thr.: " + str(
                    speed_threshold) + ")\n")

                # compute #spike binning for each event --> TODO: implement sliding window!

                event_spike_rasters, event_spike_window_lenghts = \
                        PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                        whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1,
                                                                          cell_ids=cells_ids)


                # do not need this info here
                swr_to_nrem = None

                start_times = event_times[:, 0]
                end_times = event_times[:, 1]

            else:
                raise Exception("Provided part_to_analyze is not defined!")

            # plot raster and detected SWR, example spike rasters from SWRs
            if plot_for_control:
                # ------------------------------------------------------------------------------------------------------
                # plot detected events
                # ------------------------------------------------------------------------------------------------------
                # compute pre processed data
                self.compute_raster_speed_loc()
                to_plot = np.random.randint(0, start_times.shape[0], 5)
                for i in to_plot:
                    if cells_to_use == "all":
                        raster_for_plotting = self.raster
                    else:
                        raster_for_plotting =self.raster[cells_ids, :]
                    plt.imshow(raster_for_plotting, interpolation='nearest', aspect='auto')
                    plt.vlines(start_times[i] / self.params.time_bin_size, 0, raster_for_plotting.shape[0] - 1,
                               colors="r",
                               linewidth=0.5,
                               label="START")
                    plt.vlines(end_times[i] / self.params.time_bin_size, 0, raster_for_plotting.shape[0] - 1,
                               colors="g",
                               linewidth=0.5, label="END")
                    plt.xlim(start_times[i] / self.params.time_bin_size - 10,
                             end_times[i] / self.params.time_bin_size + 10)
                    plt.ylabel("CELL IDS")
                    plt.xlabel("TIME BINS / " + str(self.params.time_bin_size) + "s")
                    plt.legend()
                    a = plt.colorbar()
                    a.set_label("SPIKES PER BIN (" + str(self.params.time_bin_size) + " s)")
                    plt.title("EVENT ID " + str(i))
                    plt.show()

                    plt.imshow(event_spike_rasters[i], interpolation='nearest', aspect='auto')
                    a = plt.colorbar()
                    a.set_label("SPIKES PER CONST. #SPIKES BIN")
                    plt.xlabel("CONST. #SPIKE POP. VEC. ID")
                    plt.ylabel("CELL ID")
                    plt.title("BINNED EVENT ID " + str(i) + "\n#SPIKES PER BIN: " + str(
                        self.params.spikes_per_bin))
                    plt.show()

                # ------------------------------------------------------------------------------------------------------
                # compute length of constant #spike windows
                # ------------------------------------------------------------------------------------------------------
                event_spike_window_lengths_avg = np.mean(np.concatenate(event_spike_window_lenghts, axis=0))
                event_spike_window_lengths_median = np.median(np.concatenate(event_spike_window_lenghts, axis=0))

                y, x, _ = plt.hist(np.concatenate(event_spike_window_lenghts, axis=0), bins=30)
                plt.xlabel("LENGTH CONST #SPIKES POP. VEC / s")
                plt.ylabel("COUNTS")
                plt.title("POPULATION VECTOR LENGTH")
                plt.vlines(event_spike_window_lengths_avg, 0, y.max(), colors="r", label="MEAN: " +
                                                                                         str(np.round(
                                                                                             event_spike_window_lengths_avg,
                                                                                             2)))
                plt.vlines(event_spike_window_lengths_median, 0, y.max(), colors="b", label="MEDIAN: " +
                                                                                            str(np.round(
                                                                                                event_spike_window_lengths_median,
                                                                                                2)))
                plt.legend()
                plt.show()

            # load pHMM model
            if cells_to_use == "stable":
                with open(self.params.pre_proc_dir + "phmm/stable_cells/" + template_file_name + '.pkl', 'rb') as f:
                    model_dic = pickle.load(f)
            elif cells_to_use == "decreasing":
                with open(self.params.pre_proc_dir + "phmm/decreasing_cells/" + template_file_name + '.pkl', 'rb') as f:
                    model_dic = pickle.load(f)
            elif cells_to_use == "increasing":
                with open(self.params.pre_proc_dir + "phmm/increasing_cells/" + template_file_name + '.pkl', 'rb') as f:
                    model_dic = pickle.load(f)
            # get means of model (lambdas) for decoding
            mode_means = model_dic.means_

            # get time bin size at time of decoding
            time_bin_size_encoding = model_dic.time_bin_size

            # check if const. #spike bins are correct for the loaded compression factor
            if not self.params.spikes_per_bin == 12:
                raise Exception("TRYING TO LOAD COMPRESSION FACTOR FOR 12 SPIKES PER BIN, "
                                "BUT CURRENT #SPIKES PER BIN != 12")

            # load correct compression factor (as defined in parameter file of the session)
            if time_bin_size_encoding == 0.01:
                compression_factor = np.round(self.session_params.sleep_compression_factor_12spikes_100ms * 10, 3)
            elif time_bin_size_encoding == 0.1:
                if cells_to_use == "stable":
                    compression_factor = self.session_params.sleep_compression_factor_12spikes_100ms_stable_cells
                elif cells_to_use == "decreasing":
                    compression_factor = self.session_params.sleep_compression_factor_12spikes_100ms_decreasing_cells
                elif cells_to_use == "increasing":
                    compression_factor = self.session_params.sleep_compression_factor_12spikes_100ms_increasing_cells
            else:
                raise Exception("COMPRESSION FACTOR NEITHER PROVIDED NOR FOUND IN PARAMETER FILE")

            # if you want to use different compression factors for PRE/POST
            # if "PRE" in result_file_name:
            #     compression_factor = 0.4
            # elif "POST" in result_file_name:
            #     compression_factor = 0.6

            if shuffling:
                print(" -- STARTED SWAPPING PROCEDURE ...")
                # merge all events
                conc_data = np.hstack(event_spike_rasters)
                nr_swaps = conc_data.shape[1]*10
                for shuffle_id in range(nr_swaps):
                    # select two random time bins
                    t1 = 1
                    t2 = 1
                    while(t1 == t2):
                        t1 = np.random.randint(conc_data.shape[1])
                        t2 = np.random.randint(conc_data.shape[1])
                    # check in both time bins which cells are active
                    act_cells_t1 = np.argwhere(conc_data[:, t1].flatten()>0).flatten()
                    act_cells_t2 = np.argwhere(conc_data[:, t2].flatten()>0).flatten()
                    # find intersect (same cells need to be firing in t1 and t2 in order to exchange
                    # spikes
                    # original code
                    # --------------------------------------------------------------------------------------
                    # cells_firing_in_both = np.intersect1d(act_cells_t1, act_cells_t2)
                    # if cells_firing_in_both.shape[0] > 1:
                    #     # select first cell to swap
                    #     cell_1 = 1
                    #     cell_2 = 1
                    #     while (cell_1 == cell_2):
                    #         cell_1 = np.random.choice(cells_firing_in_both)
                    #         cell_2 = np.random.choice(cells_firing_in_both)
                    #     # do the actual swapping
                    #     conc_data[cell_1, t1] += 1
                    #     conc_data[cell_1, t2] -= 1
                    #     conc_data[cell_2, t1] -= 1
                    #     conc_data[cell_2, t2] += 1

                    if act_cells_t1.shape[0] > 1 and act_cells_t2.shape[0] > 1:
                        # select first cell to swap
                        cell_1 = 1
                        cell_2 = 1
                        while (cell_1 == cell_2):
                            cell_1 = np.random.choice(act_cells_t2)
                            cell_2 = np.random.choice(act_cells_t1)
                        # do the actual swapping
                        conc_data[cell_1, t1] += 1
                        conc_data[cell_1, t2] -= 1
                        conc_data[cell_2, t1] -= 1
                        conc_data[cell_2, t2] += 1

                print(" -- ... DONE!")
                # split data again into list
                event_lengths = [x.shape[1] for x in event_spike_rasters]

                event_spike_rasters_shuffled = []
                start = 0
                for el in event_lengths:
                    event_spike_rasters_shuffled.append(conc_data[:,start:start+el])
                    start = el

                event_spike_rasters = event_spike_rasters_shuffled
            # start with actual decoding
            # ----------------------------------------------------------------------------------------------

            results_list = decode_using_phmm_modes(mode_means=mode_means,
                                                   event_spike_rasters=event_spike_rasters,
                                                   compression_factor=compression_factor,
                                                   cell_selection=cell_selection)

            # plot maps of some SWR for control
            if plot_for_control:
                swr_to_plot = []
                n_swr = 0
                while (len(swr_to_plot) < 10):
                    if results_list[n_swr].shape[0]>0:
                        swr_to_plot.append(n_swr)
                    n_swr += 1

                for swr in swr_to_plot:
                    res = results_list[swr]
                    plt.imshow(res.T, interpolation='nearest', aspect='auto')
                    plt.xlabel("POP.VEC. ID")
                    plt.ylabel("MODE ID")
                    a = plt.colorbar()
                    a.set_label("PROBABILITY")
                    plt.title("PROBABILITY MAP, SWR ID: " + str(swr))
                    plt.show()

            # saving results
            # --------------------------------------------------------------------------------------------------
            # create dictionary with results
            result_post = {
                "results_list": results_list,
                "event_times": event_times,
                "swr_to_nrem": swr_to_nrem
            }
            outfile = open(self.params.pre_proc_dir + result_dir +"/" + result_file_name, 'wb')
            print("  - saving new results ...")
            pickle.dump(result_post, outfile)
            outfile.close()
            print("  - ... done!\n")
        if return_results:

            while True:
                # load decoded maps
                try:
                    result = pickle.load(open(self.params.pre_proc_dir+result_dir + "/" + result_file_name,"rb"))
                    break
                except:
                    continue

            likelihood = result["results_list"]
            event_times = result["event_times"]
            swr_to_nrem = result["swr_to_nrem"]
            try:
                most_likely_state_sequence = result["most_likely_state_sequence"]
            except:
                most_likely_state_sequence = []

            return likelihood, event_times, swr_to_nrem, most_likely_state_sequence

    # </editor-fold>

    # <editor-fold desc="Sleep decoding analysis: compression factor analysis">

    def optimize_compression_factor(self, phmm_file_name, result_file):

        print(" - COMPRESSION FACTOR OPTIMIZATION ...\n")

        # get SWR timings (in sec) & compute spike rasters (constant #spikes)
        # ------------------------------------------------------------------------------------------------------
        start_times, end_times, peak_times = self.detect_swr()

        # convert to one array for event_spike_binning
        event_times = np.vstack((start_times, end_times, peak_times)).T

        # only select SWR during nrem phases
        # get nrem phases in seconds
        n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem")
        swr_in_n_rem = np.zeros(event_times.shape[0])
        swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
        for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
            n_rem_start = n_rem_phase[0]
            n_rem_end = n_rem_phase[1]
            for i, e_t in enumerate(event_times):
                event_start = e_t[0]
                event_end = e_t[1]
                if (n_rem_start < event_start) and (event_end < n_rem_end):
                    swr_in_n_rem[i] += 1
                    swr_in_which_n_rem[n_rem_phase_id, i] = 1

        event_times = event_times[swr_in_n_rem == 1]

        print(" - " + str(event_times.shape[0]) + " WERE ASSIGNED TO NREM PHASES\n")

        # compute #spike binning for each event
        event_spike_rasters, event_spike_window_lenghts = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

        # load pHMM model
        with open(self.params.pre_proc_dir + "phmm/" + phmm_file_name + '.pkl', 'rb') as f:
            model_dic = pickle.load(f)
        # get means of model (lambdas) for decoding
        mode_means = model_dic.means_

        scaling_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        max_med_list = []
        glob_med_list = []
        for awake_activity_scaling_factor in scaling_factors:

            results_list = self.decode_using_phmm_modes(mode_means=mode_means,
                                                        event_spike_rasters=event_spike_rasters,
                                                        awake_activity_scaling_factor=
                                                        awake_activity_scaling_factor)
            prob_arr = np.vstack(results_list)
            prob_arr_flat = prob_arr.flatten()
            prob_arr_z = zscore(prob_arr_flat)

            glob_med_list.append(np.mean(prob_arr_flat[prob_arr_z < 3]))

            max_prob_arr = np.max(prob_arr, axis=1)
            max_prob_arr_z = zscore(max_prob_arr)
            max_med_list.append(np.mean(max_prob_arr[max_prob_arr_z < 3]))

        results = {
            "scaling_factors": scaling_factors,
            "glob_med_list": glob_med_list,
            "max_med_list": max_med_list
        }
        outfile = open(self.params.pre_proc_dir +"compression_optimization/" + result_file+".pkl", 'wb')
        pickle.dump(results, outfile)

    def optimize_compression_factor_plot_results(self, result_file_pre, result_file_post):

        for i, (result_file, name) in enumerate(zip([result_file_pre, result_file_post], ["PRE", "POST"])):
            results = np.load(self.params.pre_proc_dir +"compression_optimization/"+result_file+".pkl",
                              allow_pickle=True)

            scaling_factors = results["scaling_factors"]
            glob_med_list = results["glob_med_list"]
            max_med_list = results["max_med_list"]
            plt.subplot(2,1,i+1)
            plt.plot(scaling_factors, max_med_list, ".-", c="blue", label="mean(max.per.vec)")
            plt.legend()
            plt.ylabel("PROB.")
            plt.grid()
        plt.xlabel("SCALING FACTOR")
        plt.show()

        for i, (result_file, name) in enumerate(zip([result_file_pre, result_file_post], ["PRE", "POST"])):
            results = np.load(self.params.pre_proc_dir +"compression_optimization/"+result_file+".pkl",
                              allow_pickle=True)

            scaling_factors = results["scaling_factors"]
            glob_med_list = results["glob_med_list"]
            max_med_list = results["max_med_list"]
            plt.subplot(2,1,i+1)
            plt.plot(scaling_factors, glob_med_list, ".-", c="r", label="global_mean - "+name)
            plt.legend()
            plt.ylabel("PROB.")
            plt.grid()
        plt.xlabel("SCALING FACTOR")
        plt.show()

    def check_compression_factor(self, phmm_file_pre, phmm_file_post):

        print(" - COMPRESSION FACTOR OPTIMIZATION ...\n")

        # get SWR timings (in sec) & compute spike rasters (constant #spikes)
        # ------------------------------------------------------------------------------------------------------
        start_times, end_times, peak_times = self.detect_swr()

        # convert to one array for event_spike_binning
        event_times = np.vstack((start_times, end_times, peak_times)).T

        # only select SWR during nrem phases
        # get nrem phases in seconds
        n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem")
        swr_in_n_rem = np.zeros(event_times.shape[0])
        swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
        for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
            n_rem_start = n_rem_phase[0]
            n_rem_end = n_rem_phase[1]
            for i, e_t in enumerate(event_times):
                event_start = e_t[0]
                event_end = e_t[1]
                if (n_rem_start < event_start) and (event_end < n_rem_end):
                    swr_in_n_rem[i] += 1
                    swr_in_which_n_rem[n_rem_phase_id, i] = 1

        event_times = event_times[swr_in_n_rem == 1]

        print(" - " + str(event_times.shape[0]) + " WERE ASSIGNED TO NREM PHASES\n")

        # compute #spike binning for each event --> TODO: implement sliding window!
        event_spike_rasters, event_spike_window_lenghts = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

        # load pHMM model
        with open(self.params.pre_proc_dir + "phmm/" + phmm_file_pre + '.pkl', 'rb') as f:
            model_dic = pickle.load(f)
        # get means of model (lambdas) for decoding
        mode_means = model_dic.means_

        for awake_activity_scaling_factor in [0.4]:

            results_list = decode_using_phmm_modes(mode_means=mode_means, event_spike_rasters=event_spike_rasters,
                                                   awake_activity_scaling_factor=awake_activity_scaling_factor)
            pre_prob_arr = np.vstack(results_list)

        # load pHMM model
        with open(self.params.pre_proc_dir + "phmm/" + phmm_file_post + '.pkl', 'rb') as f:
            model_dic = pickle.load(f)
        # get means of model (lambdas) for decoding
        mode_means = model_dic.means_

        post_lh_list = []
        for awake_activity_scaling_factor in [0.4, 0.5]:

            results_list = self.decode_using_phmm_modes(mode_means=mode_means,
                                                        event_spike_rasters=event_spike_rasters,
                                                        awake_activity_scaling_factor=
                                                        awake_activity_scaling_factor)
            post_lh_list.append(np.vstack(results_list))

        post_0_4 = post_lh_list[0]
        post_0_5 = post_lh_list[1]

        results = {
            "post_0_4": post_0_4,
            "post_0_5": post_0_5,
            "pre_prob_arr": pre_prob_arr
        }
        outfile = open(self.params.pre_proc_dir + "compression_optimization/" + "test" + ".pkl", 'wb')
        pickle.dump(results, outfile)

    def check_compression_factor_plot(self, result_file_name):
        results = np.load(self.params.pre_proc_dir +"compression_optimization/"+result_file_name+".pkl",
                          allow_pickle=True)

        pre = results["pre_prob_arr"]
        post_04 = results["post_0_4"]
        post_05 = results["post_0_5"]

        pre_max = np.max(pre, axis=1)
        post_04_max = np.max(post_04, axis=1)
        post_05_max = np.max(post_05, axis=1)


        pre_post_ratio_04 = (post_04_max - pre_max)/(post_04_max + pre_max)
        pre_post_ratio_05 = (post_05_max - pre_max)/(post_05_max + pre_max)
        p_p_r_s_04 = moving_average(a=pre_post_ratio_04, n=40)
        p_p_r_s_05 = moving_average(a=pre_post_ratio_05, n=40)

        plt.plot(p_p_r_s_04, label="POST_04")
        plt.plot(p_p_r_s_05, c="r", alpha=0.5, label="POST_05")
        plt.legend()
        plt.xlabel("POP. VEC ID")
        plt.ylabel("PRE_POST_RATIO")
        plt.show()

        plt.plot(post_04_max, label="POST_04, AVG: "+str(np.mean(post_04_max)))
        plt.plot(post_05_max, c="r", alpha=0.5, label="POST_05, AVG: "+str(np.mean(post_05_max)))
        plt.title("MAX. POST LIKELIHOOD")
        plt.xlabel("POP. VEC. ID")
        plt.ylabel("MAX. LIKELIHOOD")
        plt.legend()
        plt.show()
        nom_04 = post_04_max - pre_max
        nom_05 = post_05_max - pre_max

        plt.plot(pre_max, c="r", alpha=0.5, label="PRE, AVG: "+str(np.mean(pre_max)))
        plt.title("MAX. POST LIKELIHOOD")
        plt.xlabel("POP. VEC. ID")
        plt.ylabel("MAX. LIKELIHOOD")
        plt.legend()
        plt.show()
        exit()


        # exit()
        # plt.plot(post_04_max - pre_max, label="POST_04, AVG: "+str(np.mean(post_04_max - pre_max)))
        # plt.plot(post_05_max - pre_max, c="r", alpha=0.5, label="POST_05, AVG: "+str(np.mean(post_05_max - pre_max)))
        # plt.title("NOMINATOR")
        # plt.legend()
        # plt.show()

        denom_04 = post_04_max + pre_max
        denom_05 = post_05_max + pre_max

        a = nom_04/denom_04
        ne = moving_average(a=a, n=40)
        plt.plot(ne, label="POST_04")
        a = nom_05/denom_04
        ne = moving_average(a=a, n=40)
        plt.plot(ne, label="POST_05",c="r", alpha=0.5)
        plt.legend()
        plt.show()

        # exit()
        #
        # a = denom_04-denom_05
        # plt.hist(a, bins=100)
        # plt.show()
        # exit()

        plt.hist(nom_04, bins=10000, density=True, label="POST_04")
        plt.hist(nom_05, bins=10000, color="red", alpha=0.5, density=True, label="POST_05")
        plt.xlim(-0.05e-13, 0.1e-13)
        plt.legend()
        plt.xlabel("NOM")
        plt.ylabel("DENSITY")
        plt.title("NOMINATOR")
        plt.show()
        exit()

        plt.plot(post_04_max + pre_max, label="POST_04, AVG: "+str(np.mean(post_04_max + pre_max)))
        plt.plot(post_05_max + pre_max, c="r", alpha=0.5, label="POST_05, AVG: "+str(np.mean(post_05_max + pre_max)))
        plt.title("DENOMINATOR")
        plt.legend()
        plt.show()

        plt.plot((post_04_max - pre_max)/(post_05_max - pre_max))
        plt.yscale("log")
        plt.title("NOMINATOR")
        plt.show()

        plt.plot((post_04_max + pre_max)/((post_05_max + pre_max)*0.01))
        plt.yscale("log")
        plt.title("DENOMINATOR")
        plt.show()

    def nrem_vs_rem_compression(self):

        speed_threshold = self.session_params.sleep_phase_speed_threshold

        # get SWR timings (in sec) & compute spike rasters (constant #spikes)
        # ------------------------------------------------------------------------------------------------------
        start_times, end_times, peak_times = self.detect_swr()

        # convert to one array for event_spike_binning
        event_times = np.vstack((start_times, end_times, peak_times)).T

        # only select SWR during nrem phases
        # get nrem phases in seconds
        n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold)
        swr_in_n_rem = np.zeros(event_times.shape[0])
        swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
        for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
            n_rem_start = n_rem_phase[0]
            n_rem_end = n_rem_phase[1]
            for i, e_t in enumerate(event_times):
                event_start = e_t[0]
                event_end = e_t[1]
                if (n_rem_start < event_start) and (event_end < n_rem_end):
                    swr_in_n_rem[i] += 1
                    swr_in_which_n_rem[n_rem_phase_id, i] = 1

        event_times = event_times[swr_in_n_rem == 1]

        print(" - " + str(event_times.shape[0]) + " WERE ASSIGNED TO NREM PHASES\n")

        # compute #spike binning for each event --> TODO: implement sliding window!
        _, event_spike_window_lenghts_nrem = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

        event_spike_window_lenghts_nrem = np.hstack(event_spike_window_lenghts_nrem)
        # get rem intervals in seconds
        event_times = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold)

        print(" - " + str(event_times.shape[0]) + " REM phases found (speed thr.: " + str(speed_threshold) + ")\n")

        # compute #spike binning for each event --> TODO: implement sliding window!
        _, event_spike_window_lenghts_rem = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

        event_spike_window_lenghts_rem = np.hstack(event_spike_window_lenghts_rem)

        plt.hist(event_spike_window_lenghts_nrem,
                 color="blue", label="NREM, MEDIAN = "+str(np.median(event_spike_window_lenghts_nrem)), density=True)
        plt.hist(event_spike_window_lenghts_rem,
                 color="red", label="REM, MEDIAN = "+str(np.median(event_spike_window_lenghts_rem)), density=True)
        plt.xlabel("DURATION OF 12 SPIKE BINS / s")
        plt.ylabel("DENSITY")
        plt.legend()
        plt.show()

    # </editor-fold>
    
    # <editor-fold desc="SWR sequential content analysis">
    
    """#################################################################################################################
    #   SWR sequential content analysis
    #################################################################################################################"""

    def sequential_content_phmm(self, pHMM_file_pre, pHMM_file_post, SWR=True,
                                   plot_for_control=False):
        # --------------------------------------------------------------------------------------------------------------
        # decodes sleep activity using pHMM model and investigates sequential content of activity (how similar are the
        # sequence observed during sleep to sequences observed during behavior)
        #
        # args:     - pHMM_file_pre:    name of file containing the dictionary with pHMM model from PRE behavioral data
        #           - pHMM_file_post:    name of file containing the dictionary with pHMM model from
        #                                                   AFTER/POST awake/behavioral data
        #
        #           - SWR, bool: whether to use only SWR (=True) or all activity
        #           - plot_for_control, bool: plots intermediate results if True
        #           - n_moving_average, int: n to use for moving average
        #           - return_reslts: if True --> returns results instead of plotting
        #
        # returns:  - list with entries for each SWR:  entry contains array with [pop_vec_SWR, spatial_bin_template_map]
        #                                              probabilities
        # --------------------------------------------------------------------------------------------------------------

        print(" - CONTENT BASED MEMORY DRIFT USING PHMM MODES (HMMLEARN)...\n")

        if SWR:

            # generate results for pre
            pre_log_prob, pre_sequence_list, event_times = \
                self.decode_swr_sequence_using_phmm(phmm_file_name=pHMM_file_pre, plot_for_control=plot_for_control)
            # generate results for post
            post_log_prob, post_sequence_list, post_event_times = \
                self.decode_swr_sequence_using_phmm(phmm_file_name=pHMM_file_post, plot_for_control=plot_for_control)

            # get model from PRE and transition matrix
            model_pre = self.load_poisson_hmm(file_name=pHMM_file_pre)
            transmat_pre = model_pre.transmat_
            model_post = self.load_poisson_hmm(file_name=pHMM_file_pre)
            transmat_post = model_post.transmat_

            pre_swr_seq_similarity = []
            post_swr_seq_similarity = []

            for pre_sequence, post_sequence in zip(pre_sequence_list, post_sequence_list):

                # make sure that there is any data for the current SWR

                if pre_sequence.shape[0] > 0:

                    # check how likely observed sequence is considering transitions from model (awake behavior)
                    mode_before = pre_sequence[:-1]
                    mode_after = pre_sequence[1:]
                    transition_prob = 0
                    # go trough each transition of the sequence
                    for bef, aft in zip(mode_before, mode_after):
                        transition_prob += np.log(transmat_pre[bef, aft])

                    pre_swr_seq_similarity.append(np.exp(transition_prob))

                    # POST
                    # check how likely observed sequence is considering transitions from model (awake behavior)
                    mode_before = post_sequence[:-1]
                    mode_after = post_sequence[1:]
                    transition_prob = 0
                    # go trough each transition of the sequence
                    for bef, aft in zip(mode_before, mode_after):
                        transition_prob += np.log(transmat_post[bef, aft])

                    post_swr_seq_similarity.append(np.exp(transition_prob))

            # sequence probability
            # pre_swr_seq_similarity = moving_average(a=np.array(pre_swr_seq_similarity), n=10)
            plt.plot(pre_swr_seq_similarity)
            plt.title("PROBABILITY SWR PHMM MODE SEQUENCES \n USING VITERBI + AWAKE TRANSITION PROB. PRE")
            plt.ylabel("JOINT PROBABILITY")
            plt.xlabel("SWR ID")
            plt.show()

            # sequence probability
            # post_swr_seq_similarity = moving_average(a=np.array(post_swr_seq_similarity), n=10)
            plt.plot(post_swr_seq_similarity)
            plt.title("PROBABILITY SWR PHMM MODE SEQUENCES \n USING VITERBI + AWAKE TRANSITION PROB. POST")
            plt.ylabel("JOINT PROBABILITY")
            plt.xlabel("SWR ID")
            plt.show()

        else:
            raise Exception("TO BE IMPLEMENTED!")

    def decode_swr_sequence_using_phmm(self, phmm_file_name, plot_for_control=False,
                              time_bin_size_encoding=0.1):
        """
        decodes sleep activity sequence during SWR using pHMM model from awake activity (uses hmmlearn predict_proba)

        @param phmm_file_name: name of file containing the dictionary with template from awake/behavioral data
        @type phmm_file_name: str
        @param plot_for_control: whether to plot intermediate results for control
        @type plot_for_control: bool
        @param time_bin_size_encoding: which time bin size was used for encoding (usually 100ms --> should save this
        info with the model
        @type time_bin_size_encoding: float
        """

        print(" - DECODING SLEEP ACTIVITY SEQUENCE USING HMMLEARN CODE & "+ phmm_file_name +" ...\n")

        # load and pre-process template map (computed on awake data)
        # --------------------------------------------------------------------------------------------------------------

        # load pHMM model
        with open('temp_data_old/phmm/' + phmm_file_name + '.pkl', 'rb') as f:
             phmm_model = pickle.load(f)

        mode_means = phmm_model.means_
        nr_modes = mode_means.shape[0]

        # get SWR timings (in sec) & compute spike rasters (constant #spikes)
        # --------------------------------------------------------------------------------------------------------------
        start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

        # convert to one array for event_spike_binning
        event_times = np.vstack((start_times, end_times)).T

        # compute #spike binning for each event --> TODO: implement sliding window!
        event_spike_rasters, event_spike_window_lenghts = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)
        # plot raster and detected SWR, example spike rasters from SWRs
        if plot_for_control:
                        # compute pre processed data
            self.compute_raster_speed_loc()
            to_plot = np.random.randint(0, start_times.shape[0], 5)
            for i in to_plot:
                plt.imshow(self.raster, interpolation='nearest', aspect='auto')
                plt.vlines(start_times[i] / self.params.time_bin_size, 0, self.raster.shape[0] - 1, colors="r",
                           linewidth=0.5,
                           label="SWR START")
                plt.vlines(end_times[i] / self.params.time_bin_size, 0, self.raster.shape[0] - 1, colors="g",
                           linewidth=0.5, label="SWR END")
                plt.xlim(start_times[i] / self.params.time_bin_size - 10,
                         end_times[i] / self.params.time_bin_size + 10)
                plt.ylabel("CELL IDS")
                plt.xlabel("TIME BINS")
                plt.legend()
                a = plt.colorbar()
                a.set_label("SPIKES PER BIN (" + str(self.params.time_bin_size) + " s)")
                plt.title("SWR EVENT ID " + str(i))
                plt.show()

                plt.imshow(event_spike_rasters[i], interpolation='nearest', aspect='auto')
                a = plt.colorbar()
                a.set_label("SPIKES PER CONST. #SPIKES BIN")
                plt.xlabel("CONST. #SPIKE POP. VEC. ID")
                plt.ylabel("CELL ID")
                plt.title("BINNED SWR EVENT ID " +str(i) +"\n#SPIKES PER BIN: "+str(self.params.spikes_per_bin))
                plt.show()

        # compute firing rate factor (sleep <-> awake activity with different binning & compression)
        # --------------------------------------------------------------------------------------------------------------

        # time_window_factor:   accounts for the different window length of awake encoding and window length
        #                       during sleep (variable window length because of #spike binning)

        # compute average window length for swr activity
        event_spike_window_lengths_avg = np.mean(np.concatenate(event_spike_window_lenghts, axis=0))
        time_window_factor = event_spike_window_lengths_avg / time_bin_size_encoding

        # compression_factor: additional compression factor
        compression_factor = 1

        # firing_rate_factor: need this because we are comparing sleep activity (compressed) with awake activity
        firing_rate_factor = time_window_factor*compression_factor

        # list to store results per SWR
        sequence_list = []
        log_prob_list = []

        # main decoding part
        # --------------------------------------------------------------------------------------------------------------
        # go through all SWR events
        for event_id, spike_raster in enumerate(event_spike_rasters):

            # instead of multiplying awake activity by firing_rate_factor --> divide population vectors by
            # firing rate factor
            spike_raster /= firing_rate_factor
            res = phmm_model.decode(spike_raster.T, algorithm="viterbi")
            sequence_list.append(res[1])
            log_prob_list.append(res[0])


        return log_prob_list, sequence_list, event_times

    # </editor-fold>

    # <editor-fold desc="Predicting time bin progression">

    # predicting time bin progression
    # ------------------------------------------------------------------------------------------------------------------
    def predict_bin_progression(self, time_bin_size=None):
        """
        analysis of drift using population vectors

        @param time_bin_size: which time bin size to use for prediction --> if None:
        standard time bin size from parameter object is used
        @type time_bin_size: float
        """
        # compute pre processed data
        self.compute_raster_speed_loc()

        if time_bin_size is None:
            time_bin_size = self.params.time_bin_size

        x = self.raster

        x = (x - np.min(x, axis=1, keepdims=True)) / \
            (np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True))

        # plot activation matrix (matrix of population vectors)
        plt.imshow(x, vmin=0, vmax=x.max(), cmap='jet', aspect='auto')
        plt.imshow(x, interpolation='nearest', aspect='auto', cmap="jet",
                   extent=[0, x.shape[1], x.shape[0] - 0.5, 0.5])
        plt.ylabel("CELL ID")
        plt.xlabel("TIME BINS")
        plt.title("SPIKE BINNING RASTER")
        a = plt.colorbar()
        a.set_label("# SPIKES")
        plt.show()

        y = np.arange(x.shape[1])

        new_ml = MlMethodsOnePopulation()
        new_ml.ridge_time_bin_progress(x=x, y=y, new_time_bin_size=time_bin_size)

    def predict_time_bin_pop_vec_non_hse(self, normalize_firing_rates=True):
        # --------------------------------------------------------------------------------------------------------------
        # analysis of drift using population vectors
        #
        # args:   - normalize_firing_rates, bool: yes if true
        # --------------------------------------------------------------------------------------------------------------
        # compute pre processed data
        self.compute_raster_speed_loc()

        x = self.raster

        # find high synchrony events
        ind_hse = find_hse(x=x)

        # remove high synchrony events
        x_wo_hse = np.delete(x, ind_hse, axis=1)

        for new_time_bin_size in [0.1, 1, 5]:

            # down/up sample data
            time_bin_scaler = int(new_time_bin_size / self.params.time_bin_size)

            new_raster = np.zeros((x_wo_hse.shape[0], int(x_wo_hse.shape[1] / time_bin_scaler)))

            # down sample spikes by combining multiple bins
            for i in range(new_raster.shape[1]):
                new_raster[:, i] = np.sum(x_wo_hse[:, (i * time_bin_scaler): ((1 + i) * time_bin_scaler)], axis=1)

            plt.imshow(new_raster,  interpolation='nearest', aspect='auto')
            plt.colorbar()
            plt.show()


            if normalize_firing_rates:
                x = (new_raster-np.min(new_raster, axis=1, keepdims=True))/\
                    (np.max(new_raster, axis=1, keepdims=True)-np.min(new_raster, axis=1, keepdims=True))

            else:
                x = new_raster

            plt.imshow(x,  interpolation='nearest', aspect='auto')
            plt.colorbar()
            plt.show()

            y = np.arange(x.shape[1])*new_time_bin_size

            new_ml = MlMethodsOnePopulation()
            new_ml.ridge_time_bin_progress(x=x, y=y, new_time_bin_size=new_time_bin_size)

    def hse_predict_time_bin(self):
        # --------------------------------------------------------------------------------------------------------------
        # analysis of hse drift using population vectors
        #
        # args:
        # --------------------------------------------------------------------------------------------------------------

        # compute pre processed data
        self.compute_raster_speed_loc()

        x = self.raster

        # find high synchrony events
        ind_hse = np.array(find_hse(x=x)).flatten()

        # only select high synchrony events
        x = x[:, ind_hse]

        print("#HSE: "+str(x.shape[1]))

        y = ind_hse * self.params.time_bin_size

        new_ml = MlMethodsOnePopulation()
        new_ml.ridge_time_bin_progress(x=x, y=y, new_time_bin_size=self.params.time_bin_size, alpha_fitting=False,
                                       alpha=500)

    # </editor-fold>

    # <editor-fold desc="Instant change">
    def instant_change(self, part_to_analyze="all", only_stable_cells=False, distance_meas="cos"):

        # get population vector and sleep phases
        rem_phase = self.get_sleep_phase("rem")
        nrem_phase = self.get_sleep_phase("nrem")

        raster = self.get_raster()

        # r1 = raster[:, :10000]

        # compute distances
        distance_mat_1 = distance.squareform(distance.pdist(raster.T, metric=distance_meas))

        neigh_dist_1 = np.nan_to_num(np.diag(v=distance_mat_1, k=1))

        x_ = np.arange(neigh_dist_1.shape[0])
        # plot sleep phases
        # rem_phase_mod = rem_phase[rem_phase[:, 1] < neigh_dist_1.shape[0]]
        # nrem_phase_mod = nrem_phase[nrem_phase[:, 1] < neigh_dist_1.shape[0]]
        plt.plot(neigh_dist_1, color="gray")
        for rem_p in rem_phase:
            plt.plot(x_[int(rem_p[0]):int(rem_p[1])], neigh_dist_1[int(rem_p[0]):int(rem_p[1])], color="red", label="REM")
        for nrem_p in nrem_phase:
            plt.plot(x_[int(nrem_p[0]):int(nrem_p[1])], neigh_dist_1[int(nrem_p[0]):int(nrem_p[1])], color="blue", label="NREM")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.xlim(0, 2000)
        plt.xlabel("Time (s)")
        plt.ylabel("Cos distance")
        plt.show()

        ind = int(np.argwhere(np.logical_and(nrem_phase[:,0]>3200, nrem_phase[:,0]<3500)).flatten())
        sel_nrem = np.expand_dims(nrem_phase[ind,:],0)

        pre_res_nrem, post_res_nrem, window_len_nrem = self.decode_phmm_pre_post_using_event_times(event_times=sel_nrem)

        pre_res_nrem_sel = pre_res_nrem[0]
        post_res_nrem_sel = post_res_nrem[0]
        # compute likeli ratio
        ratio_nrem = (np.max(post_res_nrem_sel, axis=1)-np.max(pre_res_nrem_sel, axis=1))/(np.max(post_res_nrem_sel, axis=1)
                                                                            +np.max(pre_res_nrem_sel, axis=1))
        sel_nrem = sel_nrem.flatten()
        x_range = x_[int(sel_nrem[0]):int(sel_nrem[1])]
        cos_dist_to_plot = neigh_dist_1[int(sel_nrem[0]):int(sel_nrem[1])]
        fig = plt.figure()
        gs = fig.add_gridspec(10, 10)
        ax1 = fig.add_subplot(gs[:3, :])
        ax1.set_xlim(x_range[0], x_range[-1])
        ax1.set_ylim(0,1)
        ax1.plot(x_range, cos_dist_to_plot, color="blue",
                 label="NREM")
        plt.legend()
        ax1.set_ylabel("Cos dist")
        ax1.set_xlabel("Time (s)")
        ax2 = fig.add_subplot(gs[5:, :])
        ax2.plot(moving_average(ratio_nrem, 50))
        ax2.set_xlim(0, ratio_nrem.shape[0])
        ax2.set_ylabel("Likeli-ratio")
        ax2.set_xlabel("Const.#spike vec.")
        ax2.set_ylim(-1,1)
        plt.show()

        ind = int(np.argwhere(np.logical_and(rem_phase[:,0]>2100, rem_phase[:,0]<2250)).flatten())
        sel_rem = np.expand_dims(nrem_phase[ind,:],0)

        pre_res_rem, post_res_rem, window_len_rem = self.decode_phmm_pre_post_using_event_times(event_times=sel_rem)

        pre_res_rem_sel = pre_res_rem[0]
        post_res_rem_sel = post_res_rem[0]
        # compute likeli ratio
        ratio_rem = (np.max(post_res_rem_sel, axis=1)-np.max(pre_res_rem_sel, axis=1))/(np.max(post_res_rem_sel, axis=1)
                                                                            +np.max(pre_res_rem_sel, axis=1))
        sel_rem = sel_rem.flatten()
        x_range = x_[int(sel_rem[0]):int(sel_rem[1])]
        cos_dist_to_plot = neigh_dist_1[int(sel_rem[0]):int(sel_rem[1])]
        fig = plt.figure()
        gs = fig.add_gridspec(10, 10)
        ax1 = fig.add_subplot(gs[:3, :])
        ax1.set_xlim(x_range[0], x_range[-1])
        ax1.plot(x_range, cos_dist_to_plot, color="red",
                 label="REM")
        ax1.set_ylabel("Cos dist")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylim(0,1)
        plt.legend()
        ax2 = fig.add_subplot(gs[5:, :])
        ax2.plot(moving_average(ratio_rem, 50))
        ax2.set_xlim(0, ratio_rem.shape[0])
        ax2.set_ylabel("Likeli-ratio")
        ax2.set_xlabel("Const.#spike vec.")
        ax2.set_ylim(-1,1)
        plt.show()

        print("HERE")
    # </editor-fold>

    # <editor-fold desc="Others">
    def swr_times(self):
        start_times, end_times, peak_times = self.detect_swr(plot_for_control=False)
        return peak_times
    # </editor-fold>


class Exploration(BaseMethods):
    """Base class for exploration analysis"""

    def __init__(self, data_dic, cell_type, params, session_params=None, experiment_phase=None):
        # --------------------------------------------------------------------------------------------------------------
        # args: - data_dic, dictionary with standard data
        #       - cell_type, string: cell type to do analysis with
        #       - params, python class: contains all parameters
        # --------------------------------------------------------------------------------------------------------------

        # get attributes from parent class
        BaseMethods.__init__(self, data_dic, cell_type, params, session_params, experiment_phase)

        # get spatial factor: cm per .whl arbitrary unit
        self.spatial_factor = self.session_params.data_params_dictionary["spatial_factor"]

        # other data
        self.time_bin_size = params.time_bin_size

        # compute raster, location and speed data
        self.raster, self.loc, self.speed = PreProcessAwake(self.firing_times, self.params, self.whl,
                                                            spatial_factor=self.spatial_factor).get_raster_loc_vel()

        # # get dimensions of environment
        self.x_min, self.x_max, self.y_min, self.y_max = \
            min(self.loc[:, 0]), max(self.loc[:, 0]), min(self.loc[:, 1]), max(self.loc[:, 1])

    # <editor-fold desc="Getter methods">

    def get_env_dim(self):
        """
        Returns dimensions of the environment

        :return: x_min, x_max, y_min, y_max
        :rtype: float, float, float, float
        """
        return self.x_min, self.x_max, self.y_min, self.y_max

    def get_rate_maps(self, spatial_resolution=None):
        """
        Computes and returns rate maps

        :param spatial_resolution: in cm2
        :type spatial_resolution: float
        :return: list with rate maps
        :rtype: list
        """
        rate_maps, occ = self.rate_map_from_data(loc=self.loc, raster=self.raster,
                                                 spatial_resolution=spatial_resolution)
        return rate_maps

    def get_occ_map(self, spatial_resolution=None):
        """
        Returns occupancy map

        :param spatial_resolution: in cm2
        :type spatial_resolution: float
        :return: occupancy map
        :rtype: numpy.array
        """
        rate_maps, occ = self.rate_map_from_data(loc=self.loc, raster=self.raster,
                                                 spatial_resolution=spatial_resolution)
        return occ

    def get_rate_maps_occ_maps_temporal_splits(self, spatial_resolution=None, nr_of_splits=2, env_dim=None):
        """
        computes rate maps using splits of the data (in terms of time)

        :param spatial_resolution: spatial resolution of rate maps in cm2
        :type spatial_resolution: int
        :param nr_of_splits: in how many splits to divide the data
        :type nr_of_splits: int
        :return: list of rate maps (one list entry with rate maps for all cells for each split) and list of occ maps
        :rtype: list, list
        """

        len_split = int(self.loc.shape[0] / nr_of_splits)

        list_rate_maps = []
        list_occ_maps = []

        for split_id in range(nr_of_splits):
            rate_maps = None
            occ = None
            rate_maps, occ = self.rate_map_from_data(loc=self.loc[split_id * len_split:(split_id + 1) * len_split, :],
                                                     raster=self.raster[:,
                                                            split_id * len_split:(split_id + 1) * len_split],
                                                     spatial_resolution=spatial_resolution, env_dim=env_dim)

            list_rate_maps.append(rate_maps)
            list_occ_maps.append(occ)

        return list_rate_maps, list_occ_maps

    def get_spike_bin_raster(self, return_estimated_times=False):

        spike_raster, estimated_times = PreProcessAwake(self.firing_times, self.params, self.whl,
                                     spatial_factor=self.spatial_factor).spike_binning(spikes_per_bin=self.params.spikes_per_bin,
                                                                 return_estimated_times=True)
        if return_estimated_times:
            return spike_raster, estimated_times
        else:
            return spike_raster

    # </editor-fold>

    # <editor-fold desc="Plotting">

    def view_occupancy(self, spatial_resolution=None):
        """
        Plots occupancy

        :param spatial_resolution: in cm2
        :type spatial_resolution: int
        """

        rate_map, occ = self.rate_map_from_data(loc=self.loc, raster=self.raster, spatial_resolution=spatial_resolution)
        exp_time = np.round(np.sum(occ.flatten()) / 60, 3)

        plt.imshow(occ.T, interpolation='nearest', aspect='equal', vmin=0, vmax=2, origin='lower')
        a = plt.colorbar()
        a.set_label("OCCUPANCY / SEC")
        plt.ylabel("Y")
        plt.xlabel("X")
        plt.title("OCCUPANCY (EXPLORATION TIME: " + str(exp_time) + " min)")
        plt.show()

    def view_all_rate_maps(self):
        """
        Plots rate maps from all cells
        """
        rate_maps = PreProcessAwake(self.firing_times, self.params, self.whl).spatial_rate_map()
        for map in rate_maps:
            plt.imshow(map, interpolation='nearest', aspect='auto', origin="lower")
            plt.colorbar()
            plt.show()

    def view_one_rate_map(self, cell_id, sel_range=None):
        """
        Plots rate map from one cell

        :param cell_id: cell id of cell to plot
        :type cell_id: int
        :param sel_range: to use all (sel_range=None) or a subset of data to compute rate maps
        :type sel_range: None or np.array
        """
        if range is None:
            loc = self.loc
            raster = self.raster
        else:
            loc = self.loc[sel_range, :]
            raster = self.raster[:, sel_range]

        rate_map, occ = self.rate_map_from_data(loc=loc, raster=raster)

        plt.scatter(loc[:, 0], loc[:, 1])
        plt.show()

        plt.imshow(occ.T, origin="lower")
        plt.title("OCCUPANCY")
        plt.colorbar()
        plt.show()

        plt.imshow(rate_map[:, :, cell_id].T, interpolation='nearest', aspect='auto', origin="lower")
        a = plt.colorbar()
        a.set_label("FIRING RATE")
        plt.title("RATE MAP FOR CELL " + str(cell_id))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def save_raw_spike_times(self,):
        # --------------------------------------------------------------------------------------------------------------
        # determines spike times of each cell and saves them as a list of list (each list: firing times of one cell)
        # --> used for TreeHMM
        #
        # args:   - save_dir, str
        # --------------------------------------------------------------------------------------------------------------

        spike_times = self.firing_times
        # pickle in using python2 protocol
        with open(self.session_name+"_"+self.cell_type+"_"+self.experiment_phase, "wb") as f:
            pickle.dump(spike_times, f, protocol=2)

    # </editor-fold>

    # <editor-fold desc="Basic computations">

    def place_field_similarity(self, plotting=False):
        """
        Computes and returns place field similarity between cells (correlation of rate maps)
        :param plotting: whether to plot matrix or not
        :type plotting: bool
        :return: similarity matrix
        :rtype: numpy.array
        """
        rate_maps = PreProcessAwake(self.firing_times, self.params, self.whl).spatial_rate_map()
        pfs_matrix = np.zeros((len(rate_maps), len(rate_maps)))
        for i, r_m_1 in enumerate(rate_maps):
            for j, r_m_2 in enumerate(rate_maps):
                corr = pearsonr(r_m_1.flatten(), r_m_2.flatten())
                if corr[1] < 0.05:
                    pfs_matrix[i, j] = corr[0]
        if plotting:
            plt.imshow(pfs_matrix, interpolation='nearest', aspect='auto')
            a = plt.colorbar()
            a.set_label("PEARSON CORRELATION R")
            plt.xlabel("CELL ID")
            plt.ylabel("CELL ID")
            plt.title("PLACE FIELD SIMILARITY")
            plt.show()
        return pfs_matrix

    def place_field_entropy(self, plotting=False):
        """
        Computes entropy of the rate map of each cell

        :param plotting: whether to plot rate maps with maximum and minimum entropy
        :type plotting: bool
        :return: entropies
        :rtype: numpy.array
        """

        rate_maps = PreProcessAwake(self.firing_times, self.params, self.whl).spatial_rate_map()
        ent = np.zeros(len(rate_maps))
        for i, r_m in enumerate(rate_maps):
            ent[i] = entropy(r_m.flatten())

        if plotting:
            plt.subplot(2, 1, 1)
            plt.imshow(rate_maps[np.argmax(ent)], interpolation='nearest', aspect='auto')
            print(np.argmax(ent))
            a = plt.colorbar()
            a.set_label("FIRING RATE / Hz")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.subplot(2, 1, 2)
            plt.imshow(rate_maps[np.argmin(ent)], interpolation='nearest', aspect='auto')
            print(np.argmin(ent))
            plt.xlabel("X")
            plt.ylabel("Y")
            a = plt.colorbar()
            a.set_label("FIRING RATE / Hz")
            plt.show()
        return ent

    def correlate_entropy_place_field_similarity(self):
        """
        plots scatter plot with entropy and place field similarity
        """

        ent = self.place_field_entropy()
        pcf_mat = self.place_field_similarity()

        pcf_mean = np.mean(pcf_mat, axis=1)

        plt.scatter(ent, pcf_mean)
        plt.ylabel("MEAN PLACE FIELD SIMILARITY")
        plt.xlabel("RATE MAP ENTROPY")
        plt.show()

    def rate_map_from_data(self, loc, raster, spatial_resolution=None, gaussian_std=1, env_dim=None):
        """
        Computes rate maps from subset of the data

        :param loc: location data (2D) synchronized with raster data
        :type loc: numpy.array
        :param raster: raster, array [nr_cells, nr_time_bins]: #spikes per cell /time bin
        :type raster: numpy.array
        :param spatial_resolution: in cm2
        :type spatial_resolution: float
        :param gaussian_std: gaussian_std, float: std of gaussians for smoothing of rate map --> if 0: no smoothing
        :type gaussian_std: float
        :param env_dim: dimensions of the environment
        :type env_dim: numpy.array
        :return: list with rate maps, occupancy map
        :rtype: list, np.array
        """

        if spatial_resolution is None:
            spatial_resolution = self.params.spatial_resolution

        nr_cells = raster.shape[0]
        loc_ds = np.floor(loc / spatial_resolution).astype(int)

        if env_dim is None:
            # get dimensions of environment
            x_min, x_max, y_min, y_max = self.x_min, self.x_max, self.y_min, self.y_max
        else:
            x_min, x_max, y_min, y_max = env_dim[0], env_dim[1], env_dim[2], env_dim[3]

        # get size of environment
        x_span = x_max - x_min
        y_span = y_max - y_min

        # width - length ratio of environment
        w_l_ratio = y_span / x_span

        nr_spatial_bins = int(np.round(x_span / spatial_resolution))

        centers_x = np.linspace(x_min, x_max + 0.1, nr_spatial_bins)
        centers_y = np.linspace(y_min, y_max + 0.1, int(round(nr_spatial_bins * w_l_ratio)))

        dx = centers_x[1] - centers_x[0]
        dy = centers_y[1] - centers_y[0]

        # split location data into x and y coordinates
        x_loc = loc[:, 0]
        y_loc = loc[:, 1]

        x_loc[x_loc > x_max] = x_min - 0.01
        y_loc[y_loc > y_max] = y_min - 0.01

        occ = np.zeros((centers_x.shape[0], centers_y.shape[0]))
        raster_2d = np.zeros((nr_spatial_bins, int(np.round(nr_spatial_bins * w_l_ratio)), nr_cells))

        for i, (_, pop_vec) in enumerate(zip(loc_ds, raster.T)):
            xi = int(np.floor((x_loc[i] - x_min) / dx)) + 1
            yi = int(np.floor((y_loc[i] - y_min) / dy)) + 1
            if xi * yi > 0:
                occ[xi, yi] += 1
                raster_2d[xi, yi, :] += pop_vec

        # one pop vec is at time bin resolution --> need occupancy in seconds
        occ = occ * self.params.time_bin_size
        # make zeros to nan for division
        occ[occ == 0] = np.nan
        rate_map = np.nan_to_num(raster_2d / occ[..., None])
        occ = np.nan_to_num(occ)

        # rate[occ > 0.05] = rate[occ > 0.05] / occ[occ > 0.05]
        # if sigma_gauss > 0:
        #     rate = nd.gaussian_filter(rate, sigma=sigma_gauss)
        # rate[occ == 0] = 0

        # apply gaussian filtering --> smooth place fields
        for i in range(nr_cells):
            rate_map[:, :, i] = nd.gaussian_filter(rate_map[:, :, i], sigma=gaussian_std)

        return rate_map, occ

    def compare_external_binning(self):
        """
        Test binning using externally generated rasters (e.g. through matlab)
        """

        sel_range = range(1000)
        cell_to_plot = 5

        print(self.raster.shape, self.loc.shape)

        raster = self.raster[:, sel_range]
        loc = self.loc[sel_range, :]
        print(raster.shape, loc.shape)

        # load reference data

        mat_files = loadmat("matlab.mat")
        raster_ref = mat_files["UnitSp"][:, sel_range]
        occ_ref = mat_files["UnitBinOcc"][:, sel_range]
        loc_ref = mat_files["PathRely"][sel_range, :]

        print(raster_ref.shape, occ_ref.shape, loc_ref.shape)

        rate_map_ref, occ_map_ref = self.rate_map_from_data(loc=loc_ref, raster=raster_ref, gaussian_std=0)
        plt.imshow(rate_map_ref[:, :, cell_to_plot].T)
        plt.colorbar()
        plt.title("REF")
        plt.show()

        rate_map_ref, occ_map_ref = self.rate_map_from_data(loc=loc, raster=raster, gaussian_std=0)
        plt.imshow(rate_map_ref[:, :, cell_to_plot].T)
        plt.colorbar()
        plt.title("ORIG")
        plt.show()

    # </editor-fold>

    # <editor-fold desc="pHMM">

    """#################################################################################################################
    #   clustering / discrete system states analysis
    #################################################################################################################"""

    def cross_val_poisson_hmm(self, cl_ar=np.arange(1, 50, 5), cells_to_use="all_cells"):
        # --------------------------------------------------------------------------------------------------------------
        # cross validation of poisson hmm fits to data
        #
        # args:     - cl_ar, range object: #clusters to fit to data
        # --------------------------------------------------------------------------------------------------------------

        print(" - CROSS-VALIDATING POISSON HMM USING EXPLORATION DATA --> OPTIMAL #MODES ...")
        print("  - nr modes to compute: "+str(cl_ar))

        # check how many cells
        nr_cells = self.raster.shape[0]
        raster = self.raster

        # if subset of cells to use
        if cells_to_use == "stable_cells":
            # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            stable_cells = class_dic["stable_cell_ids"].flatten()
            raster = raster[stable_cells, :]

        elif cells_to_use == "decreasing_cells":
            # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            dec_cells = class_dic["decrease_cell_ids"].flatten()
            raster = raster[dec_cells, :]

        elif cells_to_use == "increasing_cells":
            # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            inc_cells = class_dic["increase_cell_ids"].flatten()
            raster = raster[inc_cells, :]

        if self.params.cross_val_splits == "custom_splits":

            # number of folds
            nr_folds = 10
            # how many times to fit for each split
            max_nr_fitting = 5
            # how many chunks (for pre-computed splits)
            nr_chunks = 10

            unobserved_lo_array = pickle.load(
                open("temp_data/unobserved_lo_cv" + str(nr_folds) + "_" + str(nr_chunks) + "_chunks", "rb"))
            unobserved_hi_array = pickle.load(
                open("temp_data/unobserved_hi_cv" + str(nr_folds) + "_" + str(nr_chunks) + "_chunks", "rb"))

            # set number of time bins
            bin_num = raster.shape[1]
            bins = np.arange(bin_num + 1)

            # length of one chunk
            n_chunks = int(bin_num / nr_chunks)
            test_range_per_fold = []
            for fold in range(nr_folds):

                # unobserved_lo: start bins (in spike data resolution) for all test data chunks
                unobserved_lo = []
                unobserved_hi = []
                for lo, hi in zip(unobserved_lo_array[fold], unobserved_hi_array[fold]):
                    unobserved_lo.append(bins[lo * n_chunks])
                    unobserved_hi.append(bins[hi * n_chunks])

                unobserved_lo = np.array(unobserved_lo)
                unobserved_hi = np.array(unobserved_hi)

                test_range = []
                for lo, hi in zip(unobserved_lo, unobserved_hi):
                    test_range += (list(range(lo, hi)))
                test_range_per_fold.append(np.array(test_range))
        elif self.params.cross_val_splits == "standard_k_fold":
            test_range_per_fold=None
        else:
            raise Exception("Cross-val split method not defined!")

        nr_cores = 12

        folder_name = self.session_name +"_"+self.experiment_phase_id+"_"+self.cell_type

        new_ml = MlMethodsOnePopulation(params=self.params)
        new_ml.parallelize_cross_val_model(nr_cluster_array=cl_ar, nr_cores=nr_cores, model_type="pHMM",
                                           raster_data=raster, folder_name=folder_name, splits=test_range_per_fold,
                                           cells_used=cells_to_use)
        # new_ml.cross_val_view_results(folder_name=folder_name)

    def find_and_fit_optimal_number_of_modes(self, cells_to_use="all_cells", cl_ar_init = np.arange(1, 50, 5)):
        # compute likelihoods with standard spacing first
        self.cross_val_poisson_hmm(cells_to_use=cells_to_use, cl_ar=cl_ar_init)
        # get optimal number of modes for coarse grained
        trials_to_use = self.default_trials
        folder_name = self.session_name + "_" + str(
            int(self.experiment_phase_id)) + "_" + self.cell_type + "_trials_"+str(trials_to_use[0])+ \
                      "_"+str(trials_to_use[-1])
        new_ml = MlMethodsOnePopulation(params=self.params)
        opt_nr_coarse = new_ml.get_optimal_mode_number(folder_name=folder_name, cells_used=cells_to_use)
        print("Coarse opt. number of modes: "+str(opt_nr_coarse))
        self.cross_val_poisson_hmm(cells_to_use=cells_to_use, cl_ar=np.arange(opt_nr_coarse - 2, opt_nr_coarse + 3, 2))
        opt_nr_fine = new_ml.get_optimal_mode_number(folder_name=folder_name, cells_used=cells_to_use)
        print("Fine opt. number of modes: " + str(opt_nr_fine))
        self.fit_poisson_hmm(nr_modes=opt_nr_fine, cells_to_use=cells_to_use)

    def view_cross_val_results(self, range_to_plot=None, save_fig=False, cells_used="all_cells"):
        # --------------------------------------------------------------------------------------------------------------
        # views cross validation results
        #
        # args:     - model_type, string: which type of model ("POISSON_HMM")
        #           - custom_splits, bool: whether custom splits were used for cross validation
        # --------------------------------------------------------------------------------------------------------------

        folder_name = self.session_name + "_" + str(
            int(self.experiment_phase_id)) + "_" + self.cell_type
        new_ml = MlMethodsOnePopulation(params=self.params)
        new_ml.cross_val_view_results(folder_name=folder_name, range_to_plot=range_to_plot, save_fig=save_fig,
                                      cells_used=cells_used)

    def fit_poisson_hmm(self, nr_modes, cells_to_use="all_cells"):
        # --------------------------------------------------------------------------------------------------------------
        # fits poisson hmm to data
        #
        # args:     - nr_modes, int: #clusters to fit to data
        #           - file_identifier, string: string that is added at the end of file for identification
        # --------------------------------------------------------------------------------------------------------------

        print(" - FITTING POISSON HMM WITH "+str(nr_modes)+" MODES ...\n")

        # check how many cells
        nr_cells = self.raster.shape[0]
        raster = self.raster

        # if subset of cells to use
        if cells_to_use == "stable_cells":
            # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            stable_cells = class_dic["stable_cell_ids"].flatten()
            raster = raster[stable_cells, :]

        elif cells_to_use == "decreasing_cells":
            # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            dec_cells = class_dic["decrease_cell_ids"].flatten()
            raster = raster[dec_cells, :]

        elif cells_to_use == "increasing_cells":
            # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            inc_cells = class_dic["increase_cell_ids"].flatten()
            raster = raster[inc_cells, :]

        log_li = -1*np.inf
        # fit 10 times to select model with best highest log-likelihood (NOT CROSS-VALIDATED!!!)
        for i in range(10):
            test_model = PoissonHMM(n_components=nr_modes)
            test_model.fit(raster.T)
            log_li_test = test_model.score(raster.T)
            if log_li_test > log_li:
                model = test_model
                log_li = log_li_test

        model.set_time_bin_size(time_bin_size=self.params.time_bin_size)

        if cells_to_use == "stable_cells":
            save_dir = self.params.pre_proc_dir+"phmm/stable_cells/"
        elif cells_to_use == "decreasing_cells":
            save_dir = self.params.pre_proc_dir+"phmm/decreasing_cells/"
        elif cells_to_use == "increasing_cells":
            save_dir = self.params.pre_proc_dir+"phmm/increasing_cells/"
        elif cells_to_use == "all_cells":
            save_dir = self.params.pre_proc_dir+"phmm/"
        file_name = self.session_name + "_" + str(
            int(self.experiment_phase_id)) + "_" + self.cell_type + "_"+str(nr_modes)+"_modes"

        with open(save_dir+file_name+".pkl", "wb") as file: pickle.dump(model, file)

        print("  - ... DONE!\n")


    def fit_spatial_gaussians_for_modes(self, file_name, min_nr_bins_active, plot_awake_fit=False):

        state_sequence, nr_modes = self.decode_poisson_hmm(file_name=file_name)

        mode_id, freq = np.unique(state_sequence, return_counts=True)
        modes_to_plot = mode_id[freq > min_nr_bins_active]

        means = np.zeros((2, nr_modes))
        cov = np.zeros((2, nr_modes))
        for mode in np.arange(nr_modes):
            mode_data = self.loc[state_sequence == mode, :]

            if len(mode_data) == 0:
                means[:, mode] = np.nan
                cov[:, mode] = np.nan
            else:
                means[:, mode] = np.mean(mode_data, axis=0)
                cov[:, mode] = np.var(mode_data, axis=0)

        loc_data = self.loc[:, :]

        center = np.min(loc_data, axis=0) + (np.max(loc_data, axis=0) - np.min(loc_data, axis=0)) * 0.5
        dist = loc_data - center

        rad = max(np.sqrt(np.square(dist[:, 0]) + np.square(dist[:, 1]))) + 1

        std_modes = np.sqrt(cov[0,:]+cov[1,:])
        std_modes[std_modes == 0] = np.nan

        if plot_awake_fit:

            for mode_to_plot in modes_to_plot:

                mean = means[:, mode_to_plot]
                cov_ = cov[:, mode_to_plot]
                std_ = std_modes[mode_to_plot]

                # Parameters to set
                mu_x = mean[0]
                variance_x = cov_[0]

                mu_y = mean[1]
                variance_y = cov_[1]

                # Create grid and multivariate normal
                x = np.linspace(center[0] - rad, center[0]+rad, int(2.2*rad))
                y = np.linspace(0, 250, 250)
                X, Y = np.meshgrid(x, y)
                pos = np.empty(X.shape + (2,))
                pos[:, :, 0] = X
                pos[:, :, 1] = Y
                rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
                rv_normalized = rv.pdf(pos) / np.sum(rv.pdf(pos).flatten())

                fig, ax = plt.subplots()
                gauss = ax.imshow(rv_normalized)
                env = Circle((center[0], center[1]), rad, color="white", fill=False)
                ax.add_artist(env)
                ax.set_ylim(center[1] - 1.1*rad, center[1]+1.1*rad)
                ax.scatter(loc_data[state_sequence == mode_to_plot, 0], loc_data[state_sequence == mode_to_plot, 1],
                           alpha=1, c="white", marker=".", s=0.3, label="MODE "+ str(mode_to_plot) +" ASSIGNED")
                cb = plt.colorbar(gauss)
                cb.set_label("PROBABILITY")
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.title("STD: "+str(np.round(std_, 2)))
                plt.legend()
                plt.show()

        # compute frequency of each mode
        mode_freq = np.zeros(nr_modes)
        mode_freq[mode_id] = freq
        mode_freq = mode_freq.astype(int)

        env = Circle((center[0], center[1]), rad, color="white", fill=False)

        return means, std_modes, mode_freq, env, state_sequence

    def phmm_mode_spatial_information_from_model(self, spatial_resolution=5, nr_modes=None, file_name=None,
                                                 plot_for_control=False):
        """
        loads poisson hmm model and weighs rate maps by lambda vectors --> then computes spatial information (sparsity,
        skaggs information)

        @param spatial_resolution: spatial resolution in cm
        @type spatial_resolution: int
        @param nr_modes: nr of modes for model file identification
        @type nr_modes: int
        @param file_name: file containing the model --> is used when nr_modes is not provided to identify file
        @type file_name: string
        @param plot_for_control: whether to plot intermediate results
        @type plot_for_control: bool
        @return: sparsity, skaggs info for each mode
        """

        print(" - SPATIAL INFORMATION OF PHMM MODES USING MODEL\n")

        if file_name is None:
            file_name = self.params.session_name + "_" + self.experiment_phase_id + \
                        "_" + self.cell_type +"_"+str(nr_modes)+"_modes"

        with open(self.params.pre_proc_dir+"phmm/"+file_name+".pkl", "rb") as file:
            model = pickle.load(file)

        # get means for all modes (lambda vectors)
        means = model.means_

        ################################################################################################################
        # get spatial information of mode by weighing rate maps
        ################################################################################################################

        # compute rate maps and occupancy
        rate_maps, occ = self.rate_map_from_data(loc=self.loc, raster=self.raster,
                                                 spatial_resolution=spatial_resolution, gaussian_std=0)
        # compute binary occupancy map --> used as a mask to filter non-visited bins
        occ_mask = np.where(occ > 0, 1, np.nan)

        sparsity_list = []
        skaggs_list = []

        # go through all modes
        for mode_id, means_mode in enumerate(means):
            # weigh rate map of each cell using mean firing from lambda vector --> compute mean across all cells
            rate_map_mode_orig = np.mean(rate_maps * means_mode, axis=2)
            # generate filtered rate map by masking non visited places
            rate_map_mode_orig = np.multiply(rate_map_mode_orig, occ_mask)
            rate_map_mode = rate_map_mode_orig[~np.isnan(rate_map_mode_orig)]
            # need to filter bins with zero firing rate --> otherwise log causes an error
            rate_map_mode = rate_map_mode[rate_map_mode > 0]

            # compute sparsity
            sparse_mode = np.round(np.mean(rate_map_mode.flatten())**2/np.mean(np.square(rate_map_mode.flatten())), 2)

            # compute Skagg's information criterium
            print("DOUBLE CHECK SKAGGS INFORMATION FORMULATION")
            skaggs_info = np.round(np.sum((rate_map_mode.flatten()/np.mean(rate_map_mode.flatten())) *
                                np.log(rate_map_mode.flatten()/np.mean(rate_map_mode.flatten()))), 4)

            skaggs_list.append(skaggs_info)
            sparsity_list.append(sparse_mode)
            if plot_for_control:
                # plot random examples
                rand_float = np.random.randn(1)
                if rand_float > 0.5:
                    plt.imshow(rate_map_mode_orig)
                    plt.colorbar()
                    plt.title(str(sparse_mode)+", "+str(skaggs_info))
                    plt.show()

        if plot_for_control:
            plt.hist(skaggs_list)
            plt.title("SKAGGS INFO.")
            plt.xlabel("SKAGGS INFO.")
            plt.ylabel("COUNTS")
            plt.show()

            plt.hist(sparsity_list)
            plt.title("SPARSITY")
            plt.xlabel("SPARSITY")
            plt.ylabel("COUNTS")
            plt.show()

            plt.scatter(skaggs_list, sparsity_list)
            plt.title("SKAGGS vs. SPARSITY\n"+str(pearsonr(skaggs_list, sparsity_list)))
            plt.xlabel("SKAGGS")
            plt.ylabel("SPARSITY")
            plt.show()

        return np.array(sparsity_list), np.array(skaggs_list)

    def phmm_mode_spatial_information_from_fit(self, nr_modes, spatial_resolution, plot_for_control=False):
        """
        get spatial information of mode by fitting model to data and analyzing "mode locations" --> e.g taking
        distances between points

        @param plot_for_control: whether to plot intermediate results
        @type plot_for_control: bool
        @param nr_modes: nr. of modes to identify file containing model
        @type nr_modes: int
        @param spatial_resolution: spatial resolution for spatial binning in cm
        @type spatial_resolution: int
        """

        print(" - SPATIAL INFORMATION OF PHMM MODES USING FIT")

        state_sequence, nr_modes_ = self.decode_poisson_hmm(nr_modes=nr_modes)

        loc_data = self.loc[:, :]

        center = np.min(loc_data, axis=0) + (np.max(loc_data, axis=0) - np.min(loc_data, axis=0)) * 0.5
        dist = loc_data - center

        rad = max(np.sqrt(np.square(dist[:, 0]) + np.square(dist[:, 1]))) + 1

        # get dimensions of environment
        x_min, x_max, y_min, y_max = min(self.loc[:, 0]), max(self.loc[:, 0]), min(self.loc[:, 1]), max(
            self.loc[:, 1])

        # get size of environment
        x_span = x_max - x_min
        y_span = y_max - y_min

        # width - length ratio of environment
        w_l_ratio = y_span / x_span

        nr_spatial_bins = int(np.round(x_span / spatial_resolution))

        centers_x = np.linspace(x_min, x_max + 0.1, nr_spatial_bins)
        centers_y = np.linspace(y_min, y_max + 0.1, int(round(nr_spatial_bins * w_l_ratio)))

        dx = centers_x[1] - centers_x[0]
        dy = centers_y[1] - centers_y[0]

        # results from fit
        sparsity_fit_list = []
        skaggs_fit_list = []

        for mode in np.arange(nr_modes_):
            mode_loc = self.loc[state_sequence == mode, :]

            # compute pairwise distances (euclidean)
            pd = upper_tri_without_diag(pairwise_distances(mode_loc))

            mean_dist = np.mean(pd)
            std_dist = np.std(pd)

            if plot_for_control:

                fig, ax = plt.subplots()
                env = Circle((center[0], center[1]), rad, color="white", fill=False)
                ax.add_artist(env)
                ax.set_ylim(center[1] - 1.1 * rad, center[1] + 1.1 * rad)
                ax.set_xlim(center[0] - 1.1 * rad, center[0] + 1.1 * rad)
                ax.scatter(mode_loc[:, 0], mode_loc[:, 1])
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.title(str(mean_dist) + ", " + str(std_dist))
                plt.show()

                # # discretize data
                # # split location data into x and y coordinates
                # x_loc = mode_loc[:, 0]
                # y_loc = mode_loc[:, 1]
                #
                # x_loc[x_loc > x_max] = x_min - 0.01
                # y_loc[y_loc > y_max] = y_min - 0.01
                #
                # mode_act = np.zeros((centers_x.shape[0], centers_y.shape[0]))
                #
                # for i in range(x_loc.shape[0]):
                #     xi = int(np.floor((x_loc[i] - x_min) / dx)) + 1
                #     yi = int(np.floor((y_loc[i] - y_min) / dy)) + 1
                #     if xi * yi > 0:
                #         mode_act[xi, yi] += 1
                #
                # # mask non-visited places
                # mode_act_orig = np.multiply(mode_act, occ_mask)
                #
                # plt.imshow(mode_act_orig.T, origin="lower")
                # plt.show()

    def phmm_mode_spatial_information_visual(self, file_name):
        """
        uses phmm fit location

        @param file_name:
        @type file_name:
        """
        print(" - SPATIAL INFORMATION OF PHMM MODES - VISUALIZATION")

        with open(self.params.pre_proc_dir+"phmm/"+file_name+".pkl", "rb") as file:
            model = pickle.load(file)

        # get means for all modes (lambda vectors)
        means = model.means_
        # get transition matrix
        transmat = model.transmat_

        means, std_modes, mode_freq, env, state_sequence = \
            self.fit_spatial_gaussians_for_modes(file_name=file_name, plot_awake_fit=False, min_nr_bins_active=20)

        # compute distance between means --> correlate with transition matrix

        dist_mat = np.zeros((means.shape[1], means.shape[1]))

        for i, mean_1 in enumerate(means.T):
            for j, mean_2 in enumerate(means.T):
                dist_mat[i, j] = np.linalg.norm(mean_1 - mean_2)

        dist_flat = upper_tri_without_diag(dist_mat)
        transition = upper_tri_without_diag(transmat)

        plt.scatter(dist_flat, transition)
        plt.xlabel("DISTANCE BETWEEN MEANS / cm")
        plt.ylabel("TRANSITION PROBABILITY")
        plt.title("DISTANCE & TRANSITION PROBABILITES BETWEEN MODES")
        plt.show()

        std = np.nan_to_num(std_modes)

        constrained_means = means[:, (std < 25) & (std > 0) & (mode_freq > 100)]
        fig, ax = plt.subplots()
        ax.scatter(means[0, :], means[1, :], c="grey", label="ALL MEANS")
        ax.scatter(constrained_means[0, :], constrained_means[1, :], c="red", label="SPATIALLY CONSTRAINED")
        ax.add_artist(env)
        ax.set_ylim(40, 220)
        ax.set_xlim(0, 175)
        ax.set_aspect("equal")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

    def phmm_single_mode_details(self, mode_id, poisson_model_file):
        # --------------------------------------------------------------------------------------------------------------
        # loads poisson hmm model from file and fits to awake behavior and analysis details of a single mode
        #
        # args:   - mode_id, integer: which mode to look at
        #               - poisson_model_file, string: file that contains the trained model
        #
        # returns: -
        # --------------------------------------------------------------------------------------------------------------

        with open(self.params.pre_proc_dir+"phmm/" + poisson_model_file+".pkl", "rb") as file:
            model = pickle.load(file)

        X = self.raster

        state_sequence = model.predict(X.T)

        trans_mat = model.transmat_

        mode_lambda = model.means_

        # plt.hist(state_sequence, bins=80)
        # plt.show()

        mode_ids, freq = np.unique(state_sequence, return_counts=True)
        nr_modes = model.means_.shape[0]
        # compute frequency of each mode
        mode_freq = np.zeros(nr_modes)
        mode_freq[mode_ids] = freq
        mode_freq = mode_freq.astype(int)

        means = np.zeros((2, nr_modes))
        cov = np.zeros((2, nr_modes))
        for mode in np.arange(nr_modes):
            mode_data = self.loc[state_sequence == mode, :]

            if len(mode_data) == 0:
                means[:, mode] = np.nan
                cov[:, mode] = np.nan
            else:
                means[:, mode] = np.mean(mode_data, axis=0)
                cov[:, mode] = np.var(mode_data, axis=0)

        loc_data = self.loc[:, :]

        center = np.min(loc_data, axis=0) + (np.max(loc_data, axis=0) - np.min(loc_data, axis=0)) * 0.5
        dist = loc_data - center

        rad = max(np.sqrt(np.square(dist[:, 0]) + np.square(dist[:, 1]))) + 1

        std_modes = np.sqrt(cov[0, :] + cov[1, :])
        std_modes[std_modes == 0] = np.nan


        mean = means[:, mode_id]
        cov_ = cov[:, mode_id]
        std_ = std_modes[mode_id]

        # Parameters to set
        mu_x = mean[0]
        variance_x = cov_[0]

        mu_y = mean[1]
        variance_y = cov_[1]

        # Create grid and multivariate normal
        x = np.linspace(center[0] - rad, center[0] + rad, int(2.2 * rad))
        y = np.linspace(0, 250, 250)
        X, Y = np.meshgrid(x, y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
        rv_normalized = rv.pdf(pos) / np.sum(rv.pdf(pos).flatten())

        fig, ax = plt.subplots()
        gauss = ax.imshow(rv_normalized)
        env = Circle((center[0], center[1]), rad, color="white", fill=False)
        ax.add_artist(env)
        ax.set_ylim(center[1] - 1.1 * rad, center[1] + 1.1 * rad)
        ax.scatter(loc_data[state_sequence == mode_id, 0], loc_data[state_sequence == mode_id,1],
                   alpha=1, c="white", marker=".", s=0.3, label="MODE " + str(mode_id) + " ASSIGNED")
        cb = plt.colorbar(gauss)
        cb.set_label("PROBABILITY")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("STD: " + str(np.round(std_, 2)))
        plt.legend()
        plt.show()

    def decode_awake_activity_time_binning(self, model_name, plot_for_control=False,
                                           cells_to_use="all"):
        """

        @param model_name: name of file containing the pHMM file
        @type model_name: str
        @param plot_for_control: plots intermediate results if True
        @type plot_for_control: bool
        @return: pre_prob (probabilities of PRE), post_prob (probabilities of ), event_times (in sec),
        swr_in_which_n_rem (assignement SWR - nrem phase)
        @rtype: list, list, list, numpy.array
        """

        # get raster
        raster = self.raster

        if cells_to_use == "all":
            # load phmm model
            with open(self.params.pre_proc_dir + "phmm/" + model_name + '.pkl', 'rb') as f:
                model_dic = pickle.load(f)
        elif cells_to_use == "stable":
            # load phmm model
            with open(self.params.pre_proc_dir + "phmm/stable_cells/" + model_name + '.pkl', 'rb') as f:
                model_dic = pickle.load(f)
        elif cells_to_use == "decreasing":
            # load phmm model
            with open(self.params.pre_proc_dir + "phmm/decreasing_cells/" + model_name + '.pkl', 'rb') as f:
                model_dic = pickle.load(f)
        elif cells_to_use == "increasing":
            # load phmm model
            with open(self.params.pre_proc_dir + "phmm/increasing_cells/" + model_name + '.pkl', 'rb') as f:
                model_dic = pickle.load(f)

        # get means of model (lambdas) for decoding
        mode_means = model_dic.means_
        time_bin_size_encoding = model_dic.time_bin_size

        if time_bin_size_encoding == self.params.time_bin_size:
            compression_factor=1
        else:
            compression_factor = self.params.time_bin_size / time_bin_size_encoding


        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+ self.params.stable_cell_method +".pickle","rb") as f:
            class_dic = pickle.load(f)

        if not cells_to_use == "all":
            if cells_to_use == "stable":
                cell_ids = class_dic["stable_cell_ids"]
            elif cells_to_use == "increasing":
                cell_ids = class_dic["increase_cell_ids"]
            elif cells_to_use == "decreasing":
                cell_ids = class_dic["decrease_cell_ids"]
            raster = raster[cell_ids,:]

        print(" - DECODING USING " + cells_to_use + " CELLS")

        # delete rasters with less than two spikes
        nr_spikes_per_bin = np.sum(raster, axis=0)
        raster=raster[:, nr_spikes_per_bin>2]

        # decode activity
        results_list = decode_using_phmm_modes(mode_means=mode_means, event_spike_rasters=[raster],
                                               compression_factor=compression_factor, cell_selection="all")

        # plot maps of some SWR for control
        if plot_for_control:
            plt.imshow(np.log(results_list[0]).T[:,:50], interpolation='nearest', aspect='auto')
            plt.xlabel("POP.VEC. ID")
            plt.ylabel("MODE ID")
            a = plt.colorbar()
            a.set_label("log-likelihood")
            plt.title("log-Likelihood")
            plt.show()

        return results_list

    # </editor-fold>


class Cheeseboard:
    """Base class for cheese board task data analysis

       ATTENTION: this is only used for the task data --> otherwise use Sleep class!

    """

    def __init__(self, data_dic, cell_type, params, session_params, experiment_phase=None):
        """
        initializes cheeseboard class

        :param data_dic: dictionary containing spike data
        :type data_dic: python dic
        :param cell_type: which cell type to use
        :type cell_type: str
        :param params: general analysis params
        :type params: class
        :param session_params: sessions specific params
        :type session_params: class
        :param exp_phase_id: which experiment phase id
        :type exp_phase_id: int
        """

        # get standard analysis parameters
        self.params = copy.deepcopy(params)
        # get session analysis parameters
        self.session_params = session_params

        self.cell_type = cell_type
        self.time_bin_size = params.time_bin_size

        # get spatial factor: cm per .whl arbitrary unit
        self.spatial_factor = self.session_params.data_params_dictionary["spatial_factor"]

        # check if list or dictionary is passed:
        # --------------------------------------------------------------------------------------------------------------
        if isinstance(data_dic, list):
            self.data_dic = data_dic[0]
        else:
            self.data_dic = data_dic

        # check if extended data dictionary is provided (contains lfp)
        if "eeg" in self.data_dic.keys():
            self.eeg = self.data_dic["eeg"]
        if "eegh" in self.data_dic.keys():
            self.eegh = self.data_dic["eegh"]

        # # get all spike times
        self.firing_times = self.data_dic["spike_times"][cell_type]
        # # get location data
        self.whl = self.data_dic["whl"]

        # get last recorded spike
        if "last_spike" in self.data_dic.keys():
            self.last_spike = self.data_dic["last_spike"]
        else:
            self.last_spike = None

        # --------------------------------------------------------------------------------------------------------------
        # get phase specific info (ID & description)
        # --------------------------------------------------------------------------------------------------------------

        # check if list is passed:
        if isinstance(experiment_phase, list):
            experiment_phase = experiment_phase[0]

        self.experiment_phase = experiment_phase
        self.experiment_phase_id = session_params.data_description_dictionary[self.experiment_phase]
        # --------------------------------------------------------------------------------------------------------------
        # extract session analysis parameters
        # --------------------------------------------------------------------------------------------------------------
        self.session_name = self.session_params.session_name
        try:
            self.nr_trials = len(self.data_dic["trial_data"])
        except:
            print("No trial data")
            pass


        # compression factor:
        #
        # compression factor used for sleep decoding --> e.g when we use constant #spike bins with 12 spikes
        # we need to check how many spikes we have in e.g. 100ms windows if this was used for awake encoding
        # if we have a mean of 30 spikes for awake --> compression factor = 12/30 --> 0.4
        # is used to scale awake activity to fit sleep activity
        # --------------------------------------------------------------------------------------------------------------

        if cell_type == "p1_l":
            self.session_params.sleep_compression_factor_12spikes_100ms = \
                self.session_params.sleep_compression_factor_12spikes_100ms_p1_l
        elif cell_type == "p1_r":
            self.session_params.sleep_compression_factor_12spikes_100ms = \
                self.session_params.sleep_compression_factor_12spikes_100ms_p1_r
        else:
            self.session_params.sleep_compression_factor_12spikes_100ms = \
                self.session_params.sleep_compression_factor_12spikes_100ms

        # default models for behavioral data
        # --------------------------------------------------------------------------------------------------------------

        if cell_type == "p1_l":
            self.session_params.default_pre_phmm_model = self.session_params.default_pre_phmm_model_p1_l
            self.session_params.default_post_phmm_model = self.session_params.default_post_phmm_model_p1_l
            self.session_params.default_pre_ising_model = self.session_params.default_pre_ising_model_p1_l
            self.session_params.default_post_ising_model = self.session_params.default_post_ising_model_p1_l
        elif cell_type == "p1_r":
            self.session_params.default_pre_phmm_model = self.session_params.default_pre_phmm_model_p1_r
            self.session_params.default_post_phmm_model = self.session_params.default_post_phmm_model_p1_r
            self.session_params.default_pre_ising_model = self.session_params.default_pre_ising_model_p1_r
            self.session_params.default_post_ising_model = self.session_params.default_post_ising_model_p1_r
        else:
            self.session_params.default_pre_phmm_model = self.session_params.default_pre_phmm_model
            self.session_params.default_post_phmm_model = self.session_params.default_post_phmm_model
            self.session_params.default_pre_ising_model = self.session_params.default_pre_ising_model
            self.session_params.default_post_ising_model = self.session_params.default_post_ising_model

        # goal locations
        # --------------------------------------------------------------------------------------------------------------
        try:
            self.goal_locations = self.session_params.goal_locations
        except:
            print("GOAL LOCATIONS NOT FOUND")

        # convert goal locations from a.u. to cm
        self.goal_locations = np.array(self.goal_locations) * self.spatial_factor

        # get pre-selected trials (e.g. last 10 min of learning 1 and first 10 min of learning 2)
        # --------------------------------------------------------------------------------------------------------------
        if session_params.experiment_phase == "learning_cheeseboard_1":
            self.default_trials = self.session_params.default_trials_lcb_1
            self.default_ising = self.session_params.default_pre_ising_model
            self.default_phmm = self.session_params.default_pre_phmm_model
            self.default_phmm_stable = self.session_params.default_pre_phmm_model_stable
        elif session_params.experiment_phase == "learning_cheeseboard_2":
            self.default_trials = self.session_params.default_trials_lcb_2
            self.default_ising = self.session_params.default_post_ising_model
            self.default_phmm = self.session_params.default_post_phmm_model


        # compute raster, location speed
        # --------------------------------------------------------------------------------------------------------------
        self.trial_loc_list = []
        self.trial_raster_list = []
        self.trial_speed_list = []

        # initialize environment dimensions --> are updated later while loading all trial data
        self.x_min = np.inf
        self.x_max = -np.inf
        self.y_min = np.inf
        self.y_max = -np.inf

        self.start_box_coordinates = \
            np.array(self.session_params.data_params_dictionary["start_box_coordinates"]) * self.spatial_factor

        # get x-max from start box --> discard all data that is smaller than x-max
        x_max_sb = self.session_params.data_params_dictionary["start_box_coordinates"][1] * self.spatial_factor
        # compute center of cheeseboard (x_max_sb + 110 cm, assumption: cheeseboard diameter: 220cm)
        x_c = x_max_sb + 110 * self.spatial_factor

        # use center of start box to find y coordinate of center of cheeseboard
        y_c = self.session_params.data_params_dictionary["start_box_coordinates"][2]+ \
              (self.session_params.data_params_dictionary["start_box_coordinates"][3] - \
              self.session_params.data_params_dictionary["start_box_coordinates"][2])/2

        cb_center = np.expand_dims(np.array([x_c, y_c]), 0)

        # compute raster, location & speed for each trial
        for trial_id, key in enumerate(self.data_dic["trial_data"]):
            # compute raster, location and speed data
            raster, loc, speed = PreProcessAwake(firing_times=self.data_dic["trial_data"][key]["spike_times"][cell_type],
                                                 params=self.params, whl=self.data_dic["trial_data"][key]["whl"],
                                                 spatial_factor=self.spatial_factor
                                                 ).interval_temporal_binning_raster_loc_vel(
                interval_start=self.data_dic["trial_timestamps"][0,trial_id],
                interval_end=self.data_dic["trial_timestamps"][1,trial_id])

            # TODO: improve filtering of spatial locations outside cheeseboard
            if self.params.additional_spatial_filter:
                # filter periods that are outside the cheeseboard
                dist_from_center = distance.cdist(loc, cb_center, metric="euclidean")

                raster = np.delete(raster, np.where(dist_from_center > 110* self.spatial_factor), axis=1)
                speed = np.delete(speed, np.where(dist_from_center > 110* self.spatial_factor))
                loc = np.delete(loc, np.where(dist_from_center > 110* self.spatial_factor), axis=0)

            if self.session_name in ["mjc163R2R_0114", "mjc163R4R_0114", "mjc169R4R_0114", "mjc163R1L_0114",
                                     "mjc163R3L_0114", "mjc169R1R_0114"]:
                raster = np.delete(raster, np.where(loc[:,0] < x_max_sb), axis=1)
                speed = np.delete(speed, np.where(loc[:, 0] < x_max_sb))
                loc = np.delete(loc, np.where(loc[:, 0] < x_max_sb), axis=0)

            # update environment dimensions
            self.x_min = min(self.x_min, min(loc[:,0]))
            self.x_max = max(self.x_max, max(loc[:, 0]))
            self.y_min = min(self.y_min, min(loc[:,1]))
            self.y_max = max(self.y_max, max(loc[:, 1]))

            self.trial_raster_list.append(raster)
            self.trial_loc_list.append(loc)
            self.trial_speed_list.append(speed)

    # <editor-fold desc="Basic computations">

    """#################################################################################################################
    #  Basic computations
    #################################################################################################################"""

    def rate_map_from_data(self, loc, raster, spatial_resolution=None, gaussian_std=1, env_dim=None):
        # --------------------------------------------------------------------------------------------------------------
        # computes #spikes per bin from input data
        #
        # args:   - loc, array: location data (2D) synchronized with raster data
        #               - raster, array [nr_cells, nr_time_bins]: #spikes per cell /time bin
        #               - spatial_resolution, int: in cm
        #               - gaussian_std, float: std of gaussians for smoothing of rate map --> if 0: no smoothing
        #
        # returns:      - rate_map, array [x_coord, y_coord, cell_id], per spatial bin: spike rate (1/s)
        # --------------------------------------------------------------------------------------------------------------

        if spatial_resolution is None:
            spatial_resolution = self.params.spatial_resolution

        nr_cells = raster.shape[0]
        loc_ds = np.floor(loc / spatial_resolution).astype(int)

        if env_dim is None:
            # get dimensions of environment
            x_min, x_max, y_min, y_max = self.x_min, self.x_max, self.y_min, self.y_max
        else:
            x_min, x_max, y_min, y_max = env_dim[0], env_dim[1], env_dim[2], env_dim[3]

        # get size of environment
        x_span = x_max - x_min
        y_span = y_max - y_min

        # width - length ratio of environment
        w_l_ratio = y_span / x_span

        nr_spatial_bins = int(np.round(x_span / spatial_resolution))

        centers_x = np.linspace(x_min, x_max + 0.1, nr_spatial_bins)
        centers_y = np.linspace(y_min, y_max + 0.1, int(round(nr_spatial_bins * w_l_ratio)))

        dx = centers_x[1] - centers_x[0]
        dy = centers_y[1] - centers_y[0]

        # split location data into x and y coordinates
        x_loc = loc[:, 0]
        y_loc = loc[:, 1]

        x_loc[x_loc > x_max] = x_min - 0.01
        y_loc[y_loc > y_max] = y_min - 0.01

        occ = np.zeros((centers_x.shape[0], centers_y.shape[0]))
        raster_2d = np.zeros((nr_spatial_bins, int(np.round(nr_spatial_bins * w_l_ratio)), nr_cells))

        for i, (_, pop_vec) in enumerate(zip(loc_ds, raster.T)):
            xi = int(np.floor((x_loc[i] - x_min) / dx)) + 1
            yi = int(np.floor((y_loc[i] - y_min) / dy)) + 1
            if xi * yi > 0:
                occ[xi, yi] += 1
                raster_2d[xi, yi, :] += pop_vec

        # one pop vec is at time bin resolution --> need occupancy in seconds
        occ = occ * self.params.time_bin_size
        # make zeros to nan for division
        occ[occ == 0] = np.nan
        rate_map = np.nan_to_num(raster_2d / occ[..., None])
        occ = np.nan_to_num(occ)

        # rate[occ > 0.05] = rate[occ > 0.05] / occ[occ > 0.05]
        # if sigma_gauss > 0:
        #     rate = nd.gaussian_filter(rate, sigma=sigma_gauss)
        # rate[occ == 0] = 0

        # apply gaussian filtering --> smooth place fields
        for i in range(nr_cells):
            rate_map[:, :, i] = nd.gaussian_filter(rate_map[:, :, i], sigma=gaussian_std)

        return rate_map, occ

    # </editor-fold>

    # <editor-fold desc="Getter methods">

    """#################################################################################################################
    #  Getter methods
    #################################################################################################################"""

    def get_raster(self, trials_to_use=None):
        # get rasters from trials

        # check how many cells
        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))

        if trials_to_use == "all":
            for trial_id, _ in enumerate(self.trial_raster_list):
                raster = np.hstack((raster, self.trial_raster_list[trial_id]))

        elif trials_to_use is None:
            trials_to_use = self.default_trials

            for trial_id in trials_to_use:
                raster = np.hstack((raster, self.trial_raster_list[trial_id]))

        else:
            for trial_id in trials_to_use:
                raster = np.hstack((raster, self.trial_raster_list[trial_id]))

        return raster

    def get_raster_stillness(self, threshold_stillness=15, time_bin_size=0.01, plot_for_control=True,
                             min_period_duration_s=1):

        # detect periods of stillness
        # --------------------------------------------------------------------------------------------------------------

        speed = self.get_speed(time_bin_size=1)
        if plot_for_control:
            plt.hist(speed, density=True, bins=150)
            plt.xlabel("Speed")
            plt.ylabel("Density")
            y_min, y_max = plt.gca().get_ylim()
            plt.vlines(threshold_stillness, 0, y_max, color="red")
            plt.show()
        good_bins = np.zeros(speed.shape[0])
        good_bins[speed < threshold_stillness] = 1

        transitions = np.diff(good_bins)

        start = []
        end = []

        if good_bins[0] == 1:
            # first data point during stillness
            start.append(0)

        for bin_nr, tran in enumerate(transitions):
            if tran == -1:
                end.append(bin_nr)
            if tran == 1:
                start.append(bin_nr+1)

        if good_bins[-1] == 1:
            # last data point during stillness
            end.append(good_bins.shape[0])

        start = np.array(start)
        end = np.array(end)
        duration = (end-start)*1

        # delete all intervals that are shorter than time bin size
        start = start[duration > min_period_duration_s]
        end = end[duration > min_period_duration_s]

        event_times = np.vstack((start, end)).T

        print("Found "+str(event_times.shape[0])+" stillness periods!")

        stillness_rasters = []
        # get stillness rasters
        for e_t in event_times:
            stillness_rasters.append(PreProcessAwake(firing_times=self.firing_times, params=self.params,
              whl=self.whl, spatial_factor=self.spatial_factor).interval_temporal_binning(interval=e_t, interval_freq=1,
                                                                            time_bin_size=time_bin_size))

        stillness_rasters = np.hstack(stillness_rasters)

        return stillness_rasters

    def get_stillness_periods(self, threshold_stillness=15, plot_for_control=False,
                             min_period_duration_s=1):

        # detect periods of stillness
        # --------------------------------------------------------------------------------------------------------------

        speed = self.get_speed(time_bin_size=1)
        if plot_for_control:
            plt.hist(speed, density=True, bins=150)
            plt.xlabel("Speed")
            plt.ylabel("Density")
            y_min, y_max = plt.gca().get_ylim()
            plt.vlines(threshold_stillness, 0, y_max, color="red")
            plt.show()
        good_bins = np.zeros(speed.shape[0])
        good_bins[speed < threshold_stillness] = 1

        transitions = np.diff(good_bins)

        start = []
        end = []

        if good_bins[0] == 1:
            # first data point during stillness
            start.append(0)

        for bin_nr, tran in enumerate(transitions):
            if tran == -1:
                end.append(bin_nr)
            if tran == 1:
                start.append(bin_nr+1)

        if good_bins[-1] == 1:
            # last data point during stillness
            end.append(good_bins.shape[0])

        start = np.array(start)
        end = np.array(end)
        duration = (end-start)*1

        # delete all intervals that are shorter than time bin size
        start = start[duration > min_period_duration_s]
        end = end[duration > min_period_duration_s]

        event_times = np.vstack((start, end)).T

        print("Found "+str(event_times.shape[0])+" stillness periods!")

        return event_times

    def get_speed(self, time_bin_size=None):
        """
        returns speed of all data (not split into trials) at time bin size
        :return:
        :rtype:
        """
        return PreProcessAwake(firing_times=self.firing_times, params=self.params,
                        whl=self.whl, spatial_factor=self.spatial_factor).get_speed(time_bin_size=time_bin_size)

    def get_spike_bin_raster(self, return_estimated_times=False, trials_to_use=None):

        spike_raster = []
        estimated_times = []

        if trials_to_use == "all":
            print("Using all trials")
            for trial_id, key in enumerate(self.data_dic["trial_data"]):
                # compute raster, location and speed data
                spike_raster_, estimated_times_ = \
                    PreProcessAwake(firing_times=self.data_dic["trial_data"][key]["spike_times"][self.cell_type],
                                                     params=self.params, whl=self.data_dic["trial_data"][key]["whl"],
                                                     spatial_factor=self.spatial_factor
                                                     ).spike_binning(spikes_per_bin=self.params.spikes_per_bin,
                                                                     return_estimated_times=True)

                estimated_times.append(estimated_times_)
                spike_raster.append(spike_raster_)
        else:
            # use default trials if no range is provided
            if trials_to_use is None:
                trials_to_use = self.default_trials

            trials_to_use_keys = ["trial"+str(i) for i in trials_to_use]
            print("Using trials: "+str(trials_to_use_keys))
            for key in trials_to_use_keys:
                # compute raster, location and speed data
                spike_raster_, estimated_times_ = \
                    PreProcessAwake(firing_times=self.data_dic["trial_data"][key]["spike_times"][self.cell_type],
                                                     params=self.params, whl=self.data_dic["trial_data"][key]["whl"],
                                                     spatial_factor=self.spatial_factor
                                                     ).spike_binning(spikes_per_bin=self.params.spikes_per_bin,
                                                                     return_estimated_times=True)

                estimated_times.append(estimated_times_)
                spike_raster.append(spike_raster_)

        if return_estimated_times:
            return spike_raster, estimated_times
        else:
            return spike_raster


    def get_spike_times(self, trials_to_use=None):
        # get rasters from trials

        # check how many cells
        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))

        if trials_to_use == "all":
            for trial_id, _ in enumerate(self.trial_raster_list):
                raster = np.hstack((raster, self.trial_raster_list[trial_id]))

        elif trials_to_use is None:
            trials_to_use = self.default_trials

            for trial_id in trials_to_use:
                raster = np.hstack((raster, self.trial_raster_list[trial_id]))

        return raster

    def get_speed_at_whl_resolution(self):
        """
        returns speed of all data (not split into trials) at 0.0256s
        :return:
        :rtype:
        """
        return PreProcessAwake(firing_times=self.firing_times, params=self.params,
                        whl=self.whl, spatial_factor=self.spatial_factor).get_speed_at_whl_resolution()

    def get_goal_locations(self):
        return self.goal_locations

    def get_raster_and_trial_times(self, trials_to_use=None):

        if trials_to_use is None:
            trials_to_use = self.default_trials

        # check how many cells
        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))

        trial_lengths = []
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            trial_lengths.append(self.trial_raster_list[trial_id].shape[1])

        return raster, trial_lengths

    def get_env_dim(self):
        return [self.x_min, self.x_max, self.y_min, self.y_max]

    def get_raster_location_speed(self, trials_to_use=None):

        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))
        speed = np.empty(0)
        location = np.empty((0, 2))

        if trials_to_use is None:
            trials_to_use = self.default_trials
        elif trials_to_use == "all":
            trials_to_use = np.arange(len(self.trial_raster_list))

        if isinstance(trials_to_use, int):
            raster = np.hstack((raster, self.trial_raster_list[trials_to_use]))
            speed = np.hstack((speed, self.trial_speed_list[trials_to_use]))
            location = np.vstack((location, self.trial_loc_list[trials_to_use]))
        else:
            for trial_id in trials_to_use:
                raster = np.hstack((raster, self.trial_raster_list[trial_id]))
                speed = np.hstack((speed, self.trial_speed_list[trial_id]))
                location = np.vstack((location, self.trial_loc_list[trial_id]))

        return raster, location, speed

    def get_basic_info(self):

        print("NUMBER CELLS: " +str(self.trial_raster_list[0].shape[0])+"\n")

        print("TRIAL LENGTH:\n")
        len_trials = []
        for trial_id, trial_raster in enumerate(self.trial_raster_list):
            len_trials.append(np.round(trial_raster.shape[1]*self.time_bin_size,2))
            print(" - TRIAL " + str(trial_id) + ": " + str(np.round(trial_raster.shape[1]*self.time_bin_size,2))+ "s")

        plt.plot(len_trials, marker=".", c="r")
        plt.title("TRIAL DURATION")
        plt.xlabel("TRIAL ID")
        plt.ylabel("TRIAL DURATION / s")
        plt.grid()
        plt.show()

        print("\n#TRIALS FOR DURATION STARTING FROM FIRST:\n")
        cs = np.cumsum(np.array(len_trials))
        for i,c in enumerate(cs):
            print(" - TRIAL " + str(i) +": "+str(np.round(c/60,2))+"min")

        print("\n#TRIALS FOR DURATION STARTING FROM LAST:\n")
        cs = np.cumsum(np.flip(np.array(len_trials)))
        for i,c in enumerate(cs):
            print(" - TRIAL " + str(cs.shape[0]-i-1) +": "+str(np.round(c/60,2))+"min")

    def get_cell_classification_labels(self):
        """
        returns cell labels for stable, increasing, decreasing cells

        @return: cell indices for stable, decreasing, increasing cells
        """
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        cell_ids_stable = class_dic["stable_cell_ids"].flatten()
        cell_ids_decreasing = class_dic["decrease_cell_ids"].flatten()
        cell_ids_increasing = class_dic["increase_cell_ids"].flatten()

        return cell_ids_stable, cell_ids_decreasing, cell_ids_increasing

    def get_info_for_trial(self, trial_id, cell_id):
        """
        shows basic data for trial and cell

        @param trial_id: which trial to use
        @type trial_id: int
        @param cell_id: which cell to use
        @type cell_id: int
        """
        loc = self.trial_loc_list[trial_id]

        plt.scatter(loc[:, 0], loc[:, 1])
        plt.show()

        raster = self.trial_raster_list[trial_id]
        rate_map, occ = self.rate_map_from_data(loc=loc, raster=raster, gaussian_std=5)

        plt.imshow(occ.T,origin="lower" )
        plt.colorbar()
        plt.show()
        plt.imshow(rate_map[:,:,cell_id].T,origin="lower")
        plt.colorbar()
        plt.show()

    def get_rate_maps(self, spatial_resolution=5, env_dim=None, trials_to_use=None, gaussian_std=1):

        raster, location, speed = self.get_raster_location_speed(trials_to_use=trials_to_use)

        rate_maps, _ = self.rate_map_from_data(loc=location, raster=raster, spatial_resolution=spatial_resolution,
                                              env_dim=env_dim, gaussian_std=gaussian_std)

        return rate_maps

    def get_nr_of_trials(self):
        return len(self.trial_loc_list)

    def get_occ_map(self, spatial_resolution=1, env_dim=None, trials_to_use=None):

        raster, location, speed = self.get_raster_location_speed(trials_to_use=trials_to_use)

        _, occ_map = self.rate_map_from_data(loc=location, raster=raster, spatial_resolution=spatial_resolution,
                                              env_dim=env_dim)

        return occ_map

    def get_rate_maps_occ_maps_temporal_splits(self, spatial_resolution=None, nr_of_splits=2, env_dim=None,
                                               exclude_first_trial=False):
        """
        computes rate maps using splits of the data (in terms of time)

        :param spatial_resolution: spatial resolution of rate maps in cm2
        :type spatial_resolution: int
        :param nr_of_splits: in how many splits to divide the data
        :type nr_of_splits: int
        :return: list of rate maps (one list entry with rate maps for all cells for each split) and list of occ maps
        :rtype: list, list
        """

        if exclude_first_trial:
            # rasters & location from all except first trial
            nr_trials = len(self.trial_raster_list)
            raster, loc, _ = self.get_raster_location_speed(trials_to_use=np.arange(1, nr_trials))

        else:
            # get all rasters & location
            raster, loc, _ = self.get_raster_location_speed(trials_to_use="all")

        len_split = int(loc.shape[0]/nr_of_splits)

        list_rate_maps = []
        list_occ_maps = []

        for split_id in range(nr_of_splits):
            rate_maps=None
            occ = None
            rate_maps, occ = self.rate_map_from_data(loc=loc[split_id*len_split:(split_id+1)*len_split, :],
                                                     raster=raster[:, split_id*len_split:(split_id+1)*len_split],
                                                     spatial_resolution=spatial_resolution, env_dim=env_dim)

            list_rate_maps.append(rate_maps)
            list_occ_maps.append(occ)

        return list_rate_maps, list_occ_maps

    def get_burstiness_subsets(self, window_s=0.008, plot_for_control=True):

        burstiness = self.get_burstiness(window_s=window_s)

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        inc_cells = class_dic["increase_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()

        burstiness_stable = burstiness[stable_cells]
        burstiness_inc = burstiness[inc_cells]
        burstiness_dec = burstiness[dec_cells]

        return burstiness_stable, burstiness_inc, burstiness_dec


    def get_burstiness(self, window_s=0.008, plot_for_control=False):

        burstiness = np.zeros(len(self.data_dic["spike_times"][self.cell_type]))

        for i_cell, cell_id in enumerate(self.data_dic["spike_times"][self.cell_type]):

            spike_times = self.data_dic["spike_times"][self.cell_type][cell_id]

            if spike_times.shape[0] == 0:
                # no spiking
                burstiness[i_cell] = np.nan
                continue
            # interspike interval in 20 kHz
            isi = np.diff(spike_times)
            isi_s = isi/20e3

            # within window
            within_win = isi_s < window_s

            # find subsequent intervals with n > 2 (number of intervals)
            trans = np.diff(within_win.astype(int))

            bursts_start = np.argwhere(trans==1).flatten()
            bursts_end = np.argwhere(trans==-1).flatten()
            if bursts_end.shape[0] == 0 or bursts_start.shape[0] == 0:
                burstiness[i_cell] = np.nan
                continue

            # if there was a burst immediately --> need to add zero at the beginning
            if bursts_start.shape[0] < bursts_end.shape[0]:
                bursts_start = np.insert(bursts_start, 0, 0)
            # if there was a burst at the end --> need to add last element to end
            elif bursts_start.shape[0] > bursts_end.shape[0]:
                bursts_end = np.insert(bursts_end, -1, trans.shape[0])

            dur = bursts_end - bursts_start

            bursts = dur[dur > 2]
            # because a string of 4 1's means 5 spikes in a burst
            spikes_in_bursts = np.sum(bursts) + bursts.shape[0]

            burstiness[i_cell] = spikes_in_bursts/spike_times.shape[0]

        if plot_for_control:
            burstiness = burstiness[~np.isnan(burstiness)]
            p = 1. * np.arange(burstiness.shape[0]) / (burstiness.shape[0] - 1)
            plt.plot(np.sort(burstiness), p)
            plt.xlabel("Burstiness")
            plt.ylabel("CDF")
            plt.tight_layout()
            plt.show()

        return burstiness

    # </editor-fold>

    # <editor-fold desc="Plotting methods">

    """#################################################################################################################
    #  Plotting methods
    #################################################################################################################"""

    def plot_speed_per_trial(self):

        avg = []
        for s in self.trial_speed_list:
            avg.append(np.mean(s))

        plt.plot(avg, marker=".")
        plt.title("AVG. SPEED PER TRIAL")
        plt.xlabel("TRIAL NR.")
        plt.ylabel("AVG. SPEED / cm/s")
        plt.show()

    def plot_rate_maps(self, spatial_resolution=5, gaussian_std=1):

        # goal locations
        gl = self.session_params.goal_locations

        # offset for plotting:
        if self.session_name == "mjc163R4R_0114":
            x_off = 50
            y_off = 15
        elif self.session_name == "mjc163R2R_0114":
            x_off = 0
            y_off = 0
        elif self.session_name == "mjc169R1R_0114":
            x_off = -10
            y_off = 5
        elif self.session_name == "mjc169R4R_0114":
            x_off = 25
            y_off = 15
        elif self.session_name == "mjc163R1L_0114":
            x_off = 38
            y_off = 5
        elif self.session_name == "mjc148R4R_0113":
            x_off = 32
            y_off = 5
        elif self.session_name == "mjc163R3L_0114":
            x_off = 25
            y_off = 20

        rate_maps = self.get_rate_maps(spatial_resolution=spatial_resolution,gaussian_std=gaussian_std)

        for cell_id, rate_map in enumerate(rate_maps.T):
            rate_map[rate_map==0] = np.nan
            plt.imshow(rate_map, cmap="Reds")
            plt.title("CELL "+str(cell_id))
            a = plt.colorbar()
            a.set_label("FIRING RATE (Hz)")
            plt.xlabel("X")
            plt.ylabel("Y")
            for g_l in gl:
                plt.scatter((g_l[0] - self.x_min - x_off) / (2/self.spatial_factor),
                            (g_l[1] - self.y_min - y_off) / (2/self.spatial_factor),
                            label="goal locations", marker="x", color="black", s=30, zorder=1000000)

            plt.show()

    def plot_rate_map(self, cell_id, spatial_resolution=5, save_fig=False, trials_to_use=None):

        plt.style.use('default')
        rate_maps = self.get_rate_maps(spatial_resolution=spatial_resolution, trials_to_use=trials_to_use)
        rate_map_to_plot = rate_maps[:,:,cell_id]
        # rate_map_to_plot[rate_map_to_plot == 0] = np.nan
        plt.imshow(rate_map_to_plot)
        a = plt.colorbar()
        a.set_label("FIRING RATE (Hz)")
        plt.xlabel("X")
        plt.ylabel("Y")

        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "rate_map_example2.svg"), transparent="True")
        else:
            plt.show()

    def plot_summary_rate_map(self, cells_to_use="all", spatial_resolution=3, normalize=False):
        # load all rate maps
        rate_maps = self.get_rate_maps(spatial_resolution=spatial_resolution)

        # load cell labels
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        if cells_to_use == "stable":

            cell_ids = class_dic["stable_cell_ids"].flatten()

        elif cells_to_use == "increasing":

            cell_ids = class_dic["increase_cell_ids"].flatten()

        elif cells_to_use == "decreasing":

            cell_ids = class_dic["decrease_cell_ids"].flatten()

        elif cells_to_use == "all":

            cell_ids = np.arange(rate_maps.shape[2])

        else:
            raise Exception("Needs to select cell subset [stable, decreasing, increasing, all")

        # normalize rate maps
        max_per_cell = np.max(np.reshape(rate_maps, (rate_maps.shape[0]*rate_maps.shape[1], rate_maps.shape[2])), axis=0)
        max_per_cell[max_per_cell == 0] = 1e-22
        norm_rate_maps = rate_maps / max_per_cell

        # compute summed up rate map
        sum_rate_map = np.sum(norm_rate_maps[:, :, cell_ids], axis=2)

        # mask with occupancy
        occ = self.get_occ_map(spatial_resolution=spatial_resolution)
        sum_rate_map[occ==0] = np.nan

        if normalize:
            sum_rate_map = sum_rate_map / np.sum(np.nan_to_num(sum_rate_map.flatten()))
            plt.imshow(sum_rate_map.T)
            a = plt.colorbar()
            a.set_label("Sum firing rate / normalized to 1")
        else:
            plt.imshow(sum_rate_map.T)
            a = plt.colorbar()
            a.set_label("SUMMED FIRING RATE")
        for g_l in self.goal_locations:
            plt.scatter((g_l[0]-self.x_min)/spatial_resolution, (g_l[1]-self.y_min)/spatial_resolution,
                        color="white", label="Goal locations")
        # plt.xlim(55,260)
        plt.title("CELLS USED: " + cells_to_use)
        plt.show()
        # exit()
        # plt.rcParams['svg.fonttype'] = 'none'
        # plt.savefig("dec_firing_changes.svg", transparent="True")

    def plot_tracking(self, ax=None, trials_to_use=None):
        if trials_to_use is None:
            trials_to_use = self.default_trials
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        # plot start box: x_min, x_max, y_min, y_max
        plt.hlines(self.start_box_coordinates[2], self.start_box_coordinates[0], self.start_box_coordinates[1])
        plt.hlines(self.start_box_coordinates[3], self.start_box_coordinates[0], self.start_box_coordinates[1])
        plt.vlines(self.start_box_coordinates[0], self.start_box_coordinates[2], self.start_box_coordinates[3])
        plt.vlines(self.start_box_coordinates[1], self.start_box_coordinates[2], self.start_box_coordinates[3])


        # get location data from trials
        nr_cells = self.trial_raster_list[0].shape[0]
        loc = np.empty((0,2))
        raster = np.empty((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            loc = np.vstack((loc, self.trial_loc_list[trial_id]))
        ax.scatter(loc[:, 0], loc[:, 1], color="grey", s=1, label="TRACKING")
        # plt.ylim(0, 300)
        # plt.xlim(0, 350)
        plt.show()

    def plot_tracking_and_goals(self, ax=None, trials_to_use=None):
        if trials_to_use is None:
            trials_to_use = self.default_trials
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        # get location data from trials
        nr_cells = self.trial_raster_list[0].shape[0]
        loc = np.empty((0,2))
        raster = np.empty((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            loc = np.vstack((loc, self.trial_loc_list[trial_id]))
        ax.scatter(loc[:, 0], loc[:, 1], color="grey", s=1, label="TRACKING")

        # plot actual trajectory
        for part in range(loc.shape[0]-1):
            ax.plot(loc[part:part+2, 0], loc[part:part+2, 1], color="red")

        for g_l in self.goal_locations:
            ax.scatter(g_l[0], g_l[1], marker="x", color="w", label="GOALS")
        plt.show()

    def plot_tracking_and_goals_first_or_last_trial(self, ax=None, save_fig=False, trial="first"):

        if save_fig:
            plt.style.use('default')
        if trial=="first":
            trials_to_use = [0]
        elif trial =="last":
            trials_to_use = [len(self.trial_raster_list) - 1]
        else:
            raise Exception("Define [first] or [last] trial")

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        # get location data from trials
        nr_cells = self.trial_raster_list[0].shape[0]
        loc = np.empty((0,2))
        raster = np.empty((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            loc = np.vstack((loc, self.trial_loc_list[trial_id]))
        # ax.scatter(loc[:, 0], loc[:, 1], color="lightblue", s=1, label="TRACKING")

        # plot actual trajectory
        for part in range(loc.shape[0]-1):
            ax.plot(loc[part:part+2, 0], loc[part:part+2, 1], color="lightblue", label="Tracking")

        for g_l in self.goal_locations:
            ax.scatter(g_l[0], g_l[1], color="black", label="Goals")

        plt.gca().set_aspect('equal', adjustable='box')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.xlim(20, 160)
        plt.ylim(10, 130)
        plt.xlabel("X (cm)")
        plt.ylabel("Y (cm)")

        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, self.experiment_phase+"_tracking_"+trial+"_trial.svg"),
                        transparent="True")
            plt.close()
        else:
            plt.show()

    def plot_spiking(self, save_fig=False):

        # used this session: mjc163R1L_0114

        if save_fig:
            plt.style.use('default')
            c="black"
        else:
            c="white"
        plt.figure(figsize=(3,3))
        for i, cell_id in enumerate(list(self.firing_times.keys())[:3]):
            cell_fir = self.firing_times[cell_id]
            for spike in cell_fir[:10]:
                plt.vlines(spike, i+0.2, i+0.8, color=c)
        plt.xlim(10000, 21000)
        plt.xticks([10000, 30000], [0, 1])
        plt.xlabel("Time (s)", color="grey")
        plt.yticks([0.5, 1.5, 2.5], ["Cell 1", "Cell 2", "Cell 3"], color="grey")
        # plt.gca().tick_params(axis=u'both', which=u'both',length=0)
        plt.tick_params(left = False)
        plt.gca().tick_params(axis='x', colors='grey')
        plt.gca().tick_params(axis='y', colors=c)
        # red lines for binning
        plt.vlines(10000, 0, 3, color="red")
        plt.vlines(14000, 0, 3, color="red")
        plt.vlines(18000, 0, 3, color="red")
        plt.vlines(22000, 0, 3, color="red")
        plt.vlines(26000, 0, 3, color="red")
        plt.vlines(30000, 0, 3, color="red")
        plt.gca().spines['bottom'].set_color('grey')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "presentation_spiking_1.svg"), transparent="True")
            plt.close()
        else:
            plt.show()


        # generate pop vectors
        bin_edges = [10000, 14000, 18000, 22000, 26000, 30000]
        pop_vecs = np.zeros((3, len(bin_edges)-1))
        for i, cell_id in enumerate(list(self.firing_times.keys())[:3]):
            cell_fir = self.firing_times[cell_id]
            # fill pop_vecs
            for bin_id in range(len(bin_edges)-1):
                nr_spikes = np.count_nonzero(np.logical_and(bin_edges[bin_id]<cell_fir, cell_fir<bin_edges[bin_id+1]))
                pop_vecs[i, bin_id] = nr_spikes

        cmap = cm.viridis
        bounds = [0, 1, 2, 3,4, 5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        # cmap = colors.ListedColormap("viridis")
        # bounds=[0,1,2,3,4,5,6]
        # norm = colors.BoundaryNorm(bounds, cmap.N)

        pop_vecs = np.flip(pop_vecs, axis=0)

        fig = plt.figure(figsize=(5, 5))
        gs = fig.add_gridspec(10, 10)
        ax1 = fig.add_subplot(gs[:9, :2])
        ax2 = fig.add_subplot(gs[:9, 2:4])
        ax3 = fig.add_subplot(gs[:9, 4:6])
        ax4 = fig.add_subplot(gs[:9, 6:8])
        ax5 = fig.add_subplot(gs[:9, 8:10])
        ax6 = fig.add_subplot(gs[9:, :8])
        axis_list = [ax1, ax2, ax3, ax4, ax5]

        for i, (pv, ax_) in enumerate(zip(pop_vecs.T, axis_list)):
            if i == 0:
                ax_.set_yticks([0,1,2], ["Cell 1", "Cell 2", "Cell 3"])
                m = ax_.imshow(np.expand_dims(pv, 1), norm=norm)
                ax_.set_xticks([0.1], ["t1"])
            else:
                ax_.set_yticks([])
                ax_.imshow(np.expand_dims(pv, 1), norm=norm)
                ax_.set_xticks([0.1], ["t"+str(i)])
            # ax_.set_xticks([])
            # ax_.set_tick_params(left = False)
        a = fig.colorbar(mappable=m, cax=ax6, orientation="horizontal", norm=norm)
        a.set_label("#spikes")
        a.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5], ["1","2","3","4","5"])
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "presentation_spiking_2.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    def plot_raster(self, save_fig=False):
        fig = plt.figure(figsize=(7, 4))
        if save_fig:
            plt.style.use('default')
        cmap = cm.viridis
        bounds = np.arange(np.max(np.max(self.trial_raster_list[0])))
        norm = colors.BoundaryNorm(bounds, cmap.N)
        plt.imshow(self.trial_raster_list[0],interpolation='nearest', aspect='auto', norm=norm)
        plt.xlabel("Time bins")
        plt.ylabel("Cell ID")
        a = plt.colorbar(norm=norm)
        a.set_label("#spikes")
        a.set_ticks(np.linspace(0.5,7.5, 8))
        plt.xlim(0,400)
        plt.tight_layout()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "presentation_example_raster.svg"), transparent="True")
            plt.close()
        else:
            plt.show()

    # </editor-fold>

    # <editor-fold desc="Saving methods">
    """#################################################################################################################
    #  Saving methods
    #################################################################################################################"""

    def save_goal_coding_all_modes(self, nr_modes, out_file_name):

        trials_to_use = self.default_trials

        file_name = self.session_name + "_" + str(
            int(self.params.experiment_phase_id[0])) + "_" + self.cell_type + "_trials_"+str(trials_to_use[0])+\
                      "_"+str(trials_to_use[-1])+"_"+str(nr_modes)+"_modes"

        frac_per_mode = []
        for mode_id in range(nr_modes):
            frac_close_to_goal = self.analyze_modes_goal_coding(file_name=file_name, mode_ids=mode_id)
            frac_per_mode.append(frac_close_to_goal)

        np.save(out_file_name, np.array(frac_per_mode))
        plt.hist(frac_per_mode, bins=20)
        plt.xlabel("FRACTION AROUND GOAL")
        plt.ylabel("COUNT")
        plt.show()

    def save_raster(self, filename, trials_to_use=None):
        raster, _, _ = self.get_raster_location_speed(trials_to_use=trials_to_use)
        with open(filename+'.pkl', 'wb') as f:
            pickle.dump(raster, f)

    def save_raster_loc_speed(self, trials_to_use=None):
        raster, loc, speed = self.get_raster_location_speed(trials_to_use=trials_to_use)
        file_name = self.session_name + "_" + self.experiment_phase + '.pickle'

        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                dic = pickle.load(f)

            if self.cell_type == "p1":
                dic["p1"] = raster
            elif self.cell_type == "b1":
                dic["b1"] = raster
            with open(file_name, 'wb') as f:
                pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

        else:
            dic = {
                "p1": None,
                "b1": None,
                "loc": loc,
                "time_bin_size": self.time_bin_size
            }

            if self.cell_type == "p1":
                dic["p1"] = raster
            elif self.cell_type == "b1":
                dic["b1"] = raster
            with open(file_name, 'wb') as f:
                pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

    def save_tracking(self, plot_for_control=False):

        tracking_per_trial = self.trial_loc_list
        file_name = os.path.join(save_path, self.session_name + "_" + self.experiment_phase + '_tracking.pickle')
        if plot_for_control:
            plt.scatter(tracking_per_trial[1][:,0],tracking_per_trial[1][:, 1])
            plt.show()

        with open(file_name, 'wb') as f:
            pickle.dump(tracking_per_trial, f, pickle.HIGHEST_PROTOCOL)


    def save_raw_spike_times(self,):
        # --------------------------------------------------------------------------------------------------------------
        # determines spike times of each cell and saves them as a list of list (each list: firing times of one cell)
        # --> used for TreeHMM
        #
        # args:   - save_dir, str
        # --------------------------------------------------------------------------------------------------------------

        spike_times_p1 = self.data_dic["spike_times"]["p1"]
        spike_times_b1 = self.data_dic["spike_times"]["b1"]
        # pickle in using python2 protocol
        with open(self.session_name+"_p1_"+self.experiment_phase, "wb") as f:
            pickle.dump(spike_times_p1, f, protocol=2)

        with open(self.session_name+"_b1_"+self.experiment_phase, "wb") as f:
            pickle.dump(spike_times_b1, f, protocol=2)

    # </editor-fold>

    # <editor-fold desc="Standard analysis">
    """#################################################################################################################
    #  Standard analysis
    #################################################################################################################"""

    def bayesian_decoding(self, test_perc=0.5):

        if self.params.stable_cell_method == "k_means":
            # load only stable cells
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_k_means.pickle", "rb") as f:
                class_dic = pickle.load(f)

        elif self.params.stable_cell_method == "mean_firing_awake":
            # load only stable cells
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_mean_firing_awake.pickle", "rb") as f:
                class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()
        inc_cells = class_dic["increase_cell_ids"].flatten()

        trials_to_use = self.default_trials

        first_trial = trials_to_use[0]
        last_trial = trials_to_use[-1]
        nr_test_trials = int((last_trial-first_trial)*test_perc)

        shuff_trials_to_use = np.array(np.copy(trials_to_use))
        np.random.shuffle(shuff_trials_to_use)

        test_trials = shuff_trials_to_use[:nr_test_trials]
        train_trials = shuff_trials_to_use[nr_test_trials:]

        # get rate map --> need to add x_min, y_min of environment to have proper location info
        rate_maps = self.get_rate_maps(spatial_resolution=1, trials_to_use=train_trials)

        # flatten rate maps
        rate_maps_flat = np.reshape(rate_maps, (rate_maps.shape[0]*rate_maps.shape[1], rate_maps.shape[2]))

        test_raster_orig, test_loc_orig, _ = self.get_raster_location_speed(trials_to_use=test_trials)

        # test_raster = test_raster_orig[:,:50]
        # test_loc = test_loc_orig[:50, :]
        test_raster = test_raster_orig
        test_loc = test_loc_orig

        test_raster_stable = test_raster[stable_cells,:]
        rate_maps_flat_stable = rate_maps_flat[:,stable_cells]
        pred_loc_stable = []
        error_stable=[]
        for pop_vec, loc in zip(test_raster_stable.T, test_loc):
            bl = bayes_likelihood(frm=rate_maps_flat_stable.T, pop_vec=pop_vec, log_likeli=False)
            bl_area = np.reshape(bl, (rate_maps.shape[0], rate_maps.shape[1]))
            pred_bin = np.unravel_index(bl_area.argmax(), bl_area.shape)
            pred_x = pred_bin[0] + self.x_min
            pred_y = pred_bin[1] + self.y_min
            pred_loc_stable.append([pred_x, pred_y])

            # plt.scatter(pred_x, pred_y, color="red")
            # plt.scatter(loc[0], loc[1], color="gray")
            error_stable.append(np.sqrt((pred_x-loc[0])**2+(pred_y-loc[1])**2))
        pred_loc = np.array(pred_loc_stable)

        test_raster_dec = test_raster[dec_cells, :]
        rate_maps_flat_dec = rate_maps_flat[:, dec_cells]
        pred_loc_dec = []
        error_dec = []
        for pop_vec, loc in zip(test_raster_dec.T, test_loc):
            bl = bayes_likelihood(frm=rate_maps_flat_dec.T, pop_vec=pop_vec, log_likeli=False)
            bl_area = np.reshape(bl, (rate_maps.shape[0], rate_maps.shape[1]))
            pred_bin = np.unravel_index(bl_area.argmax(), bl_area.shape)
            pred_x = pred_bin[0] + self.x_min
            pred_y = pred_bin[1] + self.y_min
            pred_loc_stable.append([pred_x, pred_y])

            # plt.scatter(pred_x, pred_y, color="red")
            # plt.scatter(loc[0], loc[1], color="gray")
            error_dec.append(np.sqrt((pred_x - loc[0]) ** 2 + (pred_y - loc[1]) ** 2))
        pred_loc = np.array(pred_loc_stable)

        plt.hist(error_stable, density=True, color="#ffdba1", label="STABLE")
        plt.hist(error_dec, density=True, color="#a0c4e4", label="DECREASING", alpha=0.6)

        plt.xlabel("ERROR (cm)")
        plt.ylabel("DENSITY")
        plt.legend()
        _, y_max = plt.gca().get_ylim()
        plt.vlines(np.median(np.array(error_stable)), 0, y_max, colors="y")
        plt.vlines(np.median(np.array(error_dec)), 0, y_max, colors="b")
        plt.show()

    def nr_spikes_per_time_bin(self, trials_to_use=None, cells_to_use="all_cells"):

        if trials_to_use is None:
            trials_to_use = self.default_trials

        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.zeros((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))

        if cells_to_use == "stable_cells":
            # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            stable_cells = class_dic["stable_cell_ids"].flatten()
            raster = raster[stable_cells, :]

        elif cells_to_use == "decreasing_cells":
                        # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            dec_cells = class_dic["decrease_cell_ids"].flatten()
            raster = raster[dec_cells, :]

        elif cells_to_use == "increasing_cells":
                        # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            inc_cells = class_dic["increase_cell_ids"].flatten()
            raster = raster[inc_cells, :]

        spikes_per_bin = np.sum(raster, axis=0)
        y,_,_ = plt.hist(spikes_per_bin, bins=50)
        plt.vlines(np.mean(spikes_per_bin), 0, y.max(), colors="r",
                   label="MEAN: "+str(np.round(np.mean(spikes_per_bin),2)))
        plt.vlines(np.median(spikes_per_bin), 0, y.max(), colors="blue",
                   label="MEDIAN: "+str(np.median(spikes_per_bin)))
        plt.legend()
        plt.title("#SPIKES PER "+str(self.params.time_bin_size)+"s TIME BIN \n "+cells_to_use)
        plt.xlabel("#SPIKES PER TIME BIN")
        plt.ylabel("COUNT")
        plt.show()

        return np.median(spikes_per_bin)

    def phase_preference_per_cell_subset(self, angle_20k, cell_ids, trials_to_use=None, speed_threshold=5,
                                         plot_for_control=False):
        if trials_to_use is None:
            trials_to_use = range(len(self.trial_raster_list))

        # spike times at 20kHz
        spike_times = self.data_dic["spike_times"][self.cell_type]

        # speed is at 39.0625kHz (20kHz/512) --> 0.0256s bins
        speed = self.get_speed_at_whl_resolution()
        time_axis_orig = np.arange(speed.shape[0])*0.0256
        time_axis_20k = np.arange(speed.shape[0]*512)*0.00005
        # interpolate to match 20kHz --> 0.00005s bins
        speed_20k = np.interp(x=time_axis_20k, xp=time_axis_orig, fp=speed)
        # discard spikes that are during stillness periods
        good_times = speed_20k > speed_threshold
        spikes_times_wo_stillness = {}
        for cell_id, cell_spikes in spike_times.items():
            # cut last spikes if they appeared after the location tracking
            cell_spikes = cell_spikes[cell_spikes<good_times.shape[0]]
            good_spikes = cell_spikes[good_times[cell_spikes]]
            spikes_times_wo_stillness[cell_id] = good_spikes

        if plot_for_control:
            x_axis = np.arange(good_times.shape[0]-5000000, good_times.shape[0]-4000000)
            plt.plot(x_axis, speed_20k[-5000000:-4000000], label="speed"), \
            plt.plot(x_axis, good_times[-5000000:-4000000] * 50, label="good_times")
            a =next(iter(spikes_times_wo_stillness))
            spikes = spikes_times_wo_stillness[a]
            spikes_in_window = spikes[np.logical_and((good_times.shape[0]-5000000)<spikes, spikes<(good_times.shape[0]-4000000))]
            for spike in spikes_in_window:
                plt.vlines(spike, 0, 50, color="w")
            plt.title("spikes in white")
            plt.legend()
            plt.show()

        # get keys from dictionary and get correct order
        cell_names = []
        for key in spikes_times_wo_stillness.keys():
            cell_names.append(key[4:])
        cell_names = np.array(cell_names).astype(int)
        cell_names.sort()

        # start_times, end_times of trials at .whl resolution (20kHz/512) --> up-sample to match spike frequency
        trial_timestamps_20k = self.data_dic["trial_timestamps"] * 512

        pref_angle = []

        for cell_id in cell_names[cell_ids]:
            cell_spike_times = spikes_times_wo_stillness["cell" + str(cell_id)]
            # concatenate trial data
            all_cell_spikes = []
            for trial_id in trials_to_use:
                all_cell_spikes.extend(cell_spike_times[np.logical_and(trial_timestamps_20k[0,trial_id] < cell_spike_times,
                                                       cell_spike_times < trial_timestamps_20k[1,trial_id])])

            # make array
            spk_ang = angle_20k[all_cell_spikes]
            pref_angle.append(np.angle(np.sum(np.exp(-1j * spk_ang))))

        return np.array(pref_angle)

    def phase_preference_analysis(self, oscillation="theta", tetrode=1, plot_for_control=False, plotting=True):

        # get lfp data
        # ["eeg"] --> all channels downsampled 16 times (1.25kHz)
        # ["eegh"] --> one channel per tetrode downsampled 4 times (5kHz) --> time bin size = 0.0002s
        lfp = self.eegh[:, tetrode]

        # downsample to dt = 0.001 --> 1kHz --> take every 5th value
        lfp = lfp[::5]

        # Say you have an LFP signal LFP_Data and some spikes from a cell spk_t
        # First we extract the angle from the signal in a specific frequency band
        # Frequency Range to Extract, you can also select it AFTER running the wavelet on the entire frequency spectrum,
        # by using the variable frequency to select the desired ones
        if oscillation == "theta":
            Frq_Limits = [8, 12]
        elif oscillation == "slow_gamma":
            Frq_Limits = [20, 50]
        elif oscillation == "medium_gamma":
            Frq_Limits = [60, 90]
        else:
            raise Exception("Oscillation not defined!")
        # [8,12] Theta
        # [20,50] Slow Gamma
        # [60,90] Medium Gamma
        # LFP time bin duration in seconds
        # dt = 1/5e3
        dt=0.001
        # ‘morl’ wavelet
        wavelet = "cmor1.5-1.0" # 'cmor1.5-1.0'
        scales = np.arange(1,128)
        s2f = pywt.scale2frequency(wavelet, scales) / dt
        # This block is just to setup the wavelet analysis
        scales = scales[(s2f >= Frq_Limits[0]) * (s2f < Frq_Limits[1])]
        # scales = scales[np.logical_and(s2f >= Frq_Limits[0], s2f < Frq_Limits[1])]
        print(" - started wavelet decomposition ...")
        # Wavelet decomposition
        [cfs, frequencies] = pywt.cwt(data=lfp, scales=scales, wavelet=wavelet, sampling_period=dt, axis=0)
        print(" - done!")
        # This is the angle
        angl = np.angle(np.sum(cfs, axis=0))

        # plot for control
        if plot_for_control:
            plt.plot(lfp[:200])
            plt.xlabel("Time")
            plt.ylabel("LFP")
            plt.show()

            for i in range(frequencies.shape[0]):
                plt.plot(cfs[i, :200])
            plt.xlabel("Time")
            plt.ylabel("Coeff")
            plt.show()

            plt.plot(np.sum(cfs[:, :200], axis=0), label="coeff_sum")
            plt.plot(angl[:200]/np.max(angl[:200]), label="angle")
            plt.xlabel("Time")
            plt.ylabel("Angle (norm) / Coeff_sum (norm)")
            plt.legend()
            plt.show()

        # interpolate results to match 20k
        # --------------------------------------------------------------------------------------------------------------
        x_1k = np.arange(lfp.shape[0])*dt
        x_20k = np.arange(lfp.shape[0]*20)*1/20e3
        angle_20k = np.interp(x_20k, x_1k, angl, left=np.nan, right=np.nan)

        if plot_for_control:
            plt.plot(angle_20k[:4000])
            plt.ylabel("Angle")
            plt.xlabel("Time bin (20kHz)")
            plt.show()

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        stable = class_dic["stable_cell_ids"]
        dec = class_dic["decrease_cell_ids"]
        inc = class_dic["increase_cell_ids"]

        pref_angle_stable = self.phase_preference_per_cell_subset(angle_20k=angle_20k, cell_ids=stable)
        pref_angle_dec = self.phase_preference_per_cell_subset(angle_20k=angle_20k, cell_ids=dec)
        pref_angle_inc = self.phase_preference_per_cell_subset(angle_20k=angle_20k, cell_ids=inc)

        pref_angle_stable_deg = pref_angle_stable *180/np.pi
        pref_angle_dec_deg = pref_angle_dec * 180 / np.pi
        pref_angle_inc_deg = pref_angle_inc * 180 / np.pi

        if plotting:
            plt.hist(pref_angle_stable_deg, density=True, label="stable")
            plt.hist(pref_angle_dec_deg, density=True, label="dec")
            plt.hist(pref_angle_inc_deg, density=True, label="inc")
            plt.show()

        all_positive_angles_stable = np.copy(pref_angle_stable)
        all_positive_angles_stable[all_positive_angles_stable < 0] = 2*np.pi+all_positive_angles_stable[all_positive_angles_stable < 0]

        all_positive_angles_dec = np.copy(pref_angle_dec)
        all_positive_angles_dec[all_positive_angles_dec < 0] = 2 * np.pi + all_positive_angles_dec[
            all_positive_angles_dec < 0]

        all_positive_angles_inc = np.copy(pref_angle_inc)
        all_positive_angles_inc[all_positive_angles_inc < 0] = 2 * np.pi + all_positive_angles_inc[
            all_positive_angles_inc < 0]

        if plotting:

            bins_number = 10  # the [0, 360) interval will be subdivided into this
            # number of equal bins
            bins = np.linspace(0.0, 2 * np.pi, bins_number + 1)
            angles = all_positive_angles_stable
            n, _, _ = plt.hist(angles, bins, density=True)

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

            plt.clf()
            width = 2 * np.pi / bins_number
            ax = plt.subplot(1, 1, 1, projection='polar')
            bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
            for bar in bars:
                bar.set_alpha(0.5)
            ax.set_title("inc. cells")
            plt.show()

        else:
            return all_positive_angles_stable, all_positive_angles_dec, all_positive_angles_inc

    def check_oscillation(self, oscillation="theta", plot_for_control=False, plotting=True):
        if oscillation == "theta":
            Frq_Limits = [8, 12]
        elif oscillation == "slow_gamma":
            Frq_Limits = [20, 50]
        elif oscillation == "medium_gamma":
            Frq_Limits = [60, 90]
        else:
            raise Exception("Oscillation not defined!")
        for tetrode in range(10):
            # get lfp data
            # ["eeg"] --> all channels downsampled 16 times (1.25kHz)
            # ["eegh"] --> one channel per tetrode downsampled 4 times (5kHz) --> time bin size = 0.0002s
            lfp = self.eegh[:, tetrode]

            # downsample to dt = 0.001 --> 1kHz --> take every 5th value
            lfp = lfp[::5]

            # bandpass: freq_nyquist --> sampling freq / 2
            sig_filtered = butterworth_bandpass(input_data=lfp, freq_nyquist=1000 / 2, freq_lo_bound=Frq_Limits[0],
                                                freq_hi_bound=Frq_Limits[1])

            plt.plot(sig_filtered[1000:1300], label="tet" + str(tetrode))
        plt.legend()
        plt.show()

    # </editor-fold>

    # <editor-fold desc="SWR analysis">

    """#################################################################################################################
    #  SWR analysis
    #################################################################################################################"""

    def detect_swr(self, thr=4, plot_for_control=False):
        """
        detects swr in lfp and returns start, peak and end timings at 1 second resolution
        ripple frequency: 140-240 Hz

        @param thr: nr. std. above average to detect ripple event (usually: 4-6)
        @type thr: int
        @param plot_for_control: True to plot intermediate results
        @type plot_for_control: bool
        @return: start, end, peak of each swr in seconds
        @rtype: int, int, int
        """
        if not hasattr(self.session_params, 'lfp_tetrodes'):
            self.session_params.lfp_tetrodes = None

        file_name = self.session_name + "_" + self.experiment_phase_id + "_swr_" + \
                    self.cell_type +"_tet_"+str(self.session_params.lfp_tetrodes)+ ".npy"
        # check if results exist already
        if not os.path.isfile(self.params.pre_proc_dir+"swr_periods/" + file_name):

            # check if results exist already --> if not

            # upper and lower bound in Hz for SWR
            freq_lo_bound = 140
            freq_hi_bound = 240

            # load data
            # ["eeg"] --> all channels downsampled 16 times (1.25kHz)
            # ["eegh"] --> one channel per tetrode downsampled 4 times (5kHz) --> time bin size = 0.0002s

            # check if one tetrode or all tetrodes to use
            if self.session_params.lfp_tetrodes is None:
                print(" - DETECTING SWR USING ALL TETRODES ...\n")
                data = self.eegh[:, :]
            else:
                print(" - DETECTING SWR USING TETRODE(S) "+str(self.session_params.lfp_tetrodes) +" ...\n")
                data = self.eegh[:, self.session_params.lfp_tetrodes]
            # freq of the input signal (eegh --> 5kHz --> freq=5000)
            freq = 5000
            # for low pass filtering of the signal before z-scoring (20-30Hz is good)
            low_pass_cut_off_freq = 30
            # minimum gap in seconds between events. If two events have
            # a gap < min_gap_between_events --> events are joint and become one event
            min_gap_between_events = 0.1

            # if data is too large --> need to chunk it up

            if data.shape[0] > 10000000:
                start_times = np.zeros(0)
                peak_times = np.zeros(0)
                end_times = np.zeros(0)
                size_chunk = 10000000
                for nr_chunk in range(np.ceil(data.shape[0]/size_chunk).astype(int)):
                    chunk_data = data[nr_chunk*size_chunk:min(data.shape[0], (nr_chunk+1)*size_chunk)]

                    # compute offset in seconds for current chunk
                    offset_sec = nr_chunk * size_chunk * 1/freq

                    start_times_chunk, end_times_chunk, peak_times_chunk = self.detect_lfp_events(data=chunk_data,
                                                                                freq=freq, thr=thr,
                                                                                freq_lo_bound=freq_lo_bound,
                                                                                freq_hi_bound=freq_hi_bound,
                                                                                low_pass_cut_off_freq=low_pass_cut_off_freq,
                                                                                min_gap_between_events=min_gap_between_events,
                                                                                plot_for_control=plot_for_control)

                    # check if event was detected
                    if not start_times_chunk is None:
                        start_times = np.hstack((start_times, (start_times_chunk + offset_sec)))
                        end_times = np.hstack((end_times, (end_times_chunk + offset_sec)))
                        peak_times = np.hstack((peak_times, (peak_times_chunk + offset_sec)))

            else:
                # times in seconds
                start_times, end_times, peak_times = self.detect_lfp_events(data=data, freq=freq, thr=thr,
                                                                            freq_lo_bound=freq_lo_bound,
                                                                            freq_hi_bound=freq_hi_bound,
                                                                            low_pass_cut_off_freq=low_pass_cut_off_freq,
                                                                            min_gap_between_events=min_gap_between_events,
                                                                            plot_for_control=plot_for_control)

            result_dic = {
                "start_times": start_times,
                "end_times": end_times,
                "peak_times": peak_times
            }

            outfile = open(self.params.pre_proc_dir+"swr_periods/"+file_name, 'wb')
            pickle.dump(result_dic, outfile)
            outfile.close()

        # load results from file
        infile = open(self.params.pre_proc_dir+"swr_periods/" + file_name, 'rb')
        result_dic = pickle.load(infile)
        infile.close()

        start_times = result_dic["start_times"]
        end_times = result_dic["end_times"]
        peak_times = result_dic["peak_times"]

        print(" - " + str(start_times.shape[0]) + " SWRs FOUND\n")

        return start_times, end_times, peak_times

    @staticmethod
    def detect_lfp_events(data, freq, thr, freq_lo_bound, freq_hi_bound, low_pass_cut_off_freq,
                          min_gap_between_events, plot_for_control=False):
        """
        detects events in lfp and returns start, peak and end timings at params.time_bin_size resolution

        @param data: input data (either from one or many tetrodes)
        @type data: array [nxm]
        @param freq: sampling frequency of input data in Hz
        @type freq: int
        @param thr: nr. std. above average to detect ripple event
        @type thr: int
        @param freq_lo_bound: lower bound for frequency band in Hz
        @type freq_lo_bound: int
        @param freq_hi_bound: upper bound for frequency band in Hz
        @type freq_hi_bound: int
        @param low_pass_cut_off_freq: cut off frequency for envelope in Hz
        @type low_pass_cut_off_freq: int
        @param min_gap_between_events: minimum gap in seconds between events. If two events have
         a gap < min_gap_between_events --> events are joint and become one event
        @type min_gap_between_events: float
        @param plot_for_control: plot some examples to double check detection
        @type plot_for_control: bool
        @return: start_times, end_times, peak_times of each event in seconds --> are all set to None if no event was
        detected
        @rtype: array, array, array
        """

        # check if data from one or multiple tetrodes was provided
        if len(data.shape) == 1:
            # only one tetrode
            # nyquist theorem --> need half the frequency
            sig_bandpass = butterworth_bandpass(input_data=data, freq_nyquist=freq / 2, freq_lo_bound=freq_lo_bound,
                                                freq_hi_bound=freq_hi_bound)

            # compute rectified signal
            sig_abs = np.abs(sig_bandpass)

            # if only peak position is supposed to be returned

            # low pass filter signal
            sig_lo_pass = butterworth_lowpass(input_data=sig_abs, freq_nyquist=freq / 2,
                                              cut_off_freq=low_pass_cut_off_freq)
            # z-score
            sig_z_scored = zscore(sig_lo_pass)

        else:
            # multiple tetrodes
            combined_lo_pass = []
            # go trough all tetrodes
            for tet_data in data.T:
                # nyquist theorem --> need half the frequency
                sig_bandpass = butterworth_bandpass(input_data=tet_data, freq_nyquist=freq/2, freq_lo_bound=freq_lo_bound,
                                                           freq_hi_bound=freq_hi_bound)

                # compute rectified signal
                sig_abs = np.abs(sig_bandpass)

                # if only peak position is supposed to be returned

                # low pass filter signal
                sig_lo_pass = butterworth_lowpass(input_data=sig_abs, freq_nyquist=freq/2,
                                                  cut_off_freq=low_pass_cut_off_freq)

                combined_lo_pass.append(sig_lo_pass)

            combined_lo_pass = np.array(combined_lo_pass)
            avg_lo_pass = np.mean(combined_lo_pass, axis=0)

            # z-score
            sig_z_scored = zscore(avg_lo_pass)

        # find entries above the threshold
        bool_above_thresh = sig_z_scored > thr
        sig = bool_above_thresh.astype(int) * sig_z_scored

        # find event start / end
        diff = np.diff(sig)
        start = np.argwhere(diff > 0.8 * thr)
        end = np.argwhere(diff < -0.8 * thr)

        # check that first element is actually the start (not that event started before this chunk and we only
        # observe the end of the event)
        if end[0] < start[0]:
            # if first end is before first start --> need to delete first end
            print("  --> CURRENT CHUNK: FIRST END BEFORE FIRST START --> DELETED FIRST END ELEMENT ")
            end = end[1:]

        if end[-1] < start[-1]:
            # there is another start after the last end --> need to delete last start
            print("  --> CURRENT CHUNK: LAST START AFTER LAST END --> DELETED LAST START ELEMENT ")
            start = start[:-1]

        # join events if there are less than min_gap_between_events seconds apart --> this is then one event!
        # compute differences between start time of n+1th event with end time of nth --> if < gap --> delete both
        # entries
        gap = np.squeeze((start[1:] - end[:-1]) * 1 / freq)
        to_delete = np.argwhere(gap < min_gap_between_events)
        end = np.delete(end, to_delete)
        start = np.delete(start, to_delete + 1)

        # add 25ms to the beginning of event (many spikes occur in that window)
        pad_infront = np.round(0.025/(1/freq)).astype(int)
        start -= pad_infront
        # don't want negative values (in case event happens within the 50ms of the recording)
        start[start < 0] = 0

        # # add 20ms to the end of event
        # pad_end = np.round(0.02/(1/freq)).astype(int)
        # end += pad_end
        # # don't want to extend beyond the recording
        # end[end > sig.shape[0]] = sig.shape[0]

        # check length of events --> shouldn't be shorter than 95 ms or larger than 750 ms
        len_events = (end - start) * 1 / freq
        #
        # plt.hist(len_events, bins=50)
        # plt.show()
        # exit()

        to_delete_len = np.argwhere((0.75 < len_events) | (len_events < 0.05))

        start = np.delete(start, to_delete_len)
        end = np.delete(end, to_delete_len)

        peaks = []
        for s, e in zip(start,end):
            peaks.append(s+np.argmax(sig[s:e]))

        peaks = np.array(peaks)

        # check if there were any events detected --> if not: None
        if not peaks.size == 0:
            # get peak times in s
            time_bins = np.arange(data.shape[0]) * 1 / freq
            peak_times = time_bins[peaks]
            start_times = time_bins[start]
            end_times = time_bins[end]
        else:
            peak_times = None
            start_times = None
            end_times = None

        # plot some events with start, peak and end for control
        if plot_for_control:
            a = np.random.randint(0, start.shape[0], 5)
            # a = range(start.shape[0])
            for i in a:
                plt.plot(sig_z_scored, label="z-scored signal")
                plt.vlines(start[i], 0, 15, colors="r", label="start")
                plt.vlines(peaks[i], 0, 15, colors="y", label="peak")
                plt.vlines(end[i], 0, 15, colors="g", label="end")
                plt.xlim((start[i] - 5000),(end[i] + 5000))
                plt.ylabel("LFP FILTERED (140-240Hz) - Z-SCORED")
                plt.xlabel("TIME BINS / "+str(1/freq) + " s")
                plt.legend()
                plt.title("EVENT DETECTION, EVENT ID "+str(i))
                plt.show()

        return start_times, end_times, peak_times

    def firing_rates_during_swr(self, time_bin_size=0.01, plotting=True):

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        inc_cells = class_dic["increase_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()

        start_times, end_times, peak_times = self.detect_swr()
        event_times = np.vstack((start_times, end_times)).T

        swr_rasters = []
        # get within swr rasters
        for e_t in event_times:
            swr_rasters.append(PreProcessAwake(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl).interval_temporal_binning(interval=e_t, interval_freq=1,
                                                                            time_bin_size=time_bin_size))

        swr_raster = np.hstack(swr_rasters)

        swr_mean_firing_rates = np.mean(swr_raster, axis=1)/time_bin_size

        swr_mean_firing_rates_stable = swr_mean_firing_rates[stable_cells]
        swr_mean_firing_rates_inc = swr_mean_firing_rates[inc_cells]
        swr_mean_firing_rates_dec = swr_mean_firing_rates[dec_cells]

        if plotting:
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
        else:
            return swr_mean_firing_rates_stable, swr_mean_firing_rates_dec, swr_mean_firing_rates_inc

    def firing_rates_gain_during_swr(self, time_bin_size=0.01, plotting=True, threshold_stillness=5,
                                     threshold_firing=0.1, plot_for_control=False):

        # get entire raster to determine firing rates
        # --------------------------------------------------------------------------------------------------------------
        # get entire raster to determine mean firing
        entire_raster = self.get_raster(trials_to_use="all")
        mean_firing = np.mean(entire_raster, axis=1)/self.params.time_bin_size
        cells_above_threshold = mean_firing > threshold_firing

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        inc_cells = class_dic["increase_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()

        # only use cells above the firing threshold
        dec_cells = dec_cells[cells_above_threshold[dec_cells]]
        stable_cells = stable_cells[cells_above_threshold[stable_cells]]
        inc_cells = inc_cells[cells_above_threshold[inc_cells]]

        stillness_periods = self.get_stillness_periods(threshold_stillness=threshold_stillness)
        start_times, end_times, peak_times = self.detect_swr()
        swr_times = np.vstack((start_times, end_times)).T

        # now we need to separate stillness periods and swr times

        stillness_periods_wo_swr = []
        # go through all stillness periods and check if it contains SWRs
        for still_per in stillness_periods:
            swr_in_period = False
            still_per_start = still_per[0]
            still_per_end = still_per[1]
            for swr_t in swr_times:
                swr_start = swr_t[0]
                swr_end = swr_t[1]
                if swr_start < still_per_start and (still_per_start < swr_end and swr_end < still_per_end):
                    # swr start before stillness period and extends within the stillness period --> need to change start
                    # of stillness period
                    still_per_start = swr_end
                elif still_per_start < swr_start and swr_end < still_per_end:
                    # swr is entirely contained in stillness period --> include part before swr
                    stillness_periods_wo_swr.append([still_per_start, swr_start])
                    # set end of swr as the start of the new section of the stillness period
                    still_per_start = swr_end
                elif swr_start < still_per_end and swr_end > still_per_end:
                    # swr starts at the end of the stillness period and extends beyond --> need to close last chunk
                    still_per_end = swr_start
            stillness_periods_wo_swr.append([still_per_start, still_per_end])

        if plot_for_control:
            cmap = matplotlib.cm.get_cmap('tab10')
            for sp in stillness_periods:
                plt.hlines(1, sp[0], sp[1])
            for sp in swr_times:
                plt.scatter(sp[0], 1, color="red")
            for i, sp in enumerate(stillness_periods_wo_swr):
                plt.hlines(0.9, sp[0], sp[1], color=cmap(i))
            plt.xlim(200, 280)
            plt.show()

        stillness_periods_wo_swr = np.vstack(stillness_periods_wo_swr)

        print("Computing stillness rasters ...")
        # get stillness rasters
        stillness_raster = []
        for e_t in stillness_periods_wo_swr:
            stillness_raster.append(PreProcessAwake(firing_times=self.firing_times, params=self.params,
              whl=self.whl, spatial_factor=self.spatial_factor).interval_temporal_binning(interval=e_t, interval_freq=1,
                                                                            time_bin_size=time_bin_size))
        print("Done!")
        # remove Nones (error in computing raster)
        stillness_raster = [x for x in stillness_raster if not x is None]
        stillness_raster = np.hstack(stillness_raster)
        stillness_mean_firing_rates = np.mean(stillness_raster, axis=1)/time_bin_size
        stillness_mean_firing_rates_stable = stillness_mean_firing_rates[stable_cells]
        stillness_mean_firing_rates_inc = stillness_mean_firing_rates[inc_cells]
        stillness_mean_firing_rates_dec = stillness_mean_firing_rates[dec_cells]

        # now get SWR data
        # --------------------------------------------------------------------------------------------------------------
        print("Computing SWR raster ....")
        start_times, end_times, peak_times = self.detect_swr()
        event_times = np.vstack((start_times, end_times)).T

        swr_rasters = []
        # get within swr rasters
        for e_t in event_times:
            swr_rasters.append(PreProcessAwake(firing_times=self.firing_times, params=self.params,
              whl=self.whl, spatial_factor=self.spatial_factor).interval_temporal_binning(interval=e_t, interval_freq=1,
                                                        time_bin_size=time_bin_size))

        print("Done!")
        swr_raster = np.hstack(swr_rasters)

        swr_mean_firing_rates = np.mean(swr_raster, axis=1)/time_bin_size

        swr_mean_firing_rates_stable = swr_mean_firing_rates[stable_cells]
        swr_mean_firing_rates_inc = swr_mean_firing_rates[inc_cells]
        swr_mean_firing_rates_dec = swr_mean_firing_rates[dec_cells]

        # compute swr gain

        swr_gain_stable = swr_mean_firing_rates_stable / stillness_mean_firing_rates_stable
        swr_gain_inc = swr_mean_firing_rates_inc / stillness_mean_firing_rates_inc
        swr_gain_dec = swr_mean_firing_rates_dec / stillness_mean_firing_rates_dec

        if plotting:

            p_stable = 1. * np.arange(swr_gain_stable.shape[0]) / (swr_gain_stable.shape[0] - 1)
            p_inc = 1. * np.arange(swr_gain_inc.shape[0]) / (swr_gain_inc.shape[0] - 1)
            p_dec = 1. * np.arange(swr_gain_dec.shape[0]) / (swr_gain_dec.shape[0] - 1)

            plt.plot(np.sort(swr_gain_stable), p_stable, color="violet", label="stable")
            plt.plot(np.sort(swr_gain_dec), p_dec, color="turquoise", label="dec")
            plt.plot(np.sort(swr_gain_inc), p_inc, color="orange", label="inc")
            plt.ylabel("cdf")
            plt.xlabel("SWR gain")
            plt.legend()
            plt.show()

        else:
            return swr_gain_stable, swr_gain_dec, swr_gain_inc

    def firing_rates_around_swr(self, time_bin_size=0.01, plotting=True, threshold_firing=1):

        # get gain data
        # swr_gain_stable, swr_gain_dec, swr_gain_inc = self.firing_rates_gain_during_swr(plotting=False)

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        inc_cells = class_dic["increase_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()

        start_times, end_times, peak_times = self.detect_swr()
        event_times = np.vstack((peak_times-5, peak_times+5)).T

        swr_rasters = []
        # get within swr rasters
        for e_t in event_times:
            swr_rasters.append(PreProcessAwake(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl).interval_temporal_binning(interval=e_t, interval_freq=1,
                                                                            time_bin_size=time_bin_size))
        swr_raster = np.hstack(swr_rasters)

        # z-score using all values during stillness
        stillness_raster = self.get_raster_stillness(time_bin_size=time_bin_size)
        mean_stillness=np.mean(stillness_raster, axis=1)
        std_stillness=np.std(stillness_raster, axis=1)

        # get entire raster to determine mean firing
        entire_raster = self.get_raster(trials_to_use="all")
        mean_firing = np.mean(entire_raster, axis=1)/self.params.time_bin_size
        cells_above_threshold = mean_firing > threshold_firing
        dec_cells = dec_cells[cells_above_threshold[dec_cells]]
        stable_cells = stable_cells[cells_above_threshold[stable_cells]]
        inc_cells = inc_cells[cells_above_threshold[inc_cells]]
        # compute mean firing rate for decreasing cells, than use mean and std of mean firing cells during stillness
        # to z-score
        all_dec_data = []
        for raster in swr_rasters:
            all_dec_data.append(np.mean(raster[dec_cells, :], axis=0))
        all_dec_data = np.vstack(all_dec_data)

        mean_dec_cells_stillness = np.mean(np.mean(stillness_raster[dec_cells,:], axis=1))
        std_dec_cells_stillness = np.std(np.mean(stillness_raster[dec_cells,:], axis=1))

        all_dec_data_z = (all_dec_data-mean_dec_cells_stillness)/std_dec_cells_stillness

        plot_range = 500
        plt.subplot(2, 1, 1)
        plt.plot(np.mean(all_dec_data_z, axis=0)[int((all_dec_data_z.shape[1] / 2) - int(plot_range / 2)):(int(all_dec_data_z.shape[1] / 2) + int(plot_range / 2))])
        plt.xlim(0, plot_range)
        plt.gca().get_xaxis().set_ticks([])
        plt.ylabel("Mean firing rate (z-sco.)")
        plt.title("Decreasing cells")
        plt.subplot(2, 1, 2)
        plt.imshow(all_dec_data_z[:, int((all_dec_data_z.shape[1] / 2) - int(plot_range / 2)):(
                    int(all_dec_data_z.shape[1] / 2) + int(plot_range / 2))] / time_bin_size,
            interpolation='nearest', aspect='auto')
        plt.ylabel("SWR ID")
        plt.xticks(np.arange(0, plot_range + 1, plot_range / 5),
                   (np.arange(-int(plot_range / 2), int(plot_range / 2) + 1, plot_range / 5) * time_bin_size).astype(
                       str))
        plt.xlabel("Offset from SWR peak (s)")
        plt.show()

        # compute mean firing rate for stable cells, than use mean and std of mean firing cells during stillness
        # to z-score
        all_stable_data = []
        for raster in swr_rasters:
            all_stable_data.append(np.mean(raster[stable_cells, :], axis=0))
        all_stable_data = np.vstack(all_stable_data)

        mean_stable_cells_stillness = np.mean(np.mean(stillness_raster[stable_cells,:], axis=1))
        std_stable_cells_stillness = np.std(np.mean(stillness_raster[stable_cells,:], axis=1))

        all_stable_data_z = (all_stable_data-mean_stable_cells_stillness)/std_stable_cells_stillness

        plot_range = 500
        plt.subplot(2, 1, 1)
        plt.plot(np.mean(all_stable_data_z, axis=0)[int((all_stable_data_z.shape[1] / 2) - int(plot_range / 2)):(int(all_stable_data_z.shape[1] / 2) + int(plot_range / 2))])
        plt.xlim(0, plot_range)
        plt.gca().get_xaxis().set_ticks([])
        plt.ylabel("Mean firing rate (z-sco.)")
        plt.title("Stable cells")
        plt.subplot(2, 1, 2)
        plt.imshow(all_stable_data_z[:, int((all_stable_data_z.shape[1] / 2) - int(plot_range / 2)):(
                    int(all_stable_data_z.shape[1] / 2) + int(plot_range / 2))] / time_bin_size,
            interpolation='nearest', aspect='auto')
        plt.ylabel("SWR ID")
        plt.xticks(np.arange(0, plot_range + 1, plot_range / 5),
                   (np.arange(-int(plot_range / 2), int(plot_range / 2) + 1, plot_range / 5) * time_bin_size).astype(
                       str))
        plt.xlabel("Offset from SWR peak (s)")
        plt.show()

        # for each cell compute z-scored firing around SWR and take the mean across cells

        # stable cells
        all_data_z_scored = []
        for raster in swr_rasters:
            all_data_z_scored.append(np.divide((raster-np.tile(mean_stillness,(raster.shape[1],1)).T),
                                               np.tile(std_stillness,(raster.shape[1],1)).T))


        all_data_z_scored_stable = []
        for raster in all_data_z_scored:
            all_data_z_scored_stable.append(np.mean(raster[stable_cells, :], axis=0))
        all_data_z_scored_stable = np.vstack(all_data_z_scored_stable)

        plot_range = 200
        plt.subplot(2, 1, 1)
        plt.plot(np.mean(all_data_z_scored_stable, axis=0)[int((all_data_z_scored_stable.shape[1] / 2) - int(plot_range / 2)):(int(all_data_z_scored_stable.shape[1] / 2) + int(plot_range / 2))])
        plt.xlim(0, plot_range)
        plt.gca().get_xaxis().set_ticks([])
        plt.ylabel("Mean firing rate (z-sco.)")
        plt.title("Stable cells")
        plt.subplot(2, 1, 2)
        plt.imshow(all_data_z_scored_stable[:, int((all_data_z_scored_stable.shape[1] / 2) - int(plot_range / 2)):(
                    int(all_data_z_scored_stable.shape[1] / 2) + int(plot_range / 2))] / time_bin_size,
            interpolation='nearest', aspect='auto')
        plt.ylabel("SWR ID")
        plt.xticks(np.arange(0, plot_range + 1, plot_range / 5),
                   (np.arange(-int(plot_range / 2), int(plot_range / 2) + 1, plot_range / 5) * time_bin_size).astype(
                       str))
        plt.xlabel("Offset from SWR peak (s)")
        plt.show()

        # dec cells
        all_data_z_scored_dec = []
        for raster in all_data_z_scored:
            all_data_z_scored_dec.append(np.mean(raster[dec_cells, :], axis=0))
        all_data_z_scored_dec = np.vstack(all_data_z_scored_dec)

        plot_range = 200
        plt.subplot(2, 1, 1)
        plt.plot(np.mean(all_data_z_scored_dec, axis=0)[int((all_data_z_scored_dec.shape[1] / 2) - int(plot_range / 2)):(int(all_data_z_scored_dec.shape[1] / 2) + int(plot_range / 2))])
        plt.xlim(0, plot_range)
        plt.gca().get_xaxis().set_ticks([])
        plt.ylabel("Mean firing rate (z-sco.)")
        plt.title("Decreasing cells")
        plt.subplot(2, 1, 2)
        plt.imshow(all_data_z_scored_dec[:, int((all_data_z_scored_dec.shape[1] / 2) - int(plot_range / 2)):(
                    int(all_data_z_scored_dec.shape[1] / 2) + int(plot_range / 2))] / time_bin_size,
            interpolation='nearest', aspect='auto')
        plt.ylabel("SWR ID")
        plt.xticks(np.arange(0, plot_range + 1, plot_range / 5),
                   (np.arange(-int(plot_range / 2), int(plot_range / 2) + 1, plot_range / 5) * time_bin_size).astype(
                       str))
        plt.xlabel("Offset from SWR peak (s)")
        plt.show()


        if plotting:
            plot_range = 500
            all_cell_data = []
            for raster in swr_rasters:
                all_cell_data.append(np.mean(raster[dec_cells, :], axis=0))
            all_data = np.vstack(all_cell_data)
            plt.subplot(2, 1, 1)
            plt.plot(zscore(
                np.mean(all_data, axis=0)[
                int((all_data.shape[1] / 2) - int(plot_range / 2)):(int(all_data.shape[1] / 2) + int(plot_range / 2))]))
            plt.xlim(0, plot_range)
            plt.gca().get_xaxis().set_ticks([])
            plt.ylabel("Mean firing rate (z-sco.)")
            plt.title("Decreasing cells")
            plt.subplot(2, 1, 2)
            plt.imshow(
                all_data[:, int((all_data.shape[1] / 2) - int(plot_range / 2)):(
                            int(all_data.shape[1] / 2) + int(plot_range / 2))] / time_bin_size,
                interpolation='nearest', aspect='auto')
            plt.ylabel("SWR ID")
            plt.xticks(np.arange(0, plot_range + 1, plot_range / 5),
                       (np.arange(-int(plot_range / 2), int(plot_range / 2) + 1, plot_range / 5) * time_bin_size).astype(
                           str))
            plt.xlabel("Offset from SWR peak (s)")
            plt.show()

            all_cell_data = []
            for raster in swr_rasters:
                all_cell_data.append(np.mean(raster[stable_cells, :], axis=0))
            all_data = np.vstack(all_cell_data)
            plt.subplot(2, 1, 1)
            plt.plot(zscore(
                np.mean(all_data, axis=0)[
                int((all_data.shape[1] / 2) - int(plot_range / 2)):(int(all_data.shape[1] / 2) + int(plot_range / 2))]))
            plt.xlim(0, plot_range)
            plt.gca().get_xaxis().set_ticks([])
            plt.ylabel("Mean firing rate (z-sco.)")
            plt.title("Stable cells")
            plt.subplot(2, 1, 2)
            plt.imshow(
                all_data[:, int((all_data.shape[1] / 2) - int(plot_range / 2)):(
                            int(all_data.shape[1] / 2) + int(plot_range / 2))] / time_bin_size,
                interpolation='nearest', aspect='auto')
            plt.ylabel("SWR ID")
            plt.xticks(np.arange(0, plot_range + 1, plot_range / 5),
                       (np.arange(-int(plot_range / 2), int(plot_range / 2) + 1, plot_range / 5) * time_bin_size).astype(
                           str))
            plt.xlabel("Offset from SWR peak (s)")
            plt.show()

        else:
            return swr_raster[stable_cells, :], swr_raster[dec_cells, :], swr_raster[inc_cells, :]

    # </editor-fold>

    # <editor-fold desc="Learning">

    """#################################################################################################################
    #  learning
    #################################################################################################################"""

    def map_dynamics_learning(self, nr_shuffles=500, plot_results=True, n_trials=5,
                              spatial_resolution=3, adjust_pv_size=False):

        # get maps for first trials
        initial = self.get_rate_maps(trials_to_use=range(n_trials), spatial_resolution=spatial_resolution)
        initial_occ = self.get_occ_map(trials_to_use=range(n_trials), spatial_resolution=spatial_resolution)
        # get maps for last trial
        last = self.get_rate_maps(trials_to_use=range(len(self.trial_loc_list)-n_trials, len(self.trial_loc_list)),
                                  spatial_resolution=spatial_resolution)
        last_occ = self.get_occ_map(trials_to_use=range(len(self.trial_loc_list)-n_trials, len(self.trial_loc_list)),
                                    spatial_resolution=spatial_resolution)

        # load cell labels
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()
        nr_stable_cells = stable_cells.shape[0]
        nr_dec_cells = dec_cells.shape[0]

        # compute remapping based on population vectors per bin
        # --------------------------------------------------------------------------------------------------------------

        pop_vec_initial = np.reshape(initial,
                                 (initial.shape[0] * initial.shape[1], initial.shape[2]))
        pop_vec_last = np.reshape(last, (last.shape[0] * last.shape[1], last.shape[2]))

        # only select spatial bins that were visited in PRE and POST
        comb_occ_map = np.logical_and(initial_occ.flatten() > 0, last_occ.flatten() > 0)
        pop_vec_initial = pop_vec_initial[comb_occ_map, :]
        pop_vec_last = pop_vec_last[comb_occ_map, :]

        comb_occ_map_spatial = np.reshape(comb_occ_map,(initial.shape[0], initial.shape[1]))
        common_loc = np.where(comb_occ_map_spatial)
        # stable cells

        pop_vec_initial_stable = pop_vec_initial[:, stable_cells]
        pop_vec_last_stable = pop_vec_last[:, stable_cells]

        plt.imshow(pop_vec_initial_stable.T, interpolation='none', aspect='auto')
        plt.title("First 5 trials")
        plt.ylabel("Stable cells")
        plt.xlabel("Spatial bins")
        plt.show()

        plt.imshow(pop_vec_last_stable.T, interpolation='none', aspect='auto')
        plt.title("Last 5 trials")
        plt.ylabel("Stable cells")
        plt.xlabel("Spatial bins")
        plt.show()

        remapping_pv_stable = []

        for pre, post in zip(pop_vec_initial_stable, pop_vec_last_stable):

            remapping_pv_stable.append(pearsonr(pre.flatten(), post.flatten())[0])

        remapping_pv_stable = np.nan_to_num(np.array(remapping_pv_stable))
        # remapping_pv_stable = np.array(remapping_pv_stable)



        if plot_results:
            cmap = plt.cm.get_cmap('jet')
            # plotting
            for r, x, y in zip(remapping_pv_stable, common_loc[0], common_loc[1]):
                plt.scatter(x,y,color=cmap(r))
            a = plt.colorbar(cm.ScalarMappable(cmap=cmap))
            a.set_label("PV Pearson R")

            for g_l in self.goal_locations:
                plt.scatter((g_l[0]-self.x_min)/spatial_resolution, (g_l[1]-self.y_min)/spatial_resolution,
                            color="w",marker="x" )
            plt.title("Stable cells (mean="+str(np.round(np.mean(remapping_pv_stable),4))+")")
            plt.show()

        pop_vec_initial_dec = pop_vec_initial[:, dec_cells]
        pop_vec_last_dec = pop_vec_last[:, dec_cells]

        if adjust_pv_size:
            print("Adjusting PV size ...")

            # check if there are more dec cells than stable
            if nr_dec_cells < nr_stable_cells:
                raise Exception("Cannot adjust PV size: there are less decreasing cells than stable ones")

            remapping_pv_dec = []
            for i in range(nr_shuffles):
                # pick random cells from population vector
                rand_cells = np.random.randint(low=0, high=pop_vec_initial_dec.shape[1], size=nr_stable_cells)
                pop_vec_pre_dec_sub = pop_vec_initial_dec[:, rand_cells]
                pop_vec_post_dec_sub = pop_vec_last_dec[:, rand_cells]

                remapping_pv_dec_sub = np.zeros(pop_vec_pre_dec_sub.shape[0])
                for i, (pre, post) in enumerate(zip(pop_vec_pre_dec_sub, pop_vec_post_dec_sub)):
                    remapping_pv_dec_sub[i] = pearsonr(pre.flatten(), post.flatten())[0]
                remapping_pv_dec.append(np.nan_to_num(remapping_pv_dec_sub))

            remapping_pv_dec = np.mean(np.vstack(remapping_pv_dec), axis=0)

        else:

            remapping_pv_dec = []

            for pre, post in zip(pop_vec_initial_dec, pop_vec_last_dec):
                remapping_pv_dec.append(pearsonr(pre.flatten(), post.flatten())[0])

            remapping_pv_dec = np.nan_to_num(np.array(remapping_pv_dec))

        plt.imshow(pop_vec_initial_dec.T, interpolation='none', aspect='auto')
        plt.title("First 5 trials")
        plt.ylabel("Decreasing cells")
        plt.xlabel("Spatial bins")
        plt.show()

        plt.imshow(pop_vec_last_dec.T, interpolation='none', aspect='auto')
        plt.title("Last 5 trials")
        plt.ylabel("Decreasing cells")
        plt.xlabel("Spatial bins")
        plt.show()

        if plot_results:
            cmap = plt.cm.get_cmap('jet')
            # plotting
            for r, x, y in zip(remapping_pv_dec, common_loc[0], common_loc[1]):
                plt.scatter(x,y,color=cmap(r))
            a = plt.colorbar(cm.ScalarMappable(cmap=cmap))
            a.set_label("PV Pearson R")

            for g_l in self.goal_locations:
                plt.scatter((g_l[0]-self.x_min)/spatial_resolution, (g_l[1]-self.y_min)/spatial_resolution,
                            color="w",marker="x" )
            plt.title("Decreasing cells (mean="+str(np.round(np.mean(remapping_pv_dec),4))+")")
            plt.show()

        remapping_pv_stable_shuffle = []
        for i in range(nr_shuffles):
            shuffle_res = []
            per_ind = np.random.permutation(np.arange(pop_vec_last_stable.shape[0]))
            shuffled_pop_vec_post = pop_vec_last_stable[per_ind, :]
            for pre, post in zip(pop_vec_initial_stable, shuffled_pop_vec_post):
                shuffle_res.append(pearsonr(pre.flatten(), post.flatten())[0])
            remapping_pv_stable_shuffle.append(shuffle_res)

        remapping_pv_stable_shuffle = np.array(remapping_pv_stable_shuffle)
        remapping_pv_stable_shuffle_flat = remapping_pv_stable_shuffle.flatten()

        remapping_pv_stable_sorted = np.sort(remapping_pv_stable)
        remapping_pv_stable_shuffle_sorted = np.sort(remapping_pv_stable_shuffle_flat)

        # compute statistics
        # _, p = ks_2samp(remapping_pv_stable, remapping_pv_stable_shuffle_flat)

        # dec cells

        remapping_pv_dec_shuffle = []
        for i in range(nr_shuffles):
            shuffle_res = []
            per_ind = np.random.permutation(np.arange(pop_vec_last_dec.shape[0]))
            shuffled_pop_vec_post = pop_vec_last_dec[per_ind, :]
            for pre, post in zip(pop_vec_initial_dec, shuffled_pop_vec_post):
                shuffle_res.append(pearsonr(pre.flatten(), post.flatten())[0])
            remapping_pv_dec_shuffle.append(shuffle_res)

        remapping_pv_dec_shuffle = np.array(remapping_pv_dec_shuffle)
        remapping_pv_dec_shuffle_flat = remapping_pv_dec_shuffle.flatten()

        remapping_pv_dec_sorted = np.sort(remapping_pv_dec)
        remapping_pv_dec_shuffle_sorted = np.sort(remapping_pv_dec_shuffle_flat)

        # --------------------------------------------------------------------------------------------------------------
        # compute remapping (correlation of rate maps early learning vs. late learning)
        remapping = []

        for early, late in zip(initial.T, last.T):
            if np.count_nonzero(early) > 0 and np.count_nonzero(late) > 0:
                remapping.append(pearsonr(early.flatten(), late.flatten())[0])
            else:
                remapping.append(np.nan)

        remapping = np.array(remapping)
        # compute shuffled data
        remapping_shuffle = []
        for pre, post in zip(initial.T, last.T):
            shuffle_list = []
            post_flat = post.flatten()
            for i in range(nr_shuffles):
                if np.count_nonzero(pre) > 0 and np.count_nonzero(post) > 0:
                    np.random.shuffle(post_flat)
                    shuffle_list.append(pearsonr(pre.flatten(), post_flat)[0])
                else:
                    shuffle_list.append(0)
            remapping_shuffle.append(shuffle_list)
        remapping_shuffle = np.vstack(remapping_shuffle)

        remapping_stable = remapping[stable_cells]
        remapping_stable = remapping_stable[remapping_stable != np.nan]
        remapping_shuffle_stable = remapping_shuffle[stable_cells, :]

        remapping_dec = remapping[dec_cells]
        remapping_dec = remapping_dec[remapping_dec != np.nan]
        remapping_shuffle_dec = remapping_shuffle[dec_cells, :]

        # check how many cells did not remapped
        const = 0
        for data, control in zip(remapping_stable, remapping_shuffle_stable):
            # if data is 2 std above the mean of control --> no significant remapping
            if data > np.mean(control) + 2 * np.std(control):
                const += 1

        percent_stable_place = np.round(const / nr_stable_cells * 100, 2)

        if plot_results:

            stable_cell_remap_sorted = np.sort(remapping_stable)
            stable_cell_remap_shuffle_sorted = np.sort(remapping_shuffle_stable.flatten())

            dec_cell_remap_sorted = np.sort(remapping_dec)
            dec_cell_remap_shuffle_sorted = np.sort(remapping_shuffle_dec.flatten())

            # plot on cell level
            p_stable_cell_data = 1. * np.arange(stable_cell_remap_sorted.shape[0]) / (stable_cell_remap_sorted.shape[0] - 1)
            p_stable_cell_shuffle = 1. * np.arange(stable_cell_remap_shuffle_sorted.shape[0]) / (stable_cell_remap_shuffle_sorted.shape[0] - 1)

            p_dec_cell_data = 1. * np.arange(dec_cell_remap_sorted.shape[0]) / (dec_cell_remap_sorted.shape[0] - 1)
            p_dec_cell_shuffle = 1. * np.arange(dec_cell_remap_shuffle_sorted.shape[0]) / (dec_cell_remap_shuffle_sorted.shape[0] - 1)

            plt.plot(stable_cell_remap_sorted, p_stable_cell_data, label="Stable", color="magenta")
            plt.plot(stable_cell_remap_shuffle_sorted, p_stable_cell_shuffle, label="Stable shuffle", color="darkmagenta", linestyle="dashed")
            plt.plot(dec_cell_remap_sorted, p_dec_cell_data, label="Dec", color="aquamarine")
            plt.plot(dec_cell_remap_shuffle_sorted, p_dec_cell_shuffle, label="Dec shuffle", color="lightseagreen", linestyle="dashed")
            plt.legend()
            plt.ylabel("CDF")
            plt.xlabel("PEARSON R")
            plt.title("Per cell")
            plt.show()

            # plot on population vector level
            p_pv_stable = 1. * np.arange(remapping_pv_stable_sorted.shape[0]) / (remapping_pv_stable_sorted.shape[0] - 1)
            p_pv_stable_shuffle = 1. * np.arange(remapping_pv_stable_shuffle_sorted.shape[0]) / (remapping_pv_stable_shuffle_sorted.shape[0] - 1)

            p_pv_dec = 1. * np.arange(remapping_pv_dec_sorted.shape[0]) / (remapping_pv_dec_sorted.shape[0] - 1)
            p_pv_dec_shuffle = 1. * np.arange(remapping_pv_dec_shuffle_sorted.shape[0]) / (remapping_pv_dec_shuffle_sorted.shape[0] - 1)

            plt.plot(remapping_pv_stable_sorted, p_pv_stable, label="Stable", color="magenta")
            plt.plot(remapping_pv_stable_shuffle_sorted, p_pv_stable_shuffle, label="Stable shuffle",
                     color="darkmagenta", linestyle="dashed")

            plt.plot(remapping_pv_dec_sorted, p_pv_dec, label="Dec",  color="aquamarine")
            plt.plot(remapping_pv_dec_shuffle_sorted, p_pv_dec_shuffle, label="Dec shuffle",
                     color="lightseagreen", linestyle="dashed")
            plt.legend()
            plt.ylabel("CDF")
            plt.xlabel("PEARSON R")
            plt.title("Per pop. vec.")
            plt.show()

        else:

            return remapping_stable, remapping_shuffle_stable, remapping_dec, remapping_shuffle_dec, \
                   remapping_pv_stable, remapping_pv_stable_shuffle, remapping_pv_dec, remapping_pv_dec_shuffle

    def learning_pv_corr_temporal(self, spatial_resolution=2, average_trials=False):

        # check how many trials are there
        nr_trials = len(self.trial_loc_list)

        # how many trials to skip or average
        nr_trials_in_between = 5

        nr_data = floor(nr_trials/nr_trials_in_between)

        rate_maps = []

        occ_maps = []

        for i in range(nr_data):
            if average_trials:
                new_map = self.get_rate_maps(trials_to_use=range(i*nr_trials_in_between,(i+1)*nr_trials_in_between),
                                             spatial_resolution=spatial_resolution)
                occ_map = self.get_occ_map(trials_to_use=range(i*nr_trials_in_between,(i+1)*nr_trials_in_between),
                                             spatial_resolution=spatial_resolution)

            else:
                new_map = self.get_rate_maps(trials_to_use=range(i*nr_trials_in_between, i*nr_trials_in_between+1),
                                             spatial_resolution=spatial_resolution)
                occ_map = self.get_occ_map(trials_to_use=range(i*nr_trials_in_between, i*nr_trials_in_between+1),
                                             spatial_resolution=spatial_resolution)
            rate_maps.append(new_map)
            occ_maps.append(occ_map)

        # load cell labelaverage_trials=Falses
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()

        pv_corr_stable_mean = []
        pv_corr_stable_std = []
        pv_corr_dec_mean = []
        pv_corr_dec_std = []
        # go trough all maps
        for i_map in range(len(rate_maps)-1):
            # get population vectors
            pvs_first = np.reshape(rate_maps[i_map], (rate_maps[i_map].shape[0]*rate_maps[i_map].shape[1],
                                   rate_maps[i_map].shape[2]))
            pvs_second = np.reshape(rate_maps[i_map+1], (rate_maps[i_map].shape[0]*rate_maps[i_map].shape[1],
                                   rate_maps[i_map].shape[2]))

            occ_map_first = occ_maps[i_map]
            occ_map_second = occ_maps[i_map+1]
            comb_occ_map = np.logical_and(occ_map_first.flatten() > 0, occ_map_second.flatten() > 0)

            # only use spatial bins that were visited on both trials
            pvs_first = pvs_first[comb_occ_map,:]
            pvs_second = pvs_second[comb_occ_map, :]

            # first compute for stable cells
            pvs_first_stable = pvs_first[:, stable_cells]
            pvs_second_stable = pvs_second[:, stable_cells]
            pv_corr = []
            for pv_first, pv_second in zip(pvs_first_stable, pvs_second_stable):
                if np.count_nonzero(pv_first) > 0 and np.count_nonzero(pv_second) > 0:
                    pv_corr.append(pearsonr(pv_first, pv_second)[0])
                else:
                    continue
            pv_corr_stable_mean.append(np.mean(np.array(pv_corr)))
            pv_corr_stable_std.append(np.std(np.array(pv_corr)))

            # compute for dec cells
            pvs_first_dec = pvs_first[:, dec_cells]
            pvs_second_dec = pvs_second[:, dec_cells]
            pv_corr = []
            for pv_first, pv_second in zip(pvs_first_dec, pvs_second_dec):
                if np.count_nonzero(pv_first) > 0 and np.count_nonzero(pv_second) > 0:
                    pv_corr.append(pearsonr(pv_first, pv_second)[0])
                else:
                    continue
            pv_corr_dec_mean.append(np.mean(np.array(pv_corr)))
            pv_corr_dec_std.append(np.std(np.array(pv_corr)))


        pv_corr_dec_mean = np.array(pv_corr_dec_mean)
        pv_corr_stable_mean = np.array(pv_corr_stable_mean)
        pv_corr_dec_std = np.array(pv_corr_dec_std)
        pv_corr_stable_std = np.array(pv_corr_stable_std)

        plt.errorbar(x=np.arange(pv_corr_stable_mean.shape[0]), y=pv_corr_stable_mean, yerr=pv_corr_stable_std,
                     color="magenta", label="stable", capsize=2)
        plt.errorbar(x=np.arange(pv_corr_stable_mean.shape[0]), y=pv_corr_dec_mean, yerr=pv_corr_dec_std,
                     color="turquoise", label="decreasing", capsize=2, alpha=0.8)
        plt.ylabel("Mean PV correlations")
        plt.xlabel("Comparison ID (time)")
        plt.legend()
        plt.show()

    def learning_pv_corr_stable_dec(self, spatial_resolution=2, plotting=True):

        # get maps for first 5 trials
        rate_initial = self.get_rate_maps(trials_to_use=range(5), spatial_resolution=spatial_resolution)
        # get maps for last 5 trials
        rate_last = self.get_rate_maps(trials_to_use=range(len(self.trial_loc_list) - 5, len(self.trial_loc_list)),
                                  spatial_resolution=spatial_resolution)

        # get occ for first 5 trials
        occ_initial = self.get_occ_map(trials_to_use=range(5), spatial_resolution=spatial_resolution)
        # get occ for last 5 trials
        occ_last = self.get_occ_map(trials_to_use=range(len(self.trial_loc_list) - 5, len(self.trial_loc_list)),
                                  spatial_resolution=spatial_resolution)


        # load cell labelaverage_trials=Falses
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()

        pv_corr_stable_mean = []
        pv_corr_stable_std = []
        pv_corr_dec_mean = []
        pv_corr_dec_std = []

        # process both maps

        pvs_first = np.reshape(rate_initial, (rate_initial.shape[0]*rate_initial.shape[1],
                               rate_initial.shape[2]))
        pvs_second = np.reshape(rate_last, (rate_last.shape[0]*rate_last.shape[1],
                               rate_last.shape[2]))

        occ_map_first = occ_initial
        occ_map_second = occ_last
        comb_occ_map = np.logical_and(occ_map_first.flatten() > 0, occ_map_second.flatten() > 0)

        # only use spatial bins that were visited on both trials
        pvs_first = pvs_first[comb_occ_map,:]
        pvs_second = pvs_second[comb_occ_map, :]

        # first compute for stable cells
        pvs_first_stable = pvs_first[:, stable_cells]
        pvs_second_stable = pvs_second[:, stable_cells]
        pv_corr_stable = []
        for pv_first, pv_second in zip(pvs_first_stable, pvs_second_stable):
            if np.count_nonzero(pv_first) > 0 and np.count_nonzero(pv_second) > 0:
                pv_corr_stable.append(pearsonr(pv_first, pv_second)[0])
            else:
                continue
        pv_corr_stable_mean.append(np.mean(np.array(pv_corr_stable)))
        pv_corr_stable_std.append(np.std(np.array(pv_corr_stable)))

        # compute for dec cells
        pvs_first_dec = pvs_first[:, dec_cells]
        pvs_second_dec = pvs_second[:, dec_cells]
        pv_corr_dec = []
        for pv_first, pv_second in zip(pvs_first_dec, pvs_second_dec):
            if np.count_nonzero(pv_first) > 0 and np.count_nonzero(pv_second) > 0:
                pv_corr_dec.append(pearsonr(pv_first, pv_second)[0])
            else:
                continue
        pv_corr_dec_mean.append(np.mean(np.array(pv_corr_dec)))
        pv_corr_dec_std.append(np.std(np.array(pv_corr_dec)))

        pv_corr_dec_mean = np.array(pv_corr_dec_mean)
        pv_corr_stable_mean = np.array(pv_corr_stable_mean)
        pv_corr_dec_std = np.array(pv_corr_dec_std)
        pv_corr_stable_std = np.array(pv_corr_stable_std)
        if plotting:
            res = [pv_corr_stable, pv_corr_dec]
            c="white"
            bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                                labels=["stable", "dec"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            plt.ylabel("Pearson R PV (first 5 trials vs. last 5 trials)")
            plt.show()
            print(mannwhitneyu(np.hstack(pv_corr_stable), np.hstack(pv_corr_dec)))

        else:
            return pv_corr_stable, pv_corr_dec

    def learning_multinom_log_reg(self, time_bin_size=None, nr_chunks=4, nr_fits=1,
                                  sub_sample_size=0, return_coef=False, cells_to_use="stable"):
        """
        Multinomial logistic regression to assess non-stationarity during learning

        Parameters
        ----------
        sub_sample_size : size of subset to use --> if = 0: no subsampling is done
        nr_fits : how many times to do shuffling, refitting and re-evaluation of model
        nr_chunks : in how many chunks to sleep data for classification
        time_bin_size : time bin size in seconds for temporal binning
        """

        raster = self.get_raster(trials_to_use="all")

        # check if we need to up or down sample data
        if time_bin_size is not None and not (time_bin_size == self.params.time_bin_size):
            if time_bin_size > self.params.time_bin_size:

                x_orig = down_sample_array_sum(raster, int(time_bin_size/self.params.time_bin_size))
            else:
                raise Exception("Time bin size must be smaller thatn standard time bin size")

        else:
            x_orig = raster

        good_bins = np.sum(x_orig, axis=0) > 0
        x_orig = x_orig[:, good_bins]

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        if cells_to_use == "stable":
            x_orig = x_orig[class_dic["stable_cell_ids"], :]
        elif cells_to_use == "decreasing":
            x_orig = x_orig[class_dic["decrease_cell_ids"], :]
        elif cells_to_use == "increasing":
            x_orig = x_orig[class_dic["increase_cell_ids"], :]

        if sub_sample_size > 0:
            subset = np.random.randint(0, x_orig.shape[0], sub_sample_size)
            x_orig = x_orig[subset,:]

        # assign labels
        y_orig = np.zeros(x_orig.shape[1])
        size_chunk = math.ceil(y_orig.shape[0]/nr_chunks)
        for chunk_id in range(nr_chunks):
            y_orig[chunk_id*size_chunk:(chunk_id+1)*size_chunk] = int(chunk_id)

        scores = np.zeros(nr_fits)
        coef_list = []

        for i_fit in range(nr_fits):
            # split into test and train by sampling randomly

            # shuffle data
            per_ind = np.random.permutation(np.arange(x_orig.shape[1]))
            x = x_orig[:, per_ind]
            y = y_orig[per_ind]

            # 90% for training, 10% for validation
            x_train = x[:, :int(x.shape[1] * 0.9)]
            x_test = x[:, int(x.shape[1] * 0.9):]

            y_train = y[:int(y.shape[0] * 0.9)]
            y_test = y[int(y.shape[0] * 0.9):]

            clf = LogisticRegression(max_iter=10000, solver="newton-cg").fit(x_train.T, y_train)
            scores[i_fit] = clf.score(x_test.T, y_test)
            coef_list.append(clf.coef_)

        # fraction_corr = np.zeros(nr_chunks_sleep)
        # for chunk_id in range(nr_chunks_sleep):
        #     pred_chunk = y_pred[y_test==chunk_id]
        #     total = pred_chunk.shape[0]
        #     correct = pred_chunk[pred_chunk == chunk_id].shape[0]
        #     fraction_corr[chunk_id] = correct/total

        # nr_shuffles = 20
        # chance = np.zeros((nr_shuffles, nr_chunks_sleep))
        # for i_shuffle in range(nr_shuffles):
        #     clf = LogisticRegression(random_state=0).fit(x_train.T, np.random.permutation(y_train))
        #     y_pred = clf.predict(x_test.T)
        #     # get chance level
        #     for chunk_id in range(nr_chunks_sleep):
        #         pred_chunk = y_pred[y_test==chunk_id]
        #         total = pred_chunk.shape[0]
        #         correct = pred_chunk[pred_chunk == chunk_id].shape[0]
        #         chance[i_shuffle, chunk_id] = correct/total

        # chance=np.mean(np.mean(chance, axis=0))

        # plt.scatter(range(nr_chunks_sleep), fraction_corr)
        # plt.hlines(1/nr_chunks_sleep, 0, nr_chunks_sleep-1, label="chance", color="red")
        # plt.ylim(0,1)
        # plt.ylabel("Fraction correct")
        # plt.xlabel("Sleep chunk")
        # plt.legend()
        # plt.show()
        if return_coef:
            return scores, coef_list
        else:
            return scores

    def learning_svm_first_trials_last_trials(self, time_bin_size=None, nr_chunks=4, nr_fits=1,
                                  sub_sample_size=0, return_coef=False, cells_to_use="stable"):
        """
        Multinomial logistic regression to assess non-stationarity during learning

        Parameters
        ----------
        sub_sample_size : size of subset to use --> if = 0: no subsampling is done
        nr_fits : how many times to do shuffling, refitting and re-evaluation of model
        nr_chunks : in how many chunks to sleep data for classification
        time_bin_size : time bin size in seconds for temporal binning
        """

        # get maps for first trial
        initial_x, _, _ = self.get_raster_location_speed(trials_to_use=range(5))
        initial_y = np.zeros(initial_x.shape[1])
        # get maps for last trial
        last_x, _, _ = self.get_raster_location_speed(trials_to_use=range(len(self.trial_loc_list)-5,
                                                                        len(self.trial_loc_list)))
        last_y = np.ones(last_x.shape[1])

        x_orig = np.hstack((initial_x, last_x))
        y_orig = np.hstack((initial_y, last_y))

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        if cells_to_use == "stable":
            x_orig = x_orig[class_dic["stable_cell_ids"], :]
        elif cells_to_use == "decreasing":
            x_orig = x_orig[class_dic["decrease_cell_ids"], :]
        elif cells_to_use == "increasing":
            x_orig = x_orig[class_dic["increase_cell_ids"], :]

        if sub_sample_size > 0:
            subset = np.random.randint(0, x_orig.shape[0], sub_sample_size)
            x_orig = x_orig[subset,:]

        # assign labels
        scores = np.zeros(nr_fits)
        coef_list = []

        for i_fit in range(nr_fits):
            # split into test and train by sampling randomly

            # shuffle data
            per_ind = np.random.permutation(np.arange(x_orig.shape[1]))
            x = x_orig[:, per_ind]
            y = y_orig[per_ind]

            # 90% for training, 10% for validation
            x_train = x[:, :int(x.shape[1] * 0.9)]
            x_test = x[:, int(x.shape[1] * 0.9):]

            y_train = y[:int(y.shape[0] * 0.9)]
            y_test = y[int(y.shape[0] * 0.9):]

            clf = LogisticRegression(max_iter=10000, solver="newton-cg").fit(x_train.T, y_train)
            scores[i_fit] = clf.score(x_test.T, y_test)
            coef_list.append(clf.coef_)

        # fraction_corr = np.zeros(nr_chunks_sleep)
        # for chunk_id in range(nr_chunks_sleep):
        #     pred_chunk = y_pred[y_test==chunk_id]
        #     total = pred_chunk.shape[0]
        #     correct = pred_chunk[pred_chunk == chunk_id].shape[0]
        #     fraction_corr[chunk_id] = correct/total

        # nr_shuffles = 20
        # chance = np.zeros((nr_shuffles, nr_chunks_sleep))
        # for i_shuffle in range(nr_shuffles):
        #     clf = LogisticRegression(random_state=0).fit(x_train.T, np.random.permutation(y_train))
        #     y_pred = clf.predict(x_test.T)
        #     # get chance level
        #     for chunk_id in range(nr_chunks_sleep):
        #         pred_chunk = y_pred[y_test==chunk_id]
        #         total = pred_chunk.shape[0]
        #         correct = pred_chunk[pred_chunk == chunk_id].shape[0]
        #         chance[i_shuffle, chunk_id] = correct/total

        # chance=np.mean(np.mean(chance, axis=0))

        # plt.scatter(range(nr_chunks_sleep), fraction_corr)
        # plt.hlines(1/nr_chunks_sleep, 0, nr_chunks_sleep-1, label="chance", color="red")
        # plt.ylim(0,1)
        # plt.ylabel("Fraction correct")
        # plt.xlabel("Sleep chunk")
        # plt.legend()
        # plt.show()
        if return_coef:
            return scores, coef_list
        else:
            return scores

    def learning_svm_first_trials_last_trials_stable_dec_inc(self, time_bin_size=None):
        # need to match number of neurons to have a fair comparison

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        nr_stable = class_dic["stable_cell_ids"].shape[0]
        nr_dec = class_dic["decrease_cell_ids"].shape[0]
        nr_inc = class_dic["increase_cell_ids"].shape[0]
        nr_cells = [nr_stable, nr_dec, nr_inc]

        min_nr = np.min(nr_cells)

        result_dic = {
            "stable": None,
            "decreasing": None,
            "increasing": None
        }

        for cell_subset, nr_cells_in_subset in zip(["stable", "decreasing", "increasing"], nr_cells):
            if nr_cells_in_subset == min_nr:
                # only need to compute score once
                result_dic[cell_subset] = self.learning_svm_first_trials_last_trials(time_bin_size=time_bin_size,
                                                                         cells_to_use = cell_subset, nr_fits=10)
            else:
                # compute 10 times with different subsets and save mean
                res_per_fit = np.zeros(10)
                for i in range(10):
                    res_per_fit[i] = self.learning_svm_first_trials_last_trials(time_bin_size=time_bin_size,
                                                                    cells_to_use = cell_subset, sub_sample_size=min_nr)
                result_dic[cell_subset] = res_per_fit

        return result_dic

    def learning_multinom_log_reg_stable_dec_inc(self, time_bin_size=None):
        # need to match number of neurons to have a fair comparison

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        nr_stable = class_dic["stable_cell_ids"].shape[0]
        nr_dec = class_dic["decrease_cell_ids"].shape[0]
        nr_inc = class_dic["increase_cell_ids"].shape[0]
        nr_cells = [nr_stable, nr_dec, nr_inc]

        min_nr = np.min(nr_cells)

        result_dic = {
            "stable": None,
            "decreasing": None,
            "increasing": None
        }

        for cell_subset, nr_cells_in_subset in zip(["stable", "decreasing", "increasing"], nr_cells):
            if nr_cells_in_subset == min_nr:
                # only need to compute score once
                result_dic[cell_subset] = self.learning_multinom_log_reg(time_bin_size=time_bin_size,
                                                                         cells_to_use = cell_subset, nr_fits=10)
            else:
                # compute 10 times with different subsets and save mean
                res_per_fit = np.zeros(10)
                for i in range(10):
                    res_per_fit[i] = self.learning_multinom_log_reg(time_bin_size=time_bin_size,
                                                                    cells_to_use = cell_subset, sub_sample_size=min_nr)
                result_dic[cell_subset] = res_per_fit

        return result_dic

    def learning_predict_bin_progression(self, time_bin_size=2, cells_to_use="stable",
                                         norm_firing_rates=False, save_fig=False, plotting=False, show_weights=False):
        """
        analysis of drift using population vectors

        @param time_bin_size: which time bin size to use for prediction --> if None:
        standard time bin size from parameter object is used
        @type time_bin_size: float
        """

        raster = self.get_raster(trials_to_use="all")

        # check if we need to up or down sample data
        if time_bin_size is not None and not (time_bin_size == self.params.time_bin_size):
            if time_bin_size > self.params.time_bin_size:

                x_orig = down_sample_array_sum(raster, int(time_bin_size/self.params.time_bin_size))
            else:
                raise Exception("Time bin size must be smaller thatn standard time bin size")

        else:
            x_orig = raster

        good_bins = np.sum(x_orig, axis=0) > 0
        x_orig = x_orig[:, good_bins]

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        if cells_to_use == "stable":
            x_orig = x_orig[class_dic["stable_cell_ids"], :]
        elif cells_to_use == "decreasing":
            x_orig = x_orig[class_dic["decrease_cell_ids"], :]
        elif cells_to_use == "increasing":
            x_orig = x_orig[class_dic["increase_cell_ids"], :]

        if norm_firing_rates:
            x_orig = (x_orig - np.min(x_orig, axis=1, keepdims=True)) / \
                (np.max(x_orig, axis=1, keepdims=True) - np.min(x_orig, axis=1, keepdims=True))

        # plt.imshow(x_orig, interpolation='nearest', aspect='auto', cmap="jet")
        # plt.show()
        # a = preprocessing.MinMaxScaler()
        # x_orig = a.fit_transform(X=x_orig)
        # plt.imshow(x_orig, interpolation='nearest', aspect='auto', cmap="jet")
        # plt.show()
        # print("HERE")

        # # plot activation matrix (matrix of population vectors)
        # plt.imshow(x_orig, vmin=0, vmax=x_orig.max(), cmap='jet', aspect='auto')
        # plt.imshow(x_orig, interpolation='nearest', aspect='auto', cmap="jet",
        #            extent=[0, x_orig.shape[1], x_orig.shape[0] - 0.5, 0.5])
        # plt.ylabel("CELL ID")
        # plt.xlabel("TIME BINS")
        # plt.title("SPIKE BINNING RASTER")
        # a = plt.colorbar()
        # a.set_label("# SPIKES")
        # plt.show()

        y = np.arange(x_orig.shape[1])

        new_ml = MlMethodsOnePopulation()
        if show_weights:
            weights = new_ml.ridge_time_bin_progress(x=x_orig, y=y, new_time_bin_size=time_bin_size,
                                                 plotting=plotting, save_fig=save_fig, return_weights=True)
            plt.hist(weights, density=True)
            plt.show()
            r2 = None
        else:
            r2 = new_ml.ridge_time_bin_progress(x=x_orig, y=y, new_time_bin_size=time_bin_size,
                                                 plotting=plotting, save_fig=save_fig)

        return r2

    def learning_rate_map_corr_temporal(self, spatial_resolution=5, average_trials=False, instances_to_compare=None,
                                        plotting=True, save_fig=False, nr_trials_in_between=5):

        # check how many trials are there
        nr_trials = len(self.trial_loc_list)

        nr_data = floor(nr_trials/nr_trials_in_between)

        rate_maps = []
        block_labels = []

        for i in range(nr_data):
            if average_trials:
                new_map = self.get_rate_maps(trials_to_use=range(i*nr_trials_in_between,(i+1)*nr_trials_in_between),
                                             spatial_resolution=spatial_resolution)
                block_labels.append(str(i*nr_trials_in_between)+"-"+str((i+1)*nr_trials_in_between))
            else:
                new_map = self.get_rate_maps(trials_to_use=range(i*nr_trials_in_between, i*nr_trials_in_between+1),
                                             spatial_resolution=spatial_resolution)
            rate_maps.append(new_map)

        if instances_to_compare is None:
            # compare all instances (in time)
            time_steps = np.arange(len(rate_maps))

        else:
            if instances_to_compare > (len(rate_maps)):
                raise Exception("To many instances defined - choose smaller number")
            time_steps = np.linspace(0,len(rate_maps)-1, instances_to_compare, endpoint=True).astype(int)

        shift_between = []
        # go trough all instances
        for i_map in range(time_steps.shape[0]-1):
            map_first = rate_maps[time_steps[i_map]]
            map_second = rate_maps[time_steps[i_map+1]]
            remapping = []
            # go trough all rate maps per cell
            for map_init, map_last in zip(map_first.T, map_second.T):

                if np.count_nonzero(map_init) > 0 and np.count_nonzero(map_last) > 0:
                    # plt.subplot(1,2,1)
                    # plt.imshow(map_init)
                    # plt.subplot(1,2,2)
                    # plt.imshow(map_last)
                    # plt.title(pearsonr(map_init.flatten(), map_last.flatten())[0])
                    # plt.show()
                    remapping.append(pearsonr(map_init.flatten(), map_last.flatten())[0])
                else:
                    remapping.append(0)

            shift_between.append(remapping)

        shift_between = np.array(shift_between)

        # load cell labels
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()

        stable_mean = []
        stable_std = []
        stable_nr_obs = []
        dec_mean = []
        dec_std = []
        dec_nr_obs = []

        # compute max for stable and dec (to normalized between 0 and 1)
        max_stable = np.max(shift_between[:,stable_cells])
        max_dec = np.max(shift_between[:,dec_cells])

        for i, current_shift in enumerate(shift_between):
            stable_mean.append(np.mean(current_shift[stable_cells]))
            dec_mean.append(np.mean(current_shift[dec_cells]))
            stable_std.append(np.std(current_shift[stable_cells]))
            dec_std.append(np.std(current_shift[dec_cells]))
            stable_nr_obs.append(current_shift[stable_cells].shape[0])
            dec_nr_obs.append(current_shift[dec_cells].shape[0])

        stable_mean = np.array(stable_mean)
        stable_std = np.array(stable_std)

        stable_mean_norm = (stable_mean - np.min(stable_mean))/np.max(stable_mean - np.min(stable_mean))
        stable_std_norm = stable_std / np.max(stable_mean - np.min(stable_mean))

        dec_mean_norm = (dec_mean - np.min(dec_mean))/np.max(dec_mean - np.min(dec_mean))
        dec_std_norm = dec_std / np.max(dec_mean - np.min(dec_mean))

        stats = []

        for s_mean, s_std, d_mean, d_std, s_obs, d_obs in zip(stable_mean_norm, stable_std_norm, dec_mean_norm, dec_std_norm,
                                                stable_nr_obs, dec_nr_obs):
            stats.append(np.round(ttest_ind_from_stats(mean1=s_mean, std1=s_std, mean2=d_mean, std2=d_std,
                                                       nobs1=s_obs, nobs2=d_obs)[1],2))

        m = stable_mean.shape[0]

        if plotting or save_fig:
            plt.errorbar(x=np.arange(len(stable_mean)), y=stable_mean,yerr=stable_std, color="magenta", label="stable",
                         capsize=2)
            plt.errorbar(x=np.arange(len(dec_mean)), y=dec_mean,yerr=dec_std, color="turquoise", label="decreasing",
                         capsize=2, alpha=0.8)
            plt.ylabel("Rate map correlations")
            plt.xlabel("Comparison ID (time)")
            plt.legend()
            plt.show()
            print(str(stats)+", alpha="+str(np.round(0.05/m,3)))
            # build x labels
            x_labels = []
            for i in range(len(block_labels)-1):
                x_labels.append(block_labels[i]+"\nvs\n"+block_labels[i+1])
            if save_fig:
                plt.style.use('default')
            plt.errorbar(x=np.arange(len(stable_mean_norm)), y=stable_mean_norm,yerr=stable_std_norm, color="magenta", label="stable",
                         capsize=2)
            plt.errorbar(x=np.arange(len(dec_mean_norm)), y=dec_mean_norm,yerr=dec_std_norm, color="turquoise", label="decreasing",
                         capsize=2, alpha=0.8)
            plt.ylabel("Rate map correlations (normalized)")
            plt.xticks(range(len(block_labels)),x_labels)
            plt.xlabel("Trials compared")
            plt.ylim(-1.5,3.5)
            plt.xlim(-0.3, len(stable_mean_norm)-0.7)
            # write n.s. above all
            for i, (m, s) in enumerate(zip(stable_mean_norm, stable_std_norm)):
                plt.text(i-0.1, m+s+0.2,"n.s.")

            plt.legend()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "learning_rate_map_correlations.svg"), transparent="True")
            else:
                plt.show()

        else:
            return shift_between

    def learning_rate_map_corr_across_cells(self, plotting=False, filter_low_firing=True, spatial_resolution=5):
        """
        Computes correlations between cells for first 5 and last 5 trials of learning and compares resulting matrices

        @return:
        """
        # get maps for first trial
        initial = self.get_rate_maps(trials_to_use=range(5), spatial_resolution=spatial_resolution)
        # get maps for last trial
        last = self.get_rate_maps(trials_to_use=range(len(self.trial_loc_list)-5, len(self.trial_loc_list)),
                                  spatial_resolution=spatial_resolution)

        if filter_low_firing:
        # get firing to filter quiet cells
            # get maps for first trial
            initial_fir, _, _ = self.get_raster_location_speed(trials_to_use=range(5))
            # get maps for last trial
            last_fir, _, _ = self.get_raster_location_speed(trials_to_use=range(len(self.trial_loc_list)-5,
                                                                            len(self.trial_loc_list)))


        # load cell labelaverage_trials=Falses
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()

        initial_stable = initial[:, :, stable_cells]
        last_stable = last[:, :, stable_cells]

        if filter_low_firing:
            initial_fir_stable = initial_fir[stable_cells, :]
            last_fir_stable = last_fir[stable_cells, :]
            # need to filter cells that don't fire
            active_stable = np.argwhere(np.logical_and(np.sum(initial_fir_stable, axis=1).astype(bool),
                                            np.sum(last_fir_stable, axis=1).astype(bool)) > 0).flatten()

            initial_stable = initial_stable[:,:, active_stable]
            last_stable = last_stable[:,: ,active_stable]

        initial_dec = initial[:,:,dec_cells]
        last_dec = last[:,:,dec_cells]

        if filter_low_firing:
            initial_fir_dec = initial_fir[dec_cells, :]
            last_fir_dec = last_fir[dec_cells, :]
            # need to filter cells that don't fire
            active_dec = np.argwhere(np.logical_and(np.sum(initial_fir_dec, axis=1).astype(bool),
                                                       np.sum(last_fir_dec, axis=1).astype(bool)) > 0).flatten()

            initial_dec = initial_dec[:, :, active_dec]
            last_dec = last_dec[:, :, active_dec]

        # compute correlations of rate maps for stable cells for initial and later part
        corr_rate_maps_stable_initial = np.zeros((initial_stable.shape[2], initial_stable.shape[2]))
        corr_rate_maps_stable_last = np.zeros((initial_stable.shape[2], initial_stable.shape[2]))
        for template_id in range(initial_stable.shape[2]):
            for compare_id in range(initial_stable.shape[2]):
                corr_rate_maps_stable_initial[template_id, compare_id] = \
                    pearsonr(initial_stable[:,:,template_id].flatten(),
                             initial_stable[:,:,compare_id].flatten())[0]
                corr_rate_maps_stable_last[template_id, compare_id] = \
                    pearsonr(last_stable[:,:,template_id].flatten(),
                             last_stable[:,:,compare_id].flatten())[0]

        print("Stable cells")
        print(pearsonr(upper_tri_without_diag(corr_rate_maps_stable_initial), upper_tri_without_diag(corr_rate_maps_stable_last)))
        r_stable = pearsonr(upper_tri_without_diag(corr_rate_maps_stable_initial),
                            upper_tri_without_diag(corr_rate_maps_stable_last))[0]
        r_values_initial_stable = upper_tri_without_diag(corr_rate_maps_stable_initial)
        r_values_last_stable = upper_tri_without_diag(corr_rate_maps_stable_last)
        diff_stable = r_values_last_stable - r_values_initial_stable

        # compute correlations of rate maps for dec cells for initial and later part
        corr_rate_maps_dec_initial = np.zeros((initial_dec.shape[2], initial_dec.shape[2]))
        corr_rate_maps_dec_last = np.zeros((initial_dec.shape[2], initial_dec.shape[2]))
        for template_id in range(initial_dec.shape[2]):
            for compare_id in range(initial_dec.shape[2]):
                corr_rate_maps_dec_initial[template_id, compare_id] = \
                    pearsonr(initial_dec[:,:,template_id].flatten(),
                             initial_dec[:,:,compare_id].flatten())[0]
                corr_rate_maps_dec_last[template_id, compare_id] = \
                    pearsonr(last_dec[:,:,template_id].flatten(),
                             last_dec[:,:,compare_id].flatten())[0]
        print("Decreasing cells:")
        print(pearsonr(upper_tri_without_diag(corr_rate_maps_dec_initial), upper_tri_without_diag(corr_rate_maps_dec_last)))
        r_dec = pearsonr(upper_tri_without_diag(corr_rate_maps_dec_initial),
                         upper_tri_without_diag(corr_rate_maps_dec_last))[0]
        r_values_initial_dec = upper_tri_without_diag(corr_rate_maps_dec_initial)
        r_values_last_dec = upper_tri_without_diag(corr_rate_maps_dec_last)

        diff_dec = r_values_last_dec - r_values_initial_dec

        if plotting:
            plt.hist(diff_dec, color="blue", label="Decreasing", density=True)
            plt.hist(diff_stable, color="red", alpha=0.6, label="Stable", density=True)
            plt.legend()
            plt.ylabel("Density")
            plt.xlabel("Diff. in correlations")
            plt.show()

            plt.subplot(1, 2, 1)
            plt.imshow(corr_rate_maps_stable_initial)
            plt.title("First 5 trials (corr of rate maps)")
            plt.xlabel("Cell ID")
            plt.ylabel("Cell ID")
            plt.subplot(1, 2, 2)

            plt.imshow(corr_rate_maps_stable_last)
            plt.title("Last 5 trials (corr of rate maps)")
            plt.xlabel("Cell ID")
            plt.ylabel("Cell ID")
            plt.show()

            plt.subplot(1, 2, 1)
            plt.imshow(corr_rate_maps_dec_initial)
            plt.title("First 5 trials (corr of rate maps)")
            plt.xlabel("Cell ID")
            plt.ylabel("Cell ID")
            plt.subplot(1, 2, 2)

            plt.imshow(corr_rate_maps_dec_last)
            plt.title("Last 5 trials (corr of rate maps)")
            plt.xlabel("Cell ID")
            plt.ylabel("Cell ID")
            plt.show()

        else:
            return r_stable, r_dec, diff_stable, diff_dec, r_values_initial_stable, r_values_initial_dec, \
                r_values_last_stable, r_values_last_dec

    def learning_rate_map_corr_with_first_trial_temporal(self, spatial_resolution=2, average_trials=False,
                                        plotting=True, save_fig=False):

        # check how many trials are there
        nr_trials = len(self.trial_loc_list)

        rate_maps = []
        block_labels = []

        for i in range(nr_trials):
            new_map = self.get_rate_maps(trials_to_use=range(i,(i+1)),spatial_resolution=spatial_resolution)
            rate_maps.append(new_map)
        shift_between = []
        map_first = rate_maps[0]
        # compare all rate maps with first
        for map_to_compare in rate_maps[1:]:
            remapping = []
            # go trough all rate maps per cell
            for map_init, map_last in zip(map_first.T, map_to_compare.T):

                if np.count_nonzero(map_init) > 0 and np.count_nonzero(map_last) > 0:
                    # plt.subplot(1,2,1)
                    # plt.imshow(map_init)
                    # plt.subplot(1,2,2)
                    # plt.imshow(map_last)
                    # plt.title(pearsonr(map_init.flatten(), map_last.flatten())[0])
                    # plt.show()
                    remapping.append(pearsonr(map_init.flatten(), map_last.flatten())[0])
                else:
                    remapping.append(0)

            shift_between.append(remapping)

        shift_between = np.array(shift_between)

        # load cell labels
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()

        if self.session_name == "mjc163R4R_0114":
            # delete first dec cell --> has correlation of always 1
            dec_cells = dec_cells[1:]

        stable_mean = []
        stable_std = []
        stable_nr_obs = []
        dec_mean = []
        dec_std = []
        dec_nr_obs = []

        # compute max for stable and dec (to normalized between 0 and 1)
        max_stable = np.max(shift_between[:,stable_cells])
        max_dec = np.max(shift_between[:,dec_cells])

        for i, current_shift in enumerate(shift_between):
            stable_mean.append(np.mean(current_shift[stable_cells]))
            dec_mean.append(np.mean(current_shift[dec_cells]))
            stable_std.append(np.std(current_shift[stable_cells]))
            dec_std.append(np.std(current_shift[dec_cells]))
            stable_nr_obs.append(current_shift[stable_cells].shape[0])
            dec_nr_obs.append(current_shift[dec_cells].shape[0])

        stable_mean = np.array(stable_mean)
        stable_std = np.array(stable_std)

        stable_mean_norm = (stable_mean - np.min(stable_mean))/np.max(stable_mean - np.min(stable_mean))
        stable_std_norm = stable_std / np.max(stable_mean - np.min(stable_mean))

        dec_mean_norm = (dec_mean - np.min(dec_mean))/np.max(dec_mean - np.min(dec_mean))
        dec_std_norm = dec_std / np.max(dec_mean - np.min(dec_mean))

        stats = []

        for s_mean, s_std, d_mean, d_std, s_obs, d_obs in zip(stable_mean_norm, stable_std_norm, dec_mean_norm, dec_std_norm,
                                                stable_nr_obs, dec_nr_obs):
            stats.append(np.round(ttest_ind_from_stats(mean1=s_mean, std1=s_std, mean2=d_mean, std2=d_std,
                                                       nobs1=s_obs, nobs2=d_obs)[1],2))

        m = stable_mean.shape[0]

        if plotting or save_fig:
            plt.errorbar(x=np.arange(len(stable_mean)), y=stable_mean,yerr=stable_std, color="magenta", label="stable",
                         capsize=2)
            plt.errorbar(x=np.arange(len(dec_mean)), y=dec_mean,yerr=dec_std, color="turquoise", label="decreasing",
                         capsize=2, alpha=0.8)
            plt.ylabel("Rate map correlations")
            plt.xlabel("Comparison ID (time)")
            plt.legend()
            plt.show()
            print(str(stats)+", alpha="+str(np.round(0.05/m,3)))
            # build x labels
            x_labels = []
            for i in range(len(block_labels)-1):
                x_labels.append(block_labels[i]+"\nvs\n"+block_labels[i+1])
            if save_fig:
                plt.style.use('default')
            plt.errorbar(x=np.arange(len(stable_mean_norm)), y=stable_mean_norm,yerr=stable_std_norm, color="magenta", label="stable",
                         capsize=2)
            plt.errorbar(x=np.arange(len(dec_mean_norm)), y=dec_mean_norm,yerr=dec_std_norm, color="turquoise", label="decreasing",
                         capsize=2, alpha=0.8)
            plt.ylabel("Rate map correlations (normalized)")
            plt.xticks(range(len(block_labels)),x_labels)
            plt.xlabel("Trials compared")
            plt.ylim(-1.5,3.5)
            plt.xlim(-0.3, len(stable_mean_norm)-0.7)
            # write n.s. above all
            for i, (m, s) in enumerate(zip(stable_mean_norm, stable_std_norm)):
                plt.text(i-0.1, m+s+0.2,"n.s.")

            plt.legend()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "learning_rate_map_correlations.svg"), transparent="True")
            else:
                plt.show()

        else:
            return shift_between

    def learning_place_field_peak_shift_temporal(self, plotting=True, spatial_resolution=3, average_trials=False):

        # check how many trials are there
        nr_trials = len(self.trial_loc_list)

        # how many trials to skip or average
        nr_trials_in_between = 5

        nr_data = floor(nr_trials/nr_trials_in_between)

        rate_maps = []

        for i in range(nr_data):
            if average_trials:
                new_map = self.get_rate_maps(trials_to_use=range(i*nr_trials_in_between,(i+1)*nr_trials_in_between),
                                             spatial_resolution=spatial_resolution)
            else:
                new_map = self.get_rate_maps(trials_to_use=range(i*nr_trials_in_between, i*nr_trials_in_between+1),
                                             spatial_resolution=spatial_resolution)
            rate_maps.append(new_map)

        shift_between = []
        # go trough all cells and compute shift in peak
        for i_map in range(len(rate_maps)-1):
            map_first = rate_maps[i_map]
            map_second = rate_maps[i_map+1]
            shift = []
            for map_init, map_last in zip(map_first.T, map_second.T):
                # get peak during first trials
                peak_loc_init = np.unravel_index(map_init.argmax(), map_init.shape)
                peak_loc_y_init = peak_loc_init[0]
                peak_loc_x_init = peak_loc_init[1]
                # get peak during later trials
                peak_loc_last = np.unravel_index(map_last.argmax(), map_last.shape)
                peak_loc_y_last = peak_loc_last[0]
                peak_loc_x_last = peak_loc_last[1]

                distance = np.sqrt((peak_loc_x_init - peak_loc_x_last) ** 2 + (peak_loc_y_init - peak_loc_y_last) ** 2)
                distance_cm = distance / spatial_resolution
                shift.append(distance_cm)

            shift_between.append(shift)

        shift_between = np.array(shift_between)

        # load cell labels
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()

        stable_mean = []
        dec_mean = []
        stable_std = []
        dec_std = []
        for i, current_shift in enumerate(shift_between):
            stable_mean.append(np.mean(current_shift[stable_cells]))
            dec_mean.append(np.mean(current_shift[dec_cells]))
            stable_std.append(np.std(current_shift[stable_cells]))
            dec_std.append(np.std(current_shift[dec_cells]))

        plt.errorbar(x=np.arange(len(stable_mean)), y=stable_mean,yerr=stable_std, color="magenta", label="stable",
                     capsize=2)
        plt.errorbar(x=np.arange(len(dec_mean)), y=dec_mean,yerr=dec_std, color="turquoise", label="decreasing",
                     capsize=2, alpha=0.8)
        plt.ylabel("Place field shift")
        plt.xlabel("Comparison ID (time)")
        plt.legend()
        plt.show()

    def learning_place_field_peak_shift(self, plotting=True, spatial_resolution=3, all_cells=False):

        # get maps for first trial
        initial = self.get_rate_maps(trials_to_use=range(5), spatial_resolution=spatial_resolution)
        # get maps for last trial
        last = self.get_rate_maps(trials_to_use=range(len(self.trial_loc_list)-5, len(self.trial_loc_list)),
                                  spatial_resolution=spatial_resolution)

        shift = []
        # go trough all cells and compute shift in peak
        for map_init, map_last in zip(initial.T, last.T):
            # get peak during first trials
            peak_loc_init = np.unravel_index(map_init.argmax(), map_init.shape)
            peak_loc_y_init = peak_loc_init[0]
            peak_loc_x_init = peak_loc_init[1]
            # get peak during later trials
            peak_loc_last = np.unravel_index(map_last.argmax(), map_last.shape)
            peak_loc_y_last = peak_loc_last[0]
            peak_loc_x_last = peak_loc_last[1]

            distance = np.sqrt((peak_loc_x_init - peak_loc_x_last) ** 2 + (peak_loc_y_init - peak_loc_y_last) ** 2)
            distance_cm = distance / spatial_resolution
            shift.append(distance_cm)

        shift = np.array(shift)

        if all_cells:
            if plotting:
                # plot on population vector level
                shift_sorted = np.sort(shift)
                p = 1. * np.arange(shift.shape[0]) / (shift.shape[0] - 1)

                plt.plot(shift_sorted, p, label="All cells", color="aquamarine")
                plt.legend()
                plt.ylabel("CDF")
                plt.xlabel("Place field peak shift / cm")
                plt.title("Place field peak shift during learning")
                plt.show()
            else:
                return shift

        else:

            # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            stable_cells = class_dic["stable_cell_ids"].flatten()
            dec_cells = class_dic["decrease_cell_ids"].flatten()
            nr_stable_cells = stable_cells.shape[0]
            nr_dec_cells = dec_cells.shape[0]

            shift_stable = shift[stable_cells]
            shift_stable_sorted = np.sort(shift_stable)

            shift_dec = shift[dec_cells]
            shift_dec_sorted = np.sort(shift_dec)

            if plotting:

                # plot on population vector level
                p_stable = 1. * np.arange(shift_stable.shape[0]) / (shift_stable.shape[0] - 1)

                p_dec = 1. * np.arange(shift_dec.shape[0]) / (shift_dec.shape[0] - 1)

                plt.plot(shift_stable_sorted, p_stable, label="Stable", color="magenta")

                plt.plot(shift_dec_sorted, p_dec, label="Dec",  color="aquamarine")
                plt.legend()
                plt.ylabel("CDF")
                plt.xlabel("Place field peak shift / cm")
                plt.title("Place field peak shift during learning")
                plt.show()

            else:

                return shift_stable, shift_dec

    def learning_mean_firing_rate(self, plotting=False, absolute_value=False, filter_low_firing=False):
        """
        Computes the relative change in mean firing rate between first and last 5 trials of learning

        @return: diff, relative difference in firing from first 5 trials and last 5 trials
        """
        # get maps for first trial
        initial, _, _ = self.get_raster_location_speed(trials_to_use=range(5))
        # get maps for last trial
        last, _, _ = self.get_raster_location_speed(trials_to_use=range(len(self.trial_loc_list)-5,
                                                                        len(self.trial_loc_list)))

        all, _, _ = self.get_raster_location_speed(trials_to_use="all")
        mean_fir = np.mean(all, axis=1)


        initial_mean = np.max(initial, axis=1)/self.params.time_bin_size
        last_mean = np.max(last, axis=1)/self.params.time_bin_size

        diff = (last_mean - initial_mean)/ (last_mean + initial_mean)

        if absolute_value:
            diff = np.abs(diff)

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        cell_ids_stable = class_dic["stable_cell_ids"].flatten()

        cell_ids_dec = class_dic["decrease_cell_ids"].flatten()

        diff_stable = diff[cell_ids_stable]
        if filter_low_firing:
            initial_fir_stable = initial[cell_ids_stable, :]
            last_fir_stable = last[cell_ids_stable, :]
            mean_fir_stable =mean_fir[cell_ids_stable]
            # need to filter cells that don't fire
            active_stable = np.argwhere(np.logical_and(np.sum(initial_fir_stable, axis=1).astype(bool),
                                            np.sum(last_fir_stable, axis=1).astype(bool)) > 0).flatten()

            # active_stable = np.argwhere(mean_fir_stable > 0.01).flatten()

            diff_stable = diff_stable[active_stable]

        diff_dec = diff[cell_ids_dec]

        if filter_low_firing:
            initial_fir_dec = initial[cell_ids_dec, :]
            last_fir_dec = last[cell_ids_dec, :]
            mean_fir_dec = mean_fir[cell_ids_dec]
            # need to filter cells that don't fire
            active_dec = np.argwhere(np.logical_and(np.sum(initial_fir_dec, axis=1).astype(bool),
                                            np.sum(last_fir_dec, axis=1).astype(bool)) > 0).flatten()

            # active_stable = np.argwhere(mean_fir_dec > 0.01).flatten()

            diff_dec = diff_dec[active_dec]

        if plotting:

            diff_stable_sorted = np.sort(diff_stable)
            diff_dec_sorted = np.sort(diff_dec)

            p_diff_stable = 1. * np.arange(diff_stable.shape[0]) / (diff_stable.shape[0] - 1)

            p_diff_dec = 1. * np.arange(diff_dec.shape[0]) / (diff_dec.shape[0] - 1)

            plt.plot(diff_stable_sorted, p_diff_stable, label="stable")
            plt.plot(diff_dec_sorted, p_diff_dec, label="dec")
            plt.legend()
            plt.show()

        else:

            return diff_stable, diff_dec

    def learning_correlations(self, plotting=False):
        """
        Computes correlations between cells for first 5 and last 5 trials of learning and compares resulting matrices

        @return:
        """
        # get maps for first trial
        initial, _, _ = self.get_raster_location_speed(trials_to_use=range(5))
        # get maps for last trial
        last, _, _ = self.get_raster_location_speed(trials_to_use=range(len(self.trial_loc_list)-5,
                                                                        len(self.trial_loc_list)))

        # load cell labelaverage_trials=Falses
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()

        initial_stable = initial[stable_cells, :]
        last_stable = last[stable_cells, :]

        # need to filter cells that don't fire
        active_stable = np.argwhere(np.logical_and(np.sum(initial_stable, axis=1).astype(bool),
                                        np.sum(last_stable, axis=1).astype(bool)) > 0).flatten()

        initial_stable = initial_stable[active_stable, :]
        last_stable = last_stable[active_stable, :]

        initial_dec = initial[dec_cells, :]
        last_dec = last[dec_cells, :]

        # need to filter cells that don't fire
        active_dec = np.argwhere(np.logical_and(np.sum(initial_dec, axis=1).astype(bool),
                                        np.sum(last_dec, axis=1).astype(bool)) > 0).flatten()

        initial_dec = initial_dec[active_dec, :]
        last_dec = last_dec[active_dec, :]

        corr_init_stable = np.corrcoef(initial_stable)
        corr_last_stable = np.corrcoef(last_stable)

        diff_stable = corr_last_stable - corr_init_stable
        diff_stable = upper_tri_without_diag(diff_stable)

        corr_init_dec = np.corrcoef(initial_dec)
        corr_last_dec = np.corrcoef(last_dec)

        diff_dec = corr_last_dec - corr_init_dec
        diff_dec = upper_tri_without_diag(diff_dec)

        plt.hist(diff_stable, color="blue", density=True, label="stable")
        plt.hist(diff_dec, color="red", alpha=0.6, density=True, label="dec")
        plt.legend()
        plt.xlabel("Diff. correlations (end-beginning)")
        plt.ylabel("Density")
        plt.show()

        # plt.hist(np.abs(diff_stable), color="blue", density=True, label="stable")
        # plt.hist(np.abs(diff_dec), color="red", alpha=0.6, density=True, label="dec")
        # plt.legend()
        # plt.xlabel("Abs. diff. correlations (end-beginning)")
        # plt.ylabel("Density")
        # plt.show()

        corr_corr_stable = pearsonr(upper_tri_without_diag(corr_init_stable), upper_tri_without_diag(corr_last_stable))[0]
        corr_corr_dec = pearsonr(upper_tri_without_diag(corr_init_dec), upper_tri_without_diag(corr_last_dec))[0]

        if plotting:
            plt.subplot(1,2,1)
            plt.imshow(corr_init_stable, vmin=0, vmax=1)
            plt.title("First 5 trials learning \n Corr. stable cells")
            plt.ylabel("Cell Ids")
            plt.xlabel("Cell Ids")
            plt.subplot(1,2,2)
            plt.imshow(corr_last_stable, vmin=0, vmax=1)
            plt.title("Last 5 trials learning \n Corr. stable cells")
            plt.xlabel("Cell Ids")
            plt.show()

            plt.subplot(1,2,1)
            plt.imshow(corr_init_dec, vmin=0, vmax=1)
            plt.title("First 5 trials learning \n Corr. dec cells")
            plt.ylabel("Cell Ids")
            plt.xlabel("Cell Ids")
            plt.subplot(1,2,2)
            plt.imshow(corr_last_dec, vmin=0, vmax=1)
            plt.title("Last 5 trials learning \n Corr. dec cells")
            plt.xlabel("Cell Ids")
            plt.show()
        else:
            return corr_corr_stable, corr_corr_dec, diff_stable, diff_dec

    def learning_rate_map_corr(self, spatial_resolution=3):

        # get maps for first trial
        initial = self.get_rate_maps(trials_to_use=range(5), spatial_resolution=spatial_resolution)
        # get maps for last trial
        last = self.get_rate_maps(trials_to_use=range(len(self.trial_loc_list) - 5, len(self.trial_loc_list)),
                                  spatial_resolution=spatial_resolution)

        remapping = []

        for pre, post in zip(initial.T, last.T):
            if np.count_nonzero(pre) > 0 and np.count_nonzero(post) > 0:
                remapping.append(pearsonr(pre.flatten(), post.flatten())[0])
            else:
                remapping.append(0)

        remapping = np.array(remapping)

        return remapping

    def learning_rate_map_corr_stable_dec(self, spatial_resolution=3):

        # get maps for first trial
        initial = self.get_rate_maps(trials_to_use=range(5), spatial_resolution=spatial_resolution)
        # get maps for last trial
        last = self.get_rate_maps(trials_to_use=range(len(self.trial_loc_list) - 5, len(self.trial_loc_list)),
                                  spatial_resolution=spatial_resolution)

        remapping = []

        for pre, post in zip(initial.T, last.T):
            if np.count_nonzero(pre) > 0 and np.count_nonzero(post) > 0:
                remapping.append(pearsonr(pre.flatten(), post.flatten())[0])
            else:
                remapping.append(0)

        remapping = np.array(remapping)
        # load cell labels
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()

        remapping_stable = remapping[stable_cells]
        remapping_dec = remapping[dec_cells]

        return remapping_stable, remapping_dec

    def map_heterogenity(self, nr_shuffles=500, plot_results=True,
                              spatial_resolution=3, adjust_pv_size=False, plotting=True, metric="cosine"):

        # get maps for first trial
        initial = self.get_rate_maps(trials_to_use=range(5), spatial_resolution=spatial_resolution)
        initial_occ = self.get_occ_map(trials_to_use=range(5), spatial_resolution=spatial_resolution)
        # get maps for last trial
        last = self.get_rate_maps(trials_to_use=range(len(self.trial_loc_list)-5, len(self.trial_loc_list)),
                                  spatial_resolution=spatial_resolution)
        last_occ = self.get_occ_map(trials_to_use=range(len(self.trial_loc_list)-5, len(self.trial_loc_list)),
                                    spatial_resolution=spatial_resolution)

        # load cell labels
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()
        nr_stable_cells = stable_cells.shape[0]
        nr_dec_cells = dec_cells.shape[0]

        # compute remapping based on population vectors per bin
        # --------------------------------------------------------------------------------------------------------------

        pop_vec_initial = np.reshape(initial,
                                 (initial.shape[0] * initial.shape[1], initial.shape[2]))
        pop_vec_last = np.reshape(last, (last.shape[0] * last.shape[1], last.shape[2]))

        # only select spatial bins that were visited in PRE and POST
        comb_occ_map = np.logical_and(initial_occ.flatten() > 0, last_occ.flatten() > 0)
        pop_vec_initial = pop_vec_initial[comb_occ_map, :]
        pop_vec_last = pop_vec_last[comb_occ_map, :]

        comb_occ_map_spatial = np.reshape(comb_occ_map,(initial.shape[0], initial.shape[1]))
        common_loc = np.where(comb_occ_map_spatial)
        # stable cells

        pop_vec_pre_stable = pop_vec_initial[:, stable_cells]
        pop_vec_post_stable = pop_vec_last[:, stable_cells]

        pop_vec_pre_dec = pop_vec_initial[:, dec_cells]
        pop_vec_post_dec = pop_vec_last[:, dec_cells]

        initial_pop_vec_sim_stable = distance.pdist(pop_vec_pre_stable, metric=metric)
        initial_pop_vec_sim_dec = distance.pdist(pop_vec_pre_dec, metric=metric)
        late_pop_vec_sim_stable = distance.pdist(pop_vec_post_stable, metric=metric)
        late_pop_vec_sim_dec = distance.pdist(pop_vec_post_dec, metric=metric)

        if plotting:

            initial_pop_vec_sim_stable_sorted = np.sort(initial_pop_vec_sim_stable)
            initial_pop_vec_sim_dec_sorted = np.sort(initial_pop_vec_sim_dec)

            p_init_stable = 1. * np.arange(initial_pop_vec_sim_stable.shape[0]) / (initial_pop_vec_sim_stable.shape[0] - 1)
            p_init_dec = 1. * np.arange(initial_pop_vec_sim_dec.shape[0]) / (initial_pop_vec_sim_dec.shape[0] - 1)

            plt.plot(initial_pop_vec_sim_stable_sorted, p_init_stable, label="stable")
            plt.plot(initial_pop_vec_sim_dec_sorted, p_init_dec, label="dec")
            plt.legend()
            plt.title("Before learning")
            plt.show()

            late_pop_vec_sim_stable_sorted = np.sort(late_pop_vec_sim_stable)
            late_pop_vec_sim_dec_sorted = np.sort(late_pop_vec_sim_dec)

            p_late_stable = 1. * np.arange(late_pop_vec_sim_stable.shape[0]) / (late_pop_vec_sim_stable.shape[0] - 1)
            p_late_dec = 1. * np.arange(late_pop_vec_sim_dec.shape[0]) / (late_pop_vec_sim_dec.shape[0] - 1)

            plt.plot(late_pop_vec_sim_stable_sorted, p_late_stable, label="stable")
            plt.plot(late_pop_vec_sim_dec_sorted, p_late_dec, label="dec")
            plt.legend()
            plt.title("After learning")
            plt.show()


            plt.plot(initial_pop_vec_sim_stable_sorted, p_init_stable, label="before")
            plt.plot(late_pop_vec_sim_stable_sorted, p_late_stable, label="after")
            plt.legend()
            plt.title("Stable cells: change through learning")
            plt.show()

            plt.plot(initial_pop_vec_sim_dec_sorted, p_init_dec, label="before")
            plt.plot(late_pop_vec_sim_dec_sorted, p_late_dec, label="after")
            plt.legend()
            plt.title("Dec cells: change through learning")
            plt.show()
        else:
            return initial_pop_vec_sim_stable, initial_pop_vec_sim_dec, late_pop_vec_sim_stable, late_pop_vec_sim_dec

    def learning_phmm_modes_activation(self, trials_to_use_for_decoding=None, cells_to_use="stable"):
        if cells_to_use == "stable":
            # get phmm model to decode awake
            pre_phmm_model = self.session_params.default_pre_phmm_model_stable
        elif cells_to_use == "decreasing":
            # get phmm model to decode awake
            pre_phmm_model = self.session_params.default_pre_phmm_model_dec
        elif cells_to_use == "all":
            # get phmm model to decode awake
            pre_phmm_model = self.session_params.default_pre_phmm_model
        # decode awake activity
        seq, _, post_prob = self.decode_poisson_hmm(file_name=pre_phmm_model, cells_to_use=cells_to_use,
                                                      trials_to_use=trials_to_use_for_decoding)

        return post_prob, seq

    # </editor-fold>

    # <editor-fold desc="Goal coding">

    """#################################################################################################################
    #  goal coding
    #################################################################################################################"""

    def collective_goal_coding(self, cells_to_use="all", spatial_resolution=3, max_radius=10, ring_width=1):
        # load all rate maps
        rate_maps = self.get_rate_maps(spatial_resolution=spatial_resolution)

        # load cell labels
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        if cells_to_use == "stable":

            cell_ids = class_dic["stable_cell_ids"].flatten()

        elif cells_to_use == "increasing":

            cell_ids = class_dic["increase_cell_ids"].flatten()

        elif cells_to_use == "decreasing":

            cell_ids = class_dic["decrease_cell_ids"].flatten()

        elif cells_to_use == "all":

            cell_ids = np.arange(rate_maps.shape[2])

        # normalize rate maps
        max_per_cell = np.max(np.reshape(rate_maps, (rate_maps.shape[0]*rate_maps.shape[1], rate_maps.shape[2])), axis=0)
        max_per_cell[max_per_cell == 0] = 1e-22
        norm_rate_maps = rate_maps / max_per_cell

        # compute summed up rate map
        sum_rate_map = np.sum(norm_rate_maps[:, :, cell_ids], axis=2)

        # mask with occupancy
        occ = self.get_occ_map(spatial_resolution=spatial_resolution)
        sum_rate_map[occ==0] = np.nan

        sum_rate_map = sum_rate_map / np.sum(np.nan_to_num(sum_rate_map.flatten()))

        plt.imshow(sum_rate_map.T)
        a = plt.colorbar()
        a.set_label("Sum firing rate / normalized to 1")

        all_goals = collective_goal_coding(normalized_rate_map=sum_rate_map, goal_locations=self.goal_locations,
                               env_x_min=self.x_min, env_y_min=self.y_min, spatial_resolution=spatial_resolution,
                                           max_radius=max_radius, ring_width=ring_width)

        all_goals_arr = np.vstack(all_goals)
        for i, one_goal in enumerate(all_goals):
            plt.plot(np.arange(0, max_radius*spatial_resolution, ring_width*spatial_resolution), one_goal, label="Goal " + str(i))
        plt.plot(np.arange(0, max_radius*spatial_resolution, ring_width*spatial_resolution), np.mean(all_goals_arr, axis=0), c="w", label="Mean", linewidth=3)
        plt.xlabel("Distance to goal / cm")
        plt.ylabel("Mean density")
        plt.title(cells_to_use)
        plt.legend()
        plt.show()
        # exit()
        # plt.rcParams['svg.fonttype'] = 'none'
        # plt.savefig("dec_firing_changes.svg", transparent="True")

    def phmm_mode_occurrence(self, n_smoothing=1000):

        phmm_file = self.session_params.default_pre_phmm_model

        all_trials = range(len(self.trial_loc_list))

        # get location data from trials
        nr_cells = self.trial_raster_list[0].shape[0]
        loc = np.empty((0,2))
        raster = np.empty((nr_cells, 0))
        for trial_id in all_trials:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            loc = np.vstack((loc, self.trial_loc_list[trial_id]))

        state_sequence, nr_modes, post_prob = self.decode_poisson_hmm(file_name=phmm_file,
                                                           trials_to_use=all_trials)

        # get occurences
        modes, mode_occurrences = np.unique(state_sequence, return_counts=True)

        occurrences = np.zeros(post_prob.shape[1])
        occurrences[modes] = mode_occurrences

        smooth_post_prob = []
        m = []
        # compute probabilites in moving window
        for mode_post_prob in post_prob.T:
            mode_post_prob_smooth = moving_average(a=mode_post_prob, n=n_smoothing)
            mode_post_prob_smooth_norm = mode_post_prob_smooth/np.max(mode_post_prob_smooth)
            smooth_post_prob.append(mode_post_prob_smooth_norm)
            coef = np.polyfit(np.linspace(0,1,mode_post_prob_smooth_norm.shape[0]), mode_post_prob_smooth_norm, 1)
            m.append(coef[0])
            # plt.plot(mode_post_prob_smooth_norm)
            # poly1d_fn = np.poly1d(coef)
            # plt.plot(np.arange(mode_post_prob_smooth_norm.shape[0]),
            # poly1d_fn(np.linspace(0,1,mode_post_prob_smooth_norm.shape[0])), '--w')
            # plt.title(coef[0])
            # plt.show()

        m = np.array(m)

        return m, occurrences

    def place_field_goal_distance(self, trials_to_use=None, plotting=True, spatial_resolution_rate_maps=5,
                                  plot_for_control=False, min_nr_spikes=20):

        if trials_to_use is None:
            trials_to_use = self.default_trials

        # get maps for first trial
        rate_maps = self.get_rate_maps(trials_to_use=trials_to_use, spatial_resolution=spatial_resolution_rate_maps)
        raster = self.get_raster()

        # mean_rates = np.mean(raster, axis=1)/self.params.time_bin_size
        nr_of_spikes_per_cell = np.sum(raster, axis=1)
        # compute distances
        distances = distance_peak_firing_to_closest_goal(rate_maps, goal_locations=self.goal_locations,
                                                         env_x_min=self.x_min, env_y_min=self.y_min,
                                                         spatial_resolution_rate_maps=spatial_resolution_rate_maps,
                                                         plot_for_control=plot_for_control)

        # load cell labels
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()
        inc_cells = class_dic["increase_cell_ids"].flatten()

        bad_cells = np.argwhere(nr_of_spikes_per_cell < min_nr_spikes)


        stable_cells_filtered = np.delete(stable_cells, np.isin(stable_cells, bad_cells))
        dec_cells_filtered = np.delete(dec_cells, np.isin(dec_cells, bad_cells))
        inc_cells_filtered = np.delete(inc_cells, np.isin(inc_cells, bad_cells))
        dist_stable = distances[stable_cells_filtered]
        dist_dec = distances[dec_cells_filtered]
        dist_inc = distances[inc_cells_filtered]

        if plotting:
            c = "white"
            plt.figure(figsize=(4, 5))
            res = [dist_stable, dist_dec, dist_inc]
            bplot = plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                                labels=["Stable", "Decreasing", "Increasing"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            colors = ["magenta", 'turquoise']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            plt.ylabel("Distance to closest goal")
            plt.grid(color="grey", axis="y")
            plt.show()

        else:
            return dist_stable, dist_dec, dist_inc

    def place_field_goal_distance_all_trials(self, spatial_resolution_rate_maps=5, min_nr_spikes=20, trial_chunk=1,
                                             plot_for_control=False):

        stable_res = []
        dec_res = []
        inc_res = []

        nr_trials = self.get_nr_of_trials()
        if trial_chunk == 1:
            trials = range(nr_trials)
        else:
            trials = []
            trial_id=0
            while (trial_id+trial_chunk) < nr_trials:
                trials.append(range(trial_id, trial_id+trial_chunk))
                trial_id += trial_chunk
        for trial in trials:
            dist_stable, dist_dec, dist_inc = self.place_field_goal_distance(trials_to_use=trial,
                                                                             spatial_resolution_rate_maps=
                                                                             spatial_resolution_rate_maps,
                                                                             min_nr_spikes=min_nr_spikes,
                                                                             plotting=False,
                                                                             plot_for_control=plot_for_control)
            stable_res.append(dist_stable)
            dec_res.append(dist_dec)
            inc_res.append(dist_inc)

        c = "white"
        plt.figure(figsize=(10, 5))
        bplot = plt.boxplot(stable_res, positions=np.arange(len(stable_res)), patch_artist=True,
                            labels=np.arange(len(stable_res)),
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.ylabel("Distance to closest goal")
        plt.grid(color="grey", axis="y")
        plt.xticks(rotation=45)
        plt.title("Stable cells")
        if trial_chunk > 1:
            plt.xlabel("Trial CHUNK ID")
        else:
            plt.xlabel("Trial ID")
        plt.tight_layout()
        plt.show()

        c = "white"
        plt.figure(figsize=(10, 5))
        bplot = plt.boxplot(dec_res, positions=np.arange(len(dec_res)), patch_artist=True,
                            labels=np.arange(len(dec_res)),
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.ylabel("Distance to closest goal")
        plt.grid(color="grey", axis="y")
        plt.xticks(rotation=45)
        plt.title("Decreasing cells")
        if trial_chunk > 1:
            plt.xlabel("Trial CHUNK ID")
        else:
            plt.xlabel("Trial ID")
        plt.tight_layout()
        plt.show()

        c = "white"
        plt.figure(figsize=(10, 5))
        bplot = plt.boxplot(inc_res, positions=np.arange(len(inc_res)), patch_artist=True,
                            labels=np.arange(len(inc_res)),
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        plt.ylabel("Distance to closest goal")
        plt.grid(color="grey", axis="y")
        plt.xticks(rotation=45)
        plt.title("Increasing cells")
        if trial_chunk > 1:
            plt.xlabel("Trial CHUNK ID")
        else:
            plt.xlabel("Trial ID")
        plt.tight_layout()
        plt.show()

    def place_field_goal_distance_first_x_vs_rest(self, spatial_resolution_rate_maps=5,
                                                  min_nr_spikes=20, first_x=1, plotting=False):

        nr_trials = self.get_nr_of_trials()

        if first_x > 0:
            dist_stable_first, dist_dec_first, dist_inc_first = self.place_field_goal_distance(trials_to_use=np.arange(first_x),
                                                                             spatial_resolution_rate_maps=
                                                                             spatial_resolution_rate_maps,
                                                                             min_nr_spikes=min_nr_spikes, plotting=False)
        else:
            dist_stable_first, dist_dec_first, dist_inc_first = self.place_field_goal_distance(trials_to_use=first_x,
                                                                             spatial_resolution_rate_maps=
                                                                             spatial_resolution_rate_maps,
                                                                             min_nr_spikes=min_nr_spikes, plotting=False)

        dist_stable, dist_dec, dist_inc = self.place_field_goal_distance(trials_to_use=np.arange(first_x+1, nr_trials),
                                                                         spatial_resolution_rate_maps=
                                                                         spatial_resolution_rate_maps,
                                                                         min_nr_spikes=min_nr_spikes, plotting=False)

        if plotting:
            c = "white"
            plt.figure(figsize=(10, 5))
            res = [dist_stable_first, dist_stable, dist_dec_first, dist_dec, dist_inc_first, dist_inc]
            bplot = plt.boxplot(res, positions=np.arange(len(res)), patch_artist=True,
                                labels=["stable first "+str(first_x), "stable rest", "dec first "+str(first_x),
                                        "dec rest", "inc first "+str(first_x), "inc rest"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            colors = ["magenta", "magenta", 'turquoise', 'turquoise', "orange", "orange"]
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            plt.ylabel("Distance to closest goal")
            plt.grid(color="grey", axis="y")
            plt.xticks(rotation=45)
            plt.xlabel("Trial ID")
            plt.tight_layout()
            plt.show()

        return dist_stable_first, dist_stable, dist_dec_first, dist_dec, dist_inc_first, dist_inc
    # </editor-fold>

    # <editor-fold desc="GLM">

    """#################################################################################################################
    #  glm
    #################################################################################################################"""

    def infer_glm_awake(self, trials_to_use=None, nr_gauss=20, std_gauss=10, spatial_bin_size_map=None,
                        plot_for_control=False):
        """
        infers glm (paper: Correlations and Functional Connections in a Population of Grid Cells, 2015) from awake
        data

        @param trials_to_use: which trials to use to infer glm
        @type trials_to_use: range
        @param nr_gauss: how many Gaussians to distribute in environment
        @type nr_gauss: int
        @param std_gauss: which standard deviation to use for all Gaussians
        @type std_gauss: int
        @param spatial_bin_size_map: size of a spatial bin in cm
        @type spatial_bin_size_map: int
        @param plot_for_control: if True --> plot single steps of generating maps
        @type plot_for_control: bool
        """

        # check if time bin size < 20 ms --> needed for binary assumption
        if self.params.time_bin_size != 0.01:
            raise Exception("TIME BIN SIZE MUST BE 10 MS!")

        print(" - INFERENCE GLM USING CHEESEBOARD DATA ...\n")

        if trials_to_use is None:
            trials_to_use = self.default_trials

        # params
        # --------------------------------------------------------------------------------------------------------------
        learning_rates = [100, 10, 1, 0.1, 0.01]
        likelihood = np.zeros(len(learning_rates))
        max_likelihood_per_iteration = []
        max_iter = 250
        cell_to_plot = 5

        if spatial_bin_size_map is None:
            spatial_bin_size_map = self.params.spatial_resolution

        file_name = self.session_name+"_"+self.experiment_phase_id+"_"+\
                    str(spatial_bin_size_map)+"cm_bins"+"_"+self.cell_type+ "_trials_"+str(trials_to_use[0])+\
                      "_"+str(trials_to_use[-1])

        # place Gaussians uniformly across environment
        # --------------------------------------------------------------------------------------------------------------

        # get dimensions of environment
        x_min, x_max, y_min, y_max = self.x_min, self.x_max, self.y_min, self.y_max

        # get size of environment
        x_span = x_max - x_min
        y_span = y_max - y_min

        # width - length ratio of environment
        w_l_ratio = y_span/x_span

        # tile x and y with centers of Gaussians
        centers_gauss_x = np.linspace(x_min, x_max, nr_gauss)
        centers_gauss_y = np.linspace(y_min, y_max, int(np.round(nr_gauss*w_l_ratio)))

        # compute grid with x and y values of centers
        centers_gauss_x, centers_gauss_y = np.meshgrid(centers_gauss_x, centers_gauss_y)

        # get data used to infer model --> concatenate single trial data
        # --------------------------------------------------------------------------------------------------------------

        # check how many cells
        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))
        loc = np.empty((0,2))

        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            loc = np.vstack((loc, self.trial_loc_list[trial_id]))

        # data and parameter preparation for optimization
        # --------------------------------------------------------------------------------------------------------------

        # split location data into x and y coordinates
        x_loc = loc[:, 0]
        y_loc = loc[:, 1]

        # make binary raster --> if cell fires --> 1, if cell doesn't fire --> -1 --> ISING MODEL!
        bin_raster = -1 * np.ones((raster.shape[0], raster.shape[1]))
        bin_raster[raster > 0] = 1
        bin_raster = bin_raster.T

        # x_loc_m = loadmat("matlab.mat")["posx"]
        # y_loc_m = loadmat("matlab.mat")["posy"]

        # how many time bins / cells
        nr_time_bins = bin_raster.shape[0]-1
        nr_cells = bin_raster.shape[1]

        # compute distance from center of each Gaussian for every time bin
        dist_to_center_x = matlib.repmat(np.expand_dims(centers_gauss_x.flatten("F"), 1), 1, x_loc.shape[0] - 1) - \
                   matlib.repmat(x_loc[:-1].T, centers_gauss_x.flatten().shape[0], 1)
        dist_to_center_y = matlib.repmat(np.expand_dims(centers_gauss_y.flatten("F"), 1), 1, y_loc.shape[0] - 1) - \
                   matlib.repmat(y_loc[:-1].T, centers_gauss_y.flatten().shape[0], 1)

        # compute values of each Gaussian for each time bin
        gauss_values = simple_gaussian(xd=dist_to_center_x, yd=dist_to_center_y, std=std_gauss)

        # optimization of alpha values --> maximize data likelihood
        # --------------------------------------------------------------------------------------------------------------

        # alpha --> weights for Gaussians [#gaussians, #cells]
        alpha = np.zeros((gauss_values.shape[0], bin_raster.shape[1]))

        # bias --> firing rates of neurons (constant across time points!!!)
        bias = matlib.repmat(np.random.rand(nr_cells, 1), 1, nr_time_bins)

        for iter in range(max_iter):
            # compute gradient for alpha values --> dLikeli/dalpha
            dalpha=((gauss_values @ bin_raster[1:, :]) -
                    (np.tanh(alpha.T @ gauss_values + bias) @ gauss_values.T).T)/nr_time_bins

            # compute change in cost
            dcost = np.sum(bin_raster[1:, :].T - np.tanh(alpha.T @ gauss_values+bias), axis=1)/nr_time_bins

            # try different learning rates to maximize likelihood
            for i, l_r in enumerate(learning_rates):
                # compute new alpha values with gradient and learning rate
                alpha_n = alpha + l_r * dalpha
                # compute cost using old cost and update
                bias_n = bias + matlib.repmat(l_r*np.expand_dims(dcost, 1), 1, nr_time_bins)

                likelihood[i] = np.trace((alpha_n.T @ gauss_values + bias_n) @ bin_raster[1:, :])-np.sum(
                    np.sum(np.log(2*np.cosh(alpha_n.T @ gauss_values + bias_n)), axis=1))

            max_likelihood = np.max(likelihood)
            max_likelihood_per_iteration.append(max_likelihood)
            best_learning_rate_index = np.argmax(likelihood)

            # update bias --> optimize the bias term first before optimizing alpha values
            bias = bias + matlib.repmat(learning_rates[best_learning_rate_index]*
                                        np.expand_dims(dcost, 1), 1, nr_time_bins)

            # only start optimizing alpha values after n iterations
            if iter > 50:

                alpha = alpha + learning_rates[best_learning_rate_index] * dalpha

        # generation of maps for spatial bin size defined
        # --------------------------------------------------------------------------------------------------------------

        # if spatial_bin_size_map was not provided --> use spatial_resolution from parameter file
        if spatial_bin_size_map is None:
            spatial_bin_size_map = self.params.spatial_resolution

        nr_spatial_bins = int(np.round(x_span / spatial_bin_size_map))

        # generate grid by spatial binning
        x_map = np.linspace(x_min, x_max, nr_spatial_bins)
        y_map = np.linspace(y_min, y_max, int(np.round(nr_spatial_bins * w_l_ratio)))

        # compute grid from x and y values
        x_map, y_map = np.meshgrid(x_map, y_map)

        dist_to_center_x_map = matlib.repmat(np.expand_dims(centers_gauss_x.flatten("F"), 1), 1, x_map.flatten().shape[0])\
                               -matlib.repmat(x_map.flatten("F"), centers_gauss_x.flatten().shape[0], 1)
        dist_to_center_y_map = matlib.repmat(np.expand_dims(centers_gauss_y.flatten("F"), 1), 1, y_map.flatten().shape[0])\
                               -matlib.repmat(y_map.flatten("F"), centers_gauss_y.flatten().shape[0], 1)

        # compute values of each Gaussian for each time bin
        gauss_values_map = simple_gaussian(xd=dist_to_center_x_map, yd=dist_to_center_y_map, std=std_gauss)

        # compute resulting map
        res_map = np.exp(alpha.T @ gauss_values_map + matlib.repmat(bias[:, 0], gauss_values_map.shape[1], 1).T)/ \
                   (2*np.cosh(alpha.T @ gauss_values_map + matlib.repmat(bias[:, 0], gauss_values_map.shape[1], 1).T))

        # reshape to reconstruct 2D map
        res_map = matlib.reshape(res_map, (res_map.shape[0], nr_spatial_bins, int(np.round(nr_spatial_bins * w_l_ratio))))

        # compute occupancy --> mask results (remove non visited bins)
        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        # Fede's implementation:
        # ----------------------
        # centers_x = np.linspace(x_min, x_max + 0.1, nr_spatial_bins)
        # centers_y = np.linspace(y_min, y_max + 0.1, int(round(nr_spatial_bins*w_l_ratio)))
        #
        # dx = centers_x[1] - centers_x[0]
        # dy = centers_y[1] - centers_y[0]
        #
        # occ = np.zeros((centers_x.shape[0], centers_y.shape[0]))
        #
        # x_loc[x_loc > x_max] = x_min - 0.01
        # y_loc[y_loc > y_max] = y_min - 0.01
        #
        # for i in range(x_loc.shape[0]):
        #     xi = int(np.floor((x_loc[i]-x_min)/dx))+1
        #     yi = int(np.floor((y_loc[i] - y_min) / dy)) + 1
        #     if xi*yi > 0:
        #         occ[xi, yi] += 1
        #
        # occ_mask_fede = np.where(occ > 0, 1, 0)
        # occ_mask = np.repeat(occ_mask[np.newaxis, :, :], nr_cells, axis=0)
        # --------------------------------------------------------------------------------------------------------------

        # compute actual rate maps from used data --> to validate results
        rate_map, occ = self.rate_map_from_data(loc=loc, raster=raster, spatial_resolution=spatial_bin_size_map)

        # compute binary occupancy map --> used as a mask to filter non-visited bins
        occ_mask = np.where(occ > 0, 1, 0)
        occ_mask_plot = np.where(occ > 0, 1, np.nan)
        occ_mask = np.repeat(occ_mask[np.newaxis, :, :], nr_cells, axis=0)

        # save original map before applying the mask for plotting
        res_orig = res_map

        # apply mask to results
        res_map = np.multiply(res_map, occ_mask)

        if plot_for_control:

            plt.plot(max_likelihood_per_iteration)
            plt.title("LIKELIHOOD PER ITERATION")
            plt.xlabel("ITERATION")
            plt.ylabel("LIKELIHOOD")
            plt.show()

            # compute actual rate maps from used data --> to validate results
            rate_map_to_plot = np.multiply(rate_map[:, :, cell_to_plot], occ_mask_plot)
            plt.imshow(rate_map_to_plot.T, origin="lower")
            plt.scatter((centers_gauss_x-x_min)/spatial_bin_size_map, (centers_gauss_y-y_min)/spatial_bin_size_map
                        , s=0.1, label="GAUSS. CENTERS")
            plt.title("RATE MAP + GAUSSIAN CENTERS")
            a = plt.colorbar()
            a.set_label("FIRING RATE / 1/s")
            plt.legend()
            plt.show()

            a = alpha.T @ gauss_values_map
            a = matlib.reshape(a, (a.shape[0], nr_spatial_bins, int(np.round(nr_spatial_bins * w_l_ratio))))
            plt.imshow(a[cell_to_plot, :, :].T, interpolation='nearest', aspect='auto', origin="lower")
            plt.colorbar()
            plt.title("ALPHA.T @ GAUSSIANS")
            plt.show()

            plt.imshow(res_orig[cell_to_plot, :, :].T, origin="lower")
            plt.title("RES_MAP ORIGINAL (W/O OCC. MASK)")
            a = plt.colorbar()
            a.set_label("PROB. OF FIRING IN WINDOW (" +str(self.params.time_bin_size)+"s")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

            plt.imshow(occ.T, origin="lower")
            plt.title("OCC MAP")
            a = plt.colorbar()
            a.set_label("SEC")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

            plt.imshow(occ_mask[cell_to_plot, :, :].T, origin="lower")
            plt.title("OCC MAP BINARY")
            a = plt.colorbar()
            a.set_label("OCC: YES/NO")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

            plt.imshow(res_map[cell_to_plot, :, :].T, origin="lower")
            plt.title("RES_MAP MASKED")
            a = plt.colorbar()
            a.set_label("PROB. OF FIRING IN WINDOW (" +str(self.params.time_bin_size)+"s)")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

        model_dic = {
            "rate_map": rate_map,
            "occ_map": occ,
            "occ_mask_plot": occ_mask_plot,
            "res_map": res_map,
            "alpha": alpha,
            "bias": bias,
            "centers_gauss_x": centers_gauss_x,
            "centers_gauss_y": centers_gauss_y,
            "std_gauss": std_gauss,
            "likelihood": max_likelihood_per_iteration,
            "time_bin_size": self.params.time_bin_size
        }

        with open(self.params.pre_proc_dir+'awake_ising_maps/' + file_name + '.pkl', 'wb') as f:
            pickle.dump(model_dic, f, pickle.HIGHEST_PROTOCOL)

    def load_glm_awake(self, model_name=None, cell_id=None, plotting=False):

        if model_name is None:
            model_name = self.default_ising
        with open(self.params.pre_proc_dir + 'awake_ising_maps/' + model_name + '.pkl', 'rb') as f:
            model_dic = pickle.load(f)

        centers_gauss_x = model_dic["centers_gauss_x"]
        centers_gauss_y = model_dic["centers_gauss_y"]
        std_gauss = model_dic["std_gauss"]
        alpha = model_dic["alpha"]
        rate_map = model_dic["rate_map"]
        res_map = model_dic["res_map"]
        occ_mask_plot = model_dic["occ_mask_plot"]
        time_bin_size = model_dic["time_bin_size"]

        if plotting:
            # compute actual rate maps from used data --> to validate results
            rate_map_to_plot = np.multiply(rate_map[:, :, cell_id], occ_mask_plot)

            plt.imshow(rate_map_to_plot.T, origin="lower")

            plt.title("RATE MAP")
            a = plt.colorbar()
            a.set_label("FIRING RATE / Hz")
            plt.show()


            # print res map
            plt.imshow(res_map[cell_id, :, :].T, origin='lower', interpolation='nearest', aspect='auto')
            plt.title("RES MAP")
            plt.xlabel("X")
            plt.ylabel("Y")
            a = plt.colorbar()
            a.set_label("PROB. OF FIRING IN WINDOW (" + str(time_bin_size) + "s)")
            plt.show()
        else:
            return res_map

    # </editor-fold>

    # <editor-fold desc="pHMM">

    """#################################################################################################################
    #  poisson hmm
    #################################################################################################################"""

    def cross_val_poisson_hmm(self, trials_to_use=None, cl_ar=np.arange(1, 50, 5), cells_to_use="all_cells"):
        # --------------------------------------------------------------------------------------------------------------
        # cross validation of poisson hmm fits to data
        #
        # args:     - cl_ar, range object: #clusters to fit to data
        # --------------------------------------------------------------------------------------------------------------

        print(" - CROSS-VALIDATING POISSON HMM ON CHEESEBOARD --> OPTIMAL #MODES ...")
        print("  - nr modes to compute: "+str(cl_ar))

        if trials_to_use is None:
            trials_to_use = self.default_trials

        print("   --> USING TRIALS: "+str(trials_to_use[0])+"-"+str(trials_to_use[-1])+"\n")

        # check how many cells
        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))

        trial_lengths = []
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            trial_lengths.append(self.trial_raster_list[trial_id].shape[1])

        # if subset of cells to use
        if cells_to_use == "stable_cells":
            # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            stable_cells = class_dic["stable_cell_ids"].flatten()
            raster = raster[stable_cells, :]

        elif cells_to_use == "decreasing_cells":
                        # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            dec_cells = class_dic["decrease_cell_ids"].flatten()
            raster = raster[dec_cells, :]

        elif cells_to_use == "increasing_cells":
                        # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            inc_cells = class_dic["increase_cell_ids"].flatten()
            raster = raster[inc_cells, :]

        if self.params.cross_val_splits == "trial_splitting":
            trial_end = np.cumsum(np.array(trial_lengths))
            trial_start = np.concatenate([[0], trial_end[:-1]])
            # test_range = np.vstack((trial_start, trial_end))
            test_range_per_fold = []
            for lo, hi in zip(trial_start, trial_end):
                test_range_per_fold.append(np.array(list(range(lo, hi))))

        elif self.params.cross_val_splits == "custom_splits":

            # number of folds
            nr_folds = 10
            # how many times to fit for each split
            max_nr_fitting = 5
            # how many chunks (for pre-computed splits)
            nr_chunks = 10

            unobserved_lo_array = pickle.load(
                open("temp_data/unobserved_lo_cv" + str(nr_folds) + "_" + str(nr_chunks) + "_chunks", "rb"))
            unobserved_hi_array = pickle.load(
                open("temp_data/unobserved_hi_cv" + str(nr_folds) + "_" + str(nr_chunks) + "_chunks", "rb"))

            # set number of time bins
            bin_num = raster.shape[1]
            bins = np.arange(bin_num + 1)

            # length of one chunk
            n_chunks = int(bin_num / nr_chunks)
            test_range_per_fold = []
            for fold in range(nr_folds):

                # unobserved_lo: start bins (in spike data resolution) for all test data chunks
                unobserved_lo = []
                unobserved_hi = []
                for lo, hi in zip(unobserved_lo_array[fold], unobserved_hi_array[fold]):
                    unobserved_lo.append(bins[lo * n_chunks])
                    unobserved_hi.append(bins[hi * n_chunks])

                unobserved_lo = np.array(unobserved_lo)
                unobserved_hi = np.array(unobserved_hi)

                test_range = []
                for lo, hi in zip(unobserved_lo, unobserved_hi):
                    test_range += (list(range(lo, hi)))
                test_range_per_fold.append(np.array(test_range))

        nr_cores = 12

        folder_name = self.session_name +"_"+self.experiment_phase_id+"_"+self.cell_type+\
                      "_trials_"+str(trials_to_use[0])+"_"+str(trials_to_use[-1])

        new_ml = MlMethodsOnePopulation(params=self.params)
        new_ml.parallelize_cross_val_model(nr_cluster_array=cl_ar, nr_cores=nr_cores, model_type="pHMM",
                                           raster_data=raster, folder_name=folder_name, splits=test_range_per_fold,
                                           cells_used=cells_to_use)
        # new_ml.cross_val_view_results(folder_name=folder_name)

    def plot_custom_splits(self):
        new_ml = MlMethodsOnePopulation(params=self.params)
        new_ml.plot_custom_splits()

    def find_and_fit_optimal_number_of_modes(self, cells_to_use="all_cells", cl_ar_init = np.arange(1, 50, 5)):
        # compute likelihoods with standard spacing first
        self.cross_val_poisson_hmm(cells_to_use=cells_to_use, cl_ar=cl_ar_init)
        # get optimal number of modes for coarse grained
        trials_to_use = self.default_trials
        folder_name = self.session_name + "_" + str(
            int(self.experiment_phase_id)) + "_" + self.cell_type + "_trials_"+str(trials_to_use[0])+\
                      "_"+str(trials_to_use[-1])
        new_ml = MlMethodsOnePopulation(params=self.params)
        opt_nr_coarse = new_ml.get_optimal_mode_number(folder_name=folder_name, cells_used=cells_to_use)
        print("Coarse opt. number of modes: "+str(opt_nr_coarse))
        self.cross_val_poisson_hmm(cells_to_use=cells_to_use, cl_ar=np.arange(opt_nr_coarse - 2, opt_nr_coarse + 3, 2))
        opt_nr_fine = new_ml.get_optimal_mode_number(folder_name=folder_name, cells_used=cells_to_use)
        print("Fine opt. number of modes: " + str(opt_nr_fine))
        self.fit_poisson_hmm(nr_modes=opt_nr_fine, cells_to_use=cells_to_use)

    def view_cross_val_results(self, trials_to_use=None, range_to_plot=None, save_fig=False, cells_used="all_cells"):
        # --------------------------------------------------------------------------------------------------------------
        # views cross validation results
        #
        # args:     - model_type, string: which type of model ("POISSON_HMM")
        #           - custom_splits, bool: whether custom splits were used for cross validation
        # --------------------------------------------------------------------------------------------------------------
        if trials_to_use is None:
            trials_to_use = self.default_trials

        folder_name = self.session_name + "_" + str(
            int(self.experiment_phase_id[0])) + "_" + self.cell_type + "_trials_"+str(trials_to_use[0])+\
                      "_"+str(trials_to_use[-1]+1)
        new_ml = MlMethodsOnePopulation(params=self.params)
        new_ml.cross_val_view_results(folder_name=folder_name, range_to_plot=range_to_plot, save_fig=save_fig, 
                                      cells_used=cells_used)

    def get_optimal_number_states(self, trials_to_use=None, cells_used="all_cells"):
        if trials_to_use is None:
            trials_to_use = self.default_trials


        if self.params.cross_val_splits == "custom_splits":
            res_dir = self.params.pre_proc_dir+"phmm/cross_val/"+cells_used+"/custom_splits/"
        elif self.params.cross_val_splits == "standard_k_fold":
            res_dir = self.params.pre_proc_dir + "phmm/cross_val/"+cells_used+"/standard_k_fold/"
        elif self.params.cross_val_splits == "trial_splitting":
            res_dir = self.params.pre_proc_dir + "phmm/cross_val/"+cells_used+"/trial_splitting/"


        folder_name = self.session_name + "_" + str(
            int(self.experiment_phase_id)) + "_" + self.cell_type + "_trials_" + str(trials_to_use[0]) + \
                      "_" + str(trials_to_use[-1] + 1)
        if not os.path.isdir(res_dir+folder_name):
            folder_name = self.session_name + "_" + str(
                int(self.experiment_phase_id)) + "_" + self.cell_type + "_trials_"+str(trials_to_use[0])+\
                          "_"+str(trials_to_use[-1])


        new_ml = MlMethodsOnePopulation(params=self.params)
        opt_modes = new_ml.get_optimal_mode_number(folder_name=folder_name, cells_used=cells_used)
        return opt_modes

    def fit_poisson_hmm(self, nr_modes, trials_to_use=None, cells_to_use="all_cells"):
        # --------------------------------------------------------------------------------------------------------------
        # fits poisson hmm to data
        #
        # args:     - nr_modes, int: #clusters to fit to data
        #           - file_identifier, string: string that is added at the end of file for identification
        # --------------------------------------------------------------------------------------------------------------

        print(" - FITTING POISSON HMM WITH "+str(nr_modes)+" MODES ...\n")

        if trials_to_use is None:
            trials_to_use = self.default_trials

        # check how many cells
        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))

        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))

        # if subset of cells to use
        if cells_to_use == "stable_cells":
            # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            stable_cells = class_dic["stable_cell_ids"].flatten()
            raster = raster[stable_cells, :]

        elif cells_to_use == "decreasing_cells":
                        # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            dec_cells = class_dic["decrease_cell_ids"].flatten()
            raster = raster[dec_cells, :]

        elif cells_to_use == "increasing_cells":
                        # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            inc_cells = class_dic["increase_cell_ids"].flatten()
            raster = raster[inc_cells, :]

        log_li = -1*np.inf
        # fit 10 times to select model with best highest log-likelihood (NOT CROSS-VALIDATED!!!)
        for i in range(10):
            test_model = PoissonHMM(n_components=nr_modes)
            test_model.fit(raster.T)
            log_li_test = test_model.score(raster.T)
            if log_li_test > log_li:
                model = test_model
                log_li = log_li_test

        model.set_time_bin_size(time_bin_size=self.params.time_bin_size)

        if cells_to_use == "stable_cells":
            save_dir = self.params.pre_proc_dir+"phmm/stable_cells/"
        elif cells_to_use == "decreasing_cells":
            save_dir = self.params.pre_proc_dir+"phmm/decreasing_cells/"
        elif cells_to_use == "increasing_cells":
            save_dir = self.params.pre_proc_dir+"phmm/increasing_cells/"
        elif cells_to_use == "all_cells":
            save_dir = self.params.pre_proc_dir+"phmm/"
        file_name = self.session_name + "_" + str(
            int(self.experiment_phase_id)) + "_" + self.cell_type + "_trials_"+str(trials_to_use[0])+\
                      "_"+str(trials_to_use[-1])+"_"+str(nr_modes)+"_modes"

        with open(save_dir+file_name+".pkl", "wb") as file: pickle.dump(model, file)

        print("  - ... DONE!\n")

    def evaluate_poisson_hmm(self, nr_modes=None, trials_to_use=None, save_fig=False, plotting=False):
        # --------------------------------------------------------------------------------------------------------------
        # fits poisson hmm to data and evaluates the goodness of the model by comparing basic statistics (avg. firing
        # rate, correlation values, k-statistics) between real data and data sampled from the model
        #
        # args:     - nr_modes, int: #clusters to fit to data
        #           - load_from_file, bool: whether to load model from file or to fit model again
        # --------------------------------------------------------------------------------------------------------------

        print(" - EVALUATING POISSON HMM FIT (BASIC STATISTICS) ...")

        if nr_modes is None:
            # get optimal number of modes
            nr_modes = self.get_optimal_number_states()
        if trials_to_use is None:
            trials_to_use = self.default_trials

        # check how many cells
        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
        nr_time_bins = raster.shape[1]
        # X = X[:, :1000]

        file_name = self.session_name + "_" + str(
            int(self.experiment_phase_id[0])) + "_" + self.cell_type + "_trials_"+str(trials_to_use[0])+\
                      "_"+str(trials_to_use[-1])+"_"+str(nr_modes)+"_modes"

        # check if model file exists already --> otherwise fit model again
        if os.path.isfile(self.params.pre_proc_dir+"phmm/" + file_name + ".pkl"):
            print("- LOADING PHMM MODEL FROM FILE\n")
            with open(self.params.pre_proc_dir+"phmm/" + file_name + ".pkl", "rb") as file:
                model = pickle.load(file)
        else:
            print("- PHMM MODEL FILE NOT FOUND --> FITTING PHMM TO DATA\n")
            model = PoissonHMM(n_components=nr_modes)
            model.fit(raster.T)

        samples, sequence = model.sample(nr_time_bins*50)
        samples = samples.T

        mean_dic, corr_dic, k_dic = evaluate_clustering_fit(real_data=raster, samples=samples, binning="TEMPORAL_SPIKE",
                                                            time_bin_size=0.1, plotting=False)

        print("Mean firing:")
        print(mean_dic["corr"])
        print("Correlations:")
        print(corr_dic["corr"])

        if save_fig or plotting:

            plt.style.use('default')
            k_samples_sorted = np.sort(k_dic["samples"])
            k_data_sorted = np.sort(k_dic["real"])

            p_samples = 1. * np.arange(k_samples_sorted.shape[0]) / (k_samples_sorted.shape[0] - 1)
            p_data = 1. * np.arange(k_data_sorted.shape[0]) / (k_data_sorted.shape[0] - 1)
            # plt.hlines(0.5, -0.02, 0.85, color="gray", linewidth=0.5)
            plt.plot(k_data_sorted, p_data, color="darkorange", label="Data")
            plt.plot(k_samples_sorted, p_samples, color="bisque", label="Model")
            plt.ylabel("CDF")
            plt.xlabel("% active cells per time bin")
            plt.legend()
            plt.title("Model quality: k-statistic")
            make_square_axes(plt.gca())

            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "model_qual_k.svg"), transparent="True")
            plt.close()

            # plot single figures
            plt.plot([-0.2, max(np.maximum(corr_dic["samples"].flatten(), corr_dic["real"].flatten()))],
                     [-0.2,max(np.maximum(corr_dic["samples"].flatten(),corr_dic["real"].flatten()))],
                     linestyle="dashed", c="gray")
            plt.scatter(corr_dic["samples"].flatten(), corr_dic["real"].flatten(), color="pink")
            plt.xlabel("Correlations (samples)")
            plt.ylabel("Correlations (data)")
            plt.title("Model quality: correlations ")
            plt.text(0.0, 0.9, "R = " + str(round(corr_dic["corr"][0], 4)))
            make_square_axes(plt.gca())
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "model_qual_correlations.svg"), transparent="True")
                plt.close()
            else:
                plt.show()

            # plot single figures
            plt.plot([0, max(np.maximum(mean_dic["samples"].flatten(), mean_dic["real"].flatten()))],
                     [0,max(np.maximum(mean_dic["samples"].flatten(),mean_dic["real"].flatten()))],
                     linestyle="dashed", c="gray")
            plt.scatter(mean_dic["samples"].flatten(), mean_dic["real"].flatten(), color="turquoise")
            plt.xlabel("Mean firing rate (samples)")
            plt.ylabel("Mean firing rate (data)")
            plt.title("Model quality: mean firing ")
            plt.text(1.2, 12, "R = " + str(round(mean_dic["corr"][0], 4)))
            make_square_axes(plt.gca())
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "model_qual_mean_firing.svg"), transparent="True")
                plt.close()
            else:
                plt.show()

        return mean_dic, corr_dic, k_dic

    def evaluate_multiple_poisson_hmm_models(self, nr_modes_range, trials_to_use=None):
        # --------------------------------------------------------------------------------------------------------------
        # fits poisson hmm to data and evaluates the goodness of the model by comparing basic statistics (avg. firing
        # rate, correlation values, k-statistics) between real data and data sampled from the model
        #
        # args:     - nr_modes, int: #clusters to fit to data
        #           - load_from_file, bool: whether to load model from file or to fit model again
        # --------------------------------------------------------------------------------------------------------------

        print("- EVALUATING POISSON HMM FIT (BASIC STATISTICS) ...")

        if trials_to_use is None:
            trials_to_use = self.default_trials

        # check how many cells
        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
        nr_time_bins = raster.shape[1]
        # X = X[:, :1000]

        mean_res = []
        corr_res = []
        k_res = []

        for nr_modes in nr_modes_range:
            fit_mean_res = []
            fit_corr_res = []
            fit_k_res = []
            for nr_fit in range(10):
                print(" - FITTING PHMM MODEL WITH "+str(nr_modes)+" MODES - FIT NR: "+str(nr_fit))
                model = PoissonHMM(n_components=nr_modes)
                model.fit(raster.T)

                samples, sequence = model.sample(nr_time_bins*50)
                samples = samples.T
                mean_dic, corr_dic, k_dic = evaluate_clustering_fit(real_data=raster, samples=samples,
                                                                    binning="TEMPORAL_SPIKE",
                                                                    time_bin_size=0.1, plotting=False)
                fit_mean_res.append(mean_dic["corr"][0])
                fit_corr_res.append(corr_dic["corr_triangles"][0])
                fit_k_res.append(k_dic["diff_med"])


            mean_res.append([np.mean(np.array(fit_mean_res)), np.std(np.array(fit_mean_res))])
            corr_res.append([np.mean(np.array(fit_corr_res)), np.std(np.array(fit_corr_res))])
            k_res.append([np.mean(np.array(fit_k_res)), np.std(np.array(fit_k_res))])

        mean_res = np.array(mean_res)
        corr_res = np.array(corr_res)
        k_res = np.array(k_res)

        # plot results
        plt.figure(figsize=(5,10))
        plt.subplot(3,1,1)
        plt.errorbar(nr_modes_range, mean_res[:,0],yerr=mean_res[:,1], color="r")
        plt.grid(color="gray")
        plt.ylabel("PEARSON R: MEAN FIRING")
        plt.ylim(min(mean_res[:,0])-max(mean_res[:,1]), max(mean_res[:,0])+max(mean_res[:,1]))
        plt.subplot(3,1,2)
        plt.errorbar(nr_modes_range, corr_res[:,0],yerr=corr_res[:,1], color="r")
        plt.ylabel("PEARSON R: CORR")
        plt.grid(color="gray")
        plt.ylim(min(corr_res[:,0])-max(corr_res[:,1]), max(corr_res[:,0])+max(corr_res[:,1]))
        plt.subplot(3,1,3)
        plt.errorbar(nr_modes_range, k_res[:,0],yerr=k_res[:,1], color="r")
        plt.ylabel("DIFF. MEDIANS")
        plt.xlabel("NR. MODES")
        plt.grid(color="gray")
        plt.ylim(min(k_res[:,0])-max(k_res[:,1]), max(k_res[:,0])+max(k_res[:,1]))
        plt.show()

    def decode_poisson_hmm(self, trials_to_use=None, file_name=None, cells_to_use="all"):
        # --------------------------------------------------------------------------------------------------------------
        # loads model from file and decodes data
        #
        # args:     - nr_modes, int: #clusters to fit to data --> used to identify file that fits the data
        #           - file_name, string:    is used if model from a different experiment phase is supposed to be used
        #                                   (e.g. model from awake is supposed to be fit to sleep data)
        # --------------------------------------------------------------------------------------------------------------

        if trials_to_use is None:
            trials_to_use = self.default_trials

        # check how many cells
        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
        nr_time_bins = raster.shape[1]

        if file_name is None:
            file_name = self.default_phmm
        if cells_to_use == "all":
            with open(self.params.pre_proc_dir+"phmm/"+file_name+".pkl", "rb") as file:
                model = pickle.load(file)
        elif cells_to_use == "stable":
            with open(self.params.pre_proc_dir+"phmm/stable_cells/"+file_name+".pkl", "rb") as file:
                model = pickle.load(file)
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                      "rb") as f:
                class_dic = pickle.load(f)
            raster = raster[class_dic["stable_cell_ids"], :]
        elif cells_to_use == "decreasing":
            with open(self.params.pre_proc_dir+"phmm/decreasing_cells/"+file_name+".pkl", "rb") as file:
                model = pickle.load(file)
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                      "rb") as f:
                class_dic = pickle.load(f)
            raster = raster[class_dic["decrease_cell_ids"], :]

        nr_modes_ = model.means_.shape[0]

        # compute most likely sequence
        model.implementation = "log"
        sequence = model.predict(raster.T)
        post_prob = model.predict_proba(raster.T)
        return sequence, nr_modes_, post_prob

    def load_poisson_hmm(self, trials_to_use=None, nr_modes=None, file_name=None, cells_to_use="all_cells"):
        # --------------------------------------------------------------------------------------------------------------
        # loads model from file and returns model
        #
        # args:     - nr_modes, int: #clusters to fit to data --> used to identify file that fits the data
        #           - file_name, string:    is used if model from a different experiment phase is supposed to be used
        #                                   (e.g. model from awake is supposed to be fit to sleep data)
        # --------------------------------------------------------------------------------------------------------------

        if trials_to_use is None:
            trials_to_use = self.default_trials

        if (nr_modes is None) & (file_name is None):
            raise Exception("PROVIDE NR. MODES OR FILE NAME")

        if file_name is None:
            file_name = self.params.session_name + "_" + str(
                int(self.params.experiment_phase_id[0])) + "_" + self.cell_type + "_trials_" + str(trials_to_use[0]) + \
                        "_" + str(trials_to_use[-1]) + "_" + str(nr_modes) + "_modes"
        else:
            file_name =file_name

        if cells_to_use == "stable_cells":
            save_dir = self.params.pre_proc_dir+"phmm/stable_cells"
        elif cells_to_use == "decreasing_cells":
            save_dir = self.params.pre_proc_dir+"phmm/decreasing_cells"
        elif cells_to_use == "all_cells":
            save_dir = self.params.pre_proc_dir+"phmm/"

        with open(save_dir+file_name+".pkl", "rb") as file:
            model = pickle.load(file)

        return model

    def fit_spatial_gaussians_for_modes(self, nr_modes=None, file_name=None, trials_to_use=None,
                                        min_nr_bins_active=5, plot_awake_fit=False, plot_modes=False):

        if trials_to_use is None:
            trials_to_use = self.default_trials

        # get location data from trials
        nr_cells = self.trial_raster_list[0].shape[0]
        loc = np.empty((0,2))
        raster = np.empty((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            loc = np.vstack((loc, self.trial_loc_list[trial_id]))

        state_sequence, nr_modes, _ = self.decode_poisson_hmm(file_name=file_name,
                                                           trials_to_use=trials_to_use)

        mode_id, freq = np.unique(state_sequence, return_counts=True)
        modes_to_plot = mode_id[freq > min_nr_bins_active]

        cmap = generate_colormap(nr_modes)
        if plot_modes:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for mode in np.arange(nr_modes):
                mode_data = loc[state_sequence == mode, :]
                # rgb = (random.random(), random.random(), random.random())
                ax.scatter(mode_data[:, 0], mode_data[:, 1],
                           alpha=1, marker=".", s=1, label="DIFFERENT MODES", c=np.array([cmap(mode)]))

            ax.set_ylim(self.y_min, self.y_max)
            ax.set_xlim(self.x_min, self.x_max)
            for g_l in self.goal_locations:
                ax.scatter(g_l[0], g_l[1], color="w", label="GOALS")

            plt.gca().set_aspect('equal', adjustable='box')
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.title("MODE ASSIGNMENT AWAKE DATA")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

            for mode in np.arange(nr_modes):
                mode_data = loc[state_sequence == mode, :]
                fig, ax = plt.subplots()
                ax.scatter(loc[:,0], loc[:,1], color="grey", s=1, label="TRACKING")
                ax.scatter(mode_data[:, 0], mode_data[:, 1],
                       alpha=1, marker=".", s=1, label="MODE " + str(mode) + " ASSIGNED",
                           color="red")
                for g_l in self.goal_locations:
                    ax.scatter(g_l[0], g_l[1], color="w", label="GOALS")
                ax.set_ylim(self.y_min, self.y_max)
                ax.set_xlim(self.x_min, self.x_max)
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.gca().set_aspect('equal', adjustable='box')
                handles, labels = ax.get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())
                plt.show()

        means = np.zeros((2, nr_modes))
        cov = np.zeros((2, nr_modes))
        for mode in np.arange(nr_modes):
            mode_data = loc[state_sequence == mode, :]

            if len(mode_data) == 0:
                means[:, mode] = np.nan
                cov[:, mode] = np.nan
            else:
                means[:, mode] = np.mean(mode_data, axis=0)
                cov[:, mode] = np.var(mode_data, axis=0)

        loc_data = loc[:, :]

        center = np.min(loc_data, axis=0) + (np.max(loc_data, axis=0) - np.min(loc_data, axis=0)) * 0.5
        dist = loc_data - center

        rad = max(np.sqrt(np.square(dist[:, 0]) + np.square(dist[:, 1]))) + 1

        std_modes = np.sqrt(cov[0,:]+cov[1,:])
        std_modes[std_modes == 0] = np.nan

        if plot_awake_fit:

            for mode_to_plot in modes_to_plot:

                mean = means[:, mode_to_plot]
                cov_ = cov[:, mode_to_plot]
                std_ = std_modes[mode_to_plot]

                # Parameters to set
                mu_x = mean[0]
                variance_x = cov_[0]

                mu_y = mean[1]
                variance_y = cov_[1]

                # Create grid and multivariate normal
                x = np.linspace(center[0] - rad, center[0]+rad, int(2.2*rad))
                y = np.linspace(0, 250, 250)
                X, Y = np.meshgrid(x, y)
                pos = np.empty(X.shape + (2,))
                pos[:, :, 0] = X
                pos[:, :, 1] = Y
                rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
                rv_normalized = rv.pdf(pos) / np.sum(rv.pdf(pos).flatten())

                fig, ax = plt.subplots()
                gauss = ax.imshow(rv_normalized)
                env = Circle((center[0], center[1]), rad, color="white", fill=False)
                ax.add_artist(env)
                ax.set_ylim(center[1] - 1.1*rad, center[1]+1.1*rad)
                ax.scatter(loc_data[state_sequence == mode_to_plot, 0], loc_data[state_sequence == mode_to_plot, 1],
                           alpha=1, c="white", marker=".", s=0.3, label="MODE "+ str(mode_to_plot) +" ASSIGNED")
                cb = plt.colorbar(gauss)
                cb.set_label("PROBABILITY")
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.title("STD: "+str(np.round(std_, 2)))
                plt.legend()
                plt.show()

        # compute frequency of each mode
        mode_freq = np.zeros(nr_modes)
        mode_freq[mode_id] = freq
        mode_freq = mode_freq.astype(int)

        env = Circle((center[0], center[1]), rad, color="white", fill=False)

        result_dic = {
            "std_modes": std_modes
        }

        with open("temp_data/test1", "wb") as f:
            pickle.dump(result_dic, f)


        return means, std_modes, mode_freq, env, state_sequence

    def plot_all_phmm_modes_spatial(self, nr_modes):
        for i in range(nr_modes):
            self.plot_phmm_mode_spatial(mode_id = i)

    def plot_phmm_mode_spatial(self, mode_id, ax=None, save_fig=False, use_viterbi=True, cells_to_use="all"):
        plt.style.use('default')
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        trials_to_use = self.default_trials

        # get location data from trials
        nr_cells = self.trial_raster_list[0].shape[0]
        loc = np.empty((0,2))
        raster = np.empty((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            loc = np.vstack((loc, self.trial_loc_list[trial_id]))

        # plot all tracking data
        ax.scatter(loc[:,0], loc[:,1], color="lightblue", s=1, label="Tracking", zorder=-1000, edgecolor="lightblue")

        if use_viterbi:
            if cells_to_use == "all":
                file_name = self.default_phmm
            elif cells_to_use == "stable":
                file_name = self.default_phmm_stable
            state_sequence, nr_modes, _ = self.decode_poisson_hmm(file_name=file_name,
                                                               trials_to_use=trials_to_use, cells_to_use=cells_to_use)
        else:
            cells_to_use = "stable"
            # load phmm model
            with open(self.params.pre_proc_dir + "phmm/" + self.default_phmm + '.pkl', 'rb') as f:
                model_dic = pickle.load(f)
            # get means of model (lambdas) for decoding
            mode_means = model_dic.means_
            cell_selection = "custom"
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            if cells_to_use == "stable":
                cell_ids = class_dic["stable_cell_ids"]
            results_list = decode_using_phmm_modes(mode_means=mode_means, event_spike_rasters=[raster],
                                                   compression_factor=1, cells_to_use=cell_ids,
                                                   cell_selection=cell_selection)
            likelihoods = results_list[0]
            state_sequence = np.argmax(likelihoods, axis=1)

        for g_l in self.goal_locations:
            ax.scatter(g_l[0], g_l[1], color="black", label="Goal locations")

        mode_data = loc[state_sequence == mode_id, :]
        ax.scatter(mode_data[:, 0], mode_data[:, 1],
                   alpha=1, marker=".", s=1.5, color="crimson", edgecolor="crimson")
        # ax.set_ylim(30, 230)
        # ax.set_xlim(70, 300)
        #ax.set_xlim(self.x_min, self.x_max)
        plt.xlabel("X (cm)")
        plt.ylabel("Y (cm)")
        plt.gca().set_aspect('equal', adjustable='box')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("State " + str(mode_id))
        if save_fig:
            plt.savefig(os.path.join(save_path, "state_"+str(mode_id)+".svg"), transparent="True")
            plt.rcParams['svg.fonttype'] = 'none'
        else:
            if ax is None:
                plt.show()

    def plot_phmm_state_neural_patterns(self, nr_modes):

        trials_to_use = self.default_trials

        file_name = self.params.session_name + "_" + str(
            int(self.params.experiment_phase_id[0])) + "_" + self.cell_type + "_trials_"+str(trials_to_use[0])+\
                      "_"+str(trials_to_use[-1])+"_"+str(nr_modes)+"_modes"

        with open(self.params.pre_proc_dir+"phmm/" + file_name + ".pkl", "rb") as file:
            model = pickle.load(file)

        # get means
        means = model.means_

        x_max_ = np.max(means.flatten())
        x_min_ = np.min(means.flatten())

        n_col = 10
        scaler = 0.1
        plt.style.use('default')
        fig = plt.figure(figsize=(8,6))
        gs = fig.add_gridspec(6, n_col)

        ax1 = fig.add_subplot(gs[:, 0])
        ax1.imshow(np.expand_dims(means[0, :], 1))
        ax1.set_xticks([])
        ax1.set_xlabel(str(0))
        ax1.set_aspect(scaler)
        ax1.set_ylabel("CELL ID")

        for i in range(1, n_col-1):
            ax1 = fig.add_subplot(gs[:, i])
            a = ax1.imshow(np.expand_dims(means[i,:], 1))
            ax1.set_xticks([])
            ax1.set_yticks([], [])
            ax1.set_xlabel(str(i))
            ax1.set_aspect(scaler)

        plt.tight_layout()
        ax1 = fig.add_subplot(gs[:, n_col-1])
        fig.colorbar(a, cax=ax1)
        plt.rcParams['svg.fonttype'] = 'none'
        #plt.show()
        plt.savefig(os.path.join(save_path, "state_neural_pattern.svg"), transparent="True")

        print("HERE")

    def analyze_modes_goal_coding(self, file_name, mode_ids, thr_close_to_goal=10, plotting=False):

        trials_to_use = self.default_trials

        # get location data from trials
        nr_cells = self.trial_raster_list[0].shape[0]
        loc = np.empty((0,2))
        raster = np.empty((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            loc = np.vstack((loc, self.trial_loc_list[trial_id]))

        state_sequence, nr_modes, _ = self.decode_poisson_hmm(file_name=file_name,
                                                           trials_to_use=trials_to_use)
        # only use data from mode ids provided
        mode_data = loc[np.isin(state_sequence, mode_ids), :]

        # compute distance from each goal
        close_to_goal = []
        for g_l in self.goal_locations:
            dist_g_l = np.linalg.norm(mode_data - g_l, axis=1)
            close_to_goal.append(dist_g_l < thr_close_to_goal)

        close_to_goal = np.array(close_to_goal)
        close_to_goal = np.logical_or.reduce(close_to_goal, axis=0)

        if close_to_goal.shape[0] > 0:
            # compute fraction of points that lie within border of goal location
            frac_close_to_goal = np.count_nonzero(close_to_goal) / close_to_goal.shape[0]
        else:
            frac_close_to_goal = 0

        if plotting:
            for g_l in self.goal_locations:
                plt.scatter(g_l[0], g_l[1], color="w")
            plt.scatter(mode_data[:,0], mode_data[:,1], color="gray", s=1, label="ALL LOCATIONS")
            plt.scatter(mode_data[close_to_goal,0], mode_data[close_to_goal,1], color="red", s=1, label="CLOSE TO GOAL")
            plt.title("LOCATIONS OF SELECTED MODES (GOAL RADIUS: "+str(thr_close_to_goal)+"cm)\n "
                      "FRACTION CLOSE TO GOAL: "+str(np.round(frac_close_to_goal,2)))
            plt.legend()
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

        return frac_close_to_goal

    def analyze_all_modes_goal_coding(self, thr_close_to_goal=10):

        # get default model
        phmm_file_name = self.default_phmm

        # check how many modes
        with open(self.params.pre_proc_dir + "phmm/" + phmm_file_name + ".pkl", "rb") as file:
            model = pickle.load(file)

        nr_modes = model.means_.shape[0]

        frac_per_mode = []
        for mode_id in range(nr_modes):
            frac_close_to_goal = self.analyze_modes_goal_coding(file_name=phmm_file_name, mode_ids=mode_id,
                                                                thr_close_to_goal=thr_close_to_goal)
            frac_per_mode.append(frac_close_to_goal)

        # self.plot_phmm_mode_spatial(mode_id=9)
        # plt.show()

        return np.hstack(frac_per_mode)

    def analyze_modes_spatial_information(self, file_name, mode_ids, plotting=True):

        trials_to_use = self.default_trials

        # get location data from trials
        nr_cells = self.trial_raster_list[0].shape[0]
        loc = np.empty((0,2))
        raster = np.empty((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            loc = np.vstack((loc, self.trial_loc_list[trial_id]))

        state_sequence, nr_modes_, _ = self.decode_poisson_hmm(file_name=file_name,
                                                           trials_to_use=trials_to_use)

        _, loc_data, _ = self.get_raster_location_speed()

        # only use data from mode ids provided
        # all_modes_data = loc_data[np.isin(state_sequence, mode_ids), :]
        # pd_all_modes = upper_tri_without_diag(pairwise_distances(all_modes_data))
        # all_modes_med_dist = np.median(pd_all_modes)

        med_dist_list = []

        for mode in mode_ids:
            mode_loc = loc_data[state_sequence == mode, :]
            if mode_loc.shape[0] == 0:
                med_dist_list.append(np.nan)
                continue

            # compute pairwise distances (euclidean)
            pd = upper_tri_without_diag(pairwise_distances(mode_loc))

            med_dist = np.median(pd)
            # std_dist = np.std(np.array(dist_list))
            med_dist_list.append(med_dist)

            if plotting:

                fig, ax = plt.subplots()
                ax.scatter(loc_data[:,0], loc_data[:,1], color="gray", s=1, label="TRACKING")
                ax.scatter(mode_loc[:, 0], mode_loc[:, 1], color="red", label="MODE "+str(mode)+" ASSIGNED", s=1)
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.title("MODE "+str(mode)+"\nMEDIAN DISTANCE: "+ str(np.round(med_dist,2)))
                plt.legend()
                plt.show()

        return np.array(med_dist_list)

    def analyze_all_modes_spatial_information(self):

        # get default model
        phmm_file_name = self.default_phmm

        # check how many modes
        with open(self.params.pre_proc_dir + "phmm/" + phmm_file_name + ".pkl", "rb") as file:
            model = pickle.load(file)

        nr_modes = model.means_.shape[0]

        med_distance_per_mode = []
        for mode_id in range(nr_modes):
            dist_mode = self.analyze_modes_spatial_information(file_name=phmm_file_name, mode_ids=[mode_id],
                                                               plotting=False)
            med_distance_per_mode.append(dist_mode)

        med_distance_per_modes = np.hstack(med_distance_per_mode)
        # self.plot_phmm_mode_spatial(mode_id=33)
        # plt.show()

        return med_distance_per_modes

    def phmm_mode_spatial_information_from_model(self, spatial_resolution=1, file_name=None,
                                            plot_for_control=False):
        """
        loads poisson hmm model and weighs rate maps by lambda vectors --> then computes spatial information (sparsity,
        skaggs information)

        @param spatial_resolution: spatial resolution in cm
        @type spatial_resolution: int
        @param nr_modes: nr of modes for model file identification
        @type nr_modes: int
        @param file_name: file containing the model --> is used when nr_modes is not provided to identify file
        @type file_name: string
        @param plot_for_control: whether to plot intermediate results
        @type plot_for_control: bool
        @return: sparsity, skaggs info for each mode
        """

        print(" - SPATIAL INFORMATION OF PHMM MODES USING MODEL\n")

        if file_name is None:
            file_name = self.default_phmm

        with open(self.params.pre_proc_dir+"phmm/"+file_name+".pkl", "rb") as file:
            model = pickle.load(file)

        # get means for all modes (lambda vectors)
        means = model.means_

        ################################################################################################################
        # get spatial information of mode by weighing rate maps
        ################################################################################################################

        # compute rate maps and occupancy
        rate_maps = self.get_rate_maps(spatial_resolution=spatial_resolution, trials_to_use=None)
        occ = self.get_occ_map(spatial_resolution=spatial_resolution, trials_to_use=None)
        prob_occ = occ / occ.sum()
        # compute binary occupancy map --> used as a mask to filter non-visited bins
        occ_mask = np.where(occ > 0, 1, np.nan)
        prob_occ_orig = np.multiply(prob_occ, occ_mask)

        sparsity_list = []
        skaggs_per_second_list = []
        skaggs_per_spike_list = []

        # go through all modes
        for mode_id, means_mode in enumerate(means):
            # weigh rate map of each cell using mean firing from lambda vector --> compute mean across all cells
            rate_map_mode_orig = np.mean(rate_maps * means_mode, axis=2)
            # generate filtered rate map by masking non visited places
            rate_map_mode_orig = np.multiply(rate_map_mode_orig, occ_mask)

            rate_map_mode = rate_map_mode_orig[~np.isnan(rate_map_mode_orig)]
            prob_occ = prob_occ_orig[~np.isnan(prob_occ_orig)]

            # need to filter bins with zero firing rate --> otherwise log causes an error
            rate_map_mode = rate_map_mode[rate_map_mode > 0]
            prob_occ = prob_occ[rate_map_mode > 0]

            # compute sparsity
            sparse_mode = (np.sum(prob_occ * rate_map_mode) ** 2) / np.sum(prob_occ * (rate_map_mode ** 2))

            # find good bins so that there is no problem with the log
            good_bins = (rate_map_mode / rate_map_mode.mean() > 0.0000001)
            mean_rate = np.sum(rate_map_mode[good_bins] * prob_occ[good_bins])
            skaggs_info_per_sec = np.sum(rate_map_mode[good_bins] * prob_occ[good_bins] *
                                         np.log(rate_map_mode[good_bins] / mean_rate))
            skaggs_info_per_spike = np.sum(rate_map_mode[good_bins] / mean_rate * prob_occ[good_bins] *
                                           np.log(rate_map_mode[good_bins] / mean_rate))

            skaggs_per_second_list.append(skaggs_info_per_sec)
            skaggs_per_spike_list.append(skaggs_info_per_spike)
            sparsity_list.append(sparse_mode)
            if plot_for_control:
                # plot random examples
                rand_float = np.random.randn(1)
                if rand_float > 0.5:
                    plt.imshow(rate_map_mode_orig)
                    plt.colorbar()
                    plt.title("Sparsity: "+str(sparse_mode)+"\n Skaggs per second: "+str(skaggs_info_per_sec)+
                              "\n Skaggs per spike: "+ str(skaggs_info_per_spike))
                    plt.show()

        if plot_for_control:
            plt.hist(skaggs_per_second_list)
            plt.title("SKAGGS INFO (PER SECOND)")
            plt.xlabel("SKAGGS INFO.")
            plt.ylabel("COUNTS")
            plt.show()

            plt.hist(skaggs_per_spike_list)
            plt.title("SKAGGS INFO (PER SPIKE)")
            plt.xlabel("SKAGGS INFO.")
            plt.ylabel("COUNTS")
            plt.show()

            plt.hist(sparsity_list)
            plt.title("SPARSITY")
            plt.xlabel("SPARSITY")
            plt.ylabel("COUNTS")
            plt.show()

            plt.scatter(skaggs_per_second_list, sparsity_list)
            plt.title("SKAGGS (PER SEC) vs. SPARSITY\n"+str(pearsonr(skaggs_per_second_list, sparsity_list)))
            plt.xlabel("SKAGGS (PER SEC)")
            plt.ylabel("SPARSITY")
            plt.show()

            plt.scatter(skaggs_per_spike_list, sparsity_list)
            plt.title("SKAGGS (PER SPIKE) vs. SPARSITY\n"+str(pearsonr(skaggs_per_spike_list, sparsity_list)))
            plt.xlabel("SKAGGS (PER SPIKE)")
            plt.ylabel("SPARSITY")
            plt.show()

            plt.scatter(skaggs_per_second_list, skaggs_per_spike_list)
            plt.title("SKAGGS (PER SEC) vs. SKAGGS (PER SPIKE)\n"+str(pearsonr(skaggs_per_second_list, skaggs_per_spike_list)))
            plt.xlabel("SKAGGS (PER SEC)")
            plt.ylabel("SKAGGS (PER SPIKE)")
            plt.show()

        return np.array(sparsity_list), np.array(skaggs_per_second_list), np.array(skaggs_per_spike_list)

    def nr_spikes_per_mode(self, trials_to_use=None, nr_modes=None):

        if trials_to_use is None:
            trials_to_use = self.default_trials

        model = self.load_poisson_hmm(trials_to_use, nr_modes)
        means = model.means_

        spikes_per_mode = np.sum(means, axis=1)
        y,_,_ = plt.hist(spikes_per_mode, bins=50)
        plt.vlines(np.mean(spikes_per_mode), 0, y.max(), colors="r",
                   label="MEAN: "+str(np.round(np.mean(spikes_per_mode),2)))
        plt.vlines(np.median(spikes_per_mode), 0, y.max(), colors="blue",
                   label="MEDIAN: "+str(np.median(spikes_per_mode)))
        plt.legend()
        plt.title("#SPIKES PER MODE")
        plt.xlabel("AVG. #SPIKES PER MODE")
        plt.ylabel("COUNT")
        plt.show()

    def decode_awake_activity_spike_binning(self, model_name=None, trials_to_use=None, plot_for_control=False,
                                   return_results=True, cells_to_use="all"):
        """
        decodes sleep activity using pHMM modes/spatial bins from ising model from before and after awake activity and
        computes similarity measure

        @param model_name: name of file containing the pHMM file
        @type model_name: str
        @param plot_for_control: plots intermediate results if True
        @type plot_for_control: bool
        @param return_results: whether to return results or not
        @type return_results: bool
        @return: pre_prob (probabilities of PRE), post_prob (probabilities of ), event_times (in sec),
        swr_in_which_n_rem (assignement SWR - nrem phase)
        @rtype: list, list, list, numpy.array
        """

        print(" - AWAKE DECODING USING PHMM MODES ...\n")
        result_dir = "phmm_decoding"
        # get template file name from parameter file of session if not provided
        if model_name is None:
            model_name = self.session_params.default_pre_phmm_model

        if model_name is None:
            raise Exception("MODEL FILE NOT FOUND\n NOR IN SESSION PARAMETER FILE DEFINED")

        if trials_to_use is None:
            trials_to_use = self.default_trials

        print("   --> USING TRIALS: "+str(trials_to_use[0])+"-"+str(trials_to_use[-1])+"\n")

        # check that template files are from correct session
        # --------------------------------------------------------------------------------------------------------------

        file_name = self.session_name +"_"+self.experiment_phase_id +\
                        "_"+ self.cell_type+"_AWAKE_DEC_"+cells_to_use+".npy"

        # check if PRE and POST result exists already
        # --------------------------------------------------------------------------------------------------------------
        if os.path.isfile(self.params.pre_proc_dir + result_dir + "/" + file_name):
            print(" - RESULTS EXIST ALREADY -- USING EXISTING RESULTS\n")
        else:
            # if results don't exist --> compute results
            # go trough all trials and compute constant #spike bins
            spike_rasters = []

            for trial_id in trials_to_use:
                key = "trial"+str(trial_id)
                # compute spike rasters --> time stamps here are at .whl resolution (20kHz/512 --> 0.0256s)
                spike_raster = PreProcessAwake(
                    firing_times=self.data_dic["trial_data"][key]["spike_times"][self.cell_type],
                    params=self.params, whl=self.data_dic["trial_data"][key]["whl"],
                    ).spike_binning()

                if plot_for_control:
                    if random.uniform(0, 1) > 0.5:
                        plt.imshow(spike_raster, interpolation='nearest', aspect='auto')
                        plt.title("TRIAL"+str(trial_id)+": CONST. #SPIKES BINNING, 12 SPIKES PER BIN")
                        plt.xlabel("BIN ID")
                        a = plt.colorbar()
                        a.set_label("#SPIKES")
                        plt.ylabel("CELL ID")
                        plt.show()

                        plt.imshow(self.trial_raster_list[trial_id], interpolation='nearest', aspect='auto')
                        plt.title("TIME BINNING, 100ms TIME BINS")
                        plt.xlabel("TIME BIN ID")
                        plt.ylabel("CELL ID")
                        a = plt.colorbar()
                        a.set_label("#SPIKES")
                        plt.show()

                spike_rasters.append(spike_raster)

            # load phmm model
            with open(self.params.pre_proc_dir + "phmm/" + model_name + '.pkl', 'rb') as f:
                model_dic = pickle.load(f)
            # get means of model (lambdas) for decoding
            mode_means = model_dic.means_

            time_bin_size_encoding = model_dic.time_bin_size

            # check if const. #spike bins are correct for the loaded compression factor
            if not self.params.spikes_per_bin == 12:
                raise Exception("TRYING TO LOAD COMPRESSION FACTOR FOR 12 SPIKES PER BIN, "
                                "BUT CURRENT #SPIKES PER BIN != 12")

            # load correct compression factor (as defined in parameter file of the session)
            if time_bin_size_encoding == 0.01:
                compression_factor = \
                    np.round(self.session_params.sleep_compression_factor_12spikes_100ms * 10, 3)
            elif time_bin_size_encoding == 0.1:
                compression_factor = self.session_params.sleep_compression_factor_12spikes_100ms
            else:
                raise Exception("COMPRESSION FACTOR NEITHER PROVIDED NOR FOUND IN PARAMETER FILE")

            if cells_to_use == "all":
                cell_selection = "all"
                cell_ids = np.empty(0)

            else:

                cell_selection = "custom"
                with open(self.params.pre_proc_dir + "cell_classification/" +
                          self.session_name +"_"+ self.params.stable_cell_method +".pickle","rb") as f:
                    class_dic = pickle.load(f)

                if cells_to_use == "stable":
                    cell_ids = class_dic["stable_cell_ids"]
                elif cells_to_use == "increasing":
                    cell_ids = class_dic["increase_cell_ids"]
                elif cells_to_use == "decreasing":
                    cell_ids = class_dic["decrease_cell_ids"]

            print(" - DECODING USING " + cells_to_use + " CELLS")

            # decode activity
            results_list = decode_using_phmm_modes(mode_means=mode_means, event_spike_rasters=spike_rasters,
                                                   compression_factor=compression_factor, cells_to_use=cell_ids,
                                                   cell_selection=cell_selection)

            # plot maps of some SWR for control
            if plot_for_control:
                for trial_id, res in zip(trials_to_use, results_list):
                    plt.imshow(np.log(res.T), interpolation='nearest', aspect='auto')
                    plt.xlabel("POP.VEC. ID")
                    plt.ylabel("MODE ID")
                    a = plt.colorbar()
                    a.set_label("LOG-PROBABILITY")
                    plt.title("LOG-PROBABILITY MAP: TRIAL "+str(trial_id))
                    plt.show()

            # saving results
            # --------------------------------------------------------------------------------------------------
            # create dictionary with results
            result_post = {
                "results_list": results_list,
            }
            outfile = open(self.params.pre_proc_dir + result_dir +"/" + file_name, 'wb')
            pickle.dump(result_post, outfile)
            print("  - SAVED NEW RESULTS!\n")

        if return_results:

            # load decoded maps
            result_pre = pickle.load(open(self.params.pre_proc_dir+result_dir + "/" + file_name, "rb"))

            pre_prob = result_pre["results_list"]

            return pre_prob

    def decode_awake_activity_time_binning(self, model_name=None, trials_to_use=None, plot_for_control=False,
                                   cells_to_use="all"):
        """

        @param model_name: name of file containing the pHMM file
        @type model_name: str
        @param plot_for_control: plots intermediate results if True
        @type plot_for_control: bool
        @param return_results: whether to return results or not
        @type return_results: bool
        @return: pre_prob (probabilities of PRE), post_prob (probabilities of ), event_times (in sec),
        swr_in_which_n_rem (assignement SWR - nrem phase)
        @rtype: list, list, list, numpy.array
        """

        # get template file name from parameter file of session if not provided
        if model_name is None:
            model_name = self.session_params.default_phmm_model

        if model_name is None:
            raise Exception("MODEL FILE NOT FOUND\n NOR IN SESSION PARAMETER FILE DEFINED")

        raster = self.get_raster(trials_to_use=trials_to_use)

        if cells_to_use == "all":
            # load phmm model
            with open(self.params.pre_proc_dir + "phmm/" + model_name + '.pkl', 'rb') as f:
                model_dic = pickle.load(f)
        elif cells_to_use == "stable":
            # load phmm model
            with open(self.params.pre_proc_dir + "phmm/stable_cells/" + model_name + '.pkl', 'rb') as f:
                model_dic = pickle.load(f)
        elif cells_to_use == "decreasing":
            # load phmm model
            with open(self.params.pre_proc_dir + "phmm/decreasing_cells/" + model_name + '.pkl', 'rb') as f:
                model_dic = pickle.load(f)
        elif cells_to_use == "increasing":
            # load phmm model
            with open(self.params.pre_proc_dir + "phmm/increasing_cells/" + model_name + '.pkl', 'rb') as f:
                model_dic = pickle.load(f)

        # get means of model (lambdas) for decoding
        mode_means = model_dic.means_
        time_bin_size_encoding = model_dic.time_bin_size

        if time_bin_size_encoding == self.params.time_bin_size:
            compression_factor = 1
        else:
            compression_factor = self.params.time_bin_size / time_bin_size_encoding

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+ self.params.stable_cell_method +".pickle","rb") as f:
            class_dic = pickle.load(f)

        if not cells_to_use == "all":
            if cells_to_use == "stable":
                cell_ids = class_dic["stable_cell_ids"]
            elif cells_to_use == "increasing":
                cell_ids = class_dic["increase_cell_ids"]
            elif cells_to_use == "decreasing":
                cell_ids = class_dic["decrease_cell_ids"]
            raster = raster[cell_ids,:]

        print(" - DECODING USING " + cells_to_use + " CELLS")

        # delete rasters with less than two spikes
        nr_spikes_per_bin = np.sum(raster, axis=0)
        raster=raster[:, nr_spikes_per_bin>2]

        # decode activity
        results_list = decode_using_phmm_modes(mode_means=mode_means, event_spike_rasters=[raster],
                                               compression_factor=compression_factor, cell_selection="all")

        # plot maps of some SWR for control
        if plot_for_control:
            plt.imshow(np.log(results_list[0].T[:,:200]), interpolation='nearest', aspect='auto')
            plt.xlabel("POP.VEC. ID")
            plt.ylabel("MODE ID")
            a = plt.colorbar()
            a.set_label("log-likelihood")
            plt.title("Log-likelihood")
            plt.show()

        return results_list

    def decode_awake_activity_visualization(self, cells_to_use="all", binning="spike_binning"):

        if binning == "spike_binning":
            results = self.decode_awake_activity_spike_binning(cells_to_use=cells_to_use)
        elif binning == "time_binning":
            results = self.decode_awake_activity_time_binning(cells_to_use=cells_to_use)

        a = np.vstack(results)
        b = np.argmax(a, axis=1)

        mode_ids, nr_counts = np.unique(b, return_counts=True)


        fig, ax = plt.subplots()

        lines = []
        for i in range(mode_ids.shape[0]):
            pair = [(mode_ids[i], 0), (mode_ids[i], nr_counts[i])]
            lines.append(pair)

        linecoll = matcoll.LineCollection(lines, colors="lightskyblue")

        ax.add_collection(linecoll)
        plt.scatter(mode_ids, nr_counts)
        plt.xlabel("Mode ID")
        plt.ylabel("Times assigned")
        plt.title("Awake decoding - "+cells_to_use)
        plt.show()

    def decode_awake_activity_autocorrelation_spikes_likelihood_vectors(self, plot_for_control=True, plotting=True,
                                                                        bootstrapping=False, nr_pop_vecs=10, save_fig=False):

        # get likelihood vectors
        likeli_vecs_list = self.decode_awake_activity_spike_binning()
        likeli_vecs = np.vstack(likeli_vecs_list)
        # compute correlations
        shift_array = np.arange(-1*int(nr_pop_vecs),
                                     int(nr_pop_vecs)+1)

        auto_corr, _ = cross_correlate_matrices(likeli_vecs.T, likeli_vecs.T, shift_array=shift_array)

        # fitting exponential
        # --------------------------------------------------------------------------------------------------------------
        # only take positive part (symmetric) & exclude first data point
        autocorr_test_data = auto_corr[int(auto_corr.shape[0] / 2):][1:]

        def exponential(x, a, k, b):
            return a * np.exp(x * k) + b

        popt_exponential_awake, pcov_exponential_awake = optimize.curve_fit(exponential, np.arange(autocorr_test_data.shape[0]),
                                                                        autocorr_test_data, p0=[1, -0.5, 1])

        if plotting or save_fig:

            if save_fig:
                plt.style.use('default')
            plt.plot(shift_array, (auto_corr-auto_corr[-1])/(1-auto_corr[-1]), c="y", label="Awake")
            plt.xlabel("Shift (#spikes)")
            plt.ylabel("Avg. Pearson correlation of likelihood vectors")
            plt.legend()
            # plt.xticks([-100, -75, -50, -25, 0, 25, 50, 75, 100], np.array([-100, -75, -50, -25, 0, 25, 50, 75, 100]) * 12)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "sim_ratio_autocorr_spikes.svg"), transparent="True")
                plt.close()
            else:
                print("HERE")
                plt.show()

        else:
            auto_corr_norm = (auto_corr-auto_corr[-1])/(1-auto_corr[-1])
            return auto_corr, auto_corr_norm, popt_exponential_awake[1]

        # nrem_test_data = auto_corr_nrem[int(auto_corr_nrem.shape[0] / 2)+1:]
        # rem_test_data = auto_corr_rem[int(auto_corr_rem.shape[0] / 2) + 1:]
        #
        # def exponential(x, a, k, b):
        #     return a * np.exp(x * k) + b
        #
        # popt_exponential_rem, pcov_exponential_rem = optimize.curve_fit(exponential, np.arange(rem_test_data.shape[0]),
        #                                                                 rem_test_data, p0=[1, -0.5, 1])
        # popt_exponential_nrem, pcov_exponential_nrem = optimize.curve_fit(exponential, np.arange(nrem_test_data.shape[0]),
        #                                                                 nrem_test_data, p0=[1, -0.5, 1])
        # if plotting or save_fig:
        #
        #     if save_fig:
        #         plt.style.use('default')
        #     # plot fits
        #     plt.text(3, 10, "k = " +str(np.round(popt_exponential_rem[1], 2)), c="red" )
        #     plt.scatter(np.arange(rem_test_data.shape[0]), rem_test_data, c="salmon", label="REM data")
        #     plt.plot((np.arange(rem_test_data.shape[0]))[1:], exponential((np.arange(rem_test_data.shape[0]))[1:],
        #                                                             a=popt_exponential_rem[0], k=popt_exponential_rem[1],
        #                                                             b=popt_exponential_rem[2]), c="red", label="REM fit")
        #     plt.text(0.05, 5, "k = " +str(np.round(popt_exponential_nrem[1], 2)), c="blue" )
        #     plt.scatter(np.arange(nrem_test_data.shape[0]), nrem_test_data, c="lightblue", label="NREM data")
        #     plt.plot((np.arange(nrem_test_data.shape[0]))[1:], exponential((np.arange(nrem_test_data.shape[0]))[1:],
        #                                                             a=popt_exponential_nrem[0], k=popt_exponential_nrem[1],
        #                                                             b=popt_exponential_nrem[2]), c="blue", label="NREM fit")
        #
        #     plt.legend(loc=2)
        #     plt.ylabel("Pearson R (z-scored)")
        #     plt.xlabel("nr. spikes")
        #     plt.ylim(-3, 18)
        #     if save_fig:
        #         plt.rcParams['svg.fonttype'] = 'none'
        #         plt.savefig("exponential_fit_spikes.svg", transparent="True")
        #         plt.close()
        #     else:
        #         plt.show()
        #
        # if bootstrapping:
        #
        #     # bootstrapping
        #     n_boots = 500
        #     n_samples_perc = 0.8
        #
        #     nrem_exp = []
        #     rem_exp = []
        #
        #     for boots_id in range(n_boots):
        #         per_ind = np.random.permutation(np.arange(rem_test_data.shape[0]))
        #         sel_ind = per_ind[:int(n_samples_perc*per_ind.shape[0])]
        #         # select subset
        #         x_rem = np.arange(nrem_test_data.shape[0])[sel_ind]
        #         x_nrem = np.arange(nrem_test_data.shape[0])[sel_ind]
        #         y_rem = rem_test_data[sel_ind]
        #         y_nrem = nrem_test_data[sel_ind]
        #         try:
        #             popt_exponential_rem, _ = optimize.curve_fit(exponential,x_rem, y_rem, p0=[1, -0.5, 1])
        #             popt_exponential_nrem, _ = optimize.curve_fit(exponential, x_nrem, y_nrem, p0=[1, -0.5, 1])
        #         except:
        #             continue
        #
        #         rem_exp.append(popt_exponential_rem[1])
        #         nrem_exp.append(popt_exponential_nrem[1])
        #
        #     if plotting:
        #         plt.hist(rem_exp, label="rem", color="red", bins=10, density=True)
        #         plt.xlabel("k from exp. function")
        #         plt.ylabel("density")
        #         plt.legend()
        #         plt.show()
        #         plt.hist(nrem_exp, label="nrem", color="blue", alpha=0.8, bins=10, density=True)
        #         # plt.xlim(-2,0.1)
        #         # plt.title("k from exponential fit (bootstrapped)\n"+"Ttest one-sided: p="+\
        #         #           str(ttest_ind(rem_exp, nrem_exp, alternative="greater")[1]))
        #         # plt.xscale("log")
        #         plt.show()
        #     else:
        #         return np.median(np.array(rem_exp)), np.median(np.array(nrem_exp))
        # else:
        #     return popt_exponential_rem[1], popt_exponential_nrem[1]

    def sparsity_and_distance_of_modes(self, plotting=True, use_spike_bins=False, n_spikes_per_bin=12, metric="cosine",
                                       n_segments=1, th=3, n_lim=30, q_lim=0.7):

        file_name = self.default_phmm

        with open(self.params.pre_proc_dir + "phmm/" + file_name + '.pkl', 'rb') as f:
            model_dic = pickle.load(f)
        # get means of model (lambdas) for decoding
        modes_lambdas = model_dic.means_

        # get spike rasters
        spike_raster, _ = self.get_spike_bin_raster()
        spike_raster = np.hstack(spike_raster)

        # compute compression factor
        n_spikes_per_bin = self.nr_spikes_per_time_bin()
        compression_factor = self.params.spikes_per_bin / n_spikes_per_bin

        # compute likelihoods in pre using
        likeli = decode_using_phmm_modes(mode_means=modes_lambdas, event_spike_rasters=[spike_raster],
                                         compression_factor=compression_factor)

        likeli = likeli[0]

        # NREM
        # --------------------------------------------------------------------------------------------------------------
        # Temporal interval to consider (in number of population vectors)
        t_tot = spike_raster.shape[1]
        t_int = np.floor(t_tot / 6).astype(int)

        modes_lambdas = modes_lambdas.T

        sparsity_reactivations = np.zeros((n_segments, modes_lambdas.shape[1]))
        Spars_Ratio = np.zeros((n_segments, modes_lambdas.shape[1]))  # Sparsity Comparison

        distance_modes_nrem = np.zeros((n_segments, int(modes_lambdas.shape[1]*(modes_lambdas.shape[1]-1)/2)))  # Correlation in modes and reactivations
        distance_reactivations = np.zeros((n_segments, int(modes_lambdas.shape[1]*(modes_lambdas.shape[1]-1)/2)))

        # Cycle over sleep segments
        for ii in range(n_segments):

            th_fr = th * 0.09

            Raster_subset = spike_raster[:, 0 + t_int * ii:0 + t_int * (ii + 1)]
            # Raster = zscore(Raster,axis=1)

            prob_nrem_subset = likeli[0 + t_int * ii:0 + t_int * (ii + 1)]

            # normalize likelihoods
            Nor = np.sum(prob_nrem_subset, axis=1)
            prob_nrem_subset = prob_nrem_subset / np.reshape(Nor, (t_int, 1))

            # find decoded mode
            BT = np.argmax(prob_nrem_subset, axis=1)
            # get maximum normalized likelihood
            VT = np.max(prob_nrem_subset, axis=1)
            # set modes that were decoded with a worse quality than q_lim to 1000 --> indicates that mode bin is not
            # considered for following computations
            BT[VT < q_lim] = 1000
            Best_Template = BT  # np.ones((BT.shape))*1000
            # Best_Template[0+7000*ii:0+7000*(ii+1)]=BT[0+7000*ii:0+7000*(ii+1)]

            React = np.zeros((Raster_subset.shape[0], prob_nrem_subset.shape[1]))
            Mode_Prob = np.zeros(prob_nrem_subset.shape[1])

            # go through all modes
            for te in range(modes_lambdas.shape[1]):
                # check how often each mode was reactivated --> compute probability of reactivation
                Mode_Prob[te] = len(np.where(Best_Template == te)[0]) / len(Best_Template)
                # if mode was reactivated more than n_lim times --> compute mean across all reactivations of this mode
                if (Raster_subset[:, Best_Template == te].shape[1] > n_lim):
                    React[:, te] = np.mean(Raster_subset[:, Best_Template == te], axis=1)

            React[np.isnan(React)] = 0
            React = React - th_fr
            React[React < 0] = 0

            Modes = modes_lambdas - th_fr
            Modes[Modes < 0] = np.nan
            Modes[np.isnan(Modes)] = 0

            # Compute correlation between mode and reactivation average
            # CC = np.corrcoef(React.T, Modes.T)
            # Simil_Mo[th, :] = np.diag(CC[:modes_lambdas.shape[1], modes_lambdas.shape[1]:])

            distance_reactivations[ii, :] = pdist(React.T, metric=metric)
            distance_modes_nrem[ii, :] = pdist(Modes.T, metric=metric)

            # Compute sparsity of vectors
            React_Sp = (np.nansum(React, axis=0) ** 2) / (np.nansum(React ** 2, axis=0)) / modes_lambdas.shape[0]
            Mode_Sp = (np.nansum(Modes, axis=0) ** 2) / (np.nansum(Modes ** 2, axis=0)) / modes_lambdas.shape[0]
            React_Sp[np.isnan(React_Sp)] = 0

            p1 = React_Sp
            p2 = Mode_Sp  # modes_lambdas

            Spars_Ratio[ii, :] = p1 / p2
            sparsity_reactivations[ii, :] = React_Sp# Ratio of sparsity React/Modes

        Spars_Ratio_NREM = Spars_Ratio

        distance_reactivations = distance_reactivations[~np.isnan(distance_reactivations)]
        sparsity_reactivations = sparsity_reactivations[sparsity_reactivations>0]

        # if use_spike_bins:
        #     modes_lambdas_const_n_spike = np.zeros((modes_lambdas.shape[0], modes_lambdas.shape[1]))
        #
        #     # COMPUTE POISSON 12-Vector Spike VECTOR PROBABILITES FROM MODES --> PRE MODES
        #     for mo in range(modes_lambdas.shape[1]):
        #
        #         l_i = modes_lambdas[:, mo]
        #         sp_t_all = np.zeros((0,))
        #         sp_i_all = np.zeros((0,))
        #         sp_m = np.zeros(l_i.shape[0])
        #         for nn in range(l_i.shape[0]):
        #             sp_t = np.cumsum(np.random.exponential(1 / l_i[nn], (10000, 1)))
        #             sp_m[nn] = np.max(sp_t)
        #             sp_t_all = np.concatenate((sp_t_all, sp_t))
        #             sp_i = np.ones((10000,)) * nn
        #             sp_i_all = np.concatenate((sp_i_all, sp_i))
        #             # Take the earlier last spike from any cell
        #         thr = np.min(sp_m)
        #         sp_i_all = sp_i_all[sp_t_all < thr]
        #         sp_t_all = sp_t_all[sp_t_all < thr]
        #
        #         # Rearrange spike in time
        #         aa = np.argsort(sp_t_all)
        #         sp_i_all = sp_i_all[aa]
        #
        #         # Build avarage spike occurrence
        #         n_samp = int(np.floor(sp_t_all.shape[0] / n_spikes_per_bin))
        #         raster_mod = np.zeros((l_i.shape[0], n_samp))
        #         for ss in range(n_samp):
        #             take_sp = sp_i_all[ss * n_spikes_per_bin:ss * n_spikes_per_bin + n_spikes_per_bin].astype(int)
        #             for sp in range(len(take_sp)):
        #                 raster_mod[take_sp[sp], ss] += 1
        #         modes_lambdas_const_n_spike[:, mo] = np.mean(raster_mod, axis=1)
        #
        #     modes_lambdas = modes_lambdas_const_n_spike
        #
        # sparsity = (np.sum(modes_lambdas,axis=0)**2)/\
        #                (np.sum(modes_lambdas**2,axis=0))/modes_lambdas.shape[0]
        #
        # distance_modes = pdist(modes_lambdas, metric=metric)


        return sparsity_reactivations, distance_reactivations

    # </editor-fold>

    # <editor-fold desc="Location decoding analysis">

    """#################################################################################################################
    #  location decoding analysis
    #################################################################################################################"""

    def decode_location_phmm(self, trial_to_decode, model_name=None, trials_to_use=None, save_fig=False):
        """
        decodes awake activity using pHMM modes from same experiment phase

        @param trial_to_decode: which trial to decode
        @type trial_to_decode: int
        @param model_name: name of file containing the pHMM file
        @type model_name: str
        @param plot_for_control: plots intermediate results if True
        @type plot_for_control: bool
        @param return_results: whether to return results or not
        @type return_results: bool
        @return: pre_prob (probabilities of PRE), post_prob (probabilities of ), event_times (in sec),
        swr_in_which_n_rem (assignement SWR - nrem phase)
        @rtype: list, list, list, numpy.array
        """

        print(" - DECODE LOCATION USING PHMM MODES ...\n")
        result_dir = "phmm_decoding"
        # get template file name from parameter file of session if not provided
        if model_name is None:
            model_name = self.session_params.default_pre_phmm_model

        if model_name is None:
            raise Exception("MODEL FILE NOT FOUND\n NOR IN SESSION PARAMETER FILE DEFINED")

        if trials_to_use is None:
            trials_to_use = self.default_trials

        print("   --> USING TRIALS: "+str(trials_to_use[0])+"-"+str(trials_to_use[-1])+"\n")

        # check that template files are from correct session
        # --------------------------------------------------------------------------------------------------------------
        sess_pre = model_name.split(sep="_")[0]+"_"+model_name.split(sep="_")[1]

        if not (sess_pre == self.session_name):
            raise Exception("TEMPLATE FILE AND CURRENT SESSION DO NOT MATCH!")

        # load goal locations
        goal_loc = np.loadtxt(self.params.pre_proc_dir+"goal_locations"+"/"+"mjc163R2R_0114_11.gcoords")

        # get spatial information from pHMM model
        means, _, _, _, _ = self.fit_spatial_gaussians_for_modes(file_name=model_name)

        # get location data and raster from trial
        raster = self.trial_raster_list[trial_to_decode]
        loc = self.trial_loc_list[trial_to_decode]

        model = self.load_poisson_hmm(file_name=model_name)

        model.implementation = "log"
        prob = model.predict_proba(raster.T)

        # decode activity
        # prob_poiss = decode_using_phmm_modes(mode_means=model.means_,
        #                                        event_spike_rasters=[raster],
        #                                        compression_factor=1)[0]
        #
        # prob_poiss_norm = prob_poiss / np.sum(prob_poiss, axis=1, keepdims=True)
        #
        # plt.imshow(prob_poiss_norm, interpolation='nearest', aspect='auto')
        # plt.show()
        # plt.imshow(prob, interpolation='nearest', aspect='auto')
        # plt.show()

        if save_fig:
            plt.style.use('default')
            err = np.zeros(prob.shape[0])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            prev_dec_location = None
            prev_location = None
            for i, (current_prob, current_loc) in enumerate(zip(prob, loc)):
                # compute decoded location using weighted average
                dec_location = np.average(means, weights=current_prob, axis=1)
                err[i] = np.linalg.norm(dec_location - current_loc)
                if i > 1:
                    # ax.plot([dec_location[0], prev_dec_location[0]], [dec_location[1], prev_dec_location[1]], color="mistyrose",
                    #         zorder=-1000)
                    ax.plot([current_loc[0], prev_location[0]], [current_loc[1], prev_location[1]], color="lightgray",
                            zorder=-1000)
                ax.scatter(dec_location[0], dec_location[1], color="red", label="Decoded locations")
                ax.scatter(current_loc[0], current_loc[1], color="lightgray", label="True locations")
                prev_dec_location = dec_location
                prev_location = current_loc
            for g_l in self.goal_locations:
                ax.scatter(g_l[0], g_l[1], color="black", label="Goals", marker="x")
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.title("Decoding using weighted means")
            plt.xlabel("X (cm)")
            plt.ylabel("Y (cm)")
            plt.ylim(10, 110)
            plt.xlim(30, 140)
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "decoded_locations_phmm.svg"), transparent="True")
            # plt.show()
            plt.close()

            plt.hist(err, density=True, color="indianred", bins=int(err.shape[0]/5))
            plt.title("Decoding error")
            plt.xlabel("Error (cm)")
            plt.xlim(-5,100)
            plt.ylabel("Density")
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "decoding_error_phmm.svg"), transparent="True")
            # plt.show()
        else:
            col_map_red = cm.Reds(np.linspace(0, 1, prob.shape[0]))
            close_to_goal=np.zeros(prob.shape[0])
            err = np.zeros(prob.shape[0])
            fig = plt.figure()
            ax = fig.add_subplot(111)

            for i, (current_prob, current_loc) in enumerate(zip(prob, loc)):
                # compute decoded location using weighted average
                dec_location = np.average(means, weights=current_prob, axis=1)
                err[i] = np.linalg.norm(dec_location - current_loc)
                ax.scatter(dec_location[0], dec_location[1], color="blue", label="DECODED")
                ax.scatter(current_loc[0], current_loc[1], color=col_map_red[i], label="TRUE")
                ax.plot([dec_location[0], current_loc[0]], [dec_location[1], current_loc[1]], color="gray",
                         zorder=-1000, label="ERRORS")
                for gl in goal_loc:
                    if np.linalg.norm(current_loc - gl) < 20:
                        close_to_goal[i] = 1
                        ax.scatter(current_loc[0], current_loc[1], facecolors='none', edgecolors="white", label="CLOSE TO GOAL LOC.")
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.title("DECODING USING WEIGHTED MODE MEANS")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()
            # err = moving_average(a = np.array(err), n=20)
            plt.plot(err, color="gray", label="ERROR")
            err = moving_average(a=np.array(err), n=20)
            plt.plot(err, color="lightcoral", label="ERROR SMOOTHED")
            plt.plot(close_to_goal*10, color="w", label="CLOSE TO GOAL LOC.")
            plt.ylabel("ERROR / cm")
            plt.xlabel("TIME BIN")
            plt.legend()
            plt.show()

            sequence = model.decode(raster.T, algorithm="map")[1]

            err = []
            close_to_goal=np.zeros(sequence.shape[0])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # compute error
            for i, (mode_act, current_loc) in enumerate(zip(sequence, loc)):
                ax.scatter(means[0,mode_act], means[1,mode_act], color="blue", label="DECODED")
                ax.scatter(current_loc[0], current_loc[1], color=col_map_red[i], label="TRUE")
                ax.plot([means[0,mode_act], current_loc[0]], [means[1,mode_act], current_loc[1]], color="gray",
                         zorder=-1000, label="ERROR")
                err.append(np.linalg.norm(means[:,mode_act]-current_loc))
                for gl in goal_loc:
                    if np.linalg.norm(current_loc - gl) < 20:
                        close_to_goal[i] = 1
                        ax.scatter(current_loc[0], current_loc[1], facecolors='none', edgecolors="white", label="CLOSE TO GOAL LOC.")

            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.title("DECODING USING MOST LIKELY MODE SEQUENCE")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()
            # err = moving_average(a = np.array(err), n=20)
            plt.plot(err, color="gray", label="ERROR")
            err = moving_average(a=np.array(err), n=20)
            plt.plot(err, color="lightcoral", label="ERROR SMOOTHED")
            plt.plot(close_to_goal*10, color="w", label="CLOSE TO GOAL LOC.")
            plt.legend()
            plt.ylabel("ERROR / cm")
            plt.xlabel("TIME BIN")
            plt.show()

            mean_err = []
            median_err = []

            for trial_to_decode in trials_to_use:
                raster = self.trial_raster_list[trial_to_decode]
                loc = self.trial_loc_list[trial_to_decode]
                prob = model.predict_proba(raster.T)
                err = []
                for i, (current_prob, current_loc) in enumerate(zip(prob, loc)):
                    # compute decoded location using weighted average
                    dec_location = np.average(means, weights=current_prob, axis=1)
                    err.append(np.linalg.norm(dec_location - current_loc))

                mean_err.append(np.mean(np.array(err)))
                median_err.append(np.median(np.array(err)))

            plt.plot(trials_to_use, mean_err, label="MEAN")
            plt.plot(trials_to_use, median_err, label="MEDIAN")
            plt.ylabel("ERROR / cm")
            plt.xlabel("TRIAL ID")
            plt.ylim(10,40)
            plt.legend()
            plt.show()

    def decode_location_phmm_cross_validated(self, model_name=None, save_fig=False, trials_for_test=1,
                                             plotting=False, min_error_all_sessions=0.094, max_error_all_sessions=64.28):
        """
        decodes awake activity using pHMM modes from same experiment phase. Uses all trials except for the last two ones
        for training and tests on the last two trials

        @param model_name: name of file containing the pHMM file
        @type model_name: str
        @return: pre_prob (probabilities of PRE), post_prob (probabilities of ), event_times (in sec),
        swr_in_which_n_rem (assignement SWR - nrem phase)
        @rtype: list, list, list, numpy.array
        """

        print(" - DECODE LOCATION USING PHMM MODES ...\n")
        result_dir = "phmm_decoding"
        # get template file name from parameter file of session if not provided
        if model_name is None:
            model_name = self.session_params.default_pre_phmm_model

        if model_name is None:
            raise Exception("MODEL FILE NOT FOUND\n NOR IN SESSION PARAMETER FILE DEFINED")

        # check that template files are from correct session
        # --------------------------------------------------------------------------------------------------------------
        sess_pre = model_name.split(sep="_")[0]+"_"+model_name.split(sep="_")[1]

        if not (sess_pre == self.session_name):
            raise Exception("TEMPLATE FILE AND CURRENT SESSION DO NOT MATCH!")

        # load goal locations
        goal_loc = np.loadtxt(self.params.pre_proc_dir+"goal_locations"+"/"+"mjc163R2R_0114_11.gcoords")

        # split into training and test trials
        nr_trials = len(self.trial_raster_list)
        train_trials = np.arange(0, nr_trials - trials_for_test)
        test_trials = np.arange(nr_trials-trials_for_test, nr_trials)

        # get spatial information from pHMM model
        means, _, _, _, _ = self.fit_spatial_gaussians_for_modes(file_name=model_name, trials_to_use=train_trials)

        # get location data and raster from test trial
        # check how many cells
        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))
        loc = np.empty((0, 2))
        for trial_id in test_trials:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            loc = np.vstack((loc, self.trial_loc_list[trial_id]))

        model = self.load_poisson_hmm(file_name=model_name)
        model.implementation = "log"
        prob = model.predict_proba(raster.T)

        # error
        errors = np.zeros(prob.shape[0])
        for i, (current_prob, current_loc) in enumerate(zip(prob, loc)):
            # compute decoded location using weighted average
            dec_location = np.average(means, weights=current_prob, axis=1)
            err = np.linalg.norm(dec_location - current_loc)
            errors[i] = err

        if save_fig:
            cmap_errors = matplotlib.cm.get_cmap('Reds_r')

            plt.style.use('default')
            err = np.zeros(prob.shape[0])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            prev_dec_location = None
            prev_location = None
            for i, (current_prob, current_loc) in enumerate(zip(prob, loc)):
                # compute decoded location using weighted average
                dec_location = np.average(means, weights=current_prob, axis=1)
                err[i] = np.linalg.norm(dec_location - current_loc)
                if i > 1:
                    ax.plot([current_loc[0], prev_location[0]], [current_loc[1], prev_location[1]], color="lightblue",
                            zorder=-1000)
                    # ax.plot([dec_location[0], prev_dec_location[0]], [dec_location[1], prev_dec_location[1]],
                    #         color="salmon", zorder=-1000)
                ax.scatter(dec_location[0], dec_location[1], color=cmap_errors((err[i]-min_error_all_sessions)/
                                                                               (max_error_all_sessions-
                                                                                min_error_all_sessions)),
                           label="Decoded locations")
                ax.scatter(current_loc[0], current_loc[1], color="lightblue", label="True locations")
                prev_dec_location = dec_location
                prev_location = current_loc
            for g_l in self.goal_locations:
                ax.scatter(g_l[0], g_l[1], color="black", label="Goals", s=20)
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.title("Decoding using weighted means")
            plt.xlabel("X (cm)")
            plt.ylabel("Y (cm)")
            plt.ylim(10, 110)
            plt.xlim(30, 140)
            plt.rcParams['svg.fonttype'] = 'none'
            plt.scatter(40, 20, color=cmap_errors(0))
            plt.scatter(43, 20, color=cmap_errors(0.5))
            plt.scatter(46, 20, color=cmap_errors(0.999))
            plt.savefig(os.path.join(save_path, "decoded_locations_phmm.svg"), transparent="True")
            # plt.show()
            plt.close()

            plt.hist(err, density=True, color="indianred", bins=int(err.shape[0]/5))
            plt.title("Decoding error")
            plt.xlabel("Error (cm)")
            plt.xlim(-5,100)
            plt.ylabel("Density")
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(os.path.join(save_path, "decoding_error_phmm.svg"), transparent="True")
            # plt.show()
        elif plotting:
            col_map_red = cm.Reds(np.linspace(0, 1, prob.shape[0]))
            close_to_goal=np.zeros(prob.shape[0])
            err = np.zeros(prob.shape[0])
            fig = plt.figure()
            ax = fig.add_subplot(111)

            for i, (current_prob, current_loc) in enumerate(zip(prob, loc)):
                # compute decoded location using weighted average
                dec_location = np.average(means, weights=current_prob, axis=1)
                err[i] = np.linalg.norm(dec_location - current_loc)
                ax.scatter(dec_location[0], dec_location[1], color="blue", label="DECODED")
                ax.scatter(current_loc[0], current_loc[1], color=col_map_red[i], label="TRUE")
                ax.plot([dec_location[0], current_loc[0]], [dec_location[1], current_loc[1]], color="gray",
                         zorder=-1000, label="ERRORS")
                for gl in goal_loc:
                    if np.linalg.norm(current_loc - gl) < 20:
                        close_to_goal[i] = 1
                        ax.scatter(current_loc[0], current_loc[1], facecolors='none', edgecolors="white", label="CLOSE TO GOAL LOC.")
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.title("DECODING USING WEIGHTED MODE MEANS")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()
            # err = moving_average(a = np.array(err), n=20)
            plt.plot(err, color="gray", label="ERROR")
            err = moving_average(a=np.array(err), n=20)
            plt.plot(err, color="lightcoral", label="ERROR SMOOTHED")
            plt.plot(close_to_goal*10, color="w", label="CLOSE TO GOAL LOC.")
            plt.ylabel("ERROR / cm")
            plt.xlabel("TIME BIN")
            plt.legend()
            plt.show()

            sequence = model.decode(raster.T, algorithm="map")[1]

            err = []
            close_to_goal=np.zeros(sequence.shape[0])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # compute error
            for i, (mode_act, current_loc) in enumerate(zip(sequence, loc)):
                ax.scatter(means[0,mode_act], means[1,mode_act], color="blue", label="DECODED")
                ax.scatter(current_loc[0], current_loc[1], color=col_map_red[i], label="TRUE")
                ax.plot([means[0,mode_act], current_loc[0]], [means[1,mode_act], current_loc[1]], color="gray",
                         zorder=-1000, label="ERROR")
                err.append(np.linalg.norm(means[:,mode_act]-current_loc))
                for gl in goal_loc:
                    if np.linalg.norm(current_loc - gl) < 20:
                        close_to_goal[i] = 1
                        ax.scatter(current_loc[0], current_loc[1], facecolors='none', edgecolors="white", label="CLOSE TO GOAL LOC.")

            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.title("DECODING USING MOST LIKELY MODE SEQUENCE")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()
            # err = moving_average(a = np.array(err), n=20)
            plt.plot(err, color="gray", label="ERROR")
            err = moving_average(a=np.array(err), n=20)
            plt.plot(err, color="lightcoral", label="ERROR SMOOTHED")
            plt.plot(close_to_goal*10, color="w", label="CLOSE TO GOAL LOC.")
            plt.legend()
            plt.ylabel("ERROR / cm")
            plt.xlabel("TIME BIN")
            plt.show()

            mean_err = []
            median_err = []

            for trial_to_decode in test_trials:
                raster = self.trial_raster_list[trial_to_decode]
                loc = self.trial_loc_list[trial_to_decode]
                prob = model.predict_proba(raster.T)
                err = []
                for i, (current_prob, current_loc) in enumerate(zip(prob, loc)):
                    # compute decoded location using weighted average
                    dec_location = np.average(means, weights=current_prob, axis=1)
                    err.append(np.linalg.norm(dec_location - current_loc))

                mean_err.append(np.mean(np.array(err)))
                median_err.append(np.median(np.array(err)))

            plt.scatter(test_trials, mean_err, label="MEAN", color="white")
            plt.scatter(test_trials, median_err, label="MEDIAN", color="red")
            plt.ylabel("ERROR / cm")
            plt.xlabel("TRIAL ID")
            # plt.ylim(10,40)
            plt.legend()
            plt.show()
        return errors

    def decode_location_ising(self, trial_to_decode, model_name=None, trials_to_use=None):
        """
        decodes awake activity using spatial bins from ising model from same experiment phase

        @param trial_to_decode: which trial to decode
        @type trial_to_decode: int
        @param model_name: name of file containing the pHMM file
        @type model_name: str
        @param plot_for_control: plots intermediate results if True
        @type plot_for_control: bool
        @param return_results: whether to return results or not
        @type return_results: bool
        @return: pre_prob (probabilities of PRE), post_prob (probabilities of ), event_times (in sec),
        swr_in_which_n_rem (assignement SWR - nrem phase)
        @rtype: list, list, list, numpy.array
        """

        print(" - DECODE LOCATION USING ISING ...\n")

        # get template file name from parameter file of session if not provided
        if model_name is None:
            model_name = self.params.default_pre_ising_model

        if model_name is None:
            raise Exception("MODEL FILE NOT FOUND\n NOR IN SESSION PARAMETER FILE DEFINED")

        if trials_to_use is None:
            trials_to_use = self.default_trials

        print("   --> USING TRIALS: "+str(trials_to_use[0])+"-"+str(trials_to_use[-1])+"\n")

        # check that template files are from correct session
        # --------------------------------------------------------------------------------------------------------------
        sess_pre = model_name.split(sep="_")[0]+"_"+model_name.split(sep="_")[1]

        if not (sess_pre == self.params.session_name):
            raise Exception("TEMPLATE FILE AND CURRENT SESSION DO NOT MATCH!")

            # load ising template
        with open(self.params.pre_proc_dir + 'awake_ising_maps/' + model_name + '.pkl',
                  'rb') as f:
            model_dic = pickle.load(f)

        # load goal locations
        goal_loc = np.loadtxt(self.params.pre_proc_dir+"goal_locations"+"/"+"mjc163R2R_0114_11.gcoords")

        # get location data and raster from trial
        raster = self.trial_raster_list[trial_to_decode]
        loc = self.trial_loc_list[trial_to_decode]

        # decode activity
        # get template map
        template_map = model_dic["res_map"]

        # need actual spatial bin position to do decoding
        bin_size_x = np.round((self.x_max - self.x_min)/template_map.shape[1], 0)
        bins_x = np.linspace(self.x_min+bin_size_x/2, self.x_max-bin_size_x/2, template_map.shape[1])
        # bins_x = np.repeat(bins_x[None, :], template_map.shape[2], axis=0)

        # bins_x = bins_x.reshape(-1, (template_map.shape[1] * template_map.shape[2]))

        bin_size_y = np.round((self.y_max - self.y_min)/template_map.shape[2], 0)
        bins_y = np.linspace(self.y_min+bin_size_y/2, self.y_max-bin_size_y/2, template_map.shape[2])
        # bins_y = np.repeat(bins_y[:, None], template_map.shape[1], axis=1)

        # bins_y = bins_y.reshape(-1, (template_map.shape[1] * template_map.shape[2]))

        prob = decode_using_ising_map(template_map=template_map,
                                              event_spike_rasters=[raster],
                                              compression_factor=10,
                                              cell_selection="all")[0]


        # prob_poiss_norm = prob_poiss / np.sum(prob_poiss, axis=1, keepdims=True)

        # plt.imshow(prob_poiss_norm, interpolation='nearest', aspect='auto')
        # plt.show()
        # plt.imshow(np.log(prob_poiss), interpolation='nearest', aspect='auto')
        # plt.show()

        col_map_red = cm.Reds(np.linspace(0, 1, prob.shape[0]))
        close_to_goal=np.zeros(prob.shape[0])
        err = np.zeros(prob.shape[0])
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i, (current_prob, current_loc) in enumerate(zip(prob, loc)):

            prob_map = current_prob.reshape(template_map.shape[1],template_map.shape[2])
            max_ind = np.unravel_index(prob_map.argmax(), prob_map.shape)
            # plt.imshow(prob_map.T, origin="lower")
            # plt.colorbar()
            # plt.show()
            # compute decoded location using weighted average
            dec_location_x = bins_x[max_ind[0]]
            dec_location_y = bins_y[max_ind[1]]
            dec_location = np.array([dec_location_x, dec_location_y])
            err[i] = np.linalg.norm(dec_location - current_loc)
            ax.scatter(dec_location[0], dec_location[1], color="blue", label="DECODED")
            ax.scatter(current_loc[0], current_loc[1], color=col_map_red[i], label="TRUE")
            ax.plot([dec_location[0], current_loc[0]], [dec_location[1], current_loc[1]], color="gray",
                     zorder=-1000, label="ERRORS")
            for gl in goal_loc:
                if np.linalg.norm(current_loc - gl) < 20:
                    close_to_goal[i] = 1
                    ax.scatter(current_loc[0], current_loc[1], facecolors='none', edgecolors="white", label="CLOSE TO GOAL LOC.")
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("BAYESIAN DECODING")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
        # err = moving_average(a = np.array(err), n=20)
        plt.plot(err, color="gray", label="ERROR")
        err = moving_average(a=np.array(err), n=20)
        plt.plot(err, color="lightcoral", label="ERROR SMOOTHED")
        plt.plot(close_to_goal*10, color="w", label="CLOSE TO GOAL LOC.")
        plt.ylabel("ERROR / cm")
        plt.xlabel("TIME BIN")
        plt.legend()
        plt.show()


        mean_err = []
        median_err = []

        for trial_to_decode in trials_to_use:
            raster = self.trial_raster_list[trial_to_decode]
            loc = self.trial_loc_list[trial_to_decode]
            prob = decode_using_ising_map(template_map=template_map,
                                          event_spike_rasters=[raster],
                                          compression_factor=10,
                                          cell_selection="all")[0]
            err = []
            for i, (current_prob, current_loc) in enumerate(zip(prob, loc)):
                # compute decoded location using weighted average
                prob_map = current_prob.reshape(template_map.shape[1], template_map.shape[2])
                max_ind = np.unravel_index(prob_map.argmax(), prob_map.shape)
                # plt.imshow(prob_map.T, origin="lower")
                # plt.colorbar()
                # plt.show()
                # compute decoded location using weighted average
                dec_location_x = bins_x[max_ind[0]]
                dec_location_y = bins_y[max_ind[1]]
                dec_location = np.array([dec_location_x, dec_location_y])
                err.append(np.linalg.norm(dec_location - current_loc))

            mean_err.append(np.mean(np.array(err)))
            median_err.append(np.median(np.array(err)))

        plt.plot(trials_to_use, mean_err, label="MEAN")
        plt.plot(trials_to_use, median_err, label="MEDIAN")
        plt.ylabel("ERROR / cm")
        plt.xlabel("TRIAL ID")
        plt.legend()
        plt.show()

    def decode_location_bayes(self, cell_subset=None, trials_train=None, trials_test=None, save_fig=False,
                              plotting=False):
        """
        Location decoding using Bayes

        :param trials_train: trials used to generate rate maps
        :type trials_train: iterable
        :param trials_test: trials used for testing (pop.vec. & location)
        :type trials_test: iterable
        :param save_fig: whether to save figure or not
        :type save_fig: bool
        :param plotting: whether to plot results or return results (error)
        :type plotting: bool
        :param cell_subset: subset of cells that is used for decoding
        :type cell_subset: array
        """
        print(" - DECODING LOCATION USING BAYESIAN ...\n")

        # if no trials are provided: train/test on default trials without cross-validation
        if trials_train is None:
            trials_train = self.default_trials
        if trials_test is None:
            trials_test = self.default_trials

        # get train data --> rate maps
        # --------------------------------------------------------------------------------------------------------------
        rate_maps = self.get_rate_maps(spatial_resolution=1, trials_to_use=trials_train)

        if cell_subset is not None:
            rate_maps = rate_maps[:, :, cell_subset]

        # flatten rate maps
        rate_maps_flat = np.reshape(rate_maps, (rate_maps.shape[0] * rate_maps.shape[1], rate_maps.shape[2]))

        # get test data --> population vectors and location
        # --------------------------------------------------------------------------------------------------------------
        raster = []
        loc = []
        for trial_id in trials_test:
            raster.append(self.trial_raster_list[trial_id])
            loc.append(self.trial_loc_list[trial_id])

        loc = np.vstack(loc)
        raster = np.hstack(raster)
        # need to convert spikes/bin into spikes/second
        raster = np.hstack(raster)/self.params.time_bin_size

        if cell_subset is not None:
            raster = raster[cell_subset, :]

        if plotting or save_fig:
            plt.style.use('default')
            fig = plt.figure()
            ax = fig.add_subplot(111)

        decoding_err = []
        for i, (pop_vec, current_loc) in enumerate(zip(raster.T, loc)):
            if np.count_nonzero(pop_vec) == 0:
                continue
            bl = bayes_likelihood(frm=rate_maps_flat.T, pop_vec=pop_vec, log_likeli=False)
            bl_area = np.reshape(bl, (rate_maps.shape[0], rate_maps.shape[1]))
            pred_bin = np.unravel_index(bl_area.argmax(), bl_area.shape)
            pred_x = pred_bin[0] + self.x_min
            pred_y = pred_bin[1] + self.y_min
            dec_location = np.array([pred_x, pred_y])
            decoding_err.append(np.sqrt((pred_x - current_loc[0]) ** 2 + (pred_y - current_loc[1]) ** 2))

            if plotting or save_fig:
                ax.scatter(dec_location[0], dec_location[1], color="red", label="Decoded locations")
                ax.scatter(current_loc[0], current_loc[1], color="lightgray", label="True locations")

        if plotting or save_fig:
            for g_l in self.goal_locations:
                ax.scatter(g_l[0], g_l[1], color="black", label="Goals", marker="x")
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.title("Bayesian decoding")
            plt.xlabel("X (cm)")
            plt.ylabel("Y (cm)")
            plt.ylim(10, 110)
            plt.xlim(30, 140)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "decoded_locations_bayes.svg"), transparent="True")
                plt.close()
            else:
                plt.show()

            decoding_err = np.array(decoding_err)
            plt.hist(decoding_err, density=True, color="indianred", bins=int(decoding_err.shape[0]/5))
            plt.title("Decoding error")
            plt.xlabel("Error (cm)")
            plt.ylabel("Density")
            plt.xlim(-5, 100)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "decoding_error_bayes.svg"), transparent="True")
            else:
                plt.show()

        if (not plotting) & (not save_fig):
            return decoding_err

    def decode_location_bayes_cross_validated(self, cell_subset=None, trials_for_test=1, min_error_all_sessions=0,
                                              max_error_all_sessions=100, plotting=False, save_fig=False,
                                              spatial_resolution_rate_map=1):
        """
        Location decoding using Bayes

        :param save_fig: whether to save figure or not
        :type save_fig: bool
        :param plotting: whether to plot results or return results (error)
        :type plotting: bool
        :param cell_subset: subset of cells that is used for decoding
        :type cell_subset: array
        """
        print(" - DECODING LOCATION USING BAYESIAN ...\n")

        # split into training and test trials
        nr_trials = len(self.trial_raster_list)
        train_trials = np.arange(0, nr_trials - trials_for_test)
        test_trials = np.arange(nr_trials-trials_for_test, nr_trials)

        # get train data --> rate maps
        # --------------------------------------------------------------------------------------------------------------
        rate_maps = self.get_rate_maps(spatial_resolution=spatial_resolution_rate_map, trials_to_use=train_trials)

        if cell_subset is not None:
            rate_maps = rate_maps[:, :, cell_subset]

        # flatten rate maps
        rate_maps_flat = np.reshape(rate_maps, (rate_maps.shape[0] * rate_maps.shape[1], rate_maps.shape[2]))

        # get test data --> population vectors and location
        # --------------------------------------------------------------------------------------------------------------
        raster = []
        loc = []
        for trial_id in test_trials:
            raster.append(self.trial_raster_list[trial_id])
            loc.append(self.trial_loc_list[trial_id])

        loc = np.vstack(loc)
        # need to convert spikes/bin into spikes/second
        raster = np.hstack(raster)/self.params.time_bin_size

        if cell_subset is not None:
            raster = raster[cell_subset, :]

        if plotting or save_fig:
            plt.style.use('default')
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cmap_errors = matplotlib.cm.get_cmap('Reds_r')

        # need to compute centers of spatial bins in cm to do decoding
        bin_center = spatial_resolution_rate_map/2
        x_coord = np.array([bin_center]*rate_maps.shape[0]) + \
                  np.cumsum(np.insert(np.array([spatial_resolution_rate_map]*(rate_maps.shape[0]-1)), 0,0))
        y_coord = np.array([bin_center]*rate_maps.shape[1]) + \
                  np.cumsum(np.insert(np.array([spatial_resolution_rate_map]*(rate_maps.shape[1]-1)), 0,0))

        decoding_err = np.zeros(loc.shape[0])
        prev_loc = None
        for i, (pop_vec, current_loc) in enumerate(zip(raster.T, loc)):
            if np.count_nonzero(pop_vec) == 0:
                continue
            bl = bayes_likelihood(frm=rate_maps_flat.T, pop_vec=pop_vec, log_likeli=False)
            bl_area = np.reshape(bl, (rate_maps.shape[0], rate_maps.shape[1]))
            pred_bin = np.unravel_index(bl_area.argmax(), bl_area.shape)
            pred_x = x_coord[pred_bin[0]] + self.x_min
            pred_y = y_coord[pred_bin[1]] + self.y_min
            # pred_x = pred_bin[0] + self.x_min
            # pred_y = pred_bin[1] + self.y_min
            dec_location = np.array([pred_x, pred_y])
            decoding_err[i] = np.sqrt((pred_x - current_loc[0]) ** 2 + (pred_y - current_loc[1]) ** 2)

            # plt.imshow(np.log(bl_area).T)
            # plt.scatter(np.meshgrid(x_coord, y_coord)[0], np.meshgrid(x_coord, y_coord)[1], color="black", s=0.1)
            # plt.scatter(x_coord[pred_bin[0]], y_coord[pred_bin[1]], color="red")
            # plt.scatter(current_loc[0]-self.x_min, current_loc[1]-self.y_min)
            # plt.show()

            if plotting or save_fig:
                ax.scatter(dec_location[0], dec_location[1], color=cmap_errors((decoding_err[i]-min_error_all_sessions)/
                                                                               (max_error_all_sessions-
                                                                                min_error_all_sessions)), label="Decoded locations")
                ax.scatter(current_loc[0], current_loc[1], color="lightblue", label="True locations")
                if i > 1:
                    ax.plot([current_loc[0], prev_loc[0]], [current_loc[1], prev_loc[1]], color="lightblue",
                            zorder=-1000)

            prev_loc = current_loc

        if plotting or save_fig:
            for g_l in self.goal_locations:
                ax.scatter(g_l[0], g_l[1], color="black", label="Goals", s=20)
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.title("Bayesian decoding")
            plt.xlabel("X (cm)")
            plt.ylabel("Y (cm)")
            plt.ylim(10, 110)
            plt.xlim(30, 140)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "decoded_locations_bayes.svg"), transparent="True")
                plt.close()
            else:
                plt.show()

            decoding_err = np.array(decoding_err)
            plt.hist(decoding_err, density=True, color="indianred", bins=int(decoding_err.shape[0]/5))
            plt.title("Decoding error")
            plt.xlabel("Error (cm)")
            plt.ylabel("Density")
            plt.xlim(-5, 100)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "decoding_error_bayes.svg"), transparent="True")
            else:
                plt.show()

        return decoding_err

    def compare_decoding_location_phmm_bayesian(self):

        # decoding using phmm
        # --------------------------------------------------------------------------------------------------------------
        trials_to_use = self.default_trials
        model = self.load_poisson_hmm(file_name=self.default_phmm)
                # get spatial information from pHMM model
        means, _, _, _, _ = self.fit_spatial_gaussians_for_modes(file_name=self.default_phmm, plot_awake_fit=False)

        mean_err = []
        median_err = []
        phmm_all_err = []

        for trial_to_decode in trials_to_use:
            raster = self.trial_raster_list[trial_to_decode]
            loc = self.trial_loc_list[trial_to_decode]
            prob = model.predict_proba(raster.T)
            err = []
            for i, (current_prob, current_loc) in enumerate(zip(prob, loc)):
                # compute decoded location using weighted average
                dec_location = np.average(means, weights=current_prob, axis=1)
                err.append(np.linalg.norm(dec_location - current_loc))
                phmm_all_err.append(np.linalg.norm(dec_location - current_loc))

            mean_err.append(np.mean(np.array(err)))
            median_err.append(np.median(np.array(err)))

        plt.plot(trials_to_use, mean_err, label="MEAN")
        plt.plot(trials_to_use, median_err, label="MEDIAN")
        plt.ylabel("ERROR / cm")
        plt.xlabel("TRIAL ID")
        # plt.ylim(10,40)
        plt.legend()
        plt.show()

        # plot cdf
        phmm_all_err = np.array(phmm_all_err)
        phmm_all_error_sorted = np.sort(phmm_all_err)
        p_phmm = 1. * np.arange(phmm_all_err.shape[0]) / (phmm_all_err.shape[0] - 1)
        plt.plot(phmm_all_error_sorted, p_phmm, color="#ffdba1", label="pHMM")


        # decoding using Bayesian
        # --------------------------------------------------------------------------------------------------------------
        bayes_all_err = []
        rate_maps = self.get_rate_maps(spatial_resolution=1)
        # flatten rate maps
        rate_maps_flat = np.reshape(rate_maps, (rate_maps.shape[0] * rate_maps.shape[1], rate_maps.shape[2]))
        for trial_to_decode in trials_to_use:
            raster = self.trial_raster_list[trial_to_decode]
            loc = self.trial_loc_list[trial_to_decode]
            for pop_vec, loc in zip(raster.T, loc):
                if np.count_nonzero(pop_vec) == 0:
                    continue
                bl = bayes_likelihood(frm=rate_maps_flat.T, pop_vec=pop_vec, log_likeli=False)
                bl_area = np.reshape(bl, (rate_maps.shape[0], rate_maps.shape[1]))
                pred_bin = np.unravel_index(bl_area.argmax(), bl_area.shape)
                pred_x = pred_bin[0] + self.x_min
                pred_y = pred_bin[1] + self.y_min

                # plt.scatter(pred_x, pred_y, color="red")
                # plt.scatter(loc[0], loc[1], color="gray")
                bayes_all_err.append(np.sqrt((pred_x - loc[0]) ** 2 + (pred_y - loc[1]) ** 2))

        bayes_all_err = np.array(bayes_all_err)
        bayes_all_error_sorted = np.sort(bayes_all_err)
        p_bayes = 1. * np.arange(bayes_all_err.shape[0]) / (bayes_all_err.shape[0] - 1)
        plt.plot(bayes_all_error_sorted, p_bayes, color="blue", label="bayes")
        plt.legend()
        plt.xlabel("Error (cm)")
        plt.ylabel("CDF")
        plt.show()

    def decode_location_during_learning(self, cells_to_use="stable", trial_window=3, restrict_to_goals=False,
                                        plot_for_control=False):

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        if cells_to_use == "stable":
            cell_ids = class_dic["stable_cell_ids"].flatten()
        if cells_to_use == "decreasing":
            cell_ids = class_dic["decrease_cell_ids"].flatten()

        test_trials = range(len(self.trial_loc_list)-trial_window, len(self.trial_loc_list))
        test_raster, test_loc, _ = self.get_raster_location_speed(trials_to_use=test_trials)
        test_raster = test_raster[cell_ids, :]

        if restrict_to_goals:
            radius = 20
            close_to_all_goals = np.zeros(test_loc.shape[0]).astype(bool)
            for g_l in self.goal_locations:
                dist_to_goal = np.sqrt((test_loc[:, 0] - g_l[0]) ** 2 + (test_loc[:, 1] - g_l[1]) ** 2)
                close_to_goal = dist_to_goal < radius
                close_to_all_goals = np.logical_or(close_to_all_goals, close_to_goal)
            if plot_for_control:
                plt.scatter(test_loc[:, 0], test_loc[:, 1])
                plt.scatter(test_loc[close_to_all_goals,0], test_loc[close_to_all_goals,1], color="r")
                plt.scatter(g_l[0], g_l[1], marker="x")
                plt.show()
            test_raster = test_raster[:,close_to_all_goals]
            test_loc = test_loc[close_to_all_goals,:]

        nr_windows = np.round((len(self.trial_loc_list)-trial_window)/trial_window).astype(int)

        col = plt.cm.get_cmap("jet")

        for window_id in range(nr_windows):
            train_trials = range(window_id*trial_window, (window_id+1)*trial_window)

            # get rate map --> need to add x_min, y_min of environment to have proper location info
            rate_maps = self.get_rate_maps(spatial_resolution=1, trials_to_use=train_trials)

            # flatten rate maps
            rate_maps_flat = np.reshape(rate_maps, (rate_maps.shape[0] * rate_maps.shape[1], rate_maps.shape[2]))

            rate_maps_flat = rate_maps_flat[:, cell_ids]

            error = []
            for pop_vec, loc in zip(test_raster.T, test_loc):
                if np.count_nonzero(pop_vec) == 0:
                    continue
                bl = bayes_likelihood(frm=rate_maps_flat.T, pop_vec=pop_vec, log_likeli=False)
                bl_area = np.reshape(bl, (rate_maps.shape[0], rate_maps.shape[1]))
                pred_bin = np.unravel_index(bl_area.argmax(), bl_area.shape)
                pred_x = pred_bin[0] + self.x_min
                pred_y = pred_bin[1] + self.y_min



                # plt.scatter(pred_x, pred_y, color="red")
                # plt.scatter(loc[0], loc[1], color="gray")
                error.append(np.sqrt((pred_x - loc[0]) ** 2 + (pred_y - loc[1]) ** 2))

            error_sorted = np.sort(error)
            p_error = 1. * np.arange(error_sorted.shape[0]) / (error_sorted.shape[0] - 1)
            plt.plot(error_sorted, p_error, label="WINDOW "+str(window_id), c=col(window_id/nr_windows))
            # plt.gca().set_xscale("log")
        plt.ylabel("CDF")
        plt.xlabel("Error (cm)")
        if restrict_to_goals:
            plt.title("Location decoding around goals during learning: " + cells_to_use + " cells")
        else:
            plt.title("Location decoding during learning: "+cells_to_use+" cells")
        plt.legend()
        plt.show()

    def decode_location_beginning_end_learning(self, cells_to_use="stable", trial_window=5, restrict_to_goals=False,
                                        plotting=False):

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        if cells_to_use == "stable":
            cell_ids = class_dic["stable_cell_ids"].flatten()
        if cells_to_use == "decreasing":
            cell_ids = class_dic["decrease_cell_ids"].flatten()

        test_trials = range(len(self.trial_loc_list)-trial_window, len(self.trial_loc_list))
        test_raster, test_loc, _ = self.get_raster_location_speed(trials_to_use=test_trials)
        test_raster = test_raster[cell_ids, :]

        train_trials = range(trial_window)

        # get rate map --> need to add x_min, y_min of environment to have proper location info
        rate_maps = self.get_rate_maps(spatial_resolution=1, trials_to_use=train_trials)

        # flatten rate maps
        rate_maps_flat = np.reshape(rate_maps, (rate_maps.shape[0] * rate_maps.shape[1], rate_maps.shape[2]))

        rate_maps_flat = rate_maps_flat[:, cell_ids]

        error = []
        for pop_vec, loc in zip(test_raster.T, test_loc):
            if np.count_nonzero(pop_vec) == 0:
                continue
            bl = bayes_likelihood(frm=rate_maps_flat.T, pop_vec=pop_vec, log_likeli=False)
            bl_area = np.reshape(bl, (rate_maps.shape[0], rate_maps.shape[1]))
            pred_bin = np.unravel_index(bl_area.argmax(), bl_area.shape)
            pred_x = pred_bin[0] + self.x_min
            pred_y = pred_bin[1] + self.y_min

            error.append(np.sqrt((pred_x - loc[0]) ** 2 + (pred_y - loc[1]) ** 2))

        error_sorted = np.sort(error)

        if plotting:
            p_error = 1. * np.arange(error_sorted.shape[0]) / (error_sorted.shape[0] - 1)
            # plt.gca().set_xscale("log")
            plt.ylabel("CDF")
            plt.xlabel("Error (cm)")
            plt.plot(error_sorted, p_error)

            plt.title("Location decoding during learning: "+cells_to_use+" cells")
            plt.show()
        else:
            return error_sorted

    def decode_location_end_of_learning(self, cells_to_use="stable", nr_of_trials=10, plotting=False):

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        if cells_to_use == "stable":
            cell_ids = class_dic["stable_cell_ids"].flatten()
        if cells_to_use == "decreasing":
            cell_ids = class_dic["decrease_cell_ids"].flatten()

        test_train_trials = range(len(self.trial_loc_list)-nr_of_trials, len(self.trial_loc_list))
        raster, loc, _ = self.get_raster_location_speed(trials_to_use=test_train_trials)
        raster = raster[cell_ids, :]

        # get rate map --> need to add x_min, y_min of environment to have proper location info
        rate_maps = self.get_rate_maps(spatial_resolution=1, trials_to_use=test_train_trials)
        occ_map = self.get_occ_map(spatial_resolution=1, trials_to_use=test_train_trials)

        # flatten rate maps
        rate_maps_flat = np.reshape(rate_maps, (rate_maps.shape[0] * rate_maps.shape[1], rate_maps.shape[2]))

        rate_maps_flat = rate_maps_flat[:, cell_ids]

        error = []
        pred_loc = []
        for pop_vec, loc_curr in zip(raster.T, loc):
            if np.count_nonzero(pop_vec) == 0:
                continue
            bl = bayes_likelihood(frm=rate_maps_flat.T, pop_vec=pop_vec, log_likeli=False)
            bl_area = np.reshape(bl, (rate_maps.shape[0], rate_maps.shape[1]))
            pred_bin = np.unravel_index(bl_area.argmax(), bl_area.shape)
            pred_x = pred_bin[0]
            pred_y = pred_bin[1]
            true_x = loc_curr[0] - self.x_min
            true_y = loc_curr[1] - self.y_min
            if plotting:
                plt.imshow(bl_area.T)
                plt.scatter(loc[:,0] - self.x_min, loc[:,1] - self.y_min, s=1, alpha=0.5)
                plt.scatter(true_x, true_y, c="r")
                plt.scatter(pred_bin[0],pred_bin[1], edgecolors="r", facecolor="none", s=50)
                a = plt.colorbar()
                a.set_label("Likelihood")
                plt.title(np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2))
                plt.show()
            pred_loc.append([pred_x, pred_y])
            error.append(np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2))

        error = np.array(error)
        error_sorted = np.sort(error)
        p_error = 1. * np.arange(error.shape[0]) / (error.shape[0] - 1)

        if plotting:
            plt.plot(error_sorted, p_error)
            plt.xlabel("Error (cm)")
            plt.ylabel("CDF")
            plt.show()

        pred_loc_arr = np.vstack(pred_loc)
        pos, times_decoded = np.unique(pred_loc_arr, axis=0, return_counts=True)

        times_decoded_prob = times_decoded / times_decoded.sum()

        max_prob = times_decoded_prob.max()
        times_decoded_prob_norm = times_decoded_prob / max_prob

        if plotting:
            fig, ax = plt.subplots()
            col_map = plt.cm.get_cmap("jet")
            ax.scatter(pos[:,0], pos[:,1], color=col_map(times_decoded_prob_norm))
            for g_l in self.goal_locations:
                ax.scatter(g_l[0]-self.x_min, g_l[1]-self.y_min, marker="x", color="w")
            a = fig.colorbar(cm.ScalarMappable(cmap=col_map), ticks=[0,1])
            a.ax.set_yticklabels(["0", "{:.2e}".format(max_prob)])
            a.set_label("Decoding probability: "+cells_to_use)
            plt.show()

        spatial_resolution = 1

        # get size of environment
        x_span = self.x_max - self.x_min
        y_span = self.y_max - self.y_min

        # width - length ratio of environment
        w_l_ratio = y_span / x_span

        nr_spatial_bins = int(np.round(x_span / spatial_resolution))

        centers_x = np.linspace(self.x_min, self.x_max + 0.1, nr_spatial_bins)
        centers_y = np.linspace(self.y_min, self.y_max + 0.1, int(round(nr_spatial_bins * w_l_ratio)))

        dx = centers_x[1] - centers_x[0]
        dy = centers_y[1] - centers_y[0]

        # split location data into x and y coordinates
        x_loc = loc[:, 0]
        y_loc = loc[:, 1]

        x_loc[x_loc > self.x_max] = self.x_min - 0.01
        y_loc[y_loc > self.y_max] = self.y_min - 0.01

        location_decoded_matrix = np.zeros((centers_x.shape[0], centers_y.shape[0]))

        for loc in pred_loc:
            xi = int(np.floor((loc[0]) / dx)) + 1
            yi = int(np.floor((loc[1]) / dy)) + 1
            if xi * yi > 0:
                location_decoded_matrix[xi, yi] += 1

        location_decoded_matrix_prob = location_decoded_matrix / np.sum(location_decoded_matrix.flatten())
        if plotting:
            plt.imshow(location_decoded_matrix_prob.T)
            a = plt.colorbar()
            a.set_label("Decoding probability")
            for g_l in self.goal_locations:
                plt.scatter((g_l[0]-self.x_min)/spatial_resolution, (g_l[1]-self.y_min)/spatial_resolution, marker="x", color="w")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("Decoding prob. "+cells_to_use+ " cells")
            plt.show()

        if not plotting:
            return error

    def decoding_error_stable_vs_decreasing(self, plotting=False, nr_of_trials=10, subset_range=[4,8,12,18],
                                            nr_subsets=10, cross_val=False):

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        cell_ids_stable = class_dic["stable_cell_ids"].flatten()
        cell_ids_dec = class_dic["decrease_cell_ids"].flatten()

        # use last n trials without cross-validation
        test_train_trials = np.arange(len(self.trial_loc_list)-nr_of_trials, len(self.trial_loc_list))
        if cross_val:
            test_train_trials_per = np.random.permutation(test_train_trials)
            trials_test = test_train_trials[:int(test_train_trials_per.shape[0]*0.5)]
            trials_train = test_train_trials[int(test_train_trials_per.shape[0] * 0.5):]
        else:
            trials_test = test_train_trials
            trials_train = test_train_trials

        decoding_error_stable = []
        decoding_error_dec = []

        for subset_size in subset_range:
            stable_subsets = []
            dec_subsets = []
            # generate subsets
            for n in range(nr_subsets):
                stable_subsets.append(np.random.choice(a=cell_ids_stable, size=subset_size, replace=False))
                dec_subsets.append(np.random.choice(a=cell_ids_dec, size=subset_size, replace=False))

            # compute in parallel for different subsets: first for stable cells
            with mp.Pool(nr_subsets) as p:
                multi_arg = partial(self.decode_location_bayes, trials_train=trials_train,
                                    trials_test=trials_test)
                decoding_error_stable_subset = p.map(multi_arg, stable_subsets)

            # compute in parallel for different subsets: decreasing cells
            with mp.Pool(nr_subsets) as p:
                multi_arg = partial(self.decode_location_bayes, trials_train=trials_train,
                                    trials_test=trials_test)
                decoding_error_dec_subset = p.map(multi_arg, dec_subsets)

            decoding_error_stable.append(decoding_error_stable_subset)
            decoding_error_dec.append(decoding_error_dec_subset)

        error_stable = [np.hstack(x) for x in decoding_error_stable]
        error_dec = [np.hstack(x) for x in decoding_error_dec]

        med_stable = [np.median(x) for x in error_stable]
        mad_stable = [median_abs_deviation(x) for x in error_stable]

        med_dec = [np.median(x) for x in error_dec]
        mad_dec = [median_abs_deviation(x) for x in error_dec]
        error_stable_all = np.hstack(error_stable)
        error_dec_all = np.hstack(error_dec)

        if plotting:
            p_stable_all = 1. * np.arange(error_stable_all.shape[0]) / (error_stable_all.shape[0] - 1)
            p_dec_all = 1. * np.arange(error_dec_all.shape[0]) / (error_dec_all.shape[0] - 1)
            plt.plot(np.sort(error_stable_all), p_stable_all, label="stable")
            plt.plot(np.sort(error_dec_all), p_dec_all, label="dec")
            plt.title("Error for all subsets\n p-val:" + str(mannwhitneyu(error_stable_all, error_dec_all)[1]))
            plt.xlabel("Error (cm)")
            plt.ylabel("cdf")
            plt.legend()
            plt.show()

            for i, (stable, dec) in enumerate(zip(error_stable, error_dec)):
                p_stable = 1. * np.arange(stable.shape[0]) / (stable.shape[0] - 1)
                p_dec = 1. * np.arange(dec.shape[0]) / (dec.shape[0] - 1)
                plt.plot(np.sort(stable), p_stable, label="stable")
                plt.plot(np.sort(dec), p_dec, label="dec")
                plt.title("#cells in subset: " + str(subset_range[i]) + "\n p-val:" + str(mannwhitneyu(stable, dec)[1]))
                plt.xlabel("Error (cm)")
                plt.ylabel("cdf")
                plt.legend()
                plt.show()

            plt.errorbar(x=np.array(subset_range)+0.1, y=med_stable, yerr=mad_stable, label="stable")
            plt.errorbar(x=subset_range, y=med_dec, yerr=mad_dec, label="dec")
            plt.legend()
            plt.show()
        else:
            return error_stable, error_dec


    # </editor-fold>

    # <editor-fold desc="SVM analysis">

    """#################################################################################################################
    #  SVM analysis
    #################################################################################################################"""

    def distinguishing_goals(self, radius=10, plot_for_control=False, plotting=False):

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        cell_ids_stable = class_dic["stable_cell_ids"].flatten()
        cell_ids_decrease = class_dic["decrease_cell_ids"].flatten()
        cell_ids_increase = class_dic["increase_cell_ids"].flatten()

        # get rasters
        raster, loc, speed = self.get_raster_location_speed()

        # if cells_to_use == "random_subset":
        #     cell_ids = random.sample(range(raster.shape[0]), 20)

        all_goals = []
        # select population vectors around goals
        for i, g_l in enumerate(self.goal_locations):
            close_to_goal = []
            for pv, location_pv in zip(raster.T, loc):
                if norm(location_pv - g_l) < radius:
                    close_to_goal.append(pv)
                    if plot_for_control:
                        plt.scatter(location_pv[0], location_pv[1], color="gray")
            all_goals.append(close_to_goal)
            if plot_for_control:
                plt.scatter(g_l[0], g_l[1], label="Goal "+str(i))
        if plot_for_control:
            plt.legend()
            plt.show()

        all_goal_ids = [0, 1, 2, 3]
        multi_fits = []

        for fits in range(30):
            predictability = np.zeros((4, 4, 4))
            predictability[:] = np.nan

            cell_ids_stable_subset = np.random.choice(a=cell_ids_stable, size=10, replace=False)
            cell_ids_increase_subset = np.random.choice(a=cell_ids_increase, size=10, replace=False)
            cell_ids_decrease_subset = np.random.choice(a=cell_ids_decrease, size=10, replace=False)

            for pair in itertools.combinations(all_goal_ids, r=2):

                # compare two goals using SVM
                goal_1 = np.array(all_goals[pair[0]])
                goal_2 = np.array(all_goals[pair[1]])

                all_data = np.vstack((goal_1, goal_2))

                labels = np.zeros(all_data.shape[0])
                labels[:goal_1.shape[0]] = 1

                # permute data

                per_ind = np.random.permutation(np.arange(all_data.shape[0]))
                X_orig = all_data[per_ind,:]
                y = labels[per_ind]

                for i, cell_ids in enumerate([cell_ids_stable_subset, cell_ids_decrease_subset,
                                              cell_ids_increase_subset, None]):

                    if cell_ids is not None:
                        X = X_orig[:, cell_ids]
                    else:
                        X = X_orig

                    train_per = 0.7
                    X_train = X[:int(train_per*X.shape[0]),:]
                    X_test = X[int(train_per * X.shape[0]):, :]
                    y_train = y[:int(train_per*X.shape[0])]
                    y_test = y[int(train_per * X.shape[0]):]

                    clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto', kernel="linear"))
                    clf.fit(X_train, y_train)
                    predictability[i, pair[0], pair[1]] = clf.score(X_test, y_test)
            multi_fits.append(predictability)

        multi_fits = np.array(multi_fits)
        stable = np.nanmean(multi_fits[:, 0, :, :], axis=0)
        dec = np.nanmean(multi_fits[:, 1, :, :], axis=0)
        inc = np.nanmean(multi_fits[:, 2, :, :], axis=0)
        all = np.nanmean(multi_fits[:, 3, :, :], axis=0)

        if plotting:
            plt.imshow(stable, vmin=0, vmax=1)
            plt.xticks([0, 1, 2, 3], ["Goal 0", "Goal 1", "Goal 2", "Goal 3"])
            plt.yticks([0, 1, 2, 3], ["Goal 0", "Goal 1", "Goal 2", "Goal 3"])
            a = plt.colorbar()
            a.set_label("Mean accuracy SVM")
            plt.title("stable")
            plt.show()
            plt.imshow(dec, vmin=0, vmax=1)
            plt.xticks([0, 1, 2, 3], ["Goal 0", "Goal 1", "Goal 2", "Goal 3"])
            plt.yticks([0, 1, 2, 3], ["Goal 0", "Goal 1", "Goal 2", "Goal 3"])
            a = plt.colorbar()
            a.set_label("Mean accuracy SVM")
            plt.title("decreasing")
            plt.show()
            plt.imshow(inc, vmin=0, vmax=1)
            plt.xticks([0, 1, 2, 3], ["Goal 0", "Goal 1", "Goal 2", "Goal 3"])
            plt.yticks([0, 1, 2, 3], ["Goal 0", "Goal 1", "Goal 2", "Goal 3"])
            a = plt.colorbar()
            a.set_label("Mean accuracy SVM")
            plt.title("increasing")
            plt.show()

            plt.imshow(all, vmin=0, vmax=1)
            plt.xticks([0, 1, 2, 3], ["Goal 0", "Goal 1", "Goal 2", "Goal 3"])
            plt.yticks([0, 1, 2, 3], ["Goal 0", "Goal 1", "Goal 2", "Goal 3"])
            a = plt.colorbar()
            a.set_label("Mean accuracy SVM")
            plt.title("all cells")
            plt.show()
        else:
            return stable, dec, inc, all

    def identify_single_goal(self, radius=10, plot_for_control=False, plotting=True):
        """
        identifies single goals using PV and multi-class SVM

        @param radius:
        @param plot_for_control:
        @param plotting:
        @return:
        """
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        cell_ids_stable = class_dic["stable_cell_ids"].flatten()
        cell_ids_decrease = class_dic["decrease_cell_ids"].flatten()
        cell_ids_increase = class_dic["increase_cell_ids"].flatten()

        # get rasters
        raster, loc, speed = self.get_raster_location_speed()

        # if cells_to_use == "random_subset":
        #     cell_ids = random.sample(range(raster.shape[0]), 20)

        all_goals = []
        # select population vectors around goals
        for i, g_l in enumerate(self.goal_locations):
            close_to_goal = []
            for pv, location_pv in zip(raster.T, loc):
                if norm(location_pv - g_l) < radius:
                    close_to_goal.append(pv)
                    if plot_for_control:
                        plt.scatter(location_pv[0], location_pv[1], color="gray")
            all_goals.append(close_to_goal)
            if plot_for_control:
                plt.scatter(g_l[0], g_l[1], label="Goal "+str(i))
        if plot_for_control:
            plt.legend()
            plt.show()

        data_len = [len(x) for x in all_goals]
        all_data = np.vstack(all_goals)
        labels = np.zeros(all_data.shape[0])
        start = 0
        for label, l in enumerate(data_len):
            labels[start:start+l] = label
            start = start+l

        nr_fits = 30
        mean_accuracy = np.zeros((nr_fits, 4))
        for fit_id, fits in enumerate(range(nr_fits)):


            per_ind = np.random.permutation(np.arange(all_data.shape[0]))
            X_orig = all_data[per_ind, :]
            y = labels[per_ind]

            for cell_sel_id, cell_ids in enumerate([cell_ids_stable, cell_ids_decrease, cell_ids_increase, None]):
                if cell_ids is None:
                    X = X_orig
                else:
                    X = X_orig[:, cell_ids]

                train_per = 0.7
                X_train = X[:int(train_per * X.shape[0]), :]
                X_test = X[int(train_per * X.shape[0]):, :]
                y_train = y[:int(train_per * X.shape[0])]
                y_test = y[int(train_per * X.shape[0]):]

                clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto', kernel="linear"))
                clf.fit(X_train, y_train)

                mean_accuracy[fit_id, cell_sel_id] = clf.score(X_test, y_test)

        mean_acc_stable = mean_accuracy[:,0]
        mean_acc_dec = mean_accuracy[:, 1]
        mean_acc_inc = mean_accuracy[:, 2]
        mean_acc_all = mean_accuracy[:, 3]

        if plotting:
            c="white"
            res = [mean_acc_stable, mean_acc_dec, mean_acc_inc, mean_acc_all]
            bplot = plt.boxplot(res, positions=[1,2,3,4], patch_artist=True,
                                labels=["Stable", "Dec", "Inc", "All"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c),showfliers=False)
            plt.ylabel("Mean accuracy SVM - Multi-class")
            plt.show()

        else:
            return mean_acc_stable, mean_acc_dec, mean_acc_inc, mean_acc_all

    def identify_single_goal_subsets(self, radius=10, plot_for_control=False, plotting=True, m_subset=5,
                                     nr_splits=5, nr_subsets=10):
        """
        identifies single goals using PV and multi-class SVM

        @param radius:
        @param plot_for_control:
        @param plotting:
        @return:
        """
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        cell_ids_stable = class_dic["stable_cell_ids"].flatten()
        cell_ids_decrease = class_dic["decrease_cell_ids"].flatten()
        cell_ids_increase = class_dic["increase_cell_ids"].flatten()

        # get rasters
        raster, loc, speed = self.get_raster_location_speed()

        all_goals = []
        # select population vectors around goals
        for i, g_l in enumerate(self.goal_locations):
            close_to_goal = []
            for pv, location_pv in zip(raster.T, loc):
                if norm(location_pv - g_l) < radius:
                    close_to_goal.append(pv)
                    if plot_for_control:
                        plt.scatter(location_pv[0], location_pv[1], color="gray")
            all_goals.append(close_to_goal)
            if plot_for_control:
                plt.scatter(g_l[0], g_l[1], label="Goal "+str(i))
        if plot_for_control:
            plt.legend()
            plt.show()

        data_len = [len(x) for x in all_goals]
        all_data = np.vstack(all_goals)
        labels = np.zeros(all_data.shape[0])
        start = 0
        for label, l in enumerate(data_len):
            labels[start:start+l] = label
            start = start+l

        mean_accuracy_stable = []
        mean_accuracy_dec = []
        mean_accuracy_inc = []

        for i in range(nr_splits):

            per_ind = np.random.permutation(np.arange(all_data.shape[0]))
            X_orig = all_data[per_ind, :]
            y = labels[per_ind]

            for cell_sel_id, cell_ids in enumerate([cell_ids_stable, cell_ids_decrease, cell_ids_increase, None]):
                if cell_ids is None:
                    X = X_orig
                else:
                    X = X_orig[:, cell_ids]

                train_per = 0.7
                X_train = X[:int(train_per * X.shape[0]), :]
                X_test = X[int(train_per * X.shape[0]):, :]
                y_train = y[:int(train_per * X.shape[0])]
                y_test = y[int(train_per * X.shape[0]):]

                res = MlMethodsOnePopulation().parallelize_svm(X_train=X_train, X_test=X_test, y_train=y_train,
                                                                y_test=y_test, m_subset=m_subset, nr_subsets=nr_subsets)

                if cell_sel_id == 0:
                    mean_accuracy_stable.append(res)
                elif cell_sel_id == 1:
                    mean_accuracy_dec.append(res)
                elif cell_sel_id == 2:
                    mean_accuracy_inc.append(res)

        mean_acc_stable = np.vstack(mean_accuracy_stable).flatten()
        mean_acc_dec = np.vstack(mean_accuracy_dec).flatten()
        mean_acc_inc = np.vstack(mean_accuracy_inc).flatten()

        if plotting:
            c="white"
            res = [mean_acc_stable, mean_acc_dec, mean_acc_inc]
            bplot = plt.boxplot(res, positions=[1,2,3], patch_artist=True,
                                labels=["Stable", "Dec", "Inc"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c),showfliers=False)
            plt.ylabel("Mean accuracy SVM - Multi-class")
            plt.show()
        else:
            return mean_acc_stable, mean_acc_dec, mean_acc_inc

    def identify_single_goal_multiple_subsets(self, radius=10, plotting=True,
                                              subset_range=[4, 8, 12, 18], nr_splits=10, nr_subsets=10):

        stable_mean = []
        stable_std = []
        dec_mean = []
        dec_std = []
        inc_mean = []
        inc_std = []
        stable = []
        dec = []
        inc = []

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        cell_ids_stable = class_dic["stable_cell_ids"].flatten()
        cell_ids_decrease = class_dic["decrease_cell_ids"].flatten()
        cell_ids_increase = class_dic["increase_cell_ids"].flatten()

        raster = self.get_raster()
        # compute mean
        mean_fir = np.mean(raster, axis=1)/self.params.time_bin_size

        mean_fir_stable = mean_fir[cell_ids_stable]
        mean_fir_stable = mean_fir_stable[mean_fir_stable>1]
        print(mean_fir_stable.shape[0])

        for m_subset in subset_range:
            mean_acc_stable, mean_acc_dec, mean_acc_inc = self.identify_single_goal_subsets(plotting=False,
                                                                                            m_subset=m_subset,
                                                                                            radius=radius,
                                                                                            nr_splits=nr_splits,
                                                                                            nr_subsets=nr_subsets)
            stable_mean.append(np.mean(mean_acc_stable))
            stable_std.append(np.std(mean_acc_stable))
            dec_mean.append(np.mean(mean_acc_dec))
            dec_std.append(np.std(mean_acc_dec))
            inc_mean.append(np.mean(mean_acc_inc))
            inc_std.append(np.std(mean_acc_inc))
            stable.append(mean_acc_stable)
            dec.append(mean_acc_dec)
            inc.append(mean_acc_inc)

        if plotting:
            plt.errorbar(x=subset_range, y=stable_mean, yerr=stable_std, label="stable")
            plt.errorbar(x=subset_range, y=inc_mean, yerr=inc_std, label="inc")
            plt.errorbar(x=subset_range, y=dec_mean, yerr=dec_std, label="dec")
            plt.ylabel("Mean accuracy - multiclass SVM (mean,std)")
            plt.xlabel("#cells")
            plt.legend()
            plt.show()
        else:
            return stable, inc, dec

    def detect_goal_related_activity_using_subsets(self, radius=15, plot_for_control=False, plotting=True, m_subset=3,
                                     nr_splits=30):
        """
        identifies if animal is around any goal using SVM

        @param radius:
        @param plot_for_control:
        @param plotting:
        @return:
        """
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        cell_ids_stable = class_dic["stable_cell_ids"].flatten()
        cell_ids_decrease = class_dic["decrease_cell_ids"].flatten()
        cell_ids_increase = class_dic["increase_cell_ids"].flatten()

        # get rasters
        raster, loc, speed = self.get_raster_location_speed()

        close_to_goal = []
        away_from_goal = []
        away_from_goal_loc = []
        close_to_goal_loc = []
        # select population vectors around goals
        for pv, location_pv in zip(raster.T, loc):
            close_to_goal_flag = False
            for g_l in self.goal_locations:
                if norm(location_pv - g_l) < radius:
                    close_to_goal_flag = True
            if close_to_goal_flag:
                close_to_goal.append(pv)
                close_to_goal_loc.append(location_pv)
            else:
                away_from_goal.append(pv)
                away_from_goal_loc.append(location_pv)

        if plot_for_control:
            close_to_goal_loc = np.array(close_to_goal_loc)
            plt.scatter(close_to_goal_loc[:,0],close_to_goal_loc[:,1], color="b", s=1)
            away_from_goal_loc = np.array(away_from_goal_loc)
            plt.scatter(away_from_goal_loc[:,0],away_from_goal_loc[:,1], color="r", s=1)
            plt.show()

        y = np.zeros(len(close_to_goal)+len(away_from_goal))
        y[:len(close_to_goal)] = 1
        X = np.vstack((close_to_goal, away_from_goal))

        per_ind = np.random.permutation(np.arange(X.shape[0]))
        X = X[per_ind, :]
        y = y[per_ind]

        mean_accuracy_stable = []
        mean_accuracy_dec = []
        mean_accuracy_inc = []

        sss = StratifiedShuffleSplit(n_splits=nr_splits, test_size=0.3, random_state=0)

        for train_index, test_index in sss.split(X, y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for cell_sel_id, cell_ids in enumerate([cell_ids_stable, cell_ids_decrease, cell_ids_increase]):
                res = None

                # only use data from selected cells
                X_train_sel = X_train[:, cell_ids]
                X_test_sel = X_test[:, cell_ids]

                res = MlMethodsOnePopulation().parallelize_svm(X_train=X_train_sel, X_test=X_test_sel, y_train=y_train,
                                                                y_test=y_test, m_subset=m_subset)

                if cell_sel_id == 0:
                    mean_accuracy_stable.append(res)
                elif cell_sel_id == 1:
                    mean_accuracy_dec.append(res)
                elif cell_sel_id == 2:
                    mean_accuracy_inc.append(res)

        mean_acc_stable = np.vstack(mean_accuracy_stable).flatten()
        mean_acc_dec = np.vstack(mean_accuracy_dec).flatten()
        mean_acc_inc = np.vstack(mean_accuracy_inc).flatten()

        if plotting:
            c="white"
            res = [mean_acc_stable, mean_acc_dec, mean_acc_inc]
            bplot = plt.boxplot(res, positions=[1,2,3], patch_artist=True,
                                labels=["Stable", "Dec", "Inc"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c),showfliers=False)
            plt.ylabel("Mean accuracy SVM - Multi-class")
            plt.show()
        else:
            return mean_acc_stable, mean_acc_dec, mean_acc_inc

    def detect_goal_related_activity_using_multiple_subsets(self, radius=15, plot_for_control=False, plotting=True,
                                              subset_range=[4, 8, 12, 18], nr_splits=10):

        stable_mean = []
        stable_std = []
        dec_mean = []
        dec_std = []
        inc_mean = []
        inc_std = []
        stable = []
        dec = []
        inc = []

        for m_subset in subset_range:
            mean_acc_stable, mean_acc_dec, mean_acc_inc = self.detect_goal_related_activity_using_subsets(plotting=False,
                                                                                            m_subset=m_subset,
                                                                                            radius=radius,
                                                                                            nr_splits=nr_splits)
            stable_mean.append(np.mean(mean_acc_stable))
            stable_std.append(np.std(mean_acc_stable))
            dec_mean.append(np.mean(mean_acc_dec))
            dec_std.append(np.std(mean_acc_dec))
            inc_mean.append(np.mean(mean_acc_inc))
            inc_std.append(np.std(mean_acc_inc))
            stable.append(mean_acc_stable)
            dec.append(mean_acc_dec)
            inc.append(mean_acc_inc)

        if plotting:
            plt.errorbar(x=subset_range, y=stable_mean, yerr=stable_std, label="stable")
            plt.errorbar(x=subset_range, y=inc_mean, yerr=inc_std, label="inc")
            plt.errorbar(x=subset_range, y=dec_mean, yerr=dec_std, label="dec")
            plt.ylabel("Mean accuracy - multiclass SVM (mean,std)")
            plt.xlabel("#cells")
            plt.legend()
            plt.show()
        else:
            return stable, inc, dec


    # </editor-fold>

    # <editor-fold desc="Others">
    """#################################################################################################################
    #  others
    #################################################################################################################"""

    def spatial_information_vs_firing_rate(self, spatial_resolution=5):
        nr_trials = self.get_nr_of_trials()
        # get rate maps
        rate_maps = self.get_rate_maps(spatial_resolution=spatial_resolution, trials_to_use=range(nr_trials))
        occ_map = self.get_occ_map(spatial_resolution=spatial_resolution, trials_to_use=range(nr_trials))
        raster = self.get_raster(trials_to_use="all")

        mean_firing_rate = np.mean(raster, axis=1)/self.params.time_bin_size
        max_firing_rate = np.max(raster, axis=1) / self.params.time_bin_size

        sparsity, info_per_sec, info_per_spike = compute_spatial_information(rate_maps=rate_maps, occ_map=occ_map)

        plt.scatter(mean_firing_rate, sparsity)
        plt.xlabel("mean firing rate")
        plt.ylabel("sparsity")
        plt.show()

        plt.scatter(mean_firing_rate, info_per_sec)
        plt.xlabel("mean firing rate")
        plt.ylabel("info_per_sec")
        plt.show()

        plt.scatter(mean_firing_rate, info_per_spike)
        plt.xlabel("mean firing rate")
        plt.ylabel("info_per_spike")
        plt.show()
        c_id = (mean_firing_rate- np.min(mean_firing_rate)) / np.max(mean_firing_rate- np.min(mean_firing_rate))
        plt.scatter(sparsity, info_per_sec, c = c_id)
        plt.xlabel("sparsity")
        plt.ylabel("info per sec")
        plt.set_cmap("Reds")
        a = plt.colorbar()
        a.set_label("mean firing (norm)")
        plt.show()

        plt.scatter(sparsity, info_per_spike, c=c_id)
        plt.xlabel("sparsity")
        plt.ylabel("info per spike")
        plt.set_cmap("Reds")
        a = plt.colorbar()
        a.set_label("mean firing (norm)")
        plt.show()

        c_id = (max_firing_rate- np.min(max_firing_rate)) / np.max(max_firing_rate- np.min(max_firing_rate))
        plt.scatter(sparsity, info_per_sec, c = c_id)
        plt.xlabel("sparsity")
        plt.ylabel("info per sec")
        plt.set_cmap("Reds")
        a = plt.colorbar()
        a.set_label("max firing (norm)")
        plt.show()

        plt.scatter(sparsity, info_per_spike, c=c_id)
        plt.xlabel("sparsity")
        plt.ylabel("info per spike")
        plt.set_cmap("Reds")
        a = plt.colorbar()
        a.set_label("max firing (norm)")
        plt.show()

    def occupancy_around_goals(self, radius=5, plot_for_control=False, plotting=False,
                               save_fig=False, first_half=False):
        if first_half:
            nr_trials_to_use = np.round(self.nr_trials/2).astype(int)
            occ_map = self.get_occ_map(spatial_resolution=1, trials_to_use=np.arange(nr_trials_to_use))
        else:
            occ_map = self.get_occ_map(spatial_resolution=1)


        # only for plotting
        occ_map_all_goals = np.zeros((occ_map.shape[0], occ_map.shape[1]))
        occ_map_all_goals[:] = np.nan
        occ_map_no_goals = np.copy(occ_map)

        for goal in self.goal_locations:
            occ_map_gc = np.zeros((occ_map.shape[0], occ_map.shape[1]))
            occ_map_gc[:] = np.nan
            y = np.arange(0, occ_map.shape[0])
            x = np.arange(0, occ_map.shape[1])
            cy = goal[0] - self.x_min
            cx = goal[1] - self.y_min
            # define mask to mark area around goal
            mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < radius ** 2
            # copy only masked z-values
            occ_map_gc[mask] = occ_map[mask]
            # copy also to map for all goals
            occ_map_all_goals[mask] = occ_map[mask]

            # copy only masked z-values
            occ_map_no_goals[mask] = np.nan

        if plot_for_control:
            plt.imshow(occ_map_all_goals)
            plt.colorbar()
            plt.show()

            plt.imshow(occ_map_no_goals)
            plt.colorbar()
            plt.show()

        occ_map_all_goals = np.nan_to_num(occ_map_all_goals)
        occ_map_no_goals = np.nan_to_num(occ_map_no_goals)
        # compute occupancy per cm2 around goals
        occ_around_goals_per_cm2 = np.sum(occ_map_all_goals.flatten())/(4*np.pi*radius**2)
        # compute occupancy per cm2 away from goals --> use tracking data to compute cheeseboard size
        # (or use radius of 60cm)
        min_diam = np.min([self.x_max-self.x_min, self.y_max-self.y_min])
        area_covered = np.pi*(min_diam/2)**2
        occ_wo_goals_per_cm2 = np.sum(occ_map_no_goals.flatten())/(area_covered - 4*np.pi*radius**2)

        if save_fig or plotting:
            plt.style.use('default')
            occ_map[occ_map==0] = np.nan
            b = plt.imshow(occ_map.T, cmap="Oranges")
            for goal in self.goal_locations:
                cy = goal[0] - self.x_min
                cx = goal[1] - self.y_min
                plt.scatter(cy,cx, c="black", label="Goal locations")
            a = plt.colorbar(mappable=b)
            a.set_label("Occupancy / s")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.axis('off')
            # circle1 = plt.Circle((45,40), 60)
            # plt.gca().add_patch(circle1)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(os.path.join(save_path, "occupancy_post.svg"), transparent="True")
                plt.close()
            else:
                plt.show()

        else:
            return occ_around_goals_per_cm2, occ_wo_goals_per_cm2

    def occupancy_per_goal(self, radius=10, first_half=False):
        # get number of trials
        nr_trials = len(self.trial_raster_list)

        if first_half:
            occ_map = self.get_occ_map(spatial_resolution=1, trials_to_use=range(0, int(nr_trials/2)))
        else:
            occ_map = self.get_occ_map(spatial_resolution=1)

        # only for plotting
        occ_map_all_goals = np.zeros((occ_map.shape[0], occ_map.shape[1]))
        occ_map_all_goals[:] = np.nan
        occ_map_no_goals = np.copy(occ_map)
        occ_map_gc_summary = []

        for goal_id, goal in enumerate(self.goal_locations):
            occ_map_gc = np.zeros((occ_map.shape[0], occ_map.shape[1]))
            occ_map_gc[:] = np.nan
            y = np.arange(0, occ_map.shape[0])
            x = np.arange(0, occ_map.shape[1])
            cy = goal[0] - self.x_min
            cx = goal[1] - self.y_min
            # define mask to mark area around goal
            mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < radius ** 2
            # copy only masked z-values
            occ_map_gc[mask] = occ_map[mask]

            occ_map_gc_summary.append(occ_map_gc)


        return occ_map_gc_summary

    def pvs_around_goal(self, radius=10, first_half=False, second_half=False, spatial_resolution=1, env_dim=None,
                        goal_ind_list=None):
        # get number of trials
        nr_trials = len(self.trial_raster_list)

        if first_half and second_half:
            raise Exception("Cannot select first and second half")

        if first_half:
            occ_map = self.get_occ_map(spatial_resolution=1, trials_to_use=range(0, int(nr_trials/2)), env_dim=env_dim)
            rate_maps = self.get_rate_maps(spatial_resolution=spatial_resolution,
                                           trials_to_use=range(0, int(nr_trials / 2)), env_dim=env_dim)
        elif second_half:
            occ_map = self.get_occ_map(spatial_resolution=1, trials_to_use=range(int(nr_trials/2), nr_trials), env_dim=env_dim)
            rate_maps = self.get_rate_maps(spatial_resolution=spatial_resolution, env_dim=env_dim,
                                           trials_to_use=range(int(nr_trials/2), nr_trials))
        else:
            occ_map = self.get_occ_map(spatial_resolution=1, env_dim=env_dim)
            # get rate maps
            rate_maps = self.get_rate_maps(spatial_resolution=spatial_resolution, env_dim=env_dim)

        goal_rate_maps = []

        if goal_ind_list is None:
            goal_ind_list = []

            for goal_id, goal in enumerate(self.goal_locations):
                occ_map_gc = np.zeros((occ_map.shape[0], occ_map.shape[1]))
                occ_map_gc[:] = np.nan
                y = np.arange(0, occ_map.shape[0])
                x = np.arange(0, occ_map.shape[1])
                cy = goal[0] - self.x_min
                cx = goal[1] - self.y_min
                # define mask to mark area around goal
                mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < radius ** 2
                goal_ind = np.argwhere(mask)
                goal_ind_list.append(goal_ind)
                rate_map_goal = rate_maps[goal_ind[:, 0], goal_ind[:, 1], :]
                goal_rate_maps.append(rate_map_goal)
        else:
            for goal_ind in goal_ind_list:
                rate_map_goal = rate_maps[goal_ind[:, 0], goal_ind[:, 1], :]
                goal_rate_maps.append(rate_map_goal)

        # plt.imshow(np.nansum(np.array(occ_map_gc_summary),axis=0))
        # for goal_id, goal in enumerate(self.goal_locations):
        #
        #     cy = goal[0] - self.x_min
        #     cx = goal[1] - self.y_min
        #     plt.scatter(cx, cy, color="red")
        # plt.show()

        return goal_rate_maps, goal_ind_list

    def excess_path_per_goal(self, trial_to_use=0, radius_start=5, radius_goal=5):
        trial_loc = self.trial_loc_list[trial_to_use]

        plt.scatter(trial_loc[:, 0], trial_loc[:, 1], s=0.1, c="gray")
        for goal_id, goal in enumerate(self.goal_locations):
            plt.scatter(goal[0], goal[1], marker="x", zorder=10000, c="red", label="Goals")

        # define colors for 4 paths
        cmap = matplotlib.cm.get_cmap('Set1')
        path_colors=[cmap(0), cmap(0.15), cmap(0.25), cmap(0.5)]
        color_counter=0
        # for start: draw circle of radius 10 cm around first point, once this circle is crossed start measuring
        # distance (to avoid that animal has not really started moving and measuring distance already)
        ref_point = 0
        start_point = trial_loc[0,:]
        path_length = 0
        path_length_all_goals = []
        prev_goal = None
        detected_start = False
        detected_end_prev_goal = False
        goal_order = np.zeros(4).astype(int)
        goal_numbers = np.arange(4)
        goal_counter = 0
        goal_locations = np.copy(self.goal_locations)
        for loc_id, loc in enumerate(trial_loc):
            # detect point that lies outside a certain radius of first location
            if np.linalg.norm(loc-start_point) > radius_start and np.sum(ref_point) == 0 and not detected_start:
                plt.scatter(loc[0], loc[1], color="white", label="Start Trial")
                ref_point = loc
                prev_point = loc
                detected_start = True
            # check if there was a goal before --> if yes, need to detect point when animal leaves previous goal
            if not prev_goal is None and not detected_end_prev_goal:
                if np.linalg.norm(loc - prev_goal) > radius_start:
                    plt.scatter(loc[0], loc[1], color=path_colors[color_counter], marker="D", label="Start goal")
                    ref_point = loc
                    prev_point = loc
                    detected_end_prev_goal = True

            if np.sum(ref_point) > 0:
                # start computing the path length
                path_length += np.linalg.norm(loc - prev_point)
                plt.plot([loc[0],prev_point[0]],[loc[1],prev_point[1]], color=path_colors[color_counter], alpha=0.6)
                # check which goal is visited next
                # need to make sure that we don't detect the previous goal again!
                for goal_id, goal in enumerate(goal_locations):
                    if np.linalg.norm(loc - goal) < radius_goal and np.sum(ref_point) > 0:
                        plt.scatter(loc[0], loc[1], color=path_colors[color_counter], marker="*" ,label="Goal reached")
                        # change color for next path
                        color_counter += 1
                        path_length_all_goals.append(path_length)
                        goal_order[goal_counter] = goal_numbers[goal_id]
                        goal_counter += 1
                        ref_point = 0
                        path_length = 0
                        detected_end_prev_goal = False
                        # define current goal as previous to compute distance
                        prev_goal = goal_locations[goal_id]
                        # need to delete current goal to list so that it is not detected again
                        goal_locations = np.delete(goal_locations, goal_id, axis=0)
                        goal_numbers = np.delete(goal_numbers, goal_id)
                # check if all paths have been calculated
                if len(path_length_all_goals) == 4 or goal_counter == 4:
                    break
            prev_point = loc

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.title(self.session_name+"\n"+self.experiment_phase+", trial:"+str(trial_to_use))
        plt.show()

        # compute optimal path
        optimal_path = np.zeros(4)
        optimal_path[0] = np.linalg.norm(trial_loc[0,:] - self.goal_locations[goal_order[0].astype(int)]) - radius_goal - radius_start
        # continue with other goals --> need to offset 2 times the radius (one radius at each goal) to have a fair
        # comparison

        for goal_id in range(1, goal_order.shape[0]):
            optimal_path[goal_id] = np.linalg.norm(self.goal_locations[goal_order[goal_id]]-
                                                   self.goal_locations[goal_order[goal_id-1]])- radius_goal - radius_start

        path_length_all_goals = np.array(path_length_all_goals)

        excess_path = path_length_all_goals/optimal_path

        return path_length_all_goals, optimal_path, excess_path

    def excess_path_per_goal_all_trials(self, radius_start=5, radius_goal=5, plot_for_control=False):

        excess_path_per_trial = []
        for trial_to_use in range(self.nr_trials):
            trial_loc = self.trial_loc_list[trial_to_use]
            if plot_for_control:
                plt.scatter(trial_loc[:, 0], trial_loc[:, 1], s=0.1, c="gray")
                for goal_id, goal in enumerate(self.goal_locations):
                    plt.scatter(goal[0], goal[1], marker="x", zorder=10000, c="red", label="Goals")

                # define colors for 4 paths
                cmap = matplotlib.cm.get_cmap('Set1')
                path_colors=[cmap(0), cmap(0.15), cmap(0.25), cmap(0.5)]
                color_counter=0
            # for start: draw circle of radius 10 cm around first point, once this circle is crossed start measuring
            # distance (to avoid that animal has not really started moving and measuring distance already)
            ref_point = 0
            start_point = trial_loc[0,:]
            path_length = 0
            path_length_all_goals = []
            prev_goal = None
            detected_start = False
            detected_end_prev_goal = False
            goal_order = np.zeros(4).astype(int)
            goal_numbers = np.arange(4)
            goal_counter = 0
            goal_locations = np.copy(self.goal_locations)
            for loc_id, loc in enumerate(trial_loc):
                # detect point that lies outside a certain radius of first location
                if np.linalg.norm(loc-start_point) > radius_start and np.sum(ref_point) == 0 and not detected_start:
                    if plot_for_control:
                        plt.scatter(loc[0], loc[1], color="white", label="Start Trial")
                    ref_point = loc
                    prev_point = loc
                    detected_start = True
                # check if there was a goal before --> if yes, need to detect point when animal leaves previous goal
                if not prev_goal is None and not detected_end_prev_goal:
                    if np.linalg.norm(loc - prev_goal) > radius_start:
                        if plot_for_control:
                            plt.scatter(loc[0], loc[1], color=path_colors[color_counter], marker="D",
                                        label="Start goal")
                        ref_point = loc
                        prev_point = loc
                        detected_end_prev_goal = True

                if np.sum(ref_point) > 0:
                    # start computing the path length
                    path_length += np.linalg.norm(loc - prev_point)
                    if plot_for_control:
                        plt.plot([loc[0],prev_point[0]],[loc[1],prev_point[1]],
                                 color=path_colors[color_counter], alpha=0.6)
                    # check which goal is visited next
                    # need to make sure that we don't detect the previous goal again!
                    for goal_id, goal in enumerate(goal_locations):
                        if np.linalg.norm(loc - goal) < radius_goal and np.sum(ref_point) > 0:
                            if plot_for_control:
                                plt.scatter(loc[0], loc[1], color=path_colors[color_counter],
                                            marker="*" ,label="Goal reached")
                                # change color for next path
                                color_counter += 1
                            path_length_all_goals.append(path_length)
                            goal_order[goal_counter] = goal_numbers[goal_id]
                            goal_counter += 1
                            ref_point = 0
                            path_length = 0
                            detected_end_prev_goal = False
                            # define current goal as previous to compute distance
                            prev_goal = goal_locations[goal_id]
                            # need to delete current goal to list so that it is not detected again
                            goal_locations = np.delete(goal_locations, goal_id, axis=0)
                            goal_numbers = np.delete(goal_numbers, goal_id)
                    # check if all paths have been calculated
                    if len(path_length_all_goals) == 4 or goal_counter == 4:
                        break
                prev_point = loc
            if plot_for_control:
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys())
                plt.title(self.session_name+"\n"+self.experiment_phase+", trial:"+str(trial_to_use))
                plt.show()

            # compute optimal path
            optimal_path = np.zeros(4)
            optimal_path[0] = np.linalg.norm(trial_loc[0,:] - self.goal_locations[goal_order[0].astype(int)]) - radius_goal - radius_start
            # continue with other goals --> need to offset 2 times the radius (one radius at each goal) to have a fair
            # comparison

            for goal_id in range(1, goal_order.shape[0]):
                optimal_path[goal_id] = np.linalg.norm(self.goal_locations[goal_order[goal_id]]-
                                                       self.goal_locations[goal_order[goal_id-1]])- radius_goal - radius_start

            path_length_all_goals = np.array(path_length_all_goals)

            # check if all goals were visited
            if path_length_all_goals.shape[0] == optimal_path.shape[0]:
                excess_path = path_length_all_goals/optimal_path
                excess_path_per_trial.append(excess_path)

        return excess_path_per_trial

    # </editor-fold>





