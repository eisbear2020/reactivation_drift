########################################################################################################################
#
#   MACHINE LEARNING METHODS
#
#   Description: contains classes that are used to perform analysis methods from ML
#
#   Author: Lars Bollmann
#
#   Created: 11/03/2020
#
#   Structure:
#
#               (1) MlMethodsOnePopulation: methods for one population
#
#               (2) MlMethodsTwoPopulations: methods for two populations
#
########################################################################################################################

from collections import OrderedDict
from .support_functions import upper_tri_without_diag, multi_dim_scaling, perform_PCA, perform_TSNE, \
    perform_isomap, log_multivariate_poisson_density, correlateOneWithMany, find_hse, make_square_axes

from .plotting_functions import plot_pop_clusters, plot_2D_scatter, plot_3D_scatter, \
    plot_ridge_weight_vectors, plot_true_vs_predicted, plot_pop_cluster_analysis, plot_cca_loadings, cca_video
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import sklearn as sk
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
import os
import multiprocessing as mp
import pickle
from hmmlearn.base import _BaseHMM
import numpy as np
import sklearn.cluster as cluster
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
from functools import partial
from palettable.colorbrewer import qualitative
import sklearn.svm as svm
from sklearn.model_selection import train_test_split
# from hmmlearn import hmm
from sklearn.decomposition import PCA
import rcca
import os

########################################################################################################################
#   class MlMethodsOnePopulation
########################################################################################################################


class MlMethodsOnePopulation:
    """Machine learning methods for two populations"""

    def __init__(self, act_map=None, params=None, cell_type=None):
        self.cell_type = cell_type
        self.raster = act_map
        self.params = params
        self.X = None
        self.Y = None

    def parallelize_cross_val_model(self, nr_cluster_array, nr_cores, model_type, folder_name,
                                    raster_data=None, splits=None, cells_used="all_cells"):
        # --------------------------------------------------------------------------------------------------------------
        # parallelization of gmm cross validation
        #
        # parameters:   - nr_clusters_array, np.arange: for which number of clusters GMM is fit
        #               - nr_cores, int: how many cores to run in parallel
        # --------------------------------------------------------------------------------------------------------------

        # custom_splits_array = custom_splits * np.ones(nr_clusters_array.shape[0])
        # custom_splits_array = custom_splits_array.astype("bool")

        # result location
        if model_type == "pHMM":
                if self.params.cross_val_splits == "custom_splits":
                    res_dir = self.params.pre_proc_dir+"phmm/cross_val/"+cells_used+"/custom_splits/"+folder_name
                elif self.params.cross_val_splits == "standard_k_fold":
                    res_dir = self.params.pre_proc_dir + "phmm/cross_val/"+cells_used+"/standard_k_fold/" + folder_name
                elif self.params.cross_val_splits == "trial_splitting":
                    res_dir = self.params.pre_proc_dir + "phmm/cross_val/"+cells_used+"/trial_splitting/" + folder_name
        elif model_type == "GMM":
            if self.params.cross_val_splits == "custom_splits":
                res_dir = self.params.pre_proc_dir+"gmm/cross_val/custom_splits/"+folder_name
            elif self.params.cross_val_splits == "standard_k_fold":
                res_dir = self.params.pre_proc_dir + "gmm/cross_val/standard_k_fold/" + folder_name

        # check if directory exists already, otherwise create it
        if not os.path.isdir(res_dir):
            os.mkdir(res_dir)

        with mp.Pool(nr_cores) as p:
            # use partial to pass several arguments to cross_val_model, only nr_clusters_array is a changing one, the
            # others are constant
            multi_arg = partial(self.cross_val_model, res_dir=res_dir, raster_data=raster_data,
                                model_type=model_type, splits=splits)
            p.map(multi_arg, nr_cluster_array)

    def cross_val_model(self, nr_clusters, res_dir, raster_data=None, model_type="pHMM", splits=None):
        # --------------------------------------------------------------------------------------------------------------
        # cross validation of selected model
        #
        # parameters:   - nr_clusters, int: number of clusters to fit to the data
        #               - raster_data, array: input data [nr_cells, nr_time_bins]
        #               - model_type, string ["GMM", POISSON_HMM"]: defines which model to fit to data
        #               - custom_splits, bool: use custom splits or standard k-fold CV
        #
        # returns:  - saves result in text file [mean. log-likeli, std. log-likeli, mean. per sample log-likeli,
        #                                       std. per sample log-likeli, time_bin_size in seconds]
        # --------------------------------------------------------------------------------------------------------------

        # how many times to fit for each split
        max_nr_fitting = 5

        # print current process nr
        print(" - FITTING "+ model_type +" FOR #CLUSTERS = " + str(nr_clusters)+"\n")
        # print(mp.current_process())

        # check if result file exists already
        if os.path.isfile(res_dir + "/" + str(nr_clusters)):
            print("   --> RESULT EXISTS ALREADY ... SKIPPING\n")
            return

        if raster_data is None:
            # load data
            x = self.raster.T
        else:
            x = raster_data.T

        if splits is not None:
            # cross validation using provided splits
            # ----------------------------------------------------------------------------------------------------------
            nr_folds = len(splits)
            result_array = np.zeros(len(splits))
            result_array_per_sample = np.zeros(len(splits))

            for fold, test_range in enumerate(splits):
                X_test = x[test_range,:]
                X_train = np.delete(x,test_range, axis=0)

                # fit model several times to average over the influence of initialization
                el_fitting = np.zeros(max_nr_fitting)
                el_fitting_per_sample = np.zeros(max_nr_fitting)
                for nr_fitting in range(max_nr_fitting):
                    if model_type == "GMM":
                        model = GaussianMixture(n_components=nr_clusters)
                    elif model_type == "pHMM":
                        model = PoissonHMM(n_components=nr_clusters)
                    model.fit(X_train)
                    el_fitting[nr_fitting] = model.score(X_test)
                    el_fitting_per_sample[nr_fitting] = model.score(X_test) / X_test.shape[0]
                result_array[fold] = np.mean(el_fitting)
                result_array_per_sample[fold] = np.mean(el_fitting_per_sample)

        else:

            # standard n fold cross validation
            # ----------------------------------------------------------------------------------------------------------
            # number of folds --> 10 by default
            nr_folds = 10
            result_array = np.zeros(nr_folds)
            result_array_per_sample = np.zeros(nr_folds)
            skf = KFold(n_splits=nr_folds)

            for fold, (train_index, test_index) in enumerate(skf.split(x)):
                X_train, X_test = x[train_index], x[test_index]
                # fit model several times to average over the influence of initialization
                el_fitting = np.zeros(max_nr_fitting)
                el_fitting_per_sample = np.zeros(max_nr_fitting)
                for nr_fitting in range(max_nr_fitting):
                    if model_type == "GMM":
                        model = GaussianMixture(n_components=nr_clusters)
                    elif model_type == "pHMM":
                        model = PoissonHMM(n_components=nr_clusters)
                    model.fit(X_train)
                    el_fitting[nr_fitting] = model.score(X_test)
                    el_fitting_per_sample[nr_fitting] = model.score(X_test) / X_test.shape[0]
                result_array[fold] = np.mean(el_fitting)
                result_array_per_sample[fold] = np.mean(el_fitting_per_sample)

        # print current process nr
        print(" ... DONE WITH #CLUSTERS = " + str(nr_clusters)+"\n")

        # save results
        # --------------------------------------------------------------------------------------------------------------
        with open(res_dir + "/" + str(nr_clusters), "a") as f:
            # f.write(str(np.mean(result_array[:])) + "," + str(np.std(result_array[:])) + "\n")
            f.write(str(np.round(np.mean(result_array[:]),2)) + "," + str(np.round(np.std(result_array[:]),2)) + "," +
                    str(np.round(np.mean(result_array_per_sample[:]), 5)) + "," +
                    str(np.round(np.std(result_array_per_sample[:]), 5)) + "," + str(self.params.time_bin_size)+ "," +
                    str(nr_folds)+"\n")

    def cross_val_view_results(self, folder_name, range_to_plot=None, save_fig=False, cells_used="all_cells",
                               model_type="pHMM"):
        # --------------------------------------------------------------------------------------------------------------
        # view results of GMM cross validation (optimal number of clusters)
        #
        # parameters:   - identifier, str: only used when GMM cross val results are saved with an additional identifier
        #                 (e.g. z_scored)
        #               - first_data_point_to_plot: from which #modes to plot
        # --------------------------------------------------------------------------------------------------------------

        # result location
        # result location
        # result location
        if model_type == "pHMM":
                if self.params.cross_val_splits == "custom_splits":
                    res_dir = self.params.pre_proc_dir+"phmm/cross_val/"+cells_used+"/custom_splits/"+folder_name
                elif self.params.cross_val_splits == "standard_k_fold":
                    res_dir = self.params.pre_proc_dir + "phmm/cross_val/"+cells_used+"/standard_k_fold/" + folder_name
                elif self.params.cross_val_splits == "trial_splitting":
                    res_dir = self.params.pre_proc_dir + "phmm/cross_val/"+cells_used+"/trial_splitting/" + folder_name
        elif model_type == "GMM":
            if self.params.cross_val_splits == "custom_splits":
                res_dir = self.params.pre_proc_dir+"gmm/cross_val/custom_splits/"+folder_name
            elif self.params.cross_val_splits == "standard_k_fold":
                res_dir = self.params.pre_proc_dir + "gmm/cross_val/standard_k_fold/" + folder_name

        nr_cluster_list = []
        mean_values = []
        std_values = []
        mean_values_per_sample = []
        std_values_per_sample = []
        read_time_bin_size = True

        for file in os.listdir(res_dir):
            nr_cluster_list.append(int(file))
            with open(res_dir + "/" + file) as f:
                res = f.readline()
                mean_values.append(float(res.replace("\n", "").split(",")[0]))
                std_values.append(float(res.replace("\n", "").split(",")[1]))
                mean_values_per_sample.append(float(res.replace("\n", "").split(",")[2]))
                std_values_per_sample.append(float(res.replace("\n", "").split(",")[3]))
                if read_time_bin_size:
                    time_bin_size = float(res.replace("\n", "").split(",")[4])
                    try:
                        nr_folds = int(res.replace("\n", "").split(",")[5])
                    except:
                        print("NR. FOLDS NOT FOUND")
                        nr_folds = 10
                    read_time_bin_size = False
        # sort in right order
        mean_values = [x for _, x in sorted(zip(nr_cluster_list, mean_values))]
        std_values = [x for _, x in sorted(zip(nr_cluster_list, std_values))]
        mean_values_per_sample = [x for _, x in sorted(zip(nr_cluster_list, mean_values_per_sample))]
        std_values_per_sample = [x for _, x in sorted(zip(nr_cluster_list, std_values_per_sample))]
        nr_cluster_list = sorted(nr_cluster_list)

        if save_fig:
            plt.style.use('default')
        if range_to_plot is None:
            plt.plot(nr_cluster_list, mean_values_per_sample, color="red", marker='o')
        else:
            plt.plot(nr_cluster_list[range_to_plot[0]:range_to_plot[1]],
                     mean_values_per_sample[range_to_plot[0]:range_to_plot[1]], color="red", marker='o')
        plt.xlabel("#states")
        plt.ylabel("mean per sample log-likelihood ")
        plt.grid()
        plt.tight_layout()
        if save_fig:
            make_square_axes(plt.gca())
            plt.title("cross-validated log-likelihood (10 fold)")
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("phmm_max_likelihood.svg", transparent="True")
        else:
            plt.title("GOODNESS OF FIT: "+str(nr_folds)+"-FOLD (5 FITS PER FOLD)\n PER SAMPLE, TIME BIN SIZE = "+
                      str(time_bin_size)+"s\n SPLITTING METHOD: "+self.params.cross_val_splits)
            plt.show()

        print("MAX. LOG.LIKELIHOOD: "+ str(max(mean_values)) +" --> "+
              str(nr_cluster_list[np.argmax(np.array(mean_values_per_sample))])+" CLUSTERS")

        if range_to_plot is None:
            plt.errorbar(nr_cluster_list, mean_values_per_sample, yerr = std_values_per_sample,
                         fmt='o-', label="TEST", c="#990000")
        else:
            plt.errorbar(nr_cluster_list[range_to_plot[0]:range_to_plot[1]],
                         mean_values_per_sample[range_to_plot[0]:range_to_plot[1]],
                         yerr = std_values_per_sample[range_to_plot[0]:range_to_plot[1]],
                         fmt='o-', label="TEST", c="#990000")

        plt.xlabel("#MODES")
        plt.ylabel("PER SAMPLE LOG-LIKELIHOOD (MEAN+STD OF 5 FITS)")
        plt.title("GOODNESS OF FIT: "+str(nr_folds)+"-FOLD (5 FITS PER FOLD)\n PER SAMPLE, TIME BIN SIZE = "+str(time_bin_size)+"s")
        plt.grid()
        plt.tight_layout()
        plt.show()
        print("MAX. LOG.LIKELIHOOD: "+ str(max(mean_values)) +" --> "+
              str(nr_cluster_list[np.argmax(np.array(mean_values_per_sample))])+" CLUSTERS")

    def get_optimal_mode_number(self, folder_name, cells_used="all_cells",
                               model_type="pHMM"):
        # --------------------------------------------------------------------------------------------------------------
        # view results of GMM cross validation (optimal number of clusters)
        #
        # parameters:   - identifier, str: only used when GMM cross val results are saved with an additional identifier
        #                 (e.g. z_scored)
        #               - first_data_point_to_plot: from which #modes to plot
        # --------------------------------------------------------------------------------------------------------------

        if model_type == "pHMM":
                if self.params.cross_val_splits == "custom_splits":
                    res_dir = self.params.pre_proc_dir+"phmm/cross_val/"+cells_used+"/custom_splits/"+folder_name
                elif self.params.cross_val_splits == "standard_k_fold":
                    res_dir = self.params.pre_proc_dir + "phmm/cross_val/"+cells_used+"/standard_k_fold/" + folder_name
                elif self.params.cross_val_splits == "trial_splitting":
                    res_dir = self.params.pre_proc_dir + "phmm/cross_val/"+cells_used+"/trial_splitting/" + folder_name
        elif model_type == "GMM":
            if self.params.cross_val_splits == "custom_splits":
                res_dir = self.params.pre_proc_dir+"gmm/cross_val/custom_splits/"+folder_name
            elif self.params.cross_val_splits == "standard_k_fold":
                res_dir = self.params.pre_proc_dir + "gmm/cross_val/standard_k_fold/" + folder_name

        nr_cluster_list = []
        mean_values = []
        std_values = []
        mean_values_per_sample = []
        std_values_per_sample = []
        read_time_bin_size = True

        for file in os.listdir(res_dir):
            nr_cluster_list.append(int(file))
            with open(res_dir + "/" + file) as f:
                res = f.readline()
                mean_values.append(float(res.replace("\n", "").split(",")[0]))
                std_values.append(float(res.replace("\n", "").split(",")[1]))
                mean_values_per_sample.append(float(res.replace("\n", "").split(",")[2]))
                std_values_per_sample.append(float(res.replace("\n", "").split(",")[3]))
                if read_time_bin_size:
                    time_bin_size = float(res.replace("\n", "").split(",")[4])
                    try:
                        nr_folds = int(res.replace("\n", "").split(",")[5])
                    except:
                        print("NR. FOLDS NOT FOUND")
                        nr_folds = 10
                    read_time_bin_size = False
        # sort in right order
        mean_values = [x for _, x in sorted(zip(nr_cluster_list, mean_values))]
        std_values = [x for _, x in sorted(zip(nr_cluster_list, std_values))]
        mean_values_per_sample = [x for _, x in sorted(zip(nr_cluster_list, mean_values_per_sample))]
        std_values_per_sample = [x for _, x in sorted(zip(nr_cluster_list, std_values_per_sample))]
        nr_cluster_list = sorted(nr_cluster_list)

        return nr_cluster_list[np.argmax(np.array(mean_values_per_sample))]

    def plot_custom_splits(self):
        # plot custom splits
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
        bin_num = 1000
        bins = np.arange(bin_num + 1)

        # length of one chunk
        n_chunks = int(bin_num / nr_chunks)
        fig = plt.figure()
        ax = fig.add_subplot(111)
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
            test_range = np.array(test_range)
            ax.scatter(range(bin_num), fold * np.ones(bin_num), c="b", label="TRAIN")
            ax.scatter(test_range, fold * np.ones(test_range.shape[0]), c="r", label="TEST")

        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("CUSTOM SPLITS FOR CROSS-VALIDATION")
        plt.xlabel("DATA POINTS")
        plt.ylabel("FOLD NR.")

        plt.show()

    def parallelize_svm(self, m_subset, X_train, X_test, y_train, y_test, nr_subsets=10):

        with mp.Pool(nr_subsets) as p:
            # use partial to pass several arguments to cross_val_model, only nr_clusters_array is a changing one, the
            # others are constant
            multi_arg = partial(self.svm_with_subsets, X_train=X_train, X_test=X_test,
                                y_test=y_test, y_train=y_train)

            mean_accuracy = p.map(multi_arg, (np.ones(nr_subsets)*m_subset).astype(int))

            return mean_accuracy

    def svm_with_subsets(self, m_subset, X_train, X_test, y_train, y_test):

        subset = np.random.choice(a=range(X_train.shape[1]), size=m_subset, replace=False)

        X_train_subset = X_train[:, subset]
        X_test_subset = X_test[:,subset]

        # clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto', kernel="linear"))
        clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto', kernel="linear"))
        clf.fit(X_train_subset, y_train)

        return clf.score(X_test_subset, y_test)


class PoissonHMM(_BaseHMM):
    """ Hidden Markov Model with independent Poisson emissions.
    Parameters
    ----------
    n_components : int
        Number of states.
    startprob_prior : array, shape (n_components, )
        Initial state occupation prior distribution.
    transmat_prior : array, shape (n_components, n_components)
        Matrix of prior transition probabilities between states.
    algorithm : string, one of the :data:`base.DECODER_ALGORITHMS`
        Decoder algorithm.
    random_state: RandomState or an int seed (0 by default)
        A random number generator instance.
    n_iter : int, optional
        Maximum number of iterations to perform.
    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.
    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.
    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'm' for means and 'c' for covars. Defaults
        to all parameters.
    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'm' for means and 'c' for covars.
        Defaults to all parameters.
    Attributes
    ----------
    n_features : int
        Dimensionality of the (independent) Poisson emissions.
    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.
    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.
    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.
    means_ : array, shape (n_components, n_features)
        Mean parameters for each state.
    Examples
    --------
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    PoissonHMM(algorithm='viterbi',...
    """
    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stmc", init_params="stmc"):

        self.implementation = None

        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params)

        self.means_prior = means_prior
        self.means_weight = means_weight
        self.time_bin_size = None

    def _check(self):
        super(PoissonHMM, self)._check()

        self.means_ = np.asarray(self.means_)
        self.n_features = self.means_.shape[1]

    def _compute_log_likelihood(self, obs):
        return log_multivariate_poisson_density(obs, self.means_)

    def _generate_sample_from_state(self, state, random_state=None):
        rng = check_random_state(random_state)
        return rng.poisson(self.means_[state])

    def _init(self, X, lengths=None, params='stmc'):
        super(PoissonHMM, self)._init(X)

        _, n_features = X.shape
        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (n_features, self.n_features))

        self.n_features = n_features
        if 'm' in params or not hasattr(self, "means_"):
            kmeans = cluster.KMeans(n_clusters=self.n_components, n_init="auto")
            kmeans.fit(X)
            self.means_ = kmeans.cluster_centers_

    def _initialize_sufficient_statistics(self):
        stats = super(PoissonHMM, self)._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(PoissonHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)

        if 'm' in self.params in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

    def _do_mstep(self, stats):
        super(PoissonHMM, self)._do_mstep(stats)

        means_prior = self.means_prior
        means_weight = self.means_weight

        denom = stats['post'][:, np.newaxis]
        if 'm' in self.params:
            self.means_ = ((means_weight * means_prior + stats['obs'])
                           / (means_weight + denom))
            self.means_ = np.where(self.means_ > 1e-5, self.means_, 1e-3)

    def set_time_bin_size(self, time_bin_size):
        # set time bin size in seconds for later analysis (need to know what time bin size was used to fit model)
        self.time_bin_size = time_bin_size