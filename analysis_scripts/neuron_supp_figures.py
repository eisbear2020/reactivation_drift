########################################################################################################################
#
#   Analysis for Neuron re-submission
#
########################################################################################################################

from function_files.sessions import SingleSession, MultipleSessions


def execute(params):

    # params
    # ------------------------------------------------------------------------------------------------------------------

    cell_type = "p1"

    all_sessions = ["mjc163R4R_0114", "mjc163R2R_0114", "mjc169R4R_0114", "mjc163R1L_0114", "mjc148R4R_0113",
                    "mjc163R3L_0114", "mjc169R1R_0114"]

    """#################################################################################################################
    #   Supplementary 1: Clustering stability
    #################################################################################################################"""

    # (a) example of waveforms of single cells across time
    # ------------------------------------------------------------------------------------------------------------------
    # single_ses = SingleSession(session_name="mjc163R2R_0114", params=params,
    #                            cell_type=cell_type).all_data()
    # single_ses.plot_waveforms_single_cells(cell_id=18, electrode=[0,1,2,3], y_min=-900, y_max=580, save_fig=False)
    # NOT used:
    # single_ses.plot_waveforms_single_cells(cell_id=3, electrode=[0,1,2,3], y_min=-900, y_max=580, save_fig=True)
    # single_ses.plot_waveforms_single_cells(cell_id=1, electrode=[0,1,2,3], y_min=-900, y_max=580, save_fig=True)

    # (b) feature stability over time for all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # ms = MultipleSessions(session_names=all_sessions,
    #                       cell_type=cell_type, params=params)
    # ms.assess_feature_stability(save_fig=True)

    # (c)-(d) cluster quality (separation between units) over time example
    # ------------------------------------------------------------------------------------------------------------------
    # single_ses = SingleSession(session_name="mjc163R4R_0114", params=params,
    #                            cell_type=cell_type).all_data()
    # single_ses.cluster_quality_over_time(save_fig=True)
    #
    # compute stats for all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # ms = MultipleSessions(session_names=all_sessions,
    #                       cell_type=cell_type, params=params)
    # ms.cluster_quality_over_time()

    # (e) visualizing clusters across time
    # ------------------------------------------------------------------------------------------------------------------
    # single_ses = SingleSession(session_name="mjc163R2R_0114", params=params,
    #                            cell_type=cell_type).all_data()
    # single_ses.assess_feature_stability_per_tetrode_l_ratio(plot_for_control=True)
    # tetrode 2: features 0-6 or 0-9
    # single_ses.plot_clusters_across_time(save_fig=True)

    # (f) L_ratio: all sessions - across time
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.cluster_quality_over_time_l_ratio()

    # (g)-(h) L_ratio vs. drift: all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.l_ratio_vs_drift()

    """#################################################################################################################
    #   Supplementary 2: pHMM optimal number of states and model quality
    #################################################################################################################"""

    # (b) cross-correlation of optimal #states example
    # ------------------------------------------------------------------------------------------------------------------
    # session_name = "mjc163R4R_0114"
    # c_b = SingleSession(session_name=session_name, cell_type=cell_type, params=params).cheese_board(
    #     experiment_phase=["learning_cheeseboard_1"])
    # c_b.view_cross_val_results(save_fig=False)

    # (c) #optimal states
    # ------------------------------------------------------------------------------------------------------------------
    # ms = MultipleSessions(session_names=all_sessions,
    #                       cell_type=cell_type, params=params)
    # ms.pre_post_optimal_number_states()

    # (d) & (e) phmm model quality: example
    # ------------------------------------------------------------------------------------------------------------------
    # session_name = "mjc163R4R_0114"
    # c_b = SingleSession(session_name=session_name, cell_type=cell_type, params=params).cheese_board(
    #     experiment_phase=["learning_cheeseboard_1"])
    # c_b.evaluate_poisson_hmm(plotting=True)

    # (d) & (e) phmm model quality R values from all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # ms = MultipleSessions(session_names=all_sessions,
    #                       cell_type=cell_type, params=params)
    # ms.cheeseboard_evaluate_poisson_hmm()

    """#################################################################################################################
    #   Supplementary 3: pHMM states - spatial information and decoding
    #################################################################################################################"""

    # (a) spatial information all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # ms = MultipleSessions(session_names=all_sessions,
    #                       cell_type=cell_type, params=params)
    # ms.cheeseboard_phmm_spatial_information()

    # (a) spatial information example session
    # ------------------------------------------------------------------------------------------------------------------
    # session_name = "mjc163R2R_0114"
    # c_b = SingleSession(session_name=session_name, cell_type=cell_type, params=params).cheese_board(
    #     experiment_phase=["learning_cheeseboard_1"])
    # c_b.analyze_all_modes_spatial_information()
    # c_b.plot_phmm_mode_spatial(mode_id=13, save_fig=True)
    # c_b.plot_phmm_mode_spatial(mode_id=33, save_fig=True)

    # (b) goal coding all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # ms = MultipleSessions(session_names=all_sessions,
    #                       cell_type=cell_type, params=params)
    # ms.cheeseboard_phmm_goal_coding()

    # (b) goal coding examples
    # ------------------------------------------------------------------------------------------------------------------
    # session_name = "mjc163R2R_0114"
    # c_b = SingleSession(session_name=session_name, cell_type=cell_type, params=params).cheese_board(
    #     experiment_phase=["learning_cheeseboard_1"])
    # c_b.plot_phmm_mode_spatial(mode_id=5, save_fig=True)
    # c_b.plot_phmm_mode_spatial(mode_id=18, save_fig=True)

    # (c) pHMM decoding example session
    # ------------------------------------------------------------------------------------------------------------------
    # session_name = "mjc163R2R_0114"
    # c_b = SingleSession(session_name=session_name, cell_type=cell_type, params=params).cheese_board(
    #     experiment_phase=["learning_cheeseboard_1"])
    # c_b.decode_location_phmm_cross_validated(save_fig=True)

    # (d) pHMM decoding all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # ms = MultipleSessions(session_names=all_sessions,
    #                       cell_type=cell_type, params=params)
    # ms.cheeseboard_decode_location_phmm_cross_validated(save_fig=True)

    # (e) Bayesian decoding example session
    # ------------------------------------------------------------------------------------------------------------------
    # session_name = "mjc163R2R_0114"
    # c_b = SingleSession(session_name=session_name, cell_type=cell_type, params=params).cheese_board(
    #     experiment_phase=["learning_cheeseboard_1"])
    # c_b.decode_location_bayes_cross_validated(save_fig=True, spatial_resolution_rate_map=1)

    # (f) Bayesian decoding all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # ms = MultipleSessions(session_names=all_sessions,
    #                       cell_type=cell_type, params=params)
    # ms.cheeseboard_decode_location_bayes_cross_validated(spatial_resolution_rate_map=1, save_fig=True)

    # (g) Bayesian vs. pHMM decoding
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.cheeseboard_decode_bayes_vs_phmm()

    # (h) drift for ising model: example session
    # ------------------------------------------------------------------------------------------------------------------
    # session_name = "mjc163R1L_0114"
    # cb_long_sleep = SingleSession(session_name=session_name,
    #                               cell_type=cell_type, params=params).long_sleep()
    # cb_long_sleep.memory_drift_plot_temporal_trend(template_type="ising", n_moving_average_pop_vec=400, save_fig=True)

    # (i) drift for ising model: all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_drift_temporal(save_fig=True, template_type="ising")

    # (j) cumulative sum for bayesian
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_drift_opposing_nrem_rem(save_fig=True, template_type="ising")

    # (k) zigzagging vs. net change
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_drift_short_time_scale_vs_long_time_scale(smoothing=200)

    """#################################################################################################################
    #   Supplementary 5: Features of different subsets 1
    #################################################################################################################"""
    # (a) mean firing rates per state
    # ------------------------------------------------------------------------------------------------------------------
    # pp_cb = MultipleSessions(session_names=all_sessions,
    #                       cell_type=cell_type, params=params)
    # pp_cb.pre_post_cheeseboard_modes_cell_contribution()

    # (b) number of cells per subset
    # ------------------------------------------------------------------------------------------------------------------
    # pp_cb = MultipleSessions(session_names=all_sessions,
    #                       cell_type=cell_type, params=params)
    # pp_cb.plot_classified_cell_distribution()

    # (c), (d), (e) mean firing rates and z-scored mean firing rates
    # ------------------------------------------------------------------------------------------------------------------
    # pp_cb = MultipleSessions(session_names=all_sessions,
    #                       cell_type=cell_type, params=params)
    # pp_cb.pre_long_sleep_post_firing_rates_all_cells(save_fig=True, use_log_scale=True)

    # (f) SWR gain
    # ------------------------------------------------------------------------------------------------------------------
    # cb = MultipleSessions(session_names=all_sessions, cell_type=cell_type, params=params)
    # cb.cheeseboard_firing_rates_gain_during_swr(threshold_stillness=10, experiment_phase = ["learning_cheeseboard_2"])

    # (g) stable, dec, inc cells per tetrode
    # ------------------------------------------------------------------------------------------------------------------
    # single_ses = SingleSession(session_name="mjc163R4R_0114", params=params,
    #                            cell_type=cell_type).all_data()
    # single_ses.feature_stability_per_tetrode_subsets(save_fig=True)

    # (h) recording stability stable, dec, inc
    # ------------------------------------------------------------------------------------------------------------------
    # cb = MultipleSessions(session_names=all_sessions, cell_type=cell_type, params=params)
    # cb.cluster_quality_over_time_subsets(hours_to_compute=[15, 18], save_fig=True)

    # (i)-(j) PV correlations
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, cell_type=cell_type, params=params)
    # mul_ses.pre_post_cheeseboard_remapping_all_subsets(save_fig=True)
    # mul_ses.pre_post_cheeseboard_remapping_all_subsets(normalized=True, save_fig=True, min_mean_firing_rate=0.1)

    # (k) skaggs per second
    # ------------------------------------------------------------------------------------------------------------------
    # cb = MultipleSessions(session_names=all_sessions, cell_type=cell_type, params=params)
    # cb.pre_post_cheeseboard_spatial_information(spatial_resolution=3, info_measure="skaggs_second", save_fig=True)

    """#################################################################################################################
    #   Supplementary 6: Features of different subsets 2
    #################################################################################################################"""

    # (a) theta coupling awake vs. rem
    # ------------------------------------------------------------------------------------------------------------------
    # cb = MultipleSessions(session_names=all_sessions, cell_type=cell_type, params=params)
    # cb.pre_long_sleep_theta_phase(save_fig=False)

    # (b) theta coupling awake vs. rem: stats
    # ------------------------------------------------------------------------------------------------------------------
    # cb = MultipleSessions(session_names=all_sessions, cell_type=cell_type, params=params)
    # cb.pre_long_sleep_theta_phase_separate(save_fig=True)

    # (e)-(f) burstiness
    # ------------------------------------------------------------------------------------------------------------------
    # cb = MultipleSessions(session_names=all_sessions, cell_type=cell_type, params=params)
    # cb.cheeseboard_burstiness(experiment_phase = ["learning_cheeseboard_2"], save_fig=False)

    # (g) SWR profiling: example session
    # ------------------------------------------------------------------------------------------------------------------
    # single_ses = SingleSession(session_name="mjc163R2R_0114", params=params,
    #                            cell_type=cell_type).long_sleep(data_to_use="ext_eegh")
    # single_ses.swr_profile()

    # (h) SWR profiling:all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # cb = MultipleSessions(session_names=all_sessions, cell_type=cell_type, params=params)
    # cb.long_sleep_swr_profile_subsets(save_fig=True)

    """#################################################################################################################
    #   Supplementary 7: Remapping during learning and REM/NREM analysis
    #################################################################################################################"""

    # (a)-(d) distance to goals
    # ------------------------------------------------------------------------------------------------------------------
    # cb = MultipleSessions(session_names=all_sessions, cell_type=cell_type, params=params)
    # cb.cheeseboard_place_field_goal_distance_temporal(nr_trials=2, pre_or_post="pre", spatial_resolution_rate_maps=10,
    #                                                   save_fig=True)
    # cb.cheeseboard_place_field_goal_distance_temporal(nr_trials=2, pre_or_post="post",
    #                                                   spatial_resolution_rate_maps=10, save_fig=True)

    # (e)-(h) correlation of firing probability changes of neighbouring epochs
    # (with both directions REM --> NREM and NREM --> REM
    # ------------------------------------------------------------------------------------------------------------------
    # psp = MultipleSessions(session_names=all_sessions,
    #                       cell_type=cell_type, params=params)
    # psp.long_sleep_firing_rate_changes_neighbouring_epochs(first_type="nrem", save_fig=True)
    # psp.long_sleep_firing_rate_changes_neighbouring_epochs(first_type="rem", save_fig=True)
    # psp.long_sleep_firing_rate_changes_neighbouring_epochs_stats()

    # (i)-(l) correlation of firing probability changes epoch n with epoch n+1
    # ------------------------------------------------------------------------------------------------------------------
    # psp = MultipleSessions(session_names=all_sessions, cell_type=cell_type, params=params)
    # psp.long_sleep_firing_rate_changes_neighbouring_epochs_same_sleep_phase(save_fig=True)

    # (m)-(t) firing prob vs. likelihood change
    #-------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names= all_sessions, cell_type=cell_type, params=params)
    # mul_ses.long_sleep_memory_drift_delta_log_likelihood_and_firing_prob(save_fig=True)

    """#################################################################################################################
    #   Supplementary 8: interneurons
    #################################################################################################################"""

    # (a)-(b) distributions of pearson r for firing rates vs. delta score NREM & REM, and scatter plot
    # ------------------------------------------------------------------------------------------------------------------
    # tpms = TwoPopMultipleSessions(session_names=all_sessions, params=params, cell_type_1="p1", cell_type_2="b1")
    # tpms.memory_drift_delta_score_vs_abs_delta_score_interneuron_firing()
    # tpms.memory_drift_delta_score_interneuron_firing(use_abs_delta_score=False, invert_rem_sign=False,
    #                                                  control_mul_comp=False)
    # tpms.memory_drift_delta_score_delta_interneuron_firing()