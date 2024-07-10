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
   #   Figure 1
   #################################################################################################################"""

    # (a) trajectories learning, PRE, POST
    # ------------------------------------------------------------------------------------------------------------------
    # session_name = "mjc163R3L_0114"
    # psp = SingleSession(session_name=session_name,
    #                    cell_type=cell_type, params=params).cheese_board(experiment_phase=["learning_cheeseboard_1"])
    # psp.plot_tracking_and_goals_first_or_last_trial(trial="first", save_fig=True)
    # psp.plot_tracking_and_goals_first_or_last_trial(trial="last", save_fig=True)
    # psp = SingleSession(session_name=session_name,
    #                    cell_type=cell_type, params=params).cheese_board(experiment_phase=["learning_cheeseboard_2"])
    # psp.plot_tracking_and_goals_first_or_last_trial(trial="first", save_fig=True)

    # (b)excess path
    # ------------------------------------------------------------------------------------------------------------------
    # ms = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # ms.excess_path_per_goal_all_trials(save_fig=False)

    # (c) state reactivation probability raster
    # ------------------------------------------------------------------------------------------------------------------
    # single_ses = SingleSession(session_name="mjc169R1R_0114", params=params,
    #                            cell_type=cell_type).long_sleep()
    # single_ses.memory_drift_pre_post_mode_probability_raster(save_fig=True)

    # (d) state reactivation probability acquisition vs. recall for all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_drift_memory_drift_pre_post_mode_probability(save_fig=True)

    # (e) drift all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_drift_temporal(save_fig=False)

    # compute p-values for data vs. shuffle at t=0 and t=1
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses.long_sleep_memory_drift_data_vs_shuffle()

    # (f) drift: first vs. second half
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_time_course(template_type="phmm", save_fig=True)

    """#################################################################################################################
    #   Figure 2
    #################################################################################################################"""

    # (a)-(b) jittering spikes: all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_drift_spike_jittering(save_fig=False)

    # (c)-(d) equalized firing rates per cell: all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_drift_equalized(save_fig=True, load_from_temp=True)

    # (e)-(f) cosine similarity with decoded state (z-scored using shuffle)
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_drift_decoding_quality_across_time(save_fig=False)

    """#################################################################################################################
    #   Figure 3
    #################################################################################################################"""

    # (a)-(b) decoded vs. non-decoded-model likelihood: all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_drift_decoded_vs_non_decoded_likelhoods(save_fig=True)

    # (c)-(d) decoded vs. non-decoded-model + shuffle (artifical modes): all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_drift_decoded_vs_non_decoded_likelhoods_shuffle(save_fig=False)

    """#################################################################################################################
    #   Figure 4
    #################################################################################################################"""

    # (a) plot ratio and zoom
    # ------------------------------------------------------------------------------------------------------------------
    # session_name = "mjc169R1R_0114"
    # cb_long_sleep = SingleSession(session_name=session_name,
    #                               cell_type=cell_type, params=params).long_sleep()
    # cb_long_sleep.memory_drift_plot_rem_nrem(template_type="phmm", n_moving_average_pop_vec=900, save_fig=True)

    # (b)-(c) pos./neg. sign and cumulative score for all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_drift_opposing_nrem_rem(save_fig=False, n_moving_average_pop_vec=20,
    #                                                   rem_pop_vec_threshold=10)

    # (d) cumulative vs. net effect for all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_drift_short_time_scale_vs_long_time_scale()

    # (e)-(f) REM / NREM delta score for neighbouring epochs with directionality
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_drift_neighbouring_epochs(save_fig=False, first_type="rem")
    # mul_ses.long_sleep_memory_drift_neighbouring_epochs(save_fig=False, first_type="nrem")
    # mul_ses.long_sleep_memory_drift_neighbouring_epochs_stats()

    # (g) equalized firing rates between NREM and REM pairs
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names= all_sessions, cell_type=cell_type, params=params)
    # mul_ses.long_sleep_memory_drift_neighbouring_epochs_equalized()

    # (h) neighbouring/non-neighbouring epochs of the same type
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_drift_neighbouring_epochs_same_sleep_phase(save_fig=True, template_type="phmm")

    """#################################################################################################################
    #   Figure 5
    #################################################################################################################"""

    # (a) cell classification for all cells
    # ------------------------------------------------------------------------------------------------------------------
    # psp = MultipleSessions(session_names=all_sessions, cell_type=cell_type, params=params)
    # psp.pre_long_sleep_plot_cell_classification_mean_firing_rates_awake(normalize=False, log_scale=True,
    #                                                                     midnorm_scale=False, save_fig=True)
    # psp.long_sleep_firing_rate_changes_neighbouring_epochs(save_fig=True)

    # (b) PRE-POST ratio mean firing rates
    # ------------------------------------------------------------------------------------------------------------------
    # pp_cb = MultipleSessions(session_names=all_sessions,
    #                       cell_type=cell_type, params=params)
    # pp_cb.pre_post_cheesebaord_firing_rate_ratios_all_cells(save_fig=False)

    # (c) pre-sleep and post-sleep ratio mean firing rates
    # ------------------------------------------------------------------------------------------------------------------
    # pp_cb = MultipleSessions(session_names=all_sessions,
    #                       cell_type=cell_type, params=params)
    # pp_cb.pre_long_sleep_post_firing_rate_ratios_all_cells()

    # (d)-(f) rate maps
    # ------------------------------------------------------------------------------------------------------------------
    # single_ses = SingleSession(session_name="mjc163R2R_0114", params=params,
    #                            cell_type=cell_type).pre_post()

    # single_ses.plot_rate_maps_stable(spatial_resolution=2, gaussian_std=2, save_fig=True)
    # single_ses.plot_rate_maps_inc(spatial_resolution=2, gaussian_std=2, save_fig=True)
    # single_ses.plot_rate_maps_dec(spatial_resolution=2, gaussian_std=2, save_fig=True)

    """#################################################################################################################
    #   Figure 6
    #################################################################################################################"""

    # (a) drift with subsets of cells - example session
    # ------------------------------------------------------------------------------------------------------------------
    # session_name = "mjc169R1R_0114"
    # ls = SingleSession(session_name=session_name, cell_type=cell_type, params=params).long_sleep()
    # ls.memory_drift_temporal_subsets()

    # (b) drift with subsets of cells - all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_temporal_trend_subsets(save_fig=False, n_moving_average_pop_vec=0, nr_parts_to_split_data=1)

    """#################################################################################################################
    #   Figure 7
    #################################################################################################################"""

    # (a) example session for firing rate changes
    # ------------------------------------------------------------------------------------------------------------------
    # session_name = "mjc169R1R_0114"
    # pre_s_post = SingleSession(session_name=session_name,
    #                             cell_type=cell_type, params=params).long_sleep()
    # pre_s_post.memory_drift_plot_rem_nrem_and_firing(save_fig=True)

    # (b)-(g) correlation between delta score and changes in firing probability
    # ------------------------------------------------------------------------------------------------------------------
    # ls = MultipleSessions(session_names=all_sessions,
    #                       cell_type=cell_type, params=params)
    # ls.long_sleep_memory_drift_and_firing_prob(save_fig=True, use_only_non_stationary_periods=False,
    #                                            n_smoothing_firing_rates=200)

    """#################################################################################################################
    #   Figure 8
    #################################################################################################################"""

    # (a)-(b) Interneuron firing vs. delta score --> example session
    # ------------------------------------------------------------------------------------------------------------------
    # single_ses = TwoPopSingleSession(session_name="mjc169R1R_0114", cell_type_1="p1", cell_type_2="b1",
    #                                  params=params).long_sleep()
    # single_ses.memory_drift_delta_score_interneuron_firing(interneuron_to_save=13)

    # (c)-(d) distributions of pearson r for firing rates vs. delta score NREM & REM, and scatter plot
    # ------------------------------------------------------------------------------------------------------------------
    # tpms = TwoPopMultipleSessions(session_names=all_sessions, params=params, cell_type_1="p1", cell_type_2="b1")
    # tpms.memory_drift_delta_score_interneuron_firing(use_abs_delta_score=False, invert_rem_sign=False,
    #                                                  control_mul_comp=False, save_fig=True)

    """#################################################################################################################
    #   NOT USED: related to Figure 2
    #################################################################################################################"""

    # jittering spikes: example session
    # ------------------------------------------------------------------------------------------------------------------
    # single_ses = SingleSession(session_name="mjc163R2R_0114", params=params,
    #                            cell_type=cell_type).long_sleep()
    # single_ses.memory_drift_control_jitter_combine_sleep_phases(nr_spikes_per_jitter_window=10000,
    # save_results=False, plotting=True)

    # Null model: example session
    # ------------------------------------------------------------------------------------------------------------------
    # single_ses = SingleSession(session_name="mjc163R4R_0114", params=params,
    #                            cell_type=cell_type).long_sleep()
    # single_ses.memory_drift_control_null_model(save_results=True)

    # null model: all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_drift_null_model(save_fig=True)

    # equalized firing rates per cell: example session
    # ------------------------------------------------------------------------------------------------------------------
    # single_ses = SingleSession(session_name="mjc163R2R_0114", params=params,
    #                            cell_type=cell_type).long_sleep()
    # single_ses.memory_drift_control_equalize_combine_sleep_phases(nr_chunks=5, n_moving_average_pop_vec=10000,
    #                                                               save_fig=False)

    # decoding quality: z-scored using shuffle
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_drift_decoding_quality_across_time(save_fig=False)

    # decoding quality: using likelihoods --> example session
    # ------------------------------------------------------------------------------------------------------------------
    # single_ses = SingleSession(session_name="mjc163R2R_0114", params=params,
    #                            cell_type=cell_type).long_sleep()
    # single_ses.memory_drift_decoding_quality_across_time_likelihoods(save_result=True)

    # decoding quality using likelihoods: all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=["mjc163R2R_0114", "mjc169R4R_0114", "mjc148R4R_0113",
    #                                           "mjc163R3L_0114"], params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_drift_decoding_quality_across_time_likelihoods(save_fig=True)

    """#################################################################################################################
    #   NOT USED: related to Figure 3
    #################################################################################################################"""

    # Decoded vs. non-decoded-model + shuffle (artifical modes): example session
    # ------------------------------------------------------------------------------------------------------------------
    # single_ses = SingleSession(session_name="mjc148R4R_0113", params=params,
    #                            cell_type=cell_type).long_sleep()
    # single_ses.memory_drift_combine_sleep_phases_decoded_non_decoded_likelihoods_shuffle(save_result=True)

    # Decoded vs. non-decoded-model: example session
    # ------------------------------------------------------------------------------------------------------------------
    # single_ses = SingleSession(session_name="mjc163R3L_0114", params=params,
    #                            cell_type=cell_type).long_sleep()
    # single_ses.memory_drift_combine_sleep_phases_decoded_likelihoods()

    # Decoded vs. non-decoded-model + shuffle (artifical modes): example session --> COSINE DISTANCE
    # ------------------------------------------------------------------------------------------------------------------
    # single_ses = SingleSession(session_name="mjc148R4R_0113", params=params,
    #                            cell_type=cell_type).long_sleep()
    # single_ses.memory_drift_combine_sleep_phases_decoded_non_decoded_cosine_distance_shuffle(save_result=True)

    # Decoded vs. non-decoded-model + shuffle (artifical modes): all sessions --> COSINE DISTANCE
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_drift_decoded_vs_non_decoded_cosine_distance_shuffle(save_fig=False)