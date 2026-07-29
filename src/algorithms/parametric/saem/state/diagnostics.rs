use super::*;

impl<E: Equation> SaemState<E> {
    pub(super) fn markov_variance_diagnostics(
        &self,
        estimator: &SaemEstimatorMetadata,
        information: &InformationDiagnostics,
    ) -> MarkovSimulationVarianceDiagnostics {
        self.markov_variance_diagnostics_with_seed(estimator, information, None, None)
    }

    /// Frozen-kernel diagnostic with an optional deterministic seed override.
    ///
    /// The override gives each operational checkpoint its own deterministic
    /// stream; `None` preserves the exact
    /// post-fit path seeded by the diagnostic configuration.
    pub(super) fn markov_variance_diagnostics_with_seed(
        &self,
        estimator: &SaemEstimatorMetadata,
        information: &InformationDiagnostics,
        seed_override: Option<u64>,
        candidate: Option<&DiagnosticCandidate>,
    ) -> MarkovSimulationVarianceDiagnostics {
        let Some(config) = self.config.markov_simulation_variance else {
            return MarkovSimulationVarianceDiagnostics::disabled();
        };
        let diagnostic_seed = seed_override.unwrap_or(config.seed);
        let cd = config.diagnostic_chains;
        let cf = self.initialization.n_chains;
        let mut diagnostic = MarkovSimulationVarianceDiagnostics {
            config: Some(config),
            coordinates: information.coordinates.clone(),
            chain_count: cd,
            n_avg: estimator.averaged_iterations,
            chains: Vec::new(),
            grand_score_mean: Vec::new(),
            lambda: Vec::new(),
            lambda_status: MarkovSimulationVarianceStatus::AssumptionsUnverified,
            xi: Vec::new(),
            xi_status: MarkovSimulationVarianceStatus::AssumptionsUnverified,
            simulation_covariance: Vec::new(),
            simulation_covariance_status: MarkovSimulationVarianceStatus::AssumptionsUnverified,
            status: MarkovSimulationVarianceStatus::AssumptionsUnverified,
            assumptions: MARKOV_VARIANCE_ASSUMPTIONS.into(),
            rank_diagnostics: RankMixingDiagnostics {
                diagnostic_chains: cd,
                draws_per_chain: config.draws_per_chain,
                original_chains: cf,
                traces: Vec::new(),
                lrv_per_chain: Vec::new(),
                lrv_chain_statuses: Vec::new(),
                diagnostic_mean_lrv: None,
                operational_lrv: None,
                max_trace_bytes: 0,
                accounted_peak_trace_bytes_required: 0,
                accounted_peak_trace_bytes_used: 0,
                worst_rhat: None,
                min_bulk_ess: None,
                min_avg_ess_per_split_chain: None,
                assumptions: MARKOV_VARIANCE_ASSUMPTIONS.into(),
                status: RankDiagnosticStatus::Disabled,
            },
        };
        let width = information.coordinates.len();
        let information_eligible = estimator.average_applied
            && matches!(information.status, InformationStatus::Available)
            && width > 0;
        let observed_information = if information_eligible {
            match matrix_from_rows(&information.observed_information, width) {
                Ok(matrix) => Some(matrix),
                Err(_) => {
                    diagnostic.xi_status = MarkovSimulationVarianceStatus::CoordinateMismatch;
                    None
                }
            }
        } else {
            None
        };
        if self.initialization.random_effect_indices.is_empty()
            && self.initialization.iov_effect_indices.is_empty()
        {
            let zero = Array2::zeros((width, width));
            diagnostic.lambda = rows(&zero);
            diagnostic.lambda_status = MarkovSimulationVarianceStatus::ExactZeroNoLatentState;
            diagnostic.xi = rows(&zero);
            diagnostic.xi_status = MarkovSimulationVarianceStatus::ExactZeroNoLatentState;
            diagnostic.simulation_covariance = rows(&zero);
            diagnostic.simulation_covariance_status =
                MarkovSimulationVarianceStatus::ExactZeroNoLatentState;
            diagnostic.status = MarkovSimulationVarianceStatus::ExactZeroNoLatentState;
            diagnostic.rank_diagnostics.status = RankDiagnosticStatus::NoLatent;
            diagnostic
                .rank_diagnostics
                .lrv_chain_statuses
                .fill(RankDiagnosticStatus::NoLatent);
            diagnostic.rank_diagnostics.max_trace_bytes = config.max_trace_bytes;
            return diagnostic;
        }

        // ── Pre-execution byte-cap check (checked) ───────────────────────
        let trace_shape = self
            .initialization
            .random_effect_indices
            .len()
            .checked_mul(self.initialization.subject_ids.len())
            .and_then(|n_eta| {
                self.initialization
                    .occasion_counts
                    .iter()
                    .try_fold(0usize, |total, count| total.checked_add(*count))
                    .and_then(|occasions| {
                        occasions
                            .checked_mul(self.initialization.iov_effect_indices.len())
                            .and_then(|n_kappa| width.checked_add(n_eta)?.checked_add(n_kappa))
                    })
            });
        let Some(n_traces) = trace_shape else {
            mark_diagnostic_failure(
                &mut diagnostic,
                RankDiagnosticStatus::TraceMemoryAccountingOverflow,
                MarkovSimulationVarianceStatus::TraceMemoryAccountingOverflow,
            );
            diagnostic.rank_diagnostics.max_trace_bytes = config.max_trace_bytes;
            return diagnostic;
        };
        // Deterministic requested-capacity accounting. `traces` is nested
        // coordinate-major storage, so its heap-resident Vec headers count in
        // addition to every f64 leaf payload. The peak adds the larger of:
        // (a) one nested draw-major score view, or (b) a conservative upper
        // bound for the live rank/folding/ESS workspaces. The latter is eight
        // payload-widths per retained draw (including the 24-byte ranked tuple)
        // plus sixteen Vec headers per chain. This upper-bounds all capacities
        // explicitly requested by the current rank helpers; allocator metadata
        // and allocator size-class rounding are intentionally not claimed.
        let vec_header = std::mem::size_of::<Vec<f64>>();
        let f64_bytes = std::mem::size_of::<f64>();
        let accounted = cd
            .checked_mul(config.draws_per_chain)
            .and_then(|samples_per_coordinate| {
                samples_per_coordinate
                    .checked_mul(n_traces)
                    .and_then(|values| values.checked_mul(f64_bytes))
                    .and_then(|leaf_payload| {
                        n_traces
                            .checked_mul(cd)
                            .and_then(|headers| headers.checked_mul(vec_header))
                            .and_then(|leaf_headers| leaf_payload.checked_add(leaf_headers))
                    })
                    .and_then(|bytes| {
                        n_traces
                            .checked_mul(vec_header)
                            .and_then(|middle_headers| bytes.checked_add(middle_headers))
                    })
                    .and_then(|persistent_bytes| {
                        config
                            .draws_per_chain
                            .checked_mul(width)
                            .and_then(|values| values.checked_mul(f64_bytes))
                            .and_then(|payload| {
                                config
                                    .draws_per_chain
                                    .checked_mul(vec_header)
                                    .and_then(|headers| payload.checked_add(headers))
                            })
                            .and_then(|score_transient_bytes| {
                                samples_per_coordinate
                                    .checked_mul(8 * f64_bytes)
                                    .and_then(|payload| {
                                        cd.checked_mul(16)
                                            .and_then(|headers| headers.checked_mul(vec_header))
                                            .and_then(|headers| payload.checked_add(headers))
                                    })
                                    .and_then(|rank_transient_bytes| {
                                        persistent_bytes
                                            .checked_add(
                                                score_transient_bytes.max(rank_transient_bytes),
                                            )
                                            .map(|required_bytes| {
                                                (
                                                    persistent_bytes,
                                                    score_transient_bytes,
                                                    required_bytes,
                                                )
                                            })
                                    })
                            })
                    })
            });
        let Some((persistent_bytes, score_transient_bytes, required_bytes)) = accounted else {
            mark_diagnostic_failure(
                &mut diagnostic,
                RankDiagnosticStatus::TraceMemoryAccountingOverflow,
                MarkovSimulationVarianceStatus::TraceMemoryAccountingOverflow,
            );
            diagnostic.rank_diagnostics.max_trace_bytes = config.max_trace_bytes;
            return diagnostic;
        };
        diagnostic
            .rank_diagnostics
            .accounted_peak_trace_bytes_required = required_bytes;
        diagnostic.rank_diagnostics.max_trace_bytes = config.max_trace_bytes;
        if required_bytes > config.max_trace_bytes {
            mark_diagnostic_failure(
                &mut diagnostic,
                RankDiagnosticStatus::TraceByteCapExceeded,
                MarkovSimulationVarianceStatus::InvalidConfiguration(format!(
                    "diagnostic trace accounted peak requires {required_bytes} bytes, exceeding cap {}",
                    config.max_trace_bytes
                )),
            );
            return diagnostic;
        }
        diagnostic.rank_diagnostics.lrv_per_chain = vec![None; cd];
        diagnostic.rank_diagnostics.lrv_chain_statuses =
            vec![RankDiagnosticStatus::Unavailable; cd];

        // ── Trace coordinate metadata ─────────────────────────────────────
        let mut trace_coords: Vec<DiagnosticTraceCoordinate> = Vec::with_capacity(n_traces);
        for coord in &information.coordinates {
            trace_coords.push(DiagnosticTraceCoordinate::Score {
                index: coord.index,
                name: coord.name.clone(),
                kind: coord.kind.clone(),
            });
        }
        for subject_id in &self.initialization.subject_ids {
            for (eff_idx, name) in self.initialization.random_effect_names.iter().enumerate() {
                trace_coords.push(DiagnosticTraceCoordinate::Eta {
                    subject: subject_id.clone(),
                    effect_index: eff_idx,
                    effect_name: name.clone(),
                });
            }
        }
        if !self.initialization.iov_effect_indices.is_empty() {
            for (subject_idx, subject_id) in self.initialization.subject_ids.iter().enumerate() {
                for occasion in self.data.subjects()[subject_idx].occasions() {
                    for (eff_idx, name) in self.initialization.iov_effect_names.iter().enumerate() {
                        trace_coords.push(DiagnosticTraceCoordinate::Kappa {
                            subject: subject_id.clone(),
                            occasion_index: occasion.index(),
                            effect_index: eff_idx,
                            effect_name: name.clone(),
                        });
                    }
                }
            }
        }

        // ── Cd < 2 → still execute frozen chains, LRV, and Xi ────────────
        // Only per-trace rank diagnostics are unavailable (TooFewChains).
        let rank_possible = cd >= 2;

        // ── Fresh prior-drawn chains ──────────────────────────────────────
        let omega = candidate.map_or(&self.omega, |value| &value.omega);
        let omega_iov = candidate.map_or(self.omega_iov.as_ref(), |value| value.omega_iov.as_ref());
        let omega_lower = match cholesky_lower(omega) {
            Ok(lower) => lower,
            Err(_) => {
                mark_diagnostic_failure(
                    &mut diagnostic,
                    RankDiagnosticStatus::InvalidVariance,
                    MarkovSimulationVarianceStatus::Indefinite,
                );
                return diagnostic;
            }
        };
        let iov_lower = if self.initialization.iov_effect_indices.is_empty() {
            None
        } else {
            match omega_iov.map(cholesky_lower) {
                Some(Ok(lower)) => Some(lower),
                Some(Err(_)) | None => {
                    mark_diagnostic_failure(
                        &mut diagnostic,
                        RankDiagnosticStatus::InvalidVariance,
                        MarkovSimulationVarianceStatus::Indefinite,
                    );
                    return diagnostic;
                }
            }
        };

        // Canonical storage: [score_0..score_{w-1}, eta_0.., kappa_0..].
        // A draw-major score view is created one chain at a time for LRV and
        // released before the next chain.
        let mut traces: Vec<Vec<Vec<f64>>> = (0..n_traces)
            .map(|_| vec![Vec::with_capacity(config.draws_per_chain); cd])
            .collect();
        diagnostic.rank_diagnostics.accounted_peak_trace_bytes_used = persistent_bytes;
        let score_eligible =
            width > 0 && matches!(information.status, InformationStatus::Available);

        // Initialize Cd independent diagnostic chains with domain-separated seeds.
        // Seed derivation: per-chain seed = base.wrapping_add(i).wrapping_mul(GOLDEN_RATIO)
        // where GOLDEN_RATIO = 0x9E3779B97F4A7C15 (2^64 / φ) and base is the
        // configured diagnostic seed or the deterministic checkpoint override.
        let mut chain_states: Vec<FrozenDiagnosticState> = (0..cd)
            .map(|chain| {
                let chain_seed = diagnostic_seed
                    .wrapping_add(chain as u64)
                    .wrapping_mul(0x9E3779B97F4A7C15);
                let mut chain_rng = StdRng::seed_from_u64(chain_seed);
                FrozenDiagnosticState {
                    etas: self.draw_prior_etas(&omega_lower, &mut chain_rng),
                    kappas: self.draw_prior_kappas(iov_lower.as_deref(), &mut chain_rng),
                }
            })
            .collect();
        // Independent RNG streams for transitions (offset by +1 to separate
        // from prior-initialization streams).
        let mut chain_rngs: Vec<StdRng> = (0..cd)
            .map(|chain| {
                let chain_seed = diagnostic_seed
                    .wrapping_add(chain as u64)
                    .wrapping_mul(0x9E3779B97F4A7C15)
                    .wrapping_add(1);
                StdRng::seed_from_u64(chain_seed)
            })
            .collect();
        let mut chain_counts = vec![(0usize, 0usize, 0usize); cd];

        // ── Warmup ────────────────────────────────────────────────────────
        for _ in 0..config.warmup_transitions {
            for chain in 0..cd {
                let mut single = [chain_counts[chain]];
                if self
                    .frozen_diagnostic_transition(
                        &mut chain_states[chain],
                        &mut chain_rngs[chain],
                        &mut single,
                        candidate,
                    )
                    .is_err()
                {
                    mark_diagnostic_failure(
                        &mut diagnostic,
                        RankDiagnosticStatus::Unavailable,
                        MarkovSimulationVarianceStatus::UnsupportedScore(
                            "frozen diagnostic warmup transition failed".into(),
                        ),
                    );
                    return diagnostic;
                }
                chain_counts[chain] = single[0];
            }
        }
        begin_retained_transition_accounting(&mut chain_counts);

        // ── Single retained-draw pass: transition → collect traces ──────
        for _ in 0..config.draws_per_chain {
            for chain in 0..cd {
                let mut single = [chain_counts[chain]];
                if self
                    .frozen_diagnostic_transition(
                        &mut chain_states[chain],
                        &mut chain_rngs[chain],
                        &mut single,
                        candidate,
                    )
                    .is_err()
                {
                    mark_diagnostic_failure(
                        &mut diagnostic,
                        RankDiagnosticStatus::Unavailable,
                        MarkovSimulationVarianceStatus::UnsupportedScore(
                            "frozen retained diagnostic transition failed".into(),
                        ),
                    );
                    return diagnostic;
                }
                chain_counts[chain] = single[0];

                // Score failure never discards independently valid latent draws.
                let score = if score_eligible {
                    match self.frozen_complete_score(&chain_states[chain], 0, candidate) {
                        Ok(values) if values.len() == width => Some(values),
                        Ok(_) | Err(_) => None,
                    }
                } else {
                    None
                };
                for coord_idx in 0..width {
                    traces[coord_idx][chain]
                        .push(score.as_ref().map_or(f64::NAN, |values| values[coord_idx]));
                }

                // Collect eta coordinates: subject-major, coordinate-major.
                let mut trace_idx = width;
                for subject_etas in &chain_states[chain].etas {
                    for eta_coord in &subject_etas[0] {
                        traces[trace_idx][chain].push(*eta_coord);
                        trace_idx += 1;
                    }
                }

                // Collect kappa coordinates.
                for subject_kappas in &chain_states[chain].kappas {
                    for kappa_vec in &subject_kappas[0] {
                        for kappa_coord in kappa_vec {
                            traces[trace_idx][chain].push(*kappa_coord);
                            trace_idx += 1;
                        }
                    }
                }
            }
        }

        // Preserve the raw grand complete-score mean used by the invariant
        // stationarity diagnostic. Any non-finite score leaves it unavailable.
        if score_eligible {
            let denominator = (cd * config.draws_per_chain) as f64;
            let means = (0..width)
                .map(|coordinate| {
                    traces[coordinate].iter().flatten().copied().sum::<f64>() / denominator
                })
                .collect::<Vec<_>>();
            if means.iter().all(|value| value.is_finite()) {
                diagnostic.grand_score_mean = means;
            }
        }

        // ── Per-chain score LRV from transient draw-major views ─────────
        let mut lrv_matrices: Vec<Option<Array2<f64>>> = Vec::with_capacity(cd);
        for chain in 0..cd {
            let (proposals, accepts, state_changes) = chain_counts[chain];
            let score_view = (0..config.draws_per_chain)
                .map(|draw| (0..width).map(|coord| traces[coord][chain][draw]).collect())
                .collect::<Vec<Vec<f64>>>();
            diagnostic.rank_diagnostics.accounted_peak_trace_bytes_used = persistent_bytes
                .checked_add(score_transient_bytes)
                .unwrap_or(required_bytes);
            let lrv_result = if score_eligible {
                match lugsail_batch_means(&score_view, config.batch_size, config.lugsail) {
                    Ok(value) => Some(value),
                    Err(_) => {
                        diagnostic.xi_status = MarkovSimulationVarianceStatus::UnsupportedScore(
                            "per-chain score LRV failed".into(),
                        );
                        None
                    }
                }
            } else {
                None
            };
            if let Some((coarse, fine, lrv)) = lrv_result {
                let classification = classify_psd(&lrv);
                let lrv_status = markov_matrix_status(classification);
                diagnostic
                    .chains
                    .push(MarkovSimulationVarianceChainDiagnostics {
                        chain,
                        bm_batch: rows(&coarse),
                        bm_batch_over_r: rows(&fine),
                        lugsail_lrv: rows(&lrv),
                        status: lrv_status,
                        proposals,
                        accepts,
                        state_changes,
                    });
                diagnostic.rank_diagnostics.lrv_per_chain[chain] = Some(rows(&lrv));
                diagnostic.rank_diagnostics.lrv_chain_statuses[chain] = match classification {
                    MatrixClassification::EligiblePsd => RankDiagnosticStatus::Available,
                    MatrixClassification::NonFinite => RankDiagnosticStatus::NonFiniteDraws,
                    MatrixClassification::NonSymmetric | MatrixClassification::Indefinite => {
                        RankDiagnosticStatus::InvalidVariance
                    }
                };
                lrv_matrices.push(Some(lrv));
            } else {
                diagnostic
                    .chains
                    .push(MarkovSimulationVarianceChainDiagnostics {
                        chain,
                        bm_batch: Vec::new(),
                        bm_batch_over_r: Vec::new(),
                        lugsail_lrv: Vec::new(),
                        status: MarkovSimulationVarianceStatus::UnsupportedScore(
                            "complete-score trace or information unavailable".into(),
                        ),
                        proposals,
                        accepts,
                        state_changes,
                    });
                diagnostic.rank_diagnostics.lrv_per_chain[chain] = None;
                diagnostic.rank_diagnostics.lrv_chain_statuses[chain] =
                    RankDiagnosticStatus::ScoreUnavailable;
                lrv_matrices.push(None);
            }
        }

        let stuck_chain = chain_counts
            .iter()
            .enumerate()
            .find(|(_, count)| count.2 == 0)
            .map(|(chain, _)| chain);

        // Aggregate only when every chain has an eligible score LRV.
        let all_lrvs_available = lrv_matrices.len() == cd
            && lrv_matrices.iter().all(Option::is_some)
            && diagnostic
                .rank_diagnostics
                .lrv_chain_statuses
                .iter()
                .all(|status| matches!(status, RankDiagnosticStatus::Available));
        if all_lrvs_available {
            let mut lrv_sum = Array2::zeros((width, width));
            for lrv in &lrv_matrices {
                lrv_sum += lrv
                    .as_ref()
                    .expect("all per-chain LRV matrices were checked available");
            }
            let (diag_mean, operational) = scale_lrv_sum(&lrv_sum, cd, cf);
            diagnostic.rank_diagnostics.diagnostic_mean_lrv = Some(rows(&diag_mean));
            diagnostic.lambda = rows(&diag_mean);
            diagnostic.lambda_status = markov_matrix_status(classify_psd(&diag_mean));

            // Cd != Cf is intentional: operational scale is Σ/(Cd*Cf).
            diagnostic.rank_diagnostics.operational_lrv = Some(rows(&operational));
            if let Some(observed_information) = observed_information.as_ref() {
                match transform_simulation_variance(
                    observed_information,
                    &operational,
                    estimator.averaged_iterations,
                ) {
                    Ok((xi, covariance)) => {
                        diagnostic.xi = rows(&xi);
                        diagnostic.xi_status = markov_matrix_status(classify_psd(&xi));
                        diagnostic.simulation_covariance = rows(&covariance);
                        diagnostic.simulation_covariance_status =
                            markov_matrix_status(classify_psd(&covariance));
                    }
                    Err(_) => {
                        diagnostic.xi_status = MarkovSimulationVarianceStatus::NonFinite;
                        diagnostic.simulation_covariance_status =
                            MarkovSimulationVarianceStatus::NonFinite;
                    }
                }
            } else {
                diagnostic.xi_status = MarkovSimulationVarianceStatus::InformationUnavailable(
                    format!("{:?}", information.status),
                );
                diagnostic.simulation_covariance_status = diagnostic.xi_status.clone();
            }
        } else {
            let failure = if !score_eligible {
                MarkovSimulationVarianceStatus::InformationUnavailable(format!(
                    "{:?}",
                    information.status
                ))
            } else {
                diagnostic
                    .chains
                    .iter()
                    .map(|chain| &chain.status)
                    .find(|status| {
                        !matches!(
                            status,
                            MarkovSimulationVarianceStatus::AssumptionsUnverified
                        )
                    })
                    .cloned()
                    .unwrap_or_else(|| {
                        MarkovSimulationVarianceStatus::UnsupportedScore(
                            "one or more configured diagnostic-chain score LRVs failed".into(),
                        )
                    })
            };
            diagnostic.lambda_status = failure.clone();
            diagnostic.xi_status = failure.clone();
            diagnostic.simulation_covariance_status = failure;
        }

        // ── Rank/mixing diagnostics from traces ─────────────────────────
        // The prechecked rank workspace is the accounted peak whenever rank
        // diagnostics execute; no allocator-specific byte claim is made.
        if rank_possible {
            diagnostic.rank_diagnostics.accounted_peak_trace_bytes_used = required_bytes;
            diagnostic.rank_diagnostics.traces =
                self.rank_diagnostics_from_traces(cd, &traces, &trace_coords);
        } else {
            diagnostic.rank_diagnostics.traces = trace_coords
                .iter()
                .map(|coord| RankMixingDiagnostic {
                    trace: coord.clone(),
                    rank_rhat: None,
                    rank_rhat_status: RankDiagnosticStatus::TooFewChains,
                    folded_rhat: None,
                    folded_rhat_status: RankDiagnosticStatus::TooFewChains,
                    max_rhat: None,
                    max_rhat_status: RankDiagnosticStatus::TooFewChains,
                    bulk_ess: None,
                    bulk_ess_status: RankDiagnosticStatus::TooFewChains,
                    avg_ess_per_split_chain: None,
                    tau: None,
                    status: RankDiagnosticStatus::TooFewChains,
                })
                .collect();
        }

        // ── Aggregate per-coordinate worst/min across traces ────────────
        diagnostic.rank_diagnostics.worst_rhat =
            worst_valid_max_rhat(&diagnostic.rank_diagnostics.traces);
        diagnostic.rank_diagnostics.min_bulk_ess = diagnostic
            .rank_diagnostics
            .traces
            .iter()
            .filter_map(|t| t.bulk_ess)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        diagnostic.rank_diagnostics.min_avg_ess_per_split_chain = diagnostic
            .rank_diagnostics
            .traces
            .iter()
            .filter_map(|t| t.avg_ess_per_split_chain)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Aggregate status: ineligible if any coordinate or LRV is non-available.
        let any_coord_non_available = diagnostic
            .rank_diagnostics
            .traces
            .iter()
            .any(|t| !matches!(t.status, RankDiagnosticStatus::Available));
        let any_lrv_non_available = diagnostic
            .rank_diagnostics
            .lrv_chain_statuses
            .iter()
            .any(|s| !matches!(s, RankDiagnosticStatus::Available));
        if rank_possible
            && (any_coord_non_available || any_lrv_non_available || stuck_chain.is_some())
        {
            diagnostic.rank_diagnostics.status = if diagnostic
                .rank_diagnostics
                .traces
                .iter()
                .any(|trace| matches!(trace.status, RankDiagnosticStatus::Available))
            {
                RankDiagnosticStatus::PartialAvailability
            } else {
                RankDiagnosticStatus::Unavailable
            };
        } else if !rank_possible {
            diagnostic.rank_diagnostics.status = RankDiagnosticStatus::TooFewChains;
        } else {
            diagnostic.rank_diagnostics.status = RankDiagnosticStatus::Available;
        }

        // ── Final aggregate markov status ───────────────────────────────
        diagnostic.status = if let Some(chain) = stuck_chain {
            MarkovSimulationVarianceStatus::StuckChain { chain }
        } else if !estimator.average_applied {
            MarkovSimulationVarianceStatus::AverageNotApplied
        } else if !matches!(information.status, InformationStatus::Available) {
            MarkovSimulationVarianceStatus::InformationUnavailable(format!(
                "{:?}",
                information.status
            ))
        } else {
            diagnostic
                .chains
                .iter()
                .map(|chain| &chain.status)
                .chain([
                    &diagnostic.lambda_status,
                    &diagnostic.xi_status,
                    &diagnostic.simulation_covariance_status,
                ])
                .find(|status| {
                    !matches!(
                        status,
                        MarkovSimulationVarianceStatus::AssumptionsUnverified
                    )
                })
                .cloned()
                .unwrap_or(MarkovSimulationVarianceStatus::AssumptionsUnverified)
        };
        diagnostic
    }

    /// Build per-coordinate rank/mixing diagnostics from collected trace chains.
    pub(super) fn rank_diagnostics_from_traces(
        &self,
        cd: usize,
        traces: &[Vec<Vec<f64>>],
        trace_coords: &[DiagnosticTraceCoordinate],
    ) -> Vec<RankMixingDiagnostic> {
        trace_coords
            .iter()
            .enumerate()
            .map(|(idx, coord)| {
                let chains = &traces[idx];
                let rank_result = rank_normalized_split_rhat(chains);
                let folded_result = folded_split_rhat(chains);
                let ess_result = bulk_ess(chains);

                let rank_rhat = match rank_result.as_ref() {
                    Ok(value) => Some(*value),
                    Err(_) => None,
                };
                let folded_rhat = match folded_result.as_ref() {
                    Ok(value) => Some(*value),
                    Err(_) => None,
                };
                let (bulk_ess, tau) = match ess_result.as_ref() {
                    Ok((ess, tau)) => (Some(*ess), Some(*tau)),
                    Err(_) => (None, None),
                };
                let avg_ess_per_split_chain = bulk_ess.map(|ess| ess / (2.0 * cd as f64));

                let score_unavailable = matches!(coord, DiagnosticTraceCoordinate::Score { .. })
                    && chains.iter().flatten().any(|draw| !draw.is_finite());
                let statistic_status = |result: Result<(), &RankDiagnosticError>| {
                    if score_unavailable {
                        RankDiagnosticStatus::ScoreUnavailable
                    } else {
                        result
                            .map(|()| RankDiagnosticStatus::Available)
                            .unwrap_or_else(rank_diagnostic_error_status)
                    }
                };
                let rank_rhat_status = statistic_status(rank_result.as_ref().map(|_| ()));
                let folded_rhat_status = statistic_status(folded_result.as_ref().map(|_| ()));
                let max_rhat = match (rank_rhat, folded_rhat) {
                    (Some(rank), Some(folded)) => Some(rank.max(folded)),
                    _ => None,
                };
                let max_rhat_status = if matches!(rank_rhat_status, RankDiagnosticStatus::Available)
                    && matches!(folded_rhat_status, RankDiagnosticStatus::Available)
                {
                    RankDiagnosticStatus::Available
                } else if !matches!(rank_rhat_status, RankDiagnosticStatus::Available) {
                    rank_rhat_status.clone()
                } else {
                    folded_rhat_status.clone()
                };
                let bulk_ess_status = statistic_status(ess_result.as_ref().map(|_| ()));
                let statuses = [&rank_rhat_status, &folded_rhat_status, &bulk_ess_status];
                let available = statuses
                    .iter()
                    .filter(|status| matches!(status, RankDiagnosticStatus::Available))
                    .count();
                let status = if available == statuses.len() {
                    RankDiagnosticStatus::Available
                } else if available > 0 {
                    RankDiagnosticStatus::PartialAvailability
                } else if statuses.iter().all(|status| *status == statuses[0]) {
                    statuses[0].clone()
                } else {
                    RankDiagnosticStatus::Unavailable
                };

                RankMixingDiagnostic {
                    trace: coord.clone(),
                    rank_rhat,
                    rank_rhat_status,
                    folded_rhat,
                    folded_rhat_status,
                    max_rhat,
                    max_rhat_status,
                    bulk_ess,
                    bulk_ess_status,
                    avg_ess_per_split_chain,
                    tau,
                    status,
                }
            })
            .collect()
    }

    /// Draw initial η vectors from N(0, Omega) for fresh diagnostic chains.
    pub(super) fn draw_prior_etas(
        &self,
        omega_lower: &[Vec<f64>],
        rng: &mut StdRng,
    ) -> Vec<Vec<Vec<f64>>> {
        let n_eta = self.initialization.random_effect_indices.len();
        if n_eta == 0 {
            return vec![vec![Vec::new(); 1]; self.initialization.subject_ids.len()];
        }
        self.initialization
            .subject_ids
            .iter()
            .map(|_| {
                let normals: Vec<f64> = (0..n_eta)
                    .map(|_| diagnostic_standard_normal(rng))
                    .collect();
                let eta = (0..n_eta)
                    .map(|row| {
                        (0..=row)
                            .map(|col| omega_lower[row][col] * normals[col])
                            .sum()
                    })
                    .collect::<Vec<_>>();
                vec![eta]
            })
            .collect()
    }

    /// Draw initial κ vectors from N(0, Omega_IOV) for fresh diagnostic chains.
    pub(super) fn draw_prior_kappas(
        &self,
        iov_lower: Option<&[Vec<f64>]>,
        rng: &mut StdRng,
    ) -> Vec<Vec<Vec<Vec<f64>>>> {
        let Some(iov_lower) = iov_lower else {
            return vec![vec![Vec::new(); 1]; self.initialization.subject_ids.len()];
        };
        let n_kappa = self.initialization.iov_effect_indices.len();
        self.initialization
            .occasion_counts
            .iter()
            .map(|&n_occasions| {
                let kappas: Vec<Vec<f64>> = (0..n_occasions)
                    .map(|_| {
                        let normals: Vec<f64> = (0..n_kappa)
                            .map(|_| diagnostic_standard_normal(rng))
                            .collect();
                        (0..n_kappa)
                            .map(|row| {
                                (0..=row)
                                    .map(|col| iov_lower[row][col] * normals[col])
                                    .sum()
                            })
                            .collect()
                    })
                    .collect();
                vec![kappas]
            })
            .collect()
    }

    pub(super) fn frozen_diagnostic_transition(
        &self,
        state: &mut FrozenDiagnosticState,
        rng: &mut StdRng,
        counts: &mut [(usize, usize, usize)],
        candidate: Option<&DiagnosticCandidate>,
    ) -> std::result::Result<(), String> {
        for _ in 0..self.eta_block_iterations {
            for subject in 0..self.initialization.subject_ids.len() {
                let omega = candidate.map_or(&self.omega, |value| &value.omega);
                let lower = cholesky_lower(omega).map_err(|error| error.to_string())?;
                for (chain, count) in counts.iter_mut().enumerate() {
                    let current = state.etas[subject][chain].clone();
                    let normals = (0..current.len())
                        .map(|_| diagnostic_standard_normal(rng))
                        .collect::<Vec<_>>();
                    let proposed = correlated_random_walk(
                        &current,
                        &lower,
                        &normals,
                        self.eta_block_step_sizes[subject],
                    )
                    .map_err(|error| error.to_string())?;
                    let current_score = self
                        .score_subject_latents_at(
                            subject,
                            &current,
                            &state.kappas[subject][chain],
                            candidate,
                        )
                        .map_err(|error| error.to_string())?;
                    let proposed_score = self
                        .score_subject_latents_at(
                            subject,
                            &proposed,
                            &state.kappas[subject][chain],
                            candidate,
                        )
                        .map_err(|error| error.to_string())?;
                    count.0 += 1;
                    if diagnostic_accept(rng, current_score.log_acceptance_ratio(proposed_score)) {
                        count.1 += 1;
                        if proposed != current {
                            count.2 += 1;
                        }
                        state.etas[subject][chain] = proposed;
                    }
                }
            }
        }
        for _ in 0..self.mcmc_iterations {
            for subject in 0..self.initialization.subject_ids.len() {
                for (chain, count) in counts.iter_mut().enumerate() {
                    for parameter in 0..self.initialization.random_effect_indices.len() {
                        let current = state.etas[subject][chain].clone();
                        let mut proposed = current.clone();
                        proposed[parameter] +=
                            self.proposal_step_sizes[parameter] * diagnostic_standard_normal(rng);
                        let current_score = self
                            .score_subject_latents_at(
                                subject,
                                &current,
                                &state.kappas[subject][chain],
                                candidate,
                            )
                            .map_err(|error| error.to_string())?;
                        let proposed_score = self
                            .score_subject_latents_at(
                                subject,
                                &proposed,
                                &state.kappas[subject][chain],
                                candidate,
                            )
                            .map_err(|error| error.to_string())?;
                        count.0 += 1;
                        if diagnostic_accept(
                            rng,
                            current_score.log_acceptance_ratio(proposed_score),
                        ) {
                            count.1 += 1;
                            if proposed != current {
                                count.2 += 1;
                            }
                            state.etas[subject][chain] = proposed;
                        }
                    }
                    let omega_iov =
                        candidate.map_or(self.omega_iov.as_ref(), |value| value.omega_iov.as_ref());
                    if let Some(omega_iov) = omega_iov {
                        let lower = cholesky_lower(omega_iov).map_err(|error| error.to_string())?;
                        for occasion in 0..state.kappas[subject][chain].len() {
                            let current = state.kappas[subject][chain][occasion].clone();
                            let normals = (0..current.len())
                                .map(|_| diagnostic_standard_normal(rng))
                                .collect::<Vec<_>>();
                            let proposed = correlated_random_walk(
                                &current,
                                &lower,
                                &normals,
                                self.kappa_proposal_step_sizes[subject],
                            )
                            .map_err(|error| error.to_string())?;
                            let current_score = self
                                .score_subject_latents_at(
                                    subject,
                                    &state.etas[subject][chain],
                                    &state.kappas[subject][chain],
                                    candidate,
                                )
                                .map_err(|error| error.to_string())?;
                            let mut proposed_kappas = state.kappas[subject][chain].clone();
                            proposed_kappas[occasion] = proposed.clone();
                            let proposed_score = self
                                .score_subject_latents_at(
                                    subject,
                                    &state.etas[subject][chain],
                                    &proposed_kappas,
                                    candidate,
                                )
                                .map_err(|error| error.to_string())?;
                            count.0 += 1;
                            if diagnostic_accept(
                                rng,
                                current_score.log_acceptance_ratio(proposed_score),
                            ) {
                                count.1 += 1;
                                if proposed != current {
                                    count.2 += 1;
                                }
                                state.kappas[subject][chain][occasion] = proposed;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    // ─── Operational convergence ─────────────────────────────────────────

    /// Evaluate an operational convergence checkpoint if one is due.
    pub(super) fn evaluate_operational_convergence(
        &mut self,
        iteration: usize,
        scheduled: bool,
        mandatory_final: bool,
    ) -> Result<()> {
        let Some(settings) = self.operational_settings else {
            return Ok(());
        };
        // Only check during smoothing, unless this is a mandatory final check.
        if !mandatory_final && self.initialization.schedule.phase(iteration) != SaemPhase::Smoothing
        {
            return Ok(());
        }
        let Some(ref average) = self.iterate_average else {
            return Ok(());
        };
        let n_averaged = average.count;
        if n_averaged < settings.first_eligible_averaged_iteration {
            return Ok(());
        }

        // Cadence: periodic checkpoints are evaluated every check_interval
        // iterations starting from first_eligible_averaged_iteration.
        if scheduled && !mandatory_final {
            let smoothing_start = self.initialization.schedule.pure_burn_in
                + self.initialization.schedule.exploration_iterations
                + 1;
            let smoothing_offset = iteration.saturating_sub(smoothing_start) + 1;
            if smoothing_offset < settings.first_eligible_averaged_iteration
                || !(smoothing_offset - settings.first_eligible_averaged_iteration)
                    .is_multiple_of(settings.check_interval)
            {
                return Ok(());
            }
        }

        // Defensive caching: if this is a mandatory final check and we already
        // evaluated at this iteration, reuse instead of rerunning.
        if mandatory_final {
            if let Some(last) = self.operational_diagnostics.checks.last() {
                if last.iteration == iteration {
                    self.operational_diagnostics.final_check_reused = true;
                    return Ok(());
                }
            }
        }

        // Build the deterministic per-checkpoint seed.
        let checkpoint_seed = self
            .config
            .markov_simulation_variance
            .expect("operational policy validation requires Markov diagnostics")
            .seed
            .wrapping_add(OPERATIONAL_CHECKPOINT_SEED_DOMAIN)
            .wrapping_add(iteration as u64);

        // Two-sided standard normal quantile.
        let z_quantile = normal_two_sided_z(settings.confidence_level);

        let implied_averaged_iterations =
            Some(4.0 * z_quantile * z_quantile / settings.relative_fixed_width_epsilon.powi(2));

        let info = self.information.diagnostics();
        let avg_psi = match population_psi(
            &average.population_phi,
            &self.initialization.parameter_scales,
        ) {
            Ok(psi) => psi,
            Err(_) => {
                self.record_ineligible_checkpoint(
                    iteration,
                    n_averaged,
                    scheduled,
                    mandatory_final,
                    checkpoint_seed,
                    z_quantile,
                    implied_averaged_iterations,
                    Vec::new(),
                    "averaged population psi conversion failed".to_string(),
                );
                return Ok(());
            }
        };
        let mut candidate_error_models = self.error_models.clone();
        for (output_index, model) in &average.residual_models {
            match *model {
                ResidualErrorModel::Combined { a, b } => update_estimated_combined_residual_model(
                    &mut candidate_error_models,
                    *output_index,
                    a,
                    b,
                ),
                ResidualErrorModel::CorrelatedCombined { a, b, rho } => {
                    update_estimated_correlated_combined_residual_model(
                        &mut candidate_error_models,
                        *output_index,
                        a,
                        b,
                        rho,
                    )
                }
                ResidualErrorModel::Constant { .. }
                | ResidualErrorModel::Proportional { .. }
                | ResidualErrorModel::Exponential { .. } => {
                    update_estimated_simple_residual_model_with_sigma(
                        &mut candidate_error_models,
                        *output_index,
                        primary_sigma_parameter(model),
                    )
                }
            }
        }
        let candidate_covariate_model = match (
            self.covariate_model.as_ref(),
            average.covariate_betas.as_ref(),
        ) {
            (Some(model), Some(values)) => Some(model.with_estimates(values)?),
            (None, None) => None,
            _ => anyhow::bail!("averaged covariate metadata dimension mismatch"),
        };
        let candidate = DiagnosticCandidate {
            population_parameters: avg_psi,
            covariate_model: candidate_covariate_model,
            omega: average.omega.clone(),
            omega_iov: average.omega_iov.clone(),
            error_models: candidate_error_models,
        };
        let candidate_free_coordinates = match operational_free_coordinates(&info, average) {
            Ok(values) if !values.is_empty() => values,
            Ok(_) => {
                self.record_ineligible_checkpoint(
                    iteration,
                    n_averaged,
                    scheduled,
                    mandatory_final,
                    checkpoint_seed,
                    z_quantile,
                    implied_averaged_iterations,
                    Vec::new(),
                    "no free coordinates".to_string(),
                );
                return Ok(());
            }
            Err(error) => {
                self.record_ineligible_checkpoint(
                    iteration,
                    n_averaged,
                    scheduled,
                    mandatory_final,
                    checkpoint_seed,
                    z_quantile,
                    implied_averaged_iterations,
                    Vec::new(),
                    error.to_string(),
                );
                return Ok(());
            }
        };
        if self.initialization.random_effect_indices.is_empty()
            && self.initialization.iov_effect_indices.is_empty()
        {
            self.record_ineligible_checkpoint(
                iteration,
                n_averaged,
                scheduled,
                mandatory_final,
                checkpoint_seed,
                z_quantile,
                implied_averaged_iterations,
                candidate_free_coordinates,
                "no latent coordinates".to_string(),
            );
            return Ok(());
        }

        let diagnostic_metadata = SaemEstimatorMetadata {
            policy: self.config.estimator_policy,
            average_applied: true,
            averaging_start_cycle: Some(average.start_cycle),
            averaged_iterations: n_averaged,
        };

        let markov = self.markov_variance_diagnostics_with_seed(
            &diagnostic_metadata,
            &info,
            Some(checkpoint_seed),
            Some(&candidate),
        );

        let rank = &markov.rank_diagnostics;
        let simulation_sd_fraction = operational_simulation_sd_fraction(&info, &markov);
        let fixed_width = simulation_sd_fraction.map(|fraction| 2.0 * z_quantile * fraction);
        let fixed_width_ratio =
            fixed_width.map(|width| width / settings.relative_fixed_width_epsilon);
        let newton_value = newton_displacement(&info, &markov).filter(|value| value.is_finite());
        let newton_mc_sd =
            newton_displacement_mc_sd(&info, &markov).filter(|value| value.is_finite());
        let matrix_valid = matches!(info.status, InformationStatus::Available)
            && matches!(
                markov.lambda_status,
                MarkovSimulationVarianceStatus::AssumptionsUnverified
            )
            && matches!(
                markov.xi_status,
                MarkovSimulationVarianceStatus::AssumptionsUnverified
            )
            && matches!(
                markov.simulation_covariance_status,
                MarkovSimulationVarianceStatus::AssumptionsUnverified
            );
        let every_chain_moved =
            !markov.chains.is_empty() && markov.chains.iter().all(|chain| chain.state_changes > 0);
        let every_trace_valid = !rank.traces.is_empty()
            && rank.traces.iter().all(|trace| {
                trace.rank_rhat.is_some()
                    && trace.folded_rhat.is_some()
                    && trace.max_rhat.is_some()
                    && trace.bulk_ess.is_some()
                    && matches!(trace.rank_rhat_status, RankDiagnosticStatus::Available)
                    && matches!(trace.folded_rhat_status, RankDiagnosticStatus::Available)
                    && matches!(trace.max_rhat_status, RankDiagnosticStatus::Available)
                    && matches!(trace.bulk_ess_status, RankDiagnosticStatus::Available)
            });
        let covariance_policy = self
            .config
            .covariance_stability
            .expect("operational policy validation requires covariance stability");
        let omega_boundary = covariance_boundary_rejection_summary(
            &self.cycle_diagnostics,
            covariance_policy,
            false,
        );
        let omega_iov_boundary =
            covariance_boundary_rejection_summary(&self.cycle_diagnostics, covariance_policy, true);
        let covariance_active_cycles =
            iteration.saturating_sub(self.initialization.schedule.pure_burn_in);
        let covariance_window_available =
            covariance_active_cycles >= covariance_policy.rejection_window;
        let boundary_criterion = |name: &str, longest_run: usize| {
            if covariance_window_available {
                evaluate_criterion(
                    name,
                    Some(longest_run as f64),
                    covariance_policy.rejection_window as f64,
                    |observed| observed < covariance_policy.rejection_window as f64,
                )
            } else {
                OperationalConvergenceCriterion {
                    name: name.to_string(),
                    observed: Some(longest_run as f64),
                    threshold: covariance_policy.rejection_window as f64,
                    status: OperationalConvergenceCriterionStatus::Unavailable(format!(
                        "covariance-stability window requires {} active cycles; {covariance_active_cycles} completed",
                        covariance_policy.rejection_window
                    )),
                }
            }
        };
        let criteria: Vec<OperationalConvergenceCriterion> = vec![
            evaluate_criterion(
                "valid_information_and_matrices",
                Some(matrix_valid as u8 as f64),
                1.0,
                |value| value == 1.0,
            ),
            evaluate_criterion(
                "every_diagnostic_chain_moved",
                Some(every_chain_moved as u8 as f64),
                1.0,
                |value| value == 1.0,
            ),
            evaluate_criterion(
                "every_rank_diagnostic_valid",
                Some(every_trace_valid as u8 as f64),
                1.0,
                |value| value == 1.0,
            ),
            evaluate_criterion("max_rhat", rank.worst_rhat, settings.max_rhat, |observed| {
                observed < settings.max_rhat
            }),
            evaluate_criterion(
                "min_bulk_ess",
                rank.min_bulk_ess,
                settings.min_bulk_ess,
                |observed| observed > settings.min_bulk_ess,
            ),
            evaluate_criterion(
                "min_average_bulk_ess_per_split_chain",
                rank.min_avg_ess_per_split_chain,
                settings.min_average_bulk_ess_per_split_chain,
                |observed| observed >= settings.min_average_bulk_ess_per_split_chain,
            ),
            evaluate_criterion(
                "worst_simulation_sd_fraction",
                simulation_sd_fraction,
                settings.relative_fixed_width_epsilon / (2.0 * z_quantile),
                |observed| 2.0 * z_quantile * observed <= settings.relative_fixed_width_epsilon,
            ),
            evaluate_criterion(
                "relative_fixed_width",
                fixed_width,
                settings.relative_fixed_width_epsilon,
                |observed| observed <= settings.relative_fixed_width_epsilon,
            ),
            evaluate_criterion(
                "newton_displacement",
                newton_value,
                settings.max_newton_displacement,
                |observed| observed <= settings.max_newton_displacement,
            ),
            evaluate_criterion(
                "newton_displacement_mc_sd",
                newton_mc_sd,
                settings.max_newton_displacement_mc_sd,
                |observed| observed <= settings.max_newton_displacement_mc_sd,
            ),
            boundary_criterion("omega_boundary_rejection_run", omega_boundary.longest_run),
            boundary_criterion(
                "omega_iov_boundary_rejection_run",
                omega_iov_boundary.longest_run,
            ),
        ];

        let mut ineligible_reasons = criteria
            .iter()
            .filter_map(|criterion| match &criterion.status {
                OperationalConvergenceCriterionStatus::Unavailable(reason) => {
                    Some(format!("{}: {reason}", criterion.name))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if !matrix_valid {
            ineligible_reasons.push("information or matrix validation failed".to_string());
        }
        if !every_trace_valid {
            ineligible_reasons.push("one or more rank diagnostics unavailable".to_string());
        }
        if !every_chain_moved {
            ineligible_reasons
                .push("one or more retained diagnostic chains did not move".to_string());
        }
        let failed_criteria = criteria
            .iter()
            .filter(|criterion| {
                matches!(
                    criterion.status,
                    OperationalConvergenceCriterionStatus::NotSatisfied
                )
            })
            .map(|criterion| criterion.name.clone())
            .collect::<Vec<_>>();
        let outcome = if !ineligible_reasons.is_empty() {
            OperationalConvergenceOutcome::Ineligible {
                reasons: ineligible_reasons,
            }
        } else if !failed_criteria.is_empty() {
            OperationalConvergenceOutcome::Failed {
                criteria: failed_criteria,
            }
        } else {
            OperationalConvergenceOutcome::Passed
        };

        let passed = matches!(outcome, OperationalConvergenceOutcome::Passed);
        self.operational_diagnostics.final_status = Some(outcome.clone());
        self.operational_diagnostics.worst_rhat = rank.worst_rhat;
        self.operational_diagnostics.min_bulk_ess = rank.min_bulk_ess;
        self.operational_diagnostics.fixed_width_ratio = fixed_width_ratio;
        self.operational_diagnostics.fixed_width_epsilon =
            Some(settings.relative_fixed_width_epsilon);
        self.operational_diagnostics.implied_minimum_ess = implied_averaged_iterations;
        self.operational_diagnostics.newton_displacement = newton_value;
        self.operational_diagnostics.newton_displacement_mc_sd = newton_mc_sd;
        let checkpoint = OperationalConvergenceCheck {
            iteration,
            averaged_iterations: n_averaged,
            scheduled,
            mandatory_final,
            checkpoint_seed: Some(checkpoint_seed),
            z_quantile: Some(z_quantile),
            implied_minimum_ess: implied_averaged_iterations,
            candidate_free_coordinates,
            information: Some(info),
            criteria,
            outcome,
            markov: Some(markov),
        };

        self.operational_diagnostics.checks.push(checkpoint);

        // Terminate early if converged and this was a scheduled check.
        if passed {
            self.operational_diagnostics.used_for_termination = true;
            self.status = Status::Stop(StopReason::Converged);
        }

        Ok(())
    }

    /// Record an ineligible checkpoint (candidate unavailable).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_ineligible_checkpoint(
        &mut self,
        iteration: usize,
        averaged_iterations: usize,
        scheduled: bool,
        mandatory_final: bool,
        checkpoint_seed: u64,
        z_quantile: f64,
        implied_averaged_iterations: Option<f64>,
        candidate_free_coordinates: Vec<f64>,
        reason: String,
    ) {
        let settings = self
            .operational_settings
            .expect("ineligible operational checkpoint requires configured settings");
        let unavailable = |name: &str, threshold: f64| OperationalConvergenceCriterion {
            name: name.to_string(),
            observed: None,
            threshold,
            status: OperationalConvergenceCriterionStatus::Unavailable(reason.clone()),
        };
        let criteria = vec![
            unavailable("candidate_available", 1.0),
            unavailable("valid_information_and_matrices", 1.0),
            unavailable("every_diagnostic_chain_moved", 1.0),
            unavailable("every_rank_diagnostic_valid", 1.0),
            unavailable("max_rhat", settings.max_rhat),
            unavailable("min_bulk_ess", settings.min_bulk_ess),
            unavailable(
                "min_average_bulk_ess_per_split_chain",
                settings.min_average_bulk_ess_per_split_chain,
            ),
            unavailable(
                "worst_simulation_sd_fraction",
                settings.relative_fixed_width_epsilon / (2.0 * z_quantile),
            ),
            unavailable(
                "relative_fixed_width",
                settings.relative_fixed_width_epsilon,
            ),
            unavailable("newton_displacement", settings.max_newton_displacement),
            unavailable(
                "newton_displacement_mc_sd",
                settings.max_newton_displacement_mc_sd,
            ),
        ];
        let outcome = OperationalConvergenceOutcome::Ineligible {
            reasons: vec![reason],
        };
        self.operational_diagnostics.final_status = Some(outcome.clone());
        self.operational_diagnostics
            .checks
            .push(OperationalConvergenceCheck {
                iteration,
                averaged_iterations,
                scheduled,
                mandatory_final,
                checkpoint_seed: Some(checkpoint_seed),
                z_quantile: Some(z_quantile),
                implied_minimum_ess: implied_averaged_iterations,
                candidate_free_coordinates,
                information: None,
                criteria,
                outcome,
                markov: None,
            });
    }

    pub(super) fn frozen_complete_score(
        &self,
        state: &FrozenDiagnosticState,
        chain: usize,
        candidate: Option<&DiagnosticCandidate>,
    ) -> std::result::Result<Vec<f64>, String> {
        let layout = self.information.layout();
        let population_parameters = candidate
            .map_or(self.population_parameters.as_slice(), |value| {
                value.population_parameters.as_slice()
            });
        let omega = candidate.map_or(&self.omega, |value| &value.omega);
        let omega_iov = candidate.map_or(self.omega_iov.as_ref(), |value| value.omega_iov.as_ref());
        let error_models = candidate.map_or(&self.error_models, |value| &value.error_models);
        let mut derivative = CompleteDerivative::zero(layout.len());
        for subject_index in 0..self.initialization.subject_ids.len() {
            let covariate_model = candidate
                .and_then(|value| value.covariate_model.as_ref())
                .or(self.covariate_model.as_ref());
            match covariate_model {
                Some(model) => derivative.add_covariate_population_prior(
                    &state.etas[subject_index][chain],
                    omega,
                    &self.initialization.random_effect_indices,
                    model.parameter_indices(),
                    model.subject_design()[subject_index].values(),
                    layout,
                ),
                None => derivative.add_population_prior(
                    &state.etas[subject_index][chain],
                    omega,
                    &self.initialization.random_effect_indices,
                    layout,
                ),
            }
            .map_err(|error| error.to_string())?;
            let calculated_mu = if candidate.is_some() {
                covariate_model
                    .map(|model| {
                        let phi = population_phi(
                            population_parameters,
                            &self.initialization.parameter_scales,
                        )?;
                        Ok::<_, anyhow::Error>(
                            model.subject_population_parameters(
                                &phi,
                                &self.initialization.parameter_scales,
                            )?[subject_index]
                                .phi()
                                .to_vec(),
                        )
                    })
                    .transpose()
                    .map_err(|error| error.to_string())?
            } else {
                None
            };
            let subject_mu = calculated_mu.as_deref().or_else(|| {
                self.subject_mu_phi
                    .as_ref()
                    .map(|means| means[subject_index].as_slice())
            });
            let subject = self.data.subjects()[subject_index];
            if let Some(omega_iov) = omega_iov {
                let occasions = subject.occasions();
                let kappas = &state.kappas[subject_index][chain];
                if occasions.len() != kappas.len() {
                    return Err(format!(
                        "subject {} has {} occasions but {} diagnostic kappa states",
                        subject.id(),
                        occasions.len(),
                        kappas.len()
                    ));
                }
                for (occasion, kappa) in occasions.iter().zip(kappas) {
                    derivative
                        .add_iov_prior(kappa, omega_iov, layout)
                        .map_err(|error| error.to_string())?;
                    let parameters = match subject_mu {
                        Some(mean) => occasion_psi_from_subject_mean(
                            mean,
                            &self.initialization.parameter_scales,
                            &self.initialization.random_effect_indices,
                            &state.etas[subject_index][chain],
                            &self.initialization.iov_effect_indices,
                            kappa,
                        ),
                        None => occasion_psi(
                            population_parameters,
                            &self.initialization.parameter_scales,
                            &self.initialization.random_effect_indices,
                            &state.etas[subject_index][chain],
                            &self.initialization.iov_effect_indices,
                            kappa,
                        ),
                    }
                    .map_err(|error| error.to_string())?;
                    let occasion_subject =
                        Subject::from_occasions(subject.id().to_owned(), vec![occasion.clone()]);
                    let predictions = self
                        .equation
                        .estimate_predictions_dense(&occasion_subject, &parameters)
                        .map_err(|error| error.to_string())?;
                    derivative
                        .add_predictions_strict(&predictions, error_models, layout)
                        .map_err(|error| error.to_string())?;
                }
            } else {
                let parameters = match subject_mu {
                    Some(mean) => individual_psi_from_subject_mean(
                        mean,
                        &self.initialization.parameter_scales,
                        &self.initialization.random_effect_indices,
                        &state.etas[subject_index][chain],
                    ),
                    None => individual_psi(
                        population_parameters,
                        &self.initialization.parameter_scales,
                        &self.initialization.random_effect_indices,
                        &state.etas[subject_index][chain],
                    ),
                }
                .map_err(|error| error.to_string())?;
                let predictions = self
                    .equation
                    .estimate_predictions_dense(subject, &parameters)
                    .map_err(|error| error.to_string())?;
                derivative
                    .add_predictions_strict(&predictions, error_models, layout)
                    .map_err(|error| error.to_string())?;
            }
        }
        Ok(derivative.score)
    }
}

fn diagnostic_standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn diagnostic_accept(rng: &mut StdRng, ratio: f64) -> bool {
    ratio.is_finite() && (ratio >= 0.0 || rng.random::<f64>().max(f64::MIN_POSITIVE).ln() < ratio)
}

pub(super) fn begin_retained_transition_accounting(counts: &mut [(usize, usize, usize)]) {
    counts.fill((0, 0, 0));
}

fn mark_diagnostic_failure(
    diagnostic: &mut MarkovSimulationVarianceDiagnostics,
    rank_status: RankDiagnosticStatus,
    markov_status: MarkovSimulationVarianceStatus,
) {
    diagnostic.rank_diagnostics.status = rank_status.clone();
    diagnostic
        .rank_diagnostics
        .lrv_chain_statuses
        .fill(rank_status);
    diagnostic.lambda_status = markov_status.clone();
    diagnostic.xi_status = markov_status.clone();
    diagnostic.simulation_covariance_status = markov_status.clone();
    diagnostic.status = markov_status;
}

fn markov_matrix_status(classification: MatrixClassification) -> MarkovSimulationVarianceStatus {
    match classification {
        MatrixClassification::EligiblePsd => MarkovSimulationVarianceStatus::AssumptionsUnverified,
        MatrixClassification::NonFinite => MarkovSimulationVarianceStatus::NonFinite,
        MatrixClassification::NonSymmetric => MarkovSimulationVarianceStatus::NonSymmetric,
        MatrixClassification::Indefinite => MarkovSimulationVarianceStatus::Indefinite,
    }
}

pub(super) fn worst_valid_max_rhat(traces: &[RankMixingDiagnostic]) -> Option<f64> {
    traces
        .iter()
        .filter(|trace| matches!(trace.max_rhat_status, RankDiagnosticStatus::Available))
        .filter_map(|trace| trace.max_rhat)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

fn rank_diagnostic_error_status(error: &RankDiagnosticError) -> RankDiagnosticStatus {
    match error {
        RankDiagnosticError::NoChains => RankDiagnosticStatus::NoChains,
        RankDiagnosticError::TooFewChains { .. } => RankDiagnosticStatus::TooFewChains,
        RankDiagnosticError::UnequalChainLengths { .. } => {
            RankDiagnosticStatus::UnequalChainLengths
        }
        RankDiagnosticError::OddChainLength { .. } => RankDiagnosticStatus::OddDraws,
        RankDiagnosticError::NonFiniteDraw => RankDiagnosticStatus::NonFiniteDraws,
        RankDiagnosticError::TooFewDraws { .. } => RankDiagnosticStatus::TooFewDraws,
        RankDiagnosticError::ConstantDraws => RankDiagnosticStatus::ConstantDraws,
        RankDiagnosticError::InvalidVariance => RankDiagnosticStatus::InvalidVariance,
        RankDiagnosticError::NonPositiveTau { .. } => RankDiagnosticStatus::NonPositiveTau,
    }
}
