# 7. Figure index

Figures are numbered continuously in the order in which they appear in [04_results.md](04_results.md). Each is produced by the `plot()` function of the listed script, from the listed result files only. Regenerate one with `python experiments/<script>.py --plot-only`, or all with `python experiments/run_all.py --plot-only`.

| No. | File | Script | Results | Appears in |
|---|---|---|---|---|
| 1 | `figures/fig01_ula_verification.png` | `ula_verification.py` | `results/ula_verification.*` | 04 §4.1, 03 §3.1, README |
| 2 | `figures/fig02_conventional_vs_mvdr.png` | `jammer_suppression.py` | `results/jammer_suppression.*` | 04 §4.2, 03 §3.3, README |
| 3 | `figures/fig03_sinr_vs_snapshots.png` | `snapshot_sensitivity.py` | `results/snapshot_sensitivity.*` | 04 §4.3, 03 §3.3, §3.7, README |
| 4 | `figures/fig04_loading_sensitivity.png` | `loading_sensitivity.py` | `results/loading_sensitivity.*` | 04 §4.4, 03 §3.7 |
| 5 | `figures/fig05_steering_mismatch.png` | `steering_mismatch.py` | `results/steering_mismatch.*` | 04 §4.5, 03 §3.7, README |
| 6 | `figures/fig06_jammer_angle_jnr.png` | `jammer_sweep.py` | `results/jammer_sweep.*` | 04 §4.6, 03 §3.7 |
| 7 | `figures/fig07_matched_filter.png` | `range_doppler_demo.py` | `results/range_doppler_demo.*` | 04 §4.7, 03 §3.4 |
| 8 | `figures/fig08_range_doppler_map.png` | `range_doppler_demo.py` | `results/range_doppler_demo.*` | 04 §4.8, 03 §3.4, README |
| 9 | `figures/fig09_binning_effects.png` | `range_doppler_demo.py` | `results/range_doppler_demo.*` | 04 §4.9, 03 §3.4, §3.7 |
| 10 | `figures/fig10_cfar_threshold_example.png` | `cfar_false_alarm_check.py` | `results/cfar_false_alarm_check.*` | 04 §4.10, 03 §3.8 |
| 11 | `figures/fig11_cfar_pfa_verification.png` | `cfar_false_alarm_check.py` | `results/cfar_false_alarm_check.*` | 04 §4.11, 03 §3.5–3.6, README |
| 12 | `figures/fig12_pd_vs_snr.png` | `cfar_pd_snr.py` | `results/cfar_pd_snr.*` | 04 §4.12, 03 §3.5, README |
| 13 | `figures/fig13_integrated_array_radar.png` | `integrated_array_radar.py` | `results/integrated_array_radar.*` | 04 §4.13, 03 §3.8 |

---

## Figure 1 — ULA analytical verification

![Figure 1](../figures/fig01_ula_verification.png)

**Caption.** Verification of the narrowband ULA model against the closed-form array factor (A.3).
(a) Array factor, N = 16, d = λ/2 — numerical response $|\mathbf w^H\mathbf a(\theta)|$ (lines) and closed form (circles, every 5°) for broadside and 30° steering; triangles mark the analytical null directions (A.4);
(b) Numerical vs closed-form error — maximum of $|AF_{num}-AF_{closed}|$ in 1° bins over the 0.001° evaluation grid, for four (N, θ₀) cases;
(c) Null-location error — absolute difference between each refined numerical null and its analytical position, for all interior nulls of five cases.

* **Quantities.** (a) dB magnitude vs angle; (b) absolute complex error vs angle (log scale); (c) null-location error (deg, log scale) vs null angle.
* **What to notice.** Errors stay at $10^{-16}$–$10^{-15}$ (round-off) at every angle, including near the nulls. Null positions agree to $\le 1.4\times10^{-13}$ deg. Steering to 30° shifts the pattern in $\sin\theta$, so the beam widens and the lobes compress toward endfire.
* **Caveats.** In (c), errors smaller than $10^{-16}$ deg, including exact zeros, are drawn at the $10^{-16}$ floor. The closed-form markers in (a) are a 5° subsample of the full comparison shown in (b).

## Figure 2 — Conventional vs MVDR beampattern

![Figure 2](../figures/fig02_conventional_vs_mvdr.png)

**Caption.** Conventional and MVDR beamforming with two sidelobe jammers (N = 16; target 0°; jammers +35°/40 dB and −42°/30 dB).
(a) Beampatterns — normalised response for the conventional beamformer, MVDR with the true $\mathbf R_{in}$, and one SMI realisation with K = 2N, unloaded and with 10 dB loading; vertical lines mark the target and the jammers;
(b) Output SINR loss — $10\log_{10}(\mathrm{SINR}/\mathrm{SINR}_{\rm opt})$ for each beamformer, with Monte Carlo means and 5–95 % ranges for the SMI cases (2000 trials) and theory markers (optimum, RMB mean (C.1), signal-present mean (C.2)).

* **Quantities.** (a) $20\log_{10}|\mathbf w^H\mathbf a(\theta)|$; (b) SINR loss in dB.
* **What to notice.** The conventional beamformer loses 30 dB to the jammers. Clairvoyant MVDR reaches the optimum with nulls below −100 dB. The single unloaded K = 2N draw has sidelobes raised by roughly 10 dB, and loading restores the quiescent sidelobes while keeping the nulls. In (b) the Monte Carlo means fall on the theory markers.
* **Caveats.** The patterns in (a) are single realisations. Jammer null depths for SMI vary from trial to trial; medians are given in 04 §4.2.

## Figure 3 — Output SINR versus number of snapshots

![Figure 3](../figures/fig03_sinr_vs_snapshots.png)

**Caption.** Finite-snapshot behaviour of SMI/MVDR in the baseline scenario.
(a) Normalised output SINR vs snapshots — Monte Carlo mean of $\rho$ (600 trials per point, 95 % intervals) for unloaded and 10 dB-loaded SMI, with and without the target in the training data, against the RMB law (C.1) and the signal-present relation (C.2);
(b) Conditioning of $\hat{\mathbf R}$ — median and 10–90 % range of $\mathrm{cond}(\hat{\mathbf R})$ (200 draws per K), the Marchenko–Pastur lower-edge estimate, and $\mathrm{cond}(\mathbf R_{in})$.

* **Quantities.** (a) mean $\rho$ (dB) vs K (log axis); (b) 2-norm condition number vs K.
* **What to notice.** The unloaded points track both theory curves from K = N upward. With the target present, 3 dB loss needs K ≈ 384 instead of ≈ 29. Loading makes K < N usable. The condition number at K = N is about 340× the true value.
* **Caveats.** Unloaded SMI points for K < N are absent because the estimator is undefined there, not because they were dropped. At K = N the distribution of $\rho$ is very skewed, so the plotted interval is approximate (03 §3.3).

## Figure 4 — Diagonal-loading sensitivity

![Figure 4](../figures/fig04_loading_sensitivity.png)

**Caption.** Effect of diagonal loading $\delta/\sigma^2$ on SMI/MVDR.
(a) Output SINR vs diagonal loading — mean $\rho$ for K = N/2, 2N (signal-free and target in training) and 10N (target in training), the clairvoyant loaded beamformer, and the conventional level; dash-dot lines mark the two interference eigenvalues of $\mathbf R_{in}$;
(b) Jammer-1 suppression vs loading — median response toward the 35°/40 dB jammer relative to the target response.

* **Quantities.** (a) mean $\rho$ (dB) vs loading (dB); (b) relative response (dB) vs loading (dB).
* **What to notice.** There is a broad plateau of good performance from about 0 dB to about 30 dB. Above that, loading starts to cancel the adaptivity and all curves converge to the conventional result. With the target in the training data, loading recovers 8.8 dB at K = 2N. In (b) the SMI null depth is limited to about −55…−68 dB by estimation error until loading itself becomes the limit.
* **Caveats.** The plateau edges depend on the scenario through the noise-eigenvalue spread and the weakest interference eigenvalue. They should not be read as general loading rules.

## Figure 5 — Steering-mismatch sensitivity

![Figure 5](../figures/fig05_steering_mismatch.png)

**Caption.** Pointing error and target self-nulling.
(a) SINR loss vs pointing error, SNR = 0 dB/element — conventional, MVDR from $\mathbf R_{in}$ ($S c^2$), MPDR from the full $\mathbf R$ (C.3), and Monte Carlo SMI with the target in training (K = 10N), unloaded and with 10/20 dB loading;
(b) SINR loss vs SNR, δ = 1° — same estimators against per-element SNR;
(c) Patterns, δ = 1°, SNR = 10 dB, true covariances — conventional, MVDR from $\mathbf R_{in}$, MPDR, and MPDR with 20 dB loading; vertical lines mark the look direction and the true target.

* **Quantities.** (a, b) SINR relative to the optimum for the true target direction (dB); (c) beampattern (dB).
* **What to notice.** Without the target in the covariance a 1° error costs 0.28 dB. With the target present it costs 13 dB at SNR 0 dB, and the cost grows by about 20 dB per 10 dB of SNR. In (c) the MPDR places a notch on the true target (−21 dB) and raises its sidelobes to about 0 dB. Loading moves the notch away.
* **Caveats.** The SMI curves are means of 300 trials per point. Near δ = 0 they include the finite-sample loss of (C.2), which the clairvoyant MPDR does not have.

## Figure 6 — Jammer angle and JNR

![Figure 6](../figures/fig06_jammer_angle_jnr.png)

**Caption.** Single jammer, N = 16, target at broadside, SNR 0 dB/element.
(a) SINR vs jammer angle, JNR = 30 dB — closed-form optimum (B.7) and conventional (B.5) SINR, and Monte Carlo SMI (K = 2N, 200 trials per angle) with and without 10 dB loading;
(b) MVDR beam-pointing error (true $\mathbf R_{in}$) — location of the pattern maximum within ±7° of the look direction minus the look direction, vs jammer angle; shading marks the conventional main lobe;
(c) SINR vs JNR, jammer at 35° — same estimators against JNR.

* **Quantities.** (a, c) output SINR (dB); (b) angle offset (deg).
* **What to notice.** The optimum degrades only when the jammer enters the main lobe. The conventional SINR follows the sidelobe structure and peaks wherever the jammer sits on a pattern null. Unloaded SMI keeps a constant 2.6 dB gap to the optimum, independent of angle and JNR. Pointing errors stay ≤ 0.21° for jammers outside the main lobe and reach 4.6° for jammers inside it.
* **Caveats.** The pointing error in (b) is defined as the location of the maximum, which need not coincide with the look direction even though the MVDR gain there is exactly 1.

## Figure 7 — Matched-filter range response

![Figure 7](../figures/fig07_matched_filter.png)

**Caption.** LFM pulse compression (B = 5 MHz, T = 10 µs, f_s = 10 MHz).
(a) LFM compression, noiseless, fractional delay — sampled matched-filter output (circles) for a target at 4321.7 m (288.3 samples) against the continuous closed form (D.5) at the true delay;
(b) Two equal targets, ΔR = c/(2B) — oversampled (100 MHz) matched-filter output for relative phases 0, π/2 and π;
(c) LFM range–Doppler coupling — measured matched-filter peak shift vs Doppler shift against $-f_D T/B$; shading marks the unambiguous Doppler interval of the radar model.

* **Quantities.** (a, b) $|y|/\sqrt L$ (dB) vs lag; (c) delay shift (samples) vs Doppler (kHz).
* **What to notice.** The samples lie on the continuous curve (max deviation 0.007). Whether two targets at the nominal resolution are resolved depends on their relative phase. The coupling shift is linear and matches theory to 0.015 samples. Within the unambiguous Doppler interval it is at most 0.1 sample.
* **Caveats.** Panel (c) deliberately extends to ±500 kHz, far beyond the unambiguous Doppler, to make the shift measurable. Panel (b) uses noiseless, oversampled data.

## Figure 8 — Range–Doppler map

![Figure 8](../figures/fig08_range_doppler_map.png)

**Caption.** One CPI (M = 64 pulses) with three targets in white noise, after matched filtering and a Hann-windowed Doppler FFT.
(a) Full map, Hann window — power relative to the median-based noise floor; for display only, each pixel is the maximum over 5 range cells so that single-cell peaks survive downsampling; circles mark injected targets;
(b) Zoom on target 1, every cell — unpooled cells around target 1; + marks the injected range and velocity;
(c) Doppler cut at R = 4.317 km — slow-time spectrum at the target-1 range cell for rectangular and Hann windows.

* **Quantities.** Power (dB re: noise) vs range (km) and radial velocity (m/s, positive = closing).
* **What to notice.** All three targets appear where they were injected, closing targets at positive velocity. Target 1 peaks 21.6 dB above the floor. The dark strip at ranges above 13.5 km is the partial-overlap region where the matched filter integrates less noise (§2.4).
* **Caveats.** The pooling in (a) is a display operation only; estimates and CFAR use unpooled cells. In the single-realisation Doppler cut (c), the noise hides the lower Hann sidelobes.

## Figure 9 — FFT binning effects

![Figure 9](../figures/fig09_binning_effects.png)

**Caption.** Grid quantisation in range and Doppler.
(a) Range estimate error — noiseless error of the grid-argmax and three-point parabolic estimators vs the true delay's offset from the sample grid;
(b) Doppler estimate error — same for a slow-time tone with rectangular and Hann windows, plus 8× zero padding followed by parabolic interpolation;
(c) Straddle and scalloping loss — peak loss vs offset, measured (markers) and exact or closed form (lines) for range straddle (D.5) and Doppler scalloping (D.8);
(d) Monte Carlo RMS error — RMS range and velocity error vs post-processing SNR for random sub-cell target positions (150 trials per point, rectangular window), with the uniform-quantisation level $1/\sqrt{12}$.

* **Quantities.** Errors in samples or bins; losses in dB; RMS error in samples or bins (log scale).
* **What to notice.** The grid estimator has a saw-tooth error of ±0.5 cell. Parabolic interpolation reduces the range error to 0.03 sample but the rectangular-window Doppler error only to 0.23 bin, so the shape of the peak matters. Zero padding removes the Doppler bias. Worst-case scalloping is 3.92 dB (rect) and 1.42 dB (Hann). In (d) the parabolic velocity error saturates at its bias level.
* **Caveats.** The closed form in (c) for range straddle is the continuous-time autocorrelation. The sampled result differs by up to 0.085 dB (03 §3.4).

## Figure 10 — CA-CFAR threshold example

![Figure 10](../figures/fig10_cfar_threshold_example.png)

**Caption.** One range cut through the Doppler bin of three targets at 101.3 m/s ($P_{FA} = 10^{-4}$, 8 training and 2 guard cells per side).
(a) Range cut at the target Doppler bin — cell power, CA-CFAR threshold (valid edges), CA-CFAR with truncated windows at the edges, OS-CFAR threshold (k = 12 of 16), and detections;
(b) Zoom: weak target masked by strong neighbour — the −15 dB target at 4.32 km and the −10 dB target 9 cells away (dash-dot lines).

* **Quantities.** Power and thresholds in dB relative to the noise floor vs range.
* **What to notice.** At the weak target the strong neighbour raises the CA threshold to 27.6 dB above noise, against 19.5 dB for OS-CFAR and a cell power of 22.0 dB. CA-CFAR therefore misses the weak target and OS-CFAR detects it. The CA threshold has a flat-topped "shoulder" wherever a strong target falls inside the training window. One noise cell near 6.5 km is a false alarm for both detectors.
* **Caveats.** This is a single realisation chosen to show the masking mechanism. The separation and amplitudes were selected after a 20-seed scan in which, for this geometry, CA detected the weak target in 0/20 and OS in 20/20 realisations. That scan is not part of the saved results.

## Figure 11 — Measured versus requested false-alarm probability

![Figure 11](../figures/fig11_cfar_pfa_verification.png)

**Caption.** Monte Carlo verification of the CFAR false-alarm rate.
(a) i.i.d. exponential noise, 1-D window — measured/requested $P_{FA}$ with 95 % intervals for CA-CFAR ($N_t$ = 16, 32; $2\times10^7$ cells per point) and OS-CFAR ($N_t$ = 16, k = 12; $4\times10^6$ cells);
(b) 2-D CA-CFAR on processed noise — measured/requested $P_{FA}$ ($N_t = 132$; 60 maps per case) at $10^{-3}$ and $10^{-4}$ for noise passed through the matched filter at $f_s = 2B$ or $B$ and a rectangular or Hann Doppler window, with the $N_{\rm eff}$ moment-matching prediction (E.9) (diamonds).

* **Quantities.** Ratio of measured to requested $P_{FA}$.
* **What to notice.** In (a) all points are consistent with 1 within their intervals or close to them (largest |z| = 2.9, re-run gave z = 0.67). In (b) the processed-noise cases with correlated training cells lie significantly above 1, by up to 1.45. The $N_{\rm eff}$ estimate captures the trend but overshoots for the most correlated case.
* **Caveats.** Interval widths in (a) at $10^{-5}$ reflect only about 190 false alarms per point. The partial-overlap lags are excluded from (b).

## Figure 12 — Probability of detection versus SNR

![Figure 12](../figures/fig12_pd_vs_snr.png)

**Caption.** Detection performance at $P_{FA} = 10^{-4}$.
(a) $P_D$ vs SNR — ideal (known-noise) detector and CA-CFAR ($N_t$ = 16) for Swerling 0 (Marcum Q (E.2); quadrature (E.7)) and Swerling 1 ((E.3); (E.6)), with Monte Carlo points (20 000 windows each, Clopper–Pearson 95 %);
(b) CFAR loss — extra SNR required by CA-CFAR relative to the ideal detector for a Swerling-0 target vs number of training cells, at $P_D$ = 0.5 and 0.9.

* **Quantities.** (a) $P_D$ vs single-cell SNR (dB); (b) loss (dB) vs $N_t$ (log axis).
* **What to notice.** All 38 Monte Carlo points fall on the theory curves. The 1.3 dB CFAR loss at $N_t$ = 16 is a near-constant SNR shift. A fluctuating target needs 7.6 dB more SNR than a steady one at $P_D$ = 0.9.
* **Caveats.** This is single-cell detection with independent training cells. The correlated-noise effect of Figure 11(b) is not included.

## Figure 13 — Integrated array + radar case

![Figure 13](../figures/fig13_integrated_array_radar.png)

**Caption.** Array → beamformer → matched filter → Doppler (Hann) → 2-D CA-CFAR ($P_{FA} = 10^{-4}$) for N = 16 with two targets near broadside and a 40 dB noise jammer at 35°.
(a) Conventional beam — range–Doppler map relative to its own noise-plus-interference floor (17.4 dB above element noise), CA-CFAR detections (squares) and injected targets (circles);
(b) MVDR beam (K = 20N) — the same after MVDR weights estimated from the data cube (floor −12.1 dB re: element noise).

* **Quantities.** Power (dB re: each map's own floor; display max-pooled over 5 range cells) vs range and velocity.
* **What to notice.** The two maps look alike because each is scaled to its own floor. The conventional floor, however, is 29.5 dB higher, so both targets are buried and only false alarms appear. After MVDR both targets are detected.
* **Caveats.** Single realisation. False alarms (10 and 14 cells) are in line with the expectation of 8.2 including the correlated-noise inflation; one map cannot test this. The MVDR covariance includes a small amount of target energy from the data cube.
