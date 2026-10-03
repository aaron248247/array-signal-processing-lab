# 5. Roadmap

This page separates what exists from what does not. Items under 5.2 and 5.3 are **not implemented**.

## 5.1 Completed

* Narrowband ULA model with closed-form verification of the array factor, main lobe, nulls, beamwidth and sidelobe level.
* Conventional and MVDR/MPDR beamforming with a Cholesky solver and a singularity guard. Closed-form checks for the single-jammer SINR, null depth and mismatch (Cox) expressions.
* Monte Carlo SMI studies:
  * snapshots, comparing RMB and the signal-in-training relation (C.2);
  * diagonal loading;
  * steering mismatch;
  * jammer angle and JNR;
  * covariance conditioning.
* Generic LFM pulse-Doppler model with exact fractional delay. Matched filtering, Doppler FFT and windowing, range–Doppler maps.
* Verification of the radar chain: compression gain, resolution, range–Doppler coupling, straddle and scalloping loss, estimation error against SNR.
* CA-CFAR (1-D, 2-D, three edge modes) and OS-CFAR (1-D). Monte Carlo $P_{FA}$ verification in i.i.d. noise and in processed noise, including the correlated-training-cell discrepancy. $P_D$ against SNR for Swerling 0 and 1.
* One integrated array → beamformer → matched filter → Doppler → CFAR example.

## 5.2 Natural near-term extensions

* **CFAR calibration for correlated noise.** Calibrate $\alpha$ by Monte Carlo for a given processing chain and compare with decimated training windows. This addresses §3.6 directly.
* **Additional CFAR variants.** GO- and SO-CFAR [10, 13] with their closed-form $P_{FA}$ verified the same way as CA and OS. Censored CA-CFAR for multiple-target situations.
* **2-D OS-CFAR** and a Monte Carlo study of masking probability against target separation, extending Figure 10 beyond one realization.
* **Diagonal-loading selection rules.** For example, loading relative to an estimated noise floor, or the Cox–Zeskind–Owen white-noise-gain constraint [7], evaluated over the same sweeps.
* **Weight tapering** (Chebyshev or Taylor) for sidelobe control and its interaction with MVDR.
* **Pulse integration.** Noncoherent integration across CPIs, with Albersheim [16] as a multi-pulse check.

## 5.3 Larger future extensions

* **Direction finding.** MUSIC and ESPRIT, with resolution and RMSE compared against the Cramér–Rao bound.
* **Broadband arrays.** True time delay and the aperture-transit limit of the narrowband assumption used here. Frequency-domain beamforming.
* **Nonuniform and sparse arrays.** Minimum-redundancy and coprime geometries, grating-lobe control.
* **Space–time adaptive processing.** Clutter ridge, the RMB-type sample-support requirement in $NM$ dimensions, reduced-rank methods.
* **Colored clutter models.** Gaussian-spectrum and K-distributed clutter; CFAR in non-Rayleigh backgrounds.
* **Correlated (coherent) interference and multipath.** Signal cancellation in MVDR, spatial smoothing.
* **Array calibration errors.** Element gain/phase and position errors, mutual coupling, and their effect on null depth. This connects to the electromagnetic modeling work in a separate repository.
* **Independent cross-checks.** Reproducing selected results with an independent implementation or a published dataset.
* **Real data.** Software-defined-radio recordings (for example, a small receive array or a public dataset) used to check the noise model and the CFAR behavior on measured data.
* **Hardware experiments.** A small receive array with known emitter geometry.
