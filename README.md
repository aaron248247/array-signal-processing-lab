# array-signal-processing-lab

A small Python study of narrowband array processing, adaptive beamforming, pulse-Doppler processing and CFAR detection. Each method is stated as a model, implemented, and compared with an analytical result, a limiting case or a Monte Carlo statistic. Where the comparison fails, the discrepancy is investigated and reported.

All scenarios use generic, round-number parameters. They do not represent any real system.

## Questions addressed

1. Does the ULA model reproduce the closed-form array factor, main lobe and nulls to numerical precision?
2. How much SINR does MVDR gain over conventional beamforming against sidelobe jammers? How quickly is that gain lost when the covariance is estimated from few snapshots, when the target contaminates the training data, or when the look direction is slightly wrong?
3. What does diagonal loading buy, and where does it start to remove the adaptivity it is meant to protect?
4. Are injected ranges and Doppler shifts recovered by matched filtering and pulse-Doppler processing, and how large are the grid (binning) effects?
5. Does CA-CFAR achieve its design false-alarm probability, both in idealized noise and in noise that has passed through the processing chain?

## Main equations

Steering vector and conventional array factor ($x_n = nd$, $\theta$ from broadside):

$$a_n(\theta) = e^{\,jkx_n\sin\theta},\qquad AF(\theta) = e^{j(N-1)\psi/2}\,\frac{\sin(N\psi/2)}{N\sin(\psi/2)},\quad \psi = kd(\sin\theta-\sin\theta_0)$$

MVDR weights [3] and the Reed–Mallett–Brennan loss [4] for SMI with $K$ signal-free snapshots:

$$\mathbf w = \frac{\mathbf R^{-1}\mathbf a_0}{\mathbf a_0^H\mathbf R^{-1}\mathbf a_0},\qquad E\!\left[\frac{\mathrm{SINR}(\hat{\mathbf w})}{\mathrm{SINR}_{\rm opt}}\right] = \frac{K-N+2}{K+1}$$

Monostatic delay and Doppler ($v>0$ closing), LFM resolution, and CA-CFAR threshold multiplier [12] for $N_t$ training cells:

$$\tau = \frac{2R}{c},\qquad f_D = \frac{2v}{\lambda},\qquad \Delta R = \frac{c}{2B},\qquad \alpha = N_t\left(P_{FA}^{-1/N_t}-1\right)$$

Derivations, assumptions and conventions are in [docs/01_theory.md](docs/01_theory.md). Bracketed numbers refer to [docs/06_references.md](docs/06_references.md).

## Representative results

<p align="center"><img src="figures/fig02_conventional_vs_mvdr.png" width="900" alt="Figure 2"></p>

**Figure 2.** Conventional vs MVDR against two sidelobe jammers (N = 16). (a) Beampatterns. (b) SINR loss relative to the optimum, with Monte Carlo means for sample-matrix inversion and the corresponding theory.

<p align="center"><img src="figures/fig03_sinr_vs_snapshots.png" width="900" alt="Figure 3"></p>

**Figure 3.** (a) Normalized SINR vs number of training snapshots, against the RMB law and the target-in-training relation. (b) Condition number of the sample covariance.

<p align="center"><img src="figures/fig05_steering_mismatch.png" width="900" alt="Figure 5"></p>

**Figure 5.** Pointing error with the target present in the covariance (self-nulling), with and without diagonal loading.

<p align="center"><img src="figures/fig08_range_doppler_map.png" width="900" alt="Figure 8"></p>

**Figure 8.** Range–Doppler map of three injected targets after LFM pulse compression and Doppler processing.

<p align="center"><img src="figures/fig11_cfar_pfa_verification.png" width="900" alt="Figure 11"></p>

**Figure 11.** Measured vs requested CA/OS-CFAR false-alarm probability. (a) i.i.d. noise. (b) Noise processed by the matched filter and Doppler FFT.

All 13 figures, with full captions, are in [docs/07_figure_index.md](docs/07_figure_index.md).

## Main verification results

| Check | Result |
|---|---|
| Numerical vs closed-form array factor (N = 8, 16; 5 steering cases; 0.001° grid) | max error $2.1\times10^{-15}$; nulls within $1.4\times10^{-13}$ deg |
| MVDR distortionless constraint, clairvoyant SINR, single-jammer null depth, Cox mismatch formula [6] | reproduced to $10^{-8}$–$10^{-12}$ relative |
| SMI loss vs RMB [4], $16\le K\le 640$ ([03 §3.3](docs/03_verification_validation.md#33-finite-snapshot-behavior-statistical)) | largest $\lvert z\rvert$ = 1.6; K = 2N: 0.5426 ± 0.0019 vs 0.5455 |
| Matched filter vs closed-form LFM autocorrelation | 0.007 of peak at $f_s = 2B$ |
| Pulse-compression SNR gain (L = 100) | 10.04 ± 0.07 vs 10.00 |
| Range–Doppler coupling $-f_D T/B$ | within 0.015 samples |
| CA/OS-CFAR $P_{FA}$ in i.i.d. noise, $10^{-2}$–$10^{-5}$ | measured/requested 0.92–1.08 |
| CA-CFAR $P_D$ (Swerling 0 and 1), 38 points | all inside 95 % intervals |

Details, including what each check does and does not establish, are in [docs/03_verification_validation.md](docs/03_verification_validation.md).

## Main numerical findings

* With jammers at 40 and 30 dB JNR in the sidelobes, the conventional beamformer is 30 dB below the optimum output SINR (12.0 dB). Clairvoyant MVDR reaches the optimum.
* SMI needs about $2N$ signal-free snapshots for a 3 dB average loss. With the target present in the training data (optimum SINR 12 dB), it needs about $24N$ ($K = 384$). The relation $\rho/(1+S(1-\rho))$, derived in docs/01 §C.2, matches the Monte Carlo results.
* 10 dB of diagonal loading recovers most of the finite-sample loss, down to $K = 4 < N$. Loading above about 30 dB (relative to noise) starts to cancel the jammer nulls. At +50 dB it has given up 22.7 dB of its 30 dB advantage over the conventional beamformer.
* A 1° pointing error (one sixth of a beamwidth) costs 0.3 dB when the target is absent from the covariance. When the target is present, the same error costs 13 dB at SNR 0 dB and 52 dB at SNR 20 dB.
* FFT binning limits unaided accuracy to $\Delta/\sqrt{12}$ RMS (4.3 m, 2.3 m/s here) and costs up to 0.9 dB (range straddle) and 3.9 dB (Doppler scalloping, rectangular window). A three-point parabolic fit reduces range error to 0.7 m RMS at 28 dB but leaves a 0.17-bin Doppler bias for the rectangular window.
* CA-CFAR with $N_t = 16$ costs 1.3 dB relative to a known-noise detector at $P_{FA}=10^{-4}$.
* Applied to noise that has passed through a 2×-oversampled matched filter and a Hann Doppler window, CA-CFAR exceeds its nominal false-alarm rate by up to 45 %, because the training cells are correlated.

Tables with all values are in [docs/04_results.md](docs/04_results.md).

## Limitations

* Narrowband, far-field, ideal isotropic elements. No mutual coupling, calibration error, or element pattern.
* Uncorrelated point sources and white Gaussian noise. No clutter, multipath, or coherent interference.
* Constant-amplitude point targets within a CPI (Swerling 0 or 1 only for detection). No pulse eclipsing. Envelope time-compression neglected.
* Single-CPI processing. No tracking, no multi-CPI integration.
* The integrated array + radar case is a single realization and serves as a demonstration.
* The $N_{\rm eff}$ correction for correlated CFAR training cells is a moment-matching approximation. It overestimates the measured inflation in the most correlated case.

The roadmap of extensions not implemented here is in [docs/05_roadmap.md](docs/05_roadmap.md).

## Installation

Requires Python ≥ 3.10.

```bash
git clone <repository-url> array-signal-processing-lab
cd array-signal-processing-lab
python -m pip install -e ".[test]"     # numpy, scipy, matplotlib, pytest
```

## Reproduction

```bash
python -m pytest                          # 114 tests, a few seconds
python experiments/run_all.py             # all results and figures (~2 min on 2 cores)
python experiments/run_all.py --plot-only # redraw figures from saved results
python experiments/cfar_pd_snr.py         # any single experiment
python tools/check_docs.py                # documentation link / figure / reference checks
python tools/compare_results.py A/ B/     # compare two result directories
```

Every experiment uses a fixed seed. Configuration, seed, metrics, timing and library versions are written to `results/<name>.json`, and arrays for plotting to `results/<name>.npz`. Two consecutive full runs produced bitwise-identical metrics on the same platform.

## Repository structure

```
array-signal-processing-lab/
├── README.md, LICENSE, CITATION.cff, pyproject.toml
├── src/array_signal_lab/
│   ├── array_model.py      ULA geometry
│   ├── steering.py         steering vectors, closed-form array factor, nulls
│   ├── beamforming.py      conventional and MVDR weights, patterns, angle scan
│   ├── covariance.py       snapshot simulation, sample/theoretical covariance, loading
│   ├── metrics.py          SINR, closed forms (single jammer, RMB, signal-present, Cox)
│   ├── waveforms.py        LFM and rectangular pulses, closed-form autocorrelation
│   ├── radar_signal.py     generic pulse-Doppler echo model
│   ├── matched_filter.py   FFT matched filter, peak interpolation
│   ├── doppler.py          slow-time windows, Doppler FFT, scalloping loss
│   ├── cfar.py             CA (1-D/2-D) and OS CFAR, thresholds, Pd formulas
│   └── io.py               JSON + NPZ result files
├── experiments/            one script per study (+ run_all.py, plotstyle.py, common.py)
├── tests/                  pytest suite
├── tools/                  check_docs.py, compare_results.py
├── results/                saved JSON + NPZ outputs
├── figures/                generated PNG figures (fig01 … fig13)
└── docs/
    ├── 01_theory.md
    ├── 02_numerical_method.md
    ├── 03_verification_validation.md
    ├── 04_results.md
    ├── 05_roadmap.md
    ├── 06_references.md
    └── 07_figure_index.md
```

## Documentation

1. [Theory](docs/01_theory.md): models, derivations, conventions
2. [Numerical method](docs/02_numerical_method.md): discretization, linear algebra, Monte Carlo design, result format
3. [Verification and validation](docs/03_verification_validation.md): exact, statistical, sensitivity and exploratory checks
4. [Results](docs/04_results.md): numbers and figures
5. [Roadmap](docs/05_roadmap.md): what is not implemented
6. [References](docs/06_references.md)
7. [Figure index](docs/07_figure_index.md)

## Relation to other work

This repository complements a separate computational-electromagnetics project (FDTD vs exact Mie scattering, periodic dielectric-cylinder arrays). That project models the fields. This one models what an array and a receiver do with them.

## License

MIT; see [LICENSE](LICENSE). To cite this work, see [CITATION.cff](CITATION.cff).
