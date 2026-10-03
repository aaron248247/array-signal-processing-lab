# 6. References

Numbered references cited in the documents and source code. Bibliographic details were checked on 2026-10-03 against publisher, IEEE Xplore, NASA ADS or library catalogue listings found by web search. A DOI is given only where the listing confirmed it. Where no DOI is given, the entry could not be confirmed online beyond title, venue, volume and pages, or (for books) the item has no DOI in common use.

1. H. L. Van Trees, *Optimum Array Processing* (Part IV of *Detection, Estimation, and Modulation Theory*). New York: Wiley-Interscience, 2002. ISBN 0-471-09390-4.
2. B. D. Van Veen and K. M. Buckley, "Beamforming: a versatile approach to spatial filtering," *IEEE ASSP Magazine*, vol. 5, no. 2, pp. 4–24, Apr. 1988.
3. J. Capon, "High-resolution frequency-wavenumber spectrum analysis," *Proceedings of the IEEE*, vol. 57, no. 8, pp. 1408–1418, 1969. doi:10.1109/PROC.1969.7278.
4. I. S. Reed, J. D. Mallett, and L. E. Brennan, "Rapid convergence rate in adaptive arrays," *IEEE Transactions on Aerospace and Electronic Systems*, vol. AES-10, no. 6, pp. 853–863, 1974.
5. B. D. Carlson, "Covariance matrix estimation errors and diagonal loading in adaptive arrays," *IEEE Transactions on Aerospace and Electronic Systems*, vol. 24, no. 4, pp. 397–401, 1988.
6. H. Cox, "Resolving power and sensitivity to mismatch of optimum array processors," *Journal of the Acoustical Society of America*, vol. 54, no. 3, pp. 771–785, 1973.
7. H. Cox, R. M. Zeskind, and M. M. Owen, "Robust adaptive beamforming," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 35, no. 10, pp. 1365–1376, 1987.
8. G. L. Turin, "An introduction to matched filters," *IRE Transactions on Information Theory*, vol. 6, no. 3, pp. 311–329, 1960.
9. N. Levanon and E. Mozeson, *Radar Signals*. Hoboken, NJ: Wiley-IEEE Press, 2004. ISBN 0-471-47378-2.
10. M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed. New York: McGraw-Hill, 2014. ISBN 978-0-07-179832-7.
11. F. J. Harris, "On the use of windows for harmonic analysis with the discrete Fourier transform," *Proceedings of the IEEE*, vol. 66, no. 1, pp. 51–83, 1978.
12. H. M. Finn and R. S. Johnson, "Adaptive detection mode with threshold control as a function of spatially sampled clutter-level estimates," *RCA Review*, vol. 29, no. 3, pp. 414–464, 1968.
13. H. Rohling, "Radar CFAR thresholding in clutter and multiple target situations," *IEEE Transactions on Aerospace and Electronic Systems*, vol. AES-19, no. 4, pp. 608–621, 1983.
14. J. I. Marcum, "A statistical theory of target detection by pulsed radar," *IRE Transactions on Information Theory*, vol. 6, no. 2, pp. 59–267, 1960.
15. P. Swerling, "Probability of detection for fluctuating targets," *IRE Transactions on Information Theory*, vol. 6, no. 2, pp. 269–308, 1960. doi:10.1109/TIT.1960.1057561.
16. W. J. Albersheim, "A closed-form approximation to Robertson's detection characteristics," *Proceedings of the IEEE*, vol. 69, no. 7, p. 839, 1981.
17. S. M. Kay, *Fundamentals of Statistical Signal Processing, Volume II: Detection Theory*. Upper Saddle River, NJ: Prentice Hall, 1998. ISBN 0-13-504135-X.
18. V. A. Marchenko and L. A. Pastur, "Distribution of eigenvalues for some sets of random matrices," *Mathematics of the USSR-Sbornik*, vol. 1, no. 4, pp. 457–483, 1967. doi:10.1070/SM1967v001n04ABEH001994.
19. C. J. Clopper and E. S. Pearson, "The use of confidence or fiducial limits illustrated in the case of the binomial," *Biometrika*, vol. 26, no. 4, pp. 404–413, 1934.

## Where each reference is used

| Topic | References | Where |
|---|---|---|
| ULA geometry, steering vectors, array factor, conventional beamforming | [1], [2] | 01 §A, `steering.py` |
| MVDR / Capon | [3], [1] | 01 §B.4, `beamforming.py` |
| Sample covariance, SMI convergence | [4], [1] | 01 §C.1–C.2, `metrics.py` |
| Diagonal loading, adaptive-array robustness | [5], [7] | 01 §C.3, Figure 4 |
| Steering mismatch, MPDR self-nulling | [6], [1] | 01 §C.4, `metrics.py`, Figure 5 |
| Sample-eigenvalue spread | [18] | 01 §C.3, Figure 3b |
| Matched filtering | [8], [10] | 01 §D.3, `matched_filter.py` |
| LFM ambiguity, range–Doppler coupling | [9] | 01 §D.3, `waveforms.py` |
| Radar, Doppler and range–Doppler processing, straddle | [10] | 01 §D.4–D.5 |
| Windows, scalloping loss | [11] | 01 §D.5, `doppler.py` |
| CA-CFAR | [12], [10] | 01 §E.2, `cfar.py` |
| OS-CFAR, multiple-target masking | [13] | 01 §E.3, Figure 10 |
| Detection in Gaussian noise, Marcum Q | [14], [17], [10] | 01 §E.1 |
| Fluctuating targets | [15] | 01 §E.1, E.2 |
| Independent ideal-detector cross-check | [16] | 03 §3.5, `cfar_pd_snr.py` |
| Binomial confidence intervals | [19] | 02 §2.7, `cfar.py` |
