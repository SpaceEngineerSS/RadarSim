# Scientific references

This bibliography identifies the source behind each implemented model. A citation means that RadarSim implements a stated equation or modelling approach; it does not imply validation or endorsement by the author, publisher, or standards body. Page and equation references in source docstrings take precedence when a work has several editions.

## Radar equation, detection, and fluctuating targets

1. M. I. Skolnik, editor, *Radar Handbook*, 3rd edition, McGraw-Hill, 2008, chapters 2, 3, 14, and 15. ISBN 978-0-07-148547-0.
2. P. Swerling, “Probability of Detection for Fluctuating Targets,” *IRE Transactions on Information Theory*, vol. 6, no. 2, pp. 269–308, 1960. [doi:10.1109/TIT.1960.1057561](https://doi.org/10.1109/TIT.1960.1057561).
3. P. Swerling, “Radar Probability of Detection for Some Additional Fluctuating Target Cases,” *IEEE Transactions on Aerospace and Electronic Systems*, vol. 33, no. 2, pp. 698–709, 1997. [doi:10.1109/7.588492](https://doi.org/10.1109/7.588492).
4. W. J. Albersheim, “A Closed-Form Approximation to Robertson's Detection Characteristics,” *Proceedings of the IEEE*, vol. 69, no. 7, p. 839, 1981. [doi:10.1109/PROC.1981.12025](https://doi.org/10.1109/PROC.1981.12025).

## Propagation, precipitation, and surface scattering

5. ITU-R P.676-13, “Attenuation by atmospheric gases and related effects,” International Telecommunication Union, 2022. [Recommendation page](https://www.itu.int/rec/R-REC-P.676/en).
6. ITU-R P.838-3, “Specific attenuation model for rain for use in prediction methods,” International Telecommunication Union, 2005. [Recommendation page](https://www.itu.int/rec/R-REC-P.838/en).
7. J. S. Marshall and W. McK. Palmer, “The Distribution of Raindrops with Size,” *Journal of Meteorology*, vol. 5, no. 4, pp. 165–166, 1948. [doi:10.1175/1520-0469(1948)005%3C0165:TDORWS%3E2.0.CO;2](https://doi.org/10.1175/1520-0469%281948%29005%3C0165%3ATDORWS%3E2.0.CO%3B2).
8. Y. Oh, K. Sarabandi, and F. T. Ulaby, “An Empirical Model and an Inversion Technique for Radar Scattering from Bare Soil Surfaces,” *IEEE Transactions on Geoscience and Remote Sensing*, vol. 30, no. 2, pp. 370–381, 1992. [doi:10.1109/36.134086](https://doi.org/10.1109/36.134086).
9. V. Gregers-Hansen and R. Mital, *An Improved Empirical Model for Radar Sea Clutter Reflectivity*, NRL/MR/5310--12-9346, Naval Research Laboratory, 2012. [Public report](https://apps.dtic.mil/sti/pdfs/ADA559494.pdf).

## Signal processing and adaptive detection

10. H. Rohling, “Radar CFAR Thresholding in Clutter and Multiple Target Situations,” *IEEE Transactions on Aerospace and Electronic Systems*, vol. AES-19, no. 4, pp. 608–621, 1983. [doi:10.1109/TAES.1983.309350](https://doi.org/10.1109/TAES.1983.309350).
11. F. J. Harris, “On the Use of Windows for Harmonic Analysis with the Discrete Fourier Transform,” *Proceedings of the IEEE*, vol. 66, no. 1, pp. 51–83, 1978. [doi:10.1109/PROC.1978.10837](https://doi.org/10.1109/PROC.1978.10837).
12. T. T. Taylor, “Design of Line-Source Antennas for Narrow Beamwidth and Low Side Lobes,” *IRE Transactions on Antennas and Propagation*, vol. 3, no. 1, pp. 16–28, 1955. [doi:10.1109/TAP.1955.1144274](https://doi.org/10.1109/TAP.1955.1144274).

## Estimation, association, and track fusion

13. R. E. Kalman, “A New Approach to Linear Filtering and Prediction Problems,” *Journal of Basic Engineering*, vol. 82, no. 1, pp. 35–45, 1960. [doi:10.1115/1.3662552](https://doi.org/10.1115/1.3662552).
14. Y. Bar-Shalom, F. Daum, and J. Huang, “The Probabilistic Data Association Filter,” *IEEE Control Systems Magazine*, vol. 29, no. 6, pp. 82–100, 2009. [doi:10.1109/MCS.2009.934469](https://doi.org/10.1109/MCS.2009.934469). RadarSim currently uses global nearest-neighbour assignment, not PDAF; this source defines the broader association context.
15. S. J. Julier and J. K. Uhlmann, “A Non-divergent Estimation Algorithm in the Presence of Unknown Correlations,” *Proceedings of the American Control Conference*, pp. 2369–2373, 1997. [doi:10.1109/ACC.1997.609105](https://doi.org/10.1109/ACC.1997.609105).
16. H. W. Kuhn, “The Hungarian Method for the Assignment Problem,” *Naval Research Logistics Quarterly*, vol. 2, nos. 1–2, pp. 83–97, 1955. [doi:10.1002/nav.3800020109](https://doi.org/10.1002/nav.3800020109).

## Receiver nonlinearity and electronic countermeasures

17. R. Price, “A Useful Theorem for Nonlinear Devices Having Gaussian Inputs,” *IRE Transactions on Information Theory*, vol. 4, no. 2, pp. 69–72, 1958. [doi:10.1109/TIT.1958.1057444](https://doi.org/10.1109/TIT.1958.1057444).
18. H. E. Rowe, “Memoryless Nonlinearities With Gaussian Inputs: Elementary Results,” *Bell System Technical Journal*, vol. 61, no. 7, pp. 1519–1526, 1982. [doi:10.1002/j.1538-7305.1982.tb04356.x](https://doi.org/10.1002/j.1538-7305.1982.tb04356.x).
19. F. Neri, *Introduction to Electronic Defense Systems*, 2nd edition, SciTech Publishing, 2006. ISBN 978-1-891121-49-4. The ECM implementation uses public, generic range/velocity deception and noise-jamming relationships, not equipment-specific data.

## SAR and ISAR

20. I. G. Cumming and F. H. Wong, *Digital Processing of Synthetic Aperture Radar Data: Algorithms and Implementation*, Artech House, 2005. ISBN 978-1-58053-058-3.
21. C. C. Chen and H. C. Andrews, “Target-Motion-Induced Radar Imaging,” *IEEE Transactions on Aerospace and Electronic Systems*, vol. AES-16, no. 1, pp. 2–14, 1980. [doi:10.1109/TAES.1980.308873](https://doi.org/10.1109/TAES.1980.308873).
22. X. Lv, M. Xing, C. Wan, and S. Zhang, “ISAR Imaging of Maneuvering Targets Based on the Range Centroid Doppler Technique,” *IEEE Transactions on Image Processing*, vol. 19, no. 1, pp. 141–153, 2010. [doi:10.1109/TIP.2009.2032892](https://doi.org/10.1109/TIP.2009.2032892).

## Standards and constants

23. IEEE Std 686-2017, *IEEE Standard for Radar Definitions*. [IEEE standard record](https://standards.ieee.org/ieee/686/5868/).
24. NIST, “Fundamental Physical Constants — Extensive Listing.” [NIST constants](https://physics.nist.gov/cuu/Constants/).
