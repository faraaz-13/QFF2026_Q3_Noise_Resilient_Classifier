# Noise-Resilient Quantum-Kernel Classifier

## Qiskit Fall Fest 2026 — Phase 1, PS-Q3

**Team:** Qubits  |  **Team leader:** Shaik Zaiba Faraaz

### Abstract

This project studies how synthetic gate and readout noise affect a quantum-kernel classifier and evaluates simple error-mitigation techniques across two connectivity models. A two-qubit data-encoding feature map estimates kernel entries through a compute–uncompute circuit. A support-vector classifier uses the estimated kernel matrix. The implementation compares ideal simulation, noisy simulation, readout assignment-matrix correction, quadratic noise-parameter extrapolation, and their combination.

### Problem statement

Quantum machine-learning circuits are affected by imperfect gates, measurement errors, and hardware connectivity constraints. PS-Q3 asks how classification behavior changes under synthetic device noise and whether error mitigation improves the estimated result when mapped to Processor A and Processor B. This project records classification scores, kernel-matrix deviation, circuit depth, two-qubit gate count, SWAP count, and runtime.

### Dataset and preprocessing

The experiment uses the specified `sklearn.datasets.make\_moons(n\_samples=100, noise=0.1, random\_state=42)` dataset. A fixed stratified 70/30 train/test split is used. To limit pairwise quantum-kernel simulation cost, the training subset is capped at six examples per class (12 training points total); the held-out test partition remains unchanged. The scaler is fitted on the training partition only, and both features are mapped to angles in \[0, π].

### Quantum method

The two-qubit feature map encodes each input using single-qubit rotations, an entangling CX, a data-dependent phase rotation, and a second CX. For samples x and z, the compute–uncompute circuit applies U(x) followed by U(z)†. The measured probability of `00` estimates the fidelity kernel K(x,z)=|⟨φ(z)|φ(x)⟩|². A precomputed-kernel SVM is trained on the training kernel and evaluated with the test-to-training kernel.

### Processor and noise assumptions

Processor A has a five-qubit budget and uses a participant-defined linear-chain graph. Processor B has a nine-qubit budget and uses the guide-reference graph transcribed from the participant-supplied Challenge Kit excerpt. Both use basis gates `{rz, sx, x, cx}` and the supplied synthetic error rates: one-qubit depolarizing probability 0.001, CX depolarizing probability 0.005, and symmetric readout probability 0.01. Confirm the topology details against the latest official kit before final submission. These are simulator settings, not physical calibration data.

### Error mitigation

Readout mitigation inverts a symmetric independent-bit assignment matrix and clips/renormalizes corrected outcome probabilities. ZNE is implemented as a quadratic fit to kernel probability estimates at gate-noise scales 1, 3, and 5, extrapolated to scale zero; readout error is held fixed across scales. This is **noise-parameter scaling**, not gate-folding ZNE. Extrapolation can be unstable, so raw/clipped diagnostics are saved.### Results and interpretation



The experiment used 4,096 shots per circuit, a fixed random seed of 42, and a reduced training subset of 12 samples. Results were evaluated separately for the two configured connectivity models. Processor A and Processor B produced identical classification and kernel-error metrics in this experiment.



\*\*Classification performance.\*\* The ideal condition achieved 76.67% accuracy, 72.22% precision, 86.67% recall, and an F1-score of 78.79%. All four noisy conditions achieved 80.00% accuracy, 76.47% precision, 86.67% recall, and an F1-score of 81.25%. Neither readout mitigation nor noise-parameter extrapolation changed the classification scores relative to the raw noisy condition. The higher noisy-condition scores do not establish that noise improves classification; they may reflect finite-shot variation and the sensitivity of a small test set to changes in the estimated kernel.



\*\*Kernel-estimation error.\*\* Relative to the same processor's ideal simulated kernel, the raw noisy test kernel had a mean absolute error (MAE) of 0.007753 and a root mean squared error (RMSE) of 0.010540. Readout mitigation reduced these to 0.003265 and 0.004302, respectively. Quadratic noise-parameter extrapolation alone achieved an MAE of 0.004974 and an RMSE of 0.006948. Combining extrapolation with readout mitigation achieved the lowest errors: 0.001717 MAE and 0.002168 RMSE. Compared with the raw noisy condition, this represents reductions of approximately 77.85% in MAE and 79.43% in RMSE.



These results indicate that the mitigation procedures improved agreement between the estimated kernel and the ideal simulated reference, but did not improve the measured classification scores beyond the raw noisy condition. Kernel-estimation quality and classification performance are related but distinct outcomes.



\*\*Processor comparison and computational cost.\*\* The two configured processor models yielded identical measured classification and kernel-error values. The small two-qubit feature map may not sufficiently expose differences between their connectivity graphs. The compiled depth, two-qubit gate counts, SWAP overhead, and circuit-width measurements should therefore be interpreted alongside the resource table rather than as evidence of general hardware equivalence. ZNE-style noise-parameter extrapolation also required more runtime than the raw and readout-mitigated conditions, due to additional simulations at multiple noise scales.



\### Limitations and conclusion



This study is a local, finite-shot Aer simulation using a reduced training subset and a simplified synthetic noise model. Processor A's chain and Processor B's guide-reference connectivity must be checked against the latest official Challenge Kit before submission. The reported results do not establish quantum advantage or physical hardware performance. In addition, the extrapolation method scales simulated gate-noise probabilities; it is not gate-folding ZNE.



Within these limitations, combined readout mitigation and quadratic noise-parameter extrapolation produced the closest test-kernel estimates to the ideal simulated reference. However, classification accuracy and F1-score remained unchanged across all noisy conditions. Reproducibility is supported by the fixed seed, processor configuration files, dependency list, run metadata, and exported result tables and figures.

