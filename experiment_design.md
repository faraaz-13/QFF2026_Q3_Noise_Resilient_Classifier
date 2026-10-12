# Experiment design

- Dataset: make_moons(100, noise=0.1, random_state=42).
- Split: stratified 70/30 with seed 42; training capped to 6 points per class for computational tractability.
- Feature encoding: two qubits; rotations and CX entanglement.
- Kernel: compute–uncompute, estimate P(00).
- Classifier: scikit-learn SVC with precomputed kernel.
- Conditions: ideal, noisy raw, noisy plus readout mitigation, noise-parameter ZNE, ZNE plus readout mitigation.
- Shot count: 4096 per circuit for the final run.
- Noise: p1=0.001, pCX=0.005, readout p=0.01.
- Compiler: optimization_level=1, seed_transpiler=42.
- Processor A: 5 nodes, participant-defined chain.
- Processor B: 9 nodes, guide-reference graph transcribed from supplied instructions.
- ZNE: quadratic extrapolation at scales 1, 3, 5; this is not gate folding.
- No physical backend execution and no quantum-advantage claim.
