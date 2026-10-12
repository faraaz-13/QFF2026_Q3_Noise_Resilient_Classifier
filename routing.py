"""Routing note: the main script transpiles against each config's coupling map.
The current feature map uses two qubits and one CX-connected pair, so a SWAP is
not generally needed when the compiler can choose a connected physical pair.
"""
