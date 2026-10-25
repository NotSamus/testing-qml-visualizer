# Hybrid QMLP Visualizer

An interactive 3D visualization of a hybrid quantum-classical MLP (QMLP) trained on MNIST. A PennyLane quantum layer (`Rot` + ring of `CRX` gates, followed by Pauli-Z measurements) feeds a classical linear layer, and the whole computation is laid out as a force-directed graph in the browser.

The generated page supports two data encodings side by side:

- **Angle** — `AngleEmbedding` at the start of every layer, one feature per qubit.
- **Amplitude** — `AmplitudeEmbedding` once, with `2**n_qubits` normalized amplitudes.

## Files

| Path | Description |
| --- | --- |
| `visualizer.py` | Builds the models, loads MNIST examples, exports the standalone HTML page. |
| `student_qmlp_viz.html` | Generated interactive visualization (open directly in a browser). |
| `mnist/` | MNIST idx files used for example digits. |

## Requirements

- Python 3.9+
- `numpy`, `torch`, `pennylane`
- Optional, for example digits and image resizing: `idx2numpy`, `scikit-learn`, `Pillow`, `torchvision`

The exported page loads Three.js and `3d-force-graph` from CDNs, so viewing it requires an internet connection.

## Usage

```bash
python visualizer.py
```

This precompiles every trained checkpoint named `qmlp-mnist-run1-layer<N>-<Encoding>.pt` in the folder (for example `qmlp-mnist-run1-layer2-Angle.pt`) into one `student_qmlp_viz.html`. The page has buttons to switch encoding (Angle / Amplitude) and layer count.

To include only some layer counts:

```bash
python visualizer.py --layers 2 10
```

With `--layers`, a missing checkpoint is built with random weights and the page shows a warning that its predictions mean nothing. With no checkpoints at all, the script builds 2-layer models with random weights as a preview.

## Notes

- Example digits are optional. If the MNIST files or optional dependencies are missing, the visualization still builds without them.
- The quantum layer is a 9-qubit, ring-entangling circuit; the classical head is a single `Linear(9, 10)` layer.

## License

No license specified.
