import base64
import json
import math
import struct
import zlib
import numpy as np


def _png_data_url(img):
    """Encode a 2-D image (values 0–1 or 0–255) as a grayscale PNG data URL, using only the standard library."""
    img = np.asarray(img, dtype=float)
    if img.ndim != 2:
        raise ValueError(f"Sample images must be 2-D, got shape {img.shape}.")
    # Keep true brightness (no contrast stretch) so the picture matches what the model's pipeline sees
    img = np.round(255 * np.clip(img / 255 if img.max() > 1 else img, 0, 1)).astype(np.uint8)
    h, w = img.shape
    raw = b"".join(b"\x00" + row.tobytes() for row in img)
    chunk = lambda tag, data: struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def export_student_qmlp_viz(model, n_qubits=None, n_layers=None, out_html="student_qmlp_viz.html",
                            samples=None, encoding="angle"):
    """Write one model to its own page (see build_view for the arguments)."""
    write_page([build_view(model, samples, encoding, n_qubits, n_layers)], out_html)


def write_page(views, out_html="student_qmlp_viz.html"):
    """Write the interactive page; with several views (e.g. angle and amplitude models) a button switches between them."""
    payload = json.dumps(views).replace("</", "<\\/")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE.replace("__VIEWS_JSON__", payload))
    print(f"[+] Student visualizer saved to {out_html}")


def build_view(model, samples=None, encoding="angle", n_qubits=None, n_layers=None, note=None):
    """Build the 3D graph data for one hybrid quantum-classical MLP.

    n_qubits and n_layers are read from the model's weights when left as None.
    note: optional warning shown on the page (e.g. that the weights are untrained).
    encoding: "angle" (AngleEmbedding at the start of every layer, one number per qubit) or
        "amplitude" (AmplitudeEmbedding once at the start, 2**n_qubits numbers as the state's amplitudes).
    samples: optional list of example digits to show and animate. Each is a dict with
        "image":      the digit (2-D array, e.g. 28x28),
        "features":   the numbers fed to the embedding (after scaling and PCA): n_qubits for angle,
                      2**n_qubits for amplitude (normalized here, as AmplitudeEmbedding(normalize=True) does),
        "label":      (optional) the true digit,
        "downscaled": (optional, angle only) the shrunken image (e.g. 3x3) the features came from,
        "expvals":    (optional) the ⟨Z⟩ values the quantum layer gives for it; the scores and the
                      prediction are computed from them with the model's classical layer.
    """
    encoding = encoding.lower()
    if encoding not in ("angle", "amplitude"):
        raise ValueError(f"encoding must be 'angle' or 'amplitude', got {encoding!r}.")
    amp_mode = encoding == "amplitude"

    state_dict = model.state_dict()
    if n_layers is None:
        n_layers = sum(k.startswith("qlayer.rot_layer_") for k in state_dict)
    if n_qubits is None:
        n_qubits = state_dict["qlayer.rot_layer_0"].shape[0]

    fc_weights = state_dict["fc.weight"].cpu().numpy()  # (num_classes, n_qubits)
    fc_bias = state_dict["fc.bias"].cpu().numpy() if "fc.bias" in state_dict else np.zeros(fc_weights.shape[0])
    num_classes = fc_weights.shape[0]

    # --- Example digits: check them and work out what the page needs for each ---
    samples = samples or []
    has_small = not amp_mode and bool(samples) and samples[0].get("downscaled") is not None
    n_feats = 2 ** n_qubits if amp_mode else n_qubits
    examples = []
    for i, s in enumerate(samples):
        x = np.asarray(s["features"], dtype=float).ravel()
        if x.size != n_feats:
            raise ValueError(f"Sample {i}: 'features' needs {n_feats} values for {encoding} encoding, got {x.size}.")
        if not amp_mode and (s.get("downscaled") is not None) != has_small:
            raise ValueError(f"Sample {i}: give 'downscaled' for every sample or for none.")
        ex = {"label": None if s.get("label") is None else int(s["label"]),
              "image": _png_data_url(s["image"]),
              "small": _png_data_url(s["downscaled"]) if has_small else None,
              "x": None, "amp": None, "p1": None, "z": None, "score": None, "p": None, "pred": None}
        if amp_mode:
            amp = x / (np.linalg.norm(x) or 1.0)  # What AmplitudeEmbedding(normalize=True) does
            # Chance each qubit reads |1⟩ on its own; PennyLane puts wire 0 on the leftmost bit of the basis state
            bits = (np.arange(amp.size)[:, None] >> (n_qubits - 1 - np.arange(n_qubits))) & 1
            ex.update(amp=np.round(amp, 4).tolist(), p1=np.round((amp ** 2) @ bits, 4).tolist())
        else:
            ex["x"] = np.round(x, 4).tolist()
        if s.get("expvals") is not None:
            zv = np.asarray(s["expvals"], dtype=float).ravel()
            if zv.size != n_qubits:
                raise ValueError(f"Sample {i}: 'expvals' needs {n_qubits} values (one per qubit), got {zv.size}.")
            score = fc_weights @ zv + fc_bias        # Same as the model's classical layer
            p = np.exp(score - score.max())
            p /= p.sum()                             # Softmax = exp of the model's log_softmax output
            ex.update(z=np.round(zv, 4).tolist(), score=np.round(score, 4).tolist(),
                      p=np.round(p, 4).tolist(), pred=int(np.argmax(score)))
        examples.append(ex)

    nodes = []
    links = []

    layer_spacing = 90  # Distance between stages along the X-axis
    layer_gap = 45      # Same ring spacing for every depth, so models compare fairly and depth shows as length
    ring_radius = 40    # Radius of the qubit ring so students see the circular entanglement
    meas_x = 2 * layer_spacing + (n_layers - 1) * layer_gap
    fc_x = meas_x + 1.2 * layer_spacing
    x0 = -fc_x / 2      # Center the network on the origin so the camera frames it
    meas_x, fc_x = meas_x + x0, fc_x + x0
    draw_reupload = not amp_mode and n_layers <= 10  # Beyond that the arcs (9 per layer) bury everything else

    def ring(q):
        angle = 2 * math.pi * q / n_qubits
        return angle, ring_radius * math.cos(angle), ring_radius * math.sin(angle)

    # Colors checked for colour-blind separation on the dark background (blue/red = sign, gray = plain wire)
    POS, NEG, WIRE = "#3987e5", "#e66767", "#898781"
    INPUT, QUANTUM, MEASURE, OUTPUT = "#c3c2b7", "#199e70", "#eda100", "#008300"

    def signed_color(v):
        return POS if v >= 0 else NEG

    # --- STAGE 0: the qubits as they start (Angle: one PCA feature each; Amplitude: one shared state) ---
    every_layer = " at the start of every layer" if n_layers > 1 else ""
    for q in range(n_qubits):
        _, y, z = ring(q)
        if amp_mode:
            name = f"Qubit {q} after amplitude encoding"
            info = (f"<h3>Qubit {q} after amplitude encoding</h3><p>The image is spread over the joint state of all "
                    f"{n_qubits} qubits, so no single number belongs to qubit {q}.</p>")
        else:
            name = f"PCA feature x{q} → qubit {q}"
            info = (f"<h3>Input · PCA feature x<sub>{q}</sub></h3>"
                    f"<p>Loaded onto qubit {q} as the rotation RX(x<sub>{q}</sub>){every_layer}.</p>")
        nodes.append({
            "id": f"In_Q{q}", "q": q, "name": name, "info": info,
            "group": "input",
            "fx": x0, "fy": y, "fz": z,
            "color": INPUT,
            "val": 4
        })

    # --- Example digit pictures: image → (shrunken image | amplitudes) → qubits ---
    # The page swaps in each example digit; "image"/"small"/"amp" name which picture of the example to show,
    # and "bars" which chart goes with it ("x": angle features, "p1": each qubit's chance of |1⟩).
    if examples:
        h, w = np.shape(samples[0]["image"])
        middle = None
        if amp_mode:
            rows, cols = 2 ** (n_qubits // 2), 2 ** (n_qubits - n_qubits // 2)
            step1 = f"Step 1: scale its pixels to 0–1 and let PCA compress them to {n_feats} numbers."
            middle = {
                "id": "Amps", "image": "amp", "size": 32, "aspect": cols / rows, "bars": "p1",
                "title": f"The {n_feats} amplitudes",
                "name": f"{n_feats} amplitudes of the {n_qubits}-qubit state",
                "info": f"<p>Step 2: AmplitudeEmbedding divides the {n_feats} numbers by their length, so their squares add "
                        f"up to 1, and uses them as the amplitudes of the {n_qubits}-qubit state (2<sup>{n_qubits}</sup> = {n_feats}).</p>"
                        f"<p class='muted'>Each square is one basis state |q0 q1 … q{n_qubits - 1}⟩, read left to right, top to "
                        f"bottom ({rows}×{cols}). Blue = positive, red = negative, brighter = bigger. PCA puts the most "
                        f"important information in its first numbers, so the top rows are the brightest.</p>",
                "via": f"Scale + PCA: {h * w} pixels → {n_feats} numbers"
            }
        elif has_small:
            sh, sw = np.shape(samples[0]["downscaled"])
            step1 = f"Step 1: shrink it to {sh}×{sw} = {sh * sw} pixels."
            middle = {
                "id": "Small", "image": "small", "size": 34, "bars": "x",
                "title": f"Shrunk to {sh}×{sw} pixels",
                "name": f"Shrunk to {sh}×{sw} = {sh * sw} pixels",
                "info": f"<p>Step 2: the {sh * sw} pixel values are scaled to 0–1, then PCA mixes them into {n_qubits} new numbers, "
                        f"one per qubit. Each number is a rotation angle."
                        + (" Nothing is thrown away here: PCA only re-arranges the information." if sh * sw == n_qubits else "")
                        + "</p>",
                "via": f"Resize {h}×{w} → {sh}×{sw}"
            }
        nodes.append({
            "id": "Image", "image": "image", "size": 44, "group": "input", "bars": None if middle else "x",
            "name": f"Example digit ({h}×{w} pixels)",
            "info": (f"<p>The original {h}×{w} image ({h * w} pixels). {step1}</p>" if middle else
                     f"<p>These become {n_qubits} numbers (scaling, then PCA), one per qubit. Each number is a rotation angle.</p>"),
            "fx": x0 - (150 if middle else 80), "fy": 0, "fz": 0, "val": 4
        })
        if middle:
            nodes.append({**middle, "group": "input", "fx": x0 - 75, "fy": 0, "fz": 0, "val": 4})
            links.append({"source": "Image", "target": middle["id"], "stage": "input",
                          "label": middle["via"], "color": WIRE, "width": 1.5, "particles": 2})
        for q in range(n_qubits):
            # Colour and width follow the example digit on screen (set in the page)
            links.append({
                "source": middle["id"] if middle else "Image", "target": f"In_Q{q}", "stage": "input", "enc": q,
                "label": f"Amplitude state → qubit {q}" if amp_mode else f"PCA feature x{q} → RX(x{q}) on qubit {q}",
                "color": WIRE, "width": 1.5, "particles": 2
            })

    # Wire thickness is relative to the largest value in this model: trained weights can reach ±10 or more
    crx_max = max(float(np.abs(state_dict[f"qlayer.crx_layer_{l}"].cpu().numpy()).max()) for l in range(n_layers)) or 1.0
    fc_max = float(np.abs(fc_weights).max()) or 1.0

    # --- STAGE 1..N: Quantum Layers (Rot + CRX Ring; Angle also re-encodes at every layer) ---
    for l in range(1, n_layers + 1):
        x_pos = x0 + layer_spacing + (l - 1) * layer_gap
        rot_w = state_dict[f"qlayer.rot_layer_{l-1}"].cpu().numpy()  # (n_qubits, 3)
        crx_w = state_dict[f"qlayer.crx_layer_{l-1}"].cpu().numpy()  # (n_qubits, 1)

        for q in range(n_qubits):
            angle, y, z = ring(q)
            phi, theta, omega = (float(v) for v in rot_w[q])
            nxt, prv = (q + 1) % n_qubits, (q - 1) % n_qubits
            crx_out, crx_in = float(crx_w[q][0]), float(crx_w[prv][0])
            node_id = f"L{l}_Q{q}"
            if amp_mode:
                encode = "<tr><td>Starts from</td><td>amplitude state</td></tr>" if l == 1 else ""
            else:
                encode = f"<tr><td>Encode</td><td>RX(x<sub>{q}</sub>)</td></tr>"

            nodes.append({
                "id": node_id,
                "name": f"Layer {l} · qubit {q}<br>Rot(φ={phi:.2f}, θ={theta:.2f}, ω={omega:.2f})",
                "info": f"<h3>Layer {l} · qubit {q}</h3><table>{encode}"
                        f"<tr><td>Rot φ</td><td>{phi:+.3f} rad</td></tr>"
                        f"<tr><td>Rot θ</td><td>{theta:+.3f} rad</td></tr>"
                        f"<tr><td>Rot ω</td><td>{omega:+.3f} rad</td></tr>"
                        f"<tr><td>CRX q{q} → q{nxt}</td><td class='{'pos' if crx_out >= 0 else 'neg'}'>{crx_out:+.3f} rad</td></tr>"
                        f"<tr><td>CRX q{prv} → q{q}</td><td class='{'pos' if crx_in >= 0 else 'neg'}'>{crx_in:+.3f} rad</td></tr>"
                        f"</table><p class='muted'>In q{q} → q{nxt}, this qubit is the control; in q{prv} → q{q}, it is the one rotated.</p>",
                "group": "quantum",
                "fx": x_pos, "fy": y, "fz": z,
                "color": QUANTUM,
                "val": 5
            })

            # 1. Qubit wire into this layer
            if l == 1:
                links.append({
                    "source": f"In_Q{q}", "target": node_id, "stage": "input",
                    "label": f"Qubit {q} enters Layer 1" if amp_mode else f"x{q} encoded on qubit {q} as RX(x{q}), then Rot",
                    "color": WIRE, "width": 1.5, "particles": 2
                })
            else:
                links.append({
                    "source": f"L{l-1}_Q{q}", "target": node_id, "stage": "quantum",
                    "label": f"Qubit {q} carries its quantum state into Layer {l}",
                    "color": WIRE, "width": 1.5, "particles": 2
                })
                if draw_reupload:  # Data re-uploading: AngleEmbedding runs again at the start of this layer
                    links.append({
                        "source": f"In_Q{q}", "target": node_id, "stage": "input",
                        "label": f"Data re-uploading: x{q} encoded again as RX(x{q}) in Layer {l}",
                        "color": INPUT, "width": 0.6, "particles": 1,
                        # Same small bulge for every layer (curvature is a fraction of the wire's length)
                        "curvature": 30 / (layer_spacing + (l - 1) * layer_gap), "rotation": angle
                    })

            # 2. CRX Ring Entanglement: qubit q controls qubit (q+1) % n_qubits
            links.append({
                "source": node_id, "target": f"L{l}_Q{nxt}", "stage": "quantum",
                "label": f"CRX entanglement q{q} → q{nxt}: θ = {crx_out:+.2f} rad",
                "color": signed_color(crx_out),
                "width": 0.6 + 2.4 * abs(crx_out) / crx_max,
                "particles": 4
            })

    # --- STAGE N+1: Pauli-Z Measurement ---
    for q in range(n_qubits):
        strongest = sorted(range(num_classes), key=lambda c: -abs(fc_weights[c, q]))[:3]
        nodes.append({
            "id": f"Meas_Q{q}", "q": q,
            "name": f"Measurement ⟨Z<sub>{q}</sub>⟩ (classical output)",
            "info": f"<h3>Measurement ⟨Z<sub>{q}</sub>⟩</h3>"
                    f"<p>Qubit {q} becomes one number between −1 and +1.</p>"
                    f"<p>It matters most for:</p><table>"
                    + "".join(f"<tr><td>Digit {c}</td><td class='{'pos' if fc_weights[c, q] >= 0 else 'neg'}'>"
                              f"{fc_weights[c, q]:+.3f}</td></tr>" for c in strongest)
                    + "</table>",
            "group": "measure",
            "fx": meas_x,
            "fy": ((n_qubits - 1) / 2 - q) * 18,  # Unroll ring into a vertical spine, qubit 0 at the top
            "fz": 0,
            "color": MEASURE,
            "val": 4
        })
        links.append({
            "source": f"L{n_layers}_Q{q}", "target": f"Meas_Q{q}", "stage": "measure",
            "label": f"Measure Pauli-Z on q{q}",
            "color": WIRE, "width": 1.2, "particles": 2
        })

    # --- STAGE N+2: Classical Linear Layer (n_qubits -> classes) ---
    for c in range(num_classes):
        order = sorted(range(n_qubits), key=lambda q: -abs(fc_weights[c, q]))
        nodes.append({
            "id": f"Out_{c}", "c": c,
            "name": f"Digit {c}",
            "info": f"<h3>Digit {c}</h3>"
                    f"<p>score = Σ w<sub>q</sub>·⟨Z<sub>q</sub>⟩ + b</p>"
                    f"<p>Bias b = {float(fc_bias[c]):+.3f}. Weights, strongest first:</p><table>"
                    + "".join(f"<tr><td>⟨Z<sub>{q}</sub>⟩</td><td class='{'pos' if fc_weights[c, q] >= 0 else 'neg'}'>"
                              f"{fc_weights[c, q]:+.3f}</td></tr>" for q in order)
                    + "</table>",
            "group": "output",
            "fx": fc_x,
            "fy": ((num_classes - 1) / 2 - c) * 18,  # Digit 0 at the top
            "fz": 0,
            "color": OUTPUT,
            "val": 6
        })
        for q in range(n_qubits):
            w = float(fc_weights[c, q])
            # All weights are exported; the slider in the page hides the weak ones
            links.append({
                "source": f"Meas_Q{q}", "target": f"Out_{c}", "stage": "output", "kind": "fc", "w": w,
                "label": f"Linear weight ⟨Z<sub>{q}</sub>⟩ → Digit {c}: {w:+.3f}",
                "color": signed_color(w),
                "width": 0.4 + 2.6 * abs(w) / fc_max,
                "particles": 1
            })

    meta = {"n_qubits": int(n_qubits), "n_layers": int(n_layers), "num_classes": int(num_classes), "encoding": encoding,
            "colors": {"input": INPUT, "quantum": QUANTUM, "measure": MEASURE, "output": OUTPUT, "pos": POS, "neg": NEG},
            "samples": examples, "note": note, "reupload_drawn": draw_reupload}
    return {"nodes": nodes, "links": links, "meta": meta}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hybrid QMLP Visualizer</title>
  <style>
    /* Every size below comes from fixed scales: spacing 4/8/12/16/24, type 12/14/16/18, opacity .05/.1/.4 */
    :root { --bg: #0d0d0d; --panel: #1a1a19; --raised: #383835; --line: rgba(255, 255, 255, 0.1);
            --text: #f0efec; --muted: #a3a29c; --pos: #3987e5; --neg: #e66767; --wire: #898781; --prob: #008300; --warn: #fab219; }
    body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.5 system-ui, -apple-system, sans-serif; overflow: hidden; }
    /* Floating panels stand apart from the scene by lightness, not outlines */
    .panel { position: absolute; z-index: 10; box-sizing: border-box; max-height: calc(100vh - 32px); padding: 16px;
             overflow-y: auto; background: var(--panel); border-radius: 12px; }
    #hud { top: 16px; left: 16px; width: 256px; }
    #explain { top: 16px; right: 16px; width: 384px; }  /* ~50 characters per line for the reading panel */
    summary { display: flex; justify-content: space-between; align-items: center; cursor: pointer; list-style: none; }
    summary::-webkit-details-marker { display: none; }
    summary::after { content: '▾'; color: var(--muted); }
    details:not([open]) summary::after { content: '▸'; }
    h2 { margin: 0; font-size: 18px; font-weight: 600; }
    h3 { margin: 0 0 8px; font-size: 16px; font-weight: 600; }
    h4, #explain summary h2 { margin: 24px 0 8px; font-size: 12px; font-weight: 600; color: var(--muted);
                              text-transform: uppercase; letter-spacing: .05em; }
    #explain summary h2 { margin: 0; }  /* The panel title is just a label; the content below carries the heading */
    #details { margin-top: 8px; }
    p { margin: 8px 0; }
    ol { margin: 8px 0; padding-left: 16px; } li { margin: 4px 0; }
    label { display: block; margin-top: 8px; }
    .muted { color: var(--muted); font-size: 12px; }
    .legend p { margin: 4px 0; }
    button { font: inherit; color: inherit; cursor: pointer; }
    button:focus-visible, input:focus-visible, summary:focus-visible { outline: 2px solid var(--text); outline-offset: 2px; }
    /* Hierarchy: Play is the one primary action; tour stages and choices are secondary; camera links are tertiary */
    .primary { width: 100%; margin-top: 8px; padding: 8px 12px; border: 0; border-radius: 8px;
               background: var(--text); color: var(--panel); font-weight: 600; }
    .primary:hover { background: #ffffff; }
    .stage { display: flex; align-items: center; gap: 8px; width: 100%; margin-bottom: 4px; padding: 8px 12px; border: 0;
             border-radius: 8px; background: rgba(255, 255, 255, .05); text-align: left; }
    .stage:hover { background: rgba(255, 255, 255, .1); }
    .stage[aria-pressed="true"] { background: rgba(255, 255, 255, .1); box-shadow: inset 4px 0 0 var(--text); font-weight: 600; }
    .seg { display: flex; gap: 4px; padding: 4px; border-radius: 8px; background: rgba(255, 255, 255, .05); }
    .seg button { flex: 1; padding: 4px 8px; border: 0; border-radius: 4px; background: none; color: var(--muted); }
    .seg button:hover:not(:disabled) { color: var(--text); }
    .seg button[aria-pressed="true"] { background: var(--raised); color: var(--text); font-weight: 600; }
    .seg button:disabled { opacity: .4; cursor: default; }
    .links { display: flex; flex-wrap: wrap; column-gap: 16px; margin-top: 4px; }
    .link { padding: 4px 0; border: 0; background: none; color: var(--muted); white-space: nowrap; }
    .link:hover { color: var(--text); text-decoration: underline; }
    .result { font-weight: 600; }
    .note { padding-left: 8px; border-left: 4px solid var(--warn); }
    .dot { flex: none; display: inline-block; width: 12px; height: 12px; margin-right: 8px; border-radius: 50%; vertical-align: -2px; }
    .stage .dot { margin-right: 0; }
    table { border-collapse: collapse; width: 100%; }
    td { padding: 4px 0; } td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
    /* Numbers stay in text colour; a small swatch or bar carries the sign */
    td.pos::before, td.neg::before { content: ''; display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 8px; }
    td.pos::before, .bar.pos { background: var(--pos); } td.neg::before, .bar.neg { background: var(--neg); }
    .bar.prob { background: var(--prob); }
    .bars td:nth-child(2) { width: 40%; padding: 0 8px; }
    .bar { display: inline-block; height: 8px; min-width: 2px; border-radius: 2px; vertical-align: middle; }
    .digit { display: block; width: 128px; height: 128px; margin: 8px 0; image-rendering: pixelated; border-radius: 8px; border: 1px solid var(--line); }
    .digit.amp { width: 256px; }
    input[type=range] { width: 100%; margin: 4px 0 0; accent-color: var(--text); }
    .tip { padding: 8px 12px; border-radius: 8px; background: rgba(13, 13, 13, .8); font-size: 14px; }
    @media (max-width: 760px) {
      #hud { right: 16px; width: auto; max-height: 45vh; }
      #explain { top: auto; bottom: 16px; left: 16px; width: auto; max-height: 35vh; }
    }
  </style>
  <script type="importmap">
  { "imports": {
      "three": "https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js",
      "three/webgpu": "https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.webgpu.js",
      "three/examples/": "https://cdn.jsdelivr.net/npm/three@0.180.0/examples/"
  } }
  </script>
</head>
<body>
  <details id="hud" class="panel" open>
    <summary><h2>Hybrid QMLP</h2></summary>
    <p class="muted" id="subtitle"></p>
    <p class="note" id="note" hidden></p>
    <div id="viewPick" hidden><h4>Encoding</h4><div class="seg" id="views"></div></div>
    <div id="layerPick" hidden><h4>Quantum layers</h4><div class="seg" id="layers"></div></div>
    <div id="exampleBox" hidden>
      <h4>Example digits</h4>
      <p class="result" id="result"></p>
      <button class="primary" id="play" hidden></button>
      <div id="speedBox" hidden>
        <label for="speed">Speed <b id="speedVal">1×</b></label>
        <input id="speed" type="range" min="0.25" max="3" step="0.25" value="1">
      </div>
    </div>
    <h4>Tour the model</h4>
    <div id="stages"></div>
    <div class="links">
      <button class="link" id="recenter">⌖ Recenter</button>
      <button class="link" id="overview">↺ Show whole model</button>
    </div>
    <h4>Wire colours</h4>
    <div class="legend">
      <p><span class="dot" style="background:var(--pos)"></span>Positive weight or angle</p>
      <p><span class="dot" style="background:var(--neg)"></span>Negative weight or angle</p>
      <p><span class="dot" style="background:var(--wire)"></span>Plain wire: carries a qubit</p>
      <p class="muted">Thicker wire = larger value, compared within this model.</p>
    </div>
    <h4>Classical weights</h4>
    <label for="thr">Hide weights smaller than <b id="thrVal">0.05</b></label>
    <input id="thr" type="range" min="0" max="1" step="0.01" value="0.05">
    <p class="muted" id="thrCount"></p>
  </details>
  <details id="explain" class="panel" open>
    <summary><h2>Explanation</h2></summary>
    <div id="details" aria-live="polite"></div>
  </details>
  <div id="graph"></div>
  <script type="module">
    import ForceGraph3D from 'https://esm.sh/3d-force-graph@1.80.1?external=three';
    import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
    import { Sprite, SpriteMaterial, TextureLoader, CanvasTexture, NearestFilter, SRGBColorSpace } from 'three';

    const VIEWS = __VIEWS_JSON__;     // One entry per model (e.g. angle and amplitude encoding)
    const C = VIEWS[0].meta.colors;
    const $ = id => document.getElementById(id);

    // --- State of the view on screen (filled in by loadView) ---
    let data, n_qubits, n_layers, num_classes, AMP, N_AMP, SAMPLES, hasZ, pics, textures, STAGES, fcW, span, mid;
    let cur = 0;     // Which example digit is on screen
    let shown = 3;   // How far (stage 0–3) the current digit has got; values appear as it arrives during Play
    let focus = null;  // Stage being toured or played, or null to show everything
    let thr = 0.05;
    const STEPS = ['input', 'quantum', 'measure', 'output'];

    const S = () => SAMPLES[cur];
    const fmt = (v, d = 3) => (v >= 0 ? '+' : '') + v.toFixed(d);
    const pct = v => `${Math.round(v * 100)}%`;
    const sign = v => v >= 0 ? 'pos' : 'neg';
    const title = s => s.label === null ? 'Example digit' : `Example digit: ${s.label}`;
    const verdict = s => s.label === null ? '' : s.pred === s.label ? ' ✓ correct' : ' ✗ wrong';
    const picture = (src, alt, cls = 'digit') => `<img class="${cls}" src="${src}" alt="${alt}">`;
    const bars = (rows, full) => `<table class="bars">${rows.map(([label, v, text, cls]) =>
      `<tr><td>${label}</td><td><span class="bar ${cls}" style="width:${Math.min(Math.abs(v) / full, 1) * 100}%"></span></td>` +
      `<td>${text}</td></tr>`).join('')}</table>`;
    const xMax = s => Math.max(...s.x.map(Math.abs)) || 1;
    const featureBars = s => bars(s.x.map((v, q) => [`x<sub>${q}</sub> → qubit ${q}`, v, fmt(v), sign(v)]), xMax(s));
    const p1Bars = s => `<h4>Each qubit on its own: chance of reading |1⟩</h4>` +
      bars(s.p1.map((v, q) => [`qubit ${q}`, v, pct(v), 'prob']), 1);
    const zBars = s => bars(s.z.map((v, q) => [`⟨Z<sub>${q}</sub>⟩`, v, fmt(v), sign(v)]), 1);
    const probBars = s => bars(s.p.map((v, c) => [c === s.pred ? `<b>Digit ${c} ◀</b>` : `Digit ${c}`, v, pct(v), 'prob']), 1);

    // Amplitude heatmap: one square per basis state, blue = positive, red = negative, brighter = bigger
    const rgb = hex => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
    function ampURL(a) {
      const cols = 2 ** Math.ceil(n_qubits / 2), rows = a.length / cols, big = Math.max(...a.map(Math.abs)) || 1;
      const canvas = document.createElement('canvas');
      canvas.width = cols; canvas.height = rows;
      const g = canvas.getContext('2d'), img = g.createImageData(cols, rows);
      const zero = rgb('#383835'), pos = rgb(C.pos), neg = rgb(C.neg);
      a.forEach((v, i) => {
        const t = Math.sqrt(Math.abs(v) / big), end = v >= 0 ? pos : neg;  // Square root so small amplitudes still show
        for (let k = 0; k < 3; k++) img.data[4 * i + k] = zero[k] + (end[k] - zero[k]) * t;
        img.data[4 * i + 3] = 255;
      });
      g.putImageData(img, 0, 0);
      return canvas.toDataURL();
    }
    const loadPixels = url => {
      if (!url) return null;
      const t = new TextureLoader().load(url);
      t.magFilter = NearestFilter;  // Keep pixels crisp
      t.colorSpace = SRGBColorSpace;
      return t;
    };

    function makeStages() {
      return {
        input: { title: '1 · Encode the image', color: C.input, text: (AMP ?
          `<p><b>Amplitude encoding.</b> Each image's pixels are scaled to 0–1 and PCA compresses them to ${N_AMP} numbers.
           AmplitudeEmbedding divides those numbers by their length (so their squares add up to 1) and uses them as the
           ${N_AMP} amplitudes of the ${n_qubits}-qubit state, since 2<sup>${n_qubits}</sup> = ${N_AMP}.</p>
           <p>So ${n_qubits} qubits hold ${N_AMP} numbers, but only together: no single number belongs to one qubit.
           The image is loaded once, before the first layer.</p>` :
          `<p><b>Angle encoding.</b> Each image becomes ${n_qubits} numbers, one per qubit. AngleEmbedding loads number
           x<sub>q</sub> as the rotation RX(x<sub>q</sub>) on qubit q: a larger number (up to π) rotates the qubit further
           away from |0⟩.</p>` +
           (n_layers > 1 ? `<p>The numbers are loaded again at the start of every layer. This is called <b>data re-uploading</b>` +
           (data.meta.reupload_drawn ? ' and is shown by the thin curved gray wires.</p>'
             : `. With ${n_layers} layers its wires are left out so the picture stays readable.</p>`) : '')) +
          (SAMPLES.length ? '<p>Follow the example digit on the far left, and click each picture to see each step.</p>' : '') },
        quantum: { title: '2 · Quantum layers', color: C.quantum, text:
          `<p>Each of the ${n_layers} layer${n_layers > 1 ? 's' : ''} has trainable gates:</p>
           <p><b>Rot(φ, θ, ω)</b> turns each qubit by three angles (RZ·RY·RZ). These angles are what training learns.</p>
           <p><b>CRX</b> gates link each qubit to its neighbour in a ring (q${n_qubits - 1} links back to q0). The control qubit
           decides whether its neighbour gets rotated, which <b>entangles</b> them so information mixes between qubits.
           Thicker ring wires mean a larger CRX angle.</p>` },
        measure: { title: '3 · Measure ⟨Z⟩', color: C.measure, text:
          `<p>Each qubit is measured along Z. The model keeps the average result <b>⟨Z⟩</b>, a number from −1 (always |1⟩)
           to +1 (always |0⟩).</p><p>This is where the quantum state becomes ${n_qubits} ordinary numbers.</p>` },
        output: { title: '4 · Classical layer', color: C.output, text:
          `<p>An ordinary neural-network layer. Each digit's score is Σ w<sub>q</sub>·⟨Z<sub>q</sub>⟩ + bias, and the digit
           with the highest score is the prediction.</p><p>Blue wire (positive weight): the score moves the same way as
           ⟨Z<sub>q</sub>⟩. Red wire (negative weight): it moves the opposite way. Use the weights slider to hide weak weights.</p>` },
      };
    }

    // What the current example digit looks like at each stage (empty when there is nothing to show)
    function stageSample(st) {
      const s = S();
      if (!s) return '';
      if (st === 'input') return AMP
        ? `<h4>${title(s)}: its ${N_AMP} amplitudes</h4>${picture(pics.amp[cur], 'Amplitudes', 'digit amp')}${p1Bars(s)}`
        : `<h4>${title(s)}: its ${n_qubits} numbers</h4>${featureBars(s)}`;
      if (st === 'measure' && s.z) return `<h4>${title(s)}: ⟨Z⟩ values</h4>${zBars(s)}`;
      if (st === 'output' && s.p) return `<h4>${title(s)}: predicted ${s.pred} (${pct(s.p[s.pred])})${verdict(s)}</h4>${probBars(s)}`;
      return '';
    }
    // A node's own values for the current example digit
    function nodeSample(n) {
      const s = S();
      if (!s) return '';
      if (n.group === 'input' && AMP) {
        return `<p>For the example digit, qubit ${n.q} on its own reads |1⟩ with probability ${pct(s.p1[n.q])}.</p>`;
      }
      if (n.group === 'input') {
        const v = s.x[n.q];
        return `<p>For the example digit, x<sub>${n.q}</sub> = ${fmt(v)}. On its own, before any trained gate, RX(${fmt(v)})
          would make qubit ${n.q} read |1⟩ with probability sin²(x/2) = ${pct(Math.sin(v / 2) ** 2)}.</p>`;
      }
      if (n.group === 'measure' && s.z) {
        const v = s.z[n.q];
        return `<p>For the example digit, ⟨Z<sub>${n.q}</sub>⟩ = ${fmt(v)}: qubit ${n.q} reads |0⟩ ${pct((1 + v) / 2)} of the time.</p>`;
      }
      if (n.group === 'output' && s.p) {
        return `<p>For the example digit: score ${fmt(s.score[n.c])}, probability ${pct(s.p[n.c])}` +
          `${n.c === s.pred ? ', the highest, so this is the prediction' : ''}.</p>`;
      }
      return '';
    }
    function infoFor(n) {
      if (!n.image) return n.info + nodeSample(n);
      const s = S(), t = n.title || title(s);
      const chart = n.bars === 'x' ? featureBars(s) : n.bars === 'p1' ? p1Bars(s) : '';
      return `<h3>${t}</h3>${picture(pics[n.image][cur], t, n.image === 'amp' ? 'digit amp' : 'digit')}${n.info}${chart}`;
    }
    const linkText = l => {
      if (l.enc === undefined || !S()) return l.label;
      return AMP ? `Amplitude state → qubit ${l.enc}: on its own it reads |1⟩ ${pct(S().p1[l.enc])} of the time`
        : `PCA feature x${l.enc} = ${fmt(S().x[l.enc])} → RX(${fmt(S().x[l.enc])}) on qubit ${l.enc}`;
    };

    // --- 3D view ---
    const DIM = '#383835';  // Other stages fade to this but stay visible as context
    const lit = s => !focus || s === focus;
    const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const tip = html => `<div class="tip">${html}</div>`;

    // Output balls grow with the predicted probability once the digit has reached them
    const valOf = n => n.group === 'output' && hasZ && shown >= 3 ? 3 + 15 * S().p[n.c] : n.val;
    const tagText = n => {
      const s = S();
      if (!s || !s.z) return '';
      if (n.group === 'measure') return shown >= 2 ? `q${n.q}  ${fmt(s.z[n.q], 2)}` : '';
      return shown >= 3 ? `${n.c}:  ${pct(s.p[n.c])}${n.c === s.pred ? '  ◀ predicted' : ''}` : '';
    };
    // A small text label beside a node; redrawn whenever the example digit or Play step changes
    function makeTag(n) {
      const canvas = document.createElement('canvas');
      canvas.width = 320; canvas.height = 64;
      const map = new CanvasTexture(canvas);
      map.colorSpace = SRGBColorSpace;
      const sprite = new Sprite(new SpriteMaterial({ map, depthTest: false }));
      sprite.scale.set(50, 10, 1);  // Same 5:1 shape as the canvas; readable from the whole-model view
      sprite.center.set(0, 0.5);  // Anchor the left edge so the label sits to the right of the ball
      sprite.renderOrder = 20;    // The library draws wires at 10; labels must come after them to stay readable
      n.__tag = { canvas, map, sprite };
      drawTag(n);
      return sprite;
    }
    function drawTag(n) {
      const { canvas, map, sprite } = n.__tag, g = canvas.getContext('2d'), text = tagText(n);
      g.clearRect(0, 0, canvas.width, canvas.height);
      sprite.visible = !!text;
      sprite.position.x = Math.cbrt(valOf(n)) * 4 + 2;  // Just outside the ball (the library's radius is cbrt(val) × 4)
      if (!text) return;
      g.font = '600 32px system-ui, -apple-system, sans-serif';
      g.fillStyle = 'rgba(13, 13, 13, 0.85)';
      g.beginPath(); g.roundRect(0, 6, g.measureText(text).width + 24, 52, 12); g.fill();
      g.fillStyle = '#f0efec';
      g.textBaseline = 'middle';
      g.fillText(text, 12, 33);
      map.needsUpdate = true;
    }

    const Graph = ForceGraph3D()($('graph'))
      .backgroundColor('#0d0d0d')
      .nodeColor(n => lit(n.group) ? n.color : DIM)
      .nodeVal(valOf)
      .nodeThreeObject(n => {
        if (n.image) {  // The example digit pictures replace the ball
          const mat = new SpriteMaterial({ map: textures[n.image][cur], depthTest: false });
          const sprite = new Sprite(mat);
          sprite.scale.set(n.size * (n.aspect || 1), n.size, 1);
          sprite.renderOrder = 15;  // Draw the picture on top of its own wires (the library draws wires at 10)
          n.__mat = mat;
          return sprite;
        }
        return hasZ && (n.group === 'measure' || n.group === 'output') ? makeTag(n) : null;  // Label beside the ball
      })
      .nodeThreeObjectExtend(n => !n.image)
      .nodeLabel(n => tip(n.name))
      .linkColor(l => !lit(l.stage) ? DIM : l.enc !== undefined && S() && !AMP ? (S().x[l.enc] >= 0 ? C.pos : C.neg) : l.color)
      .linkWidth(l => l.enc === undefined || !S() ? l.width
        : AMP ? 0.5 + 2.5 * S().p1[l.enc] : 0.5 + 2.5 * Math.abs(S().x[l.enc]) / xMax(S()))
      .linkOpacity(0.6)
      .linkCurvature(l => l.curvature || 0)
      .linkCurveRotation(l => l.rotation || 0)
      .linkVisibility(l => l.kind !== 'fc' || Math.abs(l.w) >= thr)
      .linkLabel(l => tip(linkText(l)))
      .linkDirectionalParticles(l => !still && lit(l.stage) ? l.particles : 0)
      .linkDirectionalParticleWidth(2.0)
      .linkDirectionalParticleSpeed(0.006)
      .onNodeClick(n => showItem(infoFor(n), n.group))
      .onLinkClick(l => showItem(`<h3>${linkText(l)}</h3>`, l.stage))
      .onBackgroundClick(() => showStage(focus));

    const bloomPass = new UnrealBloomPass();
    bloomPass.strength = 0.6;
    bloomPass.radius = 0.4;
    bloomPass.threshold = 0.3;  // Only bright things glow, so the background stays black
    Graph.postProcessingComposer().addPass(bloomPass);
    addEventListener('resize', () => Graph.width(innerWidth).height(innerHeight));

    // Re-apply everything that depends on the stage focus or the example digit
    function redraw() {
      Graph.nodeVal(Graph.nodeVal()).nodeColor(Graph.nodeColor()).linkColor(Graph.linkColor())
        .linkWidth(Graph.linkWidth()).linkDirectionalParticles(Graph.linkDirectionalParticles());
      data.nodes.forEach(n => {
        if (n.__mat) { n.__mat.map = textures[n.image][cur]; n.__mat.needsUpdate = true; }
        if (n.__tag) drawTag(n);
      });
      document.querySelectorAll('[data-stage]').forEach(b => b.setAttribute('aria-pressed', b.dataset.stage === focus));
      const s = S();
      $('result').textContent = !s || !s.p ? '' : shown < 3 ? `${title(s)} → …`
        : `${title(s)} → predicted ${s.pred} (${pct(s.p[s.pred])})${verdict(s)}`;
    }

    // --- Details panel ---
    function showIntro() {
      $('details').innerHTML = `<h3>How to explore</h3><ol>
        ${ENCODINGS.length > 1 ? '<li>Switch between <b>Angle</b> and <b>Amplitude</b> encoding at the top left.</li>' : ''}
        ${LAYERS.length > 1 ? `<li>Pick how many <b>quantum layers</b> the model has (${LAYERS.join(', ')}).</li>` : ''}
        ${SAMPLES.length > 1 ? `<li>Press <b>Play</b> to watch ${SAMPLES.length} example digits go through the model.</li>` : ''}
        <li>Click a numbered stage in the tour to fly to it and read what it does.</li>
        <li>Click any ball or wire to see its trained values here.</li>
        <li>Drag to rotate, scroll to zoom, right-drag to pan. <b>Recenter</b> brings the whole model back into view.</li></ol>
        <p class="muted">Data flows left to right: image → qubits → measurements → digit scores.</p>`;
    }
    function showStage(s) {
      if (!s) return showIntro();
      const pic = s === 'input' && S() ? picture(S().image, title(S())) : '';
      $('details').innerHTML = `<h3>${STAGES[s].title}</h3>${STAGES[s].text}${pic}${stageSample(s)}`;
    }
    function showItem(html, s) {
      $('explain').open = true;
      $('details').innerHTML = `${html}<h4>About this stage: ${STAGES[s].title}</h4>${STAGES[s].text}`;
    }
    function showPlaying() {
      const s = S(), st = STEPS[shown];
      const now = stageSample(st) || `<p class="muted">The qubits go through ${n_layers} layer${n_layers > 1 ? 's' : ''} of trained Rot and CRX gates.</p>`;
      $('details').innerHTML = `<h3>Example ${cur + 1} of ${SAMPLES.length}</h3>${picture(s.image, title(s))}
        <p>Now at <b>${STAGES[st].title}</b></p>${now}`;
    }
    function showPaused() {
      $('details').innerHTML = `<h3>Paused on example ${cur + 1} of ${SAMPLES.length}</h3>${picture(S().image, title(S()))}
        ${stageSample('output')}<p class="muted">Click any ball or wire to see this digit's values. Press Play to continue.</p>`;
    }

    // --- Camera and guided tour ---
    function setFocus(s) {
      focus = s;
      redraw();
      showStage(s);
    }
    // The strip of screen between the side panels (the whole width on phones, where the panels stack)
    function freeBand() {
      const h = $('hud').getBoundingClientRect(), e = $('explain').getBoundingClientRect();
      return e.left > h.right ? [h.right, e.left] : [0, innerWidth];
    }
    const viewAngle = () => 2 * Math.tan(Graph.camera().fov * Math.PI / 360);
    // How far to slide the camera sideways so the scene centers in the free strip, at camera distance d
    const sideShift = d => ((freeBand()[0] + freeBand()[1]) / 2 - innerWidth / 2) * d * viewAngle() / innerHeight;
    function recenter(ms = 1000) {
      // Back off until the model fills ~85% of the free strip (and fits vertically)
      const [l, r] = freeBand(), view = viewAngle();
      const d = Math.max(span * innerHeight / (0.85 * (r - l) * view), 200 / (0.6 * view));
      const x = mid - sideShift(d);
      Graph.cameraPosition({ x: x - 0.25 * d, y: 0.25 * d, z: d }, { x, y: 0, z: 0 }, ms);
    }
    function overview(ms = 1000) {  // "Show whole model": stop Play and reset to the first example
      stop();
      cur = 0;
      setFocus(null);
      recenter(ms);
    }
    function tour(s) {
      stop();
      setFocus(s);
      // Frame the stage's nodes and its wires
      const stageXs = [...data.nodes.filter(n => n.group === s).map(n => n.fx),
                       ...data.links.filter(l => l.stage === s).flatMap(l => [l.source.fx, l.target.fx])];
      const lo = Math.min(...stageXs), hi = Math.max(...stageXs);
      const d = 260 + hi - lo, sx = (lo + hi) / 2 - sideShift(d);
      Graph.cameraPosition({ x: sx - 0.45 * d, y: 0.3 * d, z: d }, { x: sx, y: 0, z: 0 }, 1000);
    }

    // --- Play: each example digit lights up the stages in order, then the next digit starts ---
    const STEP_MS = [1200, 1200, 1500, 2200];  // Time on each stage at 1× speed
    let speed = 1;
    $('speed').oninput = e => {
      speed = +e.target.value;
      $('speedVal').textContent = `${speed}×`;
      Graph.linkDirectionalParticleSpeed(0.006 * speed);  // The flowing dots keep pace with Play
    };
    const playLabel = () => `▶ Play ${SAMPLES.length} example digits`;
    let timer = null;
    function playStep(k) {
      shown = k;
      focus = STEPS[k];
      redraw();
      showPlaying();
      timer = setTimeout(() => {
        if (k < STEPS.length - 1) return playStep(k + 1);
        cur = (cur + 1) % SAMPLES.length;
        playStep(0);
      }, STEP_MS[k] / speed);
    }
    function stop() {
      clearTimeout(timer);
      timer = null;
      shown = 3;
      $('play').textContent = playLabel();
    }
    function play() {
      if (timer) {  // Second press pauses on the current digit with all its values showing
        stop();
        focus = null;
        redraw();
        return showPaused();
      }
      $('play').textContent = '❚❚ Pause';
      $('explain').open = true;
      recenter();
      playStep(0);
    }

    // --- Weight threshold slider ---
    const countShown = () => $('thrCount').textContent = `Showing ${fcW.filter(w => w >= thr).length} of ${fcW.length} weights`;
    $('thr').oninput = e => {
      thr = +e.target.value;
      $('thrVal').textContent = thr.toFixed(2);
      countShown();
      Graph.linkVisibility(Graph.linkVisibility());
    };

    // --- Switch between models (e.g. angle and amplitude encoding) ---
    function loadView(i, ms = 1000) {
      data = structuredClone(VIEWS[i]);  // Fresh copy: the 3D library adds its own fields to nodes and links
      ({ n_qubits, n_layers, num_classes } = data.meta);
      AMP = data.meta.encoding === 'amplitude';
      N_AMP = 2 ** n_qubits;
      SAMPLES = data.meta.samples;
      hasZ = SAMPLES.length > 0 && SAMPLES[0].z !== null;
      cur = 0;
      shown = 3;
      pics = { image: SAMPLES.map(s => s.image), small: SAMPLES.map(s => s.small), amp: SAMPLES.map(s => s.amp && ampURL(s.amp)) };
      textures = Object.fromEntries(Object.entries(pics).map(([k, urls]) => [k, urls.map(loadPixels)]));
      STAGES = makeStages();
      const xs = data.nodes.map(n => n.fx);
      span = Math.max(...xs) - Math.min(...xs);
      mid = (Math.max(...xs) + Math.min(...xs)) / 2;
      fcW = data.links.filter(l => l.kind === 'fc').map(l => Math.abs(l.w));
      $('thr').max = Math.ceil(Math.max(...fcW) * 100) / 100;
      countShown();
      $('subtitle').textContent = `${n_qubits} qubits · ${n_layers} quantum layer${n_layers > 1 ? 's' : ''} · ` +
        `${AMP ? 'amplitude' : 'angle'} encoding · ${num_classes} classes`;
      $('exampleBox').hidden = SAMPLES.length === 0;
      $('play').hidden = $('speedBox').hidden = SAMPLES.length < 2;
      $('note').textContent = data.meta.note ? `⚠ ${data.meta.note}` : '';
      $('note').hidden = !data.meta.note;
      view = i;
      document.querySelectorAll('[data-enc]').forEach(b => b.setAttribute('aria-pressed', b.dataset.enc === encOf(VIEWS[i])));
      document.querySelectorAll('[data-layers]').forEach(b => {
        b.setAttribute('aria-pressed', +b.dataset.layers === n_layers);
        b.disabled = !VIEWS.some(v => encOf(v) === encOf(VIEWS[i]) && layersOf(v) === +b.dataset.layers);
      });
      Graph.graphData(data);
      overview(ms);
    }

    // The page holds one precompiled view per trained model; pick by encoding and layer count
    const encOf = v => v.meta.encoding, layersOf = v => v.meta.n_layers;
    const ENCODINGS = [...new Set(VIEWS.map(encOf))];
    const LAYERS = [...new Set(VIEWS.map(layersOf))].sort((a, b) => a - b);
    let view = 0;
    function pick(enc, layers) {  // That model, or the one with the closest layer count for this encoding
      const same = VIEWS.map((v, i) => i).filter(i => encOf(VIEWS[i]) === enc);
      return same.reduce((best, i) => Math.abs(layersOf(VIEWS[i]) - layers) < Math.abs(layersOf(VIEWS[best]) - layers) ? i : best);
    }
    $('views').innerHTML = ENCODINGS.map(e =>
      `<button data-enc="${e}" aria-pressed="false">${e === 'amplitude' ? 'Amplitude' : 'Angle'}</button>`).join('');
    $('layers').innerHTML = LAYERS.map(l => `<button data-layers="${l}" aria-pressed="false">${l}</button>`).join('');
    $('viewPick').hidden = ENCODINGS.length < 2;
    $('layerPick').hidden = LAYERS.length < 2;
    $('views').onclick = e => { const b = e.target.closest('[data-enc]'); if (b) loadView(pick(b.dataset.enc, n_layers)); };
    $('layers').onclick = e => { const b = e.target.closest('[data-layers]'); if (b) loadView(pick(encOf(VIEWS[view]), +b.dataset.layers)); };
    $('overview').onclick = () => overview();
    $('recenter').onclick = () => recenter();
    $('play').onclick = play;
    loadView(0, 0);
    $('stages').innerHTML = Object.entries(STAGES).map(([k, s]) =>
      `<button class="stage" data-stage="${k}" aria-pressed="false"><span class="dot" style="background:${s.color}"></span>${s.title}</button>`).join('');
    $('stages').onclick = e => { const b = e.target.closest('[data-stage]'); if (b) tour(b.dataset.stage); };
  </script>
</body>
</html>"""


if __name__ == "__main__":
    import argparse
    import functools
    import glob
    import os
    import re
    import torch
    import torch.nn as nn
    import pennylane as qml
    from pennylane.qnn import TorchLayer

    parser = argparse.ArgumentParser(
        description="Export the student QMLP visualizer: every trained model in this folder, precompiled into one page.")
    parser.add_argument("-l", "--layers", type=int, nargs="+",
                        help="Only these layer counts, e.g. -l 2 10 (default: every qmlp-mnist-run1-layer*.pt found here)")
    args = parser.parse_args()

    n_qubits = 9
    num_classes = 10
    mnist_dir = "mnist"  # Folder with the MNIST idx files, relative to where you run this

    dev = qml.device("default.qubit", wires=n_qubits)

    def modular_w(n_qubits, n_layers):
        shapes = {}
        for n in range(n_layers):
            shapes[f"rot_layer_{n}"] = (n_qubits, 3)
            shapes[f"crx_layer_{n}"] = (n_qubits, 1)
        return shapes

    def pt_name(encoding, n_layers):
        return f"qmlp-mnist-run1-layer{n_layers}-{encoding}.pt"

    def make_model(encoding, n_layers):
        """Same circuit and classical layer as the training script for this encoding; loads its .pt file if present."""
        @qml.qnode(dev, interface="torch")
        def qnode(inputs, **weights_kwargs):
            if encoding == "Amplitude":  # Loaded once, before the first layer
                qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
            for n in range(n_layers):
                if encoding == "Angle":  # Loaded again at the start of every layer
                    qml.AngleEmbedding(inputs, wires=range(n_qubits))
                for i in range(n_qubits):
                    qml.Rot(*weights_kwargs[f"rot_layer_{n}"][i], wires=i)
                for i in range(n_qubits):
                    qml.CRX(weights_kwargs[f"crx_layer_{n}"][i][0], wires=[i, (i + 1) % n_qubits])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        class drebin(nn.Module):
            def __init__(self):
                super().__init__()
                self.qlayer = TorchLayer(qnode, modular_w(n_qubits, n_layers))
                self.fc = nn.Linear(n_qubits, num_classes)

        model = drebin()
        pt_file = pt_name(encoding, n_layers)
        if os.path.exists(pt_file):
            print(f"[*] Loading trained weights from {pt_file}...")
            model.load_state_dict(torch.load(pt_file, map_location="cpu"))
            return model, None
        print(f"[*] No {pt_file} found—using randomly initialized weights for preview...")
        return model, f"Untrained: {pt_file} was not found, so these are random weights and the predictions mean nothing."

    # Which models to precompile: every trained file found here, or the --layers counts (untrained if missing)
    found = sorted({int(re.search(r"layer(\d+)-", f).group(1)) for f in glob.glob("qmlp-mnist-run1-layer*-A*.pt")})
    models = [(enc, n) for n in (args.layers or found or [2]) for enc in ("Angle", "Amplitude")
              if args.layers or not found or os.path.exists(pt_name(enc, n))]

    # MNIST, loaded once for every model
    mnist = None
    try:
        import idx2numpy
        test_labels = idx2numpy.convert_from_file(f"{mnist_dir}/t10k-labels-idx1-ubyte")
        mnist = {
            "train": idx2numpy.convert_from_file(f"{mnist_dir}/train-images-idx3-ubyte"),
            "test": idx2numpy.convert_from_file(f"{mnist_dir}/t10k-images-idx3-ubyte"),
            "labels": test_labels,
            # The first test image of each digit 0–9
            "picks": [int(np.flatnonzero(test_labels == d)[0]) for d in range(num_classes) if (test_labels == d).any()],
        }
    except Exception as e:  # The example digits are optional; the visualizer still works without them
        print(f"[*] No example digits ({e})")

    @functools.lru_cache(maxsize=None)  # Same preprocessing for every layer count, so do it once per encoding
    def encoded_digits(encoding):
        """The example digits preprocessed exactly like that encoding's training script (scaler and PCA fitted on
        the full training set):
          Angle:     28x28 → resize to 3x3 → MinMaxScaler → PCA(n_qubits)
          Amplitude: 28x28 → 784 pixels → MinMaxScaler → PCA(2**n_qubits)"""
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.decomposition import PCA
        picks = mnist["picks"]
        if encoding == "Angle":
            from PIL import Image
            from torchvision import transforms

            def downscale_images(X, new_size=(3, 3)):  # Same steps as the training script, so the pixels match
                resize, to_tensor = transforms.Resize(new_size), transforms.ToTensor()
                return np.array([to_tensor(resize(Image.fromarray(x.astype(np.uint8)))).squeeze(0).numpy() for x in X])

            print("[*] Angle: resizing the training set to 3x3 (this takes a moment)...")
            X_train = downscale_images(mnist["train"]).reshape(len(mnist["train"]), -1).astype(np.float32)
            small = downscale_images(mnist["test"][picks])
            X_pick = small.reshape(len(picks), -1).astype(np.float32)
            n_components = n_qubits
        else:
            print("[*] Amplitude: fitting PCA on the training set...")
            X_train = mnist["train"].reshape(len(mnist["train"]), -1).astype(np.float32)
            small = [None] * len(picks)
            X_pick = mnist["test"][picks].reshape(len(picks), -1).astype(np.float32)
            n_components = 2 ** n_qubits
        scaler = MinMaxScaler().fit(X_train)
        pca = PCA(n_components=n_components).fit(scaler.transform(X_train))
        return small, pca.transform(scaler.transform(X_pick))

    def example_digits(model, encoding):
        small, features = encoded_digits(encoding)
        with torch.no_grad():  # Run the digits through this model's quantum layer to get their ⟨Z⟩ values
            expvals = model.qlayer(torch.tensor(features, dtype=torch.float32)).numpy()
        return [{"image": mnist["test"][i], "downscaled": s, "features": f, "label": int(mnist["labels"][i]), "expvals": z}
                for i, s, f, z in zip(mnist["picks"], small, features, expvals)]

    views = []
    for encoding, n_layers in models:
        model, note = make_model(encoding, n_layers)
        samples = []
        if mnist is not None:
            try:
                samples = example_digits(model, encoding)
            except Exception as e:
                print(f"[*] {encoding}: skipping the example digits ({e})")
        views.append(build_view(model, samples, encoding, note=note))  # Qubit and layer counts are read from the weights

    write_page(views, "student_qmlp_viz.html")
