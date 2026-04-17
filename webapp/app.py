"""
Raven Plot — HDF5 Signal Viewer
"""
import os
import uuid
import json
import tempfile
import h5py
import numpy as np
import plotly.graph_objects as go
from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)
app.secret_key = os.urandom(24)

# In-memory store: session_id -> {fname -> {signal_path -> np.array}}
_sessions: dict = {}

UPLOAD_DIR = tempfile.mkdtemp(prefix="ravenplot_")


def _sid():
    sid = session.get("sid")
    if not sid or sid not in _sessions:
        sid = str(uuid.uuid4())
        session["sid"] = sid
        _sessions[sid] = {}
    return sid


def collect_datasets(hf, prefix=""):
    out = {}
    def _v(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[f"{prefix}{name}" if prefix else name] = obj[()]
    hf.visititems(_v)
    return out


def to_json_safe(arr):
    """Convert numpy array to a JSON-serializable list."""
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return [float(arr)]
    if arr.dtype.kind in ("U", "S", "O"):
        return [str(x) for x in arr.tolist()]
    return arr.astype(float).tolist()


@app.route("/")
def index():
    _sid()  # ensure session
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    sid = _sid()
    mode = request.form.get("mode", "replace")  # replace | add
    f = request.files.get("file")
    if not f:
        return jsonify(error="No file"), 400

    fname = f.filename
    path = os.path.join(UPLOAD_DIR, f"{sid}_{fname}")
    f.save(path)

    try:
        hf = h5py.File(path, "r")
    except Exception as e:
        return jsonify(error=str(e)), 400

    if mode == "replace":
        _sessions[sid] = {}

    # Make filename unique within session
    existing = list(_sessions[sid].keys())
    base, ext = os.path.splitext(fname)
    unique_fname = fname
    c = 1
    while unique_fname in existing:
        unique_fname = f"{base}_{c}{ext}"
        c += 1

    multi = (len(_sessions[sid]) >= 1)
    prefix = f"{unique_fname}/" if multi else ""

    # Retroactively prefix first file if we're adding a second
    if multi and len(_sessions[sid]) == 1:
        first_fname = next(iter(_sessions[sid]))
        old = _sessions[sid].pop(first_fname)
        reprefixed = {}
        for k, v in old.items():
            new_key = k if k.startswith(first_fname + "/") else f"{first_fname}/{k}"
            reprefixed[new_key] = v
        _sessions[sid][first_fname] = reprefixed
        prefix = f"{unique_fname}/"

    datasets = collect_datasets(hf, prefix=prefix)
    # Serialise to plain python for storage
    stored = {}
    for k, arr in datasets.items():
        arr = np.squeeze(arr)
        stored[k] = arr

    if multi:
        _sessions[sid][unique_fname] = {k: v for k, v in stored.items()}
    else:
        _sessions[sid][unique_fname] = stored

    # Build flat signal list for UI
    signals = _build_signal_list(sid)
    hf.close()

    return jsonify(
        signals=signals,
        filename=unique_fname,
        total=len(signals),
        mode=mode,
    )


def _build_signal_list(sid):
    out = []
    for fname, sigs in _sessions[sid].items():
        for path, arr in sigs.items():
            arr = np.squeeze(arr)
            shape = list(arr.shape)
            dtype = str(arr.dtype)
            out.append({"path": path, "shape": shape, "dtype": dtype})
    return out


@app.route("/signals")
def signals():
    sid = _sid()
    return jsonify(signals=_build_signal_list(sid))


@app.route("/plot", methods=["POST"])
def plot():
    sid = _sid()
    body = request.get_json()
    paths = body.get("paths", [])
    options = body.get("options", {})

    skip = options.get("skip_first", False)
    use_time = options.get("use_time", True)
    plot_type = options.get("plot_type", "line")  # line | scatter | bar | step

    # Flatten all signals
    all_sigs = {}
    for fname, sigs in _sessions[sid].items():
        all_sigs.update(sigs)

    # Try to find a time vector
    time_vec = None
    if use_time:
        for path, arr in all_sigs.items():
            lp = path.lower()
            if "time" in lp or "timestamp" in lp:
                arr = np.squeeze(arr)
                if arr.ndim == 1:
                    time_vec = arr.astype(float)
                    break

    figures = []
    for path in paths:
        arr = all_sigs.get(path)
        if arr is None:
            continue
        arr = np.squeeze(arr)
        if skip and arr.ndim >= 1 and len(arr) > 1:
            arr = arr[1:]

        label = path.split("/")[-1]

        if arr.ndim == 1:
            n = len(arr)
            x = time_vec[:n].tolist() if time_vec is not None and len(time_vec) >= n else list(range(n))

            if arr.dtype.kind in ("U", "S", "O"):
                vals = [str(v) for v in arr.tolist()]
                uniq = list(dict.fromkeys(vals))
                mp = {v: i for i, v in enumerate(uniq)}
                y_num = [mp[v] for v in vals]
                fig = go.Figure(go.Scatter(
                    x=x, y=y_num, mode="lines",
                    line=dict(shape="hv", width=2), name=label,
                ))
                fig.update_yaxes(tickvals=list(range(len(uniq))), ticktext=uniq)
            else:
                y = arr.astype(float).tolist()
                if plot_type == "bar":
                    fig = go.Figure(go.Bar(x=x, y=y, name=label))
                elif plot_type == "scatter":
                    fig = go.Figure(go.Scatter(x=x, y=y, mode="markers",
                        marker=dict(size=4), name=label))
                elif plot_type == "step":
                    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines",
                        line=dict(shape="hv", width=2), name=label))
                else:  # line (default)
                    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines",
                        line=dict(width=2), name=label,
                        fill="tozeroy", fillcolor="rgba(0,113,227,0.07)"))
        elif arr.ndim == 2:
            fig = go.Figure(go.Heatmap(
                z=arr.T.astype(float).tolist(),
                colorscale="Viridis",
                name=label,
            ))
        else:
            figures.append({"path": path, "error": f"Cannot display {arr.ndim}D data"})
            continue

        fig.update_layout(
            title=dict(text=path, font=dict(size=13, color="#1c1c1c"), x=0),
            margin=dict(l=48, r=16, t=40, b=40),
            height=260,
            paper_bgcolor="white",
            plot_bgcolor="#f7f8fa",
            font=dict(family="'Inter', 'Helvetica Neue', sans-serif", size=11, color="#3a3a3a"),
            xaxis=dict(
                title="Time [s]" if use_time else "Sample",
                gridcolor="#e8e8e8", linecolor="#d0d0d0", zeroline=False,
            ),
            yaxis=dict(
                title=label,
                gridcolor="#e8e8e8", linecolor="#d0d0d0", zeroline=False,
            ),
            showlegend=False,
            hovermode="x unified",
        )
        figures.append({"path": path, "fig": fig.to_json()})

    return jsonify(figures=figures)


@app.route("/clear", methods=["POST"])
def clear_session():
    sid = _sid()
    _sessions[sid] = {}
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False, use_reloader=False, threaded=True)
