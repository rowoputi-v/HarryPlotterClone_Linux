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
# Attributes store: session_id -> {signal_path -> {attr_name -> value}}
_attrs: dict = {}

UPLOAD_DIR = tempfile.mkdtemp(prefix="ravenplot_")


def _sid():
    sid = session.get("sid")
    if not sid or sid not in _sessions:
        sid = str(uuid.uuid4())
        session["sid"] = sid
        _sessions[sid] = {}
        _attrs[sid] = {}
    return sid


def collect_datasets(hf, prefix=""):
    """Walk an HDF5 file and collect all plottable datasets.

    Handles two common structures:
    1. Flat:  dataset directly under groups  (key = group/dataset)
    2. Zenuity-style nested:
         signal_group/data/field_name/unit_name/value   → data
         signal_group/zeader/timestamp_ns                → per-signal time (ns)
         signal_group/data/timestamp/nanoseconds/value   → per-signal time (ns)

    Returns (datasets_dict, attrs_dict).
    """
    out = {}
    attrs = {}  # path -> dict of HDF5 attributes

    def _try_zenuity(name, grp):
        """Check if grp follows the zeader pattern (has 'data' + 'zeader' sub-groups)."""
        if not isinstance(grp, h5py.Group):
            return False
        keys = set(grp.keys())
        if "data" not in keys or "zeader" not in keys:
            return False
        data_grp = grp["data"]
        if not isinstance(data_grp, h5py.Group):
            return False
        return True

    def _extract_zeader_time(grp, prefix_path):
        """Extract timestamp_ns from zeader group."""
        zeader = grp.get("zeader")
        if zeader is None:
            return
        ts_ds = zeader.get("timestamp_ns")
        if ts_ds is not None and isinstance(ts_ds, h5py.Dataset):
            key = f"{prefix_path}/zeader/timestamp_ns"
            out[key] = ts_ds[()]
            attrs[key] = {"unit": "ns", "_is_zeader_time": True}

    def _extract_data_fields(grp, prefix_path):
        """Extract data fields from a signal's 'data' sub-group."""
        data_grp = grp["data"]
        for field_name in data_grp:
            obj = data_grp[field_name]
            if isinstance(obj, h5py.Dataset):
                # Direct dataset (e.g. brake_pedal_pressed_quality)
                key = f"{prefix_path}/{field_name}"
                out[key] = obj[()]
                a = {}
                for ak in obj.attrs:
                    try:
                        v = obj.attrs[ak]
                        if isinstance(v, bytes):
                            v = v.decode("utf-8", errors="replace")
                        a[ak] = v
                    except Exception:
                        pass
                attrs[key] = a
            elif isinstance(obj, h5py.Group):
                # field_name/unit_name/value pattern
                for unit_name in obj:
                    unit_grp = obj[unit_name]
                    if isinstance(unit_grp, h5py.Group) and "value" in unit_grp:
                        val_ds = unit_grp["value"]
                        if isinstance(val_ds, h5py.Dataset):
                            key = f"{prefix_path}/{field_name}"
                            out[key] = val_ds[()]
                            attrs[key] = {"unit": unit_name}
                            # If this is the embedded timestamp, mark it
                            if field_name.lower() in ("timestamp", "time"):
                                attrs[key]["_is_embedded_time"] = True
                    elif isinstance(unit_grp, h5py.Dataset):
                        # unit_name is actually a direct dataset
                        key = f"{prefix_path}/{field_name}/{unit_name}"
                        out[key] = unit_grp[()]
                        attrs[key] = {}

    # First pass: check top-level groups for zenuity pattern
    zenuity_found = False
    for top_name in hf:
        top = hf[top_name]
        if isinstance(top, h5py.Group):
            # Check if children have data+zeader
            for child_name in top:
                child = top[child_name]
                if _try_zenuity(child_name, child):
                    zenuity_found = True
                    break
            if zenuity_found:
                break

    if zenuity_found:
        # Zenuity-style: walk the expected structure
        def _walk_groups(parent, parent_path):
            for name in parent:
                grp = parent[name]
                full = f"{parent_path}/{name}" if parent_path else name
                prefixed = f"{prefix}{full}" if prefix else full
                if isinstance(grp, h5py.Group) and _try_zenuity(name, grp):
                    _extract_zeader_time(grp, prefixed)
                    _extract_data_fields(grp, prefixed)
                elif isinstance(grp, h5py.Group):
                    _walk_groups(grp, full)
                elif isinstance(grp, h5py.Dataset):
                    key = f"{prefix}{full}" if prefix else full
                    out[key] = grp[()]
                    attrs[key] = {}

        for top_name in hf:
            top = hf[top_name]
            if isinstance(top, h5py.Group):
                _walk_groups(top, top_name)
            elif isinstance(top, h5py.Dataset):
                key = f"{prefix}{top_name}" if prefix else top_name
                out[key] = top[()]
                attrs[key] = {}
    else:
        # Flat / generic style
        def _v(name, obj):
            if isinstance(obj, h5py.Dataset):
                key = f"{prefix}{name}" if prefix else name
                out[key] = obj[()]
                a = {}
                for ak in obj.attrs:
                    try:
                        v = obj.attrs[ak]
                        if isinstance(v, bytes):
                            v = v.decode("utf-8", errors="replace")
                        elif isinstance(v, np.ndarray):
                            v = v.tolist()
                        a[ak] = v
                    except Exception:
                        pass
                attrs[key] = a
        hf.visititems(_v)

    return out, attrs


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
        _attrs[sid] = {}

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
            # Also reprefix attrs
            if k in _attrs[sid]:
                _attrs[sid][new_key] = _attrs[sid].pop(k)
        _sessions[sid][first_fname] = reprefixed
        prefix = f"{unique_fname}/"

    datasets, file_attrs = collect_datasets(hf, prefix=prefix)
    # Serialise to plain python for storage
    stored = {}
    for k, arr in datasets.items():
        arr = np.squeeze(arr)
        stored[k] = arr

    # Store attributes
    for k, a in file_attrs.items():
        _attrs[sid][k] = a

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
            # Skip zeader/timestamp entries — they're time vectors, not plottable signals
            sig_attrs = _attrs.get(sid, {}).get(path, {})
            if sig_attrs.get("_is_zeader_time") or sig_attrs.get("_is_embedded_time"):
                continue
            # Also skip zeader helper fields (sequence_id etc.)
            if "/zeader/" in path:
                continue
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

    # Collect all time vectors (per-file and global)
    # For zeader-style files, each signal group has its own timestamp.
    time_vectors = {}
    time_units = {}  # path -> detected unit string
    if use_time:
        for path, arr in all_sigs.items():
            sig_attrs = _attrs.get(sid, {}).get(path, {})

            # 1. Zeader timestamps (marked by collect_datasets)
            if sig_attrs.get("_is_zeader_time") or sig_attrs.get("_is_embedded_time"):
                arr = np.squeeze(arr)
                if arr.ndim == 1 and arr.dtype.kind in ("f", "i", "u"):
                    time_vectors[path] = arr.astype(float)
                    time_units[path] = sig_attrs.get("unit")
                continue

            # 2. Generic: any path with "time" or "timestamp" in its name
            lp = path.lower()
            if "time" in lp or "timestamp" in lp:
                arr = np.squeeze(arr)
                if arr.ndim == 1 and arr.dtype.kind in ("f", "i", "u"):
                    time_vectors[path] = arr.astype(float)
                    unit = None
                    for attr_key in ("unit", "units", "Unit", "Units", "UNIT", "UNITS"):
                        if attr_key in sig_attrs:
                            unit = str(sig_attrs[attr_key]).strip().lower()
                            break
                    time_units[path] = unit

    def _find_time_vec(signal_path, signal_len):
        """Find the best matching time vector for a signal.
        For zeader-style files, match by the signal's parent group prefix.
        Returns (time_array, unit_string_or_None)."""
        if not time_vectors:
            return None, None

        # Build candidate prefixes from the signal path
        # e.g. "zen_qm_feature_a/brake_pedal/is_brake_pedal_pressed"
        #   → try "zen_qm_feature_a/brake_pedal/" first (signal group)
        #   → then "zen_qm_feature_a/" (file group)
        parts = signal_path.split("/")
        prefixes = []
        for i in range(len(parts) - 1, 0, -1):
            prefixes.append("/".join(parts[:i]) + "/")
        prefixes.append("")  # fallback: match anything

        candidates = []
        for tp, tv in time_vectors.items():
            # Score: how specific is the prefix match?
            match_depth = 0
            for depth, pfx in enumerate(prefixes):
                if pfx and tp.startswith(pfx):
                    match_depth = len(prefixes) - depth  # higher = more specific
                    break
            candidates.append((match_depth, tp, tv))

        # Sort: prefer deepest prefix match, then closest length
        best = None
        best_path = None
        best_score = (-1, float("inf"))
        for match_depth, tp, tv in candidates:
            length_match = len(tv) >= signal_len
            length_diff = abs(len(tv) - signal_len)
            score = (match_depth + int(length_match), -length_diff)
            if score > best_score:
                best_score = score
                best = tv
                best_path = tp

        return best, time_units.get(best_path) if best_path else None

    def _normalize_time(time_arr, unit):
        """Convert time to seconds, subtract the first timestamp so the axis
        starts near zero (relative time).  This keeps the numbers small and
        readable.  The label shows 'Relative time [s]' with the source unit."""
        def _to_seconds(arr, u):
            if u in ("ns", "nanosecond", "nanoseconds", "nano"):
                return arr / 1e9, "ns"
            elif u in ("us", "µs", "microsecond", "microseconds", "micro"):
                return arr / 1e6, "µs"
            elif u in ("ms", "millisecond", "milliseconds", "milli"):
                return arr / 1e3, "ms"
            elif u in ("s", "sec", "second", "seconds"):
                return arr, "s"
            elif u in ("min", "minute", "minutes"):
                return arr * 60, "min"
            elif u in ("h", "hr", "hour", "hours"):
                return arr * 3600, "h"
            return None, None

        if unit is not None:
            converted, src = _to_seconds(time_arr, unit.lower().strip())
            if converted is not None:
                # Make relative (start from 0)
                if len(converted) > 0:
                    converted = converted - converted[0]
                return converted, f"Time [s] (from {src})"
            return time_arr, f"Time [{unit}]"

        # No unit — guess from magnitude
        if len(time_arr) > 1:
            span = abs(time_arr[-1] - time_arr[0])
            if span > 1e15:
                arr_s = time_arr / 1e9
                return arr_s - arr_s[0], "Time [s] (from ns)"
            elif span > 1e9:
                arr_s = time_arr / 1e6
                return arr_s - arr_s[0], "Time [s] (from µs)"
            elif span > 1e6:
                arr_s = time_arr / 1e3
                return arr_s - arr_s[0], "Time [s] (from ms)"
        # Already small — just make relative
        rel = time_arr - time_arr[0] if len(time_arr) > 0 else time_arr
        return rel, "Time [s]"

    figures = []
    for path in paths:
        arr = all_sigs.get(path)
        if arr is None:
            continue
        arr = np.squeeze(arr)

        label = path.split("/")[-1]

        # Find matching time vector BEFORE any slicing
        raw_time, time_unit = _find_time_vec(path, len(arr) if arr.ndim >= 1 else 0)
        time_label = "Sample"

        if raw_time is not None:
            raw_time, time_label = _normalize_time(raw_time, time_unit)

        if skip and arr.ndim >= 1 and len(arr) > 1:
            arr = arr[1:]
            # Slice time vector the same way
            if raw_time is not None and len(raw_time) > 1:
                raw_time = raw_time[1:]

        if arr.ndim == 1:
            n = len(arr)
            # Build x-axis: use time vector trimmed to exactly n points
            if raw_time is not None and len(raw_time) >= n:
                x = raw_time[:n].tolist()
            else:
                x = list(range(n))

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
            title=dict(text=path, font=dict(size=11, color="#1c1c1c"), x=0),
            margin=dict(l=60, r=20, t=44, b=52),
            height=280,
            autosize=True,
            paper_bgcolor="white",
            plot_bgcolor="#f7f8fa",
            font=dict(family="'Inter', 'Helvetica Neue', sans-serif", size=9, color="#3a3a3a"),
            xaxis=dict(
                title=dict(text=time_label if use_time else "Sample", standoff=8),
                gridcolor="#e8e8e8", linecolor="#d0d0d0", zeroline=False,
                automargin=True,
            ),
            yaxis=dict(
                title=dict(text=label, standoff=8),
                gridcolor="#e8e8e8", linecolor="#d0d0d0", zeroline=False,
                automargin=True,
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
    _attrs[sid] = {}
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False, use_reloader=False, threaded=True)
