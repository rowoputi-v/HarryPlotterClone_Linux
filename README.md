# 🐦 Raven Plot

A modern HDF5 signal viewer for Linux, built with Python, Tkinter, and Matplotlib.
Also includes a web interface powered by Flask and Plotly.

## Features

- **Browse & search** all datasets inside one or more HDF5 files in a clean sidebar tree
- **Additive plotting** — click signals one by one to stack them vertically for comparison
- **Multi-file support** — open a base file then use *Add File* to load signals from additional HDF5 files side-by-side
- **Overlay mode** — plot multiple selected signals on a single shared axes
- **Drag & drop** — drop one or more `.h5` / `.hdf5` / `.hdf` files directly onto the window
- **GPS map** — renders a GPS track from `latitude`/`longitude` datasets in the browser via Folium
- **Navigation toolbar** — pan, zoom, save each individual plot
- **Responsive** — plots redraw when the window is resized
- **Catppuccin Mocha** dark theme throughout
- **Web UI** — browser-based viewer with Plotly, drag & drop, and plot type selection

## Requirements

- Python 3.9+
- `h5py`, `numpy`, `matplotlib`
- `tkinterdnd2` *(optional — drag & drop)*
- `mplcursors` *(optional — hover tooltips)*
- `folium` *(optional — GPS map)*
- `flask` + `plotly` *(optional — web UI)*

## Installation

```bash
git clone https://github.com/rowoputi-v/HarryPlotterClone_Linux.git
cd HarryPlotterClone_Linux

python -m venv .venv
source .venv/bin/activate

pip install h5py numpy matplotlib tkinterdnd2 mplcursors folium flask plotly
```

## Running

### Desktop App
```bash
./run.sh
# or
.venv/bin/python ravenplot.py
```

### Web App
```bash
cd webapp && ./run_web.sh
# then open http://localhost:5000
```

## Usage

| Action | How |
|---|---|
| Open a file | Click **Open HDF5** or drag & drop |
| Add a second file | Click **Add File** |
| Plot a signal | Click it in the tree |
| Stack another signal below | Click another signal |
| Multi-select | Ctrl+click / Shift+click, then **▶ Plot Selected** |
| Overlay signals | Enable **Overlay** checkbox, then plot |
| Clear all plots | Click **Clear Plots** or `Ctrl+L` |

## Project Structure

```
ravenplot.py      # Desktop application (Tkinter + Matplotlib)
webapp/           # Web application (Flask + Plotly)
  app.py          #   Flask backend
  templates/      #   HTML templates
  run_web.sh      #   Launch script
run.sh            # Desktop launch script
```

## License

MIT
