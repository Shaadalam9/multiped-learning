# Multi-pedestrian interaction with automated vehicle


## Citation
If you use the simulator for academic work please cite the following papers:

>  


## Getting started
[![Python Version](https://img.shields.io/badge/python-3.12.3-blue.svg)](https://www.python.org/downloads/release/python-3919/)
[![Package Manager: uv](https://img.shields.io/badge/package%20manager-uv-green)](https://docs.astral.sh/uv/)

Tested with **Python 3.12.3** and the [`uv`](https://docs.astral.sh/uv/) package manager.  
Follow these steps to set up the project.

**Step 1:** Install `uv`. `uv` is a fast Python package and environment manager. Install it using one of the following methods:

**macOS / Linux (bash/zsh):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**Alternative (if you already have Python and pip):**
```bash
pip install uv
```

**Step 2:** Fix permissions (if needed):t

Sometimes `uv` needs to create a folder under `~/.local/share/uv/python` (macOS/Linux) or `%LOCALAPPDATA%\uv\python` (Windows).  
If this folder was created by another tool (e.g. `sudo`), you may see an error like:
```lua
error: failed to create directory ... Permission denied (os error 13)
```

To fix it, ensure you own the directory:

### macOS / Linux
```bash
mkdir -p ~/.local/share/uv
chown -R "$(id -un)":"$(id -gn)" ~/.local/share/uv
chmod -R u+rwX ~/.local/share/uv
```

### Windows
```powershell
# Create directory if it doesn't exist
New-Item -ItemType Directory -Force "$env:LOCALAPPDATA\uv"

# Ensure you (the current user) own it
# (usually not needed, but if permissions are broken)
icacls "$env:LOCALAPPDATA\uv" /grant "$($env:UserName):(OI)(CI)F"
```

**Step 3:** After installing, verify:
```bash
uv --version
```

**Step 4:** Clone the repository:
```command line
git clone https://github.com/Shaadalam9/multiped-learning
cd multiped
```

**Step 5:** Ensure correct Python version. If you don’t already have Python 3.12.3 installed, let `uv` fetch it:
```command line
uv python install 3.12.3
```
The repo should contain a .python-version file so `uv` will automatically use this version.

**Step 6:** Create and sync the virtual environment. This will create **.venv** in the project folder and install dependencies exactly as locked in **uv.lock**:
```command line
uv sync --frozen
```

**Step 7:** Activate the virtual environment:

**macOS / Linux (bash/zsh):**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (cmd.exe):**
```bat
.\.venv\Scripts\activate.bat
```

**Step 8:** Ensure that dataset are present. Place required datasets (including **mapping.csv**) into the **data/** directory:


**Step 9:** Run the code:
```command line
python3 analysis.py
```

## Configuration of project
Configuration of the project needs to be defined in `multiped/config`. Please use the `default.config` file for the required structure of the file. If no custom config file is provided, `default.config` is used. The config file has the following parameters:
* `mapping`: CSV file that contains all data found in the videos.
* `plotly_template`: Template used to make graphs in the analysis.
* `output`: Directory where analysis results and intermediate output files will be saved.
* `figures`: Directory where final figures and plots are stored.
* `data`: Directory containing all raw and processed data files used in the analysis.
* `intake_questionnaire`: CSV file containing participant responses from the intake (pre-experiment) questionnaire.
* `post_experiment_questionnaire`: CSV file containing participant responses from the post-experiment questionnaire.
* `compare_trial`: Reference trial against which all other trials are compared during t-tests in the analysis.
* `kp_resolution`: Time bin size, in milliseconds, used for storing keypress data, which controls the resolution of keypress event logs.
* `yaw_resolution`: Time bin size, in milliseconds, used for storing yaw (head rotation) data, controlling the resolution of HMD orientation data.
* `smoothen_signal`:  Boolean toggle to enable or disable signal smoothing for data analysis.
* `freq`: Frequency parameter used by the One Euro Filter for signal smoothing.
* `mincutoff`: Minimum cutoff frequency for the One Euro Filter.
* `beta`: Beta value controlling the speed-versus-smoothness tradeoff in the One Euro Filter.
* `font_family`: Font family to be used in all generated figures for visual consistency.
* `font_size`: Font size to be applied to all text in generated figures.
* `p_value`: p-value threshold to be used for statistical significance testing (e.g., in t-tests).



## Results



## Contact
If you have any questions or suggestions, feel free to reach out to md_shadab_alam@outlook.com and 
