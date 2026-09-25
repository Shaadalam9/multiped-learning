# Trial order in a multi-pedestrian VR experiment

This project compares fixed and participant-specific randomised trial order in a
VR pedestrian experiment. Run one Python file to produce the analysis tables,
saved results and figures used by the manuscript.

## Run the project

From the project folder, using the existing environment:

```bash
.venv/bin/python main.py
```

The command reuses a compatible saved result snapshot when one exists. Otherwise,
it reads the raw data and runs the full analysis. Reusing a snapshot regenerates
the figures without refitting the statistical models.

To read the raw data again and recalculate everything:

```bash
.venv/bin/python main.py --refresh
```

To explicitly regenerate figures from the existing CSV result tables and create
a new snapshot:

```bash
.venv/bin/python main.py --from-saved
```

Use `--refresh` after changing raw data. A cached run does not check raw files for
changes. `--from-saved` adopts the saved numerical results; it does not recalculate
or validate them against the raw data. The two options cannot be combined.

## First-time setup

### 1. Install the uv package manager

On macOS or Linux, open Terminal and run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Alternatively, if Python and pip are already installed:

```bash
python -m pip install uv
```

Restart Terminal or PowerShell after installation, then verify:

```bash
uv --version
```

These installation methods are documented in the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).
If `uv` is not recognised, follow the installer's PATH instructions and open a new
terminal. Installation does not require an existing Python installation when
using the standalone installer.

### 2. Download the repository

With Git installed:

```bash
git clone https://github.com/Shaadalam9/multiped-learning
cd multiped-learning
```

If the repository already exists, open that folder instead. On the current Mac:

```bash
cd /Users/alam/Repos/multiped-learning
```

### 3. Install Python and the project dependencies

```bash
uv python install 3.12.3
uv sync --frozen
```

The project requires Python 3.12.3. `uv sync --frozen` creates `.venv` and installs
the dependencies recorded in `uv.lock`, including NumPy, pandas, SciPy,
statsmodels, Plotly, Kaleido and Flake8. Run these commands from the repository
folder, where `pyproject.toml` and `uv.lock` are located.

### 4. Activate the environment (optional)

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

After activation, use `python main.py`. Activation is optional:
`.venv/bin/python` on macOS/Linux or `.venv\Scripts\python.exe` on Windows runs
the project interpreter directly, including when PowerShell blocks activation.

### 5. Configure the input data

If `config` does not already exist, copy `default.config` to `config`:

```bash
cp default.config config
```

On Windows PowerShell, use `Copy-Item default.config config`. Do not overwrite an
existing configured file. Open `config` in a text editor and replace the example
paths with your own. It is JSON despite having no file extension. On Windows,
use forward slashes or escaped backslashes in JSON paths.

The repository contains code, not a guaranteed copy of all participant data.
Fresh analysis needs the participant folders, response files, sequence mappings,
timing mapping and questionnaire files specified in the configuration. OneDrive
files must be downloaded locally, not left as online-only placeholders.

### 6. Run the analysis and generate figures

```bash
.venv/bin/python main.py
```

On Windows:

```powershell
.\.venv\Scripts\python.exe main.py
```

The run prints its log location and finishes with a success message. See the run
modes above to force a fresh analysis or reuse CSV results explicitly.

## Configuration

The main script looks for `config`, then `config.comparison.json`, in the project
folder. To supply another file:

```bash
.venv/bin/python main.py /full/path/to/config.json
```

Every field in `config` and `default.config` is described below. Paths may be
absolute or relative to the folder containing the config file. Input paths are
machine-specific: replace the supplied OneDrive paths when using another computer.
`shuffled` means the randomised-order group; `unshuffled` means the fixed-order group.

### Input files and folders

The supplied input paths are under
`/Users/alam/Library/CloudStorage/OneDrive-TUEindhoven/TUe Multiped/`.
The table abbreviates that common directory as `<data>`.

| Field | Supplied value | Purpose |
| --- | --- | --- |
| `unshuffled_mapping` | `<data>/Data_unshuffled/mapping.csv` | Shared condition sequence for the fixed-order group. |
| `shuffled_mapping_filename` | `Participant_{participant_id}_mapping.csv` | Filename pattern for each randomised participant’s condition mapping; `{participant_id}` is replaced with the participant identifier. |
| `shuffled_data` | `<data>/Data_shuffled/Supplementary material/data/participant_response` | Folder containing the randomised group’s participant response data. |
| `unshuffled_data` | `<data>/Data_unshuffled/Supplementary material/data/participant_response` | Folder containing the fixed group’s participant response data. |
| `shuffled_intake_questionnaire` | `<data>/Data_shuffled/Supplementary material/data/intake-questionnaire.csv` | Randomised-group intake questionnaire used for demographic summaries. |
| `unshuffled_intake_questionnaire` | `<data>/Data_unshuffled/Supplementary material/data/intake-questionnaire.csv` | Fixed-group intake questionnaire used for demographic summaries. |
| `shuffled_post_experiment_questionnaire` | `<data>/Data_shuffled/Supplementary material/data/post-questionnaire.csv` | Randomised-group post-experiment questionnaire data. |
| `unshuffled_post_experiment_questionnaire` | `<data>/Data_unshuffled/Supplementary material/data/post-questionnaire.csv` | Fixed-group post-experiment questionnaire data. |

### Output settings

| Field | Supplied value | Purpose |
| --- | --- | --- |
| `output` | `_output` | Destination for result tables and snapshots. All generated figures go directly into `<output>/figures/`. |
| `figures` | `figures` | Destination for additional final copies when `save_final` is enabled. This does not change the primary output directory. |
| `save_final` | `true` | Copy only generated HTML and PNG files into the configured `figures` folder. Set `false` to skip these copies. If omitted, the loader defaults to `false`. |
| `auto_open` | `false` | Open each generated HTML figure in the default browser when `true`. If omitted, the loader defaults to `true`. |

### Retained settings not used by the current pipeline

These fields remain in the config for compatibility with earlier scripts. The
current `main.py` pipeline loads them but does not apply them to its calculations,
plots or logging. Changing them will not change those behaviours.

| Field | Supplied value | Earlier purpose / current behaviour |
| --- | --- | --- |
| `plotly_template` | `"plotly_white"` | Plotly visual theme. Current plotting functions set their styles in code. |
| `always_analyse` | `false` | Switch for rerunning analysis. Use `python main.py --refresh` to force a fresh analysis in the current pipeline. |
| `logger_level` | `"info"` | Logging verbosity. Use `--log-level DEBUG`, `INFO`, `WARNING` or `ERROR`; the command-line default is `INFO`. |
| `kp_resolution` | `100` | Legacy keypress/trigger resolution setting; it does not control the current time bins. Use `bin_seconds` for time-bin width. |
| `yaw_resolution` | `100` | Legacy yaw resolution setting; it does not control current head-orientation processing. |
| `smoothen_signal` | `true` | Legacy signal-smoothing switch; it does not enable or disable smoothing in the current pipeline. |
| `freq` | `120` | Legacy sampling-frequency setting in Hz; the current pipeline determines timing from the recorded data. |
| `mincutoff` | `0.1` | Legacy smoothing-filter minimum cut-off parameter; not applied by the current pipeline. |
| `beta` | `0.1` | Legacy smoothing-filter responsiveness parameter; not applied by the current pipeline. |
| `font_family` | `"verdana"` | Legacy figure font family; current plotting functions set fonts in code. |
| `font_size` | `40` | Legacy figure font size; current plotting functions set sizes in code. |
| `p_value` | `0.001` | Legacy significance-threshold setting; it does not set the current statistical tests or multiple-comparison corrections. |

For example, these settings generate all figures directly in `_output/figures`,
copy them to the project’s `figures` folder, and open each generated HTML figure
in your default browser after export finishes:

```json
"output": "_output",
"figures": "figures",
"save_final": true,
"auto_open": true
```

These are fields within the existing config object. The supplied configuration
uses `save_final: true` and `auto_open: false`: copies are enabled and browser
opening is disabled. Set `auto_open` to `true` to open the HTML figures. Changing these export preferences
does not invalidate the saved analysis snapshot.

Keep the repository's `mapping.csv`: it supplies participant passage timestamps
when no explicit timing mapping is configured. A condition-only sequence mapping
cannot replace this timing information.

### Analysis settings

Put these fields directly in the config object, alongside `output` and `figures`.
There is no `comparison_analysis` block. For example:

```json
"window_seconds": 5.0,
"bin_seconds": 0.1,
"primary_threshold": 0.1,
"sensitivity_thresholds": [0.05, 0.1, 0.5],
"require_complete_window": true,
"minimum_valid_trials_per_participant": 32
```

| Field | Default | Meaning |
| --- | --- | --- |
| `window_seconds` | `5.0` | Analyse this many seconds before the vehicle passes the participant. Must be positive. |
| `bin_seconds` | `0.1` | Width of each time bin in seconds. Must be positive and divide the analysis window exactly; the defaults give 50 bins. |
| `primary_threshold` | `0.1` | A bin is trigger-active when its maximum trigger value exceeds this threshold on the 0–1 scale. Used for the primary outcome. |
| `sensitivity_thresholds` | `[0.05, 0.1, 0.5]` | Additional trigger cut-offs used to check whether findings depend on the threshold. Values must be between 0 and 1; the primary threshold is included automatically. |
| `require_complete_window` | `true` | Require a complete analysis window when deciding whether a trial is valid. |
| `minimum_valid_trials_per_participant` | `32` | Minimum number of valid trials needed to retain a participant in the analysis. |
| `bootstrap_replicates` | `5000` | Number of participant-cluster bootstrap resamples for uncertainty estimates; minimum 1000. |
| `bootstrap_seed` | `2901` | Random seed for reproducible resampling. |
| `bayesian_bootstrap_draws` | `20000` | Number of Bayesian-bootstrap draws for the robustness analysis; minimum 5000. |
| `spline_degrees_of_freedom` | `4` | Flexibility of the spline used to describe changes over trial positions; allowed range 3–8. |
| `planning_effect_sizes_percentage_points` | `[2.5, 5.0, 7.5, 10.0]` | Positive effect sizes, in percentage points, used for sample-size planning calculations. |
| `planning_power` | `[0.8, 0.9]` | Target power levels for sample-size planning; each must be above 0.5 and below 1. |

The first six analysis fields are present in both supplied config files, with the
values shown above. The remaining six are optional fields supported by the loader
and use the listed defaults when omitted. After changing an
analysis setting, run `python main.py --refresh` to recompute results from raw data.
`--from-saved` reuses existing results and does not apply new analysis settings.

Application messages use the shared `CustomLogger` instance from `custom_logger.py`; `logmod.py`
configures its console and timestamped file handlers. Runtime warnings are also
routed through logging. Application code contains no `print()` calls.

## Outputs

The usual output locations are:

| Location | Contents |
| --- | --- |
| `_output/` | CSV result tables, analysis notes and numerical summaries |
| `_output/analysis_results.pickle` | Saved result DataFrames, not fitted model objects |
| `_output/analysis_results.json` | Snapshot provenance, checksums and compatibility information |
| `_output/figures/` | 19 analysis figures, four manuscript figures and manifests |
| Configured `figures` folder | HTML and PNG copies only, when `save_final` is `true` |
| `_logs/` | Timestamped console/file logs for each execution |
| `latex/` | Manuscript sources, bibliography and LaTeX supporting files |

Each Python-generated figure is exported as PDF, PNG, interactive HTML and an
editable Plotly pickle. Analysis figures also attempt EPS export. A failure in a
required export makes the command fail rather than report success.

The main command also exports the experimental-setup schematic as PDF, PNG,
HTML and pickle. Earlier manuscript versions embed a TikZ version of this
schematic, but no LaTeX installation is needed to generate the Python figure. Select `cas-sc-template.tex` as the main document in Overleaf; compile the
anonymous manuscript separately. Do not include one complete manuscript inside
the other.

A snapshot from fresh analysis retains unrounded table values. A snapshot adopted
from CSV retains the precision of those exported tables. Code, configuration and
library compatibility are checked before loading the snapshot, and its checksum
is verified. Only load pickle files produced by this trusted project.

## Figures: what is generated

The Python command generates 19 analysis plots and four manuscript figures for the
current complete dataset. Some analysis plots are conditional on having suitable
model results; a different or incomplete dataset may produce fewer plots.
Each filename below has `.pdf`, `.png`, `.html` and `.pickle` versions. PDF is
suited to LaTeX, PNG to presentations, HTML to interactive inspection, and pickle
to editing the Plotly figure in Python. EPS is optional for analysis plots.

### Manuscript figures

| Manuscript figure | Source or filename | What it shows |
| --- | --- | --- |
| 1 | `_output/figures/experimental_setup` (also embedded as TikZ in earlier sources) | Participant first/second, vehicle direction, pedestrian separation and stopping position |
| 2 | `_output/figures/trial_sequence` | How experimental factors are distributed over trial positions in each group |
| 3 | `_output/figures/distance_ratings` | Q2 differences for the selected yielding/eHMI-off/participant-second conditions and the position of the 2 m condition |
| 4 | `_output/figures/session_changes` | Model-adjusted responses over the three parts of the experiment |

All four manuscript figures are built by `utils/plots/manuscript.py`, called automatically
through the plotting package by `main.py`. Their inputs are `trial_level_common_window.csv`,
`condition_cell_contrasts.csv` and `session_segment_predictions.csv`.
`_output/figures/source_manifest.json` records hashes of those source tables.
The setup schematic is drawn by `utils/plots/setup.py`; all four manuscript
figures are generated by `python main.py`. Analysis plot numbers below are independent of manuscript
figure numbers.

### Analysis figures in `_output/figures/`

| Filename stem | Content |
| --- | --- |
| `figure_1_primary_marginal_estimates` | Overall perceived crossing safety estimates |
| `figure_2_participant_distributions` | Distributions of participant-level responses |
| `figure_3_trial_position_profiles` | Response profiles across trial positions |
| `figure_4_condition_effect_differences` | Differences in experimental-factor effects between groups |
| `figure_5_sequence_composition` | Experimental factors across the trial sequence |
| `figure_6_questionnaire_ratings` | Trial-level questionnaire ratings |
| `figure_7_head_heading_at_passage` | Head orientation when the vehicle reaches the participant |
| `figure_8_trigger_activation_and_magnitude` | Whether the trigger is activated and its magnitude |
| `figure_9_head_movement_outcomes` | Head movement measures |
| `figure_10_condition_adjusted_trial_profiles` | Trial profiles adjusted for experimental conditions |
| `figure_11_adjusted_temporal_effects` | Estimated changes over the session |
| `figure_12_prior_exposure_interactions` | Associations with previous experimental exposures |
| `figure_13_event_defined_trigger_outcomes` | Trigger press and release outcomes |
| `figure_14_adjusted_session_segments` | Adjusted estimates for the three session parts |
| `figure_15_ehmi_cue_learning_contrasts` | Changes in the eHMI on/off difference with experience |
| `figure_16_spline_trial_trajectories` | Exploratory nonlinear trial trajectories |
| `figure_17_primary_robustness` | Sensitivity and robustness of the primary comparison |
| `figure_18_condition_cell_followup_heatmap` | Group contrasts across individual experimental conditions |
| `figure_19_primary_condition_cell_forest` | Individual-condition contrasts for the primary outcome |

These additional plots support inspection of the analysis; they are not all
intended for inclusion in the manuscript.

## Results: where to find and interpret them

Start with `_output/analysis_results.log` for the numerical report and
`_output/analysis_decisions.md` for the analysis settings and interpretation notes.
These two reports are written by a full raw-data analysis; cached figure runs do
not rewrite them. For the currently loaded results, consult the CSV tables and
snapshot provenance. `_logs/` contains execution logs, not the separate numerical
results report.

| Question or output | Files in `_output/` |
| --- | --- |
| Which participants/trials were included? | `sample_flow.csv`, `exclusion_audit.csv` |
| Were mappings and sequences valid? | `mapping_audit.csv`, `sequence_audit.csv`, `duplicate_sequence_audit.csv` |
| What was extracted for each trial? | `trial_level_common_window.csv`, `outcome_dictionary.csv`, `outcome_quality_audit.csv` |
| How did the groups differ overall? | `primary_marginal_estimates.csv`, `primary_marginal_contrasts.csv`, `participant_level_descriptives.csv`, `participant_level_comparisons.csv` |
| What happened within individual conditions? | `condition_cell_estimates.csv`, `condition_cell_contrasts.csv`, `condition_cell_followup_summary.csv` |
| Did responses change over the session? | `session_segment_predictions.csv`, `session_segment_tests.csv`, `temporal_adjusted_tests.csv` |
| Did the eHMI difference change with experience? | `ehmi_learning_effects.csv`, `ehmi_learning_change_tests.csv` |
| What about demographic differences? | `baseline_demographic_descriptives.csv`, `baseline_demographic_comparisons.csv`, `demographic_adjusted_primary_contrast.csv` |
| Are the primary findings sensitive to analysis choices? | `primary_cluster_bootstrap_summary.csv`, `primary_bayesian_bootstrap_summary.csv`, `primary_leave_one_participant_out_summary.csv`, `primary_design_audit_sensitivity.csv` |
| Did model fitting fail? | `model_failures.csv`, `secondary_model_failures.csv`, and the other `*_failures.csv` files |

CSV tables can be opened in a spreadsheet or read with pandas. Empty failure
tables can be normal. Model coefficients and diagnostic tables provide the detail
behind the summaries; they are not additional independent findings.

### Reading the statistical columns

- Contrasts are generally fixed minus randomised; check each table's labels.
- Perceived crossing safety differences are percentage points; Q1–Q3 differences
  are rating points on the 0–100 scales.
- Distinguish ordinary `ci_low`/`ci_high` from simultaneous confidence intervals
  in the individual-condition comparisons.
- Use the reported multiplicity-adjusted p-value for the stated comparison
  family, rather than selecting an unadjusted p-value because it is smaller.
- Head orientation describes head movement, not eye gaze. Trigger responses
  describe perceived crossing safety, not observed crossing behaviour.

## Logging and troubleshooting

`logmod.py` configures console output and a UTF-8 log file for each run. The log
path appears at startup. Configuration, pipeline progress and errors are recorded
there; the numerical results summary remains a separate output of fresh analysis.

For more detail:

```bash
.venv/bin/python main.py --log-level DEBUG
```

If a snapshot is incompatible, use `--refresh` to recompute from raw data, or
`--from-saved` only if you intend to keep the existing CSV results. If raw input
files cannot be found, check the configuration paths and OneDrive download status.
For other failures, inspect the final traceback in the latest `_logs/` file.

| Problem | What to check |
| --- | --- |
| `uv: command not found` | Restart the terminal and follow the installer's PATH instructions |
| Permission error during installation | Use a user-writable installation/cache location; see the [uv installer options](https://docs.astral.sh/uv/configuration/installer/) |
| `ModuleNotFoundError` | Run `uv sync --frozen`, then use the project's `.venv` interpreter |
| Python version mismatch | Run `uv python install 3.12.3` and recreate/sync the project environment |
| Missing data files | Check configuration paths and download OneDrive files locally |
| Missing passage timestamps | Supply the timing mapping containing `cross_p1_time_s` and `cross_p2_time_s` |
| PDF/PNG export failure | Check the traceback and the pinned Plotly/Kaleido installation; HTML alone is not a successful complete export |
| Overleaf missing figures or class files | Upload the complete Overleaf package and select the correct main document |

## Code organisation

`main.py` is a 21-line entry point. All implementation lives in
`utils/`, grouped by responsibility. There is no separate top-level manuscript
plotting script and no separate test script to run.

```text
main.py       Run this file
logmod.py                   Console and file logging
common.py                  Existing configuration utilities
custom_logger.py           Logging adapter used by common.py
utils/
    cli.py                 Arguments, logging and selection of the run mode
    pipeline.py            Coordinates a full raw-data analysis
    config.py              Configuration loading and typed study settings
    constants.py           Outcome definitions, group labels and project paths
    data_io.py             Reads mappings and locates participant recordings
    features.py            Extracts trigger and head-orientation measurements
    extraction.py          Assembles trials and audits sample/sequence quality
    reporting.py           Writes result tables, reports and analysis notes
    snapshot.py            Saves, verifies and loads result-table snapshots
    models/                Statistical models grouped by research question
    plots/                 Analysis and manuscript figures plus export helpers
```

### Where to make changes

| Change | File or directory |
| --- | --- |
| Command-line options or cached/fresh route | `utils/cli.py` |
| Which analyses run and which tables are exported | `utils/pipeline.py` |
| Data locations and analysis settings | `config`, then `utils/config.py` |
| Trial-window, trigger or head-orientation extraction | `utils/features.py` |
| Primary comparisons | `utils/models/primary.py` |
| Secondary outcomes and condition follow-ups | `utils/models/secondary.py` |
| Linear changes, nonlinear trajectories or session parts | `utils/models/progression.py`, `splines.py`, `sessions.py` |
| eHMI learning or prior exposure | `utils/models/learning.py`, `exposure.py` |
| Participant summaries or demographic adjustment | `utils/models/participants.py`, `demographics.py` |
| Robustness and descriptive temporal/carryover analyses | `utils/models/robustness.py`, `descriptive.py` |
| Shared fitting and contrast utilities | `utils/models/helpers.py` |
| Figure orchestration and table-to-plot mapping | `utils/plots/__init__.py`, `utils/plots/analysis.py` |
| Overall outcomes and sequence plots | `utils/plots/outcomes.py`, `sequence.py` |
| Condition contrasts, temporal plots or robustness plots | `utils/plots/contrasts.py`, `temporal.py`, `robustness.py` |
| Manuscript plot layouts | `utils/plots/manuscript.py` |
| Analysis figure export formats | `utils/plots/export.py` |
| Console/file log format and handlers | `logmod.py` |

Follow the execution from `main.py` to `utils/cli.py`.
`run_workflow()` chooses saved CSVs, a compatible snapshot, or the full analysis
in `utils/pipeline.py`. Every route calls `regenerate_all_figures()` in
`utils/plots/`. The analysis plot dispatcher calls small plotting functions
with named inputs; the model calculations are separated from the plotting code.
There are no wildcard imports or duplicate copies of the statistical models.

Snapshots fingerprint the entry point, the entire Python package and the shared
logging/configuration helpers. Changes to an analysis module therefore invalidate
the saved-code signature just as changes to the old single script did. The
existing snapshot was explicitly migrated during the package refactor after
checking that the scientific function and plot bodies were unchanged.

To check Python formatting and lint errors:

```bash
.venv/bin/flake8
```

The root and GitHub Flake8 configurations use a 119-character line limit and
exclude generated outputs, the virtual environment and Unity's generated Library
cache. Version reporting remains available without importing the scientific
libraries:

```bash
.venv/bin/python main.py --version
```

### Finding manuscript figures by their LaTeX labels

Run `python main.py`. All manuscript figures are generated in
`_output/figures/` (or `<output>/figures/`). The run logs each label and PDF path.
Final copies controlled by `save_final` contain only HTML and PNG files.
For Overleaf, upload the manuscript PDFs from `_output/figures/` separately.

| LaTeX label | Generated filename stem |
| --- | --- |
| `fig:setup` | `experimental_setup` |
| `fig:sequence` | `trial_sequence` |
| `fig:q2` | `distance_ratings` |
| `fig:segments` | `session_changes` |

Every stem has `.pdf`, `.png`, `.html` and `.pickle` files. `figure_index.json`
records this mapping. `fig:q2` is a cross-reference label, not the output filename.
The manuscript includes it with `\includegraphics{figures/distance_ratings.pdf}`.
The Q2 panel is calculated from `condition_cell_contrasts.csv` and
`trial_level_common_window.csv`; it is not a manually created image.
