# Multi-pedestrian interaction with automated vehicle
Framework for the analysis of crossing behaviour in the interaction between multiple pedestrians and an automated vehicle, from the perspective of one of the pedestrians using a crowdsourcing approach.

## Citation
If you use the simulator for academic work please cite the following papers:

>  


## Getting started
[![Python Version](https://img.shields.io/badge/python-3.9.11-blue.svg)](https://www.python.org/downloads/release/python-3919/)
[![Package Manager: uv](https://img.shields.io/badge/package%20manager-uv-green)](https://docs.astral.sh/uv/)

Tested with **Python 3.9.11** and the [`uv`](https://docs.astral.sh/uv/) package manager.  
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
git clone https://github.com/bazilinskyy/multiped
cd multiped
```

**Step 5:** Ensure correct Python version. If you don’t already have Python 3.9.11 installed, let `uv` fetch it:
```command line
uv python install 3.9.11
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

### Realtion between trigger press and response from the participant
[![behaviour_vs_Q2_scatter](figures/crossing_risk_vs_Q2_scatter.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/crossing_risk_vs_Q2_scatter.html)
Condition-wise relationship between behavioural trigger and Q2 across all distances and factors.

[![near_minus_far_behaviour_vs_Q2](figures/near_minus_far_crossing_risk_vs_Q2.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/near_minus_far_crossing_risk_vs_Q2.html)
Near (1–2 m) minus far (4–5 m) differences in trigger and Q2 for each yielding × eHMI × position context.

[![behaviour_full_factorial](figures/crossing_risk_full_factorial.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/crossing_risk_full_factorial.html)
Behavioural trigger (0–100) across distance, split by yielding, eHMI, and participant position.

[![Q2_full_factorial](figures/Q2_full_factorial.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/Q2_full_factorial.html)
Self-reported distance influence (Q2, 0–100) across distance, split by yielding, eHMI, and participant position.

[![behaviour_full_factorial_legend_yielding](figures/crossing_risk_full_factorial_legend_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/crossing_risk_full_factorial_legend_yielding.html)
Behavioural trigger across distance with yielding shown in the legend and faceted by eHMI and position.

[![behaviour_full_factorial_legend_eHMI](figures/crossing_risk_full_factorial_legend_eHMI.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/crossing_risk_full_factorial_legend_eHMI.html)
Behavioural trigger across distance with eHMI status in the legend and faceted by yielding and position.

[![Q2_full_factorial_legend_yielding](figures/Q2_full_factorial_legend_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/Q2_full_factorial_legend_yielding.html)
Q2 across distance with yielding shown in the legend and faceted by eHMI and position.

[![Q2_full_factorial_legend_eHMI](figures/Q2_full_factorial_legend_eHMI.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/Q2_full_factorial_legend_eHMI.html)
Q2 across distance with eHMI status in the legend and faceted by yielding and position.

[![behaviour_of_the_other_pedestrian](figures/behaviour_of_the_other_pedestrian.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/behaviour_of_the_other_pedestrian.html)  
Self-reported influence of the other pedestrian’s behaviour on participants’ decision to cross (0–100), split by vehicle yielding, eHMI status, and visibility of the other pedestrian.

[![distance_between_pedestrian](figures/distance_between_pedestrian.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/distance_between_pedestrian.html)  
Self-reported influence of the distance between pedestrians on participants’ decision to cross (0–100), split by vehicle yielding, eHMI status, and visibility of the other pedestrian.

[![intention_of_the_vehicle](figures/intention_of_the_vehicle.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/intention_of_the_vehicle.html)  
Self-reported understanding of the vehicle’s intention (0–100), split by vehicle yielding, eHMI status, and visibility of the other pedestrian.



### Trigger press from the participant

[![Car is yielding](figures/all_videos_kp_slider_plot_all_values_with_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/all_videos_kp_slider_plot_all_values_with_yielding.html)
Car is yielding

[![Car is not yielding](figures/all_videos_kp_slider_plot_all_values_without_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/all_videos_kp_slider_plot_all_values_without_yielding.html)
Car is not yielding

[![Car is yielding and equipped with eHMI](figures/all_videos_kp_slider_plot_eHMI_on_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/all_videos_kp_slider_plot_eHMI_on_yielding.html)
Car is yielding and equipped with eHMI

[![Car is not yielding and equipped with eHMI](figures/all_videos_kp_slider_plot_eHMI_on_non-yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/all_videos_kp_slider_plot_eHMI_on_non-yielding.html)
Car is not yielding and equipped with eHMI

[![Car is yielding and not equipped with eHMI](figures/all_videos_kp_slider_plot_eHMI_off_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/all_videos_kp_slider_plot_eHMI_off_yielding.html)
Car is yielding and not equipped with eHMI

[![Car is not yielding and not equipped with eHMI](figures/all_videos_kp_slider_plot_eHMI_off_non-yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/all_videos_kp_slider_plot_eHMI_off_non-yielding.html)
Car is not yielding and not equipped with eHMI

[![Participant is in front with car equipped with eHMI and not yielding](figures/all_videos_kp_slider_plot_first_eHMI_on_non-yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/all_videos_kp_slider_plot_first_eHMI_on_non-yielding.html)
Participant is in front with car equipped with eHMI and not yielding

[![Participant is in front with car equipped with eHMI and yielding](figures/all_videos_kp_slider_plot_first_eHMI_on_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/all_videos_kp_slider_plot_first_eHMI_on_yielding.html)
Participant is in front with car equipped with eHMI and yielding

[![Participant is in front with car equipped with no eHMI and not yielding](figures/all_videos_kp_slider_plot_first_eHMI_off_non-yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/all_videos_kp_slider_plot_first_eHMI_off_non-yielding.html)
Participant is in front with car equipped with no eHMI and not yielding

[![Participant is in front with car equipped with no eHMI and yielding](figures/all_videos_kp_slider_plot_first_eHMI_off_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/all_videos_kp_slider_plot_first_eHMI_off_yielding.html)
Participant is in front with car equipped with no eHMI and yielding

[![Participant is in back with car equipped with eHMI and not yielding](figures/all_videos_kp_slider_plot_second_eHMI_on_non-yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/all_videos_kp_slider_plot_second_eHMI_on_non-yielding.html)
Participant is in back with car equipped with eHMI and not yielding

[![Participant is in back with car equipped with eHMI and yielding](figures/all_videos_kp_slider_plot_second_eHMI_on_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/all_videos_kp_slider_plot_second_eHMI_on_yielding.html)
Participant is in back with car equipped with eHMI and yielding

[![Participant is in back with car equipped with no eHMI and not yielding](figures/all_videos_kp_slider_plot_second_eHMI_off_non-yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/all_videos_kp_slider_plot_second_eHMI_off_non-yielding.html)
Participant is in back with car equipped with no eHMI and not yielding

[![Participant is in back with car equipped with no eHMI and yielding](figures/all_videos_kp_slider_plot_second_eHMI_off_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/all_videos_kp_slider_plot_second_eHMI_off_yielding.html)
Participant is in back with car equipped with no eHMI and yielding

[![Ratio-based heatplot showing relationships between yielding, eHMI, and pedestrian distance conditions](figures/heatmap.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/heatmap.html)
Ratio-based heatplot showing relationships between yielding, eHMI, and pedestrian distance conditions.


### Head movement from the participant

[![Car is yielding](figures/all_yaw_values_with_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/all_yaw_values_with_yielding.html)
Car is yielding

[![Car is not yielding](figures/all_yaw_values_without_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/all_yaw_values_without_yielding.html)
Car is not yielding

[![Car is yielding and equipped with eHMI](figures/yaw_eHMI_off_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_eHMI_off_yielding.html)
Car is yielding and equipped with eHMI

[![Car is not yielding and equipped with eHMI](figures/yaw_eHMI_on_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_eHMI_on_yielding.html)
Car is not yielding and equipped with eHMI

[![Car is yielding and not equipped with eHMI](figures/yaw_eHMI_off_non-yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_eHMI_off_non-yielding.html)
Car is yielding and not equipped with eHMI

[![Car is not yielding and not equipped with eHMI](figures/yaw_eHMI_on_non-yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_eHMI_on_non-yielding.html)
Car is not yielding and not equipped with eHMI

[![Participant is in front with car equipped with eHMI and not yielding](figures/yaw_first_eHMI_on_non-yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_first_eHMI_on_non-yielding.html)
Participant is in front with car equipped with eHMI and not yielding

[![Participant is in front with car equipped with eHMI and yielding](figures/yaw_first_eHMI_on_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_first_eHMI_on_yielding.html)
Participant is in front with car equipped with eHMI and yielding

[![Participant is in front with car equipped with no eHMI and not yielding](figures/yaw_first_eHMI_off_non-yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_first_eHMI_off_non-yielding.html)
Participant is in front with car equipped with no eHMI and not yielding

[![Participant is in front with car equipped with no eHMI and yielding](figures/yaw_first_eHMI_off_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_first_eHMI_off_yielding.html)
Participant is in front with car equipped with no eHMI and yielding

[![Participant is in back with car equipped with eHMI and not yielding](figures/yaw_second_eHMI_on_non-yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_second_eHMI_on_non-yielding.html)
Participant is in back with car equipped with eHMI and not yielding

[![Participant is in back with car equipped with eHMI and yielding](figures/yaw_second_eHMI_on_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_second_eHMI_on_yielding.html)
Participant is in back with car equipped with eHMI and yielding

[![Participant is in back with car equipped with no eHMI and not yielding](figures/yaw_second_eHMI_off_non-yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_second_eHMI_off_non-yielding.html)
Participant is in back with car equipped with no eHMI and not yielding

[![Participant is in back with car equipped with no eHMI and yielding](figures/yaw_second_eHMI_off_yielding.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_second_eHMI_off_yielding.html)
Participant is in back with car equipped with no eHMI and yielding

[![Participant can see the other person; yaw distribution across all yielding and eHMI conditions](figures/yaw_hist_can_see.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_hist_can_see.html)  
Participant can see the other person; yaw distribution across all yielding and eHMI conditions

[![Participant cannot see the other person; yaw distribution across all yielding and eHMI conditions](figures/yaw_hist_cannot_see.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_hist_cannot_see.html)  
Participant cannot see the other person; yaw distribution across all yielding and eHMI conditions

[![Yaw distributions by camera view and distance; 2×5 grid of conditions](figures/yaw_hist_cam_dist.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/yaw_hist_cam_dist.html)  
Yaw distributions by camera view (can / cannot see the other person) and distance; 2×5 grid of conditions


### Response of the participant


[![Age Distribution of the participants](figures/what_is_your_age_in_years.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/what_is_your_age_in_years.html)
Age Distribution of the participants

[![Gender Distribution of the participants](figures/what_is_your_gender_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/what_is_your_gender_intake.html)
Gender Distribution of the participants

[![Consent to participate in the study](figures/do_you_consent_to_participate_in_this_study_as_described_in_the_information_provided_above_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/do_you_consent_to_participate_in_this_study_as_described_in_the_information_provided_above_intake.html)
Consent to participate in the study

[![Understanding of instructions](figures/have_you_read_and_understood_the_above_instructions_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/have_you_read_and_understood_the_above_instructions_intake.html)
Understanding of instructions  

[![Use of seeing aids](figures/are_you_wearing_any_seeing_aids_during_the_experiments_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/are_you_wearing_any_seeing_aids_during_the_experiments_intake.html)
Use of seeing aids  
 
[![Hearing problems](figures/do_you_have_problems_with_hearing_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/do_you_have_problems_with_hearing_intake.html)
Hearing problems 

[![Experience with virtual reality](figures/how_often_in_the_last_month_have_you_experienced_virtual_reality_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/how_often_in_the_last_month_have_you_experienced_virtual_reality_intake.html)
Experience with virtual reality  


[![Comfort with walking in dense traffic](figures/i_am_comfortable_with_walking_in_areas_with_dense_traffic_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/i_am_comfortable_with_walking_in_areas_with_dense_traffic_intake.html)
Comfort with walking in dense traffic  


[![Effect of pedestrian presence](figures/the_presence_of_another_pedestrian_reduces_my_willingness_to_cross_the_street_when_a_car_is_driving_towards_me_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/the_presence_of_another_pedestrian_reduces_my_willingness_to_cross_the_street_when_a_car_is_driving_towards_me_intake.html)
Effect of pedestrian presence on willingness to cross  


[![Primary mode of transportation](figures/what_is_your_primary_mode_of_transportation_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/what_is_your_primary_mode_of_transportation_intake.html)
Primary mode of transportation  

 
[![Frequency of driving](figures/on_average_how_often_did_you_drive_a_vehicle_in_the_last_12_months_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/on_average_how_often_did_you_drive_a_vehicle_in_the_last_12_months_intake.html)
Frequency of driving in the last 12 months 


[![Kilometers driven](figures/about_how_many_kilometers_did_you_drive_in_last_12_months_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/about_how_many_kilometers_did_you_drive_in_last_12_month_intake_.html)
Kilometers driven in the last 12 months  


[![Age at obtaining first license](figures/at_what_age_did_you_obtain_your_first_license_for_driving_a_car_or_motorcycle.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/at_what_age_did_you_obtain_your_first_license_for_driving_a_car_or_motorcycle.html)
Age at obtaining first license  


[![Accidents in last 3 years](figures/how_many_accidents_were_you_involved_in_when_driving_a_car_in_the_last_3_years_please_include_all_accidents_regardless_of_how_they_were_caused_how_slight_they_were_or_where_they_happened_in.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/how_many_accidents_were_you_involved_in_when_driving_a_car_in_the_last_3_years_please_include_all_accidents_regardless_of_how_they_were_caused_how_slight_they_were_or_where_they_happened_in.html)
Accidents in last 3 years  


[![Risky driving behaviors](figures/how_often_do_you_do_the_following_driving_so_close_to_the_car_in_front_that_it_would_be_difficult_to_stop_in_an_emergency_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/how_often_do_you_do_the_following_driving_so_close_to_the_car_in_front_that_it_would_be_difficult_to_stop_in_an_emergency_intake.html)
How often do you do the following?: Driving so close to the car in front that it would be difficult to stop in an emergency. 

 
[![Willingness to communicate](figures/i_would_like_to_communicate_with_other_road_users_while_crossing_the_road_for_instance_using_eye_contact_gestures_verbal_communication_etc_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/i_would_like_to_communicate_with_other_road_users_while_crossing_the_road_for_instance_using_eye_contact_gestures_verbal_communication_etc_intake.html)
I would like to communicate with other road users while crossing the road (for instance, using eye contact, gestures, verbal communication, etc.).

[![Trust in automated cars](figures/i_trust_an_automated_car_more_than_a_manually_driven_car_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/i_trust_an_automated_car_more_than_a_manually_driven_car_intake.html)
Trust in automated vs manually driven cars  

[![Influence of pedestrian](figures/the_presence_of_another_pedestrian_influenced_my_willingness_to_cross_the_road_post.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/the_presence_of_another_pedestrian_influenced_my_willingness_to_cross_the_road_post.html)
Influence of another pedestrian on willingness to cross  


[![Effect of car type](figures/the_presence_of_another_pedestrian_reduces_my_willingness_to_cross_the_street_when_a_car_is_driving_towards_me_intake.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/the_presence_of_another_pedestrian_reduces_my_willingness_to_cross_the_street_when_a_car_is_driving_towards_me_intake.html)
Effect of car type (with or without eHMI)  


[![Trust in automated cars](figures/i_trust_an_automated_car_more_than_a_manually_driven_car_post.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/i_trust_an_automated_car_more_than_a_manually_driven_car_post.html)
Trust in automated vs manually driven cars (after experiment)  

 
[![Stress during the experiment](figures/how_stressful_did_you_feel_during_the_experiment.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/how_stressful_did_you_feel_during_the_experiment.html)
Stress during the experiment 


[![Anxiety during the experiment](figures/how_anxious_did_you_feel_during_the_experiment.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/how_anxious_did_you_feel_during_the_experiment.html)
Anxiety during the experiment


[![Realism of the experiment](figures/how_realistic_did_you_find_the_experiment.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/how_realistic_did_you_find_the_experiment.html)
Realism of the experiment


[![Overall experience rating](figures/how_would_you_rate_your_overall_experience_in_this_experiment.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/multiped/blob/main/figures/how_would_you_rate_your_overall_experience_in_this_experiment.html)
Overall experience rating


## Contact
If you have any questions or suggestions, feel free to reach out to md_shadab_alam@outlook.com and 
