# by Shadab Alam <shaadalam.5u@gmail.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
from helper import HMD_helper
from custom_logger import CustomLogger
from logmod import logs
import common
import pandas as pd
import os


logs(show_level="info", show_color=True)
logger = CustomLogger(__name__)  # use custom logger
HMD = HMD_helper()

template = common.get_configs("plotly_template")
data_folder = common.get_configs("data")  # new location of the csv file with participant id
mapping = pd.read_csv(common.get_configs("mapping"))  # mapping file
output_folder = common.get_configs("output")
intake_questionnaire = common.get_configs("intake_questionnaire")   # intake questionnaire
post_experiment_questionnaire = common.get_configs("post_experiment_questionnaire")  # post-experiment questionnaire

intake_columns_to_plot = [
    "Do you consent to participate in this study as described in the information provided above?",
    "Have you read and understood the above instructions?",
    "What is your gender?",
    "Are you wearing any seeing aids during the experiments?",
    "Do you have problems with hearing?",
    "How often in the last month have you experienced virtual reality?",
    "I am comfortable with walking in areas with dense traffic.",
    "The presence of another pedestrian reduces my willingness to cross the street when a car is driving towards me.",
    "What is your primary mode of transportation?",
    "On average, how often did you drive a vehicle in the last 12 months?",
    "About how many kilometers did you drive in last 12 months?",
    "How often do you do the following?: Becoming angered by a particular type of driver, and indicate your hostility by whatever means you can.",  # noqa: E501
    "How often do you do the following?: Disregarding the speed limit on a motorway.",
    "How often do you do the following?: Disregarding the speed limit on a residential road. ",
    "How many accidents were you involved in when driving a car in the last 3 years? (please include all accidents, regardless of how they were caused, how slight they were, or where they happened)",  # noqa: E501
    "How often do you do the following?: Driving so close to the car in front that it would be difficult to stop in an emergency. ",  # noqa: E501
    "How often do you do the following?: Racing away from traffic lights with the intention of beating the driver next to you. ",  # noqa: E501
    "How often do you do the following?: Sounding your horn to indicate your annoyance with another road user. ",
    "How often do you do the following?: Using a mobile phone without a hands free kit.",
    "How often do you do the following?: Doing my best not to be obstacle for other drivers.",
    "I would like to communicate with other road users while crossing the road (for instance, using eye contact, gestures, verbal communication, etc.).",  # noqa: E501
    "I trust an automated car more than a manually driven car."
]

post_columns_to_plot = [
    "The presence of another pedestrian influenced my willingness to cross the road.",
    "The type of car (with eHMI or without eHMI) affected my decision to cross the road.",
    "I trust an automated car more than a manually driven car."
]

# Put all the questions where one need to calculate mean and standard deviation
intake_columns_distribution_to_plot = [
    "What is your age (in years)?",
    "At what age did you obtain your first license for driving a car or motorcycle?",
]

post_columns_distribution_to_plot = [
    "How stressful did you feel during the experiment?",
    "How anxious did you feel during the experiment?",
    "How realistic did you find the experiment?",
    "How would you rate your overall experience in this experiment?",
]

try:
    # Check if the directory already exists
    if not os.path.exists(output_folder):
        # Create the directory
        os.makedirs(output_folder)
        logger.info(f"Directory '{output_folder}' created successfully.")
except Exception as e:
    logger.error(f"Error occurred while creating directory: {e}")

# Execute analysis
if __name__ == "__main__":
    logger.info("Analysis started.")

    # Information on participants
    # HMD.plot_gender_by_nationality(intake_questionnaire,
    #                                gender_col="What is your gender?",
    #                                nationality_col="What is your nationality?")

    # HMD.plot_column_distribution(intake_questionnaire,
    #                              intake_columns_to_plot,
    #                              output_folder="output",
    #                              save_file=True,
    #                              tag="intake")

    # HMD.plot_column_distribution(post_experiment_questionnaire,
    #                              post_columns_to_plot,
    #                              output_folder="output",
    #                              save_file=True,
    #                              tag="post")

    # HMD.distribution_plots(intake_questionnaire,
    #                        intake_columns_distribution_to_plot,
    #                        output_folder="output",
    #                        save_file=True)

    # HMD.distribution_plots(post_experiment_questionnaire,
    #                        post_columns_distribution_to_plot,
    #                        output_folder="output",
    #                        save_file=True)

    # # Read and process data
    HMD.read_slider_data(data_folder, mapping, output_folder)

    # # Keypress data for yielding criteria
    # HMD.plot_column(mapping,
    #                 parameter=None,
    #                 xaxis_range=[0, 18],
    #                 compare_trial="video_1",
    #                 xaxis_title="Time, [s]",
    #                 yaxis_title="Percentage of trials with trigger key pressed",
    #                 name="all_values_with_yielding",
    #                 margin=dict(l=120, r=2, t=12, b=12))

    # HMD.plot_column(mapping,
    #                 parameter=None,
    #                 xaxis_range=[0, 11],
    #                 compare_trial="video_21",
    #                 xaxis_title="Time, [s]",
    #                 yaxis_title="Percentage of trials with trigger key pressed",
    #                 name="all_values_without_yielding",
    #                 margin=dict(l=120, r=2, t=12, b=12))

    # # Keypress data for yielding and eHMI criteria
    # HMD.plot_column(mapping,
    #                 parameter="eHMIOn",
    #                 parameter_value=0,
    #                 xaxis_range=[0, 18],
    #                 compare_trial="video_1",
    #                 xaxis_title="Time, [s]",
    #                 yaxis_title="Percentage of trials with trigger key pressed",
    #                 name="eHMI_off_yielding",
    #                 margin=dict(l=120, r=2, t=12, b=12))

    # HMD.plot_column(mapping,
    #                 parameter="eHMIOn",
    #                 parameter_value=1,
    #                 xaxis_range=[0, 18],
    #                 compare_trial="video_11",
    #                 xaxis_title="Time, [s]",
    #                 yaxis_title="Percentage of trials with trigger key pressed",
    #                 name="eHMI_on_yielding",
    #                 margin=dict(l=120, r=2, t=12, b=12))

    # HMD.plot_column(mapping,
    #                 parameter="eHMIOn",
    #                 parameter_value=0,
    #                 xaxis_range=[0, 11],
    #                 compare_trial="video_31",
    #                 xaxis_title="Time, [s]",
    #                 yaxis_title="Percentage of trials with trigger key pressed",
    #                 name="eHMI_off_non-yielding",
    #                 margin=dict(l=120, r=2, t=12, b=12))

    # HMD.plot_column(mapping,
    #                 parameter="eHMIOn",
    #                 parameter_value=1,
    #                 xaxis_range=[0, 11],
    #                 compare_trial="video_21",
    #                 xaxis_title="Time, [s]",
    #                 yaxis_title="Percentage of trials with trigger key pressed",
    #                 name="eHMI_on_non-yielding",
    #                 margin=dict(l=120, r=2, t=12, b=12))

    # # Keypress data for yielding, eHMI and position criteria

    # # First person view
    # HMD.plot_column(mapping,
    #                 parameter="eHMIOn",
    #                 parameter_value=1,
    #                 additional_parameter="camera",
    #                 additional_parameter_value=0,
    #                 xaxis_range=[0, 11],
    #                 compare_trial="video_21",
    #                 xaxis_title="Time, [s]",
    #                 yaxis_title="Percentage of trials with trigger key pressed",
    #                 name="first_eHMI_on_non-yielding",
    #                 margin=dict(l=120, r=2, t=12, b=12))

    # HMD.plot_column(mapping,
    #                 parameter="eHMIOn",
    #                 parameter_value=1,
    #                 additional_parameter="camera",
    #                 additional_parameter_value=0,
    #                 xaxis_range=[0, 18],
    #                 compare_trial="video_11",
    #                 xaxis_title="Time, [s]",
    #                 yaxis_title="Percentage of trials with trigger key pressed",
    #                 name="first_eHMI_on_yielding",
    #                 margin=dict(l=120, r=2, t=12, b=12))

    # HMD.plot_column(mapping,
    #                 parameter="eHMIOn",
    #                 parameter_value=0,
    #                 additional_parameter="camera",
    #                 additional_parameter_value=0,
    #                 xaxis_range=[0, 11],
    #                 compare_trial="video_31",
    #                 xaxis_title="Time, [s]",
    #                 yaxis_title="Percentage of trials with trigger key pressed",
    #                 name="first_eHMI_off_non-yielding",
    #                 margin=dict(l=120, r=2, t=12, b=12))

    # HMD.plot_column(mapping,
    #                 parameter="eHMIOn",
    #                 parameter_value=0,
    #                 additional_parameter="camera",
    #                 additional_parameter_value=0,
    #                 xaxis_range=[0, 18],
    #                 compare_trial="video_1",
    #                 xaxis_title="Time, [s]",
    #                 yaxis_title="Percentage of trials with trigger key pressed",
    #                 name="first_eHMI_off_yielding",
    #                 margin=dict(l=120, r=2, t=12, b=12))

    # # Second-person view
    # HMD.plot_column(mapping,
    #                 parameter="eHMIOn",
    #                 parameter_value=1,
    #                 additional_parameter="camera",
    #                 additional_parameter_value=1,
    #                 xaxis_range=[0, 11],
    #                 compare_trial="video_26",
    #                 xaxis_title="Time, [s]",
    #                 yaxis_title="Percentage of trials with trigger key pressed",
    #                 name="second_eHMI_on_non-yielding",
    #                 margin=dict(l=120, r=2, t=12, b=12))

    # HMD.plot_column(mapping,
    #                 parameter="eHMIOn",
    #                 parameter_value=1,
    #                 additional_parameter="camera",
    #                 additional_parameter_value=1,
    #                 xaxis_range=[0, 18],
    #                 compare_trial="video_16",
    #                 xaxis_title="Time, [s]",
    #                 yaxis_title="Percentage of trials with trigger key pressed",
    #                 name="second_eHMI_on_yielding",
    #                 margin=dict(l=120, r=2, t=12, b=12))

    # HMD.plot_column(mapping,
    #                 parameter="eHMIOn",
    #                 parameter_value=0,
    #                 additional_parameter="camera",
    #                 additional_parameter_value=1,
    #                 xaxis_range=[0, 11],
    #                 compare_trial="video_36",
    #                 xaxis_title="Time, [s]",
    #                 yaxis_title="Percentage of trials with trigger key pressed",
    #                 name="second_eHMI_off_non-yielding",
    #                 margin=dict(l=120, r=2, t=12, b=12))

    # HMD.plot_column(mapping,
    #                 parameter="eHMIOn",
    #                 parameter_value=0,
    #                 additional_parameter="camera",
    #                 additional_parameter_value=1,
    #                 xaxis_range=[0, 18],
    #                 compare_trial="video_6",
    #                 xaxis_title="Time, [s]",
    #                 yaxis_title="Percentage of trials with trigger key pressed",
    #                 name="second_eHMI_off_yielding",
    #                 margin=dict(l=120, r=2, t=12, b=12))

    # # Heatplot
    # HMD.heat_plot(folder_path=output_folder, mapping_df=mapping)

    # HMD.load_and_average_Q2(
    #     trigger_summary_csv=os.path.join("_output", "trigger_summary.csv"),
    #     responses_root=common.get_configs("data"),
    #     mapping_df=mapping)

    # HMD.analyze_and_plot_distance_effect_plotly(
    #     condition_df=pd.read_csv(os.path.join(output_folder, "condition_level_trigger_Q123.csv")),
    #     mapping_df=mapping,
    #     out_dir=output_folder,
    # )

    # Violin plots
    HMD.plot_2x4_violins(
        responses_csv=os.path.join(output_folder, "slider_input_behaviour.csv"),
        mapping=mapping,
        name="behaviour_of_the_other_pedestrian",
        x_ann=0.98,
        y_ann=0.98
    )

    HMD.plot_2x4_violins(
        responses_csv=os.path.join(output_folder, "slider_input_distance.csv"),
        mapping=mapping,
        name="distance_between_pedestrian",
        x_ann=0.98,
        y_ann=0.98
    )

    HMD.plot_2x4_violins(
        responses_csv=os.path.join(output_folder, "slider_input_intention.csv"),
        mapping=mapping,
        name="intention_of_the_vehicle",
        x_ann=0.98,
        y_ann=0.28
    )

    # Head rotation
    HMD.plot_yaw(mapping,
                 parameter=None,
                 xaxis_range=[0, 18],
                 compare_trial="video_1",
                 xaxis_title="Time, [s]",
                 yaxis_title="Percentage of trials with trigger key pressed",
                 name="all_yaw_values_with_yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    HMD.plot_yaw(mapping,
                 parameter=None,
                 xaxis_range=[0, 11],
                 compare_trial="video_21",
                 xaxis_title="Time, [s]",
                 yaxis_title="Percentage of trials with trigger key pressed",
                 name="all_yaw_values_without_yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    # Keypress data for yielding and eHMI criteria

    # eHMI is off and car is yielding
    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=0,
                 xaxis_range=[0, 18],
                 compare_trial="video_1",
                 xaxis_title="Time, [s]",
                 yaxis_title="Percentage of trials with trigger key pressed",
                 name="yaw_eHMI_off_yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    # eHMI is on and car is yielding
    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=1,
                 xaxis_range=[0, 18],
                 compare_trial="video_11",
                 xaxis_title="Time, [s]",
                 yaxis_title="Percentage of trials with trigger key pressed",
                 name="yaw_eHMI_on_yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    # eHMI is off and car is not yielding
    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=0,
                 xaxis_range=[0, 11],
                 compare_trial="video_31",
                 xaxis_title="Time, [s]",
                 yaxis_title="Percentage of trials with trigger key pressed",
                 name="yaw_eHMI_off_non-yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    # eHMI is off and car is not yielding
    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=1,
                 xaxis_range=[0, 11],
                 compare_trial="video_21",
                 xaxis_title="Time, [s]",
                 yaxis_title="Percentage of trials with trigger key pressed",
                 name="yaw_eHMI_on_non-yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    # Keypress data for yielding, eHMI and position criteria

    # First person view
    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=1,
                 additional_parameter="camera",
                 additional_parameter_value=0,
                 xaxis_range=[0, 11],
                 compare_trial="video_21",
                 xaxis_title="Time, [s]",
                 yaxis_title="Percentage of trials with trigger key pressed",
                 name="yaw_first_eHMI_on_non-yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=1,
                 additional_parameter="camera",
                 additional_parameter_value=0,
                 xaxis_range=[0, 18],
                 compare_trial="video_11",
                 xaxis_title="Time, [s]",
                 yaxis_title="Percentage of trials with trigger key pressed",
                 name="yaw_first_eHMI_on_yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=0,
                 additional_parameter="camera",
                 additional_parameter_value=0,
                 xaxis_range=[0, 11],
                 compare_trial="video_31",
                 xaxis_title="Time, [s]",
                 yaxis_title="Percentage of trials with trigger key pressed",
                 name="yaw_first_eHMI_off_non-yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=0,
                 additional_parameter="camera",
                 additional_parameter_value=0,
                 xaxis_range=[0, 18],
                 compare_trial="video_1",
                 xaxis_title="Time, [s]",
                 yaxis_title="Percentage of trials with trigger key pressed",
                 name="yaw_first_eHMI_off_yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    # Second-person view
    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=1,
                 additional_parameter="camera",
                 additional_parameter_value=1,
                 xaxis_range=[0, 11],
                 compare_trial="video_26",
                 xaxis_title="Time, [s]",
                 yaxis_title="Percentage of trials with trigger key pressed",
                 name="yaw_second_eHMI_on_non-yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=1,
                 additional_parameter="camera",
                 additional_parameter_value=1,
                 xaxis_range=[0, 18],
                 compare_trial="video_16",
                 xaxis_title="Time, [s]",
                 yaxis_title="Percentage of trials with trigger key pressed",
                 name="yaw_second_eHMI_on_yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=0,
                 additional_parameter="camera",
                 additional_parameter_value=1,
                 xaxis_range=[0, 11],
                 compare_trial="video_36",
                 xaxis_title="Time, [s]",
                 yaxis_title="Percentage of trials with trigger key pressed",
                 name="yaw_second_eHMI_off_non-yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=0,
                 additional_parameter="camera",
                 additional_parameter_value=1,
                 xaxis_range=[0, 18],
                 compare_trial="video_6",
                 xaxis_title="Time, [s]",
                 yaxis_title="Percentage of trials with trigger key pressed",
                 name="yaw_second_eHMI_off_yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    HMD.plot_yaw_frequencies_by_condition(
        mapping=mapping,
        yaw_files_dir=output_folder
        )
