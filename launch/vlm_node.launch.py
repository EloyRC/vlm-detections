from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_share_dir = get_package_share_directory('vlm_detections')
    config_dir = os.path.join(package_share_dir, 'config')
    
    # Optional: specify a custom config file path here, or leave empty to use package defaults
    # model_config_path = os.path.join(config_dir, 'model_config.yaml')
    prompts_dictionary_path = os.path.join(config_dir, 'prompts_dictionary.yaml')

    return LaunchDescription([
        Node(
            package='vlm_detections',
            executable='vlm_prompt_manager_cpp',
            name='vlm_prompt_manager_node',
            output='screen',
            parameters=[{
                'input_image_topic': '/azure_kinect/rgb/image_raw/compressed',
                'people_topic': '/strawberry/people',
                'output_topic': '/vlm_prompts',
                'fps': 0.1,  # Sample rate: 0.1 Hz = 1 sample every 10 seconds
                'batch_capacity': 1,  # 1=single image mode, >1=batch mode
                'prompts_dictionary': prompts_dictionary_path,
                'enable_people_annotation': True,
                'people_sync_tolerance': 0.2,  # seconds
                'enable_faces_annotation': False,
                'faces_topic': '/faces/results',
                'debug_image_topic': '/vlm_debug_image',
            }]
        ),
        Node(
            package='vlm_detections',
            executable='vlm_ros_node',
            name='vlm_detections_node',
            output='screen',
            parameters=[{
                'paused': False,
                # 'model_config_file': model_config_path,  # Uncomment to use custom config
            }]
        ),
    ])
