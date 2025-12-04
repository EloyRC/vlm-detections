from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    
    return LaunchDescription([
        Node(
            package='vlm_detections',
            executable='vlm_ros_gui',
            name='vlm_detections_gui',
            output='screen',
        )
    ])
