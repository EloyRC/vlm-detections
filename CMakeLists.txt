cmake_minimum_required(VERSION 3.8)
project(vlm_detections)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(perception_pipeline_msgs REQUIRED)
find_package(strawberry_ros_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# C++ executable: prompt_manager_node
add_executable(vlm_prompt_manager_cpp
  src/prompt_manager_node.cpp
)

target_include_directories(vlm_prompt_manager_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(vlm_prompt_manager_cpp
  rclcpp
  std_msgs
  sensor_msgs
  cv_bridge
  perception_pipeline_msgs
  strawberry_ros_msgs
  OpenCV
)

target_link_libraries(vlm_prompt_manager_cpp
  yaml-cpp
)

# Install C++ executables
install(TARGETS
  vlm_prompt_manager_cpp
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables (entry points from setup.py)
install(PROGRAMS
  scripts/vlm_ros_node
  scripts/vlm_ros_gui
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files to share directory
install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING
  PATTERN "*.yaml"
  PATTERN "*.yml"
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
