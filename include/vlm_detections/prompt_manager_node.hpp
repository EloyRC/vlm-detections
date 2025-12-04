#ifndef VLM_DETECTIONS__PROMPT_MANAGER_NODE_HPP_
#define VLM_DETECTIONS__PROMPT_MANAGER_NODE_HPP_

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

#include "perception_pipeline_msgs/msg/vlm_prompts.hpp"
#include "perception_pipeline_msgs/msg/text_prompt.hpp"
#include "perception_pipeline_msgs/msg/image_batch.hpp"
#include "strawberry_ros_msgs/msg/people.hpp"
#include "strawberry_ros_msgs/msg/faces.hpp"

namespace vlm_detections
{

/**
 * @brief Structure to hold prompt pairs from YAML dictionary
 */
struct PromptPair
{
  std::string label;
  std::string system_prompt;
  std::string user_prompt;
};

/**
 * @brief Node for managing image batching and prompt augmentation for VLM inference
 * 
 * This node:
 * - Buffers incoming images and organizes them as batches
 * - Loads prompts from a dictionary file (YAML)
 * - Annotates images with person/face information
 * - Publishes VLMPrompts messages containing images/batches and associated prompts
 */
class PromptManagerNode : public rclcpp::Node
{
public:
  explicit PromptManagerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~PromptManagerNode() override = default;

private:
  // Callback methods
  void image_cb(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
  void people_cb(const strawberry_ros_msgs::msg::People::SharedPtr msg);
  void faces_cb(const strawberry_ros_msgs::msg::Faces::SharedPtr msg);
  void sample_timer_cb();
  void publish_vlm_prompts();
  
  // Parameter callback
  rcl_interfaces::msg::SetParametersResult on_param_update(
    const std::vector<rclcpp::Parameter> & params);

  // Utility methods
  void load_prompt_dictionary(const std::string & path);
  sensor_msgs::msg::Image::SharedPtr clone_image(const sensor_msgs::msg::Image::SharedPtr & msg);
  double stamp_to_sec(const builtin_interfaces::msg::Time & stamp);
  strawberry_ros_msgs::msg::People::SharedPtr find_closest_people(
    const builtin_interfaces::msg::Time & image_stamp);
  sensor_msgs::msg::Image::SharedPtr annotate_image_with_people(
    const sensor_msgs::msg::Image::SharedPtr & image_msg,
    const strawberry_ros_msgs::msg::People::SharedPtr & people_msg);
  sensor_msgs::msg::Image::SharedPtr annotate_image_with_faces(
    const sensor_msgs::msg::Image::SharedPtr & image_msg,
    const strawberry_ros_msgs::msg::Faces::SharedPtr & faces_msg);
  sensor_msgs::msg::Image::SharedPtr decompress_image(
    const sensor_msgs::msg::CompressedImage::SharedPtr & compressed_msg);
  double period_from_fps();

  // Parameters
  std::string input_topic_;
  std::string people_topic_;
  std::string faces_topic_;
  std::string output_topic_;
  double fps_;
  std::string prompts_dictionary_path_;
  bool enable_people_annotation_;
  bool enable_faces_annotation_;
  double people_sync_tolerance_;

  // State
  std::vector<PromptPair> prompt_pairs_;
  int batch_capacity_;
  sensor_msgs::msg::Image::SharedPtr latest_msg_;
  std::deque<sensor_msgs::msg::Image::SharedPtr> batch_buffer_;
  
  // People/Faces synchronization
  std::deque<strawberry_ros_msgs::msg::People::SharedPtr> people_cache_;
  std::mutex people_cache_mutex_;
  static constexpr size_t MAX_CACHE_SIZE = 50;

  // ROS2 objects
  rclcpp::Publisher<perception_pipeline_msgs::msg::VLMPrompts>::SharedPtr publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_subscription_;
  rclcpp::Subscription<strawberry_ros_msgs::msg::People>::SharedPtr people_subscription_;
  rclcpp::Subscription<strawberry_ros_msgs::msg::Faces>::SharedPtr faces_subscription_;
  rclcpp::TimerBase::SharedPtr sample_timer_;
  OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
};

}  // namespace vlm_detections

#endif  // VLM_DETECTIONS__PROMPT_MANAGER_NODE_HPP_
