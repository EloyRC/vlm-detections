#include "vlm_detections/prompt_manager_node.hpp"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <algorithm>

namespace vlm_detections
{

PromptManagerNode::PromptManagerNode(const rclcpp::NodeOptions & options)
: Node("vlm_prompt_manager_node", options)
{
  // Declare parameters
  this->declare_parameter<std::string>("input_image_topic", "/camera/image_raw");
  this->declare_parameter<std::string>("people_topic", "/people");
  this->declare_parameter<std::string>("faces_topic", "/faces");
  this->declare_parameter<std::string>("output_topic", "/vlm_prompts");
  this->declare_parameter<double>("fps", 5.0);
  this->declare_parameter<int>("batch_capacity", 1);
  this->declare_parameter<std::string>("prompts_dictionary", "");
  this->declare_parameter<bool>("enable_people_annotation", false);
  this->declare_parameter<bool>("enable_faces_annotation", false);
  this->declare_parameter<double>("people_sync_tolerance", 0.2);
  this->declare_parameter<std::string>("debug_image_topic", "/vlm_debug_image");

  // Get parameter values
  input_topic_ = this->get_parameter("input_image_topic").as_string();
  people_topic_ = this->get_parameter("people_topic").as_string();
  faces_topic_ = this->get_parameter("faces_topic").as_string();
  output_topic_ = this->get_parameter("output_topic").as_string();
  fps_ = this->get_parameter("fps").as_double();
  batch_capacity_ = this->get_parameter("batch_capacity").as_int();
  prompts_dictionary_path_ = this->get_parameter("prompts_dictionary").as_string();
  enable_faces_annotation_ = this->get_parameter("enable_faces_annotation").as_bool();
  people_sync_tolerance_ = this->get_parameter("people_sync_tolerance").as_double();
  std::string debug_image_topic = this->get_parameter("debug_image_topic").as_string();

  // Validate parameters
  if (fps_ <= 0.0) {
    RCLCPP_WARN(this->get_logger(), "fps must be > 0; defaulting to 1.0");
    fps_ = 1.0;
  }
  if (batch_capacity_ < 1) {
    RCLCPP_WARN(this->get_logger(), "batch_capacity must be >= 1; defaulting to 1");
    batch_capacity_ = 1;
  }

  // NOTE: enable_faces_annotation is a temporary solution until the People ROS node is fixed
  // If faces annotation is enabled, disable people annotation
  if (enable_faces_annotation_) {
    enable_people_annotation_ = false;
  } else {
    enable_people_annotation_ = this->get_parameter("enable_people_annotation").as_bool();
  }

  // Load prompts from dictionary
  if (!prompts_dictionary_path_.empty()) {
    try {
      load_prompt_dictionary(prompts_dictionary_path_);
      RCLCPP_INFO(this->get_logger(), "Loaded %zu prompts from %s",
                  prompt_pairs_.size(), prompts_dictionary_path_.c_str());
    } catch (const std::exception & e) {
      RCLCPP_ERROR(this->get_logger(), "Failed to load prompts dictionary: %s", e.what());
    }
  } else {
    RCLCPP_WARN(this->get_logger(), "No prompts_dictionary parameter provided");
  }

  // Initialize buffer management
  latest_msg_ = nullptr;

  // Create publishers
  publisher_ = this->create_publisher<perception_pipeline_msgs::msg::VLMPrompts>(
    output_topic_, 10);
  debug_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
    debug_image_topic, 10);

  // Create subscriptions based on annotation mode
  // NOTE: If faces_annotation is enabled, ignore input_image_topic (temporary solution)
  if (enable_faces_annotation_) {
    faces_subscription_ = this->create_subscription<strawberry_ros_msgs::msg::Faces>(
      faces_topic_, 10,
      std::bind(&PromptManagerNode::faces_cb, this, std::placeholders::_1));
    image_subscription_ = nullptr;
    people_subscription_ = nullptr;
  } else {
    image_subscription_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
      input_topic_, 10,
      std::bind(&PromptManagerNode::image_cb, this, std::placeholders::_1));
    people_subscription_ = this->create_subscription<strawberry_ros_msgs::msg::People>(
      people_topic_, 10,
      std::bind(&PromptManagerNode::people_cb, this, std::placeholders::_1));
    faces_subscription_ = nullptr;
  }

  // Create sample timer - this controls inference rate
  sample_timer_ = this->create_wall_timer(
    std::chrono::duration<double>(period_from_fps()),
    std::bind(&PromptManagerNode::sample_timer_cb, this));

  // Setup parameter callback
  param_callback_handle_ = this->add_on_set_parameters_callback(
    std::bind(&PromptManagerNode::on_param_update, this, std::placeholders::_1));

  // Log startup info with actual inference timing
  const char * people_annotation = enable_people_annotation_ ? "enabled" : "disabled";
  const char * faces_annotation = enable_faces_annotation_ ? "enabled" : "disabled";
  double sample_period = period_from_fps();
  double actual_inference_period = sample_period * batch_capacity_;
  const char * mode = (batch_capacity_ > 1) ? "batch" : "single image";

  if (enable_faces_annotation_) {
    RCLCPP_INFO(this->get_logger(),
                "Prompt manager ready (FACES MODE - temporary): faces=%s, output=%s, "
                "mode=%s, sample_rate=%.2f Hz (period=%.3fs), batch_capacity=%d, "
                "inference_period=%.3fs, faces_annotation=%s, prompts=%zu",
                faces_topic_.c_str(), output_topic_.c_str(), mode, fps_, sample_period,
                batch_capacity_, actual_inference_period, faces_annotation,
                prompt_pairs_.size());
  } else {
    RCLCPP_INFO(this->get_logger(),
                "Prompt manager ready: input=%s, people=%s, output=%s, "
                "mode=%s, sample_rate=%.2f Hz (period=%.3fs), batch_capacity=%d, "
                "inference_period=%.3fs, people_annotation=%s, prompts=%zu",
                input_topic_.c_str(), people_topic_.c_str(), output_topic_.c_str(),
                mode, fps_, sample_period, batch_capacity_, actual_inference_period,
                people_annotation, prompt_pairs_.size());
  }
}

void PromptManagerNode::load_prompt_dictionary(const std::string & path)
{
  prompt_pairs_.clear();

  YAML::Node config = YAML::LoadFile(path);

  if (config.IsMap()) {
    for (const auto & kv : config) {
      std::string label = kv.first.as<std::string>();
      YAML::Node value = kv.second;

      if (value.IsMap() && (value["system"] || value["user"])) {
        PromptPair pair;
        pair.label = label;
        pair.system_prompt = value["system"] ? value["system"].as<std::string>() : "";
        pair.user_prompt = value["user"] ? value["user"].as<std::string>() : "";
        prompt_pairs_.push_back(pair);
      }
    }
  } else if (config.IsSequence()) {
    for (size_t i = 0; i < config.size(); ++i) {
      YAML::Node item = config[i];
      if (item.IsMap()) {
        PromptPair pair;
        pair.label = std::to_string(i);
        pair.system_prompt = item["system"] ? item["system"].as<std::string>() : "";
        pair.user_prompt = item["user"] ? item["user"].as<std::string>() : "";
        prompt_pairs_.push_back(pair);
      }
    }
  } else {
    throw std::runtime_error("Prompt dictionary must be a mapping or list");
  }
}

sensor_msgs::msg::Image::SharedPtr PromptManagerNode::clone_image(
  const sensor_msgs::msg::Image::SharedPtr & msg)
{
  auto clone = std::make_shared<sensor_msgs::msg::Image>();
  clone->header = msg->header;
  clone->height = msg->height;
  clone->width = msg->width;
  clone->encoding = msg->encoding;
  clone->is_bigendian = msg->is_bigendian;
  clone->step = msg->step;
  clone->data = msg->data;
  return clone;
}

double PromptManagerNode::stamp_to_sec(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) / 1e9;
}

strawberry_ros_msgs::msg::People::SharedPtr PromptManagerNode::find_closest_people(
  const builtin_interfaces::msg::Time & image_stamp)
{
  std::lock_guard<std::mutex> lock(people_cache_mutex_);

  if (people_cache_.empty()) {
    return nullptr;
  }

  double image_time = stamp_to_sec(image_stamp);

  // Find closest by timestamp difference
  auto best_match = std::min_element(
    people_cache_.begin(), people_cache_.end(),
    [this, image_time](const auto & a, const auto & b) {
      double diff_a = std::abs(stamp_to_sec(a->header.stamp) - image_time);
      double diff_b = std::abs(stamp_to_sec(b->header.stamp) - image_time);
      return diff_a < diff_b;
    });

  // Check if within tolerance
  double time_diff = std::abs(stamp_to_sec((*best_match)->header.stamp) - image_time);
  if (time_diff < people_sync_tolerance_) {
    return *best_match;
  }

  return nullptr;
}

sensor_msgs::msg::Image::SharedPtr PromptManagerNode::annotate_image_with_people(
  const sensor_msgs::msg::Image::SharedPtr & image_msg,
  const strawberry_ros_msgs::msg::People::SharedPtr & people_msg)
{
  try {
    // Convert ROS image to OpenCV format
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat & cv_img = cv_ptr->image;

    // Annotate each person
    for (const auto & person : people_msg->people) {
      int person_id = person.id;
      const auto & face = person.face;

      // Check if face has valid bounding box
      if (face.id == 0) {
        continue;
      }

      const auto & bbox = face.bb_field;
      double center_x = bbox.center.position.x;
      double center_y = bbox.center.position.y;
      double size_x = bbox.size_x;
      double size_y = bbox.size_y;

      // Convert to corner coordinates
      int x1 = static_cast<int>(center_x - size_x / 2.0);
      int y1 = static_cast<int>(center_y - size_y / 2.0);
      int x2 = static_cast<int>(center_x + size_x / 2.0);
      int y2 = static_cast<int>(center_y + size_y / 2.0);

      // Draw bounding box
      cv::Scalar color(0, 255, 0);  // Green
      int thickness = 2;
      cv::rectangle(cv_img, cv::Point(x1, y1), cv::Point(x2, y2), color, thickness);

      // Draw person ID label
      std::string label = "ID: " + std::to_string(person_id);
      int font = cv::FONT_HERSHEY_SIMPLEX;
      double font_scale = 0.6;
      int font_thickness = 2;

      // Get text size for background
      int baseline;
      cv::Size text_size = cv::getTextSize(label, font, font_scale, font_thickness, &baseline);

      // Draw background rectangle for text
      cv::rectangle(
        cv_img,
        cv::Point(x1, y1 - text_size.height - baseline - 5),
        cv::Point(x1 + text_size.width, y1),
        color,
        cv::FILLED);

      // Draw text
      cv::putText(
        cv_img,
        label,
        cv::Point(x1, y1 - baseline - 2),
        font,
        font_scale,
        cv::Scalar(0, 0, 0),  // Black text
        font_thickness);
    }

    // Convert back to ROS image message
    auto annotated_msg = cv_ptr->toImageMsg();
    return std::make_shared<sensor_msgs::msg::Image>(*annotated_msg);

  } catch (const cv_bridge::Exception & e) {
    RCLCPP_WARN(this->get_logger(), "Failed to annotate image: %s", e.what());
    return image_msg;
  }
}

sensor_msgs::msg::Image::SharedPtr PromptManagerNode::annotate_image_with_faces(
  const sensor_msgs::msg::Image::SharedPtr & image_msg,
  const strawberry_ros_msgs::msg::Faces::SharedPtr & faces_msg)
{
  try {
    // Convert ROS image to OpenCV format
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat & cv_img = cv_ptr->image;

    // Annotate each face
    for (const auto & face : faces_msg->faces) {
      int face_id = face.id;
      if (face_id == 0) {
        continue;
      }

      const auto & bbox = face.bb_field;
      double center_x = bbox.center.position.x;
      double center_y = bbox.center.position.y;
      double size_x = bbox.size_x;
      double size_y = bbox.size_y;

      // Convert to corner coordinates
      int x1 = static_cast<int>(center_x - size_x / 2.0);
      int y1 = static_cast<int>(center_y - size_y / 2.0);
      int x2 = static_cast<int>(center_x + size_x / 2.0);
      int y2 = static_cast<int>(center_y + size_y / 2.0);

      // Draw bounding box
      cv::Scalar color(0, 255, 0);  // Green
      int thickness = 2;
      cv::rectangle(cv_img, cv::Point(x1, y1), cv::Point(x2, y2), color, thickness);

      // Draw face ID label
      std::string label = "Face ID: " + std::to_string(face_id);
      int font = cv::FONT_HERSHEY_SIMPLEX;
      double font_scale = 0.6;
      int font_thickness = 2;

      // Get text size for background
      int baseline;
      cv::Size text_size = cv::getTextSize(label, font, font_scale, font_thickness, &baseline);

      // Draw background rectangle for text
      cv::rectangle(
        cv_img,
        cv::Point(x1, y1 - text_size.height - baseline - 5),
        cv::Point(x1 + text_size.width, y1),
        color,
        cv::FILLED);

      // Draw text
      cv::putText(
        cv_img,
        label,
        cv::Point(x1, y1 - baseline - 2),
        font,
        font_scale,
        cv::Scalar(0, 0, 0),  // Black text
        font_thickness);
    }

    // Convert back to ROS image message
    auto annotated_msg = cv_ptr->toImageMsg();
    return std::make_shared<sensor_msgs::msg::Image>(*annotated_msg);

  } catch (const cv_bridge::Exception & e) {
    RCLCPP_WARN(this->get_logger(), "Failed to annotate faces image: %s", e.what());
    return image_msg;
  }
}

sensor_msgs::msg::Image::SharedPtr PromptManagerNode::decompress_image(
  const sensor_msgs::msg::CompressedImage::SharedPtr & compressed_msg)
{
  try {
    // Decode compressed image
    cv::Mat cv_img = cv::imdecode(cv::Mat(compressed_msg->data), cv::IMREAD_COLOR);

    if (cv_img.empty()) {
      throw std::runtime_error("Failed to decode compressed image");
    }

    // Convert to ROS Image message
    cv_bridge::CvImage cv_bridge_image;
    cv_bridge_image.header = compressed_msg->header;
    cv_bridge_image.encoding = sensor_msgs::image_encodings::BGR8;
    cv_bridge_image.image = cv_img;

    return std::make_shared<sensor_msgs::msg::Image>(*cv_bridge_image.toImageMsg());

  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to decompress image: %s", e.what());
    return nullptr;
  }
}

void PromptManagerNode::image_cb(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
{
  auto decompressed = decompress_image(msg);
  if (decompressed) {
    latest_msg_ = decompressed;
  }
}

void PromptManagerNode::people_cb(const strawberry_ros_msgs::msg::People::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(people_cache_mutex_);
  people_cache_.push_back(msg);
  
  // Maintain max cache size
  if (people_cache_.size() > MAX_CACHE_SIZE) {
    people_cache_.pop_front();
  }
}

void PromptManagerNode::faces_cb(const strawberry_ros_msgs::msg::Faces::SharedPtr msg)
{
  // NOTE: This is a temporary solution until the People ROS node is fixed.
  // Faces messages contain the original image, so no synchronization is needed.
  
  try {
    // Decompress the embedded image
    auto decompressed = decompress_image(
      std::make_shared<sensor_msgs::msg::CompressedImage>(msg->original_image));
    
    if (!decompressed) {
      RCLCPP_ERROR(this->get_logger(), "Failed to decompress Faces original_image");
      return;
    }

    // Annotate with face IDs and bounding boxes
    auto annotated = annotate_image_with_faces(decompressed, msg);
    latest_msg_ = annotated;

  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to process Faces message: %s", e.what());
  }
}

void PromptManagerNode::sample_timer_cb()
{
  if (!latest_msg_) {
    return;
  }

  auto cloned = clone_image(latest_msg_);
  cloned->header.stamp = this->now();

  // Annotate with people information if enabled
  if (enable_people_annotation_) {
    auto people_msg = find_closest_people(latest_msg_->header.stamp);
    if (people_msg) {
      cloned = annotate_image_with_people(cloned, people_msg);
    } else {
      // Log occasionally if no people data found
      if (batch_buffer_.size() % 10 == 0) {
        RCLCPP_DEBUG(this->get_logger(),
                     "No synchronized People message found for image annotation");
      }
    }
  }

  // Accumulate images in buffer
  batch_buffer_.push_back(cloned);
  
  // Maintain max buffer size
  if (static_cast<int>(batch_buffer_.size()) > batch_capacity_) {
    batch_buffer_.pop_front();
  }
  
  // Publish when batch is full (batch_capacity=1 means publish immediately)
  if (static_cast<int>(batch_buffer_.size()) >= batch_capacity_) {
    publish_vlm_prompts();
  }
}

void PromptManagerNode::publish_vlm_prompts()
{
  if (batch_buffer_.empty()) {
    return;
  }

  // Build VLMPrompts message
  auto prompts_msg = std::make_shared<perception_pipeline_msgs::msg::VLMPrompts>();

  // Set prompts from dictionary
  for (const auto & pair : prompt_pairs_) {
    perception_pipeline_msgs::msg::TextPrompt text_prompt;
    text_prompt.system_prompt = pair.system_prompt;
    text_prompt.user_prompt = pair.user_prompt;
    prompts_msg->prompts.push_back(text_prompt);
  }

  // Set image data based on batch capacity (>1 = batch mode)
  bool is_batch_mode = (batch_capacity_ > 1);
  prompts_msg->has_image_batch = is_batch_mode;

  if (is_batch_mode) {
    // Create ImageBatch from buffer
    perception_pipeline_msgs::msg::ImageBatch batch_msg;
    batch_msg.timestamp = this->now();
    batch_msg.fps = static_cast<float>(fps_);
    
    for (const auto & img : batch_buffer_) {
      batch_msg.images.push_back(*img);
    }
    
    prompts_msg->image_batch = batch_msg;
    prompts_msg->image = sensor_msgs::msg::Image();  // Empty single image
  } else {
    // Use the single image from buffer
    prompts_msg->image = *batch_buffer_.back();
    prompts_msg->image_batch = perception_pipeline_msgs::msg::ImageBatch();  // Empty batch
    
    // Publish debug image
    debug_image_pub_->publish(prompts_msg->image);
  }

  publisher_->publish(*prompts_msg);
}

rcl_interfaces::msg::SetParametersResult PromptManagerNode::on_param_update(
  const std::vector<rclcpp::Parameter> & params)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;

  for (const auto & param : params) {
    if (param.get_name() == "fps") {
      double new_fps = param.as_double();
      if (new_fps > 0.0) {
        fps_ = new_fps;
        
        // Recreate sample timer
        sample_timer_->cancel();
        sample_timer_ = this->create_wall_timer(
          std::chrono::duration<double>(period_from_fps()),
          std::bind(&PromptManagerNode::sample_timer_cb, this));
        
        double inference_period = period_from_fps() * batch_capacity_;
        RCLCPP_INFO(this->get_logger(), 
                    "Updated sample rate to %.2f Hz (period=%.3fs), inference period=%.3fs",
                    fps_, period_from_fps(), inference_period);
      } else {
        result.successful = false;
        result.reason = "fps must be > 0";
      }
    } else if (param.get_name() == "batch_capacity") {
      int new_capacity = param.as_int();
      if (new_capacity >= 1) {
        batch_capacity_ = new_capacity;
        double inference_period = period_from_fps() * batch_capacity_;
        const char * mode = (batch_capacity_ > 1) ? "batch" : "single image";
        RCLCPP_INFO(this->get_logger(), 
                    "Updated batch_capacity to %d (mode=%s, inference period=%.3fs)",
                    batch_capacity_, mode, inference_period);
      } else {
        result.successful = false;
        result.reason = "batch_capacity must be >= 1";
      }
    } else if (param.get_name() == "enable_people_annotation") {
      enable_people_annotation_ = param.as_bool();
      RCLCPP_INFO(this->get_logger(), "People annotation %s",
                  enable_people_annotation_ ? "enabled" : "disabled");
    } else if (param.get_name() == "enable_faces_annotation") {
      // NOTE: enable_faces_annotation is temporary until People node is fixed
      enable_faces_annotation_ = param.as_bool();
      if (enable_faces_annotation_) {
        enable_people_annotation_ = false;
      }
      RCLCPP_WARN(this->get_logger(), "Faces annotation %s (temporary solution)",
                  enable_faces_annotation_ ? "enabled" : "disabled");
    } else if (param.get_name() == "people_sync_tolerance") {
      double new_tolerance = param.as_double();
      if (new_tolerance > 0.0) {
        people_sync_tolerance_ = new_tolerance;
        RCLCPP_INFO(this->get_logger(), "People sync tolerance set to %.3f s",
                    people_sync_tolerance_);
      } else {
        result.successful = false;
        result.reason = "people_sync_tolerance must be > 0";
      }
    }
  }

  return result;
}

double PromptManagerNode::period_from_fps()
{
  return 1.0 / std::max(fps_, 0.0001);
}

}  // namespace vlm_detections

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<vlm_detections::PromptManagerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
