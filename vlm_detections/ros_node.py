from __future__ import annotations
"""ROS2 inference node integrating existing adapters.

This node:
- Loads model_config.yaml once at startup to determine model, variant, threshold.
- Subscribes to /vlm_prompts (VLMPrompts) containing images/batches and prompts (configurable via param).
- Publishes structured detections on /vlm_detections (VLMOutputs).
- Publishes annotated image on /vlm_detections/image (sensor_msgs/Image).
- Respects pause flag (service /vlm_detections/set_pause).
- Inference timing is controlled by the prompt_manager_node that publishes VLMPrompts.
"""
import os
import json
import time
from typing import List, Optional, Dict, Any, Deque

import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time as RosTime
from std_msgs.msg import String
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Point32

from vlm_detections.core.runtime import (
    DetectorRuntime,
    ensure_valid_variant,
    MODEL_REGISTRY,
    default_variant_for,
    is_zero_shot_detector,
    is_prompt_based_vlm,
)
from vlm_detections.core.adapter_base import Detection
from vlm_detections.core.parsed_items import EntityProperty, EntityRelation
from vlm_detections.utils.config_loader import load_model_config, resolve_config_path
from vlm_detections.core.visualize import draw_detections
from perception_pipeline_msgs.msg import (
    Detection2D, VLMOutput, VLMOutputs, VLMMetadata, CamInfo, ImageBatch,
    PersonGesture, PersonEmotion, ActorRelation, VLMPrompts, TextPrompt
)
from vision_msgs.msg import BoundingBox2D, Pose2D
from geometry_msgs.msg import Polygon
try:  # Prefer real message type from ROS2
    from rcl_interfaces.msg import SetParametersResult  # type: ignore
except Exception:  # Fallback stub for environments (e.g., CI) without ROS installed
    class SetParametersResult:  # type: ignore
        def __init__(self, successful: bool = True, reason: str = ''):
            self.successful = successful
            self.reason = reason

# Constants
HOUSEKEEPING_TIMER_PERIOD = 1.0  # seconds
EWMA_ALPHA = 0.2  # Exponential weighted moving average alpha
LATENCY_SAMPLES_MAXLEN = 200  # Maximum number of latency samples to keep


class VLMDetectionNode(Node):
    def __init__(self):
        super().__init__('vlm_detections_node')
        # Parameters
        self.declare_parameter('prompts_topic', '/vlm_prompts')
        self.declare_parameter('paused', False)
        self.declare_parameter('output_image_topic', '/vlm_detections/image')
        self.declare_parameter('output_raw_topic', '/vlm_detections')
        self.declare_parameter('model_config_file', os.environ.get('VLM_CONFIG_FILE', ''))
        self.declare_parameter('device', 'auto')

        # Resolve parameters
        self.prompts_topic = self.get_parameter('prompts_topic').get_parameter_value().string_value or '/vlm_prompts'
        self.paused = self.get_parameter('paused').get_parameter_value().bool_value
        raw_config = self.get_parameter('model_config_file').get_parameter_value().string_value
        self.config_file = resolve_config_path(raw_config) if raw_config else None
        self.get_logger().info(f"Using model config file: {self.config_file or 'default from package'}")
        self.device = self.get_parameter('device').get_parameter_value().string_value or 'auto'

        # Runtime fields
        self.bridge = CvBridge()
        self.last_infer_time = 0.0
        self.prev_infer_time: Optional[float] = None
        self.last_delta_s: Optional[float] = None
        self.total_delta_s = 0.0
        self._delta_samples = 0
        self._pause_started_at: Optional[float] = time.monotonic() if self.paused else None
        
        # End-to-end latency tracking (image timestamp → output publication)
        self.last_e2e_latency_ms: Optional[float] = None
        self.total_e2e_latency_ms = 0.0
        self.e2e_latency_samples = 0
        self.max_e2e_latency_ms: Optional[float] = None
        self.min_e2e_latency_ms: Optional[float] = None

        # Publishers
        self.pub_outputs = self.create_publisher(VLMOutputs, self.get_parameter('output_raw_topic').get_parameter_value().string_value, 10)
        self.pub_img = self.create_publisher(Image, self.get_parameter('output_image_topic').get_parameter_value().string_value, 10)
        self.pub_diag = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.pub_status = self.create_publisher(String, '/vlm_detections/status', 10)

        # Subscriber for VLMPrompts with image QoS
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.prompts_sub = self.create_subscription(
            VLMPrompts,
            self.prompts_topic,
            self.prompts_cb,
            image_qos
        )

        # Periodic housekeeping timer
        self.timer = self.create_timer(HOUSEKEEPING_TIMER_PERIOD, self.periodic_housekeeping)

        # Metrics
        self.infer_count = 0
        self.last_latency_ms: Optional[float] = None
        self.total_latency_ms = 0.0
        self.error_count = 0
        self.ewma_latency_ms: Optional[float] = None
        self.alpha_ewma = EWMA_ALPHA
        self.lat_samples: Deque[float] = deque(maxlen=LATENCY_SAMPLES_MAXLEN)
        self.has_torch = False
        self.torch_cuda = False
        try:  # Optional torch metrics
            import torch  # type: ignore
            self.has_torch = True
            self.torch_cuda = torch.cuda.is_available()
            self._torch = torch
        except Exception:
            self._torch = None
        # Service for external pause control
        self.pause_service = self.create_service(SetBool, 'vlm_detections/set_pause', self.handle_set_pause)

        # Dynamic parameter callback
        self.add_on_set_parameters_callback(self.on_param_update)

        # Initial load for runtime
        self.runtime = DetectorRuntime()
        self.current = {
            'model_name': 'OpenAI Vision (API)',
            'variant': default_variant_for('OpenAI Vision (API)'),
            'threshold': 0.25,
        }
        self.load_from_config()

    @staticmethod
    def _build_detection_msg(det: Detection) -> Detection2D:
        """Build Detection2D message from Detection object.
        
        Args:
            det: Detection object with xyxy, score, label, text, and optional polygon
            
        Returns:
            Detection2D message with BoundingBox2D, polygon_mask, score, label, and empty uids
        """
        msg = Detection2D()
        
        # Convert xyxy to BoundingBox2D
        x1, y1, x2, y2 = det.xyxy
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        size_x = x2 - x1
        size_y = y2 - y1
        
        bbox = BoundingBox2D()
        bbox.center = Pose2D()
        bbox.center.position.x = center_x
        bbox.center.position.y = center_y
        bbox.center.theta = 0.0  # No rotation
        bbox.size_x = size_x
        bbox.size_y = size_y
        msg.bbox = bbox
        
        # Set score
        msg.score = float(det.score)
        
        # Set label: use label if non-empty, otherwise use text
        if det.label:
            msg.label = str(det.label)
        else:
            msg.label = str(det.text) if det.text else ''
        
        # Convert polygon if present
        if det.polygon:
            polygon = Polygon()
            for pt in det.polygon:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2 and isinstance(pt[0], (int, float)):
                    point = Point32()
                    point.x = float(pt[0])
                    point.y = float(pt[1])
                    point.z = 0.0
                    polygon.points.append(point)
                elif isinstance(pt, (list, tuple)) and pt and isinstance(pt[0], (list, tuple)):
                    for sub in pt:
                        if isinstance(sub, (list, tuple)) and len(sub) >= 2:
                            point = Point32()
                            point.x = float(sub[0])
                            point.y = float(sub[1])
                            point.z = 0.0
                            polygon.points.append(point)
            msg.polygon_mask = polygon
        else:
            msg.polygon_mask = Polygon()  # Empty polygon
        
        # Leave uids empty for now
        msg.uids = []
        
        return msg

    @staticmethod
    def _map_gesture_to_id(gesture_value: str) -> int:
        """Map gesture string to PersonGesture constant.
        
        Args:
            gesture_value: Gesture string (e.g., "waving", "thumbs_up")
            
        Returns:
            PersonGesture constant ID, or NO_GESTURE if not recognized
        """
        gesture_map = {
            'waving': PersonGesture.WAVING,
            'thumbs_up': PersonGesture.THUMBS_UP,
            'thumbsup': PersonGesture.THUMBS_UP,
            'pointing': PersonGesture.POINTING,
            'clapping': PersonGesture.CLAPPING,
            'nodding': PersonGesture.NODDING,
            'shaking_head': PersonGesture.SHAKING_HEAD,
            'shakinghead': PersonGesture.SHAKING_HEAD,
            'raising_hand': PersonGesture.RAISING_HAND,
            'raisinghand': PersonGesture.RAISING_HAND,
            'crossing_arms': PersonGesture.CROSSING_ARMS,
            'crossingarms': PersonGesture.CROSSING_ARMS,
            'no_gesture': PersonGesture.NO_GESTURE,
            'nogesture': PersonGesture.NO_GESTURE,
            'none': PersonGesture.NO_GESTURE,
        }
        normalized = gesture_value.lower().replace(' ', '_').replace('-', '_')
        return gesture_map.get(normalized, PersonGesture.NO_GESTURE)

    @staticmethod
    def _map_emotion_to_id(emotion_value: str) -> int:
        """Map emotion string to PersonEmotion constant.
        
        Args:
            emotion_value: Emotion string (e.g., "happy", "sad")
            
        Returns:
            PersonEmotion constant ID, or NEUTRAL if not recognized
        """
        emotion_map = {
            'confused': PersonEmotion.CONFUSED,
            'happy': PersonEmotion.HAPPY,
            'sad': PersonEmotion.SAD,
            'angry': PersonEmotion.ANGRY,
            'surprised': PersonEmotion.SURPRISED,
            'neutral': PersonEmotion.NEUTRAL,
            'upset': PersonEmotion.UPSET,
            'scared': PersonEmotion.SCARED,
            'shy': PersonEmotion.SHY,
            'distressed': PersonEmotion.DISTRESSED,
            'excited': PersonEmotion.EXCITED,
        }
        normalized = emotion_value.lower().replace(' ', '_').replace('-', '_')
        return emotion_map.get(normalized, PersonEmotion.NEUTRAL)

    @staticmethod
    def _map_relation_to_id(relation_predicate: str) -> int:
        """Map relation predicate string to ActorRelation constant.
        
        Args:
            relation_predicate: Relation string (e.g., "looking_at", "holding")
            
        Returns:
            ActorRelation constant ID, or CAN_INTERACT_WITH if not recognized
        """
        relation_map = {
            'can_interact_with': ActorRelation.CAN_INTERACT_WITH,
            'caninteractwith': ActorRelation.CAN_INTERACT_WITH,
            'is_interacting_with': ActorRelation.IS_INTERACTING_WITH,
            'isinteractingwith': ActorRelation.IS_INTERACTING_WITH,
            'interacting': ActorRelation.IS_INTERACTING_WITH,
            'is_looking_at': ActorRelation.IS_LOOKING_AT,
            'islookingat': ActorRelation.IS_LOOKING_AT,
            'looking_at': ActorRelation.IS_LOOKING_AT,
            'lookingat': ActorRelation.IS_LOOKING_AT,
            'is_pointing_at': ActorRelation.IS_POINTING_AT,
            'ispointingat': ActorRelation.IS_POINTING_AT,
            'pointing_at': ActorRelation.IS_POINTING_AT,
            'pointingat': ActorRelation.IS_POINTING_AT,
            'is_attending_to': ActorRelation.IS_ATTENDING_TO,
            'isattendingto': ActorRelation.IS_ATTENDING_TO,
            'attending_to': ActorRelation.IS_ATTENDING_TO,
            'attendingto': ActorRelation.IS_ATTENDING_TO,
            'is_talking_to': ActorRelation.IS_TALKING_TO,
            'istalkingto': ActorRelation.IS_TALKING_TO,
            'talking_to': ActorRelation.IS_TALKING_TO,
            'talkingto': ActorRelation.IS_TALKING_TO,
            'calls_attention_from': ActorRelation.CALLS_ATTENTION_FROM,
            'callsattentionfrom': ActorRelation.CALLS_ATTENTION_FROM,
        }
        normalized = relation_predicate.lower().replace(' ', '_').replace('-', '_')
        return relation_map.get(normalized, ActorRelation.CAN_INTERACT_WITH)

    @staticmethod
    def _extract_entity_id(entity_name: str) -> int:
        """Extract numeric ID from entity name (e.g., 'person_1' -> 1).
        
        Args:
            entity_name: Entity identifier string
            
        Returns:
            Extracted ID or 0 if not parseable
        """
        import re
        match = re.search(r'(\d+)', entity_name)
        return int(match.group(1)) if match else 0

    def _build_vlm_output_msg(
        self,
        detections: List[Detection],
        entity_properties: List[EntityProperty],
        entity_relations: List[EntityRelation],
        raw_text: str,
        system_prompt: str,
        user_prompt: str,
        prompt_label: str,
    ) -> VLMOutput:
        """Build VLMOutput message from detections, properties, relations, and prompts.
        
        Args:
            detections: List of Detection objects
            entity_properties: List of EntityProperty objects
            entity_relations: List of EntityRelation objects
            raw_text: Raw output from the model
            system_prompt: System prompt used
            user_prompt: User prompt used
            prompt_label: Label for this prompt
            
        Returns:
            VLMOutput message with metadata, detections, gestures, emotions, relations, threshold, and raw output
        """
        msg = VLMOutput()
        
        # Build VLMMetadata
        metadata = VLMMetadata()
        metadata.model_name = str(self.current.get('model_name') or '')
        text_prompt = TextPrompt()
        text_prompt.system_prompt = system_prompt
        text_prompt.user_prompt = user_prompt
        metadata.prompt = text_prompt
        msg.vlm_metadata = metadata
        
        # Convert detections
        msg.detections = [self._build_detection_msg(d) for d in detections]
        
        # Convert entity properties to PersonGesture or PersonEmotion messages
        gestures = []
        emotions = []
        for prop in entity_properties:
            entity_id = self._extract_entity_id(prop.entity)
            
            # Check if this is a gesture property
            if prop.property_name.lower() in ['gesture', 'gestures', 'hand_gesture', 'body_gesture']:
                gesture_msg = PersonGesture()
                gesture_msg.gesture_id = self._map_gesture_to_id(prop.property_value)
                gesture_msg.person_uuid = entity_id
                gesture_msg.person_trackid = entity_id
                gestures.append(gesture_msg)
            
            # Check if this is an emotion property
            elif prop.property_name.lower() in ['emotion', 'emotions', 'facial_emotion', 'feeling']:
                emotion_msg = PersonEmotion()
                emotion_msg.emotion_id = self._map_emotion_to_id(prop.property_value)
                emotion_msg.person_uuid = entity_id
                emotion_msg.person_trackid = entity_id
                emotions.append(emotion_msg)
        
        msg.person_gestures = gestures
        msg.person_emotions = emotions
        
        # Convert entity relations to ActorRelation messages
        actor_relations = []
        for rel in entity_relations:
            relation_msg = ActorRelation()
            relation_msg.relation_id = self._map_relation_to_id(rel.predicate)
            relation_msg.subject_uuid = self._extract_entity_id(rel.subject)
            relation_msg.subject_trackid = self._extract_entity_id(rel.subject)
            relation_msg.object_uuid = self._extract_entity_id(rel.object)
            relation_msg.object_trackid = self._extract_entity_id(rel.object)
            actor_relations.append(relation_msg)
        
        msg.actor_relations = actor_relations
        
        # Set threshold, raw output, and empty caption
        msg.threshold = float(self.current.get('threshold') or 0.0)
        msg.raw_output = raw_text or ''
        msg.caption = ''
        
        return msg

    # --- Model configuration management ---
    def load_from_config(self):
        """Load model configuration from model_config.yaml file."""
        config = load_model_config(self.config_file)
        self.get_logger().info(f"Loading config from: {self.config_file or 'package defaults'}")
        
        # Validate and select model
        requested_model = config.get('model')
        model_name = requested_model if requested_model in MODEL_REGISTRY else list(MODEL_REGISTRY.keys())[0]
        if requested_model and requested_model not in MODEL_REGISTRY:
            fallback = list(MODEL_REGISTRY.keys())[0] if MODEL_REGISTRY else 'OpenAI Vision (API)'
            self.get_logger().warning(
                f"Requested model '{requested_model}' not in registry; falling back to '{fallback}'."
            )
            model_name = fallback
        
        # Extract configuration
        variant = ensure_valid_variant(model_name, config.get('model_variant', default_variant_for(model_name)))
        threshold = float(config.get('threshold', 0.25))
        
        # Load / ensure model
        self.runtime.ensure_model(model_name, variant, self.device)
        
        # Apply generation params if present
        gen_params = config.get('generation_params', {})
        if gen_params:
            self.runtime.apply_generation_params(gen_params)
        
        # Store current configuration
        self.current = {
            'model_name': model_name,
            'variant': variant,
            'threshold': threshold,
        }
        self.get_logger().info('Model configuration loaded and model ensured.')

    def prompts_cb(self, msg: VLMPrompts) -> None:
        """Handle incoming VLMPrompts message containing images and prompts."""
        if self.paused:
            return
        
        self.last_infer_time = time.time()
        
        # Extract input timestamp for end-to-end latency tracking
        has_batch = msg.has_image_batch
        input_timestamp = (
            msg.image_batch.timestamp if has_batch and msg.image_batch.images
            else msg.image.header.stamp if not has_batch
            else None
        )
        
        try:
            if has_batch:
                self._process_batch(msg.image_batch, msg.prompts, input_timestamp)
            else:
                self._process_single_image_with_prompts(msg.image, msg.prompts, input_timestamp)
        except Exception as exc:
            self.error_count += 1
            self.get_logger().error(f"Inference error in prompts_cb: {exc}")
    
    def _process_single_image_with_prompts(
        self, 
        image_msg: Image, 
        prompts: List[TextPrompt], 
        input_timestamp = None
    ) -> None:
        """Process single image with multiple prompts.
        
        Note: Classes are provided by the PromptManagerNode in the user_prompt text,
        so we pass an empty list to the adapter's infer() method.
        """
        try:
            cv_img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f"cv_bridge conversion failed: {exc}")
            return
        
        classes = []  # Classes are in the prompt text from PromptManagerNode
        threshold = self.current['threshold']
        
        detector = self.runtime.detector
        if detector is None:
            self.get_logger().error('Detector not initialized')
            return
        
        t0 = time.perf_counter()
        outputs_payload: List[Dict[str, object]] = []
        all_detections: List[Detection] = []
        
        # Run inference for each prompt
        for idx, text_prompt in enumerate(prompts):
            system_prompt = text_prompt.system_prompt
            user_prompt = text_prompt.user_prompt
            label = f'prompt_{idx}'
            
            try:
                # Call appropriate interface based on adapter protocol
                # Prefer PromptBasedVLM if we have a user prompt
                if is_zero_shot_detector(detector) and not user_prompt:
                    dets = detector.infer(cv_img, classes, threshold)
                    props, rels, raw_text = [], [], ""
                elif is_prompt_based_vlm(detector):
                    if not user_prompt:
                        self.get_logger().warning(f"PromptBasedVLM requires user prompt for prompt {idx}; skipping.")
                        continue

                    dets, props, rels, raw_text = detector.infer_from_image(
                        cv_img, user_prompt, system_prompt, threshold
                    )
                else:
                    self.get_logger().warning(f"Unknown adapter protocol for prompt {idx}")
                    continue
                
                record = {
                    'label': label,
                    'system': system_prompt,
                    'user': user_prompt,
                    'detections': dets,
                    'properties': props,
                    'relations': rels,
                    'raw': raw_text,
                }
                outputs_payload.append(record)
                all_detections.extend(dets)
                self.get_logger().debug(f"Inference completed for prompt {idx} with {len(dets)} detections.")
            except Exception as exc:
                self.get_logger().warning(f"Inference failed for prompt {idx}: {exc}")
        
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._update_latency_metrics(latency_ms)
        
        # Publish using first detection set and raw output
        raw_text = outputs_payload[0]['raw'] if outputs_payload else ''
        self._publish_outputs(cv_img, all_detections, raw_text, outputs_payload, input_timestamp)
    
    def _process_batch(
        self, 
        batch_msg: ImageBatch, 
        prompts: List[TextPrompt], 
        input_timestamp = None
    ) -> None:
        """Process an image batch with prompts."""
        frames: List[np.ndarray] = []
        timestamps: List[float] = []
        base_ros_time: Optional[RosTime] = None
        
        for idx, image_msg in enumerate(batch_msg.images):
            try:
                frames.append(self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8'))
            except Exception as exc:
                self.get_logger().error(f"Failed to convert image {idx} in batch: {exc}")
                return
            
            stamp = getattr(image_msg, 'header', None)
            stamp_msg = getattr(stamp, 'stamp', None)
            ts_value: Optional[float] = None
            if stamp_msg is not None:
                try:
                    sample_time = RosTime.from_msg(stamp_msg)
                    if base_ros_time is None:
                        base_ros_time = sample_time
                    duration = sample_time - base_ros_time
                    ts_value = max(duration.nanoseconds / 1e9, 0.0)
                except Exception:
                    ts_value = None
            if ts_value is None:
                fps_safe = batch_msg.fps if batch_msg.fps > 0 else 0.0
                ts_value = (idx / fps_safe) if fps_safe > 0 else float(idx)
            timestamps.append(float(ts_value))
        
        if not frames:
            self.get_logger().warning('Received empty image batch; skipping inference.')
            return
        
        classes = []  # Classes are in the prompt text from PromptManagerNode
        threshold = self.current['threshold']
        frame = frames[-1]  # Use last frame for visualization
        
        detector = self.runtime.detector
        if detector is None:
            self.get_logger().error('Detector not initialized')
            return
        
        t0 = time.perf_counter()
        outputs_payload: List[Dict[str, object]] = []
        all_detections: List[Detection] = []
        
        # Check if detector supports batch processing
        batch_method = getattr(detector, 'infer_from_batch', None)
        frames_with_ts = list(zip(frames, timestamps))
        
        # Run inference for each prompt
        print(f"Processing batch of {len(frames)} images with {len(prompts)} prompts")
        for idx, text_prompt in enumerate(prompts):
            system_prompt = text_prompt.system_prompt
            user_prompt = text_prompt.user_prompt
            label = f'prompt_{idx}'
            
            try:
                if callable(batch_method):
                    # Use batch inference - prefer PromptBasedVLM if we have a prompt
                    if is_prompt_based_vlm(detector) and user_prompt:
                        batch_result = batch_method(
                            frames_with_ts, user_prompt, system_prompt, threshold
                        )
                    elif is_zero_shot_detector(detector):
                        batch_result = batch_method(
                            frames_with_ts, classes, threshold
                        )
                    else:
                        self.get_logger().warning(f"Unknown adapter protocol for prompt {idx}")
                        continue
                    
                    if isinstance(batch_result, tuple) and len(batch_result) == 2:
                        dets, raw_text = batch_result
                        props = []
                        rels = []
                    else:
                        dets = []
                        raw_text = str(batch_result) if batch_result else ''
                        props = []
                        rels = []
                else:
                    # Fall back to single-frame inference on last frame
                    if is_prompt_based_vlm(detector) and user_prompt:
                        dets, props, rels, raw_text = detector.infer_from_image(
                            frame, user_prompt, system_prompt, threshold
                        )
                    elif is_zero_shot_detector(detector):
                        dets, props, rels, raw_text = detector.infer(
                            frame, classes, threshold
                        )
                    else:
                        self.get_logger().warning(f"Unknown adapter protocol for prompt {idx}")
                        continue
                
                record = {
                    'label': label,
                    'system': system_prompt,
                    'user': user_prompt,
                    'detections': dets,
                    'properties': props,
                    'relations': rels,
                    'raw': raw_text,
                }
                outputs_payload.append(record)
                all_detections.extend(dets)
            except Exception as exc:
                self.get_logger().warning(f"Batch inference failed for prompt {idx}: {exc}")
        
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._update_latency_metrics(latency_ms)
        
        # Publish using aggregated detections
        raw_text = outputs_payload[0]['raw'] if outputs_payload else ''
        self._publish_outputs(frame, all_detections, raw_text, outputs_payload, input_timestamp)

    def _update_latency_metrics(self, latency_ms: float) -> None:
        self.last_latency_ms = latency_ms
        self.total_latency_ms += latency_ms
        self.infer_count += 1
        if self.ewma_latency_ms is None:
            self.ewma_latency_ms = latency_ms
        else:
            self.ewma_latency_ms = self.alpha_ewma * latency_ms + (1 - self.alpha_ewma) * self.ewma_latency_ms
        self.lat_samples.append(latency_ms)

    def _publish_outputs(
        self,
        cv_img: np.ndarray,
        detections: List[Detection],
        raw_text: str,
        outputs_payload: List[Dict[str, object]],
        input_timestamp = None
    ) -> None:
        """Publish VLM detection outputs, annotated image, and status.
        
        Args:
            cv_img: Source image (BGR)
            detections: List of detections from the model
            raw_text: Raw output text
            outputs_payload: List of dictionaries containing detection results
            input_timestamp: Original image timestamp for e2e latency calculation
        """
        publish_time = time.monotonic()
        self._record_inference_timing(publish_time)
        
        # Calculate end-to-end latency (input image timestamp → output publication)
        if input_timestamp is not None:
            now_ros = self.get_clock().now()
            input_time_sec = float(input_timestamp.sec) + float(input_timestamp.nanosec) / 1e9
            output_time_sec = float(now_ros.seconds_nanoseconds()[0]) + float(now_ros.seconds_nanoseconds()[1]) / 1e9
            e2e_latency_ms = (output_time_sec - input_time_sec) * 1000.0
            
            if e2e_latency_ms >= 0:  # Sanity check for clock issues
                self.last_e2e_latency_ms = e2e_latency_ms
                self.total_e2e_latency_ms += e2e_latency_ms
                self.e2e_latency_samples += 1
                
                if self.max_e2e_latency_ms is None or e2e_latency_ms > self.max_e2e_latency_ms:
                    self.max_e2e_latency_ms = e2e_latency_ms
                if self.min_e2e_latency_ms is None or e2e_latency_ms < self.min_e2e_latency_ms:
                    self.min_e2e_latency_ms = e2e_latency_ms
        vis_bgr = draw_detections(cv_img, detections)
        status_dict = {
            'detections': len(detections),
            'threshold': self.current['threshold'],
            'paused': self.paused,
            'last_latency_ms': self.last_latency_ms,
            'avg_latency_ms': (self.total_latency_ms / self.infer_count) if self.infer_count else None,
            'last_delta_t_s': self.last_delta_s,
            'avg_delta_t_s': self._average_delta_interval(),
            'last_e2e_latency_ms': self.last_e2e_latency_ms,
            'avg_e2e_latency_ms': (self.total_e2e_latency_ms / self.e2e_latency_samples) if self.e2e_latency_samples else None,
            'max_e2e_latency_ms': self.max_e2e_latency_ms,
            'min_e2e_latency_ms': self.min_e2e_latency_ms,
            'inference_count': self.infer_count,
        }

        # Build VLMOutputs message with new structure
        outputs_msg = VLMOutputs()
        
        # Set header
        outputs_msg.header.stamp = self.get_clock().now().to_msg()
        outputs_msg.header.frame_id = 'camera_frame'
        
        # Set source image
        outputs_msg.source_img = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
        
        # Set camera info
        cam_info = CamInfo()
        cam_info.camera_id = 'default_camera'
        h, w = cv_img.shape[:2]
        cam_info.camera_img_width = int(w)
        cam_info.camera_img_height = int(h)
        cam_info.camera_focal_length = 0.0  # Unknown focal length
        outputs_msg.cam_info = cam_info
        
        # Build VLMOutput messages
        outputs_msg.vlm_outputs = []
        for payload in outputs_payload:
            det_list = payload.get('detections') or []
            if not isinstance(det_list, list):
                det_list = []
            det_objects = [d for d in det_list if isinstance(d, Detection)]
            
            # Extract entity properties and relations
            props_list = payload.get('properties') or []
            if not isinstance(props_list, list):
                props_list = []
            prop_objects = [p for p in props_list if isinstance(p, EntityProperty)]
            
            rels_list = payload.get('relations') or []
            if not isinstance(rels_list, list):
                rels_list = []
            rel_objects = [r for r in rels_list if isinstance(r, EntityRelation)]
            
            outputs_msg.vlm_outputs.append(
                self._build_vlm_output_msg(
                    det_objects,
                    prop_objects,
                    rel_objects,
                    str(payload.get('raw', '')),
                    str(payload.get('system', '')),
                    str(payload.get('user', '')),
                    str(payload.get('label', '')),
                )
            )
        
        self.pub_outputs.publish(outputs_msg)

        # Publish annotated image
        img_msg = self.bridge.cv2_to_imgmsg(vis_bgr, encoding='bgr8')
        self.pub_img.publish(img_msg)
        
        # Publish status
        status_msg = String()
        status_msg.data = json.dumps(status_dict)
        self.pub_status.publish(status_msg)

    def periodic_housekeeping(self):
        self.publish_diagnostics()

    # --- Services & Parameter handling ---
    def _apply_pause_state(self, new_state: bool) -> None:
        if new_state == self.paused:
            if new_state and self._pause_started_at is None:
                self._pause_started_at = time.monotonic()
            return
        now = time.monotonic()
        if new_state:
            self._pause_started_at = now
        else:
            self._pause_started_at = None
        self.paused = new_state

    def handle_set_pause(self, request: SetBool.Request, response: SetBool.Response):  # type: ignore
        self._apply_pause_state(bool(request.data))
        response.success = True
        response.message = f"pause set to {self.paused}"
        return response

    def on_param_update(self, params):  # rclpy parameter callback signature
        """Dynamic parameter update callback.

        Returns a `SetParametersResult` indicating success/failure.
        Unknown parameters are ignored (treated as success).
        """
        success = True
        reason_parts = []
        for p in params:
            if p.name == 'paused':
                self._apply_pause_state(bool(p.value))
            elif p.name == 'prompts_topic':
                new_topic = str(p.value).strip()
                if new_topic:
                    self.prompts_topic = new_topic
                    # Recreate subscription
                    if hasattr(self, 'prompts_sub') and self.prompts_sub is not None:
                        try:
                            self.destroy_subscription(self.prompts_sub)
                        except Exception:
                            pass
                    image_qos = QoSProfile(
                        reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST,
                        depth=1
                    )
                    self.prompts_sub = self.create_subscription(
                        VLMPrompts,
                        self.prompts_topic,
                        self.prompts_cb,
                        image_qos
                    )
                    self.get_logger().info(f'Updated prompts topic to {self.prompts_topic}')
            # ignore other params silently
        reason = '; '.join(reason_parts) if reason_parts else ''
        return SetParametersResult(successful=success, reason=reason)

    def publish_diagnostics(self):
        diag = DiagnosticArray()
        diag.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.name = 'vlm_detections'
        status.hardware_id = 'vlm_vlm'
        status.level = DiagnosticStatus.OK
        status.message = 'OK'
        kv = []
        kv.append(KeyValue(key='model', value=f"{self.current.get('model_name')}:{self.current.get('variant')}"))
        kv.append(KeyValue(key='threshold', value=str(self.current.get('threshold'))))
        # Inference timing controlled by prompt_manager_node
        kv.append(KeyValue(key='paused', value=str(self.paused)))
        
        # End-to-end latency (image timestamp → output publication)
        if self.last_e2e_latency_ms is not None:
            kv.append(KeyValue(key='last_e2e_latency_ms', value=f'{self.last_e2e_latency_ms:.1f}'))
        if self.e2e_latency_samples > 0:
            avg_e2e = self.total_e2e_latency_ms / self.e2e_latency_samples
            kv.append(KeyValue(key='avg_e2e_latency_ms', value=f'{avg_e2e:.1f}'))
        if self.max_e2e_latency_ms is not None:
            kv.append(KeyValue(key='max_e2e_latency_ms', value=f'{self.max_e2e_latency_ms:.1f}'))
        if self.min_e2e_latency_ms is not None:
            kv.append(KeyValue(key='min_e2e_latency_ms', value=f'{self.min_e2e_latency_ms:.1f}'))
        kv.append(KeyValue(key='inference_count', value=str(self.infer_count)))
        if self.last_latency_ms is not None:
            kv.append(KeyValue(key='last_latency_ms', value=f"{self.last_latency_ms:.2f}"))
        avg_latency = (self.total_latency_ms / self.infer_count) if self.infer_count else 0.0
        kv.append(KeyValue(key='avg_latency_ms', value=f"{avg_latency:.2f}"))
        if self.last_delta_s is not None:
            kv.append(KeyValue(key='delta_t_last_s', value=f"{self.last_delta_s:.3f}"))
        avg_delta = self._average_delta_interval()
        if avg_delta is not None:
            kv.append(KeyValue(key='delta_t_avg_s', value=f"{avg_delta:.3f}"))
        if self.ewma_latency_ms is not None:
            kv.append(KeyValue(key='ewma_latency_ms', value=f"{self.ewma_latency_ms:.2f}"))
        if self.lat_samples:
            samples_sorted = sorted(self.lat_samples)
            def q(p):
                if not samples_sorted:
                    return 0.0
                k = (len(samples_sorted)-1)*p
                f = int(k); c = min(f+1, len(samples_sorted)-1); frac = k - f
                return samples_sorted[f] + (samples_sorted[c]-samples_sorted[f])*frac
            kv.append(KeyValue(key='p50_latency_ms', value=f"{q(0.5):.2f}"))
            kv.append(KeyValue(key='p90_latency_ms', value=f"{q(0.9):.2f}"))
            kv.append(KeyValue(key='p95_latency_ms', value=f"{q(0.95):.2f}"))
        if self.torch_cuda:
            try:
                alloc = self._torch.cuda.memory_allocated() / (1024*1024)
                reserved = self._torch.cuda.memory_reserved() / (1024*1024)
                kv.append(KeyValue(key='cuda_mem_alloc_mb', value=f"{alloc:.1f}"))
                kv.append(KeyValue(key='cuda_mem_reserved_mb', value=f"{reserved:.1f}"))
            except Exception:
                pass
        kv.append(KeyValue(key='error_count', value=str(self.error_count)))
        status.values = kv
        diag.status.append(status)
        self.pub_diag.publish(diag)



    def _record_inference_timing(self, event_time: float) -> None:
        """Record timing between inferences for metrics."""
        if self.prev_infer_time is not None:
            delta = event_time - self.prev_infer_time
            if delta >= 0:
                self.last_delta_s = delta
                self.total_delta_s += delta
                self._delta_samples += 1
        else:
            self.last_delta_s = None
        self.prev_infer_time = event_time
        self.last_infer_time = time.time()

    def _average_delta_interval(self) -> Optional[float]:
        if self._delta_samples <= 0:
            return None
        return self.total_delta_s / self._delta_samples if self.total_delta_s >= 0 else None


def main(args=None):
    rclpy.init(args=args)
    node = VLMDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
