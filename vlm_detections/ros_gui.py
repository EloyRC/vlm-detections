from __future__ import annotations
"""Standalone Gradio monitor for ROS-based VLM detections."""

import os
import json
import threading
import time
from typing import Any, Dict

import gradio as gr
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from cv_bridge import CvBridge

from perception_pipeline_msgs.msg import VLMOutputs, VLMOutput, PersonGesture, PersonEmotion, ActorRelation

POLL_INTERVAL = 0.5  # seconds


def _gesture_id_to_name(gesture_id: int) -> str:
    """Convert PersonGesture ID to human-readable name."""
    gesture_names = {
        PersonGesture.WAVING: 'waving',
        PersonGesture.THUMBS_UP: 'thumbs_up',
        PersonGesture.POINTING: 'pointing',
        PersonGesture.CLAPPING: 'clapping',
        PersonGesture.NODDING: 'nodding',
        PersonGesture.SHAKING_HEAD: 'shaking_head',
        PersonGesture.RAISING_HAND: 'raising_hand',
        PersonGesture.CROSSING_ARMS: 'crossing_arms',
        PersonGesture.NO_GESTURE: 'no_gesture',
    }
    return gesture_names.get(gesture_id, f'unknown_{gesture_id}')


def _emotion_id_to_name(emotion_id: int) -> str:
    """Convert PersonEmotion ID to human-readable name."""
    emotion_names = {
        PersonEmotion.CONFUSED: 'confused',
        PersonEmotion.HAPPY: 'happy',
        PersonEmotion.SAD: 'sad',
        PersonEmotion.ANGRY: 'angry',
        PersonEmotion.SURPRISED: 'surprised',
        PersonEmotion.NEUTRAL: 'neutral',
        PersonEmotion.UPSET: 'upset',
        PersonEmotion.SCARED: 'scared',
        PersonEmotion.SHY: 'shy',
        PersonEmotion.DISTRESSED: 'distressed',
        PersonEmotion.EXCITED: 'excited',
    }
    return emotion_names.get(emotion_id, f'unknown_{emotion_id}')


def _relation_id_to_name(relation_id: int) -> str:
    """Convert ActorRelation ID to human-readable name."""
    relation_names = {
        ActorRelation.CAN_INTERACT_WITH: 'can_interact_with',
        ActorRelation.IS_INTERACTING_WITH: 'is_interacting_with',
        ActorRelation.IS_LOOKING_AT: 'is_looking_at',
        ActorRelation.IS_POINTING_AT: 'is_pointing_at',
        ActorRelation.IS_ATTENDING_TO: 'is_attending_to',
        ActorRelation.IS_TALKING_TO: 'is_talking_to',
        ActorRelation.CALLS_ATTENTION_FROM: 'calls_attention_from',
    }
    return relation_names.get(relation_id, f'unknown_{relation_id}')


class _GuiListener(Node):
    def __init__(self, cache: Dict[str, Any],
                 image_topic: str = '/vlm_detections/image',
                 raw_topic: str = '/vlm_detections',
                 status_topic: str = '/vlm_detections/status') -> None:
        super().__init__('vlm_gui_listener')
        self.cache = cache
        self.bridge = CvBridge()
        self.create_subscription(Image, image_topic, self.on_image, 10)
        self.create_subscription(VLMOutputs, raw_topic, self.on_raw, 10)
        self.create_subscription(String, status_topic, self.on_status, 10)

    def on_image(self, msg: Image) -> None:
        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv_rgb = cv_bgr[:, :, ::-1]
            with self.cache['lock']:
                self.cache['frame'] = cv_rgb
        except Exception:
            return

    def on_raw(self, msg: VLMOutputs) -> None:
        """Parse new VLMOutputs message structure with nested vlm_outputs.
        
        Args:
            msg: VLMOutputs message containing header, source_img, cam_info, and vlm_outputs
        """
        outputs_payload = []
        for vlm_detection in msg.vlm_outputs:
            detections = []
            for det in vlm_detection.detections:
                # Extract bounding box from BoundingBox2D
                bbox = det.bbox
                center_x = bbox.center.position.x
                center_y = bbox.center.position.y
                size_x = bbox.size_x
                size_y = bbox.size_y
                
                # Convert back to xyxy format for display
                x1 = center_x - size_x / 2.0
                y1 = center_y - size_y / 2.0
                x2 = center_x + size_x / 2.0
                y2 = center_y + size_y / 2.0
                
                det_dict = {
                    'xyxy': [x1, y1, x2, y2],
                    'score': float(det.score),
                    'label': det.label,
                }
                
                # Extract polygon if present
                if det.polygon_mask and det.polygon_mask.points:
                    det_dict['polygon'] = [
                        {'x': float(pt.x), 'y': float(pt.y), 'z': float(pt.z)}
                        for pt in det.polygon_mask.points
                    ]
                else:
                    det_dict['polygon'] = []
                
                detections.append(det_dict)
            
            # Extract metadata
            metadata = vlm_detection.vlm_metadata
            
            # Extract gestures
            gestures = []
            for gesture in vlm_detection.person_gestures:
                gestures.append({
                    'gesture': _gesture_id_to_name(gesture.gesture_id),
                    'person_uuid': int(gesture.person_uuid),
                    'person_trackid': int(gesture.person_trackid),
                })
            
            # Extract emotions
            emotions = []
            for emotion in vlm_detection.person_emotions:
                emotions.append({
                    'emotion': _emotion_id_to_name(emotion.emotion_id),
                    'person_uuid': int(emotion.person_uuid),
                    'person_trackid': int(emotion.person_trackid),
                })
            
            # Extract relations
            relations = []
            for relation in vlm_detection.actor_relations:
                relations.append({
                    'relation': _relation_id_to_name(relation.relation_id),
                    'subject_uuid': int(relation.subject_uuid),
                    'subject_trackid': int(relation.subject_trackid),
                    'object_uuid': int(relation.object_uuid),
                    'object_trackid': int(relation.object_trackid),
                })
            
            outputs_payload.append({
                'model_name': metadata.model_name,
                'system_prompt': metadata.prompt.system_prompt,
                'user_prompt': metadata.prompt.user_prompt,
                'threshold': float(vlm_detection.threshold),
                'detections': detections,
                'gestures': gestures,
                'emotions': emotions,
                'relations': relations,
                'raw_output': vlm_detection.raw_output,
                'caption': vlm_detection.caption,
            })
        
        # Add camera info and header metadata
        result = {
            'timestamp': {
                'sec': msg.header.stamp.sec,
                'nanosec': msg.header.stamp.nanosec,
            },
            'frame_id': msg.header.frame_id,
            'camera_info': {
                'camera_id': msg.cam_info.camera_id,
                'width': msg.cam_info.camera_img_width,
                'height': msg.cam_info.camera_img_height,
                'focal_length': float(msg.cam_info.camera_focal_length),
            },
            'vlm_detections': outputs_payload,
        }
        
        with self.cache['lock']:
            self.cache['raw'] = json.dumps(result, ensure_ascii=False, indent=2)

    def on_status(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception:
            data = {}
        with self.cache['lock']:
            self.cache['status'] = data


def _start_listener(cache: Dict[str, Any]) -> Node:
    rclpy.init(args=None)
    node = _GuiListener(cache)
    thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    thread.start()
    return node


def main() -> None:
    cache: Dict[str, Any] = {
        'frame': None,
        'raw': '',
        'status': {},
        'lock': threading.Lock(),
    }
    listener = _start_listener(cache)

    pause_client = listener.create_client(SetBool, 'vlm_detections/set_pause')
    param_client = listener.create_client(SetParameters, '/vlm_detections_node/set_parameters')
    prompt_mgr_param_client = listener.create_client(SetParameters, '/vlm_prompt_manager_node/set_parameters')

    def _call_pause(should_pause: bool) -> str:
        if not pause_client.wait_for_service(timeout_sec=1.0):
            return 'Pause service unavailable'
        req = SetBool.Request()
        req.data = bool(should_pause)
        future = pause_client.call_async(req)
        rclpy.spin_until_future_complete(listener, future, timeout_sec=2.0)
        if not future.done() or future.result() is None:
            return 'Pause call timeout'
        return future.result().message or 'Pause state updated'

    def _set_fps(fps_val: float | str) -> str:
        try:
            val = float(fps_val)
        except Exception:
            return 'Invalid FPS value'
        if val <= 0:
            return 'fps must be > 0'
        if not prompt_mgr_param_client.wait_for_service(timeout_sec=1.0):
            return 'Prompt manager param service unavailable'
        param = Parameter()
        param.name = 'fps'
        param_value = ParameterValue()
        param_value.type = ParameterType.PARAMETER_DOUBLE
        param_value.double_value = val
        param.value = param_value
        req = SetParameters.Request()
        req.parameters = [param]
        future = prompt_mgr_param_client.call_async(req)
        rclpy.spin_until_future_complete(listener, future, timeout_sec=2.0)
        if not future.done() or future.result() is None:
            return 'Set fps timeout'
        return f'Set fps={val:.2f} Hz'

    def _set_batch_capacity(capacity_val: int | str) -> str:
        try:
            val = int(capacity_val)
        except Exception:
            return 'Invalid batch capacity'
        if val < 1:
            return 'batch_capacity must be >= 1'
        if not prompt_mgr_param_client.wait_for_service(timeout_sec=1.0):
            return 'Prompt manager param service unavailable'
        param = Parameter()
        param.name = 'batch_capacity'
        param_value = ParameterValue()
        param_value.type = ParameterType.PARAMETER_INTEGER
        param_value.integer_value = val
        param.value = param_value
        req = SetParameters.Request()
        req.parameters = [param]
        future = prompt_mgr_param_client.call_async(req)
        rclpy.spin_until_future_complete(listener, future, timeout_sec=2.0)
        if not future.done() or future.result() is None:
            return 'Set batch_capacity timeout'
        mode = 'batch' if val > 1 else 'single image'
        return f'Set batch_capacity={val} (mode={mode})'

    poll_cfg = {
        'interval': max(0.1, float(os.environ.get('ROS_GUI_POLL_INTERVAL', POLL_INTERVAL))),
        'last': 0.0,
        'raw_enabled': True,
        'last_frame': None,
        'last_info': 'Waiting for status...',
        'last_raw': '',
        'last_gestures': 'No data',
        'last_emotions': 'No data',
        'last_relations': 'No data',
    }

    def _update_interval(val: float | str) -> str:
        try:
            parsed = max(0.1, float(val))
        except Exception:
            return 'Invalid poll interval'
        poll_cfg['interval'] = parsed
        return f'Poll interval set to {parsed:.2f}s'

    def _toggle_raw(flag: bool) -> tuple[str, dict[str, object]]:
        poll_cfg['raw_enabled'] = bool(flag)
        suffix = 'enabled' if poll_cfg['raw_enabled'] else 'disabled'
        return f'Raw output {suffix}.', gr.update(visible=bool(flag))

    def _format_status(status: Dict[str, Any]) -> str:
        if not status:
            return 'Waiting for status...'
        parts: list[str] = []

        def append(label: str, value: Any, digits: int | None = None) -> None:
            if value is None:
                return
            if digits is not None and isinstance(value, (int, float)):
                parts.append(f"{label}: {value:.{digits}f}")
            else:
                parts.append(f"{label}: {value}")

        
        append('Δt last(s)', status.get('last_delta_t_s'), 3)
        append('E2E last(ms)', status.get('last_e2e_latency_ms'), 1)
        append('Infer last(ms)', status.get('last_latency_ms'), 1)
        append('Δt avg(s)', status.get('avg_delta_t_s'), 3)
        append('E2E avg(ms)', status.get('avg_e2e_latency_ms'), 1)
        append('Infer avg(ms)', status.get('avg_latency_ms'), 1)
        append('E2E min(ms)', status.get('min_e2e_latency_ms'), 1)
        append('E2E max(ms)', status.get('max_e2e_latency_ms'), 1)
        append('Detections', status.get('detections'))
        append('Threshold', status.get('threshold'), 2)
        append('Inferences', status.get('inference_count'))
        append('Paused', status.get('paused'))
        return ' | '.join(parts) if parts else 'Waiting for status...'

    def _format_gestures(raw_data: str) -> str:
        """Format gestures from JSON data for display."""
        if not raw_data:
            return 'No data'
        try:
            data = json.loads(raw_data)
            vlm_detections = data.get('vlm_detections', [])
            if not vlm_detections:
                return 'No gestures detected'
            
            lines = []
            for idx, detection in enumerate(vlm_detections):
                gestures = detection.get('gestures', [])
                if gestures:
                    lines.append(f"Prompt {idx + 1}:")
                    for g in gestures:
                        lines.append(f"  Person {g['person_uuid']}: {g['gesture']}")
            
            return '\n'.join(lines) if lines else 'No gestures detected'
        except Exception:
            return 'Error parsing gestures'

    def _format_emotions(raw_data: str) -> str:
        """Format emotions from JSON data for display."""
        if not raw_data:
            return 'No data'
        try:
            data = json.loads(raw_data)
            vlm_detections = data.get('vlm_detections', [])
            if not vlm_detections:
                return 'No emotions detected'
            
            lines = []
            for idx, detection in enumerate(vlm_detections):
                emotions = detection.get('emotions', [])
                if emotions:
                    lines.append(f"Prompt {idx + 1}:")
                    for e in emotions:
                        lines.append(f"  Person {e['person_uuid']}: {e['emotion']}")
            
            return '\n'.join(lines) if lines else 'No emotions detected'
        except Exception:
            return 'Error parsing emotions'

    def _format_relations(raw_data: str) -> str:
        """Format relations from JSON data for display."""
        if not raw_data:
            return 'No data'
        try:
            data = json.loads(raw_data)
            vlm_detections = data.get('vlm_detections', [])
            if not vlm_detections:
                return 'No relations detected'
            
            lines = []
            for idx, detection in enumerate(vlm_detections):
                relations = detection.get('relations', [])
                if relations:
                    lines.append(f"Prompt {idx + 1}:")
                    for r in relations:
                        lines.append(f"  Entity {r['subject_uuid']} {r['relation']} Entity {r['object_uuid']}")
            
            return '\n'.join(lines) if lines else 'No relations detected'
        except Exception:
            return 'Error parsing relations'

    def _do_poll():
        now = time.time()
        if (now - poll_cfg['last']) < poll_cfg['interval']:
            return (
                poll_cfg['last_frame'],
                poll_cfg['last_info'],
                poll_cfg['last_raw'],
                poll_cfg['last_gestures'],
                poll_cfg['last_emotions'],
                poll_cfg['last_relations'],
            )

        with cache['lock']:
            frame = cache['frame']
            raw_text = cache['raw']
            status = cache['status']

        info = _format_status(status)
        gestures_text = _format_gestures(raw_text)
        emotions_text = _format_emotions(raw_text)
        relations_text = _format_relations(raw_text)
        
        if not poll_cfg['raw_enabled']:
            raw_text = poll_cfg['last_raw']

        poll_cfg['last'] = now
        poll_cfg['last_frame'] = frame
        poll_cfg['last_info'] = info
        poll_cfg['last_raw'] = raw_text or ''
        poll_cfg['last_gestures'] = gestures_text
        poll_cfg['last_emotions'] = emotions_text
        poll_cfg['last_relations'] = relations_text
        return frame, info, raw_text, gestures_text, emotions_text, relations_text

    base_period = min(0.25, poll_cfg['interval'])

    with gr.Blocks(title='VLM ROS Monitor') as demo:
        gr.Markdown('# VLM ROS Monitor')
        with gr.Row():
            annotated = gr.Image(label='Annotated Stream', interactive=False, type='numpy')
            with gr.Column():
                status_box = gr.Markdown('Waiting for status...')
                feedback_box = gr.Markdown('')
                with gr.Row():
                    pause_btn = gr.Button('Pause', variant='secondary')
                    resume_btn = gr.Button('Resume', variant='secondary')
                gr.Markdown('### Prompt Manager Controls')
                with gr.Row():
                    fps_input = gr.Number(value=1.0, label='fps (Hz)', precision=2, minimum=0.1)
                    apply_fps_btn = gr.Button('Apply FPS')
                with gr.Row():
                    batch_capacity_input = gr.Number(value=1, label='batch_capacity (1=single, >1=batch)', precision=0, minimum=1)
                    apply_batch_btn = gr.Button('Apply Batch Capacity')
                gr.Markdown('### GUI Controls')
                poll_interval_input = gr.Number(value=poll_cfg['interval'], label='Poll interval (s)', precision=2)
                raw_toggle = gr.Checkbox(value=True, label='Show raw output')
        raw_output = gr.Textbox(label='Latest outputs (JSON)', value='', lines=16, interactive=False)
        
        with gr.Row():
            gestures_box = gr.Textbox(label='Person Gestures', value='', lines=6, interactive=False)
            emotions_box = gr.Textbox(label='Person Emotions', value='', lines=6, interactive=False)
        relations_box = gr.Textbox(label='Actor Relations', value='', lines=6, interactive=False)

        try:
            poll_timer = gr.Timer(interval=base_period)
        except TypeError:
            poll_timer = gr.Timer(value=base_period)

        pause_btn.click(fn=lambda: _call_pause(True), outputs=[feedback_box])
        resume_btn.click(fn=lambda: _call_pause(False), outputs=[feedback_box])
        apply_fps_btn.click(fn=_set_fps, inputs=[fps_input], outputs=[feedback_box])
        apply_batch_btn.click(fn=_set_batch_capacity, inputs=[batch_capacity_input], outputs=[feedback_box])
        poll_interval_input.change(fn=_update_interval, inputs=[poll_interval_input], outputs=[feedback_box])
        raw_toggle.change(fn=_toggle_raw, inputs=[raw_toggle], outputs=[feedback_box, raw_output])
        poll_timer.tick(fn=_do_poll, outputs=[annotated, status_box, raw_output, gestures_box, emotions_box, relations_box])

    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', '7861'))
    try:
        demo.launch(server_name=host, server_port=port, quiet=False, show_error=True)
    finally:
        try:
            listener.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
