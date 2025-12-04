import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import logging
from qwen_vl_utils import smart_resize
logger = logging.getLogger(__name__)


class ColorGenerator:
    """A class for generating consistent colors for visualization.

    This class provides methods to generate colors either consistently for all elements
    or based on text content for better visual distinction.

    Args:
        color_type (str): Type of color generation strategy. Can be either "same" for consistent color
            or "text" for text-based color generation.
    """

    def __init__(self, color_type) -> None:
        self.color_type = color_type

        if color_type == "same":
            self.color = tuple((np.random.randint(0, 127, size=3) + 128).tolist())
        elif color_type == "text":
            np.random.seed(3396)
            self.num_colors = 300
            self.colors = np.random.randint(0, 127, size=(self.num_colors, 3)) + 128
        else:
            raise ValueError

    def get_color(self, text):
        """Get a color based on the text content or return a consistent color.

        Args:
            text (str): The text to generate color for.

        Returns:
            tuple: RGB color values as a tuple.

        Raises:
            ValueError: If color_type is not supported.
        """
        if self.color_type == "same":
            return self.color

        if self.color_type == "text":
            text_hash = hash(text)
            index = text_hash % self.num_colors
            color = tuple(self.colors[index])
            return color

        raise ValueError


def visualize(
    image_pil: Image,
    boxes,
    scores,
    labels=None,
    filter_score=-1,
    topN=900,
    font_size=15,
    draw_width: int = 6,
    draw_index: bool = True,
) -> Image:
    """Visualize bounding boxes and labels on an image.

    This function draws bounding boxes and their corresponding labels on the input image.
    It supports filtering by score, limiting the number of boxes, and customizing the
    visualization appearance.

    Args:
        image_pil (PIL.Image): The input image to draw on.
        boxes (List[List[float]]): List of bounding boxes in [x1, y1, x2, y2] format.
        scores (List[float]): Confidence scores for each bounding box.
        labels (List[str], optional): Labels for each bounding box. Defaults to None.
        filter_score (float, optional): Minimum score threshold for visualization. Defaults to -1.
        topN (int, optional): Maximum number of boxes to visualize. Defaults to 900.
        font_size (int, optional): Font size for labels. Defaults to 15.
        draw_width (int, optional): Width of bounding box lines. Defaults to 6.
        draw_index (bool, optional): Whether to draw index numbers for unlabeled boxes. Defaults to True.

    Returns:
        PIL.Image: The image with visualized bounding boxes and labels.
    """
    # Get the bounding boxes and labels from the target dictionary
    font_path = "tools/Tahoma.ttf"
    font = ImageFont.truetype(font_path, font_size)
    # Create a PIL ImageDraw object to draw on the input image
    draw = ImageDraw.Draw(image_pil)
    boxes = boxes[:topN]
    scores = scores[:topN]
    # Draw boxes and masks for each box and label in the target dictionary
    box_idx = 1
    color_generaor = ColorGenerator("text")
    if labels is None:
        labels = [""] * len(boxes)
    for box, score, label in zip(boxes, scores, labels):
        if score < filter_score:
            continue
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # Extract the box coordinates
        x0, y0, x1, y1 = box
        # rescale the box coordinates to the input image size
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        # Draw index when label is empty
        if draw_index and (label == ""):
            text = str(box_idx) + f" {label}"
        else:
            text = str(label)
        max_words_per_line = 10
        words = text.split()
        lines = []
        line = ""
        for word in words:
            if len(line.split()) < max_words_per_line:
                line += word + " "
            else:
                lines.append(line)
                line = word + " "
        lines.append(line)
        text = "\n".join(lines)

        draw.rectangle(
            [x0, y0, x1, y1], outline=color_generaor.get_color(text), width=draw_width
        )

        bbox = draw.textbbox((x0, y0), text, font)
        box_h = bbox[3] - bbox[1]
        box_w = bbox[2] - bbox[0]

        y0_text = y0 - box_h - (draw_width * 2)
        y1_text = y0 + draw_width
        box_idx += 1
        if y0_text < 0:
            y0_text = 0
            y1_text = y0 + 2 * draw_width + box_h
        draw.rectangle(
            [x0, y0_text, bbox[2] + draw_width * 2, y1_text],
            fill=color_generaor.get_color(text),
        )
        draw.text(
            (x0 + draw_width, y0_text),
            str(text),
            fill="black",
            font=font,
        )
    return image_pil


def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (List[float]): First bounding box in [x1, y1, x2, y2] format.
        box2 (List[float]): Second bounding box in [x1, y1, x2, y2] format.

    Returns:
        float: IoU score between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area


def return_maximum_overlap(gt_box, candidate_boxes, min_iou=0.5):
    """Find the best matching box from candidate boxes based on IoU.

    Args:
        gt_box (List[float]): Ground truth bounding box in [x1, y1, x2, y2] format.
        candidate_boxes (List[List[float]]): List of candidate bounding boxes.
        min_iou (float, optional): Minimum IoU threshold for matching. Defaults to 0.5.

    Returns:
        int or None: Index of the best matching box if IoU > min_iou, None otherwise.
    """
    max_iou = 0.0
    best_box = None
    for i, box in enumerate(candidate_boxes):
        iou = compute_iou(gt_box, box)
        if iou >= min_iou and iou > max_iou:
            max_iou = iou
            best_box = i
    return best_box


def find_best_matched_index(group1, group2):
    """Find the best matching indices between two groups of bounding boxes.

    Args:
        group1 (List[List[float]]): First group of bounding boxes.
        group2 (List[List[float]]): Second group of bounding boxes.

    Returns:
        List[int]: List of indices (1-based) indicating the best matches from group2 for each box in group1.
                   Returns 0 when no match is found or group2 is empty.
    """
    labels = []
    if not group2:
        # No candidate boxes
        return [0 for _ in group1]
    for box in group1:
        best_box = return_maximum_overlap(box, group2)
        if best_box is None:
            labels.append(0)
        else:
            labels.append(best_box + 1)
    return labels


def convert_boxes_from_absolute_to_qwen25_format(gt_boxes, ori_width, ori_height):
    """Convert bounding boxes from absolute coordinates to Qwen-25 format.

    This function resizes bounding boxes according to Qwen-25's requirements while
    maintaining aspect ratio and pixel constraints.

    Args:
        gt_boxes (List[List[float]]): List of bounding boxes in absolute coordinates.
        ori_width (int): Original image width.
        ori_height (int): Original image height.

    Returns:
        List[List[int]]: Resized bounding boxes in Qwen-25 format.
    """
    resized_height, resized_width = smart_resize(
        ori_height,
        ori_width,
        28,
        min_pixels=16 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    resized_gt_boxes = []
    for box in gt_boxes:
        # resize the box
        x0, y0, x1, y1 = box
        x0 = int(x0 / ori_width * resized_width)
        x1 = int(x1 / ori_width * resized_width)
        y0 = int(y0 / ori_height * resized_height)
        y1 = int(y1 / ori_height * resized_height)

        x0 = max(0, min(x0, resized_width - 1))
        y0 = max(0, min(y0, resized_height - 1))
        x1 = max(0, min(x1, resized_width - 1))
        y1 = max(0, min(y1, resized_height - 1))
        resized_gt_boxes.append([x0, y0, x1, y1])
    return resized_gt_boxes


def parse_json(json_output):
    """Parse JSON string containing coordinate arrays.

    Args:
        json_output (str): JSON string containing coordinate arrays.

    Returns:
        List[List[float]]: List of parsed coordinate arrays.
    """
    pattern = r"\[([0-9\.]+(?:, ?[0-9\.]+)*)\]"

    matches = re.findall(pattern, json_output)
    coordinates = [
        [float(num) if "." in num else int(num) for num in match.split(",")]
        for match in matches
    ]

    return coordinates


def postprocess_and_vis_inference_out(
    target_image,
    answer,
    proposed_box,
    gdino_boxes,
    font_size,
    draw_width,
    input_height,
    input_width,
):
    """Post-process inference results and create visualization.

    This function processes the model output, matches boxes with Grounding DINO results,
    and creates visualization images. If no GDINO boxes are provided, all VLM boxes
    are visualized with index labels.
    """
    if proposed_box is None:
        return target_image, target_image

    w, h = target_image.size
    json_output = parse_json(answer)
    final_boxes = []
    input_height = input_height.item()
    input_width = input_width.item()
    for box in json_output:
        x0, y0, x1, y1 = box
        x0 = x0 / input_width * w
        y0 = y0 / input_height * h
        x1 = x1 / input_width * w
        y1 = y1 / input_height * h

        final_boxes.append([x0, y0, x1, y1])

    # Decide labels depending on availability of GDINO boxes
    ref_labels = None
    if gdino_boxes and len(gdino_boxes) > 0:
        ref_labels = find_best_matched_index(final_boxes, gdino_boxes)
        logger.debug("ref_labels %s", ref_labels)

    ref_vis_result = visualize(
        target_image.copy(),
        final_boxes,
        np.ones(len(final_boxes)),
        labels=ref_labels,  # None => draw indices; list => draw matched indices
        font_size=font_size,
        draw_width=draw_width,
    )

    dinox_vis_result = visualize(
        target_image.copy(),
        gdino_boxes,
        np.ones(len(gdino_boxes)),
        font_size=font_size,
        draw_width=draw_width,
    )
    return ref_vis_result, dinox_vis_result
