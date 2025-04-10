import xml.etree.ElementTree as ET
import os
import argparse
from collections import defaultdict
from pathlib import Path
import logging
import math

# --- Configuration ---
KEYPOINTS_ORDER = [
    "Corner_Top_Left", "Line_Top", "Corner_Top_Right", "18_Top_Left",
    "18_Bottom_Left", "Penalty_Spot_Left", "Circle_Top", "Circle_Bottom",
    "Penalty_Spot_Right", "18_Top_Right", "18_Bottom_Right", "Corner_Bottom_Left",
    "Line_Bottom", "Corner_Bottom_Right"
]
NUM_KEYPOINTS = len(KEYPOINTS_ORDER)
LABEL_TO_INDEX = {label: i for i, label in enumerate(KEYPOINTS_ORDER)}
BBOX_PADDING = 15
MIN_BBOX_SIZE_FRACTION = 0.02

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_bounding_box(points_coords, img_w, img_h):
    if not points_coords or img_w <= 0 or img_h <= 0: return None, None, None, None
    valid_points = [(x, y) for x, y in points_coords if 0 <= x < img_w and 0 <= y < img_h]
    if not valid_points:
        valid_points = points_coords
    min_x = max(0.0, min(p[0] for p in valid_points))
    min_y = max(0.0, min(p[1] for p in valid_points))
    max_x = min(float(img_w), max(p[0] for p in valid_points))
    max_y = min(float(img_h), max(p[1] for p in valid_points))

    if len(valid_points) == 1:
        min_bbox_w, min_bbox_h = img_w * MIN_BBOX_SIZE_FRACTION, img_h * MIN_BBOX_SIZE_FRACTION
        center_x, center_y = min_x, min_y
        box_w = max(2.0, min_bbox_w + 2 * BBOX_PADDING)
        box_h = max(2.0, min_bbox_h + 2 * BBOX_PADDING)
    else:
        box_w = max(2.0, (max_x - min_x) + 2 * BBOX_PADDING)
        box_h = max(2.0, (max_y - min_y) + 2 * BBOX_PADDING)
        center_x = min_x + (max_x - min_x) / 2
        center_y = min_y + (max_y - min_y) / 2

    half_w, half_h = box_w / 2, box_h / 2
    center_x = max(half_w, min(img_w - half_w, center_x))
    center_y = max(half_h, min(img_h - half_h, center_y))
    box_w = min(img_w, 2 * min(center_x, img_w - center_x))
    box_h = min(img_h, 2 * min(center_y, img_h - center_y))
    return center_x, center_y, max(2.0, box_w), max(2.0, box_h)

def normalize_value(value, dimension):
    if dimension is None or dimension <= 0: return None
    normalized = float(value) / float(dimension)
    clamped = max(0.0, min(1.0, normalized))
    return clamped

def convert_cvat_xml_to_yolo_pose(xml_path, output_dir):
    logging.info(f"Loading XML annotations from: {xml_path}")
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        logging.error(f"Error loading XML: {e}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_path.resolve()}")

    annotations_by_image = defaultdict(lambda: {'points': [], 'width': None, 'height': None})

    for image_tag in root.findall('.//image'):
        image_name = image_tag.get('name')
        img_w = int(image_tag.get('width'))
        img_h = int(image_tag.get('height'))

        if not image_name or img_w <= 0 or img_h <= 0:
            continue

        annotations_by_image[image_name]['width'] = img_w
        annotations_by_image[image_name]['height'] = img_h

        for points_tag in image_tag.findall('points'):
            label = points_tag.get('label')
            points_str = points_tag.get('points')
            if not label or label not in LABEL_TO_INDEX or not points_str:
                continue
            try:
                x_str, y_str = points_str.split(';')[0].split(',')
                x, y = float(x_str), float(y_str)
                x = max(0.0, min(x, img_w - 1))
                y = max(0.0, min(y, img_h - 1))
                annotations_by_image[image_name]['points'].append({'label': label, 'x': x, 'y': y})
            except Exception as e:
                logging.warning(f"Failed to parse point: {e}")

    processed_files = 0
    for image_name, img_data in annotations_by_image.items():
        img_w, img_h = img_data['width'], img_data['height']
        points_list = img_data['points']
        if not points_list:
            continue

        yolo_keypoints = [[0.0, 0.0, 0] for _ in range(NUM_KEYPOINTS)]
        visible_points_coords = []
        for point in points_list:
            kpt_index = LABEL_TO_INDEX[point['label']]
            norm_x = normalize_value(point['x'], img_w)
            norm_y = normalize_value(point['y'], img_h)
            yolo_keypoints[kpt_index] = [norm_x, norm_y, 2]
            visible_points_coords.append((point['x'], point['y']))

        center_x, center_y, box_w, box_h = calculate_bounding_box(visible_points_coords, img_w, img_h)
        norm_cx = normalize_value(center_x, img_w)
        norm_cy = normalize_value(center_y, img_h)
        norm_w = normalize_value(box_w, img_w)
        norm_h = normalize_value(box_h, img_h)

        yolo_parts = ["0", f"{norm_cx:.6f}", f"{norm_cy:.6f}", f"{norm_w:.6f}", f"{norm_h:.6f}"]
        for kpt in yolo_keypoints:
            yolo_parts.extend([f"{kpt[0]:.6f}", f"{kpt[1]:.6f}", f"{kpt[2]:.0f}"])
        yolo_line = " ".join(yolo_parts)

        base_filename = Path(image_name).stem
        output_txt_path = output_path / f"{base_filename}.txt"

        with open(output_txt_path, 'w') as f:
            f.write(yolo_line + '\n')
        processed_files += 1

    logging.info(f"Processed {processed_files} label files.")

if __name__ == "__main__":
    xml_path = r"C:\Users\abdul\Desktop\YoloV8_New_Attempt\annotations.xml"
    output_dir = r"C:\Users\abdul\Desktop\YoloV8_New_Attempt\output_labels"

    convert_cvat_xml_to_yolo_pose(xml_path, output_dir)
