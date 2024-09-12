import json
import torch

from tqdm import tqdm
from shapely.geometry import box, Polygon
from shapely.ops import unary_union


def compute_union_area(boxes):
    polygons = [box(*bbox) for bbox in boxes]
    combined = unary_union(polygons)
    return combined.area if isinstance(combined, Polygon) else combined.area
    
    
def compute_total_intersection_area(gt_boxes, proposal_boxes):
    # Combine all GT boxes into a single union
    from shapely.geometry import box
    gt_combined = unary_union([box(*bbox) for bbox in gt_boxes])
    
    # Combine all proposal boxes into a single union
    prop_combined = unary_union([box(*bbox) for bbox in proposal_boxes])
    
    # Compute the intersection of the two combined geometries
    if gt_combined.intersects(prop_combined):
        return gt_combined.intersection(prop_combined).area
    else:
        return 0


def calculate_aggregated_iou(gt_boxes, proposal_boxes):
    """
    Calculates the IoU between the entire GT and proposal bounding boxes as a whole.
    
    Args:
    - gt_boxes (list of lists): List of ground truth bounding boxes, where each box is represented as a list [x1, y1, x2, y2].
    - proposal_boxes (list of lists): List of proposal bounding boxes, where each box is represented as a list [x1, y1, x2, y2].
    
    Returns:
    - float: The aggregated IoU value.
    """
    
    # Calculate union areas
    total_gt_area = compute_union_area(gt_boxes)
    #total_proposal_area = compute_union_area(proposal_boxes)
    total_union_area = compute_union_area(gt_boxes + proposal_boxes)
    total_intersection_area = compute_total_intersection_area(gt_boxes, proposal_boxes)
    #print(total_gt_area, total_proposal_area, total_intersection_area)

    aggregated_iou = total_intersection_area / total_union_area if total_union_area > 0 else 0

    gt_interpolation = total_intersection_area / total_gt_area
    return aggregated_iou, gt_interpolation



def convert(boxes) -> list:
    """
    Convert bounding boxes from cxcywh format to xyxy format.

    Parameters:
    boxes (torch.Tensor): A tensor of shape (N, 4) where each row is (cx, cy, w, h).

    Returns:
    list: A list of lists, where each inner list contains (x1, y1, x2, y2) coordinates.
    """
    cx, cy, w, h = boxes[0], boxes[1], boxes[2], boxes[3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    # Convert the resulting tensor to a list of lists
    return [x1, y1, x2, y2]



def get_topk_regions(dataset, max_bbox=3, threshold=0.4):
    '''
    recieves whole dataset, returns topk regions by the ordre of logits
    '''
    for data in tqdm(dataset):
        total_bbox = 0
        all_bbox_logits = []
        for boxes in data['new_dino_bbox']:
            total_bbox += len(boxes)
        sorted_boxes = dict()
        confident_phrase = []
        for p_idx, phrase in enumerate(data['new_dino_phrase']):
            bbox_and_logits = sorted([(box, logit) for box, logit in zip(data['new_dino_bbox'][p_idx], data['new_dino_logits'][p_idx])], key=lambda x: x[1], reverse=True)
            sorted_boxes[phrase[0]] = bbox_and_logits
            all_bbox_logits += bbox_and_logits
        bbox_to_keep = [sorted_boxes[phrase[0]][0] for phrase in data['new_dino_phrase'] if len(data['new_dino_phrase']) > 0]
        #print(len(sorted_boxes[phrase[0]]))
        confident_phrase = [phrase for phrase in data['new_dino_phrase'] if sorted_boxes[phrase[0]][0][1] > threshold]
        #confident_phrase = []
        #max_bbox -= len(confident_phrase)
        if len(bbox_to_keep) >= max_bbox:
            data[f'topk_{max_bbox}_confident'] = [data[0] for data in bbox_to_keep if len(data) > 1]
        else:
            start_idx = 1
            while True:
                candidates = []
                for p_idx, phrase in enumerate(data['new_dino_phrase']):
                    if len(sorted_boxes[phrase[0]])  > start_idx and phrase[0] not in confident_phrase and len(phrase[0]) != 0:
                        #print(len(sorted_boxes[phrase[0]]), start_idx)
                        candidates.append(sorted_boxes[phrase[0]][start_idx])
                candidates = sorted(candidates, key=lambda entry: entry[1], reverse=False)
                #print(candidates)
                while len(bbox_to_keep) < max_bbox:
                    bbox_to_keep.append(candidates)
                if len(bbox_to_keep) == max_bbox:
                    break
                start_idx += 1
            data[f'topk_{max_bbox}_confident'] = [data[0] for data in bbox_to_keep if len(data) > 1]
        data[f'topk_{max_bbox}_confident'] = [convert(item[0]) if len(item) == 2 else convert(item) for item in data[f'topk_{max_bbox}_confident']]
    return dataset

