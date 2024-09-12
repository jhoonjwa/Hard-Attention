

# Function to map image coordinates to feature map coordinates
def map_to_feature_map(bbox=[50, 50, 150, 150], patch_size=14, image_size=336):
    x1, y1, x2, y2 = bbox
    feature_map_size = image_size // patch_size  # 24x24
    
    # Mapping coordinates
    fm_x1 = int(x1 / image_size * feature_map_size)
    fm_y1 = int(y1 / image_size * feature_map_size)
    fm_x2 = int(x2 / image_size * feature_map_size)
    fm_y2 = int(y2 / image_size * feature_map_size)
    
    return fm_x1, fm_y1, fm_x2, fm_y2

fm_bbox = map_to_feature_map()

# Extracting the submatrix based on the feature map bounding box
def extract_attention(attention_weights, fm_bbox):
    fm_x1, fm_y1, fm_x2, fm_y2 = fm_bbox
    return attention_weights[fm_y1:fm_y2, fm_x1:fm_x2]

attention_submatrix = extract_attention(attention_weights, fm_bbox)
