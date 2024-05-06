import numpy as np
import os

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

# specify directories
#dir_a is the ground truth labels
#dir_b is the predicted labels such as yolov8 result output
#output_dir is the directory where the modified labels will be saved
dir_a = 'datasetsWithLabel/inference/labels/'
dir_b = 'runs/detect/yolov8_look/labels/'
output_dir = 'modified_labels/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# process all text files
for filename in os.listdir(dir_a):
    if filename.endswith('.txt') and os.path.exists(os.path.join(dir_b, filename)):
        # load the bounding boxes from file A
        with open(os.path.join(dir_a, filename), 'r') as f:
            lines_a = f.readlines()
        boxes_a = {}
        for line in lines_a:
            parts = line.strip().split()
            id = int(parts[-1])
            cx, cy, w, h = map(float, parts[1:5])
            boxes_a[id] = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]  # convert to xmin, ymin, xmax, ymax

        # load the bounding boxes from file B
        with open(os.path.join(dir_b, filename), 'r') as f:
            lines_b = f.readlines()
        boxes_b = [list(map(float, line.strip().split()[1:5])) for line in lines_b]

        # calculate IoU and assign ID if IoU > 0.55
        for id, box_a in boxes_a.items():
            for i in range(len(boxes_b)):
                box_b = boxes_b[i]
                box_b = [box_b[0]-box_b[2]/2, box_b[1]-box_b[3]/2, box_b[0]+box_b[2]/2, box_b[1]+box_b[3]/2]  # convert to xmin, ymin, xmax, ymax
                iou = bb_intersection_over_union(box_a, box_b)
                if iou > 0.55:
                    lines_b[i] = lines_b[i].strip() + ' ' + str(id) + '\n'

        # write the updated bounding boxes to the output file
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.writelines(lines_b)