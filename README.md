# load_data.py
Load and Filter Object Detections 
This code provides functions to load object detections from text files and filter them by camera ID.

## Code Explanation

The code consists of three main functions:

1. `load_all_detections(label_folder)`: This function loads all detection files from a specified folder. Each file's name is split to extract the day, start time, end time, camera ID, and frame number, which are used as keys in the returned dictionary. The values in the dictionary are the detections loaded from each file.

2. `load_detections(label_path)`: This function is used by `load_all_detections()` to load detections from a single file. Each line in the file is split and mapped to a dictionary containing the class ID, center coordinates, width, height, and confidence of the detected object.

3. `get_detection_by_cameara_id(detections, class_id)`: This function filters the detections by a specified camera ID. It returns a new dictionary containing only the detections that match the specified camera ID.

The code then demonstrates how to use these functions. It loads all detections from a folder, retrieves the detections for a specific key, and filters all detections by a specific camera ID.

## Code Usage

```python
import os

def load_all_detections(label_folder):
    all_detections = {}
    for filename in os.listdir(label_folder):
        if filename.endswith('.txt'):
            # Remove the '.txt' extension and split the filename
            day, start_time, end_time, camera_id, frame_number = filename[:-4].split('_')
            
            filepath = os.path.join(label_folder, filename)
            detections = load_detections(filepath)

            # Use a tuple of day, camera id, start time, end time, frame number as the key
            all_detections[(day, start_time, end_time, camera_id, frame_number)] = detections
    return all_detections

def load_detections(label_path):
    detections = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            class_id, x_center, y_center, width, height, confidence = map(float, line.strip().split())
            detections.append({
                'class_id': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
                'confidence': confidence
            })
    return detections

def get_detection_by_cameara_id(detections, class_id):
    camera_id_detections = {}
    for key, value in detections.items():
        day, start_time, end_time, camera_id ,frame_number = key
        if camera_id == class_id:
            camera_id_detections[key] = value
    return camera_id_detections

if __name__ == '__main__':
    # Path to the folder containing the labels
    label_folder = 'detect/yolov8n_infer/labels' 
    all_detections = load_all_detections(label_folder)
    detections = all_detections[('1016',  '150000', '151900','0', '00059')]
    
    # Get all detections of class 0
    class_0_detections = get_detection_by_cameara_id(all_detections, '0')
    print(len(class_0_detections))
    print(len(detections))
```

In this example, the label_folder is set to 'detect/yolov8n_infer/labels'. Adjust this path to match the location of your label files. The example retrieves detections for a specific key and filters all detections by a specific camera ID ('0' in this case).

# YOLO Object Detection with Ultralytics(yolo_inference.py)

This code demonstrates how to use the Ultralytics YOLO implementation for object detection.

## Code Explanation

The code loads a pre-trained YOLO model from a weights file and uses it to predict objects in images from a specified directory. The predictions are saved as text files and cropped images of detected objects.

## Code Usage

```python
from ultralytics import YOLO
from multiprocessing import freeze_support

if __name__ == '__main__':
    # Load a model
    freeze_support()
    # path to the weights file
    model_weights_path = 'runs/detect/yolov8n4/weights/best.pt'
    modelv8n = YOLO(model_weights_path)

    image_dir = 'datasets/inference/images'
    save_dir = "yolov8n_infer"

    results = modelv8n.predict(source=image_dir, name=save_dir, hide_labels=True, save_txt=True, save_conf=True, save_crop=True, save=True)
```
In this example, the model_weights_path is set to 'runs/detect/yolov8n4/weights/best.pt' and the image_dir is set to 'datasets/inference/images'. Adjust these paths to match the location of your weights file and image directory. The save_dir is set to "yolov8n_infer", adjust this to your preferred save directory.

# Add_id_to_labels.py
Add the car ID to the labels that are generated from the YOLOv8 inference result.
