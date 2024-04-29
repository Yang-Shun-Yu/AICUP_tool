import os

def load_all_detections(label_folder):
    all_detections = {}
    for filename in os.listdir(label_folder):
        if filename.endswith('.txt'):
            # Remove the '.txt' extension and split the filename
            day, start_time, end_time,camera_id, frame_number = filename[:-4].split('_')
            
            filepath = os.path.join(label_folder, filename)
            detections = load_detections(filepath)

            # Use a tuple of day, camera id, start time, end time, frame number as the key
            all_detections[(day, start_time, end_time,camera_id, frame_number)] = detections
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