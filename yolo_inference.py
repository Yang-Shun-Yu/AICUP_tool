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

    results = modelv8n.predict(source=image_dir,name=save_dir,hide_labels=True,save_txt=True,save_conf=True ,save_crop=True,save = True)

 



