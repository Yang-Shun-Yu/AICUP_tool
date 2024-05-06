from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image,deprocess_image, preprocess_image
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':


    model = torch.hub.load('b06b01073/veri776-pretrain', 'resnext101_ibn_a', fine_tuned=True).backbone # 將 fine_tuned 設為 True 會 load fine-tuned 後的 model

    model = model.to('cpu')
    model.eval() # 別忘了設成 eval model，避免 BatchNorm 追蹤 running mean

    target_layers = [model.layer4[-1]] # 'resnet', 'resnext', 'seresnet'

    # crop image is too small, need to resize large size
    transform_large_size = torchvision.transforms.Compose([
        torchvision.transforms.Resize((448, 448))
    ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # dir_path is the path of the folder containing the crops images that yolov8 created
    # save_path is the path of the folder that you want to save the cropped images with cam

    dir_path = "../AICUP/runs/detect/yolov8n_infer/crops/car"
    save_path = "test_cam"
    # cur = 0
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for filename in os.listdir(dir_path):
        if filename.endswith(".jpg"):
            # if cur > 5:
            #     break
            img = Image.open(os.path.join(dir_path, filename)).convert('RGB')
            img = transform_large_size(img)
            img = np.array(img)
            img_float_np = np.float32(img) / 255
            input_tensor = transform(img).unsqueeze(0).to(device)

            cam = EigenCAM(model=model, target_layers=target_layers)
            # cam = GradCAM(model=model, target_layers=target_layers)


            grayscale_cam = cam(input_tensor=input_tensor)
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(img_float_np, grayscale_cam, use_rgb=True)

            filename_without_ext = os.path.splitext(filename)[0]

            # cv2.imwrite(f'{save_path}/{filename_without_ext}_cam.jpg', cam_image)
            # Convert the grayscale cam to 8-bit for thresholding
            # crop the image from the bounding box which is from heatmap

            grayscale_cam = (grayscale_cam * 255).astype(np.uint8)
            _, binary_mask = cv2.threshold(grayscale_cam, 64, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            crop_img = img[y:y+h, x:x+w]
            original_image = cv2.imread(os.path.join(dir_path, filename))
            original_size = (original_image.shape[1], original_image.shape[0]) # Width, Height
            img_resized = cv2.resize(img, original_size)
            crop_img_resized = cv2.resize(crop_img, original_size)

            # cv2.imwrite(f'{save_path}/{filename_without_ext}_cam_with_bbox.jpg', img_resized)

            
            cv2.imwrite(f'{save_path}/{filename_without_ext}.jpg', crop_img_resized)  
            
            # cur += 1

        

