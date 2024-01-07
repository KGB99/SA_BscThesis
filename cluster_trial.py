from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--labels_file", required=True, type=str)
    parser.add_argument("--labels_dir", required=False, default="./output", type=str)
    parser.add_argument("--images_dir", required=False, type=str)
    args = parser.parse_args()
    labels_file = args.labels_file
    labels_dir = args.labels_dir
    images_dir = args.images_dir

    print("CUDA is available: ", torch.cuda.is_available())

    sam = sam_model_registry["vit_h"](checkpoint="/cluster/project/infk/cvg/heinj/students/kbirgi/SA_BscThesis/sam_vit_h_4b8939.pth")
    if torch.cuda.is_available():
        sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    f = open(labels_file, "r")
    labels_dict = json.load(f)
    f.close()

    for camera in labels_dict:
        camera_dict = labels_dict[camera]
        for imageNr in camera_dict:
            img_dict = camera_dict[imageNr]['img']
            img_path = images_dir + "/" + img_dict['file_name']
            processed_path = labels_dir + "/" + img_dict['file_name']
            
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = mask_generator.generate(image)
            print(masks)
            exit()

    print("OK!")