from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
import json

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--labels_file", required=True, type=str)
    parser.add_argument("--labels_dir", required=True, type=str)
    parser.add_argument("--images_dir", required=True, type=str)
    parser.add_argument("--images_type", required=False, default="png", type=str)
    args = parser.parse_args()
    labels_file = args.labels_file
    labels_dir = args.labels_dir
    images_dir = args.images_dir
    images_type = args.images_type
    
    f = open(labels_file, "r")
    labels_dict = json.load(f)
    f.close()

    for i,camera in enumerate(labels_dict):
        camera_dict = labels_dict[camera]
        for j,imageNr in enumerate(camera_dict):
            print("Camera: " + str(i) + "/" + str(len(labels_dict)) + " | Image: " + str(j) + "/" + str(len(camera_dict)), flush=True)
            image_dict = camera_dict[imageNr]["img"]
            image_path = images_dir + "/" + image_dict["file_name"]

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            masks_path = labels_dir + "/" + str.replace(image_dict["file_name"], "." + images_type, ".txt")
            f = open(masks_path, "r")
            masks = f.read()
            f.close()

            plt.figure(figsize=(20,20))
            plt.imshow(image)
            show_anns(masks)
            plt.axis('off')
            plot_path = str.replace(masks_path, ".txt", ".png")
            plt.savefig(plot_path)
            plt.close()
    
    print("OK!")