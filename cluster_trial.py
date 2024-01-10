from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
import cv2
import argparse
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--labels_file", required=True, type=str)
    parser.add_argument("--labels_dir", required=False, default="./output", type=str)
    parser.add_argument("--images_dir", required=False, type=str)
    parser.add_argument("--img_type", required=False, default="png", type=str)
    args = parser.parse_args()
    labels_file = args.labels_file
    labels_dir = args.labels_dir
    images_dir = args.images_dir
    img_type = args.img_type

    print("CUDA is available: ", torch.cuda.is_available())

    sam = sam_model_registry["vit_h"](checkpoint="/cluster/project/infk/cvg/heinj/students/kbirgi/SA_BscThesis/sam_vit_h_4b8939.pth")
    if torch.cuda.is_available():
        sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    f = open(labels_file, "r")
    labels_dict = json.load(f)
    f.close()

    for i,camera in enumerate(labels_dict):
        camera_dict = labels_dict[camera]
        for j,imageNr in enumerate(camera_dict):
            print("Camera: " + str(i) + "/" + str(len(labels_dict)) + " | Image: " + str(j) + "/" + str(len(camera_dict)), flush=True)
            img_dict = camera_dict[imageNr]['img']
            img_path = images_dir + "/" + img_dict['file_name']
            processed_path = labels_dir + "/" + str.replace(img_dict['file_name'], "." + img_type, ".txt")

            temp_img_path = img_dict['file_name']
            temp_array = []
            while (os.path.split(temp_img_path)[1] != ''):
                #print(temp_img_path)
                temp_path_split = os.path.split(temp_img_path)
                temp_array.append(temp_path_split[1])
                temp_img_path = temp_path_split[0]
            temp_array.reverse()
            temp_path = labels_dir
            for i in range(len(temp_array) - 1):
                temp_path = temp_path + "/" + temp_array[i]
                if (not os.path.exists(temp_path)):
                    os.mkdir(temp_path)
            
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = mask_generator.generate(image)
            f = open(processed_path, "w")
            f.write(str(masks))
            f.close()

    print("OK!")