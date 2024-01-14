from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
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
    parser.add_argument("--model", help="vit_h or vit_l" ,required=False, default="vit_h", type=str)
    parser.add_argument("--start_dir", required=False, default=0, type=int)
    parser.add_argument("--start_img", required=False, default=0, type=int)
    args = parser.parse_args()
    labels_file = args.labels_file
    labels_dir = args.labels_dir
    images_dir = args.images_dir
    img_type = args.img_type
    model_type = args.model
    start_dir = args.start_dir
    start_img = args.start_img

    print("CUDA is available: ", torch.cuda.is_available())

    sam = sam_model_registry[model_type](checkpoint=("/cluster/project/infk/cvg/heinj/students/kbirgi/SA_BscThesis/models/" + ("sam_vit_h_4b8939.pth" if (model_type == "vit_h") else "sam_vit_l_0b3195.pth")))

    if torch.cuda.is_available():
        sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)
    predictor = SamPredictor(sam)
    
    f = open(labels_file, "r")
    labels_dict = json.load(f)
    f.close()

    for i,camera in enumerate(labels_dict):
        camera_dict = labels_dict[camera]
        for j,imageNr in enumerate(camera_dict):
            print("Camera: " + str(i) + "/" + str(len(labels_dict)) + " | Image: " + str(j) + "/" + str(len(camera_dict)), flush=True)
            img_dict = camera_dict[imageNr]['img']
            img_path = images_dir + "/" + img_dict['file_name']

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

            masks_dict = {}
            #all masks mode
            masks_path = labels_dir + "/" + str.replace(img_dict['file_name'], "." + img_type, ".txt")
            all_masks = mask_generator.generate(image)
            masks_dict['all_masks'] = all_masks
            
            #TODO: take samples from yolact
            

            #multimask mode
            #multi_masks = 

            f = open(masks_path, "w")
            json.dump(f, masks_dict)
            f.close()

    print("OK!")