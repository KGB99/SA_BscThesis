from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import torch
import cv2
import argparse
import os
import json
import pickle

def prep_path(results_path, img_path):
    temp_img_path = img_path
    temp_array = []
    while (os.path.split(temp_img_path)[1] != ''):
        temp_path_split = os.path.split(temp_img_path)
        temp_array.append(temp_path_split[1])
        temp_img_path = temp_path_split[0]
    temp_array.reverse()
    temp_path = results_path
    for i in range(len(temp_array) - 1):
        temp_path = temp_path + "/" + temp_array[i]
        if (not os.path.exists(temp_path)):
            os.mkdir(temp_path)
    return      

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--labels_file", required=True, type=str)
    parser.add_argument("--labels_dir", required=False, default="./output", type=str)
    parser.add_argument("--images_dir", required=False, type=str)
    parser.add_argument("--img_type", required=False, default="png", type=str)
    parser.add_argument("--model", help="vit_h or vit_l" ,required=False, default="vit_h", type=str)
    parser.add_argument("--single_file_res", help="bool for saving all masks to single file, similar to labels_file", required=False, default=False, type=bool)
    parser.add_argument("--output", required=False, help="name of output file, should end with .json", default="output_sa_masks.json", type=str)
    args = parser.parse_args()
    labels_file = args.labels_file
    labels_dir = args.labels_dir
    images_dir = args.images_dir
    img_type = args.img_type
    model_type = args.model
    single_file = args.single_file_res

    print("CUDA is available: ", torch.cuda.is_available())

    sam = sam_model_registry[model_type](checkpoint=("/cluster/home/kbirgi/SA_BscThesis/models/" + ("sam_vit_h_4b8939.pth" if (model_type == "vit_h") else "sam_vit_l_0b3195.pth")))
    
    if torch.cuda.is_available():
        sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)
    #predictor = SamPredictor(sam)
    
    f = open(labels_file, "r")
    labels_dict = json.load(f)
    f.close()
    #chosen_cameras = ['001004', '016004']
    #chosen_cameras = ['016004']
    #chosen_cameras = ["001000", "001001", "001002", "001003", "001004", "001005", "001006", "016000", "016001", "016002", "016003", "016004", "016005", "016006"]
    #chosen_cameras = ["001000", "001001", "001002", "001003", "001005", "001006"]
    chosen_cameras = ["001000", "001001"]
    for i,camera in enumerate(labels_dict):
        if not (camera in chosen_cameras):
            continue

        camera_dict = labels_dict[camera]
        for j,imageNr in enumerate(camera_dict):
            print("Camera: " + str(i) + "/" + str(len(labels_dict)) + " | Image: " + str(j) + "/" + str(len(camera_dict)), flush=True)
            img_dict = camera_dict[imageNr]['img']
            img_path = images_dir + "/" + img_dict['file_name']

            prep_path(labels_dir ,img_dict['file_name'])

            # read the image for segment anything
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            masks_dict = {}
            #all masks mode
            # im using pickle files cause of the deep nested numpy ndarrays created from segment anything in the object dicts
            masks_path = labels_dir + "/" + str.replace(img_dict['file_name'], "." + img_type, ".pkl")
            all_masks = mask_generator.generate(image)
            masks_dict['all_masks'] = all_masks
            #TODO: take samples from yolact

            #multimask mode
            #multi_masks = 
            if not single_file:
                with open(masks_path, "wb") as f:
                    pickle.dump(masks_dict, f)
                #np.savetext(masks_path, all_masks, )

    print("OK!")