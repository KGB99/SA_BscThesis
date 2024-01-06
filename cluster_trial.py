from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

print("hello world")

sam = sam_model_registry["vit_h"](checkpoint="/cluster/project/infk/cvg/heinj/students/kbirgi/SA_BscThesis/sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
exit()
# need to create a list to iterate through for the mask generations
for i in range(0,10):
    image = None
    masks = mask_generator.generate(image)
    
    #store them somehow
