import numpy as np

from matplotlib import pyplot as plt

from PIL import Image

import cv2

def display_image(path_or_array, size=(10, 10)):
    if isinstance(path_or_array, str):
        image = np.asarray(Image.open(open(path_or_array, 'rb')).convert("RGB"))
    else:
        image = path_or_array
    
    plt.figure(figsize=size)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
def visualize_image(image_path,bboxs,labels):
    image = cv2.imread(image_path)
    plt.imshow(image)
    # draw = ImageDraw.Draw(image)
    ret_image = image.copy()
    for i in range(len(bboxs)):
        y1, x1, y2, x2 = int(np.floor(bboxs[i][0])), int(np.floor(bboxs[i][1])), int(np.ceil(bboxs[i][2])), int(np.ceil(bboxs[i][3]))
        ret_image = cv2.rectangle(ret_image, (x1,y1), (x2,y2), (255,255,0), 2)
        ret_image = cv2.putText(ret_image, labels[i], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255,255,0), 2, cv2.LINE_AA)
    plt.imshow(ret_image[:,:,::-1])
    plt.show()
    