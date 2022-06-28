from tqdm import tqdm
from glob import glob, iglob

from matplotlib import pyplot as plt

import os

from utils.xml_generate import inference2xml
from utils.VILD import VILD
from utils.image_manager import display_image,visualize_image

# Global matplotlib settings
SMALL_SIZE = 16#10
MEDIUM_SIZE = 18#12
BIGGER_SIZE = 20#14

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# Parameters for drawing figure.
display_input_size = (10, 10)
overall_fig_size = (18, 24)

line_thickness = 2
fig_size_w = 35
# fig_size_h = min(max(5, int(len(category_names) / 2.5) ), 10)
mask_color =   'red'
alpha = 0.5


numbered_categories = [{'name': str(idx), 'id': idx,} for idx in range(50)]
numbered_category_indices = {cat['id']: cat for cat in numbered_categories}

def main():
    ROOT = os.getcwd()
    vild_model = VILD()
    
    # image_path = '/content/sampling/Potatochip/img00001.jpg' 
    # display_image(image_path, size=display_input_size)
    # res_label, res_bbox, height, width = vild_model._detect('/content/sampling/Potatochip/img00001.jpg')
    # print(res_label)
    # print(res_bbox)
    # visualize_image('/content/sampling/Potatochip/img00001.jpg',res_bbox[3:4],res_label[3:4])
    # inference2xml('/content/','/content/sampling/Potatochip/img00001.jpg',res_label, res_bbox, height, width,res_bbox,res_label)
    
    # for loop dir
    for folder in os.listdir(os.path.join(ROOT,'sampling')):
        
        #class image PATH
        for image_path in tqdm(sorted(iglob('{}\\sampling\\{}\\images\\*.jpg'.format(ROOT,folder)))):
            
            #Detect
            res_label, res_bbox, height, width = vild_model._detect(image_path)
            print('res_label =',len(res_label))
            
            #Make .xml label files
            inference2xml('{}\\sampling\\{}\\labels'.format(ROOT,folder),image_path,res_label, res_bbox, height, width)

if __name__ == '__main__':
    main()