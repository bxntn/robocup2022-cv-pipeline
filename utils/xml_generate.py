import xml.etree.cElementTree as ET

import numpy as np
import os

import xml.etree.cElementTree as ET

def inference2xml(root_dir,image_path,label,bbox,height,width):
    root = ET.Element("annotation")
    filename = image_path.split("\\")[-1]
    ET.SubElement(root, "filename").text = filename

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"


    for i in range(len(label)):
        object_bbox = ET.SubElement(root, "object")
        ET.SubElement(object_bbox, "name").text = label[i]

        bbox2 = ET.SubElement(object_bbox, "bndbox")
        y1, x1, y2, x2 = int(np.floor(bbox[i][0])), int(np.floor(bbox[i][1])), int(np.ceil(bbox[i][2])), int(np.ceil(bbox[i][3]))
        ET.SubElement(bbox2, "xmin").text = str(x1)
        ET.SubElement(bbox2, "xmax").text = str(x2)
        ET.SubElement(bbox2, "ymin").text = str(y1)
        ET.SubElement(bbox2, "ymax").text = str(y2)

    b_xml = ET.tostring(root)
    
    with open(os.path.join(root_dir,f'{filename.split(".")[0]}.xml'), "wb") as f:
        f.write(b_xml)