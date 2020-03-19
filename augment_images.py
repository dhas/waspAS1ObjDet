import os
import albumentations as A
from xml.etree import ElementTree as ET
import json
import cv2 as cv2

def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes




BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
    return img


def visualize(annotations, category_id_to_name):
    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
        img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)


def get_aug(aug, min_area=0., min_visibility=0.):
    return A.Compose(aug, bbox_params=A.BboxParams(format='pascal_voc', min_area=min_area,
                                               min_visibility=min_visibility, label_fields=['category_id']))





'''
copy and paste from http://effbot.org/zone/element-lib.htm#prettyprint
it basically walks your tree and adds spaces and newlines so the tree is
printed in a nice way
'''
def pretty_print(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            pretty_print(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def build_pascal_voc_formart(pascal_voc_data):
    try:
        '''It the structure is correct procced further'''
       # pascal_voc_data = json.loads(pascal_voc_data)
        if len(pascal_voc_data):
            for each_pascal_voc in pascal_voc_data:
                get_folder = each_pascal_voc['folder']
                get_filename = each_pascal_voc['filename']
                get_path = each_pascal_voc['path']
                get_database = each_pascal_voc['source']['database']
                get_width = each_pascal_voc['size']['width']
                get_height = each_pascal_voc['size']['height']
                get_depth = each_pascal_voc['size']['depth']
                get_segmented = each_pascal_voc['segmented']
                get_object = each_pascal_voc['objects']
                try:
                    result = create_pascal_voc_file(get_folder, get_filename, get_path, get_database, get_width,get_height,get_depth, get_segmented, get_object)
                    if result:
                        print(f"xml file created for {get_filename} file", )
                    else:
                        print("Problem while creating xml file created for {} ", get_filename)
                except:
                    print("Problem while creating pascal voc file")


    except:
        print("Problem while reading the pascal voc data")


def create_pascal_voc_file(get_folder, get_filename, get_path, get_database, get_width,get_height,get_depth, get_segmented, get_object):
    try:
        '''Getting the basic details like folder, filename and path'''
        voc_xml = ET.Element("annotation")
        folder = ET.SubElement(voc_xml, "folder")
        folder.text = str(get_folder)
        filename = ET.SubElement(voc_xml, "filename")
        filename.text = str(get_filename)
        path = ET.SubElement(voc_xml, "path")
        path.text = str(get_path)
        source = ET.SubElement(voc_xml, "source")
        database = ET.SubElement(source, "database")
        database.text = str(get_database)
        '''Getting the image properties of width, height, depth'''
        size = ET.SubElement(voc_xml, "size")
        width = ET.SubElement(size, "width")
        width.text = str(get_width)
        height = ET.SubElement(size, "height")
        height.text = str(get_height)
        depth = ET.SubElement(size, "depth")
        depth.text = str(get_depth)
        segmented = ET.SubElement(voc_xml, "segmented")
        segmented.text = str(get_segmented)
        for each_object in get_object:
            '''Loop through each object'''
            object = ET.SubElement(voc_xml, "object")
            name = ET.SubElement(object, "name")
            name.text = str(each_object["name"])
            pose = ET.SubElement(object, "pose")
            pose.text = str(each_object["pose"])
            truncated = ET.SubElement(object, "truncated")
            truncated.text = str(each_object["truncated"])
            difficult = ET.SubElement(object, "difficult")
            difficult.text = str(each_object["difficult"])
            occluded = ET.SubElement(object, "occluded")
            occluded.text = str(each_object["occluded"])
            bndbox = ET.SubElement(object, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(each_object["bndbox"]["xmin"])
            xmax = ET.SubElement(bndbox, "ymin")
            xmax.text = str(each_object["bndbox"]["ymin"])
            ymin = ET.SubElement(bndbox, "xmax")
            ymin.text = str(each_object["bndbox"]["xmax"])
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(each_object["bndbox"]["ymax"])

        '''Make the xml pretty look'''
        pretty_print(voc_xml)
        '''Tree xml '''
        tree = ET.ElementTree(voc_xml)
        try:
            '''Write to xml file'''
            get_xml_file_name = get_filename.split('.')[0]
        except:
            print("Please cross check your filename should be like image.png, image.jpg, image_165151.jpeg")
        print(get_xml_file_name)
        tree.write(f"./data/{get_xml_file_name}.xml", xml_declaration=False, encoding='utf-8', method="xml")
        return True
    except:
        '''Problem while creating xml file'''
        return  False

def save_augmentation(image_name, augmented, augmentation_type):
    folder = './data'
    filename = image_name
    image = augmented['image']
    img_dim = image.shape
    image_height =img_dim[0]
    image_width = img_dim[1]
    image_depth = img_dim[2]
    bbox = augmented['bboxes'][0]
    pascal_voc_data = [
                                        {
                                            "folder":f"",
                                            "filename": f"{filename}_{augmentation_type}.jpg",
                                            "path":f"{folder}/{filename}",
                                            "source":{"database":"database"},
                                            "size":{"width":image_height,"height":image_width,"depth":image_depth},
                                            "segmented":0,
                                            "objects":[{"name":"name","pose":"pose","truncated":"0","difficult":0,"occluded":"occluded","bndbox":{"xmin":int(bbox[0]),"ymin":int(bbox[1]),"xmax":int(bbox[2]),"ymax":int(bbox[3])}}]
                                        }

                    ]



    '''Call the function to  build pascal formart'''
    build_pascal_voc_formart(pascal_voc_data)
    cv2.imwrite(f"./data/{pascal_voc_data[0]['filename']}", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])




def blur(image_name, image, boxes):
    bbox_params = A.BboxParams(format='pascal_voc', min_area=1, min_visibility=0.5, label_fields=['category_id'])
    aug = get_aug([A.MotionBlur(blur_limit=50, p=1)])
    augmented = aug(**annotations)
    save_augmentation(image_name, augmented, 'blur')

def horzontalFlip(image_name, image, boxes):
    bbox_params = A.BboxParams(format='pascal_voc', min_area=1, min_visibility=0.5, label_fields=['category_id'])
    aug = get_aug([ A.HorizontalFlip(p=0.7)])
    augmented = aug(**annotations)
    save_augmentation(image_name, augmented, 'hFlip')

def verticalFlip(image_name, image, boxes):
    bbox_params = A.BboxParams(format='pascal_voc', min_area=1, min_visibility=0.5, label_fields=['category_id'])
    aug = get_aug([ A.VerticalFlip(p=0.7)])
    augmented = aug(**annotations)
    save_augmentation(image_name, augmented, 'VFlip')


if __name__ == "__main__":
    image_names = ([file.split('.')[0] for file in os.listdir(os.path.join(os.getcwd(),"./data")) if ".jpg" in file])

    for image_name in image_names:
        name, boxes = read_content(f"./data/{image_name}.xml")
        image = cv2.imread(f"./data/{image_name}.jpg")

        img_dim = image.shape
        image_height =img_dim[0]
        image_width = img_dim[1]
        print('image_width', image_width)
        print('image_height', image_height)
        print('boxes', boxes)
        boxes[0][2] = min(boxes[0][2],image_width-boxes[0][1])
        boxes[0][3] = min(boxes[0][3],2000)
        annotations = {'image': image, 'bboxes': boxes, 'category_id': [1]}

        #verticalFlip(image_name, image, boxes)
        #horzontalFlip(image_name, image, boxes)
        #blur(image_name, image, boxes)
