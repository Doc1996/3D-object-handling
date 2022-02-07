import os
import xml.etree.ElementTree



def convert_xml_to_txt(xml_file_name, class_name_to_id_mapping, xml_labels_folder, txt_labels_folder):
    xml_file_rel_path = os.path.join(xml_labels_folder, xml_file_name).replace(os.sep, "/")
    root = xml.etree.ElementTree.parse(xml_file_rel_path).getroot()
    
    info_dict = {}
    info_dict["bboxes"] = []


    for elem in root:
        if elem.tag == "filename":
            info_dict["filename"] = elem.text
        
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))
            
            info_dict["image_size"] = tuple(image_size)
        
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)
            info_dict["bboxes"].append(bbox)
    
    print_buffer = []


    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Some label(s) contain(s) invalid class. All used classes must be from ", list(class_name_to_id_mapping.keys()), ".")
        
        b_center_x = (b["xmin"] + b["xmax"]) / 2
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        image_w, image_h, image_c = info_dict["image_size"]
        b_center_x /= image_w
        b_center_y /= image_h
        b_width /= image_w
        b_height /= image_h
        
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))


    save_file_name = os.path.join(txt_labels_folder, xml_file_name.replace(".xml", ".txt")).replace(os.sep, "/")
    print("\n".join(print_buffer), file=open(save_file_name, "w"))