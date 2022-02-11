from PIL import Image
import json
import os
import queue
import shutil

scale_factor = 0.33 #1.25
screen_height = 2560
screen_width = 1440

def group_by_leaves(json_dict):
    groups = []
    nodes_to_expand = queue.Queue()
    nodes_to_expand.put(json_dict["children"])
    while not nodes_to_expand.empty():
        nodes = nodes_to_expand.get()
        new_group = []
        for node in nodes:
            if node.get("children"):
                nodes_to_expand.put(node["children"])
            else:
                new_group.append(node)
        if len(new_group) > 0:
            groups.append(new_group)
    return groups

def get_all_text_and_images(semantics_json, view_hierarchy_json, ui_id, ui_image):
    text_dict = dict()
    img_bounds = []
    view_hierarchy_dict = dict()
    semantic_groups = group_by_leaves(semantics_json)
    for group in semantic_groups:
        for element in group:
            if element["componentLabel"] == "Image":
                img_bounds.append(element["bounds"])
    dir_path = os.path.join("../extracted_ui_images_test", ui_id)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)
    im = Image.open(ui_image)
    width, height = im.size
    vertical_scale = 1.*height/screen_height
    horizontal_scale = 1.*width/screen_width 
    for img_bound in img_bounds:
        left, top, right, bottom =  img_bound
        if right - left < 5 or bottom - top < 5:
            continue
        cropped_img = im.crop((left*horizontal_scale, top*vertical_scale, right*horizontal_scale, bottom*vertical_scale))
        cropped_img.save(os.path.join(dir_path, str(img_bound) + ".jpg"))

def main():
    for i in range(67219,67310):
        print(i)
        ui_id = str(i)
        semantics_path = '/Users/cusgadmin/ui_grouped_elements_study/semantic_annotations/' + ui_id + '.json'
        view_hierarchy_path = '/Users/cusgadmin/ui_grouped_elements_study/combined/' + ui_id + '.json'
        ui_image = '/Users/cusgadmin/ui_grouped_elements_study/combined/' + ui_id + '.jpg'
        if os.path.exists(semantics_path):
            semantics_view_hierarchy = json.load(open(semantics_path))
            ui_view_hierarchy = json.load(open(view_hierarchy_path))
            ui_view_hierarchy = ui_view_hierarchy["activity"]["root"]
            get_all_text_and_images(semantics_view_hierarchy, ui_view_hierarchy, ui_id, ui_image)

if __name__ == '__main__':
    main()