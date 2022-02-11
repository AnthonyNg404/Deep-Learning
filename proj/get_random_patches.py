from PIL import Image
import numpy as np
import os
import random

screen_height = 2560
screen_width = 1440
largest_uid = 72219

mean_img_width_height_all = np.array([388.7239905, 258.39138222])
mean_img_widht_height_cov_all = np.array([[165088.85026204, 61208.73027125],[61208.73027125, 74209.0172899]])

mean_img_width_height_occluded = np.array([397.78217822, 205.0990099])
mean_img_widht_height_cov_occluded = np.array([[218205.73207921, 67782.23178218], [67782.23178218, 42984.73009901]])

def get_img_size_distribution_all_img(img_path):
    all_img_widths = []
    all_img_heights = []
    for i in range(largest_uid):
        ui_image_path = os.path.join(img_path, str(i))
        if not os.path.exists(ui_image_path):
            continue
        for img_filename in os.listdir(ui_image_path):
            str_bounds = img_filename[1:img_filename.index(']')].split(', ')
            bounds = [int(x) for x in str_bounds]
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            all_img_widths.append(width)
            all_img_heights.append(height)
    width_height = [all_img_widths, all_img_heights]
    print(np.mean(width_height, axis=1))
    print(np.cov(width_height))

def get_img_size_distribution_sample(img_path):
    all_img_widths = []
    all_img_heights = []
    for img_filename in os.listdir(img_path):
        if ']' not in img_filename:
            continue
        str_bounds = img_filename[1:img_filename.index(']')].split(', ')
        bounds = [int(x) for x in str_bounds]
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        all_img_widths.append(width)
        all_img_heights.append(height)
    width_height = [all_img_widths, all_img_heights]
    print(width_height)
    print(np.mean(width_height, axis=1))
    print(np.cov(width_height))

def generate_random_patches(dir_path, num_patches, out_path):
    i = 0
    while i < num_patches:
        ui_id = str(random.choice(range(largest_uid)))
        ui_screenshot_path = os.path.join(dir_path, ui_id + '.jpg')
        if not os.path.exists(ui_screenshot_path):
            continue
        im = Image.open(ui_screenshot_path)
        width, height = im.size
        vertical_scale = 1.*height/screen_height
        horizontal_scale = 1.*width/screen_width 
        patch_width_height = np.random.multivariate_normal(mean_img_width_height_occluded, mean_img_widht_height_cov_occluded)
        patch_width = patch_width_height[0]
        patch_height = patch_width_height[1]
        if patch_width < 5 or patch_height < 5:
            continue
        x_max = screen_width - patch_width
        y_max = screen_height - patch_height
        left = np.random.uniform(0, x_max)
        top = np.random.uniform(0, y_max)
        right = left + patch_width
        bottom = top + patch_height
        cropped_img = im.crop((left*horizontal_scale, top*vertical_scale, right*horizontal_scale, bottom*vertical_scale))
        cropped_img.save(os.path.join(out_path, ui_id + "_" + str([left, top, right, bottom]) +  ".jpg"))
        i += 1





generate_random_patches('/Users/cusgadmin/ui_grouped_elements_study/combined', 100, "../random_patches")
#get_img_size_distribution_sample('/Users/cusgadmin/cs282A/occluded_images_sample')


