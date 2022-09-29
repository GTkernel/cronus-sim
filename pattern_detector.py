import sys, cv2, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from hmem_sim.profile import *

# Apps
apps = ['backprop_10000', 'kmeans_5000', 'hotspot_256', 'quicksilver_500', 'cpd_10000', 'lud_512', 'bfs_128k', 'bptree_100k', 'pennant_leblanc']
app_labels = ['backprop', 'kmeans', 'hotspot', 'quicksilver', 'cpd', 'lud', 'bfs', 'bptree', 'pennant']

# Frequencies
cori_best_perf_reqs_apps = [9000, 26000, 24900, 31100, 35500, 23200, 12400, 27600, 11700]
cori_dom_reuse_reqs_apps = [9000, 2961, 8384, 31262, 35650, 1938, 2495, 3519, 11738]



def run_pattern_detector(plotdir, image_name, resolution, colormap):
    # read the image
    image_path = plotdir + image_name
    image = cv2.imread(image_path)

    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #30 for the rest of experiments
    length = 7 # this is an important parameter, to ignore small areas of dark pixels.
    blur = cv2.GaussianBlur(imgray, (length, length), 0) # to create bigger contours, not follow the lines as-is
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # automatic detection
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area size
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    contours_to_draw = []
    areas = []
    for c in sorted_contours[1:]:
        area = cv2.contourArea(c)
        areas.append(area)
    for c in sorted_contours[1:]:
        area = cv2.contourArea(c)
        if area > np.mean(areas) / 2:
            contours_to_draw.append(c)

    # draw all contours
    imageee = cv2.drawContours(image, contours_to_draw, -1, (0, 0, 255), 2)
    save_path = plotdir + 'pattern_detect_' + image_name
    #cv2.imshow('image',imageee)
    #cv2.waitKey(0)

    # closing all open windows
    #cv2.destroyAllWindows()
    cv2.imwrite(save_path, imageee)


    selected_pixel_rows = []
    for c in contours_to_draw:
        x, y, w, h = cv2.boundingRect(c)
        selected_pixel_rows += range(y, y + h)
    return list(set(selected_pixel_rows)) # avoid duplicates

def map_pixel_to_page(pixel_ids, pages_per_pixel, num_pages):
    page_ids = []
    for id in pixel_ids:
        # 0 page id is on the bottom left of the image
        for p in range(math.ceil(pages_per_pixel)):
            pid = num_pages - math.floor(id * pages_per_pixel) + p
            page_ids.append(pid) # avoid duplicates
    return page_ids

def extract_pages_from_contours(contours, image_dims, trace_dir, app_idx):

    # IMPORTANT: the image metadata are the number of pages!
    app, app_label = apps[app_idx], app_labels[app_idx]
    trace_file = trace_dir + 'trace_' + app + '.txt'
    prof = Profile(trace_file)
    prof.init()
    num_pages = prof.traffic.num_pages
    pages_per_pixel =float(prof.traffic.num_pages) / image_dims[1] # y-axis

    selected_page_ids = []
    selected_pixel_rows = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ids = map_pixel_to_page(range(y, y+h), pages_per_pixel, prof.traffic.num_pages)
        selected_page_ids += ids
        selected_pixel_rows += range(y, y+h)
    return selected_page_ids, selected_pixel_rows

def sort_pages_by_color(image, page_ids, pixel_rows):

    # For each page == pixel row:
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # for easier color manipulation
    print(imgray.shape)
    non_white_pixels_per_row = []
    for idx in range(len(pixel_rows)):
        # Count number of non white pixels per row == hotness:
        count = np.sum(imgray[pixel_rows[idx]] != 255)
        non_white_pixels_per_row.append(count)

    ordered_ids = np.argsort(non_white_pixels_per_row)[::-1]  # bigger to smaller
    ordered_pixel_rows = [pixel_rows[i] for i in ordered_ids]
    ordered_page_ids = map_pixel_to_page(ordered_pixel_rows, )

if __name__ == "__main__":
    # Command line arguments
    trace_dir = sys.argv[1]
    plotdir = sys.argv[2]

    #for app_idx in range(len(app_labels)):
    app_idx = 1
    image_path = plotdir + 'trace_color_cronus_' + app_labels[app_idx] + '.png'
    save_path = plotdir + 'trace_color_pattern_detect_' + app_labels[app_idx] + '.png'
    contours, image = run_pattern_detector(image_path, save_path)
    selected_page_ids, selected_pixel_rows = extract_pages_from_contours(contours, image.shape, trace_dir, app_idx)
    sort_pages_by_color(image, selected_page_ids, selected_pixel_rows)
