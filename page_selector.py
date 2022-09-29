import cv2, math
import numpy as np

# Apps
apps = ['backprop_10000', 'kmeans_5000', 'hotspot_256', 'quicksilver_500', 'cpd_10000', 'lud_512', 'bfs_128k', 'bptree_100k', 'pennant_leblanc']
app_labels = ['backprop', 'kmeans', 'hotspot', 'quicksilver', 'cpd', 'lud', 'bfs', 'bptree', 'pennant']

# Frequencies
cori_best_perf_reqs_apps = [9000, 26000, 24900, 31100, 35500, 23200, 12400, 27600, 11700]
cori_dom_reuse_reqs_apps = [9000, 2961, 8384, 31262, 35650, 1938, 2495, 3519, 11738]



def map_pixel_to_page(sorted_pixel_ids, num_pages, image_size):
    pages_per_pixel = float(num_pages) / image_size
    print(len(sorted_pixel_ids), pages_per_pixel)
    sorted_page_ids = []
    set_of_dups = set()
    for id in sorted_pixel_ids:
        # 0 page id is on the bottom left of the image
        for p in range(math.ceil(pages_per_pixel)):
            pid = num_pages - math.floor(id * pages_per_pixel) + p
            if pid > num_pages - 1:
                pid = num_pages - 1
            if pid not in set_of_dups:
                set_of_dups.add(pid)
                sorted_page_ids.append(pid)
    return sorted_page_ids

def run_page_selector(plotdir, image_name, image_metadata, pixel_rows, criteria):

    # Read the image
    image_path = plotdir + image_name
    image = cv2.imread(image_path)
    # Turn to gray scale for easier color manipulation
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Criteria 1: how many non white pixels per row? == overall page hotness
    benefit_per_row = []
    for idx in range(len(pixel_rows)):
        non_white_pixels_per_row = np.sum(imgray[pixel_rows[idx]] != 255) # hotness
        avg_pixel_color = 255 - np.average(imgray[pixel_rows[idx]]) # hotness variance (give weight to darker colors)
        benefit = non_white_pixels_per_row
        if criteria == 'color':
            benefit *= avg_pixel_color
        benefit_per_row.append(benefit)
    ordered_ids = np.argsort(benefit_per_row)[::-1]  # more to less
    ordered_pixel_ids = [pixel_rows[i] for i in ordered_ids]


    # Map the pixels to the corresponding pages
    sorted_page_ids = map_pixel_to_page(ordered_pixel_ids, image_metadata['num_pages'], image.shape[1]) # y-axis
    return sorted_page_ids