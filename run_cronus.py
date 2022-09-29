from image_generator import *
from pattern_detector import *
from page_selector import *

# Apps
apps = ['backprop_10000', 'kmeans_5000', 'hotspot_256', 'quicksilver_500', 'cpd_10000', 'lud_512', 'bfs_128k', 'bptree_100k', 'pennant_leblanc']
app_labels = ['backprop', 'kmeans', 'hotspot', 'quicksilver', 'cpd', 'lud', 'bfs', 'bptree', 'pennant']

# Page Scheduling Frequencies
cori_best_perf_reqs_apps = [9000, 26000, 24900, 31100, 35500, 23200, 12400, 27600, 11700]
cori_dom_reuse_reqs_apps = [9000, 2961, 8384, 31262, 35650, 1938, 2495, 3519, 11738]

def cronus_pipeline(trace_dir, plotdir, resdir, res,  app_idx, benefit_factor, colormap):
    # Step 1: Run the image generator
    image_name, image_metadata = run_image_generator(trace_dir, app_idx, res, colormap, resdir, plotdir, 'png')

    # Step 2: Run the pattern detector
    selected_pixel_rows = run_pattern_detector(plotdir, image_name, res, colormap)

    # Step 3: Run the page selector
    sorted_page_ids = run_page_selector(plotdir, image_name, image_metadata, selected_pixel_rows, benefit_factor)

    return sorted_page_ids, image_metadata['num_pages']

if __name__ == "__main__":
    # Command line arguments
    trace_dir = sys.argv[1]
    plotdir = sys.argv[2]
    resdir = sys.argv[3]

    for app_idx in range(len(apps)):
        # Returns a list of pages ordered with priority for machine intelligent management
        sort_page_ids, num_pages = cronus_pipeline(trace_dir, plotdir, resdir, 256, app_idx, 'color', 'viridis')
