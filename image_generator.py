import sys
from hmem_sim.profile import *
from hmem_sim.perf_model import *
from visuals.plot_lib import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Apps
apps = ['backprop_10000', 'kmeans_5000', 'hotspot_256', 'quicksilver_500', 'cpd_10000', 'lud_512', 'bfs_128k', 'bptree_100k', 'pennant_leblanc']
app_labels = ['backprop', 'kmeans', 'hotspot', 'quicksilver', 'cpd', 'lud', 'bfs', 'bptree', 'pennant']

# Frequencies
cori_best_perf_reqs_apps = [9000, 26000, 24900, 31100, 35500, 23200, 12400, 27600, 11700]
cori_dom_reuse_reqs_apps = [9000, 2961, 8384, 31262, 35650, 1938, 2495, 3519, 11738]

def get_ordered_pages_cronus(sim):
    benefit_per_page = []
    page_ids = range(sim.profile.hmem.num_pages)
    for id in page_ids:
      page = sim.profile.hmem.page_list[id]
      benefit = page.hotness #* page.hotness_var
      benefit_per_page.append(benefit)
    norm_benefit_per_page = NormalizeData(benefit_per_page)

    sorted_idxs = np.argsort(norm_benefit_per_page)[::-1] # descending order
    ordered_page_ids = [page_ids[i] for i in sorted_idxs]
    sorted_benefit_per_page = [norm_benefit_per_page[i] for i in ordered_page_ids]

    return ordered_page_ids, sorted_benefit_per_page


def plot_trace_color(ordered_page_list, prof, pixel_size, colormap, outname, format):

    plt.rc('axes.spines', **{'bottom': False, 'left': False, 'right': False, 'top': False})
    #plt.figure(figsize=(2.5, 2), dpi=350)
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    plt.figure(figsize=(pixel_size * px, pixel_size * px))
    colormap = cm.get_cmap(colormap)
    plt.rcParams["image.interpolation"] = 'nearest'


    #Plot all trace in a light color
    plt.plot(range(prof.traffic.num_reqs), [req.page_id for req in prof.traffic.req_seq], '.', markersize=0.01, c=colormap(0.99))

    #Plot the selected pages in their priority ordering
    idx = 0
    for page_id in ordered_page_list:
        dataX = prof.hmem.page_list[page_id].req_ids
        dataY = [page_id for p in range(len(dataX))]
        c = float(idx / prof.traffic.num_pages)
        plt.plot(dataX, dataY, '.', markersize=0.01, c=colormap(c))
        idx += 1

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig(outname + "." + format, format=format)


def run_image_generator(trace_dir, app_idx, resolution, colormap, resdir, plotdir, format):
    # Read trace
    app, app_label = apps[app_idx], app_labels[app_idx]
    trace_file = trace_dir + 'trace_' + app + '.txt'
    prof = Profile(trace_file)
    prof.init()

    # Plot trace in color (hotness + hotness_var)
    freq = cori_best_perf_reqs_apps[app_idx] # to match with the evaluation
    sim = PerfModel(prof, 'Fast:NearSlow', 'history', 0.2, freq)
    sim.init()
    ordered_page_ids, ordered_norm_benefit = get_ordered_pages_cronus(sim)

    image_name = 'trace_color_cronus_' + str(resolution) + '_' + app_labels[app_idx] + '.png'
    plot_trace_color(ordered_page_ids, prof, resolution, colormap, plotdir + image_name, format)
    image_name += '.png'

    image_metadata = {}
    image_metadata['num_pages'] = prof.traffic.num_pages

    return image_name, image_metadata





