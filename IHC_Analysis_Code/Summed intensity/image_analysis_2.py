#%%
#This is my pipeline to do image upscaling and segmentation using Cellpose 3 from Janelia labs (doi: https://doi.org/10.1101/2024.02.10.579780). The basic steps are:
#1. Take the .czi files and extract the brightest slice from each channel and convert them to .tif files
#2. Use Cellpose 3 to first upscale the images and then segment the cells in the .tif files
#3. Measure the intensity of the cells in the .tif files

#to do these three steps there are three major functions that are needed and implemented below. They are invidually commented on
#%%

from cellpose import core, utils, io, models, metrics
from glob import glob
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 200
from cellpose import utils, io
import pandas as pd
#first, we need to import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os
from aicsimageio import AICSImage
from aicsimageio.readers import CziReader
from aicsimageio.writers import OmeTiffWriter

from cellpose import denoise
from cellpose import models

from skimage.transform import resize
import importlib
from skimage import io, filters
from skimage.measure import label, regionprops, find_contours
import cv2

#%%
#here is the first set of functions we need, first we need to get the masks from the files that contain TH
#now we want to write this into a function
def get_cell_masks_upsample(img_path_seg, model_type='cyto3', diam_mean=30.0, channels=[0, 0]):
    #load in the image
    tiff_img = io.imread(img_path_seg)
    #upsample the image
    dn = denoise.DenoiseModel(model_type="upsample_cyto3", gpu=False)
    imgs_dn = dn.eval(tiff_img, channels=None, diameter=10.0)
    #fit the masks
    mask_model = models.CellposeModel(gpu=False, pretrained_model=False, model_type=model_type, diam_mean=diam_mean,
                                      device=None, backbone='default')
    masks = mask_model.eval(imgs_dn, channels=channels)
    cells = masks[0]
    # plt.imshow(cells)
    #get the unique cells
    unique_cells = np.unique(cells)
    unique_cells = unique_cells[unique_cells != 0]  # Assuming 0 is the background
    return cells, unique_cells


#now lets create a function that will take the cells and unique cells, and a second image, and return the intensities of the second image within the cells
def get_cell_intensities_from_image_upsample(cells, unique_cells, img_path_measure, marker_name):
    #load in the image
    tiff_img = io.imread(img_path_measure)
    #upsample the image
    dn = denoise.DenoiseModel(model_type="upsample_cyto3", gpu=False)
    imgs_dn = dn.eval(tiff_img, channels=None, diameter=10.0)
    # plt.imshow(imgs_dn)
    #get the intensities
    cell_intensities = {}
    for cell_id in unique_cells:
        cell_mask = (cells == cell_id)
        cell_intensity_values = imgs_dn[cell_mask]
        cell_intensities[cell_id] = {
            'mean_intensity': np.mean(cell_intensity_values),
            'sum_intensity': np.sum(cell_intensity_values),
            'max_intensity': np.max(cell_intensity_values),
            'min_intensity': np.min(cell_intensity_values),
            'std_intensity': np.std(cell_intensity_values),
        }
    # Compute statistics for the entire image
    overall_intensity_values = imgs_dn[cells > 0]  # Consider only the cells, exclude background
    overall_intensity_stats = {
        'mean_intensity': np.mean(overall_intensity_values),
        'sum_intensity': np.sum(overall_intensity_values),
        'mean_sum_intensity': np.sum(overall_intensity_values) / len(unique_cells),  # Mean intensity per cell
        'max_intensity': np.max(overall_intensity_values),
        'min_intensity': np.min(overall_intensity_values),
        'std_intensity': np.std(overall_intensity_values),
        'cell_count': len(unique_cells),
    }
    return cell_intensities, overall_intensity_stats

#now we need to write the wrapper function that will call the two functions above from a filepath containing the images
def collect_segmented_image_data_upsample(tiff_path, marker_name):
    overall_files_intensities = []
    th_intensities = []
    all_cells = []
    filenames = []
    for files in sorted(os.listdir(tiff_path)):
        if files.endswith('.tif') and 'TH' in files:
            #from the TH containing .tif file, we want to get the cell masks
            cells, unique_cells = get_cell_masks_upsample(tiff_path+files)
            all_cells.append(cells)
            #from the SK containing .tif file, we want to get the intensities of the cells
            cell_intensities, overall_intensity_stats = get_cell_intensities_from_image_upsample(cells, unique_cells, tiff_path+files.replace('TH', marker_name), marker_name)
            overall_files_intensities.append(overall_intensity_stats)

            #lets also collect the th intensity from this slice
            th_cell_intensities, th_overall_intensity_stats = get_cell_intensities_from_image_upsample(cells, unique_cells, tiff_path+files, 'TH')
            th_intensities.append(th_overall_intensity_stats)

    #lets make a dataframe from the overall_files_intensities
    intensities_df = pd.DataFrame(overall_files_intensities)
    th_intensities_df = pd.DataFrame(th_intensities)
    # plt.show()
    return intensities_df, th_intensities_df, all_cells



#nonupsampled to follow
def get_cell_masks(img_path_seg, model_type='cyto3', diam_mean=80.0, channels=[0, 0]):
    #load in the image
    tiff_img = io.imread(img_path_seg)
    #upsample the image

    #fit the masks
    mask_model = models.CellposeModel(gpu=False, pretrained_model=False, model_type=model_type, diam_mean=diam_mean,
                                      device=None, backbone='default')
    masks = mask_model.eval(tiff_img, channels=channels)
    cells = masks[0]
    # plt.imshow(cells)
    #get the unique cells
    unique_cells = np.unique(cells)
    unique_cells = unique_cells[unique_cells != 0]  # Assuming 0 is the background
    return cells, unique_cells

def get_cell_intensities_from_image(cells, unique_cells, img_path_measure, marker_name):
    #load in the image
    tiff_img = io.imread(img_path_measure)
    # plt.imshow(imgs_dn)
    #get the intensities
    cell_intensities = {}
    for cell_id in unique_cells:
        cell_mask = (cells == cell_id)
        cell_intensity_values = tiff_img[cell_mask]
        cell_intensities[cell_id] = {
            'mean_intensity': np.mean(cell_intensity_values),
            'sum_intensity': np.sum(cell_intensity_values),
            'max_intensity': np.max(cell_intensity_values),
            'min_intensity': np.min(cell_intensity_values),
            'std_intensity': np.std(cell_intensity_values),
        }
    # Compute statistics for the entire image
    overall_intensity_values = tiff_img[cells > 0]  # Consider only the cells, exclude background
    overall_intensity_stats = {
        'mean_intensity': np.mean(overall_intensity_values),
        'sum_intensity': np.sum(overall_intensity_values),
        'mean_sum_intensity': np.sum(overall_intensity_values) / len(unique_cells),  # Mean intensity per cell
        'max_intensity': np.max(overall_intensity_values),
        'min_intensity': np.min(overall_intensity_values),
        'std_intensity': np.std(overall_intensity_values),
        'cell_count': len(unique_cells),
    }
    return cell_intensities, overall_intensity_stats




#%%
#now we need to write the wrapper function that will call the two functions above from a filepath containing the images
def collect_segmented_image_data(tiff_path, marker_name):
    overall_files_intensities = []
    th_intensities = []
    all_cells = []
    filenames = []
    for files in sorted(os.listdir(tiff_path)):
        if files.endswith('.tif') and 'TH' in files:
            #from the TH containing .tif file, we want to get the cell masks
            cells, unique_cells = get_cell_masks(tiff_path+files)
            all_cells.append(cells)
            #from the SK containing .tif file, we want to get the intensities of the cells
            cell_intensities, overall_intensity_stats = get_cell_intensities_from_image(cells, unique_cells, tiff_path+files.replace('TH', marker_name), marker_name)
            overall_files_intensities.append(overall_intensity_stats)

            #lets also collect the th intensity from this slice
            th_cell_intensities, th_overall_intensity_stats = get_cell_intensities_from_image(cells, unique_cells, tiff_path+files, 'TH')
            th_intensities.append(th_overall_intensity_stats)

    #lets make a dataframe from the overall_files_intensities
    intensities_df = pd.DataFrame(overall_files_intensities)
    th_intensities_df = pd.DataFrame(th_intensities)
    # plt.show()
    return intensities_df, th_intensities_df, all_cells




#here we take the upsample to get the mask, resize with skimage, and then we get the intensity from the original image!
#%%
def get_cell_masks_resize(img_path_seg, model_type='cyto3', diam_mean=30.0, channels=[0, 0]):
    #load in the image
    tiff_img = io.imread(img_path_seg)
    #upsample the image
    dn = denoise.DenoiseModel(model_type="upsample_cyto3", gpu=False)
    imgs_dn = dn.eval(tiff_img, channels=None, diameter=10.0)
    #fit the masks
    mask_model = models.CellposeModel(gpu=False, pretrained_model=False, model_type=model_type, diam_mean=diam_mean,
                                      device=None, backbone='default')
    masks = mask_model.eval(imgs_dn, channels=channels)
    cells = masks[0]

    cells = resize(cells, tiff_img.shape, anti_aliasing=False)

    # plt.imshow(cells)
    #get the unique cells
    unique_cells = np.unique(cells)
    unique_cells = unique_cells[unique_cells != 0]  # Assuming 0 is the background
    return cells, unique_cells





def get_cell_intensities_from_original_tiff(cells, unique_cells, img_path_measure, marker_name):
    #load in the image
    tiff_img = io.imread(img_path_measure)


    # plt.imshow(imgs_dn)
    #get the intensities
    cell_intensities = {}
    for cell_id in unique_cells:
        cell_mask = (cells == cell_id)
        cell_intensity_values = tiff_img[cell_mask]
        cell_intensities[cell_id] = {
            'mean_intensity': np.mean(cell_intensity_values),
            'sum_intensity': np.sum(cell_intensity_values),
            'max_intensity': np.max(cell_intensity_values),
            'min_intensity': np.min(cell_intensity_values),
            'std_intensity': np.std(cell_intensity_values),
        }
    # Compute statistics for the entire image
    overall_intensity_values = tiff_img[cells > 0]  # Consider only the cells, exclude background
    overall_intensity_stats = {
        'mean_intensity': np.mean(overall_intensity_values),
        'sum_intensity': np.sum(overall_intensity_values),
        'mean_sum_intensity': np.sum(overall_intensity_values) / len(unique_cells),  # Mean intensity per cell
        'max_intensity': np.max(overall_intensity_values),
        'min_intensity': np.min(overall_intensity_values),
        'std_intensity': np.std(overall_intensity_values),
        'cell_count': len(unique_cells),
    }
    return cell_intensities, overall_intensity_stats


#now we need to write the wrapper function that will call the two functions above from a filepath containing the images
def collect_segmented_image_data_original(tiff_path, marker_name):

    overall_files_intensities = []
    overall_individual_cells = []

    th_intensities = []
    overall_individual_th_cells = []

    all_cells = []
    filenames = []
    for files in sorted(os.listdir(tiff_path)):
        if files.endswith('.tif') and 'TH' in files:
            #from the TH containing .tif file, we want to get the cell masks
            cells, unique_cells = get_cell_masks_resize(tiff_path+files)
            all_cells.append(cells)
            #from the SK containing .tif file, we want to get the intensities of the cells
            cell_intensities, overall_intensity_stats = get_cell_intensities_from_original_tiff(cells, unique_cells, tiff_path+files.replace('TH', marker_name), marker_name)
            overall_files_intensities.append(overall_intensity_stats)
            overall_individual_cells.append(cell_intensities)

            #lets also collect the th intensity from this slice
            th_cell_intensities, th_overall_intensity_stats = get_cell_intensities_from_original_tiff(cells, unique_cells, tiff_path+files, 'TH')
            th_intensities.append(th_overall_intensity_stats)
            overall_individual_th_cells.append(th_cell_intensities)

    #lets make a dataframe from the overall_files_intensities
    intensities_df = pd.DataFrame(overall_files_intensities)
    th_intensities_df = pd.DataFrame(th_intensities)
    # plt.show()
    return intensities_df, overall_individual_cells, th_intensities_df, overall_individual_th_cells, all_cells


#%%
def collect_cum_sorted(measurement, list_of_dictionaries):

    intensities_all = []
    for indivudual_cells in list_of_dictionaries:
        intensities = [info[measurement] for info in indivudual_cells.values()]

        intensities_all.extend(intensities)

    intensities_all_sorted = np.sort(intensities_all)

    return intensities_all_sorted


#%%
#%%
#now we need to write the wrapper function that will call the two functions above from a filepath containing the images
def mip_binarize_and_fluor_by_area(tiff_path, marker_name):
    total_intensity_append = []
    total_intensit = []
    all_cells = []
    filenames = []
    for files in sorted(os.listdir(tiff_path)):

        if files.endswith('.tif') and marker_name in files:

            tiff_img = io.imread(files)
            #lets increase the intensity of the tiff im
            tiff_img = tiff_img/np.max(tiff_img)
            # plt.imshow(tiff_img)
            #first lets otsu threshold on th
            threshold = filters.threshold_otsu(tiff_img)
            binary = tiff_img > threshold
            #lets fill the binary mask
            from scipy import ndimage
            binary = ndimage.binary_fill_holes(binary)
            #lets create a rectangle around the binary mask

            binary = ndimage.binary_erosion(binary, iterations=1)



            # plt.imshow(np.rot90(binary))
            plt.show()
            #now lets calculate the total intensity, from the original tiff_img using the binary as the mask
            #lets rescale the tiff_img to 0-1

            total_intensity = np.sum(tiff_img)
            #can we get the area of the binary mask?
            area = np.sum(binary)
            print(area)
            fluor_per_area = total_intensity/area





            total_intensity_append.append(total_intensity)
    #lets convert total intensities into a dataframe
    int_df = pd.DataFrame(total_intensity_append, columns=['Fluorescence per Area'])
    return int_df
#%%
def pval_to_asteriks(pval):
    if pval < 0.0001:
        return '****'
    elif pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return 'ns'

    #lets find out of the data is normal or not

#%%
def paired_data_stats(total_df, df_1, df_2):
    #test if normal
    norm_stat, norm_p = shapiro(total_df)
    # print('Statistics=%.3f, p=%.3f' % (stat, p))

    #print whether it is normal or not
    if norm_p > 0.05:
        test_name = 'Paired t-test'
        #run rel t test
        stat, p = ttest_rel(df_1, df_2)
        # print('T-test: t=%.5f, p=%.5f' % (stat, p))
        asterik = pval_to_asteriks(p)
    else:
        test_name = 'wilcoxon'
        #run wilcoxon

        stat, p = wilcoxon(df_1, df_2)
        # print('Wilcoxon: Statistics=%.5f, p=%.5f' % (stat, p))
        asterik = pval_to_asteriks(p)
    return stat, p, test_name, asterik

#%%
def unpaired_data_stats(total_df, df_1, df_2):
    from scipy.stats import shapiro
    from scipy.stats import ttest_ind
    from scipy.stats import ttest_rel
    from scipy.stats import wilcoxon
    #test if normal
    norm_stat, norm_p = shapiro(total_df)
    # print('Statistics=%.3f, p=%.3f' % (stat, p))

    #print whether it is normal or not
    if norm_p > 0.05:
        test_name = 'Unpaired t-test'
        #run rel t test
        stat, p = ttest_ind(df_1, df_2)
        # print('T-test: t=%.5f, p=%.5f' % (stat, p))
        asterik = pval_to_asteriks(p)
    else:
        test_name = 'Mann Whitney U'
        #run wilcoxon

        stat, p = mannwhitneyu(df_1, df_2)
        # print('Mann Whitney U: Statistics=%.5f, p=%.5f' % (stat, p))
        asterik = pval_to_asteriks(p)
    return stat, p, test_name, asterik