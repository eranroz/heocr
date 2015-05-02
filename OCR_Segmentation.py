# -*- coding: utf-8 -*-
__author__ = 'ravidziv'
import cv2
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.measure import regionprops
from skimage.color import label2rgb
from scipy import ndimage
from scipy.signal import convolve2d
import numpy as np
import matplotlib.patches as mpatches
from image_extractor import word_segmentation, down_sample
from matplotlib import pyplot as plt
from localSVM import *
import skimage
def opening(mask, struc):
  return ndimage.binary_opening(mask, structure=struc).astype(mask.dtype)


#Build struct (matrix) with the give size and the given num of ones
def build_struct(size, num_of_ones):
    struc = np.zeros((size,size))
    for i in range(0,num_of_ones):
        struc[(size//2), i] = 1
    return struc

def findHeight(regions):
    #Find the height according to the max value in the histogram
    max_val = 1000
    min_val = 8
    length_of_bin =4
    heights =[]
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        current_hegiht = abs(maxr-minr)
        if current_hegiht>=min_val and current_hegiht<=max_val:
            heights.append(current_hegiht)
    [hist, bins] = np.histogram(heights, bins=range(min(heights), max(heights) + length_of_bin, length_of_bin))
    #plt.hist(heights, bins=range(min(heights), max(heights) + length_of_bin, length_of_bin))
    #plt.show()
    counts = np.bincount(heights)
    height = max(np.where(counts>=4)[0])
    index = np.argmax(hist)
    #height = bins[index+1]
    return height

def sortedByHeight(regions):
    new_regions = []
    height = findHeight(regions)
    factor = 0.7
    height_max = height + height*factor
    height_min = max(5,height - height*factor)
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        curent_height = abs(maxr-minr)
        if  (curent_height<=height_max and curent_height>=height_min):
            new_regions.append(region)
    return new_regions

def findWidth(regions, num=0):
    #Find the height according to the max value in the histogram
    max_val = 100000
    min_val = 8
    length_of_bin = 5
    widths =[]
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        current_width = abs(maxc-minc)
        if current_width>min_val and current_width<max_val:
            widths.append(current_width)
    [hist, bins] = np.histogram(widths, bins=range(min(widths), max(widths) + length_of_bin, length_of_bin))
    sort_ind = np.argsort(hist)
    index = sort_ind[num]
    width = bins[index+1]
    #plt.hist(widths, bins=range(min(widths), max(widths) + length_of_bin, length_of_bin))
    #plt.show()
    width = max(widths)
    return width
#Sorted according to the width_threshold
def sortedBywidth(regions):
    new_regions = []
    width_threshold = findWidth(regions)

    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        curent_width = abs(maxc-minc)
        if  (curent_width>=width_threshold):
            new_regions.append(region)
    return new_regions

def findArea(regions):
     #Find the height according to the max value in the histogram
    max_val = 1000
    min_val = 8
    length_of_bin = 5
    areas =[]
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        current_hegiht = abs(maxr-minr)
        current_area = region.area
        if current_hegiht>min_val and current_hegiht<max_val:
            areas.append(current_area)
    [hist, bins] = np.histogram(areas, bins=range(int(round(min(areas), -1)), int(round(max(areas),-1)) + length_of_bin, length_of_bin))
    index = np.argmax(hist)
    height = bins[index+1]
    return height

def getPictureInTheimageRestrinct(regions, image):
    width_threshold = findWidth(regions)
    height_threshold = findHeight(regions)
    area_threshold = findArea(regions)
    regions_after_filters = []
    for region in regions:
        min_pic_r, min_pic_c, max_pic_r, max_pic_c = region.bbox
        cureent_he = max_pic_r - min_pic_r
        current_width = max_pic_c - min_pic_c
        if (height_threshold*1.5>cureent_he):
            continue
        regions_after_filters.append(region)
    return regions_after_filters

def getPictureInTheimage(regions, image):
    image_width = len(image[1])

    height_threshold = 50
    area_threshold = findArea(regions)
    regions_after_filters = []
    for region in regions:
        min_pic_r, min_pic_c, max_pic_r, max_pic_c = region.bbox
        cureent_he = max_pic_r - min_pic_r
        current_width = max_pic_c - min_pic_c
        if (region.area < area_threshold *1.5 or height_threshold*1.5>cureent_he) or current_width > (image_width/2)*.8:
            continue
        regions_after_filters.append(region)
    return regions_after_filters

def filter_according_to_pic(regions, region_with_pics):
    region_in_pic = []
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        for picture in region_with_pics:
            min_pic_r, min_pic_c, max_pic_r, max_pic_c = picture.bbox
            #Its inside the picature
            if minr>min_pic_r and maxr< max_pic_r and minc>min_pic_c and maxc< max_pic_c:
                region_in_pic.append(region)
                break

    regions_without_pic = [region for region in regions if region not in region_in_pic]
    return regions_without_pic

def build_close_struct(size, second):
    struct = np.zeros((size,size))
    struct[(size//2), 0] = 1
    index = 1
    if second:
        index = 5
    for i in range(0,index):
        struct[(size//2)+i, 0] = 1
        struct[(size//2)-i, 0] = 1
    return struct

def segmentToRegions(image, num_of_ones, bw, second = False):
    #apply threshold
    struct = build_struct(30, num_of_ones)
    img_close =opening(bw, struct)
    struct  =build_close_struct(30, second)
    img_close =  ndimage.binary_closing(img_close, structure=struct).astype(img_close.dtype)
    if (num_of_ones == 0):
        img_close = bw
    # remove artifacts connected to image border
    cleared = img_close.copy()
    #clear_border(cleared)
    # label image regions
    label_image = label(cleared)
    borders = np.logical_xor(img_close, cleared)
    label_image[borders] = -1
    regions = regionprops(label_image)
    if (not second):
        regions_sec = segmentToRegions(image, num_of_ones, bw, second= True)
    else:
        regions_sec =set()
    regions = set(regions)
    regions_sec = set(regions_sec)
    regions = regions.union(regions_sec)
    """
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(image_label_overlay)
    #ax.imshow(img_close)


    for region in regions:
        # skip small images
        if region.area < 0:
            continue
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox

        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)

    plt.show()
    """
    return regions

def removeRegionsFromImage(image, regions):
    thresh = threshold_otsu(image)
    bw = np.any(image > thresh, -1)
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        bw[minr:maxr, minc:maxc]=1
    return bw

def removePic(regions, image):
    image_heighet = len(image)
    image_width = len(image[1])
    factor = 2
    threshold = 100
    regions_with_pic = []
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        current_height = abs(maxr-minr)
        current_width = abs(maxc-minc)
        if (current_height> threshold and current_width< image_width/2 and current_height<image_heighet/2):
            regions_with_pic.append(region)

    region_wihtout_pic = [region for region in regions if region not in regions_with_pic]
    return region_wihtout_pic

def plotImage(regions, image, num = 0):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))
    ax.imshow(image)
    index = 0
    for row in regions:
        global_minr=len(image)
        global_minc=len(image[0])
        global_maxr=-1
        global_maxc=-1
        if num!=0 and index>num:
                break
        index +=1
        if (isinstance(row, skimage.measure._regionprops._RegionProperties)):
            row = [row]
        for region in row:
            # skip small images
            if region.area < 2:
                continue
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            global_minr = min(global_minr, minr)
            global_minc = min(global_minc, minc)
            global_maxr = max(global_maxr, maxr)
            global_maxc = max(global_maxc, maxc)

        rect = mpatches.Rectangle((global_minc, global_minr), global_maxc - global_minc, global_maxr - global_minr,
                                  fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)

def needToInsert(region, max_index_row, min_index_row, max_index_col, min_index_col, printT = True ):
     minr, minc, maxr, maxc = region.bbox
     spces = 40
     if (minc+spces >= min_index_col and maxc <= max_index_col + spces):
         if (max_index_row >= minr and  min_index_row <= minr ):
            return True
     return False

def getThreshold(regions, image,width_threshold ):
    if (len(regions)==0):
        return 0,0
    regions_sorted_by_cols = sorted(regions, key=lambda x:  x.bbox[3], reverse=True)
    most_right_region = regions_sorted_by_cols[0]
    right_index = most_right_region.bbox[3]
    left_index = most_right_region.bbox[1]
    for region in regions_sorted_by_cols:
         minr, minc, maxr, maxc = region.bbox
         if (minc <= left_index and maxc >= left_index):
             left_index = minc

    min_index = left_index
    max_index = right_index
    return min_index, max_index

def findOtherRegionsToInsert(regions, region, min_index_col, max_index_col):
    regions_in_row = []
    minr, minc, maxr, maxc = region.bbox
    for region_t in regions:
        if needToInsert(region_t, maxr,minr,  max_index_col,min_index_col, True):
            regions_in_row.append(region_t)
    regions_by_row = sorted(regions_in_row, key=lambda x:  x.bbox[3])
    return regions_by_row

def filtred_rows_by_height(row):
     row= sorted(row, key=lambda x:  x.bbox[0])
     min_row = row[0].bbox[0]
     max_row = row[0].bbox[2]
     buffer_space =  (max_row-min_row) / 3
     region_in_row = []
     for region in row:
        minr, minc, maxr, maxc = region.bbox
        if (maxr<max_row + buffer_space):
            region_in_row.append(region)
     return  region_in_row

def get_regions_end(row):
    return [region for region in row ]

def check_cols(last_rows):
    #goes over the rows and check their ending points
    if len(last_rows)<3:
        return last_rows
    need_to_split = True
    max_region = 0
    min_region = 0
    for i in range(1,3):
        current_row = last_rows[-i]
        last_row = last_rows[-i-1]
        num_of_cols, max_region, min_region = get_num_of_cols(current_row, last_row, min_region, max_region)
        if (num_of_cols<=1):
            need_to_split = False
    if (need_to_split):
        print "Split!!!!!!!!!!!"
        for i in range(1,4):
            current_row = last_rows[-i]
            update_row = []
            for region in current_row:
                minr, minc, maxr, maxc = region.bbox
                if (minc>=min_region -5):
                    update_row.append(region)
                    print "Insert___"
            last_rows[-i] = update_row
    return last_rows


def get_num_of_cols(current_row, last_row, min_region, max_region):
    buffer_size = 20
    current_row = sorted(current_row, key=lambda x:  x.bbox[3], reverse=True)
    last_row = sorted(last_row, key=lambda x:  x.bbox[3], reverse=True)
    for index, region in enumerate(current_row):
        minr, minc, maxr, maxc = region.bbox
        for index_scond, region_second  in enumerate(last_row):
            minr_second, minc_seoncd, maxr_second, maxc_second = region_second.bbox
            if (len(last_row)>index_scond+1 and len(current_row)>index+1 and minc+buffer_size>= minc_seoncd and minc-buffer_size<=minc_seoncd):
                next_maxc = current_row[index+1].bbox[3]
                next_maxc_second = last_row[index_scond+1].bbox[3]
                if (next_maxc + buffer_size>=next_maxc_second and next_maxc - buffer_size<= next_maxc_second ):
                    if (max_region==0 or next_maxc + buffer_size>=max_region and next_maxc - buffer_size<= max_region ):
                        return 2, next_maxc, minc
    return 1, -1, -1

def delete_regions(regions, regions_to_insert):
    regions_to_insert = [region for region in regions_to_insert if region not in regions]
    return regions_to_insert

def get_plat_list(regions):
    plat_list = []
    for row in regions:
        for region in row:
            plat_list.append(region);
    return plat_list

def orderRows(regions, image):
    regions = set(regions)
    width_threshold = findWidth(regions)
    #Todo-change it
    sorted_region = []
    regions_to_insert = sorted(regions, key=lambda x:  x.bbox[0])
    height = len(image)
    while( len(regions_to_insert)!=0):
        for index_row in range(0,height):
            min_index, max_index = getThreshold(regions_to_insert, image, width_threshold)
            regions_already_inside =[]
            region_to_insert_update = regions_to_insert
            for region in regions_to_insert:
                if (region not in regions_already_inside and needToInsert(region, index_row, index_row - 2, max_index, min_index)):
                    current_row =[]
                    current_row.append(region)
                    region_in_the_row = findOtherRegionsToInsert(regions_to_insert, region, min_index,max_index )
                    current_row.extend( region_in_the_row)
                    current_row = filtred_rows_by_height(current_row)
                    current_row = sorted(current_row, key=lambda x:  x.bbox[3], reverse=True)
                    sorted_region.append(current_row)
                    #sorted_region = check_cols(sorted_region)
                    regions_already_inside = get_plat_list(sorted_region)
                    region_to_insert_update =[region for region in regions_to_insert if region not in regions_already_inside]
            regions_to_insert = region_to_insert_update
    return sorted_region

#Find all the rows and Order them
def segmentRows(image):
    thresh = threshold_otsu(image)
    bw = np.any(image > thresh, -1)
    print 1
    regions = segmentToRegions(image, 0, bw);

    print 2
    region_with_pics = getPictureInTheimage(regions, image)

    print 3
    bw = removeRegionsFromImage(image, region_with_pics)
    print 4
    regions = segmentToRegions(image, 25, bw)
    #plotImage(regions, image)

    #regions_withoutWidth = sortedBywidth(regions)
    regions_1 = sortedByHeight(regions)
    #plotImage(regions_1, image)

    print "Here",  len(regions_1)
    region_with_pics = getPictureInTheimageRestrinct(regions_1, image)
    region_after_filter = filter_according_to_pic(regions_1, region_with_pics)
    plotImage(region_after_filter, image)

    rows = orderRows(region_after_filter, image)
    plotImage(rows, image)
    print 5
    #Plot the rows in the rihgt order

    for i in range(1,len(rows)):
        pass
        #plotImage(rows, image, i)
        #plt.show()

    plt.show()
    #width_threshold = findWidth(regions)

    return rows

def segment_letters_from_row(row, image):
    letters_in_row = []
    for region in row:
        minr, minc, maxr, maxc = region.bbox
        new_image = image[minr-5:maxr+5,minc-5:maxc+5]
        thresh = threshold_otsu(new_image)
        new_image = image.copy()

        new_image[0:minr, 0:minc] = 0
        new_image[0:minr-5, minc:maxc] = 0
        new_image[0:minr, maxc:len(image[0])] = 0

        new_image[minr:maxr, 0:minc] = 0
        new_image[minr:maxr, maxc+5:len(image[0])] = 0

        new_image[maxr:len(image), 0:minc] = 0
        new_image[maxr+5:len(image), minc: maxc] = 0
        new_image[maxr:len(image), maxc:len(image[0])] = 0

        bw = np.any(new_image > thresh, -1)
        # remove artifacts connected to image border
        cleared = bw.copy()
        clear_border(cleared)
        # label image regions
        label_image = label(cleared)
        borders = np.logical_xor(bw, cleared)
        label_image[borders] = -1
        for letter in regionprops(label_image):
            if letter.area < 5:
                continue
            letters_in_row.append(letter)
    print "Finise"
    return letters_in_row

#For each row extract the letters
def segmentLetters(rows, image):
    words = []
    for row in rows:
        words_in_row = segment_letters_from_row(row, image)
        words.extend(words_in_row)
    return words

def inRegion(letter, region):
    minr, minc, maxr, maxc = region.bbox
    minr_letter, minc_letter, maxr_letter, maxc_letter = letter.bbox
    buffer_val = 5
    if (minc_letter + buffer_val >= minc and maxc_letter - buffer_val <= maxc and minr_letter + buffer_val >= minr and maxr_letter - buffer_val <= maxr ):
        return True
    return  False

def calLetters(image):
    # apply threshold
    thresh = threshold_otsu(image)
    bw = np.any(image > thresh, -1)
    #bw = ndimage.binary_opening(bw, structure= np.zeros((3,3))).astype(bw.dtype)
    # remove artifacts connected to image border
    cleared = bw.copy()
    clear_border(cleared)
    # label image regions
    label_image = label(cleared)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1
    image_label_overlay = label2rgb(label_image, image=image)
    letters = regionprops(label_image)
    return letters

def sorte_letters_in_region(letters):
    sorted_letters = []
    next_line_letters =[]
    letters_sorted_by_cols = sorted(letters, key=lambda x:  x.bbox[3])
    higher_letter = sorted(letters, key=lambda x:  x.bbox[0])[0]
    min_height = higher_letter.bbox[0]
    max_height = higher_letter.bbox[2]
    for letter in letters_sorted_by_cols:
        if (letter.bbox[0]<=max_height):
            sorted_letters.append(letter)
        else:
            next_line_letters.append(letter)
    sorted_letters.extend(next_line_letters)
    return  sorted_letters


def segment_letters_fast(rows, image, letters = None, old_letters = 0):
    words = []
    if (letters == None):
        letters = calLetters(image)

    i = 0
    for row in rows:
        letters_in_row = []
        for region in row:
            print "Region", i
            i+=1
            lettersToRemove =[]
            for letter in letters:
                if (inRegion(letter, region)):
                    letters_in_row.append(letter)
                    lettersToRemove.append(letter)
            letters = [letter for letter in letters if letter not in lettersToRemove]
        letters_in_region_sort = sorte_letters_in_region(letters_in_row)
        #letters_in_region_sort = sorted(letters_in_row, key=lambda x:  x.bbox[3], reverse=True)
        words.extend(letters_in_region_sort)
    if (len(letters)>0 and len(letters)!= old_letters):
        print "New Round!!!!!"
        segment_letters_fast(rows, image, letters, len(letters))
    return words , letters

def OCRWith_SVM(letters):
    images = [letter['image'] for letter in letters]
    images_resize =[resize(letter, (40, 15)) for letter in images]
    data = []
    for image in images_resize:
        reshape_image = image.reshape((1,-1))
        data.extend(reshape_image)
    #Load from dataSetMerge

    y_pred = getSVMPerdit(data)
    return y_pred

#order all the lteers
def imageSegmentation(image_name):
    #Read image
    img_data = cv2.imread(image_name)
    #im = down_sample(img_data)  # make it smaller to avoid out of memory. TODO: later on we can skip it?
    im  = img_data
    #The rows are in the right order
    rows = segmentRows(im)
    #The words are in the right order
    #letters = segmentLetters(rows, im)
    """
    letters, letters_remain = segment_letters_fast(rows, im)
    y_pred = OCRWith_SVM(letters)
    for i in range(1,20):
        plt.imshow(letters[i]['image'])
        print y_pred[i]
    """
    plt.show()

if __name__ == '__main__':
    image_name = 'train_data/wikiText15.jpg'
    imageSegmentation(image_name)
    plt.show()