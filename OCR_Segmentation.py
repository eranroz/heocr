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

def opening(mask, struc):
  return ndimage.binary_opening(mask, structure=struc).astype(mask.dtype)


#Build struct (matrix) with the give size and the given num of ones
def buildStruct(size, num_of_ones):
    struc = np.zeros((30,30))
    for i in range(0,num_of_ones):
        struc[i] = 1

    return struc

def findHeight(regions):
    #Find the height according to the max value in the histogram
    max_val = 1000
    min_val = 10
    length_of_bin = 10
    heights =[]
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        current_hegiht = abs(maxr-minr)
        if current_hegiht>min_val and current_hegiht<max_val:
            heights.append(current_hegiht)
    [hist, bins] = np.histogram(heights, bins=range(min(heights), max(heights) + length_of_bin, length_of_bin))
    index = np.argmax(hist)
    height = bins[index+1]
    return height

def sortedByHeight(regions, regions_withoutWidth):
    new_regions = []
    height = findHeight(regions)
    factor = 0.5
    height_max = height + height*factor
    height_min = height - height*factor
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        curent_height = abs(maxr-minr)
        if  (curent_height<=height_max and curent_height>=height_min):
            new_regions.append(region)
    return new_regions

def findWidth(regions, num=0):
    #Find the height according to the max value in the histogram
    max_val = 100000
    min_val = 10
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
    min_val = 10
    length_of_bin = 10
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
    width_threshold = findWidth(regions)
    height_threshold = findHeight(regions)
    image_height = len(image)
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

def segmentToRegions(image, num_of_ones, bw):
    #apply threshold
    struct = buildStruct(30, num_of_ones)
    img_close =opening(bw, struct)
    if (num_of_ones == 0):
        img_close = bw
    # remove artifacts connected to image border
    cleared = img_close.copy()
    clear_border(cleared)
    # label image regions
    label_image = label(cleared)
    borders = np.logical_xor(img_close, cleared)
    label_image[borders] = -1
    image_label_overlay = label2rgb(label_image, image=image)
    regions = regionprops(label_image)
    """
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(image_label_overlay)
    for region in regions:
        # skip small images
        if region.area < 10:
            continue
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox

        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
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
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
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

        for region in row:
            # skip small images
            if region.area < 10:
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
     spces = 50
     if (minc+spces >= min_index_col and maxc <= max_index_col + spces):
         if (max_index_row >= minr and  min_index_row <= minr ):
            if printT:
                print "Good Region", minr, minc, maxr, maxc,  minr , min_index_col, max_index_col, min_index_row, max_index_row
            return True
     return False

def getThreshold(regions, image,width_threshold ):
    regions_sorted_by_cols = sorted(regions, key=lambda x:  x.bbox[3], reverse=True)
    most_right_region = regions_sorted_by_cols[0]
    right_index = most_right_region.bbox[3]
    left_index = most_right_region.bbox[1]
    for region in regions_sorted_by_cols:
         minr, minc, maxr, maxc = region.bbox
         if (minc <= left_index and maxc >= left_index):
             left_index = minc
    """
    image_height = len(image)
    image_width = len(image[0])
    startCols = sorted( [region.bbox[3] for region in regions])
    #If we got somthing bigger then helf of the page try the next common width
    if (image_width/2< width_threshold):
        pass
        #width_threshold = findWidth(regions,1)

    max_index =startCols[-1]
    min_index = min([region.bbox[1] for region in regions_sorted_by_cols[0:2]])
    #min_index = max_index -width_threshold
    """
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

def orderRows(regions, image):
    width_threshold = findWidth(regions)
    height_threshold = findHeight(regions)
    #Todo-change it
    sorted_region = []
    regions_to_insert = sorted(regions, key=lambda x:  x.bbox[0])
    max_index =0
    height_row = 60
    min_index = len(image[0])
    height = len(image)
    current_row = []
    rowse = [region.bbox for region in regions_to_insert]
    print rowse[0:6]
    while( len(regions_to_insert)!=0):
        min_index, max_index = getThreshold(regions_to_insert, image, width_threshold)
        print "startand End indexs",min_index, max_index
        for index_row in range(0,height):
            for region in regions_to_insert:
                #print index_row, region.bbox[0]
                if (needToInsert(region, index_row, index_row - 2, max_index, min_index)):
                    current_row =[]
                    current_row.append(region)
                    regions_to_insert.remove(region)
                    current_row.extend( findOtherRegionsToInsert(regions_to_insert, region, min_index,max_index ))
                    current_row = sorted(current_row, key=lambda x:  x.bbox[3], reverse=True)
                    #sorted_region.append(region)
                    print "Row: "
                    for row in current_row:
                        print row.bbox[1],row.bbox[3]
                    print "END"

                    sorted_region.append(current_row)
                    #regions_to_insert.remove(region)
                    regions_to_insert = [region for region in regions_to_insert if region not in current_row]

    return sorted_region

#Find all the rows and Order them
def segmentRows(image):
    thresh = threshold_otsu(image)
    bw = np.any(image > thresh, -1)
    print 1
    regions = segmentToRegions(image, 0, bw)
    print 2
    region_with_pics = getPictureInTheimage(regions, image)
    print 3
    bw = removeRegionsFromImage(image, region_with_pics)
    print 4
    plt.imshow(bw)
    regions = segmentToRegions(image, 6, bw)
    #regions_withoutWidth = sortedBywidth(regions)
    regions_1 = sortedByHeight(regions, [])
    region_with_pics = getPictureInTheimageRestrinct(regions_1, image)
    region_after_filter = filter_according_to_pic(regions_1, region_with_pics)
    rows = orderRows(region_after_filter, image)
    plotImage(rows, image)
    print 5
    #Plot the rows in the rihgt order
    for i in range(1,len(rows),2):
        pass
        #plotImage(rows, image, i)
    #width_threshold = findWidth(regions)
    return rows

def segmentLettersFromRow(row, image):
    letters_in_row = []
    for region in row:
        print "Row!!!!!!!!!!!!!!!!!!!a"
        minr, minc, maxr, maxc = region.bbox
        new_image = image[minr-5:maxr+5,minc-5:maxc+5]
        #new_image = image
        #new_image[0:len(image), 0:len(image[0])]
        #new_image[1:minr]
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
        #bw = ndimage.binary_opening(bw, structure= np.zeros((3,3))).astype(bw.dtype)
        # remove artifacts connected to image border
        cleared = bw.copy()
        clear_border(cleared)
        # label image regions
        label_image = label(cleared)
        borders = np.logical_xor(bw, cleared)
        label_image[borders] = -1
        image_label_overlay = label2rgb(label_image, image=new_image)
        #fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        #ax.imshow(label_image)
        for letter in regionprops(label_image):
            if letter.area < 5:
                continue
            # draw rectangle around segmented coins
            #minr1, minc1, maxr1, maxc1 = letter.bbox
            #roi = image[ minr: maxr, minc:maxc]
            #plt.figure()
            #plt.imshow(roi)
            """
            rect = mpatches.Rectangle((minc1, minr1), maxc1 - minc1, maxr1 - minr1,
                                      fill=False, edgecolor='red', linewidth=2)
            """
            #ax.add_patch(rect)
            letters_in_row.append(letter)
    print "Finise"
    return letters_in_row

#For each row extract the letters
def segmentLetters(rows, image):
    words = []
    for row in rows:
        words_in_row = segmentLettersFromRow(row, image)
        words.extend(words_in_row)
    return words

def inRegion(letter, region):
    minr, minc, maxr, maxc = region.bbox
    minr_letter, minc_letter, maxr_letter, maxc_letter = letter.bbox
    buffer_val = 100
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

def segmentLettersFast(rows, image, letters = None, old_letters = 0):
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
        letters_in_region_sort = sorted(letters_in_row, key=lambda x:  x.bbox[3], reverse=True)
        words.extend(letters_in_region_sort)
    print len(letters)
    if (len(letters)>0 and len(letters)!= old_letters):
        print "New Round!!!!!"
        segmentLettersFast(rows, image, letters, len(letters))
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
    im = down_sample(img_data)  # make it smaller to avoid out of memory. TODO: later on we can skip it?
    #The rows are in the right order
    rows = segmentRows(im)
    #The words are in the right order
    #letters = segmentLetters(rows, im)
    letters, letters_remain = segmentLettersFast(rows, im)
    y_pred = OCRWith_SVM(letters)
    for i in range(1,20):
        plt.imshow(letters[i]['image'])
        print y_pred[i]
    plt.show()

if __name__ == '__main__':
    image_name = 'train_data/alice/0073.jpg'
    imageSegmentation(image_name)