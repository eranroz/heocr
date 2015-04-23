# -*- coding: utf-8 -*-
__author__ = 'ravidziv'
import IPython
import sklearn as sk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Tkinter import *
import sys
from skimage.transform import resize
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
import pickle
from sklearn import svm
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

class mainWindow(object):
    def __init__(self,master):
        self.master=master
        self.labeled = 0
        def onclick(event=None):
            self.labeled = self.e1.get()
            self.e1.delete(0, END)
            master.quit()

        master.title("Labeling")
        Label(master, text="Insert Label").grid(row=0)
        self.e1 = Entry(master)
        self.e1.grid(row=0, column=1)
        self.finish = False
        Button(master, text='OK', command=onclick).grid(row=1, column=0, sticky=W, pady=4)
        def func(event):
            print("You hit return.")



        master.bind('<Return>', onclick)

        def exitParams():
            self.finish = True
            master.quit()
        Button(master, text='Exit', command=exitParams).grid(row=1, column=1, sticky=W, pady=4)

    def entryValue(self):
        return self.labeled

class PersonalLetters():

    def __init__(self):
        self.target = []
        self.data = []
        self.images = []

    def addImage(self, image, tragetImg):
        self.target.append(tragetImg)
        self.images.append(image)
        reshape_image = image.reshape((1,-1))
        self.data.extend(reshape_image)

    def merge(self, Personal_letters):
        self.target.extend(Personal_letters.target)
        self.images.extend(Personal_letters.images)
        self.data.extend(Personal_letters.data)

def trainSVM(X, y, file_name = 'objs.pickle'):
    clf = svm.SVC()
    clf.fit(X, y)
    with open(file_name, 'w') as f:
        pickle.dump(clf, f)

def loadSVM(file_name = 'objs.pickle'):
    with open(file_name) as f:
        clf = pickle.load(f)
        return  clf


class popupWindow(object):
    def __init__(self,master):
        top=self.top=Toplevel(master)
        self.l=Label(top,text="Hello World")
        self.l.pack()
        self.e=Entry(top)
        self.e.pack()
        self.b=Button(top,text='Ok',command=self.cleanup)
        self.b.pack()


    def cleanup(self):
        self.value=self.e.get()

        self.top.destroy()


def getLetters(image):
    # apply threshold
    thresh = threshold_otsu(image)
    bw = np.any(image > thresh, -1)
    # remove artifacts connected to image border
    cleared = bw.copy()
    clear_border(cleared)

    # label image regions
    label_image = label(cleared)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1
    image_label_overlay = label2rgb(label_image, image=image)
    letters_region =  regionprops(label_image)
    letters = []
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(image_label_overlay)
    i=0
    Personal_letters= PersonalLetters()
    for region in sorted(letters_region, key=lambda x:  x.bbox[3], reverse=True):
        minr, minc, maxr, maxc = region.bbox

        # skip small images
        if (region.area < 15 or maxr-minr<15 or maxc-minc<15 or region.area>2000) :
            continue
        # draw rectangle around segmented coins
        letter = region['image']
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        letter = resize(letter, (40, 15))
        plt.figure()
        plt.imshow(letter)
        plt.waitforbuttonpress()
        if (i==0):
            root=Tk()
            m=mainWindow(root)
        root.mainloop()
        i=i+1
        print "M: ", m.entryValue()
        #letter_label = raw_input("Insert the letters of the image")
        letter_label=m.entryValue()
        plt.close()
        if (letter_label == "-1" or m.finish == True):
            break
        if (letter_label =="0" or letter_label =="" ):
            continue
        print len(letter), len(letter[0])
        Personal_letters.addImage(letter,letter_label)
        ax.add_patch(rect)
    return Personal_letters

def print_letters(images, target, top_n):
    # set up the figure size in inches
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(top_n):
        # plot the images in a matrix of 20x20
        p = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
        # label the image with the target value
        print target[i].decode('utf-8')
        p.text(0, 50, "i_" + str(target[i]).decode('utf-8') )
        #p.text(0, 60, str(i))

def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold croos validation iterator
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print scores
    print ("Mean score: {0:.3f} (+/-{1:.3f})").format(
        np.mean(scores), sem(scores))

def train_and_evaluate(clf, X_train, X_test, y_train, y_test):

    clf.fit(X_train, y_train)

    print "Accuracy on training set:"
    print clf.score(X_train, y_train)
    print "Accuracy on testing set:"
    print clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    print "Classification Report:"
    print metrics.classification_report(y_test, y_pred)
    print "Confusion Matrix:"
    print metrics.confusion_matrix(y_test, y_pred)

def saveData(obj, file_name= 'data_set'):
    with open(file_name, 'w') as f:
        pickle.dump(obj, f)

def loadData(file_name = 'data_set' ):
     with open(file_name) as f:
        obj = pickle.load(f)
        return obj
def labeledLetters(image_name):
    #Read image
    img_data = cv2.imread(image_name)
    img = down_sample(img_data)
    Personal_letters = getLetters(img)
    saveData(Personal_letters, file_name=image_name.split('.')[0])

def trainSVMWithLetters(data_to_load):
    Personal_letters = loadData(data_to_load)
    svc_1 = SVC(kernel='linear')
    X_train, X_test, y_train, y_test = train_test_split(
        Personal_letters.data, Personal_letters.target, test_size=0.25, random_state=0)
    svc_2 = SVC(kernel='linear')
    print  "Before Cross"
    evaluate_cross_validation(svc_2, X_train, y_train, 2)
    train_and_evaluate(svc_2, X_train, X_test, y_train, y_test)
    print "Finish"
    plt.show()

def mergeDataSet(datasets, new_file_name = 'dataSetMerge'):
    letters = PersonalLetters()
    for dataset_path in datasets:
        Personal_letters = loadData(dataset_path)
        letters.merge(Personal_letters)
    saveData(letters, new_file_name)

def getSVMPerdit(X_test, data_name = 'dataSetMerge'):
     print "Perdit:"
     Personal_letters = loadData(data_name)
     X_train = Personal_letters.data
     y_train = Personal_letters.target
     clf = SVC(kernel='linear')
     clf.fit(X_train, y_train)
     y_pred = clf.predict(X_test)
     return y_pred

if __name__ == '__main__':
    datasets = ['train_data/alice/0079', 'train_data/alice/0080']
    #mergeDataSet(datasets, new_file_name = 'dataSetMerge')
    #labeledLetters('train_data/alice/0080.jpg')
    #Personal_letters_test = loadData('dataSetMerge')
    #Personal_letters = loadData('train_data/alice/0079')
    #y_pred = getSVMPerdit(Personal_letters_test.data, data_name = 'train_data/alice/0079')
    #print metrics.classification_report(Personal_letters_test.target,  y_pred)
    #print_letters(Personal_letters.images, Personal_letters.target, 7)
    #plt.show()
    #trainSVMWithLetters('train_data/alice/0079')
