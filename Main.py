import os
import cv2
import numpy as np
from Plot_Result import plotConvResults, Plot_ROC_Curve, Plots_Results, Plot_Proposed_Results, Table, Confusion

# Read Dataset 1
an = 0
if an == 1:
    Path = './Datasets/Dataset1/Data/'
    Image = []
    Target = []
    in_dir = os.listdir(Path)
    for i in range(len(in_dir)):
        path1 = Path + '/' + in_dir[i]
        sub_dir1 = os.listdir(path1)
        for j in range(488):  # len(sub_dir1)
            path2 = path1 + '/' + sub_dir1[j]
            img = cv2.imread(path2)
            img = cv2.resize(img, (512, 512))
            img = np.uint8(img)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)

            Tar = i
            Image.append(img)
            Target.append(Tar)
            print(i, j)

    Images = np.asarray(Image)
    Target = np.asarray(Target)
    uniq = np.unique(Target)
    target = np.zeros((Target.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Target == uniq[uni])
        target[index[0], uni] = 1
    index = np.arange(len(Images))
    np.random.shuffle(index)
    Org_Img = np.asarray(Images)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = target[index]
    np.save('Index_1.npy', index)
    np.save('Images_1.npy', Shuffled_Datas)
    np.save('Target_1.npy', Shuffled_Target)

# Read Dataset 2
an = 0
if an == 1:
    Path = './Datasets/Dataset2/OriginalDataset/'
    Image = []
    Target = []
    in_dir = os.listdir(Path)
    for i in range(len(in_dir)):
        path1 = Path + '/' + in_dir[i]
        sub_dir1 = os.listdir(path1)
        for j in range(len(sub_dir1)):
            path2 = path1 + '/' + sub_dir1[j]
            img = cv2.imread(path2)
            img = cv2.resize(img, (512, 512))
            img = np.uint8(img)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)

            Tar = i
            Image.append(img)
            Target.append(Tar)
            print(i, j)

    Images = np.asarray(Image)
    Target = np.asarray(Target)
    uniq = np.unique(Target)
    target = np.zeros((Target.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Target == uniq[uni])
        target[index[0], uni] = 1
    index = np.arange(len(Images))
    np.random.shuffle(index)
    Org_Img = np.asarray(Images)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = target[index]
    np.save('Index_2.npy', index)
    np.save('Images_2.npy', Shuffled_Datas)
    np.save('Target_2.npy', Shuffled_Target)

# Read Dataset3
an = 0
if an == 1:
    Path = './Datasets/Dataset3/combined_images'
    Image = []
    Target = []
    in_dir = os.listdir(Path)
    for i in range(len(in_dir)):
        path1 = Path + '/' + in_dir[i]
        sub_dir1 = os.listdir(path1)
        for j in range(800):  # len(sub_dir1)
            path2 = path1 + '/' + sub_dir1[j]
            img = cv2.imread(path2)
            img = cv2.resize(img, (512, 512))
            img = np.uint8(img)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)

            Tar = i
            Image.append(img)
            Target.append(Tar)
            print(i, j)

    Images = np.asarray(Image)
    Target = np.asarray(Target)
    uniq = np.unique(Target)
    target = np.zeros((Target.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Target == uniq[uni])
        target[index[0], uni] = 1
    index = np.arange(len(Images))
    np.random.shuffle(index)
    Org_Img = np.asarray(Images)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = target[index]
    np.save('Index_3.npy', index)
    np.save('Images_3.npy', Shuffled_Datas)
    np.save('Target_3.npy', Shuffled_Target)

plotConvResults()
Confusion()
Plot_ROC_Curve()
Plots_Results()
Plot_Proposed_Results()
Table()
