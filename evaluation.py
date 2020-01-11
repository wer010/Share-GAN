import os
import numpy as np
import scipy as scp
import imageio
import ntpath
import time
import cv2 as cv
from options.test_options import TestOptions
from PIL import Image

def evalExp(gtBin, cur_prob, thres, validMap=None, validArea=None):
    '''
    Does the basic pixel based evaluation!
    :param gtBin:
    :param cur_prob:
    :param thres:
    :param validMap:
    '''
    # print(len(cur_prob.shape))
    # print(len(gtBin.shape))
    assert len(cur_prob.shape) == 2, 'Wrong size of input prob map'
    assert len(gtBin.shape) == 2, 'Wrong size of input prob map'
    thresInf = np.concatenate(([-np.Inf], thres, [np.Inf]))

    # Merge validMap with validArea
    if validMap is not None:
        if validArea is not None:
            validMap = (validMap == True) & (validArea == True)
    elif validArea is not None:
        validMap = validArea

    # histogram of false negatives
    if validMap is not None:
        fnArray = cur_prob[(gtBin == True) & (validMap == True)]
    else:
        fnArray = cur_prob[(gtBin == True)]
    fnHist = np.histogram(fnArray, bins=thresInf)[0]
    fnCum = np.cumsum(fnHist)
    FN = fnCum[0:0 + len(thres)];

    if validMap is not None:
        fpArray = cur_prob[(gtBin == False) & (validMap == True)]
    else:
        fpArray = cur_prob[(gtBin == False)]

    fpHist = np.histogram(fpArray, bins=thresInf)[0]
    fpCum = np.flipud(np.cumsum(np.flipud(fpHist)))
    FP = fpCum[1:1 + len(thres)]

    # count labels and protos
    # posNum = fnArray.shape[0]
    # negNum = fpArray.shape[0]
    if validMap is not None:
        posNum = np.sum((gtBin == True) & (validMap == True))
        negNum = np.sum((gtBin == False) & (validMap == True))
    else:
        posNum = np.sum(gtBin == True)
        negNum = np.sum(gtBin == False)
    return FN, FP, posNum, negNum


def eval_image(gt_image, cnn_image):
    thresh = np.array(range(0, 256))/255.0

    road_color = np.array([255, 0,255])
    background_color = np.array([255, 0, 0])
    gt_road = np.all(gt_image == road_color, axis=2)
    # print(len(gt_road.shape))
    # print()

    gt_bg = np.all(gt_image == background_color, axis=2)
    # print(len(gt_bg.shape))
    valid_gt = gt_road + gt_bg

    FN, FP, posNum, negNum = evalExp(gt_road, cnn_image, thresh, validMap=None, validArea=valid_gt)

    return FN, FP, posNum, negNum


def three2two(image):
    H,W,S = image.shape
    image_new = np.zeros((H,W),dtype=int)
    for i in range(H):
        for j in range(W):
            image_new[i,j] = image[i,j,2]

    return (image_new)




def zero_one(image):
    H,W = image.shape[:2]
    for i in range(H):
        for j in range(W):
            if image[i,j] > 155: image[i,j] = 255
            else:image[i, j] = 0
    return image

def count_pixel(image,thresh):
    H,W = image.shape[:2]
    zero = 0
    one = 0
    for i in range(H):
        for j in range(W):
            if image[i,j] > thresh: one = one + 1
            else: zero = zero + 1
    return zero, one


def accuracy(dirroot):
    imagelist = os.listdir(dirroot)
    image_root = dirroot + imagelist[1]
    image_zero = cv.imread(image_root,0)
    image_one = cv.imread(image_root,0)
    #print(image_one)
    for _, root in enumerate(imagelist):
        #print(root)
        if '_1.png'  in  root:
            print('loading the photo {}'.format(root))
            image_root = dirroot + root
        elif '_1.PNG' in root:
            print('loading the photo {}'.format(root))
            image_root = dirroot + root
        elif '_1.jpg' in root:
            print('loading the photo {}'.format(root))
            image_root = dirroot + root
        elif '_1.JPG' in root:
            print('loading the photo {}'.format(root))
            image_root = dirroot + root
        image_temp = cv.imread(image_root,0)
        #print(image_temp)
        H,W = image_temp.shape[:2]
        for i in range(H):
            for j in range(W):
                if image_one[i,j] < image_temp[i,j]: image_one[i,j] = image_temp[i,j]
                if image_zero[i,j] > image_temp[i,j]: image_zero[i,j] = image_temp[i,j]
        #print(image_temp)
    zero_0,one_0 = count_pixel(image_zero,156)
    zero_1, one_1 = count_pixel(image_one,156)
    return one_0/one_1


def acc(image_test,image_gost):
    Wt,Ht = image_test.shape[:2]
    Wg,Hg = image_gost.shape[:2]
    if Ht != Hg or Wt != Wg:
        fy = Wg / Wt
        fx = Hg / Ht
        image_test = cv.resize(image_test, None, fx=fx, fy=fy, interpolation=cv.INTER_CUBIC)
        # print('ERROR: Images is not the same size')
        # return
    image_test = zero_one(image_test)
    # image_gost = zero_one(image_gost)
    TP,FP,FN,TN = 0,0,0,0
    for i in range(Wg):
        for j in range(Hg):
            if image_test[i,j] == 255:
                if image_gost[i,j] == 255: TP = TP + 1
                else: FP = FP + 1
            else:
                if image_gost[i,j] == 255: FN = FN + 1
                else: TN = TN + 1
    sum_size = Wg * Hg
    sum_size = float(sum_size)
    return TP/sum_size, FP/sum_size, FN/sum_size, TN/sum_size

def reset(image, thres = 128):
    H,W,S = image.shape
    image_new = np.zeros((H,W,S),dtype=int)
    for i in range(H):
        for j in range(W):
            for z in range(S):
                if image[i,j,z] > thres:
                    image_new[i,j,z] = 255
                else: image_new[i,j,z] = 0
    return image_new

def culc_acc(real_image, fake_image):
    #print(real_image.shape)
    real_image = reset(real_image)
    fake_image = reset(fake_image)
    H,W,S = real_image.shape
    count = 0
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for i in range(H):
        for j in range(W):
            if (real_image[i,j,0] == 255 and real_image[i,j,1] == 0 and real_image[i,j,2] == 255) or \
                    (real_image[i,j,0] == 255 and real_image[i,j,1] == 0 and real_image[i,j,2] == 0):
                count += 1
                real = real_image[i,j,2]
                fake = fake_image[i,j,2]

                if real == 255 and fake == 255:
                    TP += 1
                elif real == 255 and fake == 0:
                    FN +=1
                elif real == 0 and fake ==255:
                    FP += 1
                elif real == 0 and fake == 0:
                    TN += 1

    return TP/count, FP/count, FN/count, TN/count



if __name__ == '__main__':

    opt = TestOptions().parse()
    if 'txt' not in opt.phase:
        opt.phase = opt.phase + '.txt'
    opt.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
    opt.AB_paths = open(opt.dir_AB).read().split('\n')  # get image paths

    #list = ['real_B','fake_B']
    #list = ['data_B','output']
    list = ['mask_B','mask_fake_B']
    sum_value =0.0
    for i,path in enumerate(opt.AB_paths):
        short_path = path.split()
        name = ntpath.basename(short_path[0])
        name = os.path.splitext(name)[0]
        # print(name)git

        val_dir = './results/experiment_name/branch/val.txt_latest_iter400/images/'

        name_A = '%s_%s.png' % (name, list[0])
        name_B = '%s_%s.png' % (name, list[1])
        path_A = os.path.join(val_dir,name_A)
        path_B = os.path.join(val_dir,name_B)

        real_image = Image.open(path_A).convert('RGB')
        fake_image = Image.open(path_B).convert('RGB')
        real_image = np.asarray(real_image)
        fake_image = np.asarray(fake_image)

        # #print(real_image[2])
        # real_image = three2two(real_image)
        # fake_image = three2two(fake_image)
        #
        #
        #
        #
        # # FN, FP, posNum, negNum = eval_image(real_image, fake_image)
        # # acc = posNum / (FP + posNum + negNum)
        #
        # TP, FP, FN, TN = acc(fake_image, real_image)
        # acc_value = TP / (TP + FP + FN)
        #
        # sum_value += acc_value
        #
        # print(name,':  FN:',FN, '\tFP:',FP,'\tTP:',TP,'\tTN:',TN,'\tacc',acc_value)
        TP,FP,FN,TN = culc_acc(real_image, fake_image)
        acc_value = TP / (TP + FP + FN)
        sum_value += acc_value
        print(name, ':   TP:%f\tFP:%f\tFN:%f\tTN:%f\tacc:%f' %(TP,FP,FN,TN,acc_value))
    print(sum_value / (i + 1))