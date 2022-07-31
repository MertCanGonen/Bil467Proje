from statistics import median
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math


#Reading the all images
bw = cv.imread("bw.jpg")
rgb = cv.imread("rgb.jpg")

def boxFilter(image, kernel, border):
    img = image
    (h, w) = image.shape[:2]
    output = np.zeros((h, w, 3), dtype = np.uint8)
    k = int((kernel - 1) / 2)
    if (border == "0"):
        img = cv.copyMakeBorder(image, k, k, k, k, cv.BORDER_CONSTANT)
    elif (border == "M"):
        img = cv.copyMakeBorder(image, k, k, k, k, cv.BORDER_REFLECT)
    arr = np.ones((kernel, kernel), dtype=int)
    for i in range(k, h+k):
        for j in range(k, w+k):
            top = i - k
            bottom = i + k + 1
            left = j - k
            right = j + k + 1
            tmp1 = 0
            tmp2 = 0
            tmp3 = 0
            for x in range(top, bottom):
                for y in range(left, right):
                    tmp1 = tmp1 + (int(img[x][y][0]) * int(arr[x-top][y-left]))
                    tmp2 = tmp2 + (int(img[x][y][1]) * int(arr[x-top][y-left]))
                    tmp3 = tmp3 + (int(img[x][y][2]) * int(arr[x-top][y-left]))
            tmp1 = int(tmp1 / kernel**2)
            tmp2 = int(tmp2 / kernel**2)
            tmp3 = int(tmp3 / kernel**2)
            pixel = [tmp1, tmp2, tmp3]
            output[i-k][j-k] = pixel
    return output
        
def gaussianBlur(image, kernel, sigma, border):
    arr = np.zeros((kernel, kernel))
    center = (kernel - 1) / 2
    value = 0
    for i in range(kernel):
        for j in range(kernel):
            r = abs(center-i)**2 + abs(center-j)**2
            v = math.exp(-(r / ( 2 * (sigma**2) )))
            arr[i][j] = v
            value = value + v
    for i in range(kernel):
        for j in range(kernel):
            arr[i][j] = arr[i][j] / value
    img = image
    k = int((kernel - 1) / 2)
    (h, w) = image.shape[:2]
    output = np.zeros((h, w, 3), dtype = np.uint8)
    if (border == "0"):
        img = cv.copyMakeBorder(image, k, k, k, k, cv.BORDER_CONSTANT)
    elif (border == "M"):
        img = cv.copyMakeBorder(image, k, k, k, k, cv.BORDER_REFLECT)
    for i in range(k, h+k):
        for j in range(k, w+k):
            top = i - k
            bottom = i + k + 1
            left = j - k
            right = j + k + 1
            tmp1 = 0
            tmp2 = 0
            tmp3 = 0
            for x in range(top, bottom):
                for y in range(left, right):
                    tmp1 = tmp1 + img[x][y][0] * arr[x-top][y-left]
                    tmp2 = tmp2 + img[x][y][1] * arr[x-top][y-left]
                    tmp3 = tmp3 + img[x][y][2] * arr[x-top][y-left]
            pixel = [tmp1, tmp2, tmp3]
            output[i-k][j-k] = pixel
    return output
    
def medianFilter(image, kernel, border):
    img = image
    (h, w) = image.shape[:2]
    k = int((kernel - 1) / 2)
    output = np.zeros((h, w, 3), dtype = np.uint8)
    if (border == "0"):
        img = cv.copyMakeBorder(image, k, k, k, k, cv.BORDER_CONSTANT)
    elif (border == "M"):
        img = cv.copyMakeBorder(image, k, k, k, k, cv.BORDER_REFLECT)
    for i in range(k, h+k):
        for j in range(k, w+k):
            top = i - k
            bottom = i + k + 1
            left = j - k
            right = j + k + 1
            l1 = []
            l2 = []
            l3 = []
            for x in range(top, bottom):
                for y in range(left, right):
                    l1.append(img[x][y][0])
                    l2.append(img[x][y][1])
                    l3.append(img[x][y][2])
            m1 = median(l1)
            m2 = median(l2)
            m3 = median(l3)
            pixel = [m1, m2, m3]
            output[i-k][j-k] = pixel
    return output

def createMask(image, blur):
    result = image - blur
    return result

def output(image, blurredImage, k):
    mask = createMask(image, blurredImage)
    cv.imshow("Mask", mask)
    output = image + (k * mask)
    if (k > 1):
        o = image + mask
        cv.imshow("Unsharp Masking", o)
        cv.imshow("High Boost Filtering", output)
    else:
        cv.imshow("Unsharp Masking", output)
    cv.waitKey(0)
    print()

#Example for output
"""
bwBoxW3k0p = boxFilter(bw, 3, "0")
output(bw, bwBoxW3k0p, 1)
"""

#Blurred images

"""
#Masks created by using box filter
 
bwBoxW3k0p = boxFilter(bw, 3, "0")

bwBoxW3kMp = boxFilter(bw, 3, "M")

rgbBoxW3k0p = boxFilter(rgb, 3, "0")

rgbBoxW3kMp = boxFilter(rgb, 3, "M")

bwBoxW5k0p = boxFilter(bw, 5, "0")

rgbBoxW5k0p = boxFilter(rgb, 5, "0")

#################################################################
#Masks created by using gaussian

bwGausW3k0ps1 = gaussianBlur(bw, 3, 1, "0")

bwGausW3kMps1 = gaussianBlur(bw, 3, 1, "M")

rgbGausW3k0ps1 = gaussianBlur(rgb, 3, 1, "0")

rgbGausW3kMps1 = gaussianBlur(rgb, 3, 1, "M")

bwGausW5k0ps1 = gaussianBlur(bw, 5, 1, "0")

rgbGausW5k0ps1 = gaussianBlur(rgb, 5, 1, "0")

bwGausW5k0ps02 = gaussianBlur(bw, 5, 0.2, "0")

rgbGausW5k0ps02 = gaussianBlur(rgb, 5, 0.2, "0")

###################################################################
#Masks created by using median filter

bwMedianW3k0p = medianFilter(bw, 5, "0")

bwMedianW3kMp = medianFilter(bw, 5, "M")

rgbMedianW5k0p = medianFilter(rgb, 5, "0")

rgbMedianW5kMp = medianFilter(rgb, 5, "M")

bwMedianW11k0p = medianFilter(bw, 11, "0")

rgbMedianW11k0p = medianFilter(rgb, 11, "0")

#############################################################
"""



                                                                #SAME OPERATIONS WITH OPENCV METHODS# 

"""
                                                                    #CREATING ALL BLURRED IMAGES
        #BOX FILTER
#0 padding
bw_pad0_bf1 = cv.blur(bw, (3,3), cv.BORDER_CONSTANT)
bw_pad0_bf2 = cv.blur(bw, (11,11), cv.BORDER_CONSTANT)
bw_pad0_bf3 = cv.blur(bw, (25,25), cv.BORDER_CONSTANT)

rgb_pad0_bf1 = cv.blur(rgb, (3,3), cv.BORDER_CONSTANT)
rgb_pad0_bf2 = cv.blur(rgb, (11,11), cv.BORDER_CONSTANT)
rgb_pad0_bf3 = cv.blur(rgb, (25,25), cv.BORDER_CONSTANT)

        #MIRROR PADDING
bw_padM_bf1 = cv.blur(bw, (3,3), cv.BORDER_REFLECT)
bw_padM_bf2 = cv.blur(bw, (11,11), cv.BORDER_REFLECT)
bw_padM_bf3 = cv.blur(bw, (25,25), cv.BORDER_REFLECT)

rgb_padM_bf1 = cv.blur(rgb, (3,3), cv.BORDER_REFLECT)
rgb_padM_bf2 = cv.blur(rgb, (11,11), cv.BORDER_REFLECT)
rgb_padM_bf3 = cv.blur(rgb, (25,25), cv.BORDER_REFLECT)

        #GAUSSIAN FILTER
#0 padding
bw_pad0_g1_1 = cv.GaussianBlur(bw, (3,3), 0, cv.BORDER_CONSTANT)
bw_pad0_g1_2 = cv.GaussianBlur(bw, (3,3), 1, cv.BORDER_CONSTANT)
bw_pad0_g2_1 = cv.GaussianBlur(bw, (5,5), 0, cv.BORDER_CONSTANT)
bw_pad0_g2_2 = cv.GaussianBlur(bw, (5,5), 1, cv.BORDER_CONSTANT)

rgb_pad0_g1_1 = cv.GaussianBlur(rgb, (3,3), 0, cv.BORDER_CONSTANT)
rgb_pad0_g1_2 = cv.GaussianBlur(rgb, (3,3), 1, cv.BORDER_CONSTANT)
rgb_pad0_g2_1 = cv.GaussianBlur(rgb, (5,5), 0,  cv.BORDER_CONSTANT)
rgb_pad0_g2_2 = cv.GaussianBlur(rgb, (5,5), 1, cv.BORDER_CONSTANT)

        #MIRROR PADDING
bw_padM_g1_1 = cv.GaussianBlur(bw, (3,3), 0, cv.BORDER_REFLECT)
bw_padM_g1_2 = cv.GaussianBlur(bw, (3,3), 1, cv.BORDER_REFLECT)
bw_padM_g2_1 = cv.GaussianBlur(bw, (5,5), 0, cv.BORDER_REFLECT)
bw_padM_g2_2 = cv.GaussianBlur(bw, (5,5), 1, cv.BORDER_REFLECT)

rgb_padM_g1_1 = cv.GaussianBlur(rgb, (3,3), 0, cv.BORDER_REFLECT)
rgb_padM_g1_2 = cv.GaussianBlur(rgb, (3,3), 1, cv.BORDER_REFLECT)
rgb_padM_g2_1 = cv.GaussianBlur(rgb, (5,5), 0,  cv.BORDER_REFLECT)
rgb_padM_g2_2 = cv.GaussianBlur(rgb, (5,5), 1, cv.BORDER_REFLECT)

        #MEDIAN FILTER
bw_m1 = cv.medianBlur(bw, 3)
bw_m2 = cv.medianBlur(bw, 11)
rgb_m1 = cv.medianBlur(rgb, 3)
rgb_m2 = cv.medianBlur(rgb, 11)

        #MASKS FOR B&W IMAGE
mask1_bw = createMask(bw, bw_pad0_bf1)
mask2_bw = createMask(bw, bw_pad0_bf2)
mask3_bw = createMask(bw, bw_pad0_bf3)
mask4_bw = createMask(bw, bw_padM_bf1)
mask5_bw = createMask(bw, bw_padM_bf2)
mask6_bw = createMask(bw, bw_padM_bf3)

mask7_bw = createMask(bw, bw_pad0_g1_1)
mask8_bw = createMask(bw, bw_pad0_g1_2)
mask9_bw = createMask(bw, bw_pad0_g2_1)
mask10_bw = createMask(bw, bw_pad0_g2_2)
mask11_bw = createMask(bw, bw_padM_g1_1)
mask12_bw = createMask(bw, bw_padM_g1_2)
mask13_bw = createMask(bw, bw_padM_g2_1)
mask14_bw = createMask(bw, bw_padM_g2_2)

mask15_bw = createMask(bw, bw_m1)
mask16_bw = createMask(bw, bw_m2)

        #MASKS FOR RGB IMAGE
mask1_rgb = createMask(rgb, rgb_pad0_bf1)
mask2_rgb = createMask(rgb, rgb_pad0_bf2)
mask3_rgb = createMask(rgb, rgb_pad0_bf3)
mask4_rgb = createMask(rgb, rgb_padM_bf1)
mask5_rgb = createMask(rgb, rgb_padM_bf2)
mask6_rgb = createMask(rgb, rgb_padM_bf3)

mask7_rgb = createMask(rgb, rgb_pad0_g1_1)
mask8_rgb = createMask(rgb, rgb_pad0_g1_2)
mask9_rgb = createMask(rgb, rgb_pad0_g2_1)
mask10_rgb = createMask(rgb, rgb_pad0_g2_2)
mask11_rgb = createMask(rgb, rgb_padM_g1_1)
mask12_rgb = createMask(rgb, rgb_padM_g1_2)
mask13_rgb = createMask(rgb, rgb_padM_g2_1)
mask14_rgb = createMask(rgb, rgb_padM_g2_2)

mask15_rgb = createMask(rgb, rgb_m1)
mask16_rgb = createMask(rgb, rgb_m2)

        #OUTPUT FOR B&W IMAGES
output1 = highboostFiltering(bw, mask1_bw, 1)
output2 = highboostFiltering(bw, mask2_bw, 1)
output3 = highboostFiltering(bw, mask3_bw, 1)
output4 = highboostFiltering(bw, mask4_bw, 1)
output5 = highboostFiltering(bw, mask5_bw, 1)
output6 = highboostFiltering(bw, mask6_bw, 1)
output7 = highboostFiltering(bw, mask7_bw, 1)
output8 = highboostFiltering(bw, mask8_bw, 1)
output9 = highboostFiltering(bw, mask9_bw, 1)
output10 = higboostFiltering(bw, mask10_bw, 1)
output11 = higboostFiltering(bw, mask11_bw, 1)
output12 = higboostFiltering(bw, mask12_bw, 1)
output13 = higboostFiltering(bw, mask13_bw, 1)
output14 = higboostFiltering(bw, mask14_bw, 1)
output15 = higboostFiltering(bw, mask15_bw, 1)
output16 = higboostFiltering(bw, mask16_bw, 1)

        #OUTPUT FOR RGB IMAGES
output1 = higboostFiltering(rgb, mask1_rgb, 1)
output2 = higboostFiltering(rgb, mask2_rgb, 1)
output3 = higboostFiltering(rgb, mask3_rgb, 1)
output4 = higboostFiltering(rgb, mask4_rgb, 1)
output5 = higboostFiltering(rgb, mask5_rgb, 1)
output6 = higboostFiltering(rgb, mask6_rgb, 1)
output7 = higboostFiltering(rgb, mask7_rgb, 1)
output8 = higboostFiltering(rgb, mask8_rgb, 1)
output9 = higboostFiltering(rgb, mask9_rgb, 1)
output10 = higboostFiltering(rgb, mask10_rgb, 1)
output11 = higboostFiltering(rgb, mask11_rgb, 1)
output12 = higboostFiltering(rgb, mask12_rgb, 1)
output13 = higboostFiltering(rgb, mask13_rgb, 1)
output14 = higboostFiltering(rgb, mask14_rgb, 1)
output15 = higboostFiltering(rgb, mask15_rgb, 1)
output16 = higboostFiltering(rgb, mask16_rgb, 1)

cv.imshow("original", rgb)
cv.imshow("1", output1)
cv.imshow("2", output2)
cv.imshow("3", output3)
cv.imshow("4", output4)
cv.imshow("5", output5)
cv.imshow("6", output6)
cv.imshow("7", output7)
cv.imshow("8", output8)
cv.imshow("9", output9)
cv.imshow("10", output10)
cv.imshow("11", output11)
cv.imshow("12", output12)
cv.imshow("13", output13)
cv.imshow("14", output14)
cv.imshow("15", output15)
cv.imshow("16", output16)

cv.waitKey()
cv.destroyAllWindows()
"""