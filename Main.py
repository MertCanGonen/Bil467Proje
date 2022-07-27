from statistics import median
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math


#Reading the all images
bw = cv.imread("bw.jpg")
rgb = cv.imread("rgb1.jpg")

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
    cv.imshow("Output", output)
    cv.waitKey()
        
"""
boxFilter(rgb, 3, "0")
boxFilter(bw, 3, "M")
boxFilter(bw, 11, "0")
boxFilter(bw, 11, "M")

boxFilter(rgb, 3, "0")
boxFilter(rgb, 3, "M")
boxFilter(rgb, 11, "0")
boxFilter(rgb, 11, "M")
"""

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
    print(arr)
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
    cv.imshow("1", output)
    cv.waitKey(0)
    return output
    
"""
gaussianBlur(rgb, 3, 1, "0")
gaussianBlur(bw, 11, 1, "M")
gaussianBlur(bw, 5, 0, "0")
gaussianBlur(bw, 5, 0, "M")
"""           

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
    cv.imshow("Output", output)
    cv.waitKey()

"""
medianFilter(rgb, 11, "0")
medianFilter(bw, 11, "0")
medianFilter(bw, 3, "M")
medianFilter(bw, 11, "M")

medianFilter(bw, 3, "0")
medianFilter(bw, 3, "M")
medianFilter(bw, 11, "0")
medianFilter(bw, 11, "M")
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
mask1_bw = bw - bw_pad0_bf1
mask2_bw = bw - bw_pad0_bf2
mask3_bw = bw - bw_pad0_bf3
mask4_bw = bw - bw_padM_bf1
mask5_bw = bw - bw_padM_bf2
mask6_bw = bw - bw_padM_bf3

mask7_bw = bw - bw_pad0_g1_1
mask8_bw = bw - bw_pad0_g1_2
mask9_bw = bw - bw_pad0_g2_1
mask10_bw = bw - bw_pad0_g2_2
mask11_bw = bw - bw_padM_g1_1
mask12_bw = bw - bw_padM_g1_2
mask13_bw = bw - bw_padM_g2_1
mask14_bw = bw - bw_padM_g2_2

mask15_bw = bw - bw_m1
mask16_bw = bw - bw_m2

        #MASKS FOR RGB IMAGE
mask1_rgb = rgb - rgb_pad0_bf1
mask2_rgb = rgb - rgb_pad0_bf2
mask3_rgb = rgb - rgb_pad0_bf3
mask4_rgb = rgb - rgb_padM_bf1
mask5_rgb = rgb - rgb_padM_bf2
mask6_rgb = rgb - rgb_padM_bf3

mask7_rgb = rgb - rgb_pad0_g1_1
mask8_rgb = rgb - rgb_pad0_g1_2
mask9_rgb = rgb - rgb_pad0_g2_1
mask10_rgb = rgb - rgb_pad0_g2_2
mask11_rgb = rgb - rgb_padM_g1_1
mask12_rgb = rgb - rgb_padM_g1_2
mask13_rgb = rgb - rgb_padM_g2_1
mask14_rgb = rgb - rgb_padM_g2_2

mask15_rgb = rgb - rgb_m1
mask16_rgb = rgb - rgb_m2

        #OUTPUT FOR B&W IMAGES
output1 = bw + mask1_bw
output2 = bw + mask2_bw
output3 = bw + mask3_bw
output4 = bw + mask4_bw
output5 = bw + mask5_bw
output6 = bw + mask6_bw
output7 = bw + mask7_bw
output8 = bw + mask8_bw
output9 = bw + mask9_bw
output10 = bw + mask10_bw
output11 = bw + mask11_bw
output12 = bw + mask12_bw
output13 = bw + mask13_bw
output14 = bw + mask14_bw
output15 = bw + mask15_bw
output16 = bw + mask16_bw

        #OUTPUT FOR RGB IMAGES
output1 = rgb + mask1_rgb
output2 = rgb + mask2_rgb
output3 = rgb + mask3_rgb
output4 = rgb + mask4_rgb
output5 = rgb + mask5_rgb
output6 = rgb + mask6_rgb
output7 = rgb + mask7_rgb
output8 = rgb + mask8_rgb
output9 = rgb + mask9_rgb
output10 = rgb + mask10_rgb
output11 = rgb + mask11_rgb
output12 = rgb + mask12_rgb
output13 = rgb + mask13_rgb
output14 = rgb + mask14_rgb
output15 = rgb + mask15_rgb
output16 = rgb + mask16_rgb

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