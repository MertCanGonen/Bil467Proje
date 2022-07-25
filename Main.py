from statistics import median
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math

#Reading the all images
bw = cv.imread("bw1.jpg")
rgb = cv.imread("rgb.jpg")

#Creating all blurred images
#Box Filter
#0 padding
bw_pad0_bf1 = cv.blur(bw, (3,3), cv.BORDER_CONSTANT)
bw_pad0_bf2 = cv.blur(bw, (11,11), cv.BORDER_CONSTANT)
bw_pad0_bf3 = cv.blur(bw, (25,25), cv.BORDER_CONSTANT)

rgb_pad0_bf1 = cv.blur(rgb, (3,3), cv.BORDER_CONSTANT)
rgb_pad0_bf2 = cv.blur(rgb, (11,11), cv.BORDER_CONSTANT)
rgb_pad0_bf3 = cv.blur(rgb, (25,25), cv.BORDER_CONSTANT)

#Mirror padding
bw_padM_bf1 = cv.blur(bw, (3,3), cv.BORDER_REFLECT)
bw_padM_bf2 = cv.blur(bw, (11,11), cv.BORDER_REFLECT)
bw_padM_bf3 = cv.blur(bw, (25,25), cv.BORDER_REFLECT)

rgb_padM_bf1 = cv.blur(rgb, (3,3), cv.BORDER_REFLECT)
rgb_padM_bf2 = cv.blur(rgb, (11,11), cv.BORDER_REFLECT)
rgb_padM_bf3 = cv.blur(rgb, (25,25), cv.BORDER_REFLECT)

#Gaussian Filter
#0 padding
bw_pad0_g1_1 = cv.GaussianBlur(bw, (3,3), 0, cv.BORDER_CONSTANT)
bw_pad0_g1_2 = cv.GaussianBlur(bw, (3,3), 1, cv.BORDER_CONSTANT)
bw_pad0_g2_1 = cv.GaussianBlur(bw, (5,5), 0, cv.BORDER_CONSTANT)
bw_pad0_g2_2 = cv.GaussianBlur(bw, (5,5), 1, cv.BORDER_CONSTANT)

rgb_pad0_g1_1 = cv.GaussianBlur(rgb, (3,3), 0, cv.BORDER_CONSTANT)
rgb_pad0_g1_2 = cv.GaussianBlur(rgb, (3,3), 1, cv.BORDER_CONSTANT)
rgb_pad0_g2_1 = cv.GaussianBlur(rgb, (5,5), 0,  cv.BORDER_CONSTANT)
rgb_pad0_g2_2 = cv.GaussianBlur(rgb, (5,5), 1, cv.BORDER_CONSTANT)

#Mirror padding
bw_padM_g1_1 = cv.GaussianBlur(bw, (3,3), 0, cv.BORDER_REFLECT)
bw_padM_g1_2 = cv.GaussianBlur(bw, (3,3), 1, cv.BORDER_REFLECT)
bw_padM_g2_1 = cv.GaussianBlur(bw, (5,5), 0, cv.BORDER_REFLECT)
bw_padM_g2_2 = cv.GaussianBlur(bw, (5,5), 1, cv.BORDER_REFLECT)

rgb_padM_g1_1 = cv.GaussianBlur(rgb, (3,3), 0, cv.BORDER_REFLECT) #Hocam sigma degeri parametreler icerisinde var ancak K degeri yok. Metodun implemantation'ında yok o yüzden koyamadım ancak kendi tanımladığım manuel fonksiyonda mevcut.
rgb_padM_g1_2 = cv.GaussianBlur(rgb, (3,3), 1, cv.BORDER_REFLECT)
rgb_padM_g2_1 = cv.GaussianBlur(rgb, (5,5), 0,  cv.BORDER_REFLECT)
rgb_padM_g2_2 = cv.GaussianBlur(rgb, (5,5), 1, cv.BORDER_REFLECT)

#Median Filter
bw_m1 = cv.medianBlur(bw, 3)
bw_m2 = cv.medianBlur(bw, 11)
rgb_m1 = cv.medianBlur(rgb, 3)
rgb_m2 = cv.medianBlur(rgb, 11)


#Masks for b&w image
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

#Masks for rgb image
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



def gaussianBlur(image, kernel, K, sigma, border):
    arr = np.zeros((kernel, kernel))
    center = (kernel - 1) / 2
    for i in range(kernel):
        for j in range(kernel):
            r = (center-i)**2 + (center-j)**2
            value = K * math.exp(r / ( 2 * (sigma**2) ))
            arr[i][j] = value
    img = image
    (h, w) = image.shape[:2]
    output = np.zeros((h, w, 3), dtype = np.uint8)
    if (kernel == 3 and border == "0"):
        img = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_CONSTANT, (0,0,0)) #Hocam pixel sayısını belirlemek icin degisken koyunca hata verdigi icin böyle bir if-else state koymak zorunda kaldim.
    elif (kernel == 5 and border == "0"):
        img = cv.copyMakeBorder(image, 2, 2, 2, 2, cv.BORDER_CONSTANT, (0,0,0))
    elif (kernel == 3 and border == "M"):
        img = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_REFLECT, (0,0,0))
    elif (kernel == 5 and border == "M"):
        img = cv.copyMakeBorder(image, 2, 2, 2, 2, cv.BORDER_REFLECT, (0,0,0))
    k = int((kernel - 1) / 2)
    for i in range(k, h-k):
        for j in range(k, w-k):
            top = i - k
            bottom = i + k + 1
            left = j - k
            right = j + k + 1
            tmp = 0
            a = 0
            b = 0
            for x in range(top, bottom):
                b = 0
                for y in range(left, right):
                    tmp = tmp + img[x][y][0] * arr[a][b]
                    b = b + 1
                a = a + 1
            pixel = [tmp, tmp, tmp]
            print(pixel)
            output[i-1][j-1] = pixel
    cv.imshow("1", output)
    cv.waitKey(0)
    return output
    
            
            






def medianFilter(image, kernel, border): #kernel = 3 icin (max kernel = 11 icin)
    img = image
    (h, w) = image.shape[:2]
    output = np.zeros((h, w, 3), dtype = np.uint8)
    if (kernel == 3 and border == "0"):
        img = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_CONSTANT, (0,0,0)) #Hocam pixel sayısını belirlemek icin degisken koyunca hata verdigi icin böyle bir if-else state koymak zorunda kaldim.
    elif (kernel == 11 and border == "0"):
        img = cv.copyMakeBorder(image, 5, 5, 5, 5, cv.BORDER_CONSTANT, (0,0,0))
    elif (kernel == 3 and border == "M"):
        img = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_REFLECT, (0,0,0))
    elif (kernel == 11 and border == "M"):
        img = cv.copyMakeBorder(image, 5, 5, 5, 5, cv.BORDER_REFLECT, (0,0,0)) 
    k = int((kernel - 1) / 2)
    for i in range(k, h-k):
        for j in range(k, w-k):
            top = i - k
            bottom = i + k + 1
            left = j - k
            right = j + k + 1
            l = []
            for x in range(top, bottom):
                for y in range(left, right):
                    l.append(img[x][y][0])
            m = median(l)
            pixel = [m, m, m]
            output[i-1][j-1] = pixel
    return output



gaussianBlur(bw, 5, 1, 1, "0")
"""gaussianBlur(bw, 3, 1, 0, "M")
gaussianBlur(bw, 5, 1, 0, "0")
gaussianBlur(bw, 5, 1, 0, "M")"""


"""i = medianFilter(bw, 3, "0")
medianFilter(bw, 11, "0")
medianFilter(bw, 3, "M")
medianFilter(bw, 11, "M")"""

















"""cv.imshow("original", rgb)
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
cv.destroyAllWindows()"""


#Histogram calculation before the process
"""
r, g, b = cv.split(rgb4)
plt.hist(r.ravel(), 256, [0, 256])
plt.hist(g.ravel(), 256, [0, 256])
plt.hist(b.ravel(), 256, [0, 256])
plt.title("Color Histogram")
plt.xlabel("Color value")
plt.ylabel("Pixel count")
plt.show()
cv.waitKey()
cv.destroyAllWindows()


def box_filter_blurring(origin, image, kernel_size):
    i1 = 50
    i2 = 50
    (h, w) = image.shape[:2]
    kernel = np.ones((kernel_size,kernel_size),np.float32)/kernel_size**2
    k = (kernel_size - 1) / 2
    (q, z) = origin.shape[:2]
    output = np.zeros((q,z,3), dtype=np.float32)
    for x in range(50, h-50):
        i2 = 50
        for y in range(50, w-50):
            topX = int(x - k)
            bottomX = int(x + k + 1)
            leftY = int(y - k)
            rightY = int(y + k + 1)
            img = image[topX:bottomX, leftY:rightY]
            tmp = 0
            for i in range(kernel_size):
                for j in range(kernel_size):
                    tmp = tmp + (img[i][j] * kernel[i][j])  
            if (i2 == 266):
                break
            output[i1-50][i2-50] = tmp
            i2 = i2 + 1
        i1 = i1 + 1
    c = origin - output
    cv.imshow("1", output)

"""

