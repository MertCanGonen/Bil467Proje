BIL 467 - Image Processing / Final Project

# Box filter filtering

# Gaussian filtering

# Median filtering

# Blurring
1) Blur uygula
    a)Box filter
    b)Gaussian filter
    c)Median filter

# Create Mask
2) Mask = original - blur

# Sharpening
3) Result = original + mask

# Image sharpening and highboost filtering
g(x,y) = mask
f(x,y) = original image
f'(x,y) = blurred image
output = f(x,y) + k[mask]

if k=1; unsharp masking
if k>1; highboost filtering
