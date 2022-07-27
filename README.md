BIL 467 - Image Processing / Final Project

# Box filter filtering

# Gaussian filtering

# Median filtering

# Blurring
1) Blur uygula<br/>
    a)Box filter<br/>
    b)Gaussian filter<br/>
    c)Median filter

# Create Mask
2) Mask = original - blur

# Sharpening
3) Result = original + mask

# Image sharpening and highboost filtering
g(x,y) = mask<br/>
f(x,y) = original image<br/>
f'(x,y) = blurred image<br/>
output = f(x,y) + k[mask]<br/>
<br/>
if k=1; unsharp masking<br/>
if k>1; highboost filtering
