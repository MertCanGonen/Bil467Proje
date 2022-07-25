BIL 467 - Image Processing
Project

Hepsini hem opencv metotlarıyla hem de elle yap!!!

# Box filter filtering
operatorların hepsi (sobel vs.)

# Gaussian filtering

# Median filtering

combining specail enhancement methods (türet sürekli yap hepsini)

# Blurring
1) Blur uygula
    a)Box filter (3x3) (11x11) (21x21)
    b)Gaussian filter
    c)Median filter

# Create Mask
2) Original'dan blur'u çıkart ve mask'i oluştur   

# Sharpening
3) Mask'i original' a ekle

# Formula
g(x,y) = mask
f(x,y) = original image
f'(x,y) = blurred image
output = f(x,y) + k[mask]

if k=1; unsharp masking
if k>1; highboost filtering

reshape ile grafik olarak da bastır 
bw1.reshape(-1)

farklı k değerleri için dene ve raporla !!!