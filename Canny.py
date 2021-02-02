import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from matplotlib.pyplot import figure
figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
# do the mult
def filterGS(img , filter):
    temp = np.zeros((423, 419))
    for i in range(420):
        for j in range(416):
            for l in range(len(filter[0])):
                t = 0
                for k in range(len(filter[0])):
                    t += img[i + l][k + j] * filter[k][l]
                temp[i][j] += t
    return temp

# organize
def gaussians():
    image = cv2.imread('sudoku-original.jpg ' , 0)
    padding = np.zeros((427, 423))
    temp = np.copy(image)
    new_image = padding
    new_image[2:425, 2:421] = temp

    Gaussians = np.array(
        [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256

    clean_image = filterGS(new_image, Gaussians)
    afterClean = clean_image[2:425, 2:421]
    result = afterClean[:423, :419]
    return result , image



def gradient_intensity(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32)
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    G = np.hypot(Ix, Iy)
    D = np.arctan2(Iy, Ix)
    return (G, D)

def round_angle(angle):
    angle = np.rad2deg(angle)%180
    if (0<=angle<22.5) or (157.5<=angle<180):
        angle=0
    elif 22.5 <= angle < 67.5:
        angle=45
    elif 67.5 <=angle < 112.5:
        angle=90
    elif 112.5<=angle<157.5:
        angle=135
    return  angle

def suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    for i in range(M):
        for j in range(N):
            # find neighbour pixels to visit from the gradient directions
            where = round_angle(D[i, j])
            try:
                if where == 0:
                    if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
                        Z[i, j] = img[i, j]
            except IndexError as e:
                pass

    return Z


def threshold(img, t, T):
    # define gray value of a WEAK and a STRONG pixel
    cf = {
        'WEAK': np.int32(25),
        'STRONG': np.int32(255),
    }
    # get strong pixel indices
    strong_i, strong_j = np.where(img > T)
    # get weak pixel indices
    weak_i, weak_j = np.where((img >= t) & (img <= T))
    # get pixel indices set to be zero
    zero_i, zero_j = np.where(img < t)
    # set values
    img[strong_i, strong_j] = cf.get('STRONG')
    img[weak_i, weak_j] = cf.get('WEAK')
    img[zero_i, zero_j] = np.int32(0)
    return (img, cf.get('WEAK'))

def tracking(img, weak, strong=255):
    M, N = img.shape
    for i in range(M-1):
        for j in range(N-1):
            if img[i, j] == weak:
                # check if one of the neighbours is strong (=255 by default)
                try:
                    if ((img[i + 1, j] == strong) or (img[i - 1, j] == strong)
                    or (img[i, j + 1] == strong) or (img[i, j - 1] == strong)
                    or (img[i + 1, j + 1] == strong) or (img[i - 1, j - 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img
# find the 4 line
def findLine(image ,num):
    start , end   = (0,0)
    M ,N  = image.shape
    for i in range(num ,M):
        if image[i][num]!=0:
            start = (50 ,i)
            end = (50 ,M)
            break
    return  start ,end
# find the distance
def findDistanceBtweenCulomn(image , start , mid):
    M = image.shape[1]
    counter=0
    for i in range(start+5 , M):
        if image[mid][i]==0:
            counter +=1
        if image[mid][i]!=0:
            break
    return counter+5

if  __name__ == "__main__":

   img1   , ORG = gaussians()
   img2, D = gradient_intensity(img1)
   img3 = suppression(np.copy(img2), D)
   img4, weak = threshold(np.copy(img3),80 ,130)
   img5 = tracking(np.copy(img4), weak)
   color = (255, 0, 0)
   thickness = 3
   new = cv2.imread('sudoku-original.jpg')
   image  = np.copy(new)
   num = 50 # to find the first  column
   mid  = 72 # mid of the Y to ignore the  interruption
   distance  = findDistanceBtweenCulomn(img5 ,num , mid) # calculate the distance between two column
    # column 1
   start_point , end_point = findLine(img5 ,num)
   image = cv2.line(new, start_point, end_point, color, thickness)
   # column 2
   start_point = (num +distance,start_point[1])
   end_point = (num+distance ,end_point[1])
   image = cv2.line(new, start_point, end_point, color, thickness)
   # column 3
   start_point = (num+distance*2, start_point[1])
   end_point = (num+distance*2, end_point[1])
   image = cv2.line(new, start_point, end_point, color, thickness)
   # column 4
   start_point = (num +distance*3, start_point[1])
   end_point = (num+distance*3, end_point[1])
   image = cv2.line(new, start_point, end_point, color, thickness)

   plt.subplot(3,2,1), plt.imshow(ORG, cmap="gray"), plt.title('Org')
   plt.xticks([]), plt.yticks([])
   plt.subplot(3,2,2), plt.imshow(img1, cmap="gray"), plt.title('afterGS')
   plt.xticks([]), plt.yticks([])
   plt.subplot(3,2,3), plt.imshow(img2, cmap="gray"), plt.title('afterGradient')
   plt.xticks([]), plt.yticks([])
   plt.subplot(3, 2, 4), plt.imshow(img3, cmap="gray"), plt.title('suppression')
   plt.xticks([]), plt.yticks([])
   plt.subplot(3, 2,5 ), plt.imshow(img5, cmap="gray"), plt.title('after tracking')
   plt.xticks([]), plt.yticks([])
   plt.subplot(3, 2, 6), plt.imshow(image, cmap="gray"), plt.title('finalImage')
   plt.xticks([]), plt.yticks([])
   plt.show()

