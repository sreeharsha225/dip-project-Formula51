import numpy as np
import cv2
from ConstructingTextures import texturesComponents

Green_Threshold = 240
# Manual Implementation
def erode(img, SE):
    center = SE.shape[0] // 2
    result = np.zeros(img.shape, dtype=np.uint8)
    rows, cols = img.shape
    for r in range(center, rows-center):
        for c in range(center, cols-center):
            segment = img[r-center:r+center+1, c-center:c+center+1] % 2
            if np.allclose(segment & SE, SE):
                result[r, c] = 255
    return result

# This function returns the number of pyramid levels required in the multiscale method based on occlusion size and patch size

def getNLevels(H, p_size): # H is occlusion image (binary)
    temp = H.copy()
    SE = np.ones((3, 3), dtype=np.uint8)
    numOnes = np.sum(H)
    numErosions = 0
    while numOnes > 0:
        temp = cv2.erode(temp, SE)
        numErosions += 1
        numOnes = np.sum(temp)
    # print(numErosions)
    numLevels = np.ceil(np.log2(2*numErosions / p_size))
    return max(int(numLevels),2)
    # return 3



# Changed by TH
# Reason: Using ceil messes up.
# This function takes an image and new dimensions as input and returns the resized image as output.
# The purpose of this function is to get images in the coarser level of the pyramid.

def scale(img, new_r, new_c, default = 1):
    m = img.shape[0]
    n = img.shape[1]
    factor = 1/2
    if default:
    	out_image = np.zeros((int(new_r), int(new_c), 3), dtype=np.uint8)
    else:
    	out_image = np.zeros((int(new_r), int(new_c)), dtype=np.uint8)
    for r in range(new_r):
        for c in range(new_c):
            target_r = int(np.round(r/factor))
            target_c = int(np.round(c/factor))
            out_image[r, c] = img[target_r, target_c]
    return out_image

# This function upsamples the shift map obatined for level l in the pyramid and use it as starting point in level l-1

def upSample(shiftmapold,shiftmapnew, BI, preBI):
	m,n,_=shiftmapnew.shape
	for i in range(m):
		for j in range(n):

			# coarserx = i//2
			# coarsery = j//2
			# coaserfinalx = coarserx + shiftmapold[coarserx, coarsery][0]
			# coaserfinaly = coarsery + shiftmapold[coarserx, coarsery][1]
			# if preBI[coaserfinalx, coaserfinaly] != 0:
			# 	print("Now this is an Avengers level Threat!!!")
			# finerfinalx = 2*coaserfinalx
			# finerfinaly = 2*coaserfinaly
			# shiftmapnew[i,j][0] =  finerfinalx - i
			# shiftmapnew[i,j][0] =  finerfinaly - j
			# if BI[finerfinalx, finerfinaly] != 0:
			# 	print("Unexpected behaviour!!!")
			

			tempx,tempy=shiftmapold[i//2,j//2]*2
			sx=tempx
			sy=tempy
			if i+tempx<0:
				sx=-i 
			if i+tempx>m-1:
				sx=m-1-i	
			if j+tempy<0:
				sy=-j 
			if j+tempy>n-1:
				sy=n-1-j
			shiftmapnew[i,j][0]=sx
			shiftmapnew[i,j][1]=sy
			# if shiftmapnew[i,j,0]==0 and shiftmapnew[i,j,1]==0:
			# 	print("invalid")
	return shiftmapnew

# This function returns the gaussian pyramid for a given image.
# Gaussian kernel used is 3 by 3 with sigma = 1.5

def imagePyramid(img, nLevels):
    images = [img]
    for i in range(1, nLevels):
        upper_level = scale(cv2.GaussianBlur(images[-1], (3, 3), 1.5, 1.5), 1/2)
        images.append(upper_level)
    return images

def custompyrDown(image, newm, newn):
	out = scale(cv2.GaussianBlur(image, (3, 3), 1.5, 1.5), newm, newn)
	return out

# Used Libraries
# Input: img --> original image, Hbinary --> Occluded binary image.

def imagePyramidOcclusionPyramid(img, Hbinary, p_size):
	nLevels = getNLevels(Hbinary, p_size)
	images = [img]

	for i in range(nLevels-1):
		images.append(custompyrDown(np.copy(images[-1]),(images[-1].shape[0]+1)//2, (images[-1].shape[1]+1)//2))
	images = np.array(images,dtype=object)

	binaries = [Hbinary]
	for i in range(nLevels-1):
		binaries.append(scale(binaries[-1],(binaries[-1].shape[0]+1)//2, (binaries[-1].shape[1]+1)//2, 0))
	binaries = np.array(binaries, dtype = object)

	Hvalues = []
	temp = []
	for i in range(nLevels):
		temp = []
		for j in range(images[i].shape[0]):
			for k in range(images[i].shape[1]):
				# print(binaries[i][j,k])
				if binaries[i][j,k] == 1:
					temp.append([j,k])
					images[i][j,k] = [0,0,0]

		Hvalues.append(temp)
	Hvalues = np.array(Hvalues,dtype=object)

	return images, Hvalues, binaries

# This function takes an image as input and returns its texture pyramid as output. Used in multiscale scheme.

def TextureFeaturePyramid(img, Hbinary, p_size):
	nLevels = getNLevels(Hbinary, p_size)
	tx, ty = texturesComponents(np.copy(img), p_size)
	tem = np.square(tx) + np.square(ty)
	tem = np.sqrt(tem)
	for i in range(img.shape[2]):
		tem[:,:,i][tem[:,:,i]>255] = 255

	textures = [tem]

	for i in range(nLevels-1):
		textures.append(custompyrDown(np.copy(textures[-1]), (textures[-1].shape[0]+1)//2, (textures[-1].shape[1]+1)//2))
	textures = np.array(textures,dtype=object)
	HHB=Hbinary.copy()
	Hbinary=cv2.dilate(HHB,np.ones((5,5),dtype='int'))
	binaries = [Hbinary]
	for i in range(nLevels-1):
		binaries.append(scale(binaries[-1],(binaries[-1].shape[0]+1)//2, (binaries[-1].shape[1]+1)//2, 0))
	binaries = np.array(binaries, dtype = object)

	for i in range(nLevels):

		for j in range(textures[i].shape[0]):
			for k in range(textures[i].shape[1]):
				# print(binaries[i][j,k])
				if binaries[i][j,k] == 1:
					textures[i][j,k] = [0,0,0]
					pass

	return textures


