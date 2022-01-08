import numpy as np
import cv2
import matplotlib.pyplot as plt
from ANNwithPatchMatch import initialization, boundaryConditions, getDilatedOccludedRegion,getANNShiftmap, neighborhoodH1

# This function calculates the partial patch distance.
# Partial patch of a pixel is the neighborhood around the pixel excluding occluded pixels which have not been assigned a color yet.

def partialPatchDistanceH1(img, p, T, shiftmap, wt, binaryimg, p_size, recontructiontype):
	distance = 0
	m,n, _ = img.shape
	pixelneighbourhoodH1 = neighborhoodH1(m,n,p, p_size, binaryimg)

	length = len(pixelneighbourhoodH1)
	# print(length)
	for q in pixelneighbourhoodH1:
		utemp = img[q[0], q[1]]
		Ttemp = T[q[0], q[1]]
		if boundaryConditions(m,n, q + shiftmap[p[0],p[1]]):
			temp = q + shiftmap[p[0],p[1]]
			utemp -= img[temp[0], temp[1]]
			Ttemp -= T[temp[0], temp[1]]
		
		if recontructiontype == 0:
			distance += utemp**2 + wt*Ttemp**2
		else:
			distance += wt*utemp**2 + Ttemp**2
		 

	return distance.sum()/length


def spq(weights, sigma):
	return np.exp(-weights/(2*sigma*sigma))

# This function assigns the color to the pixels in the occluded region based on its neighboring pixels which are in the known region. 
# Known region involves pixels from the unoccluded region as well as occluded region pixels which have been assigned a color.

def partialReconstruction(img, shiftmap, differentialH, binaryimg, p_size, T,wt, recontructiontype):
	m,n, _ = img.shape
	image = np.copy(img)
	outimage = np.copy(image)
	for pixel in differentialH:
		weights = []
		neigh = neighborhoodH1(m,n,pixel, p_size, binaryimg)
		for q in neigh:
			weights.append(partialPatchDistanceH1(np.copy(image), q, T, shiftmap, wt, binaryimg, p_size, recontructiontype))
			# outimage[q[0],q[1]] = [255,255,255]
		# break
		weights = np.array(weights)
		# print(weights)
		sigma = np.percentile(weights, 75)
		if sigma==0:
			continue
		numerator = 0
		denominator = 0
		cou = 0
		weights = spq(weights, sigma)
		# Assigning color values in the current layer
		for i in range(len(neigh)):
			q = neigh[i]
			curwei = weights[i]
			if boundaryConditions(m,n, pixel + shiftmap[q[0],q[1]]):
				newp = pixel + shiftmap[q[0],q[1]]
				if binaryimg[newp[0], newp[1]] == 0:
					cou +=1 
					denominator += curwei
					numerator += curwei * image[newp[0], newp[1]]
		if denominator!= 0:
			outimage[pixel[0], pixel[1]] = numerator/denominator
		else:
			weights = []
			neigh = neighborhoodH1(m,n,pixel, p_size+4, binaryimg)
			for q in neigh:
				weights.append(partialPatchDistanceH1(np.copy(image), q, T, shiftmap, wt, binaryimg, p_size, recontructiontype))
				# outimage[q[0],q[1]] = [255,255,255]
			# break
			weights = np.array(weights)
			# print(weights)
			sigma = np.percentile(weights, 75)
			if sigma==0:
				continue
			numerator = 0
			denominator = 0
			cou = 0
			weights = spq(weights, sigma)
			for i in range(len(neigh)):
				q = neigh[i]
				curwei = weights[i]
				if boundaryConditions(m,n, pixel + shiftmap[q[0],q[1]]):
					newp = pixel + shiftmap[q[0],q[1]]
					if binaryimg[newp[0], newp[1]] == 0:
						cou +=1 
						denominator += curwei
						numerator += curwei * image[newp[0], newp[1]]
			if denominator!= 0:
				outimage[pixel[0], pixel[1]] = numerator/denominator
	return outimage


# Onion peel initialization erodes the occlusion one layer at a time. This method of initialization gives best inpainting results.	

def OnionPeel(imgL, T, HL, p_size,wt=50):
	m, n, _ = imgL.shape
	Htilda = getDilatedOccludedRegion(m, n, np.copy(HL), p_size)
	binaryimg1 = np.zeros((m,n), dtype='uint8')
	
	for x,y in Htilda:
		binaryimg1[x,y] = 1
	

	binaryimg = np.zeros((m,n), dtype='uint8')
	
	for x,y in HL:
		binaryimg[x,y] = 1
	img = np.copy(imgL)
	SE = np.ones((3,3), dtype = 'uint8')

	shiftmapL = initialization(m,n,np.copy(Htilda), binaryimg.copy())
	
	# return imgL,T,shiftmapL
	
	# SE = np.array([[0,1,0], [1,1,1], [0,1,0]], dtype = 'uint8')
	while np.sum(binaryimg) > 0:
		
		temp = cv2.erode(np.copy(binaryimg), SE)
		
		tempcopy = np.copy(temp)
		temp = binaryimg - temp
		# finding the layer to inpaint
		dHL = []
		for i in range(temp.shape[0]):
			for j in range(temp.shape[1]):
				if temp[i,j] != 0:
					dHL.append([i,j])

		dHL = np.array(dHL, dtype = 'int')
		print(len(dHL))
		# for i,j in dHL:
		# 	img[i,j] = [255,255,255]
		shiftmapL = getANNShiftmap(img.copy(),HL.copy(),np.copy(shiftmapL),10,T.copy(),p_size,wt)

		img_copy  = partialReconstruction(np.copy(img), shiftmapL, np.copy(dHL), binaryimg, p_size, T,wt, 0)
		plt.imshow(img_copy)
		# plt.show()
		T = partialReconstruction(T, shiftmapL, np.copy(dHL), binaryimg, p_size, np.copy(img),wt,1)
		img = np.copy(img_copy)
		binaryimg = np.copy(tempcopy)

		# break
		
	# shiftmapL = initialization(m,n,np.copy(Htilda), binaryimg1.copy())
	shiftmapL = getANNShiftmap(img.copy(),HL.copy(),np.copy(shiftmapL),10,T.copy(),p_size,wt)

		# break
	return img,T,shiftmapL
