import numpy as np
from ANNwithPatchMatch import patchMatchWithTexture, boundaryConditions, zero_padding,getDilatedOccludedRegion

def neighborhood(m,n,pixel, p_size):
	neigh = []
	for x in range(pixel[0]-p_size//2, pixel[0]+p_size//2+1):
		for y in range(pixel[1]-p_size//2, pixel[1]+p_size//2+1):
			if boundaryConditions(m,n,np.array([x,y])):
				neigh.append([x,y])
	neigh = np.array(neigh,dtype='int64')

	return neigh

# This function is to randomly assign a color to a pixel in dilated occluded region from the image. (Random initialization)

def randomInitialisation(image,H1):
	img=np.copy(image)
	m,n,_=img.shape
	for pixel in H1:
		random_r = np.random.randint(0, m)
		random_c = np.random.randint(0, n)
		img[pixel[0],pixel[1]]=img[random_r,random_c]
	return img

# This function implements the final reconstruction step of the algorithm

def reconstruct(image,T,shift_map, H,p_size,wt):
	img = np.copy(image)
	m, n, _ = img.shape
	# img=randomInitialisation(img,H)
	padded_img = zero_padding(img.copy(), p_size // 2)
	padded_T = zero_padding(T.copy(), p_size // 2)
	# padded_T=np.zeros(padded_T.shape)
	H1=getDilatedOccludedRegion(m, n, H, p_size)

	for p in H1:
		neighborhoodOfPixel = neighborhood(m,n,p,p_size)

		patch_distance = []
		for q in neighborhoodOfPixel:
			patch_distance.append(patchMatchWithTexture(padded_img.copy(),q,q+shift_map[q[0],q[1]],p_size,padded_T.copy(),wt))

		patch_distance = np.array(patch_distance)

		qstar = neighborhoodOfPixel[np.argsort(patch_distance)]
		index=0
		while index<len(qstar):
			if boundaryConditions(m,n,p + shift_map[qstar[index,0], qstar[index,1]]):
				newp = p + shift_map[qstar[index,0], qstar[index,1]]
				padded_img[p[0]+p_size//2,p[1]+p_size//2] = padded_img[newp[0]+p_size//2, newp[1]+p_size//2]
				padded_T[p[0]+p_size//2,p[1]+p_size//2] = padded_T[newp[0]+p_size//2, newp[1]+p_size//2]
				# img[p[0],p[1]] = img[newp[0], newp[1]]
				break
			index+=1

	return padded_img,img
	pass

# This function returns the weights for calculating the weighted mean which is to be assigned to the pixels in the occluded region.

def spq(weights, sigma):
	return np.exp(-weights/(2*sigma*sigma))

# This function assigns the color to the pixels in the occluded region using the shift map found using PatchMatch.
# Weighted mean is used.

def Reconstruction(oriimg,T,image, shift_map, HL,p_size,binaryimg,wt):
	p_size = max(3, p_size)
	m,n,_=oriimg.shape
	img=oriimg.copy()
	outimage = np.copy(img)
	padded_img=zero_padding(image.copy(),p_size//2)
	padded_T=zero_padding(T.copy(),p_size//2)
	# For finding sigma (75th percentile)
	for p in HL:
		weights=[]
		neigh=neighborhood(m,n,p,p_size)
		for q in neigh:
			weights.append(patchMatchWithTexture(padded_img.copy(),q,q+shift_map[q[0],q[1]],p_size,padded_T.copy(),wt))
		weights=np.array(weights)
		sigma=np.percentile(weights,75)
		if sigma==0:
			continue
		numerator=0
		denominator=0
		cou=0
		weights = spq(weights, sigma)
		# Calculating the weighted mean
		for i in range(len(neigh)):
			q = neigh[i]
			curwei = weights[i]
			if boundaryConditions(m,n, p + shift_map[q[0],q[1]]):
				newp = p + shift_map[q[0],q[1]]
				if binaryimg[newp[0], newp[1]] == 0:
					cou +=1 
					denominator += curwei
					numerator += curwei * image[newp[0], newp[1]]
		if denominator!= 0:
			outimage[p[0], p[1]] = numerator/denominator
		else:
			weights=[]
			neigh=neighborhood(m,n,p,p_size+4)
			for q in neigh:
				weights.append(patchMatchWithTexture(padded_img.copy(),q,q+shift_map[q[0],q[1]],p_size,padded_T.copy(),wt))
			weights=np.array(weights)
			sigma=np.percentile(weights,75)
			if sigma==0:
				continue
			numerator=0
			denominator=0
			cou=0
			weights = spq(weights, sigma)
			for i in range(len(neigh)):
				q = neigh[i]
				curwei = weights[i]
				if boundaryConditions(m,n, p + shift_map[q[0],q[1]]):
					newp = p + shift_map[q[0],q[1]]
					if binaryimg[newp[0], newp[1]] == 0:
						cou +=1 
						denominator += curwei
						numerator += curwei * image[newp[0], newp[1]]
			if denominator!= 0:
				outimage[p[0], p[1]] = numerator/denominator

	return outimage


def finalReconstruction(image,T, shift_map, H,p_size,wt):
	return reconstruct(image,T, shift_map, H,p_size,wt)

