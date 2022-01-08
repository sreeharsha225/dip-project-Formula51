import numpy as np

# neighborhoodH1 takes in a pixel position and patch size as input
# returns the pixel positions in its neighborhood which are not a part of occluded region

def neighborhoodH1(m,n,pixel, p_size, binaryimg):
    neigh = []
    for x in range(pixel[0]-p_size//2, pixel[0]+p_size//2+1):
        for y in range(pixel[1]-p_size//2, pixel[1]+p_size//2+1):
            if boundaryConditions(m,n,np.array([x,y])) and binaryimg[x,y] == 0:
                neigh.append([x,y])
    neigh = np.array(neigh,dtype='int64')

    return neigh

# patchMatchWithTexture functions takes 2 pixel positions and patch size as input
# returns the patch distance between the patches of the 2 given pixels
# This also includes textures

def patchMatchWithTexture(img, p11, p22, p_size, T, wt=50):
    center = p_size//2
    # print(p1,p2)
    p1=np.copy(p11)
    p2=np.copy(p22)
    p1 += np.array([center, center])
    p2 += np.array([center, center])
    
    patch_1 = img[-center+p1[0]:p1[0]+center+1, -center+p1[1]:p1[1]+center+1]
    patch_2 = img[-center+p2[0]:p2[0]+center+1, -center+p2[1]:p2[1]+center+1]

    t_patch_1 = T[-center+p1[0]:p1[0]+center+1, -center+p1[1]:p1[1]+center+1]
    t_patch_2 = T[-center+p2[0]:p2[0]+center+1, -center+p2[1]:p2[1]+center+1]
    
    temp = patch_1 - patch_2
    texture = t_patch_1 - t_patch_2

    return np.sum(np.square(temp) + wt*np.square(texture))/(p_size**2)


# patchMatchWithTexture functions takes 2 pixel positions and patch size as input
# returns the patch distance between the patches of the 2 given pixels

def patchDistance(img, p11, p22, p_size):
    center = p_size//2
    # print(p1,p2)
    p1=np.copy(p11)
    p2=np.copy(p22)
    p1 += np.array([center, center])
    p2 += np.array([center, center])
    # print("patcheddis")
    # print(p1,p2)
    patch_1 = img[-center+p1[0]:p1[0]+center+1, -center+p1[1]:p1[1]+center+1]
    patch_2 = img[-center+p2[0]:p2[0]+center+1, -center+p2[1]:p2[1]+center+1]
    temp = patch_1 - patch_2
    return np.sum(np.square(temp))

# This functions returns whether a pixel is in dilated occluded region or not.

def belongsToRegion(p, mark):
    if mark[p[0],p[1]]==1:
            return False
    return True

# This function returns whether a given position (r, c) is present in the image or not.

def boundaryConditions(m,n,q):
    if 0<=q[0]<m and 0<=q[1]<n:
        return True
    return False


# Changed by TH
# Reason: Only occuluded region needs shiftmap

def initialization(m, n, H, binaryimg):  # this function randomly initialises a shift map
    shift_map = np.zeros((m, n, 2), dtype=np.int64) 
    for pixel in H:
        x, y = pixel
        cou = 3
        while 1:
            neigh = neighborhoodH1(m,n,pixel, cou, binaryimg)
            cou += 2
            if len(neigh) != 0:
                num = len(neigh)
                ANN = pixel.copy()
                randomind = np.random.choice(num)
                random_r, random_c = neigh[randomind]
                ANN = np.array([random_r, random_c])
                shift_map[pixel[0], pixel[1]] = ANN - pixel
                break

        
        # print(shift_map[pixel[0], pixel[1]])
    return shift_map

# zero_padding function is used to pad the image for taking neighborhoods of every pixel in the image

def zero_padding(img, size):
    m, n, c = img.shape
    padded_img = np.zeros((m+2*size, n+2*size, c), dtype=np.uint8)
    padded_img[size: m+size, size:n+size, :] = img
    return padded_img

# Will return an np array which tells us if pixel is part of occluded: 1 or unoccluded: 0

def markEachPixel(m, n, H):
	markOccluded = np.zeros((m, n), dtype = int)
	for i in H:
		markOccluded[i[0],i[1]] = 1

	return markOccluded

# dilated occluded regions includes pixels whose patch neighborhood contains pixels from the occluded region
# returns the list of dilated occuleded pixel positions

def getDilatedOccludedRegion(m, n, H, p_size):
	H1 = []
	isItUnique = np.zeros((m,n), dtype = int)

	for i in H:
		xcor = i[0]
		ycor = i[1]
		for neighbourx in range(xcor-p_size//2, xcor+p_size//2 + 1):
			for neighboury in range(ycor-p_size//2, ycor+p_size//2 + 1):
				if boundaryConditions(m,n,[neighbourx, neighboury]) and isItUnique[neighbourx, neighboury] == 0:
					isItUnique[neighbourx, neighboury] = 1
					H1.append([neighbourx, neighboury])

	return np.array(H1)



# Changed by TH
# Reason: We think for getANNshoft map we need to pass occluded region it doenst make sense to pass H'

# This functions contains the implementation of PatchMatch algorithm which helps to find the approximate nearest neighbors.


def getANNShiftmap(img, H, shift_map,iter,T, p_size=7, wt=50): # returns approximate nearest neighbour shift map
    m, n, _ = img.shape
    ro = 0.5
    rmax=max(m,n)
    # returns Dilated occluded region H'
    markOccluded = markEachPixel(m, n, H)

    H1 = getDilatedOccludedRegion(m, n, H, p_size)
    markDiaOcculuded= markEachPixel(m,n,H1)
    # # random initialization
    # shift_map = initialization(m,n,np.copy(H)) 

    padded_img = zero_padding(img.copy(), p_size // 2)
    T = zero_padding(T.copy(), p_size // 2)
    for k in range(iter):
        for i in range(len(H1)):

            if k%2==0: # lexographic order on even iterations
                p = H1[i]
                a = p - np.array([1,0])
                b = p - np.array([0,1])
                # have to find q which has min distance from this patch
            else: # inverse lexographic order on odd iterations
                p = H1[len(H1)-i-1]
                a = p + np.array([1,0])
                b = p + np.array([0,1])
                # have to find q which has min distance from this patch
            patch_distance = []
            # the below code performs the propogation step
            f=0
            if boundaryConditions(m,n,p)  and boundaryConditions(m,n,p+shift_map[p[0], p[1]]):
                patch_distance.append(patchMatchWithTexture(padded_img, p, p+shift_map[p[0], p[1]], p_size, T, wt))
                f=1
            else:
                patch_distance.append(1e13)

            if boundaryConditions(m,n,a)  and boundaryConditions(m,n,p+shift_map[a[0], a[1]]):
                f=1
                patch_distance.append(patchMatchWithTexture(padded_img, p, p+shift_map[a[0], a[1]], p_size , T, wt))
            else:
                patch_distance.append(1e13)
            
            if boundaryConditions(m,n,b) and boundaryConditions(m,n,p+shift_map[b[0], b[1]]):
                f=1
                patch_distance.append(patchMatchWithTexture(padded_img, p, p+shift_map[b[0], b[1]], p_size,  T, wt))
            else:
                patch_distance.append(1e13)
            

            if f==1:
                patch_distances = np.array(patch_distance)
                q = np.argmin(patch_distances)
                if q == 0 :
                    q = p
                elif q == 1 :
                    q = a
                elif q==2 :
                    q = b
                if boundaryConditions(m,n,p+shift_map[q[0], q[1]]) and belongsToRegion(p+shift_map[q[0], q[1]], markDiaOcculuded):
                    shift_map[p[0], p[1]] = shift_map[q[0], q[1]]
                
            # random search
            zmax = int(np.ceil(-np.log(max(m,n))/np.log(ro)))

            for z in range(1,zmax):
                q=p+shift_map[p[0], p[1]]+np.floor(rmax*(ro**z)*np.random.uniform(-1,1,size=2))

                q=q.astype('int64')
                try:
                    if boundaryConditions(m,n,q) and boundaryConditions(m,n,p+shift_map[q[0], q[1]]) and \
                boundaryConditions(m,n,p+shift_map[p[0], p[1]]) and         \
                     patchMatchWithTexture(padded_img, p, p+shift_map[q[0], q[1]], p_size,  T, wt) <\
                         patchMatchWithTexture(padded_img, p, p+shift_map[p[0], p[1]], p_size,  T, wt) and \
                             belongsToRegion(p+shift_map[q[0], q[1]], markDiaOcculuded):
                        shift_map[p[0], p[1]] = shift_map[q[0], q[1]]
                except Exception as e:
                    print(q,p)
                    print(shift_map[q[0],q[1]], shift_map[p[0],p[1]])
                    print(p+shift_map[q[0],q[1]], p+shift_map[p[0],p[1]])
                    print(e)
                    print()
                    return 0
                
    return shift_map
