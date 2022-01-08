import numpy as  np
from ReconstructingTheImage import neighborhood
from ANNwithPatchMatch import boundaryConditions, patchDistance, zero_padding
import cv2

##############################################################################################################
# To avoid smoothing(blurring) of the inpainted image, we introduce texture features into the patch distance.#
# The textures features depends upon the the derivatives of the image in the x and y directions.             #
# The texture features are incorporated into the calculation of patch distamce.                              #
##############################################################################################################
# To apply the mask(filter) across the image
def slidethrough(image, mask):
    k = mask.shape[0]
    m, n = image.shape

    # Image is padded according to the size of the filter
    img = np.pad(image, (((k-1)//2,k//2),((k-1)//2,k//2)), 'constant')

    img_new = np.zeros([m, n])

    for i in range(0, m):
        for j in range(0, n):
            img_new[i, j] = np.sum(np.multiply(img[i:i+k,j:j+k], mask))


   # img_new = (img_new.astype(np.uint8))

   # if there any abruptions, they would be brought into valid range
    for i in range(0, m):
        for j in range(0, n):
            if(img_new[i][j] > 255):
                img_new[i][j] = 255
            elif (img_new[i][j] < 0):
                img_new[i][j] = 0-img_new[i][j]
    return img_new

# To obtain the image derivatives we use sobel filter 
def obtainDerivatives(img):
    
    Ix = np.zeros(img.shape, dtype=np.uint8)
    Iy = np.zeros(img.shape, dtype=np.uint8)
    copy_image = img.copy()
    Ix = cv2.Sobel(src=copy_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    Iy = cv2.Sobel(src=copy_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    return Ix, Iy

# Texture of an image are constructed using image derivatives. The formula to calculate is given in the paper
def texturesComponents(image, p_size):
    m,n,_ = image.shape
    Tx = np.zeros(image.shape, dtype=float)
    Ty = np.zeros(image.shape, dtype=float)

    Ix, Iy = obtainDerivatives(image)
    for i in range(m):
        for j in range(n):
            neighborhoodPatch = neighborhood(m, n, [i, j], p_size)
            card = len(neighborhoodPatch)
            for k in range(card):
                a, b = neighborhoodPatch[k]
                
                Tx[i][j] += abs(Ix[a][b]/card)
                Ty[i][j] += abs(Iy[a][b]/card)
        

    return Tx, Ty


# To retain texture components in the inpainted image, we introduce texture components
# in the calculation of patch distance.

def modifiedPatchDistance(img, p11, p22, p_size, Tx, Ty, wt):
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

    t_patchx_1 = Tx[-center+p1[0]:p1[0]+center+1, -center+p1[1]:p1[1]+center+1]
    t_patchy_1 = Ty[-center+p1[0]:p1[0]+center+1, -center+p1[1]:p1[1]+center+1]

    t_patchx_2 = Tx[-center+p2[0]:p2[0]+center+1, -center+p2[1]:p2[1]+center+1]
    t_patchy_2 = Ty[-center+p2[0]:p2[0]+center+1, -center+p2[1]:p2[1]+center+1]

    t_patch_1 = np.sqrt( np.square(t_patchx_1) + np.square(t_patchy_1) )
    t_patch_2 = np.sqrt( np.square(t_patchx_2) + np.square(t_patchy_2) )
    
    temp = patch_1 - patch_2
    texture = t_patch_1 - t_patch_2

    return np.sum(np.square(temp) + wt*np.square(texture))

# Texture image is formed in the below function 

def constructingTextures(image, H, p_size, sigma, shift_map, wt):
    Tx, Ty = texturesComponents(image.copy(), p_size)
    T = np.sqrt(np.square(Tx)+np.square(Ty))
    m, n, _ = image.shape
    padded_image = zero_padding(image.copy(), p_size//2)
    padded_Tx = zero_padding(Tx, p_size//2)
    padded_Ty = zero_padding(Ty, p_size//2)

    # for the constuction for textureimages the coefficient Spq are calculated as mentioned in paper.

    for i in H:
        neighborhoodPatch = neighborhood(m, n, [i[0], i[1]], p_size)
        ans = np.zeros(T[i[0]][i[1]].shape)
        denominator = 0
        
        for j in neighborhoodPatch:
            if boundaryConditions(m, n , j+shift_map[j[0], j[1]]) and boundaryConditions(m, n , j) and boundaryConditions(m, n , i+shift_map[j[0], j[1]]):
                s = np.exp( (-1)*modifiedPatchDistance(padded_image, j, j+shift_map[j[0], j[1]], p_size, padded_Tx, padded_Ty, wt)/(2 * np.square(sigma)) )
                co_ord = i+shift_map[j[0], j[1]]
                ans += s*T[co_ord[0], co_ord[1]]
                denominator += s
        if(denominator!=0):
            T[i[0]][i[1]] = (ans/denominator)

    # for p in H:
    #     neighborhoodOfPixel = neighborhood(m,n,p,p_size)

    #     patch_distance = []
    #     for j in neighborhoodOfPixel:
    #         patch_distance.append(modifiedPatchDistance(padded_image, j, j+shift_map[j[0], j[1]], p_size, padded_Tx, padded_Ty, wt))

    #     patch_distance = np.array(patch_distance)

    #     qstar = neighborhoodOfPixel[np.argsort(patch_distance)]
    #     index=0
    #     while index<len(qstar):
    #         if boundaryConditions(m,n,p + shift_map[qstar[index,0], qstar[index,1]]):
    #             newp = p + shift_map[qstar[index,0], qstar[index,1]]
    #             padded_Tx[p[0]+p_size//2,p[1]+p_size//2] = padded_Tx[newp[0]+p_size//2, newp[1]+p_size//2]
    #             padded_Ty[p[0]+p_size//2,p[1]+p_size//2] = padded_Ty[newp[0]+p_size//2, newp[1]+p_size//2]
    #     # img[p[0],p[1]] = img[newp[0], newp[1]]
    #             break
    #         index+=1


    return T




