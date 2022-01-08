import os, sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from ANNwithPatchMatch import getANNShiftmap
from ReconstructingTheImage import Reconstruction
from pyramids import imagePyramidOcclusionPyramid, TextureFeaturePyramid, upSample
from onionPeel import OnionPeel
import numpy as np

# This function combines all the functions required for the inpainting algorithm and returns the final output image.

def inpaint(img, occ, p_size=7, lamda=50):
    occ[occ != 0] = 1
    print("Pyramid levels started!")
    imgPyramid, H_pixels, H_pyramid = imagePyramidOcclusionPyramid(img.copy(), occ.copy(), p_size)

    textures = TextureFeaturePyramid(img.copy(), occ, p_size)
    print("Pyramid levels ended!")
    print("Onion peel started!")
    imgPyramid[-1], textures[-1], shift_map = OnionPeel(imgPyramid[-1], textures[-1], H_pixels[-1], p_size, lamda)
    print("Onion peel ended!")
    L = len(imgPyramid) # number of levels in the pyramid
    pp = 3
    for l in range(L-1, -1, -1):
        k, e = 0, 1
        while k < 10 and e > 0.1:
            curr_img = imgPyramid[l].copy()
            shift_map = getANNShiftmap(imgPyramid[l].copy(), H_pixels[l].copy(), shift_map, 10, 
                                    textures[l], pp, lamda) # finding approximate nearest neighbours
            imgPyramid[l] = Reconstruction(imgPyramid[l], textures[l].copy(), 
                                    imgPyramid[l].copy(), shift_map, H_pixels[l].copy(), pp, 
                                    H_pyramid[l].copy(), lamda)
            textures[l] = Reconstruction(textures[l], textures[l], imgPyramid[l].copy(), shift_map, 
                                    H_pixels[l].copy(), pp, H_pyramid[l].copy(), lamda)
            e = np.mean(imgPyramid[l][H_pyramid[l] != 0] - curr_img[H_pyramid[l] != 0])
            k += 1
        if l > 0:
            new_h, new_w = H_pyramid[l-1].shape
            shift_map_new = np.zeros((new_h, new_w, 2), np.int64)
            shift_map = upSample(shift_map, shift_map_new, H_pyramid[l-1].copy(), H_pyramid[l-1].copy())
            imgPyramid[l-1] = Reconstruction(imgPyramid[l-1], textures[l-1].copy(), 
                                    imgPyramid[l-1].copy(), shift_map, H_pixels[l-1].copy(), pp, 
                                    H_pyramid[l-1].copy(), lamda)
            textures[l-1] = Reconstruction(textures[l-1], textures[l-1], imgPyramid[l-1].copy(), shift_map, 
                                    H_pixels[l-1].copy(), pp, H_pyramid[l-1].copy(), lamda)
        # increasing the patch size for the finer pyramid levels.
        pp += 2
        pp = min(p_size, pp)
        print(f"level:{l} completed")
    return imgPyramid[0]
