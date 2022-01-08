# Non-Local Patch-Based Image Inpainting

Image inpainting is the task of filling in an unknown region in an image. This is useful, for example, if an area is damaged, or if one wishes to remove an unwanted object from the image. Inpainting can be used for personal purposes or for professional image restoration. Two of the main goals of image inpainting are the convincing restitution of structures and textures which are coherent with the unoccluded part of the image

# Links

## Input Images folder : 

https://drive.google.com/drive/folders/1aRwHv7Htnn3KOXRn3FZ8ijzrxOve6kZs?usp=sharing

## Output Images folder :

https://drive.google.com/drive/folders/1wkzILOEARgR90bcAcMb4x8gC2A4XVz_n?usp=sharing


## Presentation slides :
https://drive.google.com/drive/folders/1ep6zo36RXSEoXO3RxAmb6_VBySFE8Q2X?usp=sharing


# Required Documents
    Documentation of project 
    >./Documents/documentation.md
    >./Documetns/documentation.pdf

    Project Proposal 
    > ./Documents/Project Proposal.pdf

    Presentation slides
    > ./Documents/Presentation Slides.pdf
    > ./Documents/Presentation Slides.pptx

# Navigation Through code

### ANN
> ./src/ANNwithPatchMatch.py
* Implemented Patch Match algorithm for approximate nearest neighbour.
* Ouptuts the shiftmap which maps occluded pixels to some pixel in unoccluded region
* shiftmap is usefull inother parts of the algorithm including reconstruction of image etc

### Reconstruction of image
> ./src/ReconstructingTheImage.py
* Will assign the intensities of the occluded region using the shift map which is caclulated using previous algorithm.
* Returns reconstrcted image with modified intensities in occluded region.

### Onion peel initialization
> ./src/onionPeel.py
* We initialize the regions of occluded region layerwise. To understand this better we first initialize the outermost layer in occluded region using already known pixel intensity and texture values.
* But in the next go when we initialize deeper layers we use the previously calculated layers and known pixels to compute the current layers values.
* We take the coarser level image as input and this function will output an onion peel intialized image.


### Creating pyramid levels
> ./src/pyramids.py
* We subsample the image to get images of lower dimension. Using lower dimension makes it easier for initializing the intensities, it also helps us to get out of suboptimal solutions.
* We pass the details of the lower dimension image to higher dimension image by upsampling the mapping we used in the lower dimension.
* This function will give l levels of images, which are sub sampled with a factor of 1/2 and at each level blurred using gaussian
* We used coarse level images to construct higher dimensional finer level images.

### Inpainting
> ./src/GUI/inpaintingGUI.py
* This is the file where we used the above defined functions to get the over-all desired image.
* We used onion peel initialization , pyramids, ANN patch match and reconstruction to reconstruct the occluded region as given in the pseudo code(which is provided in research paper).

### GUI
> ./src/GUI/main.py
* Declared functions to maintain the functionality and display of the GUI.





# Goals

The main goal of the project is to implement an image inpainting algorithm by minimizing a patch-based function. We would try to improve the algorithm by using the following components of the algorithm.  
* Approximate Nearest neighbor Search
* Reconstructing the Image
* Inpainting with Textures
* Initialization
* Multiscale Scheme
* Algorithm Parameters
* Apart from these we also see the effect of various parameters on the output of the images, like patch size etc.
* We will apply this framework on a dataset of images and observe that our framework produces best results when compared to existing algorithms.

# Onion peel initialization:
Image with mask             |  Onion Peel Initialization         
:-------------------------:|:------------------------:
![doginp](https://user-images.githubusercontent.com/59763897/144001173-898a66ab-d05e-4353-92e1-d7a673c3817f.png)|![dogonion](https://user-images.githubusercontent.com/59763897/144001202-18e05e63-6500-48a1-9e9b-68aeb2c2ed8a.png)
![footbalinp](https://user-images.githubusercontent.com/59763897/144001237-bf772711-228a-4048-852a-7f5a61d8e76e.png)|![footballonion](https://user-images.githubusercontent.com/59763897/144001279-f08e64e3-fb8e-4b5d-9a38-2f9a4dc7fd60.png)
![girl](https://user-images.githubusercontent.com/59763897/144001406-d96010fe-3cff-4ec5-a98b-e9d544d7a698.png)|![girlonion](https://user-images.githubusercontent.com/59763897/144001422-f294fd28-b446-40a8-a3b1-19e137eaf5f4.png)
![image](https://user-images.githubusercontent.com/65012587/144026458-b5c0217a-df70-4c59-8fc1-67967428a481.png)|![image](https://user-images.githubusercontent.com/65012587/144026551-9afcb859-a6e2-4623-84c7-095648c5ccfc.png)






# Best of the Results:
Original Image             |  Image with mask         |  Final Image
:-------------------------:|:------------------------:|:-------------------------:
![image_2021-11-30_12-31-37](https://user-images.githubusercontent.com/59763897/144000898-fb482b23-3983-499f-a40c-2d7611d88aa7.png)|![doginp](https://user-images.githubusercontent.com/59763897/143999261-6a1a2f7a-f384-437f-8581-f96fc3e1222d.png)|![dogfinal](https://user-images.githubusercontent.com/59763897/143999367-aeef74e5-2fa7-4d64-bb19-37a054a5a880.png)
 ![image_2021-11-30_12-33-00](https://user-images.githubusercontent.com/59763897/144001044-abfe9d45-127c-4410-b229-390ddc2b0c3a.png)|![footbalinp](https://user-images.githubusercontent.com/59763897/143999978-186c91c6-bda9-4c90-a747-916e539cdbd8.png)|![football](https://user-images.githubusercontent.com/59763897/144000022-756ba62b-a722-49c9-9c44-f87b956c4d23.png)
![image_2021-11-30_12-34-33](https://user-images.githubusercontent.com/59763897/144001460-7aa80327-0aa6-4fee-878f-eec28982a117.png)| ![girl](https://user-images.githubusercontent.com/59763897/144000650-d3a06a0d-391e-4f6d-947a-ea5c6d41efc2.png)| ![girlfinal](https://user-images.githubusercontent.com/59763897/144000684-18040552-46a4-4563-a9a0-b71e9f94a7f5.png)
![image](https://user-images.githubusercontent.com/65012587/144026716-0dc1ed7b-12ca-4c30-b536-ca230e15bd98.png)|![image](https://user-images.githubusercontent.com/65012587/144026338-42f739c8-e565-4484-9cbf-ab4d356c5db9.png)|![image](https://user-images.githubusercontent.com/65012587/144026399-f9e3f0dd-a36c-4508-aba2-c9c183c0709b.png)



### How to run

1. Make sure to install all the libraries mentioned below. Depending upon the operating system, the installation procedure may change. We have written the installation procedure for Ubuntu 20.04. Make sure python3 is installed and of version 3.6.9 or plus. Make sure pip3 is installed and updated.

2. To run the GUI, run the ```main.py``` file present in src/GUI folder. Make sure to be present in project directory and run the following command to invoke the  GUI.

        python3 ./src/GUI/main.py

3. In the GUI, there would be a option to select an image file ```open file```(please make sure to select an valid format image file) at the top center of GUI, and an option to choose one among the recommended images(which are displayed at the bottom of GUI). The choosen image would be displayed on screen under the title input image.

4. The occluded part (part of image that needed to be removed) has be drawn/traced on the input image. We can draw on the input image using cursor. Please make sure to draw bit slowly on the input image, so that it would be easy to trace the boundary drawn by user. It would be recommended to draw a closed shape on input image to better visualize the results.
5. After drawing a boundary on the image, click on ```inpaint``` button, to run the algorithm. The output would be shown under the tile output image. The algorithm may take long time to execute(5minutes - 10 minutes), please be patient until the results appears. To track the progress, we would be printing the stage of execution of the algorithm in console(terminal).

### Libraries Used :

1. numpy :

    To install numpy libraries, run :

        pip3 install numpy
2. tkinter :

    To install tkinter library, run :
    
    On Linux:

        sudo apt-get install python3-tk
    
    
3. os:

    os library would be installed/present along with the python.

4. PIL:

    To install PIL library , run :

        sudo apt-get install python3-pil python3-pil.imagetk

5. sys

    The sys module comes packaged with Python.

6. matplotlib

    To install matplotlib, run :
         
        pip3 install matplotlib
7. cv2 

    To install cv2, run :

        pip3 install opencv-python

8. functools

    The functools module comes packaged with Python.
