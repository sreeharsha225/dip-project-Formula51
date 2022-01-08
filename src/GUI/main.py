from os import lstat, path
from tkinter import *
from tkinter import filedialog as fd
from tkinter import ttk
import numpy as np
import cv2
from PIL import Image, ImageTk
from functools import partial
from inpaintingGUI import inpaint
import matplotlib.pyplot as plt

#########################################################################################################################################
##
## In the below code 'occImage, image_shape, image_file, image, output_image, last_x, last_y' are used as global variables.
## occImage  ---> binary image containing the choosen area (patch) by the user which is to be inpainted.
## image_shape --> To maintain uniformity across the resultant images created.
## image_file --> when a file is selected , image_file variable holds the name of the file.
## image --> when a file is selected, image variable holds the data of the image
## last_x, last_y --> when tracing the line drawn by the user ( on the input image ), last_x and last_y holds the previous values of pointer position.
##
#########################################################################################################################################

## After inpainting the image , ''clear'' function removes the image which are outputted
def clear():
    global occImage,cntInterrupt
    occImage = np.zeros(image_shape[::-1], dtype=np.uint8)
    # occluded images is set to all zeroes
    # The objects with tag of 'lines' and 'image' which are used in constructing the occluded image are deleted.
    # The object with line tag is used to show the boundary drawn by the user
    canvas_1.delete("lines")
    canvas_2.delete("image")
    cntInterrupt=0


# constructing the occluded area from the boundary drawn by the user on the input image.
def getOclusionPixels():
    global occImage, image_shape

    m , n = image_shape[::-1]

    ## We construct a binary image occImage containing values of 255 at occluded pixels and 0 at remaining pixels.

    for i in range(m):
        minn = -1
        maxx = -1

        # for each row in the image, we calculate first and last pixel which is part
        # of the boundary drawn by the user, and the take the all the pixels between them
        # to be the part of occluded image.

        for j in range(n):
            if(occImage[i, j] != 0):
                if(minn < 0):
                    minn = j
                else:
                    maxx = j
        if(maxx > 0):
            for k in range(minn, maxx+1):
                occImage[i, k] = 255
        elif minn>0:
            occImage[i, minn] = 255
    

# Below function is called after we obtain the input image
# All the processing which is needed to be done after, is called from below function
def apply():
    global Label_right, Label_left
    global occImage, output_image
    global occImage, output_image,cntInterrupt
    if cntInterrupt!=0:
        return
    cntInterrupt=1
    occImage = cv2.dilate(occImage, np.ones((3, 3), np.uint8), 2)
    ## The binary Image containing removed area(patch) is stored in occImage after calling getOclusionPixels function
    getOclusionPixels()
    
    # image_file contains the name of file, which contains the image data
    # The image file would be read and stored

    img = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    occ = occImage.copy()
    plt.imshow(occ,'gray')
    ## TODO
    # Both the input image and occluded image are 
    img = cv2.resize(img, send_shape, interpolation=cv2.INTER_NEAREST)
    occ = cv2.resize(occ, send_shape, interpolation=cv2.INTER_NEAREST)

    # inpaint function - takes binary occluded image and input image and outputs the resultant image
    output_image = inpaint(img.copy(), occ.copy())
    
    output_image = Image.fromarray(output_image.copy())
    output_image = output_image.resize(image_shape, Image.ANTIALIAS)

    # resultant image is displayed in canvas-2.
    output_image = ImageTk.PhotoImage(output_image)
    Label_right.config(text="Output Image")
    displayImage(canvas_2, output_image)
    print("Process completed!")

# canvas are similiar to containers, the images are displayed  upon canvas
def displayImage(canvas, image):
    canvas.create_image(0, 0, image=image, anchor="nw", tags="image")

# select file from internal memory
def select_file():
    global image_file, image_shape, image, occImage,cntInterrupt
    # all the formats of files, which may contain contain images are considered
    filetypes = (
        ("PNG files", "*.png"),
        ('All files', '*.*')
    )

    image_file = fd.askopenfilename(
        title='Open a file',
        initialdir='./',
        filetypes=filetypes)
    # Image_file holds the path to the selected file
    # Image is resized and displayed on canvas_1
    if image_file is not None:
        image = Image.open(image_file)
        image = image.resize(image_shape, Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)
        occImage = np.zeros(image_shape[::-1], np.uint8)
        displayImage(canvas_1, image)
        cntInterrupt=0

## recommendFile would display images , which the user can use as an input
# It takes imageName as input, we call the function by passing the path of images, that we want to show
def recommendFile(imageName):
    global image_file, image_shape, image, occImage,cntInterrupt
    image_file = imageName
    temp = cv2.imread(image_file)
    temp = cv2.resize(temp, send_shape, interpolation=cv2.INTER_NEAREST)
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(temp)
    image = image.resize(image_shape, Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)
    occImage = np.zeros(image_shape[::-1], np.uint8)
    displayImage(canvas_1, image)
    cntInterrupt=0
   

# On click on any part of canvas_1(input image),get_x_and_y get invoked and position of click will be stored
def get_x_and_y(event):
    global last_x, last_y
    # boundary conditions check for the image
    if image is not None and event.x < image_shape[0] and event.y < image_shape[1]:
        last_x, last_y = event.x, event.y

# The following functions trace the line drawn by the user
def draw_smth(event):
    global last_x, last_y, occImage
    if image is not None and event.x < image_shape[0] and event.y < image_shape[1]:
        canvas_1.create_line((last_x, last_y, event.x, event.y), 
                        fill='red',
                        tags="lines", 
                        width=2)
        occImage[last_y, last_x] = 255
        last_x, last_y = event.x, event.y

## Initialization  of global variables
image_file = None
image = None
output_image = None
send_shape = (250, 250)

app = Tk()
width = app.winfo_screenwidth()
height = app.winfo_screenheight()
app.geometry(f"{width}x{height}")

# To display the image on canvas and to maintain uniformity, we resize all the images 
image_shape = (width//3, height//3)
occImage = np.zeros(image_shape[::-1], np.uint8)



mainCanvas = Canvas(app, width=app.winfo_screenwidth(), height=height//2)
mainCanvas.pack(side='top', padx="10px",expand=True)

canvas_top = Canvas(mainCanvas, bg="black", width=app.winfo_screenwidth(), height=50)
canvas_top.pack(side="top", padx="5px", pady = "10px")

global Label_right, Label_left

Label_right = Label(canvas_top, text ='Output Image', bg="black", fg="white", font="30px")
Label_right.place(relx = 0.85,
                  rely = 0.37,
                  anchor ='ne')

 
Label_left = Label(canvas_top, text ='Input Image', bg="black", fg="white", font="30px")
Label_left.place(relx = 0.2,
                  rely = 0.37,
                  anchor ='ne')

canvas_1 = Canvas(mainCanvas, bg="black", width=image_shape[0]-1, height=image_shape[1]-1)
canvas_1.pack(side="left", padx="2px")

# To trace the boundary drawn by user
canvas_1.bind("<Button-1>", get_x_and_y)
canvas_1.bind("<B1-Motion>", draw_smth)

canvas_2 = Canvas(mainCanvas, bg="black", width=image_shape[0]-1, height=image_shape[1]-1)
canvas_2.pack(side="right", padx="5px")


canvas_3 = Canvas(app, bg="black", width = app.winfo_screenwidth(), height=height//3)
canvas_3.pack(side="bottom", padx="5px")

# image = image.resize((100, 100), Image.ANTIALIAS)

# on clicking the open_button, a file picker would open
open_button = ttk.Button(
    mainCanvas,
    text='Open file',
    command=select_file
)
open_button.pack(anchor="center", pady="2px")

# After taking the input, upon clicking the apply_button the inpainting algorithm would be called
apply_button = ttk.Button(
    mainCanvas,
    text="Inpaint",
    command=apply
)

apply_button.pack(pady="2px")

# Clear the output and the trace
clear_button = ttk.Button(
    mainCanvas,
    text="Clear",
    command=clear
)

clear_button.pack(pady="2px")

## imageArray stores the path of the recommended image files
global cntInterrupt
cntInterrupt=0

dirname, filename = path.split(path.abspath(__file__))
imageArray  = [dirname + '/images/cup.png', dirname+'/images/f.jpg', dirname+'/images/dog.png', dirname+'/images/cycle.png']
cnt = 0

images = []
imagebtn = []

# Each recommended image is clickable. Upon clicking those images would be selected as input
for imageName in imageArray:
    images.append(Image.open(imageName))
    
    images[-1] = images[-1].resize((200, 200), Image.ANTIALIAS)
    images[-1] = ImageTk.PhotoImage(images[-1])

    # Upon clicking, the name of recommended image is passed to function recommendFile
    imagebtn.append(ttk.Button(
        canvas_3,
        image=images[-1],
        command=partial(recommendFile, imageName)
    ))
    imagebtn[-1].grid(row=1, column=cnt, pady = 20, padx = 20)
    cnt += 1

app.mainloop()
