import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import matplotlib as mpl
from scipy import ndimage, misc

def reset():
    global first_point_selected, first_point_x, first_point_y
    global second_point_x, second_point_y, line_length
    first_point_selected = False
    first_point_x, first_point_y = None, None
    second_point_x, second_point_y = None, None
    line_length = None

def key_press(event):
    global first_point_selected, first_point_x, first_point_y
    global second_point_x, second_point_y, line_length, img_rgb
    global xlabel_hint, degree, orig_img_rgb, start_num_for_naming
    if event.key == '1':
        reset()
        plt.clf()
        plt.xlabel(xlabel_hint)
        plt.ylabel("degree: " + str(degree))
        plt.imshow(img_rgb)
        plt.draw()
    elif event.key == '2':
        reset()
        plt.clf()
        orig_img_rgb = cv2.flip(orig_img_rgb, 1)
        img_rgb = cv2.flip(img_rgb, 1)
        plt.xlabel(xlabel_hint)
        plt.ylabel("degree: " + str(degree))
        plt.imshow(img_rgb)
        plt.draw()
    elif str(event.key) == '3' or str(event.key) == '4' or str(event.key) == '5' or str(event.key) == '6' or \
         str(event.key) == '7' or str(event.key) == '8' or str(event.key) == '9' or str(event.key) == '0':
        reset()
        plt.clf()
        plt.xlabel(xlabel_hint)
        if str(event.key) == '3': degree += 1
        elif str(event.key) == '4': degree -= 1
        elif str(event.key) == '5': degree += 5
        elif str(event.key) == '6': degree -= 5
        elif str(event.key) == '7': degree += 30
        elif str(event.key) == '8': degree -= 30
        elif str(event.key) == '9': degree += 90
        elif str(event.key) == '0': degree -= 90
        plt.ylabel("degree: " + str(degree))
        img_rgb = orig_img_rgb
        img_rgb = ndimage.rotate(img_rgb, degree, cval=255)
        plt.imshow(img_rgb)
        plt.draw()
    elif event.key == 'q':
        plt.close()

def mouse_press(event):
    global first_point_selected, first_point_x, first_point_y
    global second_point_x, second_point_y
    if event.xdata != None and event.ydata != None and first_point_selected == False:
        first_point_selected = True
        first_point_x = event.xdata
        first_point_y = event.ydata
    elif second_point_x != None and second_point_y != None and first_point_selected == True:
        global img_rgb, folder_name, line_length, start_num_for_naming, also_save_mirror_img
        x0 = int(round(min(first_point_x, second_point_x)))
        y0 = int(round(min(first_point_y, second_point_y)))

        # now I use cv2 for output, so change bgr to rgb again
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        img_cropped = img_rgb[y0:y0+int(line_length), x0:x0+int(line_length)]
        
        if save_bias == 'y':
            bias, ratio = [], 15
            left_bias_x0, right_bias_x0 = x0 - int(round(line_length / ratio)), x0 + int(round(line_length / ratio))
            up_bias_y0, down_bias_y0 = y0 - int(round(line_length / ratio)), y0 + int(round(line_length / ratio))
            if left_bias_x0 >= 0:
                # left
                bias.append(img_rgb[y0:y0+int(line_length), left_bias_x0:left_bias_x0+int(line_length)])
                if up_bias_y0 >= 0:
                    # left up
                    bias.append(img_rgb[up_bias_y0:up_bias_y0+int(line_length), left_bias_x0:left_bias_x0+int(line_length)])
                if down_bias_y0+int(line_length) <= img_rgb.shape[0]:
                    # left down
                    bias.append(img_rgb[down_bias_y0:down_bias_y0+int(line_length), left_bias_x0:left_bias_x0+int(line_length)])
            if right_bias_x0+int(line_length) <= img_rgb.shape[1]:
                #right
                bias.append(img_rgb[y0:y0+int(line_length), right_bias_x0:right_bias_x0+int(line_length)])
                if up_bias_y0 >= 0:
                    # right up
                    bias.append(img_rgb[up_bias_y0:up_bias_y0+int(line_length), right_bias_x0:right_bias_x0+int(line_length)])
                if down_bias_y0+int(line_length) <= img_rgb.shape[0]:
                    # right down
                    bias.append(img_rgb[down_bias_y0:down_bias_y0+int(line_length), right_bias_x0:right_bias_x0+int(line_length)])
            if up_bias_y0 >= 0:
                # up
                bias.append(img_rgb[up_bias_y0:up_bias_y0+int(line_length), x0:x0+int(line_length)])
            if down_bias_y0+int(line_length) <= img_rgb.shape[0]:
                # down
                bias.append(img_rgb[down_bias_y0:down_bias_y0+int(line_length), x0:x0+int(line_length)])

        if also_save_mirror_img == 'y':
            cv2.imwrite(folder_name + '\\' + str(start_num_for_naming) + ".jpg", img_cropped)
            start_num_for_naming = int(start_num_for_naming) + 1
            if save_bias == 'y':
                for i in range(len(bias)):
                    cv2.imwrite(folder_name + '\\' + str(start_num_for_naming) + ".jpg", bias[i])
                    start_num_for_naming = int(start_num_for_naming) + 1
            cv2.imwrite(folder_name + '\\' + str(start_num_for_naming) + ".jpg", cv2.flip(img_cropped, 1))
            start_num_for_naming = int(start_num_for_naming) + 1
            if save_bias == 'y':
                for i in range(len(bias)):
                    cv2.imwrite(folder_name + '\\' + str(start_num_for_naming) + ".jpg", cv2.flip(bias[i], 1))
                    start_num_for_naming = int(start_num_for_naming) + 1
                
        else:
            cv2.imwrite(folder_name + '\\' + str(start_num_for_naming) + ".jpg", img_cropped)
            start_num_for_naming = int(start_num_for_naming) + 1
            if save_bias == 'y':
                for i in range(len(bias)):
                    cv2.imwrite(folder_name + '\\' + str(start_num_for_naming) + ".jpg", bias[i])
                    start_num_for_naming = int(start_num_for_naming) + 1
        plt.close()

def draw_line(event):
    global first_point_selected, img_rgb, xlabel_hint
    if event.xdata != None and event.ydata != None and first_point_selected == False:
        plt.clf()
        plt.imshow(img_rgb)
        plt.plot(event.xdata, event.ydata, marker = '+', \
            markersize=15, fillstyle='none', color = "black")
        plt.xlabel(xlabel_hint)
        plt.ylabel("degree: " + str(degree))
        plt.draw()
    if event.xdata != None and event.ydata != None and first_point_selected == True:
        global first_point_x, first_point_y
        if event.xdata - first_point_x != 0 and event.ydata - first_point_y != 0:
            plt.clf()
            plt.imshow(img_rgb)
            #plt.plot(first_point_x, first_point_y, marker = 'o', \
            #        markersize=7, fillstyle='none', color = "black")
            global line_length
            line_length = min(abs(event.xdata - first_point_x), abs(event.ydata - first_point_y))

            x_direction = (event.xdata - first_point_x) / abs(event.xdata - first_point_x)
            y_direction = (event.ydata - first_point_y) / abs(event.ydata - first_point_y)

            x1, y1 = [first_point_x, first_point_x + x_direction * line_length] \
                , [first_point_y, first_point_y]
            x2, y2 = [first_point_x, first_point_x + x_direction * line_length] \
                , [first_point_y + y_direction * line_length, first_point_y + y_direction * line_length]
            plt.plot(x1, y1, x2, y2, color="black", linewidth=0.7)

            x3, y3 = [first_point_x, first_point_x], \
                     [first_point_y, first_point_y + y_direction * line_length]
            x4, y4 = [first_point_x + x_direction * line_length, first_point_x + x_direction * line_length], \
                     [first_point_y, first_point_y + y_direction * line_length]
            plt.plot(x3, y3, x4, y4, color="black", linewidth=0.7)

            plt.xlabel(xlabel_hint)
            plt.ylabel("degree: " + str(degree))
            plt.draw()
            
            global second_point_x, second_point_y
            second_point_x = first_point_x + x_direction * line_length
            second_point_y = first_point_y + y_direction * line_length

# set backend to TkAgg
mpl.use('TkAgg') 

# input folder name
folder_name = input("Input folder name: ")
if folder_name == '':
    folder_name = "folder"
folder_name = "./" + folder_name
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Global variable
xlabel_hint = "press: (1) cancel select, (2) mirror image, (q) skip this image \
\n(3) rotate image (<-.  1 degree), (4) rotate image (.->  1 degree), (5) rotate image (<-.  5 degree), (6) rotate image (.->  5 degree)\
\n(7) rotate image (<-. 30 degree), (8) rotate image (.-> 30 degree), (9) rotate image (<-. 90 degree), (0) rotate image (.-> 90 degree)"
start_num_for_naming = input("Input start number for naming: ")
if start_num_for_naming == '':
    start_num_for_naming = 1
also_save_mirror_img = input("save mirror cropped image at the same time? (y/n): ")
save_bias = input("save multiple images with little bias from selected position? (y/n): ")

# serach filename with extension .jpg and .png
img_list = []
for filename in os.listdir(os.getcwd()):
    if filename[-4:].lower().find('.jpg') != -1 or filename[-4:].lower().find('.png') != -1:
        img_list.append(filename)

for i in range(len(img_list)):
    # Global variables
    degree = 0
    first_point_selected = False
    first_point_x, first_point_y = None, None
    second_point_x, second_point_y = None, None
    line_length = None

    # image (also global variables)
    img_bgr = cv2.imread(img_list[i])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # save the origin img_rgb to solve rotation issue
    orig_img_rgb = img_rgb

    # set event
    plt.connect('motion_notify_event', draw_line)
    plt.connect('button_press_event', mouse_press)
    plt.connect('key_press_event', key_press)
    
    # set full screen
    plt.xlabel(xlabel_hint)
    plt.ylabel("degree: " + str(degree))
    plt.get_current_fig_manager().window.state('zoomed')

    # show image
    plt.imshow(img_rgb)
    plt.show()