import glob, random
import numpy as np
import cv2
import math
import imutils
import os
import shutil

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx,qy

def check_if_b_box_outside_of_image(b_box_corners):
    if b_box_corners[0] < 0:
        b_box_corners[0] = 0
    if b_box_corners[0] > width:
        b_box_corners[0] = width
    if b_box_corners[1] < 0:
        b_box_corners[1] = 0
    if b_box_corners[1] > width:
        b_box_corners[1] = width
    if b_box_corners[2] < 0:
        b_box_corners[2] = 0
    if b_box_corners[2] > height:
        b_box_corners[2] = height
    if b_box_corners[3] < 0:
       b_box_corners[3] = 0
    if b_box_corners[3] > height:
        b_box_corners[3] = height
    return b_box_corners

def rotate_image():
    global image
    global height, width
    global original_with_visual_b_box, rotated_with_visual_b_box, rotated_no_b_box
    global rotated, annotation, annotation_rot

    annotation = open(("data_augmentation/original_images/"+image_name+".txt"), "r")
    annotation_rot = open(("data_augmentation/rotated/"+image_name+"_rot.txt"), "w")
    random_rotation = random.randint(-30,30)
    data = cv2.imread(image,1)
    height, width, channels = data.shape

    rotated = imutils.rotate(data, random_rotation)

    rotated_no_b_box = rotated
    for line in annotation:
        line = line.split()
        top_left_corner_x = int((float(line[1])-float(line[3])/2)*width)
        top_left_corner_y = int((float(line[2])-float(line[4])/2)*height)
        top_right_corner_x = int(top_left_corner_x + float(line[3])*width)
        top_right_corner_y = top_left_corner_y
        bottom_right_corner_x = int((float(line[1])+float(line[3])/2)*width)
        bottom_right_corner_y = int((float(line[2])+float(line[4])/2)*height)
        bottom_left_corner_x = int(bottom_right_corner_x - float(line[3])*width)
        bottom_left_corner_y = bottom_right_corner_y
        original_with_visual_b_box = cv2.rectangle(data, (top_left_corner_x, top_left_corner_y), (bottom_right_corner_x, bottom_right_corner_y), (0, 255, 0), 3)
        b_box_corners = np.array([top_left_corner_x,top_left_corner_y,top_right_corner_x,top_right_corner_y,bottom_right_corner_x,bottom_right_corner_y,bottom_left_corner_x,bottom_left_corner_y], np.int32)

        rot_ax,rot_ay = rotate((width/2,height/2),(top_left_corner_x,top_left_corner_y),math.radians(-random_rotation))
        rot_bx,rot_by = rotate((width/2,height/2),(top_right_corner_x,top_right_corner_y),math.radians(-random_rotation))
        rot_cx,rot_cy = rotate((width/2,height/2),(bottom_right_corner_x,bottom_right_corner_y),math.radians(-random_rotation))
        rot_dx,rot_dy = rotate((width/2,height/2),(bottom_left_corner_x,bottom_left_corner_y),math.radians(-random_rotation))

        list_x = [rot_ax,rot_bx,rot_cx,rot_dx]
        list_y = [rot_ay,rot_by,rot_cy,rot_dy]

        minim_x, maxim_x, minim_y, maxim_y  = min(list_x),max(list_x),min(list_y),max(list_y)
        new_b_box = [minim_x, maxim_x, minim_y, maxim_y]
        check_if_b_box_outside_of_image(new_b_box)
        #rotated_with_visual_b_box = cv2.rectangle(rotated, (int(new_b_box[0]), int(new_b_box[2])), (int(new_b_box[1]), int(new_b_box[3])), (0, 0, 255), 3)
        b_box_center_x = (new_b_box[0] + (new_b_box[1] - new_b_box[0])/2)/width
        b_box_center_y = (new_b_box[2] + (new_b_box[3] - new_b_box[2])/2)/height
        #print((new_b_box[2] + (new_b_box[3] - new_b_box[2])/2)/height)
        b_box_width = (new_b_box[1] - new_b_box[0])/width
        #print((new_b_box[1] - new_b_box[0])/width)
        b_box_height = (new_b_box[3] - new_b_box[2])/height
        #print((new_b_box[3] - new_b_box[2])/height)
        #if (b_box_width * b_box_height) >= float(line[3]) * float(line[2]) * 0.25:
        annotation_rot.write(line[0] + ' ' + str(round(b_box_center_x,6)) + ' ' + str(round(b_box_center_y,6)) + ' ' + str(round(b_box_width,6)) + ' ' + str(round(b_box_height,6)) + '\n')

        #show_rotations()
        #print('rotated\\'+image_name+'_rot.jpg')


def show_rotations():
    cv2.namedWindow("original_with_visual_b_box", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("original_with_visual_b_box", 1000, 600)
    cv2.namedWindow("rotated_with_visual_b_box", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("rotated_with_visual_b_box", 1000, 600)
    cv2.imshow('original_with_visual_b_box', original_with_visual_b_box)
    cv2.imshow('rotated_with_visual_b_box', rotated_with_visual_b_box)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def histogram():
    ###############
    # Histogram Equalization
    channels = cv2.split(image)
    eq_channels = []
    for ch, color in zip(channels, ['B', 'G', 'R']):
        eq_channels.append(cv2.equalizeHist(ch))
    global eq_image
    eq_image = cv2.merge(eq_channels)
    return eq_image

def histogram_hsv():
    # Histogram Equalization (HSV)
    H, S, V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    eq_V = cv2.equalizeHist(V)
    global eq_image_hsv
    eq_image_hsv = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)
    return eq_image_hsv

def histogram_clahe_l():
    # Histogram Equalization (CLAHE-L)
    # convert image to LAB color model
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # split the image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(image_lab)
    # apply CLAHE to lightness channel
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    # merge the CLAHE enhanced L channel with the original A and B channel
    merged_channels = cv2.merge((cl, a_channel, b_channel))
    # convert iamge from LAB color model back to RGB color model
    global eq_image_clahe
    eq_image_clahe = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    return eq_image_clahe

def resize_translate():
    resize_ratio = random.randint(7000, 9000)
    resize_ratio = resize_ratio / 10000
    image_name = image.split('/')[-1]
    image_name = image_name.replace(image_type, '')
    annotation = open(("data_augmentation/original_images/" + image_name + ".txt"), "r")
    annotation_rt = open(("data_augmentation/resized_translated/" + image_name + "_rt.txt"), "w")
    new_height, new_width = int(height * resize_ratio), int(width * resize_ratio)
    resized_img = cv2.resize(data, (new_width, new_height))
    blank_image = np.zeros((height, width, 3), np.uint8)
    x_offset = random.randint(0, width - new_width)
    y_offset = random.randint(0, height - new_height)
    blank_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img
    #print(x_offset, y_offset)
    for line in annotation:
        line = line.split()
        annotation_rt.write(line[0] + ' ' + str(round(float(line[1]) * resize_ratio + x_offset / width, 6)) + ' ' + str(
            round(float(line[2]) * resize_ratio + y_offset / height, 6)) + ' ' + str(
            round(float(line[3]) * resize_ratio, 6)) + ' ' + str(round(float(line[4]) * resize_ratio, 6)) + '\n')
    cv2.imwrite('data_augmentation/resized_translated/' + image_name + '_rt'+image_type, blank_image)

def show_results():
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original", 1000, 600)
    cv2.namedWindow("Equalized Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Equalized Image", 1000, 600)
    cv2.namedWindow("Equalized Image_HSV", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Equalized Image_HSV", 1000, 600)
    cv2.namedWindow("Equalized Image_CLAHE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Equalized Image_CLAHE", 1000, 600)
    cv2.imshow("Original", image)
    cv2.imshow("Equalized Image", eq_image)
    cv2.imshow("Equalized Image_HSV", eq_image_hsv)
    cv2.imshow("Equalized Image_CLAHE", eq_image_clahe)
    cv2.waitKey()
    cv2.destroyAllWindows()




    # image = cv2.imread('data_augmentation/original_images/1574949804.459803.bmp',1)
    # image = cv2.imread('../data/test_files/agco_1.bmp',1)


rotated_folder = 'data_augmentation/rotated/'
try:
    os.mkdir(rotated_folder)
except OSError as error:
    shutil.rmtree(rotated_folder)
    os.mkdir(rotated_folder)

equalized_folder = 'data_augmentation/equalized/'
try:
    os.mkdir(equalized_folder)
except OSError as error:
    shutil.rmtree(equalized_folder)
    os.mkdir(equalized_folder)

resized_translated_folder = 'data_augmentation/resized_translated/'
try:
    os.mkdir(resized_translated_folder)
except OSError as error:
    shutil.rmtree(resized_translated_folder)
    os.mkdir(resized_translated_folder)

images_bmp = glob.glob('data_augmentation/original_images/*.bmp')
images_jpg = glob.glob('data_augmentation/original_images/*.jpg')

print('Applying data augmentation...')
print("'Patience is bitter, but it's fruit is sweet - Aristotle'")
counter_0,counter_1,counter_2 = 0,0,0

global image
global image_type
for image in images_bmp:
    choices = [0,1,2]
    random_augmentation = random.choice(choices)
    image_name = image.split('/')[2].rstrip('.bmp')
    if random_augmentation == 0:
        counter_0 += 1
        rotate_image()
        cv2.imwrite('data_augmentation/rotated/'+image_name+'_rot.bmp',rotated_no_b_box)
        annotation.close()
        annotation_rot.close()
    elif random_augmentation == 1:
        counter_1 += 1
        image = cv2.imread(image, 1)
        # histogram()
        # histogram_hsv()
        histogram_clahe_l()
        cv2.imwrite('data_augmentation/equalized/' + image_name + '_eq_clahe_l.bmp', eq_image_clahe)
        shutil.copy("data_augmentation/original_images/"+image_name+".txt","data_augmentation/equalized/"+image_name+"_eq_clahe_l.txt")
        # show_results()
    elif random_augmentation == 2:
        data = cv2.imread(image, 1)
        height, width, channels = data.shape
        image_type = '.bmp'
        if height < 500 and width < 500:
            counter_1 += 1
            image = cv2.imread(image, 1)
            histogram_clahe_l()
            cv2.imwrite('data_augmentation/equalized/' + image_name + '_eq_clahe_l.bmp', eq_image_clahe)
            shutil.copy("data_augmentation/original_images/" + image_name + ".txt",
                        "data_augmentation/equalized/" + image_name + "_eq_clahe_l.txt")
        else:
            counter_2 += 1
            resize_translate()

for image in images_jpg:
    choices = [0, 1, 2]
    #random_augmentation = random.choice(choices)
    random_augmentation = 1
    image_name = image.split('/')[2].rstrip('.jpg')
    if random_augmentation == 0:
        counter_0 += 1
        rotate_image()
        cv2.imwrite('data_augmentation/rotated/'+image_name+'_rot.jpg',rotated_no_b_box)
        annotation.close()
        annotation_rot.close()
    elif random_augmentation == 1:
        counter_1 += 1
        image = cv2.imread(image, 1)
        # histogram()
        # histogram_hsv()
        histogram_clahe_l()
        cv2.imwrite('data_augmentation/equalized/' + image_name + '_eq_clahe_l.jpg', eq_image_clahe)
        shutil.copy("data_augmentation/original_images/" + image_name + ".txt", "data_augmentation/equalized/" + image_name + "_eq_clahe_l.txt")
        # show_results()
    elif random_augmentation == 2:
        data = cv2.imread(image, 1)
        height, width, channels = data.shape

        image_type = '.jpg'
        if height < 300 and width < 300:
            image = cv2.imread(image, 1)
            counter_1 += 1
            histogram_clahe_l()
            cv2.imwrite('data_augmentation/equalized/' + image_name + '_eq_clahe_l.jpg', eq_image_clahe)
            shutil.copy("data_augmentation/original_images/" + image_name + ".txt",
                        "data_augmentation/equalized/" + image_name + "_eq_clahe_l.txt")
        else:
            counter_2 += 1
            resize_translate()

print(counter_0,counter_1,counter_2)





