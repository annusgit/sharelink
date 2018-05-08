

"""

    The following is our project for our instruments and measurements course
"""

from __future__ import print_function
from __future__ import division

import time
import cv2
import numpy as np

import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *

def get_diff_image(prev_prev, prev, current):

    """
        basically calculates the difference of 3 frames for background cancellation
    :param prev_prev: at t-2
    :param prev: at t-1
    :param current: at t
    :return: diff image of the last three frames
    """

    diff_1 = cv2.absdiff(current, prev)
    diff_2 = cv2.absdiff(prev, prev_prev)
    return cv2.bitwise_and(diff_1, diff_2)


def get_bounding_boxes(img, contours):
    """
        this function converts contours into bounding rectangles
    :param img:
    :param contours:
    :return:
    """
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        print(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    return img


def get_enclosing_circle(img, contours):
    """
        this function converts contours into bounding circles
    :param img:
    :param contours:
    :return:
    """

    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        if radius >= 40:
            cv2.circle(img, center, radius, (0, 255, 0), 5)
    return img



def detect_motion():

    # get the image feed and store 3 frames at each time step
    camera = cv2.VideoCapture(0)
    _, current_frame = camera.read() # get some initial settings
    # we'll make our diff_frame a difference of 3 consecutive frames for background cancellation
    # and after reading them, must convert them to gray scale first
    prev_prev_frame = cv2.cvtColor(cv2.medianBlur(camera.read()[1], ksize=5), cv2.COLOR_BGR2GRAY)
    prev_frame = cv2.cvtColor(cv2.medianBlur(camera.read()[1], ksize=5), cv2.COLOR_BGR2GRAY)
    current_frame_rbg = cv2.medianBlur(camera.read()[1], ksize=5)
    current_frame = cv2.cvtColor(current_frame_rbg, cv2.COLOR_BGR2GRAY)

    window_name = 'Video Feed'; cv2.namedWindow(winname=window_name)
    while(True):
        diff_frame = get_diff_image(prev_prev=prev_prev_frame, prev=prev_frame, current=current_frame)
        # gray_frame = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
        # (thresh, binarized_frame) = cv2.threshold(diff_frame, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # binarize with adaptive threshold
        binarized_frame = cv2.adaptiveThreshold(diff_frame.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, 41, 3)
        # find the contours in this image
        # contours = None
        im, contours, hierarchy = cv2.findContours(binarized_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # draw those contours now
        # cv2.drawContours(current_frame_rbg, contours, -1, (0,255,0), 3)
        # cv2.drawContours(current_frame_rbg, contours, -1, (0, 255, 0), 3)

        # get circles bounding these contours
        current_frame_rbg = get_enclosing_circle(img=current_frame_rbg, contours=contours)

        # get rects bounding these contours
        # current_frame_rbg = get_bounding_boxes(img=current_frame_rbg, contours=contours)

        # or may be draw bounding boxes
        if False:
            for component in zip(contours, hierarchy):
                currentContour = component[0]
                # currentHierarchy = component[1]
                x, y, w, h = cv2.boundingRect(currentContour)
            #     if len(currentHierarchy) > 1:
                    # if currentHierarchy[2] < 0:
                        # these are the innermost child components
                cv2.rectangle(current_frame_rbg, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    # elif currentHierarchy[3] < 0:
                        # these are the outermost parent components
                        # cv2.rectangle(current_frame_rbg, (x, y), (x + w, y + h), (0, 255, 0), 3)




        # cv2.drawContours(current_frame_rbg, contours, -1, (0, 255, 0), 3)
        # prev_frame = current_frame
        cv2.imshow(window_name, current_frame_rbg)

        # update the image frames
        prev_prev_frame = prev_frame
        prev_frame = current_frame
        current_frame_rbg = cv2.medianBlur(camera.read()[1], ksize=5)
        current_frame = cv2.cvtColor(current_frame_rbg, cv2.COLOR_BGR2GRAY)

        if cv2.waitKey(delay=10) == 27:
            break # basically this means ESC key

    cv2.destroyAllWindows()
    pass


def flip_horizontal(source):
    #flip the img horizontally
    img = source.reshape(-1, 96, 96)
    img = np.flip(img, axis=2)
    img = img.reshape(-1, 96*96)
    return img


def detect_lines(source):

    current_frame = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    # detector = cv2.LineSegmentDetector()
    detector = cv2.createLineSegmentDetector()
    lines = detector.detect(current_frame)[0]
    return detector.drawSegments(source, lines)



def function():
    """
        We will use this function to apply some transformations on our camera feed
    :return:
    """

    cam = cv2.VideoCapture(1)
    # while cam.read()[1].empty:
    #     pass

    kernel_1 = np.asarray([[-1,-2,-1],[0,0,0],[1,2,1]])
    kernel_2 = np.transpose(kernel_1)
    while True:
        frame = cam.read()[1]
        # frame = np.flip(cam.read()[1], axis=1)
        trans_frame_1 = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel_1)
        trans_frame_2 = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel_2)
        combined_trans = np.add(trans_frame_1, trans_frame_2)
        edge_enhanced_image = np.add(frame, combined_trans)
        # show_frame = np.hstack((frame, combined_trans))
        show_frame = np.hstack((frame, combined_trans))
        # cv2.imshow('Video feed', show_frame)
        cv2.imshow('Video feed', detect_lines(frame))
        if cv2.waitKey(1) == 27: # esc key
            break
    pass


def detect_lines_hough(source):
    current_frame = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(current_frame, 75, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50)
    # print(lines)
    if lines is not None:
        for line in lines:
            x0, y0, x1, y1 = line[0]
            cv2.line(source, (x0, y0), (x1, y1), (0, 255, 0), 3)
    return source


def no_grid_function():

    verticies = (
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, -1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, -1, 1),
        (-1, 1, 1)
    )

    edges = (
        (0, 1),
        (0, 3),
        (0, 4),
        (2, 1),
        (2, 3),
        (2, 7),
        (6, 3),
        (6, 4),
        (6, 7),
        (5, 1),
        (5, 4),
        (5, 7)
    )

    def Cube():
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(verticies[vertex])
        glEnd()

    def thisone():
        pygame.init()
        display = (800, 600)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

        gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

        glTranslatef(0.0, 0.0, -5)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            glRotatef(1, 3, 1, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            Cube()
            pygame.display.flip()
            pygame.time.wait(10)

    thisone()


def grid():
    scr_width = 1200; scr_height = 700; grid_spacing = 50; line_width = 1; delay = 0.05
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode([scr_width,scr_height])
    pygame.display.set_caption('localize that shit!')
    colors = {'white': (255, 255, 255), 'black': (0, 0, 0), 'red': (255, 0, 0),
              'green': (0, 255, 0), 'blue': (0, 0, 255)}
    screen.fill(colors['black'])

    # get all the vertices for our localization grid
    localization_grid = []
    for col in range(1, scr_width // grid_spacing):
        for row in range(1, scr_height // grid_spacing):
            # the first two points are for the column and the other two are for the row
            localization_grid.append([(col * grid_spacing, 0), (col * grid_spacing, scr_height),
                                      (0, row * grid_spacing), (scr_width, row * grid_spacing)])
    localization_grid = np.asarray(localization_grid)
    over = False
    while not over:
        # get some events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                over = True
            elif event.type == pygame.KEYDOWN:
                # let's see which key was pressed
                key_pressed = event.key

        # now draw the grid
        for points in localization_grid[:,]:
            pygame.draw.lines(screen, colors['white'], False, [points[0], points[1]], line_width)
            pygame.draw.lines(screen, colors['white'], False, [points[2], points[3]], line_width)
        pygame.display.flip()
        time.sleep(delay)


def images_for_dataset():

    import os
    import shutil

    cam = cv2.VideoCapture(1)
    save_folder = 'dataset/images/fool'
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.mkdir(save_folder)
    counter = 0

    while True:
        frame = cam.read()[1]
        # handle key events
        key_pressed = cv2.waitKey(1)
        if key_pressed == 27:
            break

        elif key_pressed == 32:
            counter += 1
            cv2.imwrite(os.path.join(save_folder, '{}.jpg'.format(counter)), frame)
            print('your image {} has been saved'.format(os.path.join(save_folder, '{}.jpg'.format(counter))))

        cv2.imshow('Video feed', frame)


def main():

    cam = cv2.VideoCapture(1)
    while True:
        # detections = detect_lines_hough(cam.read()[1])
        frame = cam.read()[1]
        # resized_frame = cv2.resize(frame, (128, 128))
        # handle key events
        key_pressed = cv2.waitKey(1)
        if key_pressed == 27:
            break
        # show the frames

        cv2.imshow('Video feed', frame)

    # another_one()
    # grid()
    pass


def imageman():
    size = 128
    image = cv2.imread('/home/annus/PycharmProjects/instruments_chair_project/dataset/images/IMG-20180427-00082.jpg')
    image = cv2.resize(image, (size, size))
    cv2.imshow('this', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


















