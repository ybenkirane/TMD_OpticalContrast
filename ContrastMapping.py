# -*- coding: utf-8 -*-
"""
Author: Yacine Benkirane
Supervisor: Peter Grutter
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


def mouse_callback(event, x, y, flags, param):
    global ref_point, cropping, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        img_copy = img.copy()
        cv2.rectangle(img_copy, ref_point[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("image", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        cropping = False
        cv2.rectangle(img_copy, ref_point[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("image", img_copy)
        ref_point.append((x, y))


def plot_gray_value_vs_position(images, titles, extended_crop):
    fig = plt.figure(figsize=(20, 10))

    # Plot the extended crop with the green crop box
    ax1 = fig.add_subplot(2, 4, (1, 4))
    ax1.imshow(cv2.cvtColor(extended_crop, cv2.COLOR_BGR2RGB))
    ax1.set_title('Cropped Region with Surrounding Area')

    # Plot the gray value vs position for the original and individual channels
    for i, (image, title) in enumerate(zip(images, titles)):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        long_side_length = max(gray_image.shape)
        long_side_idx = np.argmax(gray_image.shape)
        positions = np.arange(long_side_length)

        if long_side_idx == 0:
            gray_values = np.mean(gray_image, axis=1)
        else:
            gray_values = np.mean(gray_image, axis=0)

        ax = fig.add_subplot(2, 4, i + 5)
        ax.plot(positions, gray_values)
        ax.set_xlabel('Position')
        ax.set_ylabel('Gray Value')
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


def process_image():
    if len(ref_point) != 2:
        return

    # Crop the image
    cropped_image = img[ref_point[0][1]:ref_point[1]
                        [1], ref_point[0][0]:ref_point[1][0]]

    if cropped_image.size == 0:
        return

    # Get the extended crop with the green crop box
    padding = np.abs(np.subtract(*ref_point)) * 2
    extended_crop_start = np.maximum(np.subtract(ref_point[0], padding), 0)
    extended_crop_end = np.minimum(np.add(ref_point[1], padding), [
                                   img.shape[1], img.shape[0]])
    extended_crop = img[extended_crop_start[1]:extended_crop_end[1],
                        extended_crop_start[0]:extended_crop_end[0]].copy()
    cv2.rectangle(extended_crop, tuple(ref_point[0] - extended_crop_start), tuple(
        ref_point[1] - extended_crop_start), (0, 255, 0), 2)

    # Split the image into 3 channels
    r_channel, g_channel, b_channel = cv2.split(cropped_image)

    # Plot the Gray Value vs Position for the original and individual channels
    images = [
        cropped_image,
        cv2.merge([r_channel, np.zeros_like(
            g_channel), np.zeros_like(b_channel)]),
        cv2.merge([np.zeros_like(r_channel), g_channel,
                  np.zeros_like(b_channel)]),
        cv2.merge([np.zeros_like(r_channel),
                  np.zeros_like(g_channel), b_channel])
    ]
    titles = ['Original Image', 'R Channel', 'G Channel', 'B Channel']
    plot_gray_value_vs_position(images, titles, extended_crop)


# Load the tif image
file_path = "Pos015_100x.tif"
img = cv2.imread(file_path)
img_copy = img.copy()

# Initialize global variables
ref_point = []
cropping = False
center = (img.shape[1] // 2, img.shape[0] // 2)
scale = 1

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_callback)

while True:
    cv2.imshow("image", img_copy)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("d"):  # press 'd' for done
        process_image()
    elif key == ord("q"):  # press 'q' for quit
        break

cv2.destroyAllWindows()
