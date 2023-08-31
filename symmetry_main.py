import cv2
import numpy as np
import matplotlib.pyplot as plt

#Load the image
img_path = 'pic.jpeg'
img = cv2.imread(img_path, 0)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.show()

#Binarization
_, img_bin = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
print("Binarization is completed")
plt.imshow(img_bin, cmap='gray')
plt.title('Binarized Image')
plt.show()

#Contour Detection
contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
max_cnt =max(contours, key=lambda x: cv2.contourArea(x))
img_contour = cv2.drawContours(img_bin, [max_cnt], -1, 255, -1)
print("Contour detection is completed.")
plt.imshow(img_contour, cmap='gray')
plt.title('Contour Image')
plt.show()

#Calcurate the Centerline
M = cv2.moments(img_contour)
if M["m00"] != 0:
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # print(f"Centerline coordinates: ({cX}, {cY})")
else:
    print("No object was detected in the image.")

# Draw the Centerline
img_with_centerline = cv2.line(img_contour.copy(), (cX, 0), (cX, img.shape[0] - 1), (255), 1)
plt.imshow(img_with_centerline, cmap='gray')
plt.title('Image with Centerline')
plt.show()

# Preparation for inverting the image around the centerline axis
height, width = img.shape
img_centered = np.zeros_like(img_contour)
shift_x = width // 2 - cX

# Move the object to the center of the image
M = np.float32([[1, 0, shift_x], [0, 1, 0]])
img_centered = cv2.warpAffine(img_contour, M, (width, height))
print("the object moved to the center of the image.")
plt.imshow(img_centered, cmap='gray')
plt.title('Centered Image')
plt.show()

#Invert image
img_flipped = cv2.flip(img_centered, 1)
print("Inverted image")
plt.imshow(img_flipped, cmap='gray')
plt.title('Flipped Image')
plt.show()

#Calculate differences between images
img_difference = cv2.absdiff(img_centered, img_flipped)
print("Differences between images were calculated.")
plt.imshow(img_difference, cmap='gray')
plt.title('Calculated differences between images')
plt.show()

#Calculation of overhang area
difference_num_piexl=cv2.countNonZero(img_difference)
print(f'The overhang area is {difference_num_piexl} pixels')

# Overlay the original image and the inverted image (AND operation)
img_overlapped = cv2.bitwise_and(img_centered, img_flipped)
plt.imshow(img_overlapped, cmap='gray')
plt.title('Overlapped Image')
plt.show()

#Calculation of area of overlapped area
and_num_piexl=cv2.countNonZero(img_overlapped)

#Overall area
sum_num_piexl = and_num_piexl + difference_num_piexl
print(f'The overall area is {sum_num_piexl} pixels')

symmetry = difference_num_piexl / sum_num_piexl
symmetry = symmetry * -1 + 1
print(f'The symmetry of this image is {symmetry}')
