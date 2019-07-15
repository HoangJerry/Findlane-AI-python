import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def canny_image(image):
	lane_image = np.copy(image)
	gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	canny = cv2.Canny(blur, 50, 70)
	return canny
def make_coordinates(image, line_parameters):
	print (line_parameters)
	slope, intercept = line_parameters
	y1 = image.shape[0]
	y2 = int(y1*(3/5))
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	return np.array([x1,y1,x2,y2])


def average_line(image, lines):
	left_fit = []
	right_fit = []
	for line in lines:
		x1, y1, x2, y2 = line.reshape(4)
		parameters = np.polyfit((x1, x2), (y1, y2), 1)
		slope = parameters[0]
		intercept = parameters[1]
		if slope <0:
			left_fit.append((slope,intercept))
		else:
			right_fit.append((slope,intercept))
	left_fit_average = np.average(left_fit, axis=0)
	right_fit_average = np.average(right_fit, axis=0)

	left_line = make_coordinates(image, left_fit_average)
	right_line = make_coordinates(image, right_fit_average)
	return np.array([left_line, right_line])

def display_lines(image, lines):
	line_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)
			cv2.line(line_image, (x1,y1), (x2,y2), [110,50,50], 10)
	return line_image

def forcus_region(image):
	height = image.shape[0]
	triangle = np.array([[(200,height),(1100,height),(550,250)]])
	mask = np.zeros_like(image) #Black image same size
	cv2.fillPoly(mask, triangle, 255)
	masked_mage = cv2.bitwise_and(image, mask)
	return masked_mage
# # Image
# image = cv2.imread('screenshot.png')
# lane_image = np.copy(image)
# canny = canny_image(image)
# masked_mage = forcus_region(canny)

# # Detect lines
# lines = cv2.HoughLinesP(masked_mage, 2, np.pi/180, 50, np.array([]), minLineLength=10, maxLineGap=1)
# # Average lines
# average_lines = average_line(lane_image,lines)

# line_image = display_lines(lane_image,lines)


# # Combo image
# print (lane_image.shape, line_image.shape)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

# cv2.imshow('result', combo_image)
# # Waiting for press a key
# cv2.waitKey(0)


# plt.imshow(canny)
# plt.show()

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened):
	_, frame = cap.read()
	canny = canny_image(frame)
	masked_mage = forcus_region(canny)

	# Detect lines
	lines = cv2.HoughLinesP(masked_mage, 2, np.pi/180, 50, np.array([]), minLineLength=10, maxLineGap=5)
	# Average lines
	average_lines = average_line(frame,lines)

	line_image = display_lines(frame,average_lines)


	# Combo image
	# print (frame.shape, line_image.shape)
	combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
	cv2.imshow('result', combo_image)
	cv2.waitKey(3)


