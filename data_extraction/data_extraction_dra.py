from PIL import Image
import cv2
import numpy as np
import os
import math
import pandas as pd


def angle_between_line_and_vertical_line(m):
    angle_radians = math.atan(m)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def alpha_line(x, y, alpha):

    if math.isnan(alpha):
        return 0, 0

    length=150
    # Convert angle from degrees to radians
    alpha_rad = np.radians(alpha)

    # Calculate direction vector (dx, dy)
    # Since alpha is with respect to vertical, we swap sin and cos
    dx = np.sin(alpha_rad)
    dy = np.cos(alpha_rad)

    # Calculate end point of the line
    x_end = x + length * dx
    y_end = y + length * dy

    return (x,y), (int(x_end), int(y_end))

def extract_data(image):
     # Blur the img
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Erode the blurred image
    kernel = np.ones((7, 7), np.uint8)
    erosion = cv2.erode(blurred, kernel, iterations=1)

    # Detect edges on the eroded image
    edges = cv2.Canny(erosion, 50, 80)

    # Apply the Hough Line Transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 30)  # 250

    # Count the number of detected lanes
    if lines is not None:
        num_lanes = len(lines)
        print(f"Number of detected lanes: {num_lanes}")

    # Extract line parameters and draw bold lines
    m_neg_list, n_neg_list = [],  []
    m_pos_list, n_pos_list = [],  []

    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            m, n = rho / np.sin(theta), -1 / np.tan(theta)

            if n < 0:
                m_neg_list.append(m)
                n_neg_list.append(n)
            else:
                m_pos_list.append(m)
                n_pos_list.append(n)
    
        m_neg, n_neg = np.mean(m_neg_list), np.mean(n_neg_list)
        m_pos, n_pos = np.mean(m_pos_list), np.mean(n_pos_list)
        print(f'y = {m_neg:.2f} + {n_neg:.2f}*x')
        print(f'y = {m_pos:.2f} + {n_pos:.2f}*x')

        # Function to calculate y for a given x using the line equation
        def calculate_y(m, n, x):
            if math.isnan(m) or math.isnan(n):
                return 0
            y = int(n * x + m)
            return y
        
        # Function to calculate x for a given y using the line equation
        def calculate_x(m, n, y):
            if math.isnan(m) or math.isnan(n):
                return 0
            return int((y - m) / n) if n != 0 else 0
    #     def calculate_x(y, m, n):
    # # Check if any of the inputs are NaN
    #         if any(pd.isna(value) for value in [y, m, n]):
    #             return 0  # or return some other appropriate value or action
            
    #         # Perform the calculation if n is not zero
    #         try:
    #             result = (y - m) / n if n != 0 else 0
    #             return int(result)  # Convert to int, but this will fail if result is NaN
    #         except ValueError:
    #             # Handle the case where result is NaN
    #             return 0 

        # Define the y-interval
        y1_interval = height - 100
        y2_interval = height

        x1_neg = calculate_x(m_neg, n_neg, y1_interval)
        x2_neg = calculate_x(m_neg, n_neg, y2_interval)
        x1_pos = calculate_x(m_pos, n_pos, y1_interval)
        x2_pos = calculate_x(m_pos, n_pos, y2_interval)

        cv2.line(color_image, (x1_neg, y1_interval), (x2_neg, y2_interval), (0, 255, 255), 5)  # Negative slope line
        cv2.line(color_image, (x2_neg, y1_interval), (x2_neg, y2_interval), (0, 100, 255), 5)  # Negative vertical line
        cv2.line(color_image, (x1_pos, y1_interval), (x2_pos, y2_interval), (0, 255, 255), 5)  # Positive slope line
        cv2.line(color_image, (x2_pos, y1_interval), (x2_pos, y2_interval), (0, 100, 255), 5)  # Positive vertical line

        # Draw line of direction 
        alpha_neg = angle_between_line_and_vertical_line(n_neg)
        alpha_pos = angle_between_line_and_vertical_line(n_pos)
        alpha = (alpha_neg + alpha_pos) / 2
        print('▶️ alpha:', alpha)
        x1_alpha = (x2_neg + x2_pos)//2
        y1_alpha = height
        x2_alpha, y2_alpha = alpha_line(x1_alpha, y1_alpha, alpha)[1]
        cv2.line(color_image, (x1_alpha, y1_alpha), (x2_alpha, y1_alpha - abs(y2_alpha - y1_alpha)), (255, 0, 0), 5)  # alpha line
        cv2.line(color_image, (x1_alpha, y1_alpha), (x1_alpha, y1_alpha - (y2_alpha - y1_alpha)), (255, 0, 100), 5)  # vertical line

    else:
        print('▶️ alpha:', 100)

    return color_image