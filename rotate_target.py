import cv2
import numpy as np
from PIL import Image


def rotate_image(image : np.ndarray):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Display the grayscale image
    # cv2.imshow('Grayscale', gray)
    # cv2.waitKey(0)

    # Threshold the image to get the white regions
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # # Display the thresholded image
    # cv2.imshow('Thresholded', thresh)
    # cv2.waitKey(0)

    # Find the largest rectangle that fits inside the white region
    # First, perform distance transform to find the distance to the nearest black pixel
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

    # Find the maximum distance and its location
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)

    # The maximum value in the distance transform is the radius of the largest inscribed circle
    # The location of this maximum is the center of the largest inscribed circle
    radius = int(max_val)
    region_center_x, region_center_y = max_loc


    # # Create a visualization image to show the largest inscribed circle
    circle_image = image.copy()
    # Draw the largest inscribed circle
    # cv2.circle(circle_image, (region_center_x, region_center_y), radius, (0, 255, 0), 2)  # Draw the circle
    # cv2.circle(circle_image, (region_center_x, region_center_y), 3, (0, 0, 255), -1)  # Mark the center
    # cv2.imshow('Largest Inscribed Circle', circle_image)
    # cv2.waitKey(0)

    # Calculate moments of the binary image to find center of mass of white pixels
    M = cv2.moments(thresh)
    if M["m00"] != 0:
        # Calculate center of mass
        cx = int((int(M["m10"] / M["m00"]) + region_center_x) / 2)
        cy = int((int(M["m01"] / M["m00"]) + region_center_y) / 2)

        # print(cx, cy)
        
        # Draw the center of mass on a copy of the original image
        center_of_mass_image = image.copy()
        # cv2.circle(center_of_mass_image, (cx, cy), 5, (0, 0, 255), -1)
        # cv2.imshow('Center of Mass of White Pixels', center_of_mass_image)
        # cv2.waitKey(0)
        
        # Get the image center
        height, width = image.shape[:2]
        center_x = width // 2
        center_y = height // 2
        
        # Draw the image center and vector to center of mass
        center_image = center_of_mass_image.copy()
        # cv2.circle(center_image, (center_x, center_y), 5, (0, 255, 255), -1)
        # cv2.line(center_image, (center_x, center_y), (cx, cy), (255, 255, 0), 2)
        # cv2.imshow('Center and Vector to Mass', center_image)
        # cv2.waitKey(0)
        # Calculate the angle between the horizontal line from the left edge to center
        # and the line from center to center of mass
        
        dx = cx - center_x
        dy = cy - center_y
        
        theta = np.degrees(np.arctan2(dy, dx))

        # print(theta)
        
        # Compute the rotation angle so that the red point lies directly to the left.
        # "Left" means an angle of 180° from the positive x-axis.
        rotation_angle = 180 + theta
        # print(f"Current angle: {theta:.2f} degrees, Rotating by: {rotation_angle:.2f} degrees")
        
        # Create the rotation matrix using the yellow point as the center of rotation
        M = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
        
        # Apply the rotation to the image
        (h, w) = image.shape[:2]
        rotated = cv2.warpAffine(image, M, (w, h))
        
        # Display the rotated image (this works if you're running in an environment that supports GUI windows)
        # cv2.imshow("Rotated Image", rotated)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return rotated
    return image

if __name__ == "__main__":
    for i in range(10):
        image = cv2.imread(f"{i}.png")
        rotated = rotate_image(image)
        cv2.imwrite(f"rotated_{i}.png", rotated)