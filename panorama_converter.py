import cv2
import numpy as np

# Load the spherical panorama image
spherical_image_path = "./data/__FwscCqAFl8v3uNUXf4ow.jpeg"
spherical_image = cv2.imread(spherical_image_path)
height, width, _ = spherical_image.shape

# Define cylindrical panorama dimensions
cylindrical_width = width
cylindrical_height = height

# Define field of view (FoV) parameters
fov_x = 360.0  # degrees
fov_y = (float(cylindrical_height) / cylindrical_width) * fov_x

# Create empty cylindrical image
cylindrical_image = np.zeros_like(spherical_image)

# Create coordinate grid
x_cylinder, y_cylinder = np.meshgrid(np.arange(cylindrical_width), np.arange(cylindrical_height))

# Convert cylindrical coordinates to spherical coordinates
theta = (x_cylinder / cylindrical_width) * (2 * np.pi) - np.pi
phi = (y_cylinder / cylindrical_height) * np.pi - (np.pi / 2)

# Convert spherical coordinates to Cartesian coordinates
x_sphere = np.cos(phi) * np.sin(theta)
y_sphere = np.sin(phi)
z_sphere = np.cos(phi) * np.cos(theta)

# Map Cartesian coordinates to spherical image coordinates
u = 0.5 * width * (1 + np.arctan2(x_sphere, z_sphere) / np.pi)
v = 0.5 * height * (1 - np.arcsin(y_sphere) / (np.pi / 2))

# Ensure coordinates are within image bounds
u = np.clip(u, 0, width - 1).astype(np.int32)
v = np.clip(v, 0, height - 1).astype(np.int32)

# Assign pixel values from spherical image to cylindrical image
cylindrical_image[y_cylinder, x_cylinder] = spherical_image[v, u]

# Save the cylindrical panorama image
cylindrical_image_path = "./output/result.jpeg"
cv2.imwrite(cylindrical_image_path, cylindrical_image)
