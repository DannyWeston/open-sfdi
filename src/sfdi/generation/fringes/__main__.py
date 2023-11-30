import numpy as np
import cv2
import os

from sfdi.definitions import FRINGES_DIR

def binary(width, height, freq, phase, orientation):
    """
    Creates a binary fringe pattern, similar to a PWM signal.

    Args:
        width (int): Width of the fringe pattern (pixels).
        height (int): Height of the fringe pattern (pixels).
        freq (float): Spatial frequency of the fringes.
        orientation (float): Orientation of the fringes (0 = horizontal, 2pi = vertical).

    Returns:
        ndarray[uint8]: Binary fringe pattern.
    """ 

    # Maybe not a good idea to rely upon sinusoidal function
    # But works for now :)

    img = sinusoidal(width, height, freq, phase, orientation)
    width, height = img.shape
    for col in range(width):
        for row in range(height):
            img[col][row] = 0 if img[col][row] < 0 else 1

    return img #img.astype(np.uint8)

def sinusoidal(width, height, freq, phase=0, orientation=(np.pi / 2.0)):
    """
    Creates a sinusoidal fringe pattern (values between -1 and 1)
 
    Args:
        width (int): Width of the fringe pattern (pixels).
        height (int): Height of the fringe pattern (pixels).
        freq (float): Spatial frequency of the fringes.
        orientation (float): Orientation of the fringes (0 = horizontal, 2pi = vertical).
 
    Returns:
        ndarray[float32]: Sinusoidal fringe pattern.
    """
    x, y = np.meshgrid(np.arange(width, dtype=int), np.arange(height, dtype=int))

    gradient = np.sin(orientation) * x - np.cos(orientation) * y

    return np.sin(((2.0 * np.pi * gradient) / freq) + phase)

def normalise_image(img):
    """
    Normalises an image's pixel values to be: 0 <= x <= 1

    Args:
        img (np.ndarray): Image to be normalised.

    Returns:
        np.ndarray: Normalised image.
    """
    return ((img - img.min()) / (img.max() - img.min()))

def phase_animation(f_type, width: int, height: int, freq: float, orientation: float, frame_rate: int=30):
    """
    Displays a preview of a given fringe pattern with changing phase.
 
    Args:
        f_type (function): Type of fringe pattern type (which function to use).
        width (int): Width of image.
        height (int): Height of image.
        freq (float): Spatial frequency of the fringes.
        orientation (float): Orientation of the fringes (0 = horizontal, 2pi = vertical).
        frame_rate (int): Frame rate of the animation which will be played.
 
    Returns:
        None
    """
    
    counter = 0

    while True:
        img = f_type(width, height, freq, counter, orientation)
        img = cv2.resize(img, (int(width / 3), int(height / 3)))  

        cv2.imshow('Phase Shifting', img)
        key = cv2.waitKey(int(1000 / frame_rate)) # Roughly 30 fps
        counter += 0.05
        if key == 27:
            cv2.destroyAllWindows()
            break

def show_image(img, title='Preview'):
    cv2.imshow(title, img)
    cv2.waitKey(0)

def generate_images(f_type, width: int, height: int, freq: float, orientation: float, n: int=3):
    """
    Generates a collection of fringe images with phase = 2pi * k / n for a given n,
    such that k ∈ {1..n}.
 
    Args:
        f_type (function): Type of fringe pattern to generate.
        width (int): Width of image.
        height (int): Height of image.
        freq (float): Spatial frequency of the fringes.
        orientation (float): Orientation of the fringes (0 = horizontal, 2pi = vertical).
        n (int): Number of different phases.
 
    Returns:
        list[nd.array]: List of n fringe patterns (images). 
    """

    imgs = []

    for i in range(n):
        img = f_type(width, height, freq, 2 * i * np.pi / n, orientation)

        # Normalise and convert to rgb8 range (change to correct datatype)
        img = (normalise_image(img) * 255).astype(np.uint8)

        imgs.append(img)

    return imgs

def save_image(img, path):
    """
    Save an image to disk.
 
    Args:
        path (str): Path to the new file to save the image in.
        img (list[nd.array]): Image to save.
 
    Returns:
        bool: Whether it was successful or not.
    """
    cv2.imwrite(path, img)

# Useful sf values for certain projectors with properties: 
#   - Resolution:                           1024x1024
#   - Number of line pairs:                 8
#   - Size:                                 1m x 1m
#   - Camera tilt:                          0 degrees
#   - Projector distance to baseline stage: 0.5m
#   - Projector FOV:                        -
projector_sfs = {
    'LG' : 32,
    'BLENDER' : 128
}

width = 1024
height = 1024

freq = projector_sfs['BLENDER']
orientation = np.pi / 2.0
imgs = generate_images(binary, width, height, freq / 2, orientation)

for i, img in enumerate(imgs):
    save_image(img, os.path.join(FRINGES_DIR, f'fringes_{i}.jpg'))

#phase_animation(binary, width, height, freq, orientation)