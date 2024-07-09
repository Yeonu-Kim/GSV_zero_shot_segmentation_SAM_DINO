import numpy as np
import cv2

def clamp(x, min_val, max_val):
    return max(min_val, min(max_val, x))

def mod(x, n):
    return ((x % n) + n) % n

def copy_pixel_nearest(read, write):
    height, width = read.shape[:2]

    def func(x_from, y_from, to):
        nearest_x = clamp(int(round(x_from)), 0, width - 1)
        nearest_y = clamp(int(round(y_from)), 0, height - 1)
        write[to] = read[nearest_y, nearest_x]

    return func

def copy_pixel_bilinear(read, write):
    height, width = read.shape[:2]

    def func(x_from, y_from, to):
        xl = clamp(int(np.floor(x_from)), 0, width - 1)
        xr = clamp(int(np.ceil(x_from)), 0, width - 1)
        xf = x_from - xl

        yl = clamp(int(np.floor(y_from)), 0, height - 1)
        yr = clamp(int(np.ceil(y_from)), 0, height - 1)
        yf = y_from - yl

        for channel in range(3):
            p0 = read[yl, xl, channel] * (1 - xf) + read[yl, xr, channel] * xf
            p1 = read[yr, xl, channel] * (1 - xf) + read[yr, xr, channel] * xf
            write[to + channel] = p0 * (1 - yf) + p1 * yf

    return func

def kernel_resample(read, write, filter_size, kernel):
    height, width = read.shape[:2]
    two_filter_size = 2 * filter_size

    def func(x_from, y_from, to):
        xl = int(np.floor(x_from))
        yl = int(np.floor(y_from))
        x_start = xl - filter_size + 1
        y_start = yl - filter_size + 1

        x_kernel = np.array([kernel(x_from - (x_start + i)) for i in range(two_filter_size)])
        y_kernel = np.array([kernel(y_from - (y_start + i)) for i in range(two_filter_size)])

        for channel in range(3):
            q = 0
            for i in range(two_filter_size):
                y = y_start + i
                y_clamped = clamp(y, 0, height - 1)
                p = 0
                for j in range(two_filter_size):
                    x = x_start + j
                    x_clamped = clamp(x, 0, width - 1)
                    p += read[y_clamped, x_clamped, channel] * x_kernel[j]
                q += p * y_kernel[i]
            write[to + channel] = q

    return func

def copy_pixel_bicubic(read, write):
    b = -0.5
    def kernel(x):
        x = abs(x)
        if x <= 1:
            return (b + 2) * x**3 - (b + 3) * x**2 + 1
        elif x < 2:
            return b * x**3 - 5 * b * x**2 + 8 * b * x - 4 * b
        else:
            return 0

    return kernel_resample(read, write, 2, kernel)

def copy_pixel_lanczos(read, write):
    filter_size = 5
    def kernel(x):
        if x == 0:
            return 1
        else:
            xp = np.pi * x
            return filter_size * np.sin(xp) * np.sin(xp / filter_size) / (xp * xp)

    return kernel_resample(read, write, filter_size, kernel)

def orientations(face):
    if face == 'pz':
        return lambda out, x, y: (out.__setitem__('x', -1), out.__setitem__('y', -x), out.__setitem__('z', -y))
    elif face == 'nz':
        return lambda out, x, y: (out.__setitem__('x', 1), out.__setitem__('y', x), out.__setitem__('z', -y))
    elif face == 'px':
        return lambda out, x, y: (out.__setitem__('x', x), out.__setitem__('y', -1), out.__setitem__('z', -y))
    elif face == 'nx':
        return lambda out, x, y: (out.__setitem__('x', -x), out.__setitem__('y', 1), out.__setitem__('z', -y))
    elif face == 'py':
        return lambda out, x, y: (out.__setitem__('x', -y), out.__setitem__('y', -x), out.__setitem__('z', 1))
    elif face == 'ny':
        return lambda out, x, y: (out.__setitem__('x', y), out.__setitem__('y', -x), out.__setitem__('z', -1))

def render_face(read_image, face, rotation, interpolation, max_width=np.inf):
    height, width = read_image.shape[:2]
    face_width = min(max_width, width // 4)
    face_height = face_width

    write_image = np.zeros((face_height, face_width, 3), dtype=np.uint8)  # Use 3 channels for RGB

    copy_pixel_func = {
        'nearest': copy_pixel_nearest,
        'linear': copy_pixel_bilinear,
        'cubic': copy_pixel_bicubic,
        'lanczos': copy_pixel_lanczos
    }[interpolation](read_image, write_image)

    orientation_func = orientations(face)

    for x in range(face_width):
        for y in range(face_height):
            to = (y, x)  # Remove slice(None) for 3-channel image

            # Get position on cube face
            cube = {}
            orientation_func(cube, (2 * (x + 0.5) / face_width - 1), (2 * (y + 0.5) / face_height - 1))

            # Project cube face onto unit sphere
            r = np.sqrt(cube['x']**2 + cube['y']**2 + cube['z']**2)
            lon = mod(np.arctan2(cube['y'], cube['x']) + rotation, 2 * np.pi)
            lat = np.arccos(cube['z'] / r)

            copy_pixel_func(width * lon / (2 * np.pi) - 0.5, height * lat / np.pi - 0.5, to)

    return write_image

def panorama_to_cubemap(image):
    # Load the spherical panorama image
    image_path = './data/__FwscCqAFl8v3uNUXf4ow.jpeg'
    read_image = cv2.imread(image_path)
    read_image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)  # Convert to RGB if it's BGR

    # Define the size of each cubemap face
    face_size = 512

    # Define the faces of the cubemap
    faces = ['pz', 'nz', 'px', 'nx', 'py', 'ny']
    rotations = [0, 0, 0, 0, 0, 0]

    # Generate cubemap faces
    cubemap_faces = {}
    for face, rotation in zip(faces, rotations):
        cubemap_faces[face] = render_face(read_image, face, rotation, 'nearest', face_size)

    return cubemap_faces
    # Display the cubemap faces

