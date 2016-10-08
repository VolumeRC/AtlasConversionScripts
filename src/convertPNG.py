#!/usr/bin/env python
print """
This code was created by Luis Kabongo, Vicomtech-IK4 Copyright 2012-2013.
This application converts the slices found in a folder into a tiled 2D texture 
image in PNG format (it assumes all files in the folder are of the same type 
and dimensions). It uses Python with PIL, numpy and pydicom packages are 
recommended for other formats.
Information links:
http://www.volumerc.org
http://demos.vicomtech.org
Contact mailto:volumerendering@vicomtech.org
"""

import os
import errno
import sys
import math
from multiprocessing import cpu_count
import tempfile
# this is required to manage the images
try:
    from PIL import Image
except ImportError:
    import Image

# This is the default size when loading a Raw image
sizeOfRaw = (512, 512)
# This determines if the endianness should be reversed
rawByteSwap = True
# Standard deviation for Gaussian kernel 
sigmaValue = 1


# This function simply loads a PNG file and returns a compatible Image object
def load_png(filename):
    im = Image.open(filename)
    if im.mode != 1:
        return im.convert("L", palette=Image.ADAPTIVE, colors=256)
    return im

# Simple decrement function
def decr(x, y):
    return x - y

# Normalize values between [0-1]
def normalize(block):
    old_min = delayed(block.min())
    r = delayed(decr)(block.max(), old_min)
    t0 = decr(block, old_min.compute())
    return t0 / r.compute()

# Calculate derivatives function
def gaussian_filter(block, axis):
    return ndimage.gaussian_filter1d(block, sigma=sigmaValue, axis=axis, order=1)

# This function calculates the gradient from a 3 dimensional dask array
def calculate_gradient(arr):
    axises = [1, 0, 2]  # Match RGB
    g = da.ghost.ghost(arr, depth={0: 1, 1: 1, 2: 1},  boundary={0: 'periodic', 1: 'periodic', 2: 'reflect'})
    derivatives = [g.map_blocks(gaussian_filter, axis) for axis in axises]
    derivatives = [da.ghost.trim_internal(d, {0: 1, 1: 1, 2: 1}) for d in derivatives]
    gradient = da.stack(derivatives, axis=3)
    return normalize(gradient)


def resize_image(im, width, height, _filter=Image.BICUBIC):
    if width is not None or height is not None:
        original_size = im.size
        if width is None:
            width = original_size[0]
        if height is None:
            height = original_size[1]
        size = (width, height)
        return im.resize(size, _filter)
    return im


def make_square_image(im):
    mode = im.mode
    width, height = im.size
    new_background = 0  # L, 1
    if len(mode) == 3:  # RGB
        new_background = (0, 0, 0)
    if len(mode) == 4:  # RGBA, CMYK
        new_background = (0, 0, 0, 0)
    new_resolution = max(width, height)
    offset = ((new_resolution - width) / 2, (new_resolution - height) / 2)
    t_im = Image.new("L", (new_resolution, new_resolution), new_background)
    t_im.paste(im, offset)
    return t_im


def read_image(filename, load_img_func=load_png, r_width=None, r_height=None):
    # Load the image
    im = load_img_func(filename)
    # Perform resize if required
    im = resize_image(im, r_width, r_height)
    # Create an square image if required
    width, height = im.size
    if width != height:
        return make_square_image(im)
    return im


# This function uses the images retrieved with loadImgFunction (whould return a PIL.Image) and
# writes them as tiles within a new square Image.
# Returns a set of Image, size of a slice, number of slices and number of slices per axis
def ImageSlices2TiledImage(filenames, loadImgFunction=load_png, cGradient=False, r_width=None, r_height=None):
    filenames = sorted(filenames)
    print "Desired load function=", loadImgFunction.__name__
    size = read_image(filenames[0], loadImgFunction, r_width, r_height).size
    numberOfSlices = len(filenames)
    slicesPerAxis = int(math.ceil(math.sqrt(numberOfSlices)))
    imout = Image.new("L", (size[0] * slicesPerAxis, size[1] * slicesPerAxis))

    i = 0
    for filename in filenames:
        im = read_image(filename, loadImgFunction, r_width, r_height)

        row = int((math.floor(i / slicesPerAxis)) * size[0])
        col = int((i % slicesPerAxis) * size[1])

        box = (int(col), int(row), int(col + size[0]), int(row + size[1]))
        imout.paste(im, box)

        i += 1
        print "processed slice  : " + str(i) + "/" + str(numberOfSlices)  # filename

    gradient = None
    if cGradient:
        print "Starting to compute the gradient: Loading the data..."
        image_list = [da.from_array(np.array(read_image(f, loadImgFunction, r_width, r_height),
                                             dtype='uint8'), chunks=size) for f in filenames]
        data = da.stack(image_list, axis=-1)
        cpus = cpu_count()
        chunk_size = [x//cpus for x in data.shape]
        print "Calculated chunk size: "+str(chunk_size)
        data = da.rechunk(data, chunks=chunk_size)
        print "Loading complete. Data size: "+str(data.shape)
        print "Computing the gradient..."
        data = data.astype(np.float32)
        gradient_data = calculate_gradient(data)
        # Normalize values to RGB values
        gradient_data = gradient_data * 255
        gradient_data = gradient_data.astype(np.uint8)
        # Keep the RGB information separated, uses less RAM memory
        channels = ['/r', '/g', '/b']
        f = tempfile.NamedTemporaryFile(delete=False)
        [da.to_hdf5(f.name, c, gradient_data[:, :, :, i]) for i, c in enumerate(channels)]
        print "Computed gradient data saved in cache file."
        # Create atlas image
        gradient = Image.new("RGB", (size[0] * slicesPerAxis, size[1] * slicesPerAxis))

        channels = ['/r', '/g', '/b']
        handle = h5py.File(f.name)
        dsets = [handle[c] for c in channels]
        arrays = [da.from_array(dset, chunks=chunk_size) for dset in dsets]
        gradient_data = da.stack(arrays, axis=-1)

        for i in range(0, numberOfSlices):
            row = int((math.floor(i / slicesPerAxis)) * size[0])
            col = int((i % slicesPerAxis) * size[1])
            box = (int(col), int(row), int(col + size[0]), int(row + size[1]))

            s = gradient_data[:, :, i, :]
            im = Image.fromarray(np.array(s))
            gradient.paste(im, box)
            print "processed gradient slice  : " + str(i) + "/" + str(numberOfSlices)  # filename

        try:
            handle.close()
            f.close()
        finally:
            try:
                os.remove(f.name)
            except OSError as e:  # this would be "except OSError, e:" before Python 2.6
                if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
                    raise  # re-raise exception if a different error occurred
    return imout, gradient, size, numberOfSlices, slicesPerAxis


# This functions takes a (tiled) image and writes it to a png file with base filename outputFilename.
# It also writes several versions in different sizes determined by dimensions
def write_versions(tileImage, tileGradient, outputFilename, dimensions=None):
    if dimensions is None:
        dimensions = [8192, 4096, 2048, 1024, 512]
    try:
        print 'Creating folder', os.path.dirname(outputFilename), '...',
        os.makedirs(os.path.dirname(outputFilename))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(os.path.dirname(outputFilename)):
            print 'was already there.'
        else:
            print ', folders might not be created, trying to write anyways...'
    except:
        print "Could not create folders, trying to write anyways..."

    print "Writing complete image: " + outputFilename + "_full.png"
    try:
        tileImage.save(outputFilename + "_full.png", "PNG")
        if tileGradient:
            tileGradient.save(outputFilename + "_gradient_full.png", "PNG")
    except:
        print "Failed writing ", outputFilename + "_full.png"
    for dim in dimensions:
        if tileImage.size[0] > dim:
            print "Writing " + str(dim) + "x" + str(dim) + " version: " + outputFilename + "_" + str(dim) + ".png"
            try:
                tmpImage = tileImage.resize((dim, dim))
                tmpImage.save(outputFilename + "_" + str(dim) + ".png", "PNG")
            except:
                print "Failed writing ", outputFilename, "_", str(dim), ".png"
            if tileGradient:
                try:
                    tmpImage = tileGradient.resize((dim, dim))
                    tmpImage.save(outputFilename + "_gradient_" + str(dim) + ".png", "PNG")
                except:
                    print "Failed writing ", outputFilename, "_gradient_", str(dim), ".png"


# This function lists the files within a given directory dir
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


# This is the main program, it takes at least 2 arguments <InputFolder> and <OutputFilename>
def main(argv=None):
    print "Parsing arguments..."
    if argv is None:
        argv = sys.argv

    if len(argv) < 3:
        print "Usage: command <InputFolder> <OutputFilename> [width] [height]"
        print "	<InputFolder> must contain only one set of PNG files to be processed"
        print "	<OutputFilename> must contain the path and base name of the desired output, extension will be added " \
              "automatically"
        print " [width] max width for the slices, it will resize the slice before computing the atlas"
        print " [height] max height for the slices, it will resize the slice before computing the atlas"
        print "Note: this version does not process several folders recursively. "
        print "You typed: ", argv
        return 2

    # Filter only png files in the given folder
    filenames_png = filter(lambda x: ".png" in x, listdir_fullpath(argv[1]))

    width = int(argv[3]) if len(argv) > 3 else None
    height = int(argv[4]) if len(argv) > 4 else None

    # Convert into a tiled image
    if len(filenames_png) > 0:
        try:
            global ndimage, misc, np, da, delayed, h5py
            import numpy as np
            import dask.array as da
            import h5py
            from dask import delayed
            from scipy import ndimage, misc
            c_gradient = True
        except ImportError:
            print "You need the following dependencies to also calculate the gradient: scipy, numpy, h5py, dask"
            c_gradient = False

        # From png files
        imgTile, gradientTile, sliceResolution, numberOfSlices, slicesPerAxis = ImageSlices2TiledImage(filenames_png,
                                                                                                       load_png,
                                                                                                       c_gradient,
                                                                                                       width,
                                                                                                       height)
    else:
        print "No PNG files found in that folder, check your parameters or contact the authors :)."
        return 2

    # Write a text file containing the number of slices for reference
    try:
        try:
            print 'Creating folder', os.path.dirname(argv[2]), '...',
            os.makedirs(os.path.dirname(argv[2]))
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(os.path.dirname(argv[2])):
                print 'was already there.'
            else:
                print ', folders might not be created, trying to write anyways...'
        except:
            print ", could not create folders, trying to write anyways..."
        with open(argv[2] + "_AtlasDim.txt", 'w') as f:
            f.write(str((numberOfSlices, (slicesPerAxis, slicesPerAxis))))
    except:
        print "Could not write a text file", argv[2] + "_AtlasDim.txt", \
            "containing dimensions (total slices, slices per axis):", (numberOfSlices, (slicesPerAxis, slicesPerAxis))
    else:
        print "Created", argv[2] + "_AtlasDim.txt", "containing dimensions (total slices, slices per axis):", (
            numberOfSlices, (slicesPerAxis, slicesPerAxis))

    # Output is written in different sizes
    write_versions(imgTile, gradientTile, argv[2])

if __name__ == "__main__":
    sys.exit(main())
