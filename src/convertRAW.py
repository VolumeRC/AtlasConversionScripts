#!/usr/bin/env python
# coding=utf-8
print """
This code was created by Luis Kabongo, Vicomtech-IK4 Copyright 2012-2013.
This application converts the slices found in a folder into a tiled 2D texture 
image in PNG format (it assumes all files in the folder are of the same type 
and dimensions). It uses Python with PIL, numpy and pydicom packages are 
recommended for other formats.
Information links:
http://www.volumerc.org
http://demos.vicomtech.org
Code was inspired by: https://github.com/nopjia/WebGL-Volumetric/blob/master/stitching/stitch.py
Contact mailto:volumerendering@vicomtech.org
"""

import os, errno
import sys
import math
import argparse
from argparse import RawTextHelpFormatter
from multiprocessing import cpu_count
import tempfile# this is required to manage the images
try:
    from PIL import Image
except ImportError:
    import Image
# Scipy dependencies
try:
    import numpy as np
    from scipy import ndimage, misc
except ImportError:
    print "You need SciPy and Numpy (http://numpy.scipy.org/)!"

# This is the default size when loading a Raw image
sizeOfRaw = (256, 256)
# Number of slices per RAW image
slices = 128
# This determines if the endianness should be reversed
rawByteSwap = False

# Standard deviation for Gaussian kernel
sigmaValue = 2

# Simple decrement function
def decr(x, y):
    return x - y


# Normalize values between [0-1]
def normalize(block):
    old_min = delayed(block.min())
    old_max = delayed(block.max())
    r = delayed(decr)(old_max, old_min)
    minimum = old_min.compute()
    t0 = decr(block, minimum)
    return t0/r.compute(), -minimum/r.compute()


# Calculate derivatives function
def gaussian_filter(block, axis):
    return ndimage.gaussian_filter1d(block, sigma=sigmaValue, axis=axis, order=1)


# This function calculates the gradient from a 3 dimensional dask array
def calculate_gradient(arr):
    axises = [1, 0, 2]  # Match RGB
    g = da.ghost.ghost(arr, depth={0: 1, 1: 1, 2: 1},  boundary={0: 'reflect', 1: 'reflect', 2: 'reflect'})
    derivatives = [g.map_blocks(gaussian_filter, axis) for axis in axises]
    derivatives = [da.ghost.trim_internal(d, {0: 1, 1: 1, 2: 1}) for d in derivatives]
    gradient = da.stack(derivatives, axis=3)
    return normalize(gradient)


# This function takes a filename and returns a compatible multidemensional array
def loadRAW2Numpy(filename):
    f = open(filename, "rb")
    try:
        first_time = True
        for _ in range((slices - 1)):
            if first_time:
                data = np.fromfile(f, 'uint8', sizeOfRaw[0] * sizeOfRaw[1]).reshape(sizeOfRaw)
                first_time = False
            raw = np.fromfile(f, 'uint8', sizeOfRaw[0] * sizeOfRaw[1]).reshape(sizeOfRaw)
            data = np.dstack((data, raw))
        return data
    except EOFError:
        return data
    except ValueError:
        print 'Warning!! ValueError when reshaping the data, continuing anyway!'
    finally:
        f.close()


# This function uses the images retrieved with loadImgFunction (whould return a PIL.Image) and
#	writes them as tiles within a new square Image. 
#	Returns a set of Image, size of a slice, number of slices and number of slices per axis
def ImageSlices2TiledImage(filenames, loadImgFunction=loadRAW2Numpy, cGradient=False):
    filenames = sorted(filenames)
    print "Desired load function=", loadImgFunction.__name__
    size = sizeOfRaw
    numberOfSlices = slices * len(filenames)
    slicesPerAxis = int(math.ceil(math.sqrt(numberOfSlices)))

    data = loadRAW2Numpy(filenames[0])
    for f in range(1, len(filenames)):
        data = np.dstack((data, loadRAW2Numpy(filenames[f])))

    atlasArray = np.zeros((size[0] * slicesPerAxis, size[1] * slicesPerAxis))

    for i in range(0, numberOfSlices):
        row = int((math.floor(i / slicesPerAxis)) * size[0])
        col = int((i % slicesPerAxis) * size[1])

        box = (int(row), int(col), int(row + size[0]), int(col + size[1]))
        atlasArray[box[0]:box[2], box[1]:box[3]] = data[:, :, i]
        print "processed slice  : " + str(i + 1) + "/" + str(numberOfSlices)  # filename

    imout = misc.toimage(atlasArray, mode="L")

    gradient = None
    if cGradient:
        print "Starting to compute the gradient: Loading the data..."
        cpus = cpu_count()
        chunk_size = [x // cpus for x in data.shape]
        print "Calculated chunk size: " + str(chunk_size)
        data = da.from_array(data, chunks=chunk_size)
        print "Computing the gradient..."
        data = data.astype(np.float32)
        gradient_data, g_background = calculate_gradient(data)
        # Normalize values to RGB values
        gradient_data *= 255
        g_background = int(g_background * 255)
        gradient_data = gradient_data.astype(np.uint8)
        # Keep the RGB information separated, uses less RAM memory
        channels = ['/r', '/g', '/b']
        f = tempfile.NamedTemporaryFile(delete=False)
        [da.to_hdf5(f.name, c, gradient_data[:, :, :, i]) for i, c in enumerate(channels)]
        print "Computed gradient data saved in cache file."
        # Create atlas image
        gradient = Image.new("RGB",
                             (size[0] * slicesPerAxis, size[1] * slicesPerAxis),
                             (g_background, g_background, g_background))

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
            print "processed gradient slice  : " + str(i + 1) + "/" + str(numberOfSlices)  # filename

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
def main():
    # Define th CLI
    parser = argparse.ArgumentParser(prog='RAW Atlas Generator',
                                     description='''
RAW Atlas generation utility
----------------------------\n

This application converts the slices found in a folder into a tiled 2D texture
image in PNG format.\nIt uses Python with PIL and numpy.
\n
Note1: RAW Support may require adaptation, check values for sizeOfRaw and rawByteSwap at the beginning of this file.
Note2: this version does not process several RAW folders recursively.
''',
                                     epilog='''
This code was created by Luis Kabongo, Vicomtech-IK4 Copyright 2012-2013.
Modified by Ander Arbelaiz to add gradient calculation.\n
Information links:
 - https://github.com/VolumeRC/AtlasConversionScripts/wiki
 - http://www.volumerc.org
 - http://demos.vicomtech.org
Contact mailto:volumerendering@vicomtech.org''',
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('input', type=str, help='must contain only one set of RAW files to be processed')
    parser.add_argument('output', type=str,
                        help='must contain the path and base name of the desired output,\n'
                             'extension will be added automatically')
    parser.add_argument('--gradient', '-g', action='store_true',
                        help='calculate and generate the gradient atlas')
    parser.add_argument('--standard_deviation', '-std', type=int, default=2,
                        help='standard deviation for the gaussian kernel used for the gradient computation')

    # Obtain the parsed arguments
    print "Parsing arguments..."
    arguments = parser.parse_args()

    # Convert into a tiled image
    filenamesRAW = listdir_fullpath(arguments.input)
    if not len(filenamesRAW) > 0:
        print "No files found in that folder, check your parameters or contact the authors :)."
        return 2

    # Update global value for standard_deviation
    sigmaValue = arguments.standard_deviation

    c_gradient = False
    if arguments.gradient:
        try:
            global da, delayed, h5py
            import dask.array as da
            import h5py
            from dask import delayed
            c_gradient = True
        except ImportError:
            print "You need the following dependencies to also calculate the gradient: numpy, h5py and dask"

    # From nrrd files
    imgTile, gradientTile, sliceResolution, numberOfSlices, slicesPerAxis = ImageSlices2TiledImage(filenamesRAW,
                                                                                                   loadRAW2Numpy,
                                                                                                   c_gradient)

    # Write a text file containing the number of slices for reference
    try:
        try:
            print 'Creating folder', os.path.dirname(arguments.output), '...',
            os.makedirs(os.path.dirname(arguments.output))
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(os.path.dirname(arguments.output)):
                print 'was already there.'
            else:
                print ', folders might not be created, trying to write anyways...'
        except:
            print ", could not create folders, trying to write anyways..."
        with open(str(arguments.output) + "_AtlasDim.txt", 'w') as f:
            f.write(str((numberOfSlices, (slicesPerAxis, slicesPerAxis))))
    except:
        print "Could not write a text file", str(arguments.output) + "_AtlasDim.txt", \
            "containing dimensions (total slices, slices per axis):", (
            numberOfSlices, (slicesPerAxis, slicesPerAxis))
    else:
        print "Created", arguments.output + "_AtlasDim.txt", "containing dimensions (total slices, slices per axis):", \
            (numberOfSlices, (slicesPerAxis, slicesPerAxis))

    # Output is written in different sizes
    write_versions(imgTile, gradientTile, arguments.output)


if __name__ == "__main__":
    sys.exit(main())
