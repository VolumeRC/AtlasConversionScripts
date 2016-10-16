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
import getopt
import math
import array
# this is required to manage the images
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
sigmaValue = 1

# Normalize a numpy array
def normalize(inputData):
    old_min = inputData.min()
    old_range = inputData.max() - old_min
    return (inputData - old_min) / old_range


# This function calculates the gradient from a 3 dimensional numpy array ussing a gaussian filter
def calculateGradient(arr):
    r = np.zeros(arr.shape)
    g = np.zeros(arr.shape)
    b = np.zeros(arr.shape)
    ndimage.gaussian_filter1d(arr, sigma=sigmaValue, axis=1, order=1, output=r)
    ndimage.gaussian_filter1d(arr, sigma=sigmaValue, axis=0, order=1, output=g)
    ndimage.gaussian_filter1d(arr, sigma=sigmaValue, axis=2, order=1, output=b)
    return normalize(np.concatenate((r[..., np.newaxis], g[..., np.newaxis], b[..., np.newaxis]), axis=3))


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
def ImageSlices2TiledImage(filenames, loadImgFunction=loadRAW2Numpy):
    filenames = sorted(filenames)
    print "Desired load function=", loadImgFunction.__name__
    size = sizeOfRaw
    numberOfSlices = slices * len(filenames)
    slicesPerAxis = int(math.ceil(math.sqrt(numberOfSlices)))

    data = loadRAW2Numpy(filenames[0])
    for f in range(1, len(filenames)):
        data = np.dstack((data, loadRAW2Numpy(filenames[f])))
    gradientData = calculateGradient(data)

    atlasArray = np.zeros((size[0] * slicesPerAxis, size[1] * slicesPerAxis))
    atlasGradientArray = np.zeros((size[0] * slicesPerAxis, size[1] * slicesPerAxis, 3))

    for i in range(0, numberOfSlices):
        row = int((math.floor(i / slicesPerAxis)) * size[0])
        col = int((i % slicesPerAxis) * size[1])

        box = (int(row), int(col), int(row + size[0]), int(col + size[1]))
        atlasArray[box[0]:box[2], box[1]:box[3]] = data[:, :, i]
        atlasGradientArray[box[0]:box[2], box[1]:box[3], :] = gradientData[:, :, i, :]
        print "processed slice  : " + str(i + 1) + "/" + str(numberOfSlices)  # filename

    imout = misc.toimage(atlasArray, mode="L")
    gradient = misc.toimage(atlasGradientArray, mode="RGB")

    return imout, gradient, size, numberOfSlices, slicesPerAxis


# This functions takes a (tiled) image and writes it to a png file with base filename outputFilename.
#	It also writes several versions in different sizes determined by dimensions
def WriteVersions(tileImage, tileGradient, outputFilename, dimensions=[8192, 4096, 2048, 1024]):
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
        print "Usage: command <InputFolder> <OutputFilename>"
        print "	<InputFolder> must contain only one set of RAW files to be processed"
        print "	<OutputFilename> must contain the path and base name of the desired output, extension will be added automatically"
        print "Note1: RAW Support may require adaptation, check values for sizeOfRaw and rawByteSwap at the beginning of this file."
        print "Note2: this version does not process several RAW folders recursively."
        print "You typed:", argv
        return 2

    # Convert into a tiled image
    filenamesRAW = listdir_fullpath(argv[1])
    if len(filenamesRAW):
        # From RAW files
        imgTile, gradientTile, sliceResolution, numberOfSlices, slicesPerAxis = ImageSlices2TiledImage(filenamesRAW,
                                                                                                       loadRAW2Numpy)
    else:
        print "No files found in that folder, check your parameters or contact the authors :)."
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
        print "Could not write a text file", argv[
                                                 2] + "_AtlasDim.txt", "containing dimensions (total slices, slices per axis):", (
        numberOfSlices, (slicesPerAxis, slicesPerAxis))
    else:
        print "Created", argv[2] + "_AtlasDim.txt", "containing dimensions (total slices, slices per axis):", (
        numberOfSlices, (slicesPerAxis, slicesPerAxis))

    # Output is written in different sizes
    WriteVersions(imgTile, gradientTile, argv[2])


if __name__ == "__main__":
    sys.exit(main())
