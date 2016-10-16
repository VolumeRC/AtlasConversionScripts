#!/usr/bin/env python
# coding=utf-8
print """
This application converts a nrrd file into a tiled 2D texture 
image in PNG format (it assumes all slices are of the same type 
and dimensions). It uses Python with PIL, pynrrd and numpy.
Information links:
http://www.volumerc.org
http://demos.vicomtech.org
Contact mailto:volumerendering@vicomtech.org
"""

import os, errno
import sys
import math
import array
# this is required to manage the images
try:
    from PIL import Image
except ImportError:
    import Image
try:
    import numpy as np
    from scipy import ndimage, misc
except ImportError:
    print 'You need scipy.ndimage, scipy.misc and Numpy (http://numpy.scipy.org/)'
    exit()

# Standard deviation for Gaussian kernel 
sigmaValue = 1


def normalize(inputData):
    old_min = inputData.min()
    old_range = inputData.max() - old_min
    return (inputData - old_min) / old_range


# This function calculates the gradient from a 3 dimensional numpy array
def calculateGradient(arr):
    r = np.zeros(arr.shape)
    g = np.zeros(arr.shape)
    b = np.zeros(arr.shape)
    ndimage.gaussian_filter1d(arr, sigma=sigmaValue, axis=1, order=1, output=r)
    ndimage.gaussian_filter1d(arr, sigma=sigmaValue, axis=0, order=1, output=g)
    ndimage.gaussian_filter1d(arr, sigma=sigmaValue, axis=2, order=1, output=b)
    return normalize(np.concatenate((r[..., np.newaxis], g[..., np.newaxis], b[..., np.newaxis]), axis=3))


# This function simply loads a NRRD file and returns a compatible Image object
def loadNRRD(filename):
    try:
        data, info = nrrd.read(filename)
        return data
    except:
        print 'Error reading the nrrd file!'
        exit()


# This function uses the images retrieved with loadImgFunction (whould return a PIL.Image) and
#	writes them as tiles within a new square Image. 
#	Returns a set of Image, size of a slice, number of slices and number of slices per axis
def ImageSlices2TiledImage(filename, loadImgFunction=loadNRRD):
    print "Desired load function=", loadImgFunction.__name__
    data = loadImgFunction(filename)
    volumeSize = (data.shape[0], data.shape[1])
    numberOfSlices = data.shape[2]
    slicesPerAxis = int(math.ceil(math.sqrt(numberOfSlices)))
    atlasArray = np.zeros((volumeSize[0] * slicesPerAxis, volumeSize[1] * slicesPerAxis))

    gradientData = calculateGradient(data)
    atlasGradientArray = np.zeros((volumeSize[0] * slicesPerAxis, volumeSize[1] * slicesPerAxis, 3))

    for i in range(0, numberOfSlices):
        row = int((math.floor(i / slicesPerAxis)) * volumeSize[0])
        col = int((i % slicesPerAxis) * volumeSize[1])
        box = (row, col, int(row + volumeSize[0]), int(col + volumeSize[1]))
        atlasArray[box[0]:box[2], box[1]:box[3]] = data[:, :, i]
        atlasGradientArray[box[0]:box[2], box[1]:box[3], :] = gradientData[:, :, i, :]

    # From numpy to PIL image
    imout = misc.toimage(atlasArray, mode="L")
    gradient = misc.toimage(atlasGradientArray, mode="RGB")

    return imout, gradient, volumeSize, numberOfSlices, slicesPerAxis


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
        print "	<InputFile> must contain only the NRRD file to be processed"
        print "	<OutputFilename> must contain the path and base name of the desired output, extension will be added automatically"
        print "Note: this version does not process several folders recursively. "
        print "You typed:", argv
        return 2

    filenameNRRD = argv[1]

    # Convert into a tiled image
    if len(filenameNRRD) > 0:
        try:
            global nrrd
            import nrrd
        except:
            print "You need pynrrd package (sudo easy_install pynrrd) to do this!"
            return 2
        # From nrrd files
        imgTile, gradientTile, sliceResolution, numberOfSlices, slicesPerAxis = ImageSlices2TiledImage(filenameNRRD,
                                                                                                       loadNRRD)
    else:
        print "No NRRD file found in that folder, check your parameters or contact the authors :)."
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
    WriteVersions(imgTile, gradientTile, argv[2])


if __name__ == "__main__":
    sys.exit(main())
