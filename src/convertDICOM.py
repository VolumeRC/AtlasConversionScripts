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

# This is the default size when loading a Raw image
sizeOfRaw = (512, 512)
# This determines if the endianness should be reversed
rawByteSwap = True
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


# This function returns the data array values mapped to 0-256 using window/level parameters
#	If provided it takes into account the DICOM flags:
#	- Rescale Intercept http://dicomlookup.com/lookup.asp?sw=Tnumber&q=(0028,1052)
#	- Rescale Slope http://dicomlookup.com/lookup.asp?sw=Tnumber&q=(0028,1053)
#	Code adapted from pydicom, requires numpy
#	http://code.google.com/p/pydicom/source/browse/source/dicom/contrib/pydicom_PIL.py
def get_LUT_value(data, window, level, rescaleIntercept=0, rescaleSlope=1):
    if isinstance(window, list):
        window = window[0]
    if isinstance(level, list):
        level = level[0]
    return np.piecewise(data,
                        [((data * rescaleSlope) + rescaleIntercept) <= (level - 0.5 - (window - 1) / 2),
                         ((data * rescaleSlope) + rescaleIntercept) > (level - 0.5 + (window - 1) / 2)],
                        [0, 255, lambda VAL: ((((VAL * rescaleSlope) + rescaleIntercept) - (level - 0.5)) / (
                        window - 1) + 0.5) * (255 - 0)])


# This function loads a DCM file and returns a compatible Image object
# Implemented from: http://stackoverflow.com/questions/119684/parse-dicom-files-in-native-python
def loadDICOM(filename, arr=False):
    try:
        dicomFile = dicom.read_file(filename)
    except:
        print "Pydicom function for reading file not found."
        return None

    data = dicomFile.pixel_array

    rescaleIntercept = dicomFile[0x0028, 0x1052].value
    rescaleSlope = dicomFile[0x0028, 0x1053].value
    print "Rescale intercept/slope", (rescaleIntercept, rescaleSlope),
    if dicomFile.RescaleIntercept == None or dicomFile.RescaleSlope == None:
        rescaleIntercept = 0.0
        rescaleSlope = 1.0

    # Since we are opening a DICOM file with a possible data value range that exceeds the output format range, we try to use one of the provided window/level values to rescale values
    if dicomFile.Modality == "CT":
        print "CT modality, rescaling 1500 500"
        data = get_LUT_value(data, 1500, 500, rescaleIntercept, rescaleSlope)
    elif dicomFile.WindowWidth != None and dicomFile.WindowCenter != None:
        print "Rescaling", dicomFile.WindowWidth, dicomFile.WindowCenter
        data = get_LUT_value(data, dicomFile.WindowWidth, dicomFile.WindowCenter, rescaleIntercept, rescaleSlope)
    else:
        print "No rescaling applied"

    bits = dicomFile.BitsAllocated
    samples = dicomFile.SamplesPerPixel
    if bits == 8 and samples == 1:
        mode = "L"
    elif bits == 8 and samples == 3:
        mode = "RGB"
    elif bits == 16:
        mode = "I;16"

    if not arr:
        im = Image.frombuffer(mode=mode, size=(dicomFile.Columns, dicomFile.Rows), data=data)
    else:
        im = data
    return im


# This function uses the images retrieved with loadImgFunction (whould return a PIL.Image) and
#	writes them as tiles within a new square Image. 
#	Returns a set of Image, size of a slice, number of slices and number of slices per axis
def ImageSlices2TiledImage(filenames, loadImgFunction=loadDICOM, cGradient=False):
    filenames = sorted(filenames)
    print "Desired load function=", loadImgFunction.__name__
    size = loadImgFunction(filenames[0]).size
    numberOfSlices = len(filenames)
    slicesPerAxis = int(math.ceil(math.sqrt(numberOfSlices)))
    imout = Image.new("L", (size[0] * slicesPerAxis, size[1] * slicesPerAxis))

    i = 0
    for filename in filenames:
        im = loadImgFunction(filename)

        row = int((math.floor(i / slicesPerAxis)) * size[0])
        col = int((i % slicesPerAxis) * size[1])

        box = (int(col), int(row), int(col + size[0]), int(row + size[1]))
        imout.paste(im, box)
        i += 1
        print "processed slice  : " + str(i) + "/" + str(numberOfSlices)  # filename

    if cGradient:
        data = loadDICOM(filenames[0], arr=True)
        for f in range(1, len(filenames)):
            data = np.dstack((data, loadDICOM(filenames[f], arr=True)))

        gradientData = calculateGradient(data)
        atlasArray = np.zeros((size[0] * slicesPerAxis, size[1] * slicesPerAxis, 3))

        for i in range(0, numberOfSlices):
            row = int((math.floor(i / slicesPerAxis)) * size[0])
            col = int((i % slicesPerAxis) * size[1])

            box = (int(row), int(col), int(row + size[0]), int(col + size[1]))
            atlasArray[box[0]:box[2], box[1]:box[3], :] = gradientData[:, :, i, :]

        gradient = misc.toimage(atlasArray)

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
        print "	<InputFolder> must contain only one series of DICOM files to be processed"
        print "	<OutputFilename> must contain the path and base name of the desired output, extensions will be added automatically"
        print "Note: this version does not process several DICOM folders recursively. "
        print "You typed:", argv
        return 2

    # Filter only DCM files in the given folder
    filenamesDCM = filter(lambda x: ".DCM" in x or ".dcm" in x, listdir_fullpath(argv[1]))

    # Convert into a tiled image
    if len(filenamesDCM) > 0:
        # Dicom dependencies
        try:
            global dicom
            global np
            import dicom
            import numpy as np
        except:
            print "You need dicom package (http://code.google.com/p/pydicom/) and numpy (http://numpy.scipy.org/) to do this!"
            return 2

        # Scipy dependdencies
        try:
            global ndimage, misc
            from scipy import ndimage, misc

            gradient = True
        except ImportError:
            print "You need ndimage and misc modules from SciPy to also calculate the gradient!"
            gradient = False

        # From dcm files
        if gradient:
            imgTile, gradientTile, sliceResolution, numberOfSlices, slicesPerAxis = ImageSlices2TiledImage(filenamesDCM,
                                                                                                           loadDICOM,
                                                                                                           True)
        else:
            imgTile, gradientTile, sliceResolution, numberOfSlices, slicesPerAxis = ImageSlices2TiledImage(filenamesDCM,
                                                                                                           loadDICOM)
    else:
        print "No DICOM files found in that folder, check your parameters or contact the authors :)."
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
        print "Could not write a text file", \
            argv[2] + "_AtlasDim.txt", \
            "containing dimensions (total slices, slices per axis):", \
            (numberOfSlices, (slicesPerAxis, slicesPerAxis))
    else:
        print "Created", argv[2] + "_AtlasDim.txt", "containing dimensions (total slices, slices per axis):", (
        numberOfSlices, (slicesPerAxis, slicesPerAxis))

    # Output is written in different sizes
    WriteVersions(imgTile, gradientTile, argv[2])


if __name__ == "__main__":
    sys.exit(main())
