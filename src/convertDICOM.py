#!/usr/bin/env python
# coding=utf-8
import os, errno
import sys
import math
import argparse
from argparse import RawTextHelpFormatter
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

    gradient = None
    if cGradient:
        print "Starting to compute the gradient: Loading the data..."
        image_list = [da.from_array(np.array(loadImgFunction(f, arr=True), dtype='uint8'), chunks=size) for f in filenames]
        data = da.stack(image_list, axis=-1)
        cpus = cpu_count()
        chunk_size = [x // cpus for x in data.shape]
        print "Calculated chunk size: " + str(chunk_size)
        data = da.rechunk(data, chunks=chunk_size)
        print "Loading complete. Data size: " + str(data.shape)
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
    parser = argparse.ArgumentParser(prog='DICOM Atlas Generator',
                                     description='''
DICOM Atlas generation utility
------------------------------\n

This application converts the slices found in a folder into a tiled 2D texture
image in DICOM format.\nIt uses Python with PIL, numpy and pydicom packages are recommended for other formats.
\n
Note: this version does not process several folders recursively.''',
                                 epilog='''
This code was created by Luis Kabongo, Vicomtech-IK4 Copyright 2012-2013.
Modified by Ander Arbelaiz to add gradient calculation.\n
Information links:
 - https://github.com/VolumeRC/AtlasConversionScripts/wiki
 - http://www.volumerc.org
 - http://demos.vicomtech.org
Contact mailto:volumerendering@vicomtech.org''',
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('input', type=str, help='must contain a path to one set of DICOM files to be processed')
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

    # Filter only DCM files in the given folder
    filenamesDCM = filter(lambda x: ".DCM" in x or ".dcm" in x, listdir_fullpath(arguments.input))

    if not len(filenamesDCM) > 0:
        print "No DICOM files found in that folder, check your parameters or contact the authors :)."
        return 2

    # Update global value for standard_deviation
    sigmaValue = arguments.standard_deviation

    # Dicom dependencies
    try:
        global dicom
        global np
        import dicom
        import numpy as np
    except:
        print "You need dicom package (http://code.google.com/p/pydicom/) and numpy (http://numpy.scipy.org/) to do this!"
        return 2

    c_gradient = False
    if arguments.gradient:
        try:
            global ndimage, misc, np, da, delayed, h5py
            import numpy as np
            import dask.array as da
            import h5py
            from dask import delayed
            from scipy import ndimage, misc
            c_gradient = True
        except ImportError:
            print "You need the following dependencies to also calculate the gradient: scipy, numpy, h5py and dask"

    imgTile, gradientTile, sliceResolution, numberOfSlices, slicesPerAxis = ImageSlices2TiledImage(filenamesDCM,
                                                                                                   loadDICOM,
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
            "containing dimensions (total slices, slices per axis):", (numberOfSlices, (slicesPerAxis, slicesPerAxis))
    else:
        print "Created", arguments.output + "_AtlasDim.txt", "containing dimensions (total slices, slices per axis):", \
            (numberOfSlices, (slicesPerAxis, slicesPerAxis))

    # Output is written in different sizes
    write_versions(imgTile, gradientTile, arguments.output)


if __name__ == "__main__":
    sys.exit(main())
