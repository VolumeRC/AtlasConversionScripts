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

def decr(x, y):
    return x - y

def normalize(block):
    old_min = delayed(block.min())
    r = delayed(decr)(block.max(), old_min)
    t0 = decr(block, old_min.compute())
    return t0 / r.compute()

def gaussian_filter(block, axis):
    return ndimage.gaussian_filter1d(block, sigma=sigmaValue, axis=axis, order=1)

# This function calculates the gradient from a 3 dimensional numpy array
def calculate_gradient(arr):
    axises = [1, 0, 2]  # Match RGB
    g = da.ghost.ghost(arr, depth={0: 2, 1: 2, 2: 1},  boundary={0: 'periodic', 1: 'periodic', 2: 'reflect'})
    derivatives = [g.map_blocks(gaussian_filter, axis) for axis in axises]
    derivatives = [da.ghost.trim_internal(d, {0: 2, 1: 2, 2: 1}) for d in derivatives]
    gradient = da.stack(derivatives, axis=3)
    return normalize(gradient)


# This function simply loads a PNG file and returns a compatible Image object
def load_png(filename):
    im = Image.open(filename)
    width, height = im.size
    # Create an square image if required
    if width == height:
        if im.mode != 1:
            t_im = Image.new("L", im.size, 0)
            t_im.paste(im)
            return t_im
        return im
    else:
        mode = im.mode
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


# This function uses the images retrieved with loadImgFunction (whould return a PIL.Image) and
# writes them as tiles within a new square Image.
# Returns a set of Image, size of a slice, number of slices and number of slices per axis
def ImageSlices2TiledImage(filenames, loadImgFunction=load_png, cGradient=False):
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
        # data = ndimage.imread(filenames[0], flatten=True)
        # data = loadImgFunction(filenames[0])
        # for f in range(1, len(filenames)):
        #     # data = np.dstack((data, ndimage.imread(filenames[f], flatten=True)))
        #     data = da.stack((data, loadImgFunction(filenames[f])))

        print "Loading the data..."
        image_list = [da.from_array(np.array(loadImgFunction(f), dtype='uint8'), chunks=size) for f in filenames]
        data = da.stack(image_list, axis=-1)
        cpus = cpu_count()
        chunk_size = [x//cpus for x in data.shape]
        print "Calculated chunk size: "+str(chunk_size)
        data = da.rechunk(data, chunks=chunk_size)
        print "Loading complete. Data size: "+str(data.shape)
        print "Computing the gradient..."
        data = data.astype(np.float32)
        gradientData = calculate_gradient(data)
        # Normalize to image values RGB values
        gradientData = gradientData * 255
        gradientData = gradientData.astype(np.uint8)
        # Keep the RGB information separated
        channels = ['/r', '/g', '/b']
        [da.to_hdf5('gradient_cache.hdf5', c, gradientData[:, :, :, i]) for i, c in enumerate(channels)]
        #plt.imshow(gradientData[:, :, 2], cmap=cm.get_cmap("gray"))
        #plt.show()
        #return imout, gradient, size, numberOfSlices, slicesPerAxis
        #atlasArray = da.from_array(np.zeros((size[0] * slicesPerAxis, size[1] * slicesPerAxis, 3)), chunks=512)
        # Create atlas image
        gradient = Image.new("RGB", (size[0] * slicesPerAxis, size[1] * slicesPerAxis))

        boxes = []
        for i in range(0, numberOfSlices):
            row = int((math.floor(i / slicesPerAxis)) * size[0])
            col = int((i % slicesPerAxis) * size[1])

            box = (int(col), int(row), int(col + size[0]), int(row + size[1]))
            boxes.append(box)

        channels = ['/r', '/g', '/b']
        dsets = [h5py.File('gradient_cache.hdf5')[c] for c in channels]
        arrays = [da.from_array(dset, chunks=chunk_size) for dset in dsets]
        gradient_data = da.stack(arrays, axis=-1)
        for ind, box in enumerate(boxes):
            s = gradient_data[:, :, ind, :]
            im = Image.fromarray(np.array(s))
            gradient.paste(im, box)
            print "processed gradient slice  : " + str(ind) + "/" + str(numberOfSlices)  # filename

        # Remove cache file
        try:
            os.remove('gradient_cache.hdf5')
        except OSError as e:  # this would be "except OSError, e:" before Python 2.6
            if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
                raise  # re-raise exception if a different error occured

    return imout, gradient, size, numberOfSlices, slicesPerAxis


# This functions takes a (tiled) image and writes it to a png file with base filename outputFilename.
# It also writes several versions in different sizes determined by dimensions
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
        print "	<InputFolder> must contain only one set of PNG files to be processed"
        print "	<OutputFilename> must contain the path and base name of the desired output, extension will be added " \
              "automatically"
        print "Note: this version does not process several folders recursively. "
        print "You typed:", argv
        return 2

    # Filter only png files in the given folder
    filenamesPNG = filter(lambda x: ".png" in x, listdir_fullpath(argv[1]))

    # Convert into a tiled image
    if len(filenamesPNG) > 0:
        try:
            global ndimage, misc
            global h5py
            global np, da, delayed
            import numpy as np
            import dask.array as da
            import h5py
            from dask import delayed
            from scipy import ndimage, misc

            gradient = True
        except ImportError:
            print "You need SciPy and Numpy (http://numpy.scipy.org/) to also calculate the gradient!"
            gradient = False

        # From png files
        if gradient:
            imgTile, gradientTile, sliceResolution, numberOfSlices, slicesPerAxis = ImageSlices2TiledImage(filenamesPNG,
                                                                                                           load_png,
                                                                                                           True)
        else:
            imgTile, gradientTile, sliceResolution, numberOfSlices, slicesPerAxis = ImageSlices2TiledImage(filenamesPNG,
                                                                                                           load_png)
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
    WriteVersions(imgTile, gradientTile, argv[2])


if __name__ == "__main__":
    sys.exit(main())
