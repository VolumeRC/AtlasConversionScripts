#!/usr/bin/env python
# coding=utf-8
#
# Copyright 2012 Vicomtech-IK4
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import errno
import math
import argparse
from argparse import RawTextHelpFormatter
from multiprocessing import cpu_count
import tempfile
from itertools import izip_longest
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


# This function simply loads a PNG file and returns a compatible Image object
def load_png(filename):
    im = Image.open(filename)
    if im.mode != 1:
        return im.convert("L", palette=Image.ADAPTIVE, colors=256)
    return im


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)


def grouper_with_repeat(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABCD DEFG Gxxx"
    args = [iter(iterable)] * n + [iter([iterable[i] for i in range(n, len(iterable), n)])]
    return izip_longest(*args, fillvalue=fillvalue)


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


# This function uses the images retrieved with loadImgFunction (returns a PIL.Image) and
# writes them as tiles within a new square Image.
# Returns a set of Image, size of a slice, number of slices and number of slices per axis
def ImageSlices2TiledImage(filenames, loadImgFunction=load_png, cGradient=False, r_width=None, r_height=None):
    filenames = sorted(filenames)
    print "Desired load function=", loadImgFunction.__name__
    size = read_image(filenames[0], loadImgFunction, r_width, r_height).size
    channel_option = "rgba"
    numberOfSlices = len(filenames)

    _grouper = grouper

    if "r" == channel_option:  # Single channel
        len_channels = 1
        image_mode = "L"
    elif "r+g" == channel_option:
        len_channels = 1
        image_mode = "RGB"
        _grouper = grouper_with_repeat
    elif "rg+b" == channel_option:
        len_channels = 2
        image_mode = "RGB"
        _grouper = grouper_with_repeat
    elif "rgb+a" == channel_option:
        len_channels = 3
        image_mode = "RGBA"
        _grouper = grouper_with_repeat
    elif "rgba" == channel_option:
        len_channels = 4
        image_mode = "RGBA"

    numberOfSlicesPerChannel = int(math.ceil(numberOfSlices / len_channels))
    slicesPerAxis = int(math.ceil(math.sqrt(numberOfSlicesPerChannel)))
    imout = Image.new(image_mode, (size[0] * slicesPerAxis, size[1] * slicesPerAxis))

    i = 0
    for files in _grouper(filenames, len_channels):
        idx = int(math.floor(i / len_channels))
        image_stack = []
        for image in files:
            image_stack.append(read_image(image, loadImgFunction, r_width, r_height)) if image else Image.new("L", size)
        while len(image_stack) != len(image_mode):
            image_stack.append(Image.new("L", size))
        im = Image.merge(image_mode, image_stack)
        row = int((math.floor(idx / slicesPerAxis)) * size[0])
        col = int((idx % slicesPerAxis) * size[1])
        box = (int(col), int(row), int(col + size[0]), int(row + size[1]))
        imout.paste(im, box)
        i += len_channels
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
            print "processed gradient slice  : " + str(i+1) + "/" + str(numberOfSlices)  # filename

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


######################################
# Main program - CLI with argparse - #
######################################
def main():
    # Define th CLI
    parser = argparse.ArgumentParser(prog='PNG Atlas Generator',
                                     description='''
PNG Atlas generation utility
----------------------------\n

This application converts the slices found in a folder into a tiled 2D texture
image in PNG format.\nIt uses Python with PIL, numpy and pydicom packages are recommended for other formats.
\n
Note: this version does not process several folders recursively.''',
                                     epilog='''
This code was created by Luis Kabongo.
Modified by Ander Arbelaiz to add gradient calculation.\n
Information links:
 - https://github.com/VolumeRC/AtlasConversionScripts/wiki
 - http://www.volumerc.org
 - http://demos.vicomtech.org
Contact mailto:volumerendering@vicomtech.org''',
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('input', type=str, help='must contain a path to one set of PNG files to be processed')
    parser.add_argument('output', type=str,
                        help='must contain the path and base name of the desired output,\n'
                             'extension will be added automatically')
    parser.add_argument('--resize', '-r', type=int, nargs=2, metavar=('x', 'y'),
                        help='resizing of the input images x y, before processing')
    parser.add_argument('--gradient', '-g', action='store_true',
                        help='calculate and generate the gradient atlas')
    parser.add_argument('--standard_deviation', '-std', type=int, default=2,
                        help='standard deviation for the gaussian kernel used for the gradient computation')
    parser.add_argument('--channels', '-ch', type=str, default="r",
                        help='disposition of slices along atlas color channels')

    # Obtain the parsed arguments
    print "Parsing arguments..."
    arguments = parser.parse_args()

    # Filter only png files in the given folder
    filenames_png = filter(lambda x: ".png" in x, listdir_fullpath(arguments.input))

    if not len(filenames_png) > 0:
        print "No PNG files found in that folder, check your parameters or contact the authors :)."
        return 2

    if arguments.resize:
        width, height = arguments.resize[0], arguments.resize[1]
    else:
        width, height = None, None

    # Update global value for standard_deviation
    sigmaValue = arguments.standard_deviation

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

    # From png files
    imgTile, gradientTile, sliceResolution, numberOfSlices, slicesPerAxis = ImageSlices2TiledImage(filenames_png,
                                                                                                   load_png,
                                                                                                   c_gradient,
                                                                                                   width,
                                                                                                   height)

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
        print "Created", arguments.output + "_AtlasDim.txt", "containing dimensions (total slices, slices per axis):",\
            (numberOfSlices, (slicesPerAxis, slicesPerAxis))

    # Output is written in different sizes
    write_versions(imgTile, gradientTile, arguments.output)

if __name__ == "__main__":
    sys.exit(main())
