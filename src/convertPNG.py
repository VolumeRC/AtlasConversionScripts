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

import os, errno
import sys
import getopt
import math
import array
#this is required to manage the images
try:
	from PIL import Image 
except ImportError:
	import Image

#This is the default size when loading a Raw image
sizeOfRaw = (512, 512)
#This determines if the endianness should be reversed
rawByteSwap = True
# Standard deviation for Gaussian kernel 
sigmaValue = 1

def normalize(inputData):
	old_min = inputData.min()
	old_range = inputData.max()-old_min
	return (inputData-old_min)/old_range

#This function calculates the gradient from a 3 dimensional numpy array
def calculateGradient(arr):
	r = np.zeros(arr.shape)
	g = np.zeros(arr.shape)
	b = np.zeros(arr.shape)
	ndimage.gaussian_filter1d(arr, sigma=sigmaValue, axis=1, order=1, output=r)
	ndimage.gaussian_filter1d(arr, sigma=sigmaValue, axis=0, order=1, output=g)
	ndimage.gaussian_filter1d(arr, sigma=sigmaValue, axis=2, order=1, output=b)
	return normalize(np.concatenate((r[...,np.newaxis],g[...,np.newaxis],b[...,np.newaxis]),axis=3))

#This function simply loads a PNG file and returns a compatible Image object
def loadPNG(filename):
	im = Image.open(filename)
	width, height = im.size
	if (width == height):
		return im
	else:
		mode = im.mode
		if len(mode) == 1:  # L, 1
			new_background = (0)
		if len(mode) == 3:  # RGB
			new_background = (0, 0, 0)
		if len(mode) == 4:  # RGBA, CMYK
			new_background = (0, 0, 0, 0)
		new_resolution = max(width, height)
		offset = ((new_resolution - width) / 2, (new_resolution - height) / 2)
		t_im = Image.new(mode, (new_resolution, new_resolution), new_background)
		t_im.paste(im, offset)
		return t_im

#This function uses the images retrieved with loadImgFunction (whould return a PIL.Image) and
#	writes them as tiles within a new square Image. 
#	Returns a set of Image, size of a slice, number of slices and number of slices per axis
def ImageSlices2TiledImage(filenames, loadImgFunction=loadPNG, cGradient=False):
	filenames=sorted(filenames)
	print "Desired load function=", loadImgFunction.__name__
	size = loadImgFunction(filenames[0]).size
	numberOfSlices = len(filenames)
	slicesPerAxis = int(math.ceil(math.sqrt(numberOfSlices)))
	imout = Image.new("L", (size[0]*slicesPerAxis, size[1]*slicesPerAxis))

	i = 0
	for filename in filenames:
		im = loadImgFunction(filename)
		
		row = int( (math.floor(i/slicesPerAxis)) * size[0] )
		col = int( (i%slicesPerAxis) * size[1] )

		box = ( int(col), int(row), int(col+size[0]), int(row+size[1]) )
		imout.paste(im, box)

		i+=1
		print "processed slice  : "+str(i)+"/"+str(numberOfSlices) #filename
	
	gradient = None
	if cGradient:
		#data = ndimage.imread(filenames[0], flatten=True)
		data = loadImgFunction(filenames[0])
		for f in range(1, len(filenames)):
			#data = np.dstack((data, ndimage.imread(filenames[f], flatten=True)))
			data = np.dstack((data, loadImgFunction(filenames[f])))

		gradientData = calculateGradient(data)
		atlasArray = np.zeros((size[0]*slicesPerAxis, size[1]*slicesPerAxis, 3))
		
		for i in range(0, numberOfSlices):
			row = int( (math.floor(i/slicesPerAxis)) * size[0] )
			col = int( (i%slicesPerAxis) * size[1] )

			box = ( int(row), int(col), int(row+size[0]), int(col+size[1]) )
			atlasArray[box[0]:box[2],box[1]:box[3],:] = gradientData[:,:,i,:]

		gradient = misc.toimage(atlasArray)

	return imout, gradient, size, numberOfSlices, slicesPerAxis

#This functions takes a (tiled) image and writes it to a png file with base filename outputFilename.
#	It also writes several versions in different sizes determined by dimensions
def WriteVersions(tileImage, tileGradient, outputFilename,dimensions=[8192,4096,2048,1024]):
	try:
		print 'Creating folder',os.path.dirname(outputFilename),'...',
		os.makedirs(os.path.dirname(outputFilename))
	except OSError as exc:
		if exc.errno == errno.EEXIST and os.path.isdir(os.path.dirname(outputFilename)):
			print 'was already there.'
		else:
			print ', folders might not be created, trying to write anyways...'
	except:
		print "Could not create folders, trying to write anyways..."

	print "Writing complete image: "+outputFilename+"_full.png"
	try:
		tileImage.save(outputFilename+"_full.png", "PNG")
		if tileGradient:
			tileGradient.save(outputFilename+"_gradient_full.png", "PNG")
	except:
		print "Failed writing ",outputFilename+"_full.png"
	for dim in dimensions:
		if tileImage.size[0] > dim :
			print "Writing "+str(dim)+"x"+str(dim)+" version: "+outputFilename+"_"+str(dim)+".png"
			try:
				tmpImage = tileImage.resize((dim,dim))
				tmpImage.save(outputFilename+"_"+str(dim)+".png", "PNG")
			except:
				print "Failed writing ",outputFilename,"_",str(dim),".png"
			if tileGradient:
				try:
					tmpImage = tileGradient.resize((dim,dim))
					tmpImage.save(outputFilename+"_gradient_"+str(dim)+".png", "PNG")
				except:
					print "Failed writing ",outputFilename,"_gradient_",str(dim),".png"

#This function lists the files within a given directory dir
def listdir_fullpath(d):
	return [os.path.join(d, f) for f in os.listdir(d)]

#This is the main program, it takes at least 2 arguments <InputFolder> and <OutputFilename>
def main(argv=None):
	print "Parsing arguments..."
	if argv is None:
		argv = sys.argv

	if len(argv) < 3:
		print "Usage: command <InputFolder> <OutputFilename>"
		print "	<InputFolder> must contain only one set of PNG files to be processed"
		print "	<OutputFilename> must contain the path and base name of the desired output, extension will be added automatically"
		print "Note: this version does not process several folders recursively. "
		print "You typed:", argv
		return 2

	#Filter only png files in the given folder
	filenamesPNG = filter(lambda x: ".png" in x, listdir_fullpath(argv[1]))
	
	#Convert into a tiled image
	if len(filenamesPNG) > 0:
		try:
			global ndimage, misc
			global np
			import numpy as np
			from scipy import ndimage, misc
			gradient = True
		except ImportError:
			print "You need SciPy and Numpy (http://numpy.scipy.org/) to also calculate the gradient!"
			gradient = False

		#From png files
		if gradient:
			imgTile, gradientTile, sliceResolution, numberOfSlices, slicesPerAxis = ImageSlices2TiledImage(filenamesPNG,loadPNG, True)
		else:
			imgTile, gradientTile, sliceResolution, numberOfSlices, slicesPerAxis = ImageSlices2TiledImage(filenamesPNG,loadPNG)
	else:
		print "No PNG files found in that folder, check your parameters or contact the authors :)."
		return 2
	
	#Write a text file containing the number of slices for reference
	try:
		try:
			print 'Creating folder',os.path.dirname(argv[2]),'...',
			os.makedirs(os.path.dirname(argv[2]))
		except OSError as exc:
			if exc.errno == errno.EEXIST and os.path.isdir(os.path.dirname(argv[2])):
				print 'was already there.'
			else:
				print ', folders might not be created, trying to write anyways...'
		except:
			print ", could not create folders, trying to write anyways..."
		with open(argv[2]+"_AtlasDim.txt",'w') as f:
			f.write(str((numberOfSlices,(slicesPerAxis,slicesPerAxis))))
	except:
		print "Could not write a text file",argv[2]+"_AtlasDim.txt","containing dimensions (total slices, slices per axis):",(numberOfSlices,(slicesPerAxis,slicesPerAxis))
	else:
		print "Created",argv[2]+"_AtlasDim.txt","containing dimensions (total slices, slices per axis):",(numberOfSlices,(slicesPerAxis,slicesPerAxis))

	#Output is written in different sizes
	WriteVersions(imgTile, gradientTile, argv[2])

if __name__ == "__main__":
	sys.exit(main())
