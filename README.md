Atlas Conversion Scripts
========================

Introduction
-------------
This repository contains some scripts to help converting 3D volume data into a WebGL compatible 2D texture atlas.

The scripts allow you to generate an atlas from different types of volume data sources. This are the supported data types:
*	Common image formats like __PNG__, __JPEG__... which are supported by _PIL_
*	__DICOM__
*	__RAW__ 
*	__NRRD__ 

Also, there is a template script you can use to adapt it to your own volume data type.  

Documentation
--------------
You can found the necessary information about how you can use the scripts and how to visualize the atlas on the [wiki](https://github.com/VolumeRC/AtlasConversionScripts/wiki).

Set-up
------
To easy install the required dependencies we provide [Anaconda](https://www.continuum.io/downloads) `environment` files.

### Windows

    conda env create -f environment.yml
    activate volren-atlas

### Linux

    conda env create -f environment-linux.yml
    source activate volren-atlas

Related Publication
-------------------
*	_John Congote, Alvaro Segura, Luis Kabongo, Aitor Moreno, Jorge Posada, and Oscar Ruiz. 2011_. __Interactive visualization of volumetric data with WebGL in real-time__. In Proceedings of the 16th International Conference on 3D Web Technology (Web3D '11). ACM, New York, NY, USA, 137-146.
