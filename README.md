# GEG2_Chap22_ClothDemo
This is Visual Studio 2019 compile able code from Game Engine Gems 2 Chapter 22 demo by Marco Fratarcangeli. The original demo was written with CUDA 3.0 SDK with cutil 
library and many other libraries which are not available anymore. Luckily, I managed to get the required libraries oclUtils and shrUtils from [NVIDIA OpenCL SDK Samples](https://developer.download.nvidia.com/compute/cuda/4_2/rel/sdk/website/OpenCL/html/samples.html). This code is self contained as it includes everything required. 

## Requirements
This code has been tested on Windows 10 x64 machine with CUDA 11.2 on VisualStudio 2019.

## How to run 
Simply open src/phys_sim.sln and press the play button. 

## Demo Screenshot
![image](https://user-images.githubusercontent.com/1354859/216827419-8ce7f1bf-69e7-4435-84fc-2430b15f9703.png)

## Author Details
Main Author's Publication Page: https://mfratarcangeli.github.io/publication/

Game Engine Gems 2 Article Page: https://mfratarcangeli.github.io/pdf/geg2.Fratarcangeli.pdf

Game Engine Gems 2 Article Demo: https://mfratarcangeli.github.io/code/1008.geg2.zip

## About this Repository
This repository was compiled by Dr. Muhammad Mobeen Movania for the larger good of community.

## Original Article ReadMe Starts Here ##
GPGPU Cloth simulation using GLSL, OpenCL, and CUDA - Game Engine Gems 2
by Marco Fratarcangeli - 2010
  marco@fratarcangeli.net
  www.fratarcangeli.net
  
The demo allows you to change both the computing platform and the number of simulated particles at run time. 

Hold down the left mouse button to rotate the camera. 

Hold down the middle mouse button and move up and down to zoom.

Hold down the right mouse button to chose the collision primitive(s).

The demo starts in CPU mode, showing a grid of 32x32 particles.

Press numbers from 1 to 4 to increase the resolution of the grid, from 32x32 to 256x256.

Press function key from F1 to F4 to switch computing platform:

F1 for CPU; F2 for GLSL; F3 for OpenCL; F4 for CUDA.

Press ‘p’ for entering in profiling mode

Press ‘w’ for toggling wireframe 

Press Space to freeze the simulation

Discuss, ask, appreciate and criticize here: marco@fratarcangeli.net
