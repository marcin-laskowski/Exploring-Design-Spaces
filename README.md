# Exploring Design Spaces
The goal of the project is to explore the design spaces using neural networks, which will be able to predict the functional behavior of the object.


## Project Specification
During the first part of the project, an analysis will be carried out to examine different neural network architectures that would act as a model capable of replacing current methods of calculating physical properties such as the Finite Element Method.

As a proof of concept, a strength analysis of a 2D plate will be carried out, to which various types of weight will be applied. Output from the neural network will be compared with the output from the FEM software. If that part will work, in the later stage we would like to explore the design space (design and CAD) e.g. by Autoencoders. During the second part of the project we would like to answer two questions: What is the most optimal way to input 3D structure to the network, and what is the most convenient architecture of the Neural Network to operate on the 3D objects.

The part that will be used to prepare training data (3D solids and FEM analysis) will be carried out in the MARC MSC Software. The program is capable of solving nonlinear problems, allows for an easy and quick execution of a fully automated analysis of the deformation of several hundred thousand blocks at the same time.

As far as the neural networks is concerned PyTorch will be used as the main framework. PyTorch is Python’s library for building Dynamic Computation Graphs, it allows to use dynamic data structures in the middle of the network. This means that we have the opportunity to enter a different number of inputs in any place during the training. In Pytorch, building a network is modular, which means that we implement and debug each one separately.


## Files
- **2D_Data_Generator** - folder that consists of files needed to generate the 2D files needed to perform investigation of the problem.

- **2D_Neural_Networks** - folder that consists of files with neural network architectures.
    - **Stage_0** - Stage 0 is based on creating Autoencoders for Images and Labels
    <p align="center">
        <img width="1000" src="https://github.com/mlaskowski17/Exploring-Design-Spaces/blob/master/images/stage_0.jpg">
    </p>

    - **Stage_1_1** - Stage 1 is approach during which there was investigated different neural network architectures when as the input we give 3 channels (shape, X-coordinates of Forces, Y-coordinates of forces) with dimensions 64px x 64px and as the output we get stress distribution (1 channel, 64px 64px)
    <p align="center">
        <img width="1000" src="https://github.com/mlaskowski17/Exploring-Design-Spaces/blob/master/images/stage_1_1.jpg">
    </p>
    
    - **Stage_1_2** - Stage 1 is approach during which there was investigated different neural network architectures when as the input we
    <p align="center">
        <img width="1000" src="https://github.com/mlaskowski17/Exploring-Design-Spaces/blob/master/images/stage_1_2.jpg">
    </p>

    - **Stage_1_3** - Stage 1 is approach during which there was investigated different neural network architectures when as the input we
    <p align="center">
        <img width="1000" src="https://github.com/mlaskowski17/Exploring-Design-Spaces/blob/master/images/stage_1_3.jpg">
    </p>

- **Report_Generator** - folder with functions needed to automatically generate report after each run of the neural network script.
