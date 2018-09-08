# Exploring Design Spaces
The goal of the project is to explore the design spaces using neural networks, which will be able to predict the functional behavior of the object.


## Project Specification
During the first part of the project, an analysis will be carried out to examine different neural network architectures that would act as a model capable of replacing current methods of calculating physical properties such as the Finite Element Method.

As a proof of concept, a strength analysis of a 2D plate will be carried out, to which various types of weight will be applied. Output from the neural network will be compared with the output from the FEM software. If that part will work, in the later stage we would like to explore the design space (design and CAD) e.g. by Autoencoders. During the second part of the project we would like to answer two questions: What is the most optimal way to input 3D structure to the network, and what is the most convenient architecture of the Neural Network to operate on the 3D objects.

The part that will be used to prepare training data (3D solids and FEM analysis) will be carried out in the MARC MSC Software. The program is capable of solving nonlinear problems, allows for an easy and quick execution of a fully automated analysis of the deformation of several hundred thousand blocks at the same time.

As far as the neural networks is concerned PyTorch will be used as the main framework. PyTorch is Python’s library for building Dynamic Computation Graphs, it allows to use dynamic data structures in the middle of the network. This means that we have the opportunity to enter a different number of inputs in any place during the training. In Pytorch, building a network is modular, which means that we implement and debug each one separately.


## Files
- **2D_Data_Generator** - program which allows us to generate input files for the neural network. The basic idea of the process is visualized below.
    <p align="center">
        <img width="1000" src="https://github.com/mlaskowski17/Exploring-Design-Spaces/blob/master/images/generating_data_complex3.png">
    </p>
    Finally the whole research was performed od three datasets presented below:
    <p align="center">
        <img width="1000" src="https://github.com/mlaskowski17/Exploring-Design-Spaces/blob/master/images/dataset5.png">
    </p>


- **2D_Neural_Networks** - folder that consists of files with neural network architectures.
    - **Stage_0** - Stage 0 is based on creating Autoencoders for Images and Labels
    <p align="center">
        <img width="1000" src="https://github.com/mlaskowski17/Exploring-Design-Spaces/blob/master/images/stage_0.png">
    </p>

    - **Stage_1_2** - Stage 1_2 is approach during which input is pentagon picture (1 x 64 x 64) and output is stress distribution (1 x 64 x 64). However in this approach forces and boundary conditions are added inside neural network.
    <p align="center">
        <img width="1000" src="https://github.com/mlaskowski17/Exploring-Design-Spaces/blob/master/images/stage_1_2.png">
    </p>

    - **Stage_1_3** - Stage 1_3 is approach during which there was investigated different neural network architectures when as the input we give matrix 5x6 with coordinates of the nodes, forces and fixation points.
    <p align="center">
        <img width="1000" src="https://github.com/mlaskowski17/Exploring-Design-Spaces/blob/master/images/stage_1_3.png">
    </p>

    - **Stage_1_4** - Stage 1_4 was inspired by the [Uber paper](http://arxiv.org/abs/1807.03247) where it was investigated transformation from coordinate space to pixel space. This stage is similar to stage 1.2, however, input and output consists of two additional channels (with x coordinates mesh grid, and y coordinates mesh gird). On the top of that in the middle of the neural network latent space is concatenated with not only forces and fixations but also with x and y coordinates of applied boundary conditions.
    <p align="center">
        <img width="1000" src="https://github.com/mlaskowski17/Exploring-Design-Spaces/blob/master/images/stage_1_4.png">
    </p>

    - **Stage_1_5** - Similar to stage 1.3, however, in this case it was added additional two channels at the output of the neural network.
    <p align="center">
        <img width="1000" src="https://github.com/mlaskowski17/Exploring-Design-Spaces/blob/master/images/stage_1_5.png">
    </p>

- **Report_Generator** - folder with functions needed to automatically generate report after each run of the neural network script.
