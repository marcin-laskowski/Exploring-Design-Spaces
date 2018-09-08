# Exploring Design Spaces
The goal of the project is to explore the design spaces using neural networks, which will be able to predict the functional behavior of the 2D object.


## Project Specification
During the first part of the project, an analysis was carried out to examine different neural network architectures that would act as a model capable of replacing current methods of calculating physical properties such as the Finite Element Method.

The whole project was divided into couple stages where thorough analysis was performed. The research begun with the process of generating data where three datasets was created. This step was very time consuming and very demanding because it was necessary to connect many different environments. Stress distribution was calculated by the Marc Software. However it was necessary to convert input and output files needed for training in the most convenient format for the neural network. That is why special converters was created. After first phase of the project we switched to investigation of different neural networks playing with different types of autoencoders. This analysis was very important at the beginning because it gave us general overview how neural networks behaves on our datasets. During the whole process we also explored how much we can decrease latent spaces for each task. After that task we jumped to the main phase of the project where form shape and boundary conditions (fixation and forces) we want to obtain von Mises stresses of the objects. This phase was performed on three datasets and was and was investigated using four different methods: Stage 1.2, Stage 1.3, Stage 1.4, Stage 1.5.

<p align="center">
    <img width="650" src="https://github.com/mlaskowski17/Exploring-Design-Spaces/blob/master/images/project_path.png">
</p>

As far as the neural networks is concerned PyTorch was used as the main framework. PyTorch is Python’s library for building Dynamic Computation Graphs, it allows to use dynamic data structures in the middle of the network. This means that we have the opportunity to enter a different number of inputs in any place during the training. In Pytorch, building a network is modular, which means that we implement and debug each one separately.

All results and final reports from the research are available uner the link: [https://laskowskimarcin.com/exploring-design-spaces/](https://laskowskimarcin.com/exploring-design-spaces/)


## Files
- **2D_Data_Generator** - program which allows us to generate input files for the neural network. The basic idea of the process is visualized below.
    <p align="center">
        <img width="650" src="https://github.com/mlaskowski17/Exploring-Design-Spaces/blob/master/images/generating_data_complex3.png">
    </p>
    Finally the whole research was performed od three datasets presented below:
    <p align="center">
        <img width="600" src="https://github.com/mlaskowski17/Exploring-Design-Spaces/blob/master/images/dataset5.png">
    </p>


- **2D_Neural_Networks** - folder that consists of files with neural network architectures.
    - **Stage_0** - Stage 0 is based on creating Autoencoders for Images and Labels
    <p align="center">
        <img width="650" src="https://github.com/mlaskowski17/Exploring-Design-Spaces/blob/master/images/stage_0.png">
    </p>

    - **Stage_1_2** - Stage 1_2 is approach during which input is pentagon picture (1 x 64 x 64) and output is stress distribution (1 x 64 x 64). However in this approach forces and boundary conditions are added inside neural network.
    <p align="center">
        <img width="650" src="https://github.com/mlaskowski17/Exploring-Design-Spaces/blob/master/images/stage_1_2.png">
    </p>

    - **Stage_1_3** - Stage 1_3 is approach during which there was investigated different neural network architectures when as the input we give matrix 5x6 with coordinates of the nodes, forces and fixation points.
    <p align="center">
        <img width="650" src="https://github.com/mlaskowski17/Exploring-Design-Spaces/blob/master/images/stage_1_3.png">
    </p>

    - **Stage_1_4** - Stage 1_4 was inspired by the [Uber paper](http://arxiv.org/abs/1807.03247) where it was investigated transformation from coordinate space to pixel space. This stage is similar to stage 1.2, however, input and output consists of two additional channels (with x coordinates mesh grid, and y coordinates mesh gird). On the top of that in the middle of the neural network latent space is concatenated with not only forces and fixations but also with x and y coordinates of applied boundary conditions.
    <p align="center">
        <img width="650" src="https://github.com/mlaskowski17/Exploring-Design-Spaces/blob/master/images/stage_1_4.png">
    </p>

    - **Stage_1_5** - Similar to stage 1.3, however, in this case it was added additional two channels at the output of the neural network.
    <p align="center">
        <img width="650" src="https://github.com/mlaskowski17/Exploring-Design-Spaces/blob/master/images/stage_1_5.png">
    </p>

- **Report_Generator** - folder with functions needed to automatically generate report after each run of the neural network script.
