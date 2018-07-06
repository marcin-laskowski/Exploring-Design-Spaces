# Exploring Design Spaces / 2D_Data_Generator

## Setup
To run the data generator you need:
- Python 2.7 with matplotlib, PIL
- Marc Software 2017 with py_post library

## Files
- **Main.py** - main file needed to run the simulation
- **BlackBox_FE.py** - function which creates the procedure file that run the Marc Software and perform simulation
- **Generate_Files_Func.py** - function which creates several InputParams.txt files
- **GenerateDataFunctions.py** - functions that perform operations on the files to obtain matrices need for NN
- **ModelGenerator.bat**, **ModelRun.bat**, **baseline.proc**, **original.proc** - files needed to create procedure file
