# Exploring Design Spaces  -  Report_Generator

## Files:
- **neural_net.py** - main file with the architecture of the neural network, parameters and hyper-parameters.
- **VisualizationFunction.py** - file with all functions needed to create a 'PDF' report
- **template_test.xlsx** - template which is needed to input all data generate by the *VisualizationFunction.py*

To run the program you also need folder with dataset which if needed please contact: marcin.laskowski17@imperial.ac.uk


## Environment preparation:
Before running main file you need to have following libraries (for Linux please find commands below):
- **Anaconda**
    ```
    wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh

    bash Anaconda3-5.2.0-Linux-x86_64.sh

    conda create --no-default-packages -n sil-env python=3.5.2
    source activate sil-env
    ```
- **PyTorch 0.4.0**
    ```
    conda install pytorch torchvision cuda91 -c pytorch
    ```
- **Matplotlib**
    ```
    python -mpip install -U matplotlib
    ```
- **SciPy**
    ```
    pip install scipy
    ```
- **OpenPyXL**
    ```
    pip install openpyxl
    ```
