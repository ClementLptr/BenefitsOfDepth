# Experimental Validation of "Benefits of Depth in Neural Networks"

## Course Project - Theory of Deep Learning (TDL)
This repository contains the implementation of experimental tests validating theoretical results from the paper "Benefits of Depth in Neural Networks" by Matus Telgarsky (JMLR 2016).

### Project Overview
The goal of this project is to empirically verify the theoretical benefits and limitations of neural network depth as presented in the paper, specifically focusing on:
- Theorem 1.1 demonstrates that increasing the depth of a neural network enhances its ability to approximate complex functions. 

### Implementation Details
The code is implemented in Python using:
- NumPy for numerical computations
- Scikit-learn for neural network implementation
- Matplotlib and Seaborn for visualization

### Experiment: Benefits of Depth
#### Target Function
The target functions are defined as sin(10πx), a highly oscillatory function that presents a challenge for the neural network to approximate and sin(5πx) a lower oscillatory function.

#### Network Architectures
- **Shallow Network**: Consists of one hidden layer with 1000 neurons.
- **Deep Network**: Consists of four hidden layers, each with 20 neurons.

#### Evaluation
After training, the models are evaluated on a test set of 500 points, and the Mean Squared Error (MSE) of each model's predictions is calculated.

#### Results
The performance of both models is visualized alongside the target function. The MSE values for both models are compared and plotted for analysis.

### Results Summary
Our experiment demonstrates the benefits of increasing network depth, as shown by:
1. The ability of the deep network to better approximate the oscillatory target function compared to the shallow network.
2. A comparison of MSE values for both networks.

### Usage
```bash
# Install requirements
pip install -r requirements.txt
```
And then run the experiment:
```bash
python benefits_of_the_depth.py
```

### Project Structure
```
├── README.md
├── requirements.txt
├── main.py
├── model.py
├── model_visualisation.py
├── utils.py
├── results/
│   ├── figures/
├── report.pdf
└── Benefits of the depth in neural network.pdf
```

### References
Telgarsky, M. (2016). Benefits of depth in neural networks. JMLR: Workshop and Conference Proceedings, 49:1-23.

### Author  
Clément Leprêtre  
Student ID: 2101396  
Theory of Deep Learning  
CentraleSupélec  
SD10

### License
This project is provided for academic purposes only. Please cite both the original paper and this implementation if you use this code.
