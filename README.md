# Experimental Validation of "Benefits of Depth in Neural Networks"

## Course Project - Theory of Deep Learning (TDL)
This repository contains the implementation of experimental tests validating theoretical results from the paper "Benefits of Depth in Neural Networks" by Matus Telgarsky (JMLR 2016).

### Project Overview
The goal of this project is to empirically verify the theoretical benefits and limitations of neural network depth as presented in the paper, specifically focusing on:
- Theorem 1.1: This theorem establishes that for any positive integer kk, there exist neural networks with O(k3) layers, O(1) nodes per layer, and Θ(1) distinct parameters that cannot be approximated by networks with O(k) layers unless they are exponentially large—requiring Ω(2k) nodes. 
- Theorem 3.12: Existence of labelings realizable by deep networks but not shallow ones
- Lemma 4.1: Probability bounds for random labeling classification
- Relationship between network depth and approximation capabilities

### Implementation Details
The code is implemented in Python using:
- NumPy for numerical computations
- Scikit-learn for neural network implementation
- Matplotlib and Seaborn for visualization
- Tqdm for progress tracking


### Experiments

#### Benefits of depth experiment
Target Function: The target function is defined as sin(10πx), a function with oscillations. This makes the task more challenging for the neural network.

Network Architectures:
    Shallow Network: The shallow network consists of one hidden layer with 1000 neurons.
    Deep Network: The deep network consists of four hidden layers, each with 20 neurons.


Evaluation: After training, the models are evaluated on a test set (500 points) and the MSE of each model's predictions is calculated.

Results: The performance of both models is plotted alongside the target function. The MSE values for both models are printed for comparison.

#### Limits of depth experiment

The implementation tests several network architectures:
- Shallow network (1 hidden layer)
- Medium network (2 hidden layers)
- Deep network (3 hidden layers)
- Very deep network (4 hidden layers)

For each architecture, we:
1. Generate random 2D points
2. Assign random binary labels
3. Train networks of varying depth
4. Compare empirical errors with theoretical bounds

### Results
Our experiments demonstrate:
1. Validation of theoretical bounds from Lemma 4.1
2. Empirical evidence of depth limitations
3. Relationship between network complexity and performance
4. Confirmation of the paper's theoretical predictions

### Usage
```bash
# Install requirements
pip install -r requirements.txt
```
And then run experiments
```bash
python benefits_of_the_depth.py
```
or
```bash
python limits_of_the_depth.py
```

### Project Structure
```
├── README.md
├── requirements.txt
├── benefits_of_the_depth.py
├── limits_of_the_depth.py
├── results/
│   ├── figures/
└── report.pdf
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