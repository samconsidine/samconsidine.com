---
title: "Can Neural Cellular Automata Predict Cloud Cover?"
date: 2022-03-23
draft: false
tags: ["machine learning", "neural style transfer", "edge computing", "TPU", "ReCoNet"]
categories: ["AI", "Computer Vision", "Edge Computing"]
math: true
---


# Introduction

In the intersection of AI and climate science, I'm constantly exploring novel approaches to complex problems. Today, I'd like to share an experiment that applies Neural Cellular Automata (NCA) to cloud cover prediction. This unconventional approach combines concepts from computational biology with meteorology, potentially offering new insights into weather forecasting.

## Background: Cellular Automata and Their Neural Variants

Before delving into cloud prediction, let's review Cellular Automata (CA) and their neural counterparts.

Cellular Automata are discrete models studied across various scientific disciplines. Developed by Stanislaw Ulam and John von Neumann in the 1940s, a cellular automaton consists of a grid of cells, each in one of a finite number of states. As time progresses, the states of the cells change based on fixed rules that depend on the current state of the cell and its neighbors.

A well-known example of cellular automata is Conway's Game of Life:

![Conway's Game of Life](https://upload.wikimedia.org/wikipedia/commons/e/e5/Gospers_glider_gun.gif)
*Figure 1: Conway's Game of Life, demonstrating complex behavior emerging from simple rules*

Neural Cellular Automata (NCA), introduced by Mordvintsev et al. in their 2020 paper "Growing Neural Cellular Automata", extend this concept by replacing fixed update rules with learned rules implemented by neural networks. This allows for more complex and adaptive behavior, enabling the system to learn to produce specific patterns or behaviors.

![Neural Cellular Automata](https://distill.pub/2020/growing-ca/fig1_1.mp4)
*Figure 2: Neural Cellular Automata growing a pattern (Source: Mordvintsev et al., 2020)*

In an NCA, each cell's state is updated based on its current state and the states of its neighbors, but the update function is a trainable neural network. This makes NCAs a powerful tool for modeling complex, self-organizing systems.

## Applying NCAs to Cloud Cover Prediction

The application of NCAs to cloud cover prediction is novel and potentially valuable. Cloud formations often exhibit complex, self-organizing behavior that aligns well with the NCA paradigm.

### Data and Task

For this experiment, I used data from the Climate Hack.AI competition. The dataset consists of high-resolution satellite imagery of the UK and northwestern mainland Europe, collected using EUMETSAT's Spinning Enhanced Visible and InfraRed Imager Rapid Scanning Service.

The task involves predicting the next 2 hours of satellite images given an hour of previous data. Specifically, the input is 12 128x128 crops of satellite images at 5-minute intervals, and the goal is to predict the next 24 64x64 crops at 5-minute intervals.

![Example Satellite Image](example_image.png)
*Figure 3: An uncropped example image of the UK and the region of Europe directly to the south east*

### Cloud Formation as a System of PDEs

To approach this problem using NCAs, I first modeled cloud formation as a system of partial differential equations (PDEs). The model assumes that the change in reflectance (the value at each pixel of the image) is a function of the current reflectance and the vector fields that define the directional change of reflectance over each image in the image stack.

Mathematically, this is expressed as:

$$
\frac{\partial \textbf{a}}{\partial t} = \phi \left (\mathbf{a}, \frac{\partial \mathbf{a}}{\partial x},  \frac{\partial \mathbf{a}}{\partial y} \right )
$$

where $\mathbf{a}_t$ is a vector whose first element is the bidirectional reflectance if a cloud exists at that point $(x, y)$ at time $t$.

### Connection to NCAs

To discretize this formulation and connect it to NCAs, I modeled the 2D plane as a grid of $N \times N$ cells, each holding a vector $\mathbf{a}_t : t \in \{1\ldots T\}$. The tensor $\mathbf{A} \in \mathbb{R}^{N \times N \times T}$ represents the 2D grid of these vectors.

The update function for the NCA is:

$$
\mathbf{a}_{t+1} = \mathbf{a}_t + \phi(\mathbf{a}_t, \Delta_x \mathbf{a}_t, \Delta_y \mathbf{a}_t)
$$

Here, $\Delta_x \mathbf{a}$ and $\Delta_y \mathbf{a}$ are approximations of the partial derivatives, obtained by convolving the tensor with Sobel operators:

$$
\Delta_x \mathbf{A} = \begin{bmatrix}
-1 & 0 & 1\\
-2 & 0 & 2\\
-1 & 0 & 1
\end{bmatrix} 
\ast \mathbf{A}, \quad
\Delta_y \mathbf{A} = \begin{bmatrix}
1 & 2 & 1\\
0 & 0 & 0\\
-1 & -2 & -1
\end{bmatrix} 
\ast \mathbf{A}
$$

By using a multilayer perceptron to approximate $\phi : \mathbb{R}^T \rightarrow \mathbb{R}^T$, the system is effectively modeled as an NCA.

## Experimental Setup

The NCA model takes 12 128x128 images as input and predicts the next 24 128x128 images. I used 10 update steps between frames, corresponding to a temporal resolution of 30 seconds in real time.

The neural network $\phi$ consists of 2 layers of 128 neurons with a ReLU activation for nonlinearity and a tanh output activation to scale the update between -1 and 1. I employed 1x1 convolution filters for the MLP, which are equivalent to dense layers.

### Loss Function

I used a multiscale structural similarity (SSIM) loss function:

$$
L(\mathbf{y}, \mathbf{\hat{y}}; \theta) = \sum_{k=1}^{K} \gamma^k (1 - \text{SSIM}(y_k, \hat{y}_k))
$$

where

$$
\text{SSIM}(y, \hat{y}) = \frac{(2 \mu_y \mu_{\hat{y}} + c_1) + (2 \sigma_{y \hat{y}} + c_2)} {(\mu_{y}^2 + \mu_{\hat{y}}^2 + c_1) (\sigma_y^2 + \sigma_{\hat{y}}^2 + c_2)}
$$

$y_k$ and $\hat{y_k}$ are the target image and prediction at time step $k$, and $\gamma$ is a discount factor set to 0.97.

## Results

I compared the NCA model to two benchmarks: a "last seen image" model and an optical flow model. The results are as follows:

![Results Comparison](results.png)
*Figure 4: The NCA model compared to benchmarks*

The NCA model achieved an average structural similarity of 0.710638, outperforming both the "last seen image" model (0.651985) and the optical flow model (0.705232).

Here are some example outputs from the model:

![Example Outputs](nca_out.png)
*Figure 5: Example outputs from the model for the 1st, 5th, 9th, 13th, 17th and 21st target images (left to right). The target images (top) and the predicted images from the NCA (bottom).*

## Discussion

While the NCA model shows promise by outperforming simple benchmarks, it's important to note that state-of-the-art machine learning solutions in the Climate Hack.AI competition achieved similarity scores around 0.8 for the center 64x64 crop of the images. However, the NCA approach offers some unique advantages:

1. It uses a very small neural network, making it computationally efficient.
2. It requires no non-local information, operating purely on local update rules.
3. It provides a biologically-inspired approach to climate modeling.

The self-organizing nature of NCAs is evident in the output images. Over time, intense clusters spread out and decrease in intensity, likely due to increasing uncertainty about cloud direction in future predictions.

## Conclusion

This experiment demonstrates the feasibility of applying Neural Cellular Automata to cloud cover prediction, with results that outperform simple benchmarks. While there's room for improvement compared to state-of-the-art methods, the NCA approach offers a novel, efficient, and interpretable method for modeling complex atmospheric phenomena.

The application of NCAs to this domain opens up intriguing possibilities for future research. Could more sophisticated NCA models, perhaps incorporating additional meteorological data or using more complex network architectures, further improve performance? Could the self-organizing properties of NCAs be leveraged to model other atmospheric or climate phenomena?

As we continue to face the challenges of climate change and the need for accurate weather prediction, innovative approaches like this could contribute to advancing our understanding and forecasting capabilities. The intersection of cellular automata, neural networks, and climate science represents a rich area for further exploration, and I'm eager to see how this line of research develops.
