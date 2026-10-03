# Week 1 Contribution

**Member:** Kyegombe Shafiq .J. 25/U/08737/PS 
**Project:** Machine Learning 3 - Graph Neural Network from Scratch  
**Group:** 12  
**Week:** 1  
**Task:** Project Coordination and Architecture  

## 1. Understanding of the Project

The project is to design and implement a Graph Neural Network (GNN) from scratch using C++. The main objective is to understand and implement the major stages of a GNN without depending on a ready-made GNN library to perform the main learning operations.

The system will represent data as a graph consisting of nodes and edges. Each node will have a feature vector, and the connections between nodes will allow information to be passed between neighboring nodes.

The implemented GNN will be used for node classification. The system will process the graph, propagate information between connected nodes, generate predictions, calculate the loss, calculate gradients manually, update the model parameters using gradient descent, and evaluate the trained model.

---

## 2. Overall Project Pipeline

The planned workflow of the system is:

```text
Dataset
   |
   v
Data Loading
   |
   v
Graph Representation
   |
   v
Graph Preprocessing
   |
   +--> Adjacency Matrix / Adjacency List
   |
   +--> Degree Matrix
   |
   +--> Normalized Adjacency Matrix
   |
   v
Train/Test Split
   |
   v
GNN Forward Propagation
   |
   v
Node Representations
   |
   v
Output Layer
   |
   v
Softmax
   |
   v
Node Predictions
   |
   v
Loss Calculation
   |
   v
Backpropagation
   |
   v
Manual Gradient Calculation
   |
   v
Gradient Descent
   |
   v
Updated Weights
   |
   v
Evaluation