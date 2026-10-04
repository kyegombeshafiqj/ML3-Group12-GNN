# Machine Learning 3 – Group 12

## Week 1 Progress Report

**Project:** Graph Neural Network from Scratch  
**Programming Language:** C++  
**Week:** 1  
**Period:** 2–4 October 2026

---

## 1. Objectives

The main objectives for Week 1 were:

- Understand the project requirements.
- Understand Graph Neural Networks and node classification.
- Investigate a suitable dataset.
- Understand graph representation.
- Understand graph preprocessing.
- Research GNN forward propagation.
- Research training, loss and backpropagation.
- Establish the C++ project architecture.
- Set up the GitHub repository and team workflow.

---

## 2. Project Understanding

The project aims to implement a Graph Neural Network from scratch using C++.

The system will represent data as a graph consisting of nodes and edges. Node features will be processed using graph-based operations, allowing information from neighbouring nodes to contribute to node representations.

The final system is expected to perform node classification or prediction and evaluate its performance using suitable metrics.

The main project pipeline is:

Dataset → Graph Representation → Graph Preprocessing → GNN Forward Propagation → Softmax → Prediction → Loss → Backpropagation → Gradient Descent → Evaluation

---

## 3. Project Architecture

The proposed project is divided into several major components:

1. Data loading
2. Graph representation
3. Graph preprocessing
4. GNN forward propagation
5. Softmax and prediction
6. Loss calculation
7. Backpropagation
8. Manual gradient calculation
9. Gradient descent
10. Evaluation

The project will use C++ data structures such as vectors and matrices to represent graph data, node features and model parameters.

The repository is organized into folders for source code, header files, tests, data and reports.

---

## 4. Team and GitHub Workflow

The Group 12 GitHub repository was created and organized for collaborative development.

The team will use feature branches for individual contributions. Members will work on their assigned tasks, commit their changes and push them to GitHub. Pull requests will then be reviewed and merged into the main branch.

Week 1 contributions were divided among the group members as follows:

- **Shafiq:** Project coordination and architecture
- **Prossy:** GNN fundamentals
- **Joshua:** Dataset investigation
- **Arnold:** Graph representation
- **Sharon:** Graph preprocessing
- **Samalie:** Forward propagation
- **Nesta:** Training and backpropagation
- **Tendo:** Evaluation and documentation

---

## 5. GNN Fundamentals

A Graph Neural Network operates on graph-structured data.

A graph can be represented as:

G = (V, E)

where V is the set of nodes and E is the set of edges.

Nodes represent entities while edges represent relationships between entities. Nodes may also contain numerical features.

A major concept in GNNs is message passing. During message passing, a node receives information from neighbouring nodes, aggregates the information and updates its representation.

A typical message-passing process therefore consists of:

1. Message generation
2. Aggregation
3. Node update

This allows the model to use both node features and graph connectivity for prediction.

---

## 6. Dataset Investigation

The group investigated datasets suitable for a node-classification GNN project.

The dataset must provide graph relationships, node information or features, and labels suitable for training and testing.

The final dataset will be selected based on its suitability for implementing the required GNN operations manually in C++ and the availability of manageable graph data.

Dataset preparation will include loading the data, organizing nodes and edges, preparing node features and separating data for training and testing.

---

## 7. Graph Representation

A graph consists of nodes and edges.

Nodes represent objects or entities while edges represent relationships between them.

Edges may be directed or undirected and may also be weighted or unweighted.

Two important representations investigated were the adjacency matrix and adjacency list.

For a graph with N nodes, an adjacency matrix A has dimensions N × N. For an unweighted graph:

A[i][j] = 1

indicates that a connection exists between node i and node j, while:

A[i][j] = 0

indicates that there is no connection.

An adjacency list stores the neighbours of each node. For example:

A: B, C  
B: D  
C: D  
D: none

An adjacency list can be represented in C++ using:

    vector<vector<int>> adjacency;

Node features can similarly be stored using:

    vector<vector<float>> features;

The graph structure and node feature matrix together form important inputs to the GNN.

---

## 8. Graph Preprocessing

Graph preprocessing prepares the graph before it is passed through the GNN.

Important preprocessing operations include:

- Organizing nodes and edges.
- Constructing the adjacency representation.
- Constructing the node feature matrix.
- Constructing the degree matrix.
- Normalizing the adjacency matrix.
- Preparing training and testing data.

The degree matrix contains the degree of each node on its diagonal.

Normalization is important because it helps control the scale of information being aggregated from neighbouring nodes.

The preprocessing stage will provide the graph representation required by the forward propagation stage.

---

## 9. Forward Propagation

Forward propagation is the process through which input node features are passed through the GNN to produce node representations and predictions.

A simplified GNN layer can be represented conceptually as:

H' = σ(AH W)

where:

- A represents graph connectivity or a normalized adjacency matrix.
- H represents the input node features.
- W represents learnable weights.
- σ represents an activation function.
- H' represents the updated node representations.

The general process is:

1. Gather information from neighbouring nodes.
2. Aggregate the information.
3. Apply a weight transformation.
4. Apply an activation function.
5. Produce updated node representations.

The final layer can produce class scores which can then be converted into probabilities using softmax.

---

## 10. Training and Backpropagation

Training involves adjusting the model's weights so that the predicted classes become closer to the true labels.

A loss function measures the difference between predictions and the correct labels.

For multiclass node classification, cross-entropy loss can be used.

Backpropagation calculates how the loss changes with respect to the model parameters.

The project will manually calculate the required gradients rather than relying on an automatic differentiation framework.

Gradient descent can then be used to update the weights:

    W_new = W_old - learning_rate × gradient

This process is repeated over multiple training iterations until the model learns useful parameters.

---

## 11. Evaluation

The trained GNN will be evaluated using appropriate performance measures.

The planned evaluation measures include:

- Accuracy
- Precision
- Recall
- Training loss
- Training time

The dataset will be divided into training and testing portions so that the model can be trained on one portion and evaluated on unseen data.

The evaluation results will be used to determine how well the implemented GNN performs.

---

## 12. Challenges and Observations

During Week 1, the main challenges were understanding the complete GNN pipeline and coordinating the different research tasks.

The team also had to determine how graph structures, node features, preprocessing, forward propagation and training would connect together in the final C++ implementation.

GitHub collaboration was also part of the initial setup. The group established a workflow using branches, commits and pull requests to organize contributions.

The Week 1 research has provided the foundation required to begin implementation in Week 2.

---

## 13. Week 2 Plan

During Week 2, the group plans to:

- Finalize the dataset.
- Implement the dataset loader in C++.
- Implement node and feature representations.
- Implement edge and graph representations.
- Implement the adjacency structure.
- Implement graph preprocessing.
- Implement degree matrix calculation.
- Implement adjacency normalization.
- Begin testing the graph representation.
- Integrate the first implemented components.

---

## 14. Conclusion

Week 1 focused mainly on understanding the project and designing the structure of the GNN implementation.

The group investigated GNN fundamentals, datasets, graph representation, graph preprocessing, forward propagation, training and evaluation.

The GitHub repository and collaborative workflow were also established.

The research completed during Week 1 provides the theoretical foundation for beginning the implementation of the Graph Neural Network in C++ during Week 2.

---

## References

1. Hamilton, W. L. *Graph Representation Learning*. Morgan & Claypool Publishers. Chapter 5: The Graph Neural Network Model.

2. NetworkX Documentation. *Introduction and Graphs*.

3. PyTorch Geometric Documentation. *Introduction by Example – Data Handling of Graphs*.

4. PyTorch Geometric Documentation. *Message Passing Networks*.
