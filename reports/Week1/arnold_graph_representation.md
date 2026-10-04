# Week 1 Contribution – Graph Representation

**Member:** Arnold Zziwa  
**Project:** Machine Learning 3 – Graph Neural Network from Scratch  
**Group:** 12  
**Week:** 1  
**Task:** Graph Representation

## 1. Introduction

A graph is a mathematical structure used to represent relationships between objects. A graph is commonly represented as:

G = (V, E)

where:

- V represents the set of nodes (vertices).
- E represents the set of edges connecting the nodes.

Graphs are useful for representing data where relationships between objects are important. In a Graph Neural Network (GNN), both the graph structure and node features are used to generate useful representations for prediction and classification.

## 2. Nodes

A node represents an individual object or entity in a graph.

For example, in a social network, each person can be represented as a node. In a citation network, each research paper can be represented as a node.

Each node can also have features that describe it. These features are represented numerically so that they can be processed by the GNN.

## 3. Edges

An edge represents a relationship or connection between two nodes.

For example:

- In a social network, an edge can represent a friendship.
- In a citation network, an edge can represent one paper citing another.
- In a communication network, an edge can represent a connection between devices.

Edges can be:

- **Directed:** the connection has a direction, such as A → B.
- **Undirected:** the connection works in both directions, such as A — B.
- **Weighted:** an edge has a numerical value representing the strength or importance of the connection.
- **Unweighted:** the edge simply indicates that a connection exists.

The choice between directed and undirected representation depends on the dataset and the type of relationship being modelled.

## 4. Adjacency Matrix

An adjacency matrix is a matrix used to represent the connections between nodes.

For a graph containing N nodes, the adjacency matrix A has dimensions N × N.

For an unweighted graph:

- A[i][j] = 1 means that an edge exists between node i and node j.
- A[i][j] = 0 means that no edge exists.

For example, consider three nodes:

A, B and C

with directed edges:

A → B  
B → C

The adjacency matrix is:

    A  B  C
A   0  1  0
B   0  0  1
C   0  0  0

The rows represent the source nodes and the columns represent the destination nodes.

For an undirected graph, an edge between two nodes appears in both corresponding positions of the matrix.

The adjacency matrix provides a convenient mathematical representation of graph connectivity and is important in GNN operations such as neighbourhood aggregation.

## 5. Adjacency List

An adjacency list stores the neighbours of each node instead of storing all possible pairs of nodes.

For the example above:

    A: B
    B: C
    C: none

An adjacency list can require considerably less memory than an N × N adjacency matrix when the graph is sparse, because it stores only the connections that actually exist.

For our C++ implementation, an adjacency-list representation can be implemented using:

    vector<vector<int>>

where the outer vector represents the nodes and each inner vector contains the neighbouring nodes.

For example:

    vector<vector<int>> adjacency = {
        {1},
        {2},
        {}
    };

This represents:

    Node 0 → Node 1
    Node 1 → Node 2
    Node 2 → no neighbours

## 6. Node Feature Matrix

The graph structure describes how nodes are connected, but a GNN also requires information describing the nodes themselves.

These values are called node features.

If there are N nodes and each node has F features, the feature matrix can be represented as:

    X ∈ R^(N × F)

Each row corresponds to one node, while each column represents one feature.

For example, consider three nodes with two features each:

    X =
    [ 1   0.5 ]
    [ 0   1.0 ]
    [ 1   1.5 ]

The first row contains the features of node 0, the second row contains the features of node 1, and the third row contains the features of node 2.

The feature matrix is passed through the GNN layers together with the graph structure.

## 7. Example Graph Representation

Consider a graph with four nodes:

    A, B, C, D

and the edges:

    A → B
    A → C
    B → D
    C → D

The adjacency list is:

    A: B, C
    B: D
    C: D
    D: none

The adjacency matrix is:

       A  B  C  D
    A  0  1  1  0
    B  0  0  0  1
    C  0  0  0  1
    D  0  0  0  0

Suppose every node has two features. A possible feature matrix is:

    X =
    [ 1   0 ]
    [ 0   1 ]
    [ 1   1 ]
    [ 0   0 ]

The graph structure and feature matrix together provide the main input information required by the GNN.

## 8. Graph Representation in C++

For the project, the graph can be represented using standard C++ data structures.

An adjacency list can be represented as:

    vector<vector<int>> adjacency;

The node feature matrix can be represented as:

    vector<vector<float>> features;

For a weighted graph, the adjacency information can additionally store the edge weights.

The final representation used by the project will depend on the selected dataset and the graph structure agreed upon by the group.

## 9. Importance to the GNN

Graph representation is important because the GNN uses the connections between nodes to determine which neighbouring information should be combined.

During message passing, a node receives information from its neighbours, aggregates that information, and uses it to produce an updated representation.

Therefore, the graph representation provides the structure that controls how information flows between nodes.

## 10. Summary

The main components of the graph representation are:

1. Nodes – represent entities in the dataset.
2. Edges – represent relationships between nodes.
3. Adjacency matrix – represents graph connectivity using a matrix.
4. Adjacency list – stores the neighbours of each node.
5. Node feature matrix – stores numerical information describing each node.

These representations form the foundation for the graph preprocessing and GNN forward-propagation stages of the project.

## References

1. Hamilton, W. L. *Graph Representation Learning*. Morgan & Claypool Publishers. Chapter 5: The Graph Neural Network Model.

2. NetworkX Documentation. *Introduction and Graphs*.

3. PyTorch Geometric Documentation. *Introduction by Example – Data Handling of Graphs*.
