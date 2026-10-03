# Week1 contribution
**Member:** NABUKEERA PROSCOVIA 25/U/0720
**Project:** Machine Learning 3 - Graph Neural Network from scartch
**Group:** 12
**Week:** 1
**Task:** GNN fundamentals

## 1. What is a Graph Neural Network (GNN)?

A Graph Neural Network is a type of neural network that operates directly on graph-structured data. Unlike standard CNNs (Convolutional Neural Networks) that work on grids/images or RNNs (Recurrent Neural Networks) that work on sequences, GNNs are designed to learn from data where relationships between entities matter.

GNNs maintain structural relationships by propagating information across connected nodes and edges, making them essential for domains such as social networks, citation networks, molecular graphs, and traffic modeling.

Many traditional machine-learning methods are designed for settings where observations can be treated as independent or relationships between observations are not explicitly represented. In graph data, the relationship between nodes are an important part of the data.

## 2. Graphs, Nodes, Edges and Node Features

* **Graph:** A graph is defined as G = (V, E) where V is set of vertices/nodes and E is set of edges.

* **Nodes (V):** These are individual entities within a network. Example: In a citation network, each node is a research paper. In a social network, each node is a person. Number of nodes = N.

* **Edges (E):** These are the connections/relationships between nodes. These can be directed or undirected. Example: If person A is friends with B, there is an edge A-B. Represented by adjacency.

* **Node Features (X):** Each node has a feature vector that describes it. This is the input data for learning.

Example: For a paper node, features could be [number_of_words, contains_word_neural, year]. For a person node, [age, interest_score].

## 3. Node Classification and Message Passing

## Node Classification

Node classification is the process of predicting the category or class of a node in a graph.

For example, in a social network:
- **Nodes** represent people.
- **Edges** represent friendships or connections.
- **Node features** can include information such as age, interests, or education level.
- **Node labels** can be categories such as Student, Teacher, or Other.
A Graph Neural Network (GNN) uses the features of a node and information from its neighboring nodes to predict the node's class.

## Message Passing

Message passing is a process used by Graph Neural Networks (GNNs) to allow nodes to exchange and gather information from their neighboring nodes.

Each node has its own features, but the information from its neighboring nodes can also help it understand its role in the graph. During message passing, a node receives information from its connected neighbors and combines this information with its own features.

The message passing process generally involves three main steps:

1. **Message:** Each node sends information about its features to its neighboring nodes.

2. **Aggregation:** A node collects and combines the information received from its neighbors.

3. **Update:** The node combines the aggregated information with its own features to create an updated representation.

This process can be repeated for several layers. With each layer, a node can gather information from nodes that are further away in the graph.

Message passing is important because it allows a GNN to consider both the features of individual nodes and the relationships between nodes. This helps the GNN make better predictions for tasks such as node classification.

In simple terms, **message passing allows a node to learn from its neighbors.**

## 4.  Small Example: 3-Node Citation Network

Consider a small academic citation network consisting of three research papers: **Paper A with 2500 words, Paper B with 1800 words, and Paper C with 4000**. **Paper A and B have the keyword nueral but C doesn't have**

The citation relationships are:

- Paper A cites Paper B.
- Paper B cites Paper C.
- Paper A and Paper C are not directly connected.

### 1. Graph Representation

The graph can be represented using an adjacency matrix. Since the citations are directed, the matrix indicates the direction of each citation.

The adjacency matrix is:

$$
A =
\begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 1 \\
0 & 0 & 0
\end{bmatrix}
$$

Each paper has two node features:

- **Feature 1:** Word count in thousands.
- **Feature 2:** Whether the paper contains a neural-network-related keyword, where `1` means yes and `0` means no.

The node features are:

| Paper | Word Count (thousands) | Neural Keyword |
|---|---:|---:|
| A | 2.5 | 1 |
| B | 1.8 | 1 |
| C | 4.0 | 0 |

Therefore, the feature matrix is:

$$
X =
\begin{bmatrix}
2.5 & 1 \\
1.8 & 1 \\
4.0 & 0
\end{bmatrix}
$$
X is the feature matrix.

### 2. One Step of Message Passing Using Mean Aggregation

Consider updating the representation of **Paper B** using information from the paper it cites, **Paper C**.

The feature vector of Paper C is:

\[
h_C = [4.0,0]
\]

Since Paper C is the only neighbor considered for Paper B in this example, the message received by Paper B is:

\[
m_B = \text{Mean}(\{h_C\})
\]

\[
m_B = \text{Mean}(\{[4.0,0]\})
\]

\[
m_B = [4.0,0]
\]

Paper B's original feature vector is:

\[
h_B = [1.8,1]
\]

The original features and the aggregated message are combined using element-wise addition:

\[
h_B^{(\text{new})}=h_B+m_B
\]

\[
h_B^{(\text{new})}=[1.8,1]+[4.0,0]
\]

\[
\boxed{h_B^{(\text{new})}=[5.8,1]}
\]

After this message-passing step, the representation of Paper B contains information from both **Paper B itself and Paper C**, its connected neighbor. This illustrates the fundamental principle of a Graph Neural Network: node representations can be updated by aggregating information from neighboring nodes.

## Sources

1. Hamilton, W. L. *Graph Representation Learning — Chapter 5: Graph Neural Networks*.  
   https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book-Chapter_5-GNNs.pdf

2. PyTorch Geometric. *Message Passing Networks*.  
   https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.MessagePassing.html

