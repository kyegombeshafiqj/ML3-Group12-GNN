Week 1 Contribution
Nalukwago Phiona Sharon
Group 12
Graph preprocessing
In graph preprocessing and graph neural networks (GNNs), the structural layout of a network is converted into mathematical matrices so that algorithms can process them efficiently.
1. The Degree Matrix Explanation
The Degree Matrix (\(D\)) is a diagonal matrix that contains information about the "degree" (number of connected edges) of each node in the graph.
• Diagonal elements (\(D_{ii}\)): Represent the degree of node \(i\).
• Off-diagonal elements (\(D_{ij}\) where \(i \neq j\)): Are always zero.
For an undirected and unweighted graph, the degree of a node is simply the count of its direct neighbors.
2. Calculating the Degree Matrix for a Small Graph
 Using a simple, undirected 4-node graph for  calculations.
The Graph Structure:
• Node 1 is connected to: Node 2, Node 3 (Degree = 2)
• Node 2 is connected to: Node 1, Node 3, Node 4 (Degree = 3)
• Node 3 is connected to: Node 1, Node 2 (Degree = 2)
• Node 4 is connected to: Node 2 (Degree = 1)
First, we express this as a standard Adjacency Matrix (\(A\)), where a 1 represents an edge and 0 represents no edge:
\(A=\left(\begin{matrix}0&1&1&0\\ 1&0&1&1\\ 1&1&0&0\\ 0&1&0&0\end{matrix}\right)\)
To get the Degree Matrix (\(D\)), we sum the rows of \(A\) (or columns, since it's undirected) and place those sums on the diagonal:
\(D=\left(\begin{matrix}2&0&0&0\\ 0&3&0&0\\ 0&0&2&0\\ 0&0&0&1\end{matrix}\right)\)
3. What is Normalized Adjacency?
When passing graph data into a machine learning model, using the raw adjacency matrix \(A\) directly causes structural issues. Normalized Adjacency scales the edge values based on node degrees to prevent mathematical instability.
There are two primary ways to normalize the adjacency matrix:
. Symmetric Normalization (Common in Graph Convolutional Networks / GCNs):
\(A_{sym}=D^{-1/2}AD^{-1/2}\)
. Random Walk / Left Normalization:
\(A_{rw}=D^{-1}A\)
   
. Step-by-Step Symmetric Normalization
To compute \(A_{sym} = D^{-1/2} A D^{-1/2}\), we need to follow these matrix operations:
Step A: Calculating \(D^{-1/2}\)
To find \(D^{-1/2}\), take the reciprocal of the square root of each diagonal element in \(D\). If a node has a degree of 0, it stays 0.
• \(2 \rightarrow \frac{1}{\sqrt{2}} \approx 0.707\)
• \(3 \rightarrow \frac{1}{\sqrt{3}} \approx 0.577\)
• \(1 \rightarrow \frac{1}{\sqrt{1}} = 1\)
\(D^{-1/2}=\left(\begin{matrix}0.707&0&0&0\\ 0&0.577&0&0\\ 0&0&0.707&0\\ 0&0&0&1\end{matrix}\right)\)
Step B: Computing \(D^{-1/2} A\)
Multiplying \(D^{-1/2}\) on the left scales each row \(i\) by \(D_{ii}^{-1/2}\):
\(D^{-1/2}A=\left(\begin{matrix}0&0.707&0.707&0\\ 0.577&0&0.577&0.577\\ 0.707&0.707&0&0\\ 0&1&0&0\end{matrix}\right)\)
Step C: Computing \((D^{-1/2} A) D^{-1/2}\)
Multiplying by \(D^{-1/2}\) on the right scales each column \(j\) by \(D_{jj}^{-1/2}\):
\(A_{sym}=\left(\begin{matrix}0&(0.707\times 0.577)&(0.707\times 0.707)&0\\ (0.577\times 0.707)&0&(0.577\times 0.707)&(0.577\times 1)\\ (0.707\times 0.707)&(0.707\times 0.577)&0&0\\ 0&(1\times 0.577)&0&0\end{matrix}\right)\)
The Final Symmetric Normalized Adjacency Matrix:
\(A_{sym}\approx \left(\begin{matrix}0&0.408&0.500&0\\ 0.408&0&0.408&0.577\\ 0.500&0.408&0&0\\ 0&0.577&0&0\end{matrix}\right)\)
(Note: Elements are calculated using the formula \((A_{sym})_{ij} = \frac{A_{ij}}{\sqrt{\text{deg}(i)\text{deg}(j)}}\)

Why Normalized Adjacency is Useful
1. Prevents Exploding/Vanishing Gradients: If you pass the raw adjacency matrix \(A\) through multiple neural network layers, multiplying feature vectors by \(A\) repeatedly will scale up the features of highly connected nodes exponentially. Normalization keeps the eigenvalues scaled within a stable numerical range (typically \([-1, 1]\)).
2. Balances Influence of High vs. Low Degree Nodes: A hub node with 1,000 connections shouldn't dominate message passing just because of its volume. Normalization reduces the weight of edges connected to high-degree nodes.
3. Averages Neighboring Information: It transforms the operation from a harsh summation of neighbor features into a weighted or localized average, which makes feature aggregation physically meaningful across different graph sizes.
 
