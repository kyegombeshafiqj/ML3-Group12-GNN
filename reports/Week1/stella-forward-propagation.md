Week 1 Contribution **Member:** KYOMUHENDO SAMALIE **REG NO.**25/u/08738/ps
**Project:** Machine Learning 3 — Graph Neural Network from Scratch
**Group:** 12 **Week:** 1
**Task:** Forward Propagation
I would like to declare my use of AI to help me figure out the description of this part of the project.
Explain GNN forward pass
A forward pass answers one question: how does each node update its numbers using its neighbours' numbers?
Every layer does the same three things in order.
Gather: each node collects its neighbours' feature vectors (aggregation, using the graph structure).
Transform: multiply the result by a learned weight matrix W (the weights).
Squash: apply a nonlinearity such as ReLU (the activation).
Worked example: Alice's update, with real numbers

Using the 4-person graph from before, Alice's friends are Bob {0, 1, 1.5} and Carol {2, 0, 1}.

Gather (average):
avg = ((0+2)/2, (1+0)/2, (1.5+1)/2) = (1, 0.5, 1.25)

Transform with W = {{0.5, -0.2}, {0.1, 0.8}, {-0.3, 0.4}}:

new₀ = 1×0.5 + 0.5×0.1 + 1.25×(−0.3) = 0.175
new₁ = 1×(−0.2) + 0.5×0.8 + 1.25×0.4 = 0.7

Squash (ReLU): both numbers are positive, so they stay.

Alice's features changed from (1, 2, 0.5) to (0.175, 0.7). They are now built from her friends' information instead of her own.
PS:The forward pass makes a prediction, a loss function measures how wrong it was, 
and backpropagation works out how to nudge each number in W to make the prediction slightly better.
This repeats thousands of times. After training, W holds values that turned out to be useful.

Explain features, graph structure, aggregation, weights and activation
Think of the graph as a social network. The five ingredients answer five separate questions.

Features:	What do we know about each node?
Graph structure:	Who is connected to whom?
Aggregation:	How do we combine the neighbours' information?
Weights:	How do we mix the numbers into something useful?
Activation:	How do we let the network learn non-straight-line patterns?

1. Features: what each node knows

Every node carries a list of numbers describing it. In a social network this could be age, number of posts, and activity level. 
In a power grid it could be voltage, load, and capacity. In a molecule it could be the atom type.
The table is the input to the network. After each layer the numbers change, so the features become richer. 
The first layer's output is a new features table, which the next layer takes as input.
Graph structure: who talks to whom

2.Graph structure
This is the list of connections. It does not change during the forward pass, and a basic GNN does not learn it.
It acts as a routing map that says where information is allowed to flow.
This is what makes a GNN different from an ordinary neural network. An ordinary network treats every input as independent. 
A GNN lets a node's output depend on the nodes it is connected to

3.Aggregation: combining the neighbours

A node has several neighbours, so it needs a rule to squash their feature lists into one list.
The common rules are sum, average, and max.
The rule must give the same answer no matter what order the neighbours are visited in.
Neighbours have no natural order, so the result must not depend on one. Sum, average, and max all pass this test.
This step is the only place where the graph structure matters.
It is where "who is connected to whom" turns into actual number-crunching

4.Weights:The learned Mixing Tape
After aggregation, a node has a list of numbers, but they may not be in the most useful form. 
The weight matrix W converts them into a new list, where each new number is a weighted sum of the old ones.
The values in W are what the network learns during training. They start random and get adjusted until the outputs become useful.
The same W is used by every node, so the number of learned values does not grow with the size of the graph.
This is the same idea as a convolution kernel being reused at every position in an image.


5. Activtaion
The activation is a simple function applied to each number on its own. ReLU is the most common one:
Without it, there is a problem. Aggregating and multiplying by W are both linear operations, and stacking linear operations just gives another linear operation.
Ten layers without activations would be no more powerful than one layer.
The activation bends the output so each layer can add something new, which lets the network represent complicated patterns

The Mathematical Equations
Main equations

These use row-vector convention, so h_i is a row of H.

Normalised adjacency
Â=D^(-1/2) (A+I) D^(-1/2)

Here D is the diagonal degree matrix of A + I.
GCN layer (matrix form)
H^((l+1))=σ(Â H^((l)) W^((l)))

GCN layer (single node i)
h_i^((l+1))=σ(∑_(j∈N(i)∪{i}) 1/√(d_i d_j) h_j^((l)) W^((l)))

General message-passing form
m_i^((l))=AGG({h_j^((l)):j∈N(i)})
h_i^((l+1))=σ(h_i^((l)) W_self^((l))+m_i^((l)) W_neigh^((l))+b^((l)))

Here AGG can be sum, mean or max. This form covers GraphSAGE-style networks.

Output (node classification example)
Ŷ=softmax(H^((L)))


Implementation in C++
How to implement it in C++

1. Store the features. Keep the feature matrix as a single std::vector<float> of size N×F in row-major order, so node i's features start at position i*F.
2. Read the data from file into it.
3. Store the graph. Store the connections as a list of neighbours for each node, for example a std::vector<std::vector<int>> where entry i holds the neighbours of node i.
    Add node i to its own list as the self-loop. Real graphs are sparse, so this is far smaller than an N×N matrix.
5. Compute the degrees. Make a std::vector<float> of size N. Set each entry to the size of that node's neighbour list.
6. Do this once before running any layer.
7. Store the weights. For each layer, store W as a std::vector<float> of size F_in×F_out and b as a std::vector<float> of size F_out.
8.  Fill them from a trained weights file, or with small random values for testing.
9. Write the aggregation function. Create an output array of size N×F_in filled with zeros.
10.  For each node i, loop over its neighbours j. Compute the coefficient 1/sqrt(deg[i]*deg[j]) using std::sqrt. Then, for each feature f, add coefficient × H[j][f] to the output at node i, feature f.
11. Write the transform function. Create an output array of size N×F_out. For each node i and each output feature k, start from b[k].
12. Then loop over the input features f and add aggregated[i][f] × W[f][k].
13. This is a standard matrix multiplication written as three nested loops.
14. Write the activation function. Loop over every element of the result and replace it with std::max(0.0f, value).
15. Combine into one layer function. The function takes the graph, the current features, W and b, and returns the new features.
16. Inside, it calls aggregation, then transform, then activation.
17. Write the forward pass. Call the layer function L times in a loop, passing each layer's output as the next layer's input features.
18. The graph and degrees stay the same in every call. Only W and b change.
19. Apply softmax (classification only). For each node, subtract the maximum value in its output row (for numerical stability), apply std::exp to each entry, then divide each entry by the row sum.
21. Test it. Use a tiny graph of 3 or 4 nodes with hand-calculable features and weights.
22. Compute one layer by hand using the equations above and check that the C++ output matches.
23. We can also use library functions intead of modules where possible.
