# Week 1 Contribution
**Member:** MULONDO JOSHUA JRDAN 25/U/07948/PS
**Project:** Machine Learning 3 — Graph Neural Network from Scratch
**Group:** 12
**Week:** 1
**Task:** Dataset investigation

## 1. Objective
Find 2-3 node-classification datasets suitable for our from-scratch C++ GNN implementation before deadline 5 November 2026. Record nodes, edges, features, classes, format and recommend one.

## 2. Dataset Candidates

### Candidate 1: Cora Citation Network
**Source / Location:** https://linqs.soe.ucsc.edu/data and https://relational.fit.cvut.cz/dataset/CORA
Original paper: Sen et al., 2008. Also available on PyTorch Geometric as Planetoid/Cora

**Description:** Machine learning papers classified into 7 research areas. Each node is a paper, edge represents a citation relationship; for our GNN implementation, the graph can be represented as undirected. This is the standard benchmark for GNN node classification.

**Statistics:**
- Nodes: 2,708
- Edges: 5,429 citation links in the dataset. If represented as an undirected graph using a symmetric adjacency matrix, each edge contributes two off-diagonal adjacency entries.
- Node Features: 1,433 (binary bag-of-words, 0/1 whether word appears)
- Classes: 7 (Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory)
- Labels: One label per node, only 140 nodes used for training in original semi-supervised split, 500 validation, 1000 test
- File Format: Content file: `<paper_id> <word_features> <label>` + Cites file: `<paper_id> <cited_paper_id>` . Tab-separated . Can be converted to CSV easily for C++ parsing. Also available as .npz

**Practicality for C++ from scratch:**
Highly practical. N=2708 is small enough to store full adjacency matrix 2708x2708 (~7.3M floats ~29 MB) or adjacency list (much smaller) in C++. Feature matrix 2708x1433 is ~3.8M entries, can be stored sparse because binary. Training time in C++ will be seconds per epoch. This is ideal for our 5 Nov deadline. Well-documented preprocessing steps.

### Candidate 2: Zachary's Karate Club
**Source / Location:** https://networkx.org/documentation/stable/reference/readwrite/edgelist.html and SNAP: https://snap.stanford.edu/data/ or Zachary 1977
Built-in in NetworkX and available as karate.edgelist

**Description:** Small social network of 34 members of a karate club, split after conflict. Classic toy dataset.

**Statistics:**
- Nodes: 34
- Edges: 78 (undirected)
- Node Features: No inherent features - we must create features e.g., identity matrix or degree-based features. F = 34 if using one-hot, or 1-2 handcrafted.
- Classes: 2 (Mr. Hi's club vs John A's club) or 4 if using modularity
- Labels: 34 labels
- File Format: Edge list .txt: `src dst` per line. Very simple to parse in C++ with ifstream.

**Practicality for C++ from scratch:**
Extremely practical for debugging and testing our GNN implementation. Small enough to fit comfortably in memory for development and unit testing. Can be used for initial unit tests before moving to Cora. However too small to demonstrate real GNN learning - accuracy will be 100% quickly and not convincing for final report. Best suited for unit testing and debugging rather than the primary training experiment.

### Candidate 3: CiteSeer Citation Network
**Source / Location:** https://linqs.soe.ucsc.edu/data and Planetoid
Paper: Giles et al.

**Description:** Similar to Cora, computer science papers.

**Statistics:**
- Nodes: 3,327
- Edges: 4,732
- Node Features: 3,703 binary bag-of-words
- Classes: 6 (Agents, AI, DB, IR, ML, HCI)
- Labels: 6 classes
- File Format: Same format as Cora: content + cites files

**Practicality for C++ from scratch:**
Practical but slightly harder than Cora. Feature dimension 3703 is larger than Cora's 1433, so weight matrix W will be bigger (3703 x hidden_dim). This increases memory and compute ~2.5x vs Cora. Still feasible before Nov 5, but Cora is more efficient. Also has more isolated nodes which makes training slightly unstable for from-scratch implementation.

## 3. Comparison Table

| Dataset | Nodes | Edges | Features | Classes | Format Complexity | C++ Memory | Best For |
|---|---|---|---|---|---|---|---|
| **Cora** | 2,708 | 5,429 | 1,433 | 7 | Easy (2 files) | Main training |
| **Karate** | 34 | 78 | 0 (need to make) | 2 | Very Easy | Very Low | Unit testing |
| **CiteSeer** | 3,327 | 4,732 | 3,703 | 6 | Easy | Alternative |

## 4. Preliminary Recommendation

**Based on the dataset size, avaliable node features file format, and suitability for a from-scratch C++ implementation, Cora appears to be a strong candidate for the primary dataset. The final dataset selection will be made by the group after discussion.**

**Reasons:**
1. Size: 2,708 nodes is small enough for from-scratch C++ without GPU before 5 November.
2. No large dataset upload needed: Original files are ~2MB total, fits GitHub limit (PDF rule: do not upload large dataset yet - we can just link).
3. Features included: Unlike Karate, Cora already has meaningful node features, so we don't need feature engineering.
4. Standard benchmark: Most GNN from-scratch tutorials use Cora, so we can compare accuracy (Expected Performance: Cora is a widely used benchmark dataset for node-classification tasks in Graph Neural Networks. Its established use in GNN research makes it suitable for evaluating our from-scratch C++ implementation).
5. File format: Simple tab-separated, easy to parse with `std::ifstream`, `std::stringstream` in C++.

Implementation plan for C++: 
- data/cora/ folder (gitignored for large files, keep only sample)
- Parse `cora.content` to map paper_id -> index, build X matrix vector<vector<float>>
- Parse `cora.cites` to build adjacency list vector<vector<int>>
- Build degree matrix and normalized adjacency later by Sharon's task

## 5. Do Not Upload Yet
As per instruction, I have NOT uploaded dataset files. Only documented links and candidates. Group will agree on Sunday.

## References / Sources
1. Sen, P. et al. (2008). Collective Classification in Network Data. AI Magazine.
2. https://linqs.soe.ucsc.edu/data - LINQS Cora and CiteSeer repository
3. https://snap.stanford.edu/data/ - SNAP Stanford datasets
4. Kipf & Welling (2017) uses Cora for GCN evaluation - https://arxiv.org/abs/1609.02907
5. Zachary, W. W. (1977). An information flow model for conflict and fission in small groups. Karate Club.