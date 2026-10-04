# Week 1 Contribution

**Member:** Tendo
**Project:** Machine Learning 3 — Graph Neural Network from Scratch
**Group:** 12
**Week:** 1
**Task:** Evaluation and documentation

---

## Why we need evaluation

Our GNN will predict the class of each node. After training, we need to check if the predictions are good. We should check using test nodes that the model did not see during training, so the result is fair.

## Example I will use

Say we test the model on 10 nodes, and we are checking class A.

- 5 nodes are really class A.
- The model said "A" for 4 nodes, and only 3 of them were really A.

From this:

- **TP** (True Positive) = 3. The model said A and it was A.
- **FP** (False Positive) = 1. The model said A but it was not A.
- **FN** (False Negative) = 2. It was A but the model missed it.
- **TN** (True Negative) = 4. The model said not A and it was not A.

## Accuracy

How many predictions were correct overall.

**Formula:** Accuracy = (TP + TN) / total

**Example:** (3 + 4) / 10 = 0.70, so 70%.

## Precision

When the model says "A", how often is it right?

**Formula:** Precision = TP / (TP + FP)

**Example:** 3 / (3 + 1) = 0.75

## Recall

Out of all the real A nodes, how many did the model find?

**Formula:** Recall = TP / (TP + FN)

**Example:** 3 / (3 + 2) = 0.60

## F1-score

One number that combines precision and recall. It is only high when both are high.

**Formula:** F1 = 2 × (Precision × Recall) / (Precision + Recall)

**Example:** 2 × (0.75 × 0.60) / (0.75 + 0.60) = 0.67

Our data has more than two classes, so we calculate these for each class and then take the average.

## Training loss

The loss is a number that shows how wrong the model is while it is learning. A big loss means bad predictions. It should go down as training continues. If it does not go down, something is wrong. For classification, we normally use cross-entropy loss. Nesta explains how the loss is calculated.

## Training time

This is how long the training takes, in seconds. In C++ we can measure it with the `<chrono>` library. We note the time before training and after training, then subtract.

```cpp
# include <chrono>
auto start = std::chrono::steady_clock::now();
// training happens here
auto end = std::chrono::steady_clock::now();
std::chrono::duration<double> elapsed = end - start;
std::cout << "Training time: " << elapsed.count() << " s\n";
```

## What the final C++ program should display

```
Dataset: <name> | Nodes: <N> | Classes: <C>

Epoch 1   | Loss: 1.10 | Accuracy: 0.33
Epoch 100 | Loss: 0.35 | Accuracy: 0.90
Epoch 200 | Loss: 0.20 | Accuracy: 0.96

Training time: 4.8 s

Test accuracy: 0.80
Class   Precision   Recall   F1
A       0.75        0.60     0.67
B       ...         ...      ...
Average ...         ...      ...
```

(These numbers are just an example of the layout. The real ones will come from our model.)

## Helping with the weekly documentation

To keep our report neat, I suggest that:

- Everyone uses the same header at the top of their file.
- Everyone adds their sources at the end.
- Arnold and Sharon use the same example graph.
- We all use the same letters, for example A for adjacency matrix.
- I keep a simple checklist of who has submitted their file and opened a Pull Request, so Shafiq can see what is missing.

## Sources

1. scikit-learn documentation, Model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
2. Google Machine Learning Crash Course, Classification: https://developers.google.com/machine-learning/crash-course
3. C++ reference, chrono library: https://en.cppreference.com/w/cpp/chrono
