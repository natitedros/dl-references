# Assignment 1: A Guide to What's Going On

This guide explains the five notebooks you completed (`knn_`, `svm_`, `softmax_`, `two_layer_net_`, `features_`): what problem each one solves, the idea behind it, the math, and what the NumPy code is doing. It assumes you're comfortable with linear algebra and calculus but new to neural networks.

It **doesn't answer the inline questions**. Where a section covers the idea a question depends on, it says so (look for 📝).

---

## Contents

0. [The big picture: image classification](#0-the-big-picture-image-classification)
1. [The shared framework: data, shapes, splits, hyperparameters](#1-the-shared-framework)
2. [kNN: classify by looking at the neighbours](#2-knn-knn_ipynb)
3. [Linear classifiers: scores = Wx](#3-linear-classifiers-the-common-core)
4. [Multiclass SVM](#4-multiclass-svm-svm_ipynb)
5. [Softmax classifier](#5-softmax-classifier-softmax_ipynb)
6. [Optimisation: gradient descent, SGD, gradient checking](#6-optimisation-how-w-is-learned)
7. [Two-layer neural network and backpropagation](#7-two-layer-neural-network-two_layer_net_ipynb)
8. [Hand-crafted image features](#8-image-features-features_ipynb)
9. [NumPy cheat sheet (every trick used)](#9-numpy-cheat-sheet)
10. [Your results side by side](#10-your-results-side-by-side)
11. [Which section helps with which inline question](#11-inline-question-map)

---

## 0. The big picture: image classification

**The problem.** You're given a small colour image and must say which of 10 categories it belongs to:

```
plane, car, bird, cat, deer, dog, frog, horse, ship, truck
```

The dataset is **CIFAR-10**: 50,000 training images and 10,000 test images, each 32×32 pixels with 3 colour channels (R, G, B). Each pixel value is a number from 0 to 255.

To a computer, one image is just a block of $32 \times 32 \times 3 = 3072$ numbers. So mathematically the task is:

> Find a function $f: \mathbb{R}^{3072} \to \{0, 1, \dots, 9\}$ that gives the right label for images it has **never seen**.

This is **supervised learning**: we have example pairs $(x_i, y_i)$ (image, correct label) and want to learn $f$ from them.

**How the assignment builds up.** Each notebook is a slightly smarter version of $f$:

| Notebook | Approach | Core idea |
|---|---|---|
| `knn_` | k-Nearest Neighbours | "Find the training images that look most similar and copy their label." No learning. |
| `svm_` | Linear classifier + SVM loss | Learn a weight matrix $W$ so that $Wx$ gives a score per class. |
| `softmax_` | Linear classifier + softmax loss | Same model, but the scores are read as probabilities. |
| `two_layer_net_` | 2-layer neural network | Add a non-linear hidden layer between the input and the scores. |
| `features_` | Features + SVM / neural net | Don't feed in raw pixels. Feed in smarter summaries (edges, colours). |

Accuracy goes up at each step: roughly 28% → 38% → 53% → 60%. For reference, random guessing gets 10%.

---

## 1. The shared framework

### 1.1 Shapes: the letters used everywhere

| Symbol | Meaning | Typical value |
|---|---|---|
| $N$ | number of examples in a batch/dataset | 500, 49000 |
| $D$ | dimension of one input vector | 3072 (3073 with bias) |
| $C$ | number of classes | 10 |
| $H$ | number of hidden units (neural net) | 50–500 |

The conventions (row = one example):

- `X` has shape `(N, D)`: each **row** is one flattened image.
- `y` has shape `(N,)`: integers in `0..C-1`.
- `W` has shape `(D, C)`: each **column** belongs to one class.
- `scores = X @ W` has shape `(N, C)`: `scores[i, j]` = how much example `i` looks like class `j`.

**Tip:** when you're lost in a derivation, check the shapes. Very often only one arrangement of the matrices has shapes that fit, and that arrangement is the right answer.

### 1.2 Flattening images

```python
X_train = np.reshape(X_train, (X_train.shape[0], -1))   # (5000, 32, 32, 3) -> (5000, 3072)
```

`-1` means "work out this dimension yourself". Flattening throws away the 2-D layout: pixel (0,0) and pixel (0,1) become just coordinates 0 and 3 of a vector. None of the models in this assignment "know" that neighbouring pixels are related. (Convolutional networks, later in the course, fix that.)

### 1.3 Train / validation / test / dev splits

| Split | Size | Used for |
|---|---|---|
| **train** | 49,000 | fitting the parameters ($W$, etc.) |
| **validation** | 1,000 | choosing **hyperparameters** (learning rate, regularisation, $k$, hidden size…) |
| **test** | 1,000 | measured **once** at the very end to report honest performance |
| **dev** | 500 (random subset of train) | fast debugging of loss and gradient code |

Why a separate validation set? If you picked hyperparameters by looking at test accuracy, you'd be tuning to the test set and the reported number would be over-optimistic. The test set has to stay untouched until the end to measure **generalisation**.

**Cross-validation** (used in kNN): when data is scarce, split the training set into $F$ folds (you used $F=5$). For each fold, train on the other $F-1$ and validate on that one, then average the $F$ accuracies. The estimate is more reliable, but it costs $F$ times the compute.

### 1.4 Parameters vs. hyperparameters

- **Parameters** are learned by the algorithm (the entries of $W$, $b$).
- **Hyperparameters** are chosen by *you* before training ($k$, learning rate, regularisation strength, hidden size, number of epochs, batch size). The nested `for lr in ...: for reg in ...:` loops in every notebook are a **grid search** over hyperparameters, scored on the validation set.

### 1.5 Preprocessing: mean subtraction and the bias trick

In `svm_` and `softmax_`:

```python
mean_image = np.mean(X_train, axis=0)       # (3072,)  the "average image"
X_train -= mean_image                       # broadcast: subtract from every row
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])   # append a column of 1s
```

- **Mean subtraction** centres the data around 0. Every pixel is otherwise around +130, so all the inputs are large and positive, which makes gradient descent zig-zag and converge slowly. The mean is computed **only from training data** and then applied to val/test. Otherwise information from val/test would "leak" into training.
- **Bias trick**: a linear model is really $s = W^\top x + b$. If you append a constant 1 to every $x$ and add one extra row to $W$, then
  $$[W; b^\top]^\top [x; 1] = W^\top x + b,$$
  so you only have to handle one matrix. That's why $D$ becomes 3073.

---

## 2. kNN (`knn_.ipynb`)

### 2.1 Idea

- **"Training"**: just store all training images (`self.X_train = X`). Nothing is learned.
- **Prediction**: for a test image, compute its distance to *every* training image, take the $k$ closest ones, and have them **vote** on the label.

### 2.2 The distance: L2 (Euclidean)

$$d(x, z) = \|x - z\|_2 = \sqrt{\sum_{p=1}^{3072} (x_p - z_p)^2}$$

For $N_{te}=500$ test and $N_{tr}=5000$ training images you build a **distance matrix** `dists` of shape `(500, 5000)`, where `dists[i, j]` = distance from test $i$ to train $j$.

📝 *Inline Q1 (bright rows/columns)*: a row is one test image against all training images; a column is one training image against all test images. Consider what makes one image far away from *everything* in raw-pixel distance.

The **L1** (Manhattan) distance is $d_1(x,z)=\sum_p |x_p - z_p|$.
📝 *Inline Q2* asks which preprocessing steps leave nearest-neighbour **rankings** unchanged. The test is whether $|\tilde{x}_p - \tilde{z}_p|$ equals $|x_p - z_p|$, or at least a constant times it, for *every* pixel $p$.

### 2.3 Three implementations of the same matrix

This is really a lesson in **vectorisation**: making NumPy do the loops in optimised C instead of in Python.

**Two loops**: one Python iteration per (test, train) pair = 2.5 million iterations:

```python
dists[i, j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))
```

**One loop**: one test image against *all* training images at once, using **broadcasting**:

```python
dists[i, :] = np.sqrt(np.sum((self.X_train - X[i]) ** 2, axis=1))
#                            (5000,3072) - (3072,) -> (5000,3072); sum over axis=1 -> (5000,)
```

`X[i]` has shape `(3072,)`, and NumPy "stretches" it to subtract from each of the 5000 rows. `axis=1` sums across columns, giving one number per row.

> Your timing showed the one-loop version was *not* faster (47s vs 42s). Each iteration allocates a temporary `(5000, 3072)` array of float64 (~120 MB) and makes several passes over it, so memory traffic dominates. The notebook itself warns this can happen.

**No loops**: the clever part. Expand the square:

$$\|x - z\|^2 = \|x\|^2 - 2\,x\cdot z + \|z\|^2$$

Do this for all pairs at once. With $X$ (test, $500\times D$) and $Z$ (train, $5000\times D$):

$$\text{dists}^2 = \underbrace{\text{rowsums}(X^2)}_{(500,1)} \;-\; 2\underbrace{X Z^\top}_{(500,5000)} \;+\; \underbrace{\text{rowsums}(Z^2)}_{(5000,)}$$

```python
dists = np.sqrt((X**2).sum(axis=1, keepdims=True)          # (500, 1)
                - 2 * X @ self.X_train.T                     # (500, 5000)
                + (self.X_train**2).sum(axis=1))             # (5000,) broadcasts as a row
```

- `keepdims=True` keeps the result as a **column** `(500,1)` rather than a flat `(500,)` vector, so it broadcasts *down the columns*.
- The `(5000,)` vector broadcasts as a **row**, across the rows.
- All the heavy work is one matrix multiply, `X @ X_train.T`, which runs on highly optimised BLAS code. Result: **0.67 s vs 42 s**.

The Frobenius-norm check `np.linalg.norm(dists - dists_one, ord='fro')` is just $\sqrt{\sum_{ij}(A_{ij}-B_{ij})^2}$: one number measuring how different two matrices are.

### 2.4 Voting

```python
k_nearest_indices = np.argsort(dists[i])[:k]     # indices of the k smallest distances
closest_y = self.y_train[k_nearest_indices]       # their labels (fancy indexing)
y_pred[i] = np.bincount(closest_y).argmax()       # most common label
```

- `np.argsort` returns the **indices** that would sort the array, not the sorted values.
- `np.bincount([3,1,3,7])` → `[0,1,0,2,0,0,0,1]`: counts of each integer. `argmax` picks the most frequent, and on ties it returns the **first** (smallest) label, which is exactly the tie-breaking rule the assignment asks for.

### 2.5 Choosing $k$ by cross-validation

```python
X_train_folds = np.array_split(X_train, num_folds)                  # list of 5 arrays
X_tr_cv = np.concatenate(X_train_folds[:i] + X_train_folds[i+1:])    # glue the other 4 back together
```

(`list + list` is Python list concatenation, not arithmetic.) Your plot showed the accuracy vs. $k$ curve; you picked $k=10$ and got **28.2%** test accuracy.

### 2.6 Key properties of kNN (useful for 📝 *Inline Q3*)

- **Training cost** $O(1)$, **prediction cost** $O(N_{tr} \cdot D)$ per test image. This is the opposite of what you want in practice, where you'd rather pay once during training and predict fast.
- **Small $k$** follows every quirk of the training data (including noise), and the decision boundary is jagged. **Large $k$** averages more and gives a smoother boundary.
- Think about what happens when you classify a *training* point with $k=1$. What is its nearest neighbour?
- **Why it's weak on images**: pixel distance says little about meaning. A shifted, darkened, or recoloured version of the same cat can be far away in L2, while an unrelated image with a similar background can be close.

---

## 3. Linear classifiers: the common core

The SVM and softmax notebooks share one model and differ only in the **loss function**.

### 3.1 The score function

$$s = f(x; W) = W^\top x \quad\in \mathbb{R}^{C}, \qquad \text{or for a whole batch:}\quad S = XW \in \mathbb{R}^{N\times C}$$

Predict the class with the highest score:

```python
scores = X.dot(self.W)                  # (N, C)
y_pred = np.argmax(scores, axis=1)      # (N,)  index of max in each row
```

### 3.2 Two ways to picture it

1. **Template matching.** Column $w_c$ of $W$ is a 3072-vector, i.e. an image. The score $w_c \cdot x$ is a dot product, which is large when $x$ "lines up" with $w_c$. So each class learns **one template image**, and classification asks "which template does this image match best?" That's why the notebooks can reshape `W[:-1, i]` back to `32×32×3` and display it.
   📝 *SVM Inline Q2* and the softmax weight visualisation: with only one template per class, what does a template have to look like to match *all* horses, facing left and right, on different backgrounds?
2. **Geometry.** Each class score $w_c \cdot x = 0$ defines a hyperplane in $\mathbb{R}^{3072}$. The classifier cuts space into regions with flat (linear) boundaries.

### 3.3 Loss function and regularisation

We need a number that says how bad $W$ is on the training data. Then learning = minimising that number:

$$L(W) = \underbrace{\frac{1}{N}\sum_{i=1}^N L_i(W)}_{\text{data loss}} \;+\; \underbrace{\lambda \sum_{d,c} W_{dc}^2}_{\text{regularisation}}$$

- $L_i$ measures how badly example $i$ is classified (SVM and softmax define it differently).
- **Regularisation** ($\lambda$ = `reg`) penalises large weights. It prefers *spread out, small* weights over a few huge ones, which usually **generalises** better. Without it there are infinitely many equally good $W$: if $W$ classifies everything perfectly with margin, so does $2W$.
- Its gradient is $\frac{\partial}{\partial W} \lambda \|W\|^2 = 2\lambda W$. That's the `dW += 2 * reg * W` line.

---

## 4. Multiclass SVM (`svm_.ipynb`)

### 4.1 The hinge loss

For example $i$ with correct class $y_i$ and scores $s = W^\top x_i$:

$$L_i = \sum_{j \ne y_i} \max\bigl(0,\; s_j - s_{y_i} + \Delta\bigr), \qquad \Delta = 1$$

In words: the correct class score should beat every wrong class score by at least a **margin** $\Delta$. Each wrong class $j$ that comes within the margin (or beats the correct class) adds a penalty equal to how badly it violated the margin. Wrong classes that are safely below contribute **zero**.

$\max(0, \cdot)$ is the **hinge**: flat at zero, then a straight line. It has a **kink** (not differentiable) exactly at 0.
📝 *SVM Inline Q1* is about this kink: what happens to a numerical derivative $\frac{f(x+h)-f(x-h)}{2h}$ when $x$ sits within $h$ of a kink?

**Sanity check at initialisation**: with tiny random $W$ all scores ≈ 0, so each of the 9 wrong classes contributes $\max(0, 0 - 0 + 1) = 1$ and $L \approx 9$. Your notebook printed 9.07. ✓

### 4.2 Deriving the gradient

Write $s_j = w_j \cdot x_i$ (with $w_j$ = column $j$ of $W$). For one term with $j \ne y_i$, if the margin is positive ($s_j - s_{y_i} + 1 > 0$):

$$\frac{\partial}{\partial w_j} = x_i, \qquad \frac{\partial}{\partial w_{y_i}} = -x_i$$

and if the margin ≤ 0, both are 0. Summing over all $j$:

- **Wrong class $j$**: $\nabla_{w_j} L_i = \mathbb{1}[\text{margin}_j > 0]\; x_i$ (push its score *down*)
- **Correct class**: $\nabla_{w_{y_i}} L_i = -\Big(\sum_{j\ne y_i} \mathbb{1}[\text{margin}_j > 0]\Big)\, x_i$ (push its score *up*, once per violator)

This is exactly what the naive loop does:

```python
if margin > 0:
    loss += margin
    dW[:, j]    += X[i]
    dW[:, y[i]] -= X[i]
```

### 4.3 Vectorised version

Build everything as `(N, C)` matrices:

```python
rows = np.arange(N)
scores = X.dot(W)                                        # (N, C)
correct = scores[rows, y][:, np.newaxis]                  # (N, 1)  correct-class score per row
margins = np.maximum(0, scores - correct + 1)             # (N, C)  broadcast subtraction
margins[rows, y] = 0                                      # don't count j == y_i
loss = margins.sum() / N + reg * np.sum(W * W)
```

For the gradient, build a **coefficient matrix** $M$ (`mask`) where $M_{ij}$ says "how many times to add $x_i$ into column $j$":

```python
mask = (margins > 0).astype(float)          # 1 where a wrong class violates the margin
mask[rows, y] = -mask.sum(axis=1)           # correct class gets minus the number of violators
dW = X.T.dot(mask) / N + 2 * reg * W        # (D, N) @ (N, C) -> (D, C)
```

Why does `X.T @ mask` work? Look at one entry:
$$(X^\top M)_{dj} = \sum_i X_{id} M_{ij},$$
i.e. column $j$ of $dW$ = $\sum_i M_{ij}\, x_i$, a weighted sum of input rows. That's exactly the loop's "add $x_i$ into column $j$ with coefficient $M_{ij}$". **This `X.T @ (something N×C)` pattern shows up again in softmax and in the neural net.**

Key NumPy trick: **`scores[rows, y]`** is *integer-array (fancy) indexing*. It picks `scores[0, y[0]], scores[1, y[1]], …`: one element per row. `[:, np.newaxis]` turns shape `(N,)` into `(N,1)` so it broadcasts across columns.

---

## 5. Softmax classifier (`softmax_.ipynb`)

### 5.1 From scores to probabilities

Same scores $s = W^\top x$, but interpret them as unnormalised **log-probabilities**:

$$p_k = \frac{e^{s_k}}{\sum_{j=1}^{C} e^{s_j}} \qquad (\text{softmax})$$

The $p_k$ are positive and sum to 1, so they form a probability distribution over classes.

### 5.2 Cross-entropy loss

Penalise a low probability on the correct class:

$$L_i = -\log p_{y_i} = -s_{y_i} + \log\sum_j e^{s_j}$$

- If the model is confident and right ($p_{y_i}\to 1$), $L_i \to 0$.
- If it gives the right class probability near 0, $L_i \to \infty$.

Information-theory view: this is the **cross-entropy** between the "true" distribution (all mass on $y_i$) and the predicted $p$. It's also the **negative log-likelihood**, so minimising it is maximum-likelihood estimation.

📝 *Softmax Inline Q1* (why the loss starts near $-\log(0.1)$): work out what $p$ looks like when all scores are approximately equal.

### 5.3 Numerical stability

$e^{s}$ overflows for large $s$ (e^1000 = inf). But softmax doesn't change if you shift all scores by a constant $m$:

$$\frac{e^{s_k - m}}{\sum_j e^{s_j - m}} = \frac{e^{-m}e^{s_k}}{e^{-m}\sum_j e^{s_j}} = p_k$$

So subtract the row max first (largest exponent becomes $e^0=1$):

```python
scores -= np.max(scores, axis=1, keepdims=True)
```

### 5.4 The gradient: a very clean result

Differentiate $L_i = -s_{y_i} + \log\sum_j e^{s_j}$ with respect to score $s_k$:

$$\frac{\partial L_i}{\partial s_k} = -\mathbb{1}[k = y_i] + \frac{e^{s_k}}{\sum_j e^{s_j}} = p_k - \mathbb{1}[k=y_i]$$

"Predicted probability minus the truth." Then because $s_k = w_k \cdot x_i$, $\partial s_k/\partial w_k = x_i$ (chain rule):

$$\nabla_{w_k} L_i = (p_k - \mathbb{1}[k = y_i])\, x_i$$

Vectorised, with $P$ the `(N,C)` probability matrix and $Y$ the one-hot label matrix:

$$\nabla_W L = \frac{1}{N} X^\top (P - Y) + 2\lambda W$$

```python
dscores = probs.copy()
dscores[rows, y] -= 1                 # P - Y without building Y explicitly
dW = X.T.dot(dscores) / N + 2 * reg * W
```

The same `X.T @ (N×C matrix)` structure as the SVM. The only difference is *what goes in the coefficient matrix*: SVM uses 0/1/−count, softmax uses probabilities minus 1.

### 5.5 SVM vs. softmax: the key difference

- **SVM** is satisfied once the correct class wins by margin Δ. After that the loss is exactly 0 and the example stops influencing $W$.
- **Softmax** is never fully satisfied: $-\log p_{y_i} > 0$ unless $p_{y_i}=1$ exactly, which needs infinite score gaps. It keeps pushing, a little.

📝 This difference is what *Softmax Inline Q2* is about.

In practice they reach about the same accuracy (38.1% vs 38.0% in your runs).

---

## 6. Optimisation: how $W$ is learned

### 6.1 Gradient descent

The gradient $\nabla_W L$ points in the direction where $L$ increases fastest. So step the other way:

$$W \leftarrow W - \eta\, \nabla_W L$$

$\eta$ is the **learning rate** (`learning_rate`). Too small: painfully slow (the loss falls in a straight line). Too large: overshoots and diverges (the loss blows up, you get NaN/overflow warnings).

### 6.2 Stochastic (mini-batch) gradient descent: SGD

Computing $L$ over all 49,000 examples for every step is expensive. Instead, estimate the gradient from a random **mini-batch** of 200:

```python
batch_indices = np.random.choice(num_train, batch_size, replace=True)
X_batch, y_batch = X[batch_indices], y[batch_indices]
loss, grad = self.loss(X_batch, y_batch, reg)
self.W -= learning_rate * grad
```

The mini-batch gradient is an unbiased but noisy estimate of the full gradient. You take many more cheap steps, and that works far better in practice. The noise is why your loss curve is jittery rather than smooth.

- **Iteration** = one mini-batch update.
- **Epoch** = enough iterations to see ~all training data once = $49000 / 200 = 245$ iterations. That's why 10 epochs = 2450 iterations in the two-layer net output.

**How learning rate and regularisation interact.** Look at just the regularisation part of the update:
$$W \leftarrow W - \eta \cdot 2\lambda W = (1 - 2\eta\lambda)\,W.$$
Every step multiplies $W$ by $(1 - 2\eta\lambda)$, which is "weight decay". If $2\eta\lambda$ gets near or above 1, the weights oscillate or explode (the source of the overflow warnings the notebooks mention). At the other extreme, when $\eta$ is tiny the model barely moves from its random start in 1500 iterations. That's why, in `features_`, the runs with `lr=1e-9, reg=5e4` stayed at ~10%, as good as random guessing. The product $\eta\lambda$ matters as much as either value alone.

### 6.3 Gradient checking

Deriving gradients by hand is error-prone, so we verify them numerically with the **centred difference**:

$$\frac{\partial f}{\partial w} \approx \frac{f(w + h) - f(w - h)}{2h}, \quad h \approx 10^{-5}$$

(The centred version has error $O(h^2)$, better than the one-sided $O(h)$.) The `gradient_check.py` helpers nudge one entry at a time, re-run the loss, and compare. They report **relative error**:

$$\text{rel\_error} = \frac{|a - n|}{\max(10^{-8}, |a| + |n|)}$$

Relative error is used rather than absolute because gradients can be huge or tiny. Rough guide: < 1e-7 great, ~1e-4 suspicious, > 1e-2 almost certainly a bug. (A kink in the function can cause occasional legitimate mismatches. See SVM 📝 above.)

`eval_numerical_gradient_array(f, x, dout)` is the version for layers that output arrays. It numerically computes $\sum \frac{\partial\,\text{out}}{\partial x} \cdot \text{dout}$, which is exactly what a backward function should return (see §7.3).

---

## 7. Two-layer neural network (`two_layer_net_.ipynb`)

### 7.1 Why go beyond linear?

A linear classifier gives each class one template and linear decision boundaries. Stacking linear layers doesn't help: $W_2^\top(W_1^\top x) = (W_1W_2)^\top x$ is still linear. You need a **non-linearity** in between.

### 7.2 The architecture

```
x ──► affine (W1, b1) ──► ReLU ──► affine (W2, b2) ──► scores ──► softmax loss
(N,D)        (N,H)           (N,H)          (N,C)
```

$$h = \max(0,\; xW_1 + b_1), \qquad s = hW_2 + b_2$$

- **Affine layer** (also called fully connected or dense): $\text{out} = xW + b$. Note that the bias trick isn't used here; $b$ is a separate vector.
- **ReLU** (Rectified Linear Unit): $\text{ReLU}(z) = \max(0, z)$, element-wise.

Intuition: the first layer learns $H$ templates (your $H$ = 200 best). ReLU keeps only the positive matches, i.e. "is feature $k$ present?". The second layer combines these detections, so "horse" can be "left-facing-horse template **or** right-facing-horse template". That's something one linear template can't express.

**Weight initialisation** (`fc_net.py`):

```python
self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)   # small Gaussian, std 1e-3
self.params['b1'] = np.zeros(hidden_dim)
```

Weights have to be random. If they started identical, every hidden unit would compute the same thing and receive the same gradient, so they would stay identical forever ("symmetry").

**Regularisation convention changes here**: the net uses $\frac{1}{2}\lambda\|W\|^2$, so the gradient is $\lambda W$ (no factor 2). Same idea, different constant.

### 7.3 Backpropagation: the chain rule, organised

You need $\partial L/\partial W_1$, but $L$ depends on $W_1$ only through a chain of operations. **Backprop** applies the chain rule one step at a time, going backward through the chain.

The modular design:

- `layer_forward(inputs) -> (out, cache)`: compute the output and **save** whatever the backward pass will need.
- `layer_backward(dout, cache) -> dinputs`: given $\text{dout} = \partial L/\partial\,\text{out}$ (the "upstream gradient"), return $\partial L/\partial\,\text{input}$ for each input.

Each layer only needs to know **its own local derivative**. Chaining them together gives the full gradient. So any architecture can be built from reusable blocks.

#### Affine backward

Forward: $\text{out} = XW + b$ with $X:(N,D)$, $W:(D,M)$, $b:(M,)$, $\text{out}:(N,M)$.

Entry-wise: $\text{out}_{nm} = \sum_d X_{nd}W_{dm} + b_m$. So:

- $\dfrac{\partial L}{\partial W_{dm}} = \sum_n \dfrac{\partial L}{\partial \text{out}_{nm}} X_{nd}$ ⟹ $\;dW = X^\top \, \text{dout}$  (D,N)@(N,M) = (D,M) ✓
- $\dfrac{\partial L}{\partial X_{nd}} = \sum_m \dfrac{\partial L}{\partial \text{out}_{nm}} W_{dm}$ ⟹ $\;dX = \text{dout}\, W^\top$  (N,M)@(M,D) = (N,D) ✓
- $\dfrac{\partial L}{\partial b_m} = \sum_n \dfrac{\partial L}{\partial \text{out}_{nm}}$ ⟹ $\;db = $ column-sums of dout  (M,) ✓

Notice that the bias was *broadcast* to all $N$ rows in the forward pass, so its gradient is *summed* over rows in the backward pass. That's a general rule: **broadcast forward ⇔ sum backward**.

```python
x_flat = x.reshape(N, -1)
dx = dout.dot(w.T).reshape(x.shape)    # reshape back to whatever shape x came in with
dw = x_flat.T.dot(dout)
db = np.sum(dout, axis=0)
```

(The two-layer net's data is shaped `(N, 3, 32, 32)`. `data_utils.get_CIFAR10_data` transposes channels first. The affine layer flattens it internally, which is why `x` has to be reshaped.)

#### ReLU backward

$\frac{d}{dz}\max(0,z) = 1$ if $z>0$, else $0$. So the gradient passes through where the input was positive and is **blocked** where it was negative:

```python
dx = dout * (x > 0)       # boolean mask auto-converts to 0/1
```

📝 *Two-layer Inline Q1* is about exactly this: when is the local derivative of an activation (sigmoid $\sigma(z)=1/(1+e^{-z})$, ReLU, leaky ReLU $\max(0.01z, z)$) zero or nearly zero? Sketch each function and its derivative for large positive, large negative, and near-zero $z$.

#### The loss layers

`svm_loss(x, y)` and `softmax_loss(x, y)` in `layers.py` are the same math as §4 and §5, but written for **scores** as input and returning `dx = ∂L/∂scores` (no $W$ inside). The softmax version computes log-probabilities directly (`shifted - log(sum(exp(shifted)))`), which is even more numerically stable.

### 7.4 The full forward and backward pass (`TwoLayerNet.loss`)

```python
# forward
hidden, cache1 = affine_relu_forward(X, W1, b1)     # (N,D) -> (N,H)
scores, cache2 = affine_forward(hidden, W2, b2)     # (N,H) -> (N,C)
data_loss, dscores = softmax_loss(scores, y)        # dscores = (P - Y)/N
loss = data_loss + 0.5 * reg * (sum(W1²) + sum(W2²))

# backward, in reverse order, feeding each dout into the previous layer
dhidden, dW2, db2 = affine_backward(dscores, cache2)
dX, dW1, db1      = affine_relu_backward(dhidden, cache1)
grads['W1'] = dW1 + reg * W1                        # + regularisation gradient
grads['W2'] = dW2 + reg * W2
```

When `y is None` (test time), it just returns `scores`. That's why prediction is `np.argmax(model.loss(X), axis=1)`.

`affine_relu_forward/backward` in `layer_utils.py` is a "sandwich layer": it calls the two in sequence and bundles both caches.

### 7.5 The Solver

`solver.py` wraps the training loop so the model only needs to provide `loss(X, y) -> (loss, grads)` and a `params` dict. Each `_step`:

1. sample a mini-batch,
2. call `model.loss`,
3. for each parameter, apply the update rule (`optim.sgd`: `w -= lr * dw`).

At the end of every epoch it checks train/val accuracy, keeps the **best parameters seen** (by val accuracy), and multiplies the learning rate by `lr_decay` (0.95). The **learning-rate decay** takes big steps early and finer steps later.

### 7.6 Reading the training plots (debugging)

- **Loss falling almost linearly, not flattening** → learning rate probably too low.
- **Train and val accuracy nearly identical** → model is **underfitting** (not enough capacity). Make it bigger or train longer.
- **Train accuracy ≫ val accuracy** → **overfitting**: the model memorises training data.

📝 *Two-layer Inline Q2* asks how to shrink a train/test gap. For each option, ask whether it makes memorising the training set harder or easier.

Your tuning moved from the default (H=50, ~36–44%) to H=200, lr=1e-3, reg=1e-2 → **52.4% val / 52.9% test**.

---

## 8. Image features (`features_.ipynb`)

### 8.1 Idea

Instead of making the classifier smarter, make the **input** smarter. Replace 3072 raw pixel values with a hand-designed summary that captures what matters (shapes and colours) and ignores what doesn't (small shifts, exact brightness).

### 8.2 HOG: Histogram of Oriented Gradients (shape and texture)

1. Convert to greyscale.
2. Compute the image **gradient** at every pixel using differences of neighbouring pixels (`np.diff`):
   $g_x = I(r, c+1) - I(r,c)$, $g_y = I(r+1,c) - I(r,c)$.
   Magnitude $\sqrt{g_x^2+g_y^2}$ = edge strength; angle $\arctan(g_y/g_x)$ = edge direction.
3. Split the 32×32 image into 8×8-pixel **cells** → a 4×4 grid of cells.
4. In each cell, build a **histogram over 9 orientation bins** (0–180°), each pixel voting with its edge strength.

Result: $4\times4\times9 = 144$ numbers saying "in this region of the image, edges mostly point this way". Because it pools over each cell, a small shift of the object barely changes the features. It ignores colour.

### 8.3 Colour histogram (colour, no shape)

Convert RGB → HSV and take the **hue** channel ("which colour", separate from brightness/saturation). Build a 10-bin histogram over the whole image: "how much of the image is red-ish, green-ish, blue-ish…". It ignores *where* colours are.

Final feature vector = concatenation: $144 + 10 = 154$ (plus 1 bias = 155, matching your printout).

### 8.4 Standardisation

```python
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)   # (1, 154)
std_feat  = np.std(X_train_feats,  axis=0, keepdims=True)
X_train_feats = (X_train_feats - mean_feat) / std_feat
```

Each feature gets mean 0 and standard deviation 1. HOG magnitudes and histogram fractions live on very different scales, and without this the large-scale features would dominate the dot products and the gradient steps.

### 8.5 Results

- SVM on features: **41.6%** test (vs 38.1% on raw pixels). A better input helps even a linear model.
- Two-layer net on features: **59.7%** test (vs 52.9% on raw pixels). Best of all: good features plus a non-linear model.

(You removed the bias column before the neural net because `TwoLayerNet` has its own $b_1$, $b_2$.)

📝 *Features Inline Q1* (misclassifications): recall that the features only know *edge orientations per region* and the *overall hue distribution*. Which classes would look alike through that lens? Think about backgrounds (sky, water, grass) and shapes.

**Historical note:** this "hand-engineered features + classifier" pipeline was state of the art before ~2012. Deep networks (CNNs) replaced it by *learning* the features from pixels. The first layer of a trained CNN typically ends up looking like edge and colour detectors, much like HOG and colour histograms.

---

## 9. NumPy cheat sheet

Every non-obvious NumPy idiom in the assignment:

| Code | What it does | Example |
|---|---|---|
| `A @ B`, `A.dot(B)` | Matrix multiply | `(N,D)@(D,C)→(N,C)` |
| `A.T` | Transpose | `(N,D)→(D,N)` |
| `x.reshape(N, -1)` | Change shape, `-1` = infer | `(N,3,32,32)→(N,3072)` |
| `A.sum(axis=0)` | Sum **down** columns (collapse rows) | `(N,C)→(C,)` |
| `A.sum(axis=1)` | Sum **across** each row (collapse columns) | `(N,C)→(N,)` |
| `keepdims=True` | Keep the summed axis as size 1 | `(N,C)→(N,1)` (ready to broadcast) |
| **Broadcasting** | Arrays with size-1 or missing dims are virtually stretched to match | `(N,C) - (N,1)`; `(N,M) + (M,)` |
| `x[:, np.newaxis]` | Add a size-1 axis | `(N,)→(N,1)` |
| `A[rows, y]` with `rows=np.arange(N)` | **Fancy indexing**: pick one element per row | correct-class scores `(N,)` |
| `A[rows, y] = 0` / `-= 1` | Write to those elements | zero out correct-class margins |
| `np.maximum(0, A)` | Element-wise max with 0 (≠ `np.max`, which reduces) | hinge, ReLU |
| `(A > 0)` | Boolean array; in arithmetic True→1, False→0 | ReLU backward mask |
| `.astype(float)` | Convert dtype | booleans → 0.0/1.0 |
| `np.argmax(A, axis=1)` | Index of the max in each row | predicted class |
| `np.argsort(v)` | Indices that sort `v` ascending | k nearest neighbours |
| `np.bincount(v)` | Count occurrences of each non-negative int | kNN voting |
| `np.random.choice(n, k, replace=True)` | k random indices from `0..n-1` | mini-batch sampling |
| `np.array_split(A, k)` | Split into k (nearly) equal chunks | CV folds |
| `np.concatenate(list)` | Glue arrays along axis 0 | re-join CV folds |
| `np.hstack([A, ones])` | Stack column-wise | bias trick |
| `np.linalg.norm(A, ord='fro')` | $\sqrt{\sum A_{ij}^2}$ | compare two matrices |
| `A -= m` | In-place; also modifies the original array | mean subtraction |
| `.copy()` | Real copy (slices/views share memory) | `dscores = probs.copy()` |

**Broadcasting rule:** compare shapes from the **right**. Two dimensions are compatible if they're equal or one of them is 1 (a missing leading dimension counts as 1). The size-1 dimension is stretched. So:

- `(500, 1)` + `(5000,)` → treat as `(500,1)` + `(1,5000)` → `(500, 5000)`
- `(N, C)` − `(N, 1)` → subtract the column from every column
- `(N, C)` − `(N,)` → **error or bug**: `(N,)` is read as `(1, N)`. This is why `keepdims`/`newaxis` matter.

**Why vectorise at all?** A Python `for` loop pays interpreter overhead on every iteration. A NumPy operation runs one tight compiled loop (and for `@`, a multi-threaded BLAS routine using SIMD instructions). You saw the difference directly: **42 s → 0.67 s**.

---

## 10. Your results side by side

| Model | Input | Test accuracy |
|---|---|---|
| Random guess | – | 10% |
| kNN, k=10 | raw pixels | 28.2% |
| Linear SVM | raw pixels | 38.1% |
| Softmax | raw pixels | 38.0% |
| Linear SVM | HOG + colour hist | 41.6% |
| Two-layer net (H=200) | raw pixels | 52.9% |
| Two-layer net (H=500) | HOG + colour hist | 59.7% |

What the table shows:
1. **Learning beats memorising** (kNN → linear): one template per class, fitted by optimisation, beats pixel lookup and predicts much faster.
2. **The loss choice matters little here** (SVM ≈ softmax).
3. **Non-linearity matters a lot** (linear → 2-layer): +15 points.
4. **Input representation matters** (pixels → features): better for both model types.

---

## 11. Inline question map

| Notebook | Question | Read |
|---|---|---|
| knn_ | Q1 bright rows/cols | §2.2, §1.5 (raw pixel values) |
| knn_ | Q2 preprocessing & L1 | §2.2 |
| knn_ | Q3 true statements about kNN | §2.6, §3.2 (what "linear boundary" means) |
| svm_ | Q1 gradcheck mismatch | §4.1 (kink), §6.3 |
| svm_ | Q2 weight visualisation | §3.2 (template matching) |
| softmax_ | Q1 initial loss ≈ −log(0.1) | §5.1–5.2, compare with §4.1's "≈ 9" sanity check |
| softmax_ | Q2 adding a point, SVM vs softmax loss | §4.1, §5.5 |
| two_layer_net_ | Q1 activations with ~zero gradient | §7.3 (ReLU backward) |
| two_layer_net_ | Q2 closing the train/test gap | §3.3 (regularisation), §7.6 |
| features_ | Q1 misclassifications | §8.2–8.3 |

(Your knn_ notebook already has answers for its three questions; the others are still open.)
