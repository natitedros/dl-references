# Assignment 2 — Reference Guide

This document records everything that was implemented for Assignment 2, explains what each notebook teaches, and lists the code that goes into the notebooks themselves (which are run in Google Colab).

---

## Table of Contents

1. [How the assignment is organized](#1-how-the-assignment-is-organized)
2. [File map — what was changed](#2-file-map--what-was-changed)
3. [Running in Colab (setup + Cython)](#3-running-in-colab-setup--cython)
4. [FullyConnectedNets.ipynb](#4-fullyconnectednetsipynb)
5. [BatchNormalization.ipynb](#5-batchnormalizationipynb)
6. [Dropout.ipynb](#6-dropoutipynb)
7. [ConvolutionalNetworks.ipynb](#7-convolutionalnetworksipynb)
8. [PyTorch.ipynb](#8-pytorchipynb)
9. [Still to do by hand](#9-still-to-do-by-hand)

---

## 1. How the assignment is organized

The assignment has two halves:

- **Notebooks 1–4 (numpy).** You build a small deep learning library from scratch in the `cisc6000/` package: layers (forward + backward), optimizers, and models. The notebooks only *test and use* that library — most of the real work lives in `.py` files.
- **Notebook 5 (PyTorch).** You leave the hand-written library behind and rebuild the same ideas with PyTorch, where autograd computes all backward passes for you.

Every layer in the numpy library follows the same contract:

```python
out, cache = layer_forward(x, params...)   # cache = whatever backward needs
dx, dparams... = layer_backward(dout, cache)
```

`dout` is the upstream gradient ∂L/∂out; backward returns ∂L/∂(each input). Chaining these is backpropagation. Every notebook checks analytic gradients against **numerical gradients** (finite differences, `gradient_check.py`) and reports a *relative error* — tiny numbers (≈1e-7 or less) mean the backward pass is correct.

---

## 2. File map — what was changed

| File | Status | What it contains |
|---|---|---|
| `cisc6000/layers.py` | **Implemented in this session** (normalization, dropout, conv, pool sections) | All layer forward/backward functions |
| `cisc6000/classifiers/cnn.py` | **Implemented in this session** | `ThreeLayerConvNet` |
| `cisc6000/setup.py` | **Fixed in this session** | Cython build script (`distutils` → `setuptools`) |
| `cisc6000/classifiers/fc_net.py` | Already complete | `FullyConnectedNet` (supports batchnorm / layernorm / dropout) |
| `cisc6000/optim.py` | Already complete | SGD, SGD+momentum, RMSProp, Adam |
| `cisc6000/layer_utils.py` | Provided / unchanged | "Sandwich" layers (affine-relu, conv-relu-pool, …) |
| `cisc6000/fast_layers.py`, `im2col*.py/.pyx` | Provided / unchanged | Fast conv & pool (need Cython compile) |
| `cisc6000/solver.py` | Provided / unchanged | Training loop (`Solver`) |
| `pytorch_cifar.py` | Not used | Old standalone script; no notebook imports it |

Functions implemented in `layers.py` this session:

| Function | Used by notebook |
|---|---|
| `batchnorm_forward`, `batchnorm_backward`, `batchnorm_backward_alt` | BatchNormalization |
| `layernorm_forward`, `layernorm_backward` | BatchNormalization |
| `dropout_forward`, `dropout_backward` | Dropout |
| `conv_forward_naive`, `conv_backward_naive` | ConvolutionalNetworks |
| `max_pool_forward_naive`, `max_pool_backward_naive` | ConvolutionalNetworks |
| `spatial_batchnorm_forward`, `spatial_batchnorm_backward` | ConvolutionalNetworks |
| `spatial_groupnorm_forward`, `spatial_groupnorm_backward` | ConvolutionalNetworks |

All of these were verified locally with the notebooks' own test cells (see each section for results).

---

## 3. Running in Colab (setup + Cython)

**First cell of every notebook** — set your Drive path:

```python
FOLDERNAME = 'dl-assignments/assignment2/'
```

**Cython extension (ConvolutionalNetworks only).** `conv_backward_fast` calls `col2im_6d_cython`, which exists only after compiling `im2col_cython.pyx`.

- *The bug:* the original `setup.py` imported `distutils`, which was removed in Python 3.12 (Colab's version). The compile cell failed quietly (`!` shell cells don't turn red), so the extension was never built and you got `NameError: name 'col2im_6d_cython' is not defined`. Note `conv_forward_fast` still worked because it doesn't use Cython — which is why only the backward call failed.
- *The fix:* `setup.py` now uses `from setuptools import setup, Extension`.
- *Correct order (it matters):*
  1. Run the first cell, then **run the compile cell** (`python setup.py build_ext --inplace`). Confirm an `im2col_cython...so` file appears in `cisc6000/`.
  2. **Restart** the runtime — modules imported before the build are cached, so the new extension is only picked up after a restart.
  3. Run the notebook from the top (you can skip the compile cell from now on; the `.so` stays in Drive).

Diagnostic cell if it still fails:

```python
import os
print([f for f in os.listdir('cisc6000') if 'im2col_cython' in f])
from cisc6000.im2col_cython import col2im_6d_cython
print("Cython extension OK")
```

---

## 4. FullyConnectedNets.ipynb

**What it teaches:** building an arbitrarily deep fully connected network out of modular layers, and comparing optimizers.

**Architecture** (`FullyConnectedNet` in `fc_net.py`):

```
{affine - [batch/layer norm] - relu - [dropout]} x (L-1) - affine - softmax
```

**Notebook flow:**
1. Gradient-check the multi-layer net; overfit tiny datasets with 3- and 5-layer nets (deeper nets are much more sensitive to `weight_scale` and learning rate). Values to use (untested starting points, adjust if the loss doesn't fall to ~0):
   - Three-layer (`[100, 100]`): `weight_scale = 1e-2`, `learning_rate = 1e-2`
   - Five-layer (`[100, 100, 100, 100]`): `weight_scale = 1e-1`, `learning_rate = 2e-3`. At `1e-5` the activations shrink toward zero through 4 hidden layers and nothing learns, so the five-layer net needs a much larger init.
2. **SGD vs SGD+momentum.** Momentum keeps a velocity `v = μv − lr·dw; w += v`, which smooths noisy gradients and speeds progress along consistent directions.
3. **RMSProp and Adam.** Per-parameter adaptive learning rates:
   - RMSProp: `cache = ρ·cache + (1−ρ)·dw²;  w −= lr·dw / (√cache + ε)`
   - Adam: momentum (`m`) + RMSProp (`v`) + *bias correction* `m̂ = m/(1−β₁ᵗ)`, `v̂ = v/(1−β₂ᵗ)` (since `m`, `v` start at zero they're biased low early on).
4. **Train a good model** (≥ 50% validation accuracy).

**Final TODO cell** (`best_model`) — uses batchnorm + dropout now that those are implemented:

```python
best_val = -1
for lr in [1e-3, 5e-4]:
    for keep in [0.75, 0.9]:
        model = FullyConnectedNet(
            [512, 256, 128],
            normalization='batchnorm',
            dropout_keep_ratio=keep,
            reg=1e-3,
            weight_scale=5e-2,
        )
        solver = Solver(
            model,
            data,
            num_epochs=10,
            batch_size=200,
            update_rule='adam',
            optim_config={'learning_rate': lr},
            lr_decay=0.9,
            print_every=1000,
            verbose=False,
        )
        solver.train()
        print(f"lr {lr:.0e}  keep {keep}  ->  best val acc {solver.best_val_acc:.4f}")
        if solver.best_val_acc > best_val:
            best_val = solver.best_val_acc
            best_model = model

print("Best validation accuracy:", best_val)
```

Why these choices: batchnorm makes the deep net insensitive to initialization and lets Adam use a larger learning rate; dropout + small L2 reduce overfitting (the net has ~1.7M parameters vs 49k training images); `lr_decay=0.9` shrinks the step each epoch. `Solver.train()` restores the best-validation weights at the end, so `best_model` holds its best checkpoint.

---

## 5. BatchNormalization.ipynb

**What it teaches:** normalizing activations inside the network to make deep nets train faster and more robustly; and layer normalization as a batch-size-independent alternative.

### Batch normalization — forward

For a minibatch `x` of shape (N, D), per feature (column):

```
μ = mean(x, axis=0)            σ² = var(x, axis=0)
x̂ = (x − μ) / √(σ² + ε)
out = γ · x̂ + β                 (γ, β learnable, shape (D,))
```

- **Train mode** uses minibatch statistics and updates running averages:
  `running_mean = momentum·running_mean + (1−momentum)·μ` (same for var).
- **Test mode** uses the running averages instead (a single test image has no "batch").
- γ, β let the network undo the normalization if that's optimal, so no representational power is lost.

### Batch normalization — backward

Two versions are implemented:

1. **`batchnorm_backward` (computation graph).** Backprop node by node: `out → x̂ → (x−μ) and 1/σ → σ² → μ → x`, summing gradients where a value branches (e.g. `x − μ` feeds both `x̂` and `σ²`).
2. **`batchnorm_backward_alt` (simplified on paper).** Collapsing the graph gives one line:

   ```
   dx̂ = dout · γ
   dx  = (N·dx̂ − Σdx̂ − x̂·Σ(dx̂·x̂)) / (N·σ)
   ```

   Same result (diff ≈ 1e-12), faster. `fc_net.py` uses this one.

Also: `dβ = Σ dout`, `dγ = Σ dout·x̂` (sums over the batch axis).

### Layer normalization

Identical math, but statistics are taken **per data point across its features** (axis=1) instead of per feature across the batch. Consequences: no running averages, train = test, works with any batch size (even 1). The backward pass is the simplified batchnorm formula with sums over axis 1 and `D` instead of `N`.

### Experiments in the notebook
- **Deep net with vs without BN:** BN converges much faster.
- **Weight-init scale sweep:** without BN, accuracy collapses unless the init scale is just right (activations vanish or explode through layers); with BN, a wide range of scales works because every layer re-normalizes.
- **Batch size:** BN degrades with small batches (noisy μ, σ estimates; mismatch with test-time running stats). Layer norm is much less sensitive.

**Verified:** BN train-mode output means/stds = β/γ exactly; dx error 1.7e-9; alt vs graph diff 1.4e-12; layernorm dx error 1.4e-9; full-net gradient checks within the notebook's expected ranges for both normalizations.

---

## 6. Dropout.ipynb

**What it teaches:** dropout as a regularizer — randomly zeroing activations during training so the network can't rely on any single unit.

### Inverted dropout (`p` = probability of **keeping** a unit)

```python
# train
mask = (np.random.rand(*x.shape) < p) / p
out  = x * mask
# test
out  = x
# backward
dx = dout * mask
```

Dividing by `p` at train time keeps the expected activation unchanged (`E[out] = x`), so **test time needs no scaling** — that's the "inverted" part. Without the `/p`, test-time activations would be ~1/p times larger than what later layers saw during training.

`FullyConnectedNet` inserts dropout after every ReLU when `dropout_keep_ratio != 1`.

**Experiment:** two nets on 500 training images, keep=1 vs keep=0.25. Without dropout, training accuracy shoots up while validation lags (overfitting); with dropout the train/val gap is smaller.

**Verified:** fraction of zeros ≈ 1−p; output mean ≈ input mean (≈10) for p = 0.25/0.4/0.7; dx error 5.4e-11; full-net checks ≤ ~1e-6.

---

## 7. ConvolutionalNetworks.ipynb

**What it teaches:** the layers that make CNNs work, then a small CNN, then normalization variants for conv feature maps.

### Convolution — naive forward/backward

Input `x (N, C, H, W)`, filters `w (F, C, HH, WW)`, bias `b (F,)`:

```
H' = 1 + (H + 2·pad − HH) / stride        W' likewise
out[n, f, i, j] = Σ_{c,u,v} x_pad[n, c, i·s+u, j·s+v] · w[f, c, u, v] + b[f]
```

Implementation pads with `np.pad`, loops over output positions `(i, j)`, and at each position does one matrix multiply between all windows and all filters. Backward, at each position:
- `dw += doutᵀ · window` (each filter's gradient accumulates over every place it was applied)
- `dx_pad[window] += dout · w` (each input pixel gets gradient from every window that covered it)
- `db = Σ dout` over N, H', W'; finally strip the padding from `dx_pad`.

### Max pooling — naive forward/backward

Forward takes the max of each `pool_height × pool_width` window. Backward routes each upstream gradient **only to the element that was the max** (a mask of `window == max`); all others get zero.

### Fast layers & sandwich layers
`fast_layers.py` (im2col + Cython) gives the same outputs much faster; `layer_utils.py` chains them into `conv_relu_forward`, `conv_relu_pool_forward`, etc.

### ThreeLayerConvNet (`cnn.py`)

```
conv(7x7, 32 filters, "same" padding) - relu - 2x2 maxpool - affine(hidden) - relu - affine - softmax
```

- Init: `W1 (F, C, k, k)`, `W2 (F·H/2·W/2, hidden)` (pooling halves each spatial dim), `W3 (hidden, classes)`; biases zero.
- Forward: `conv_relu_pool_forward → affine_relu_forward → affine_forward`.
- Loss: softmax + `0.5·reg·Σ‖W‖²`; backward mirrors the forward with the matching `_backward` functions and adds `reg·W` to each weight gradient.

Sanity checks: initial loss ≈ log(10) = 2.30 (random guessing over 10 classes); goes up slightly with regularization; can overfit 100 images.

### Spatial batch normalization
For conv features, one channel = one filter, so statistics are shared **per channel over N, H, W**. Implementation: transpose to `(N, H, W, C)`, reshape to `(N·H·W, C)`, call ordinary `batchnorm_forward`/`batchnorm_backward_alt`, reshape back.

### Spatial group normalization
Layer norm over a whole conv feature map works poorly (border vs center units have very different statistics). Group norm splits the C channels into G groups and normalizes **per data point, per group**. Implementation: reshape to `(N·G, C/G·H·W)`, apply the layer-norm forward/backward formulas row-wise, reshape back; γ, β of shape `(1, C, 1, 1)` are per channel. No dependence on batch size.

**Verified:** conv forward 2.2e-8, conv backward ≤ 8e-9, pool forward 4.2e-8, pool backward 3.3e-12, spatial BN dx 2.8e-7, group norm dx 7.4e-8; CNN initial loss 2.3026 / 2.508 (reg 0.5) and all CNN gradients ≤ 1.8e-2 (the notebook allows ~1e-2).

---

## 8. PyTorch.ipynb

**What it teaches:** the same models at three levels of abstraction in PyTorch, with autograd doing every backward pass and GPUs doing the compute.

| Level | You manage | Example |
|---|---|---|
| Barebones tensors | every weight tensor, the forward function, the SGD update | `F.conv2d(x, w, b, padding=2)` |
| `nn.Module` | layers declared in `__init__`, connectivity in `forward` | `self.conv1 = nn.Conv2d(...)` |
| `nn.Sequential` | just the list of layers | `nn.Sequential(nn.Conv2d(...), nn.ReLU(), ...)` |

No `.py` files are involved — every TODO is in the notebook. The cells below go between the `START/END OF YOUR CODE` markers.

**Barebones `three_layer_convnet`:**
```python
    x = F.relu(F.conv2d(x, conv_w1, conv_b1, padding=2))
    x = F.relu(F.conv2d(x, conv_w2, conv_b2, padding=1))
    scores = flatten(x).mm(fc_w) + fc_b
```

**Barebones parameter init** (padding keeps 32×32, so FC input = `channel_2·32·32`):
```python
conv_w1 = random_weight((channel_1, 3, 5, 5))
conv_b1 = zero_weight((channel_1,))
conv_w2 = random_weight((channel_2, channel_1, 3, 3))
conv_b2 = zero_weight((channel_2,))
fc_w = random_weight((channel_2 * 32 * 32, 10))
fc_b = zero_weight((10,))
```

**Module API `ThreeLayerConvNet.__init__`:**
```python
        self.conv1 = nn.Conv2d(in_channel, channel_1, kernel_size=5, padding=2)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.fc = nn.Linear(channel_2 * 32 * 32, num_classes)
        nn.init.kaiming_normal_(self.fc.weight)
```

**Module API `forward`:**
```python
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        scores = self.fc(flatten(x))
```

**Module API training cell:**
```python
model = ThreeLayerConvNet(in_channel=3, channel_1=channel_1, channel_2=channel_2, num_classes=10)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
```

**Sequential API three-layer ConvNet:**
```python
model = nn.Sequential(
    nn.Conv2d(3, channel_1, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1),
    nn.ReLU(),
    Flatten(),
    nn.Linear(channel_2 * 32 * 32, 10),
)
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=0.9, nesterov=True)
```

**Part V — open-ended challenge (target ≥ 70% val in 10 epochs):**
```python
def conv_bn_relu(c_in, c_out):
    # Bias is redundant right before BatchNorm
    return [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)]

model = nn.Sequential(
    *conv_bn_relu(3, 64),    *conv_bn_relu(64, 64),   nn.MaxPool2d(2), nn.Dropout(0.1),  # 32 -> 16
    *conv_bn_relu(64, 128),  *conv_bn_relu(128, 128), nn.MaxPool2d(2), nn.Dropout(0.2),  # 16 -> 8
    *conv_bn_relu(128, 256), *conv_bn_relu(256, 256), nn.MaxPool2d(2), nn.Dropout(0.3),  # 8 -> 4
    nn.AdaptiveAvgPool2d(1),   # global average pooling
    Flatten(),
    nn.Linear(256, 10),
)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
```

Design rationale: stacked 3×3 convs (two 3×3 = one 5×5 receptive field with fewer parameters and an extra nonlinearity); BatchNorm2d after each conv for fast, stable training; max pooling halves resolution while channels double; increasing dropout regularizes deeper stages; global average pooling replaces a huge FC layer (~1.1M parameters total); Adam + weight decay. Expect roughly 80%+ validation accuracy in 10 epochs on a GPU runtime (not verified locally — no GPU/CIFAR here). The final `best_model = model` cell must run right after this training.

**Verified locally:** all snippets produce `(64, 10)` / correct shapes and gradients flow to every barebones parameter.

---

## 9. Still to do by hand

These are written answers / results, not code:

| Notebook | Item |
|---|---|
| FullyConnectedNets | Inline Questions 1–2 (Q2: AdaGrad's `cache` only grows, so steps shrink toward zero; Adam uses a decaying average so it doesn't) |
| BatchNormalization | Inline Questions 1–4 |
| Dropout | Inline Questions 1–2 |
| PyTorch | "Describe what you did" — summarize the Part V design above plus your actual val/test accuracy |
| All | Run every notebook in Colab (with CIFAR-10 and, for CNNs, the compiled Cython extension) to produce the training results and plots |
