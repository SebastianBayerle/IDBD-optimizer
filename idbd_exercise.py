# Exercise: implement classic LINEAR IDBD as a torch.optim.Optimizer.
# Scope: one observation, one scalar prediction, no bias or momentum.
# Each scalar weight has its own learning rate (not just one per tensor).
# Use loss = 0.5 * (prediction - target)^2 for both SGD and IDBD.
# Not NetworkIDBD or AutoStep; do not add normalization or rate clipping yet.

# INITIALIZATION
# Accept parameters, initial_lr > 0, and meta_lr >= 0.
# Call the base Optimizer initializer to register parameters and defaults.
# For each weight tensor, store same-shaped tensors in self.state[weight]:
#     beta  <- log(initial_lr), filled across every weight
#     h     <- zeros
#     alpha <- exp(beta)  # Can also be computed on demand.
# The model's weights themselves are initialized separately, to zero.

# CLASSIC PSEUDOCODE (all vector operations are elementwise)
# For each incoming feature vector x and noisy target y:
#     prediction <- dot(weight, x)  # BEFORE any update.
#     delta      <- y - prediction
#     beta       <- beta + meta_lr * delta * x * h
#     alpha      <- exp(beta)       # Use the NEW beta immediately.
#     weight     <- weight + alpha * delta * x
#     h          <- h * max(0, 1 - alpha * x^2) + alpha * delta * x
# Save beta and h for the next observation; never reset them each step.
# Keep delta fixed during this entire update; do not recompute it afterward.
# The max clips the decay factor, NOT h: h may be negative.

# TRANSLATE THIS INTO A PYTORCH step(...) METHOD
# The training loop calls loss.backward() BEFORE optimizer.step(...).
# Pass the current x into step, or pass curvature = x^2 explicitly.
# Match its shape to the weight tensor: [1, n_features] for Linear(n, 1).
# Run optimizer updates without autograd tracking (e.g. torch.no_grad).
# Retrieve grad = weight.grad; skip parameters whose grad is None.
# For half-squared error: grad = -delta*x, curvature = x^2.
# Thus the same update, using gradients already computed by autograd, is:
#     beta   <- beta - meta_lr * grad * h
#     alpha  <- exp(beta)
#     weight <- weight - alpha * grad
#     h      <- h * max(0, 1 - alpha * curvature) - alpha * grad
# Mutate the original weight tensor in place; persist state between steps.
# If using unhalved squared error instead, curvature must be 2*x^2.

# SMALL CHECKS BEFORE THE FULL EXPERIMENT
# With h initially zero, the first weight update equals SGD at initial_lr.
# With meta_lr = 0, ALL weight updates match fixed-rate SGD on the same loss.
# With x = 0, that weight's gradient is zero and its state stays unchanged.
# Consistently aligned updates tend to raise rates; opposing updates lower them.
# Hand-calculate two updates for one weight and compare weight, beta, and h.
