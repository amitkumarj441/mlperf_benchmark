## Mathematics behind Transformer

Here are some additional details on how the proportional limit enables relating discrete covariance updates to the continuous neural SDE,

* In a proportional limit, the number of layers $\(L\)$ and width $\(n\)$ satisfy $\(L/n \rightarrow d/w\)$ for some constant ratio $\(d/w\)$ as $\(n \rightarrow \infty\)$.
* This constant ratio allows defining a continuous time $\(t = l/n \in [0, T]\)$ where $\(T = d/w\)$ and $\(l\)$ is the layer index.
* As $\(n \rightarrow \infty\)$, there are infinitesimally small gaps $\(1/n\)$ between discrete layers $\(l\)$ and $\(l+1\)$ in this pseudo-continuous time.
* The attention operation gives covariance updates of size $\(O(1/n)\)$.
* This means the discrete covariance updates $\(V_{l+1} - V_l\)$ are small $O(1/n)$ for large $\(n\)$.
* These small updates can be embedded into the continuous time $\(t \in [0, T]\)$ as $\(n \rightarrow \infty\)$.
* The evolution of $\(V_l\)$ in discrete layers can be approximated by discretizing the SDE $\(dV_t\)$ using step size $1/n$.
* This gives updates like $\(V_{l+1} \approx V_l + \text{(SDE drift and diffusion terms)}\)$ that converge to the continuous SDE.
* So the constant $\(d/w\)$ ratio provides a continuous notion of time to embed discrete steps into.
* The $\(O(1/n)\)$ update size allows approximating the SDE via these discrete updates.

Together this connects the discrete covariance evolution to the solution of the continuous SDE.

In summary, the key is that the temperature scaling leads to $O(1/n)$ terms in the Taylor expansion of the centered softmax attention matrix \(A\). This provides the small update sizes needed to relate the discrete updates to the continuous SDE.

### Taylor Expansion of Softmax

The following is the mathematical derivation for the Taylor expansion approximation for the centered softmax attention matrix $\(A\)$:

We start with the definition of $\(A\)$,

$\[ A = I + \text{Softmax}(\tau^{-1}Y) - \frac{1}{m}11^\top \]$, where 
$\(\text{Softmax}(Z)_{ij} = \frac{\exp(Z_{ij})}{\sum_{k} \exp(Z_{ik})}\)$.

The softmax function is used in machine learning to convert a vector of arbitrary values to a probability distribution. It is defined as follows:

For a vector $\(z\)$ of length $\(n\)$, the softmax of its $\(i\)$-th component is defined as,
$\text{softmax}(z_i) = \frac{\exp(z_i)}{\sum_{j=1}^{n} \exp(z_j)}$. The partial derivative of the softmax function with respect to its inputs can be a bit tricky because of the summation in the denominator. The derivative will be different depending on whether we're taking the derivative with respect to the same input that we're applying the softmax to $i=j$ or a different input $(i \neq j)$.

1. **Case $\(i = j\)$:**
$$\[ \frac{\partial \text{softmax}(z_i)}{\partial z_i} = \text{softmax}(z_i) \cdot (1 - \text{softmax}(z_i)) \]$$
2. **Case $\(i \neq j\)$:**
$$\[ \frac{\partial \text{softmax}(z_i)}{\partial z_j} = - \text{softmax}(z_i) \cdot \text{softmax}(z_j) \]$$

So, the derivative of the softmax function with respect to its inputs can be compactly represented as follows:
$$\[ \frac{\partial \text{softmax}(z_i)}{\partial z_j} = \text{softmax}(z_i) \cdot (\delta_{ij} - \text{softmax}(z_j)) \]$$
where $\(\delta_{ij}\)$ is the Kronecker delta, which is 1 when $\(i=j\)$ and 0 when $\(i \neq j\)$.

The second derivative, $\(\frac{\partial^2 \text{softmax}(z)}{\partial z \partial z}\)$, can be derived similarly.

In matrix form:
$$\[ \frac{\partial^2 \text{softmax}(z)}{\partial z \partial z} = \text{diag}(\text{softmax}(z)) - \text{softmax}(z) \cdot \text{softmax}(z)^\top \]$$
To get the second derivative, we take the derivative of the above:
$$\[ 
\frac{\partial^2 \text{softmax}(z)}{\partial z^2} = \text{diag}(\text{diag}(\text{softmax}(z)) - \text{softmax}(z) \cdot \text{softmax}(z)^\top) - (\text{diag}(\text{softmax}(z)) - \text{softmax}(z) \cdot \text{softmax}(z)^\top) \cdot \text{softmax}(z)^\top - \text{softmax}(z) \cdot \left(\text{diag}(\text{softmax}(z)) - \text{softmax}(z) \cdot \text{softmax}(z)^\top\right)^\top 
\]$$
Evaluating at $\(z=0\)$, where,
$$\[
\text{diag}(\text{softmax}(0)) = \frac{1}{m}I \\
\text{softmax}(0) = \frac{1}{m}11^\top
\]$$

The first term:
$\[ \text{diag}(\text{diag}(\text{softmax}(0)) - \text{softmax}(0) \cdot \text{softmax}(0)^\top) = \text{diag}\left(\frac{1}{m}I - \frac{1}{m^2}11^\top\right) \]$
