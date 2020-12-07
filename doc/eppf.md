[Caron, 3.2](https://arxiv.org/pdf/1401.1137v3.pdf) samples directed graph using urn-process
$$
\begin{array}{l}
1 . W_{\alpha}^{*} \sim P_{W_{\alpha}^{*}} \\
2 . D_{\alpha}^{*} \mid W_{\alpha}^{*} \sim \mathrm{Poisson}\left(W_{\alpha}^{* 2}\right) \\
3 .\left(U_{k j}\right)_{k=1, \ldots, D_{\alpha}^{*} ; j=1,2} \mid W_{\alpha}^{*} \sim \text { Urn process } \\
4 . D_{\alpha}=\sum_{k=1}^{D_{\alpha}^{*}} \delta_{\left(U_{k 1}, U_{k 2}\right)}
\end{array}
$$
[Caron, 5.3](https://arxiv.org/pdf/1401.1137v3.pdf) defines EPPF of $\operatorname{PK}(\rho \mid t)$ as
$$
\Pi_{k}^{(n)}\left(m_{1}, \ldots, m_{k} \mid t\right)=\frac{\sigma^{k} t^{-n}}{\Gamma(n-k \sigma) g_{\sigma}(t)} \int_{0}^{t} s^{n-k \sigma-1} g_{\sigma}(t-s) d s\left(\prod_{i=1}^{k} \frac{\Gamma\left(m_{i}-\sigma\right)}{\Gamma(1-\sigma)}\right)
$$
where $g_\sigma $ is the pdf of the positive stable distribution.

In the urn-process,
$$
\begin{aligned}

\frac{\Pi_{n+1}^{(k+1)}\left(m_{1}, \ldots, m_{k}, 1 \mid W_{\alpha}^{*}\right)}{\Pi_{n}^{(k)}\left(m_{1}, \ldots, m_{k} \mid W_{\alpha}^{*}\right)} &=  \frac{\sigma t^{-1} \gamma(n-k\sigma)}{\Gamma(n+1-(k+1)\sigma)} \frac{\int_{0}^{t} s^{n-(k+1) \sigma} g_{\sigma}(t-s) d s}{\int_{0}^{t} s^{n-k \sigma-1} g_{\sigma}(t-s) d s}

\\

&= \frac{\sigma t^{-1} \gamma(n-k\sigma)}{\Gamma(n+1-(k+1)\sigma)} \frac{\mathbb E \left[(t-s)^{n-(k+1)\sigma} \mathbf 1(0\leq s \leq t) \right]}{\mathbb E \left[(t-s)^{n-k\sigma-1} \mathbf 1(0\leq s \leq t) \right]}
\end{aligned}
$$
However, the MC approximation of the value is not converged because $(t-s)^{n-k\sigma-1}$ becomes large for large urn samples.