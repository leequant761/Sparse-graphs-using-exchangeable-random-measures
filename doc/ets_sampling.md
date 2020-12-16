If $\sigma \geq 0, \quad W_{\alpha}^{*}$ is an exponentially tilted stable random variable so that there exist exact samplers.

[Jim Pitman, 5.3](https://projecteuclid.org/download/pdf_1/euclid.lnms/1215091133) said 

​	if Levy density has the form $\rho_{\alpha}(x)=\frac{\alpha x^{-\alpha-1}}{\Gamma(1-\alpha)} \quad(x>0)$, then total length $T=:S_\alpha$  follows $\operatorname{stable}(\alpha)$ with pdf $f(t)$

​	Furthermore, from Levy khintchine formula, we can get a Laplace transform as
$$
\mathbb{E}\left\{\mathrm{e}^{-t T}\right\}=

\exp \left\{ \int_{(0, \infty]}\left(1-\mathrm{e}^{-t x}\right) \rho_\alpha(dx) \right\}


= \exp \left(-t^{\alpha}\right)
$$
[Jim Pitman, 4.2](https://projecteuclid.org/download/pdf_1/euclid.lnms/1215091133) said

​	If we set $\rho_{\alpha, b}(x)=\rho_\alpha(x) e^{-b x}$ as a Levy density, then $T=: S_{\alpha, b}$'s pdf $f^{(b)}(t)=f(t) e^{\psi(b)-b t}$ where $\psi(b)=\int_{0}^{\infty}\left(1-e^{-b x}\right) \rho_\alpha(x) d x = \exp \left(-b^{\alpha}\right)$

​	Therefore, using a simple trick, we can get a Laplace transform as
$$
\mathbb{E}\left\{e^{-t S_{\alpha, \lambda}}\right\}=\mathbb{E}_{\alpha}\left\{e^{-\lambda^{\alpha}-(t+\lambda) S_{\alpha}}\right\}=e^{-\lambda^{\alpha}-(t+\lambda)^{\alpha}}
$$

---

[Caron, 5.3](https://arxiv.org/pdf/1401.1137v3.pdf) considers Levy measure
$$
\begin{aligned}

\rho(d w) \lambda ([0, \alpha]) &= \frac{\alpha}{\Gamma(1-\sigma)} w^{-1-\sigma} \exp (-\tau w) d w
\\

& = \frac{\alpha}{\sigma} \times \frac{\sigma}{\Gamma(1-\sigma)} w^{-1-\sigma} \exp (-\tau w) d w

\\

&= \frac{\alpha}{\sigma}\rho_{\sigma, \tau}(dw)

\end{aligned}
$$
Then, from Levy khintchine formula,
$$
\begin{aligned}

\mathbb{E}\left\{\mathrm{e}^{-t W_{\alpha}^{*}}\right\} &=

\exp \left\{ \int_{(0, \infty]}\left(1-\mathrm{e}^{-t x}\right) \rho(d x) \lambda ([0, \alpha]) \right\}

\\\\

&= \exp \left\{ \frac{\alpha}{\sigma} \times \int_{(0, \infty]} \left(1-\mathrm{e}^{-t x}\right) \rho_{\sigma, \tau}(dx)  \right\}

\\\\

&=  \exp \left\{  \frac{\alpha}{\sigma} \times \left (-\tau^{\sigma}-(t + \tau)^{\sigma} \right ) \right\}

\\\\

&=: \exp \left\{  M \times \left (-\tau^{\sigma}-(t+\tau)^{\sigma} \right ) \right\}

\\\\

&=  \exp \left\{ \left ( -\left(M^{\frac{1}{\sigma}}\tau \right)^{\sigma}- \left(M^{\frac{1}{\sigma}} t+ M^{\frac{1}{\sigma}}\tau \right)^{\sigma} \right ) \right\}

\\\\

&= \mathbb E \left[ \exp \left( - t M^{\frac{1}{\sigma}} S_{\sigma, M ^{1 /\sigma} \tau} \right) \right] 
\end{aligned}
$$
Hence, we can sample $W_{\alpha}^{*}$ as

1. Sample $ T= S_{\sigma, M ^{1/\sigma} \tau}$
2. Multiply $T = T \times M ^{\frac{1}{\sigma}}$
