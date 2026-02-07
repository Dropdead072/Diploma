# Адаптивные методы первого порядка

##  AdaGrad (Adaptive Gradient)

$$
\mathbb{E} [g_k] = \nabla f(x_k) \\
H_k = [ \sum_{i=0}^k \text{diag}(g_i)^2 ]^{\frac{-1}{2}} \\
x_{k+1} = x_k - \eta H_k g_k
$$

Член $H_k$ называется оператором предобуславливания. 

## RMSProp

## Adam

$$

$$