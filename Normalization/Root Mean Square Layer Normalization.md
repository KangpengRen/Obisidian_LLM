---
发表年份: "2019"
注释: RMSNorm均方根归一化，证明了中心化并非必需，简化归一化范式
---
# 背景
`LayerNorm`中，给定神经元$a$：
$$
\begin{align}
\mu = \frac{1}{n} \sum a_i, \\
\sigma = \sqrt {\frac {1}{n} \sum (a_i - \mu)^2} \\
LayerNorm(a) = \frac{a - \mu}{\sigma} \cdot g + b
\end{align}
$$
完成：中心化（均值）和标准化（缩方差），其优点是稳定训练、收敛快，缺点是计算方差必须先算均值，开销大，尤其在RNN、大模型上明显拖慢。
这篇论文假设，`LayerNorm`能稳定训练，靠的是缩放不变形，而不是中心化，于是直接去掉均值，只用`RMS`归一化。
# `RMSNorm`
公式：
$$
\begin{align}
RMS(a) = \sqrt {\frac {1}{n} \sum_{i=1}^n a_i^2} \\
RMSNorm(a) = \frac {a}{RMS(a)} \cdot g + b
\end{align}
$$
与`LayerNorm`的区别：减少了均值$\mu$的计算，减少了一次逐元素减法。
满足缩放不变性：
$$
RMS(\alpha x) = \alpha RMS(x)
$$
无中心化不变性：输入平移会改变结果，但论文证明不影响效果。其梯度更加平滑，对权重缩放有负相关，相当于隐式学习率调整。

---
`pRMSNorm`：只对前$p\%$的元素估算`RMS`，论文推荐$p=6.25\%$ 就足够，理论上更快，但受框架切片实现影响，实际不一定更快，不变性和`RMSNorm`完全一致。
# 总结
证明了“中心化不是必需”，简化归一化范式。现在几乎所有大模型都在使用，`LLaMA`、`GPT`、`T5`等都把`LayerNorm`换成`RMSNorm`，几乎无精度损失，训练、推理更快，可以直接替换`LayerNorm`。