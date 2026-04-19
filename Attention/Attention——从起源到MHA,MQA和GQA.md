---
注释: 注意力综述
---
<a herf="https://zhuanlan.zhihu.com/p/686149289">理解Attention:从起源到MHA,MQA和GQA</a>
# `Transformer`中的`attention`
> 彻底抛弃了`RNN`在time step上的迭代计算，完全拥抱了`attention`机制，只用最简单粗暴的方式同步计算每个输入的`hidden state`，其他的交给`attention`解决

`Attention`的一般形式写作：$Attention(Q, K, V) = Score(Q, K)V$，其中$Q = W_Q Y, K = W_K X, V = W_V X$：
- 对于`cross-attention`，$X$是`encoder`的`hidden states`，$Y$是`decoder`的`hidden states`；
- 对于`self-attention`，$X = Y$。
标准公式为：
$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt d})V
$$
对于`self-attention`，`Q`、`K`、`V`都来自输入`X`，在计算$QK^T$是，模型很容易关注到自身位置，即$QK^T$对角线上的激活值明显比较大。削弱了模型关注其他高价值位置的能力，也就限制了模型的理解和表达能力（过多关注自己本身），`MHA`对这个问题有一些缓解作用。
# `MHA`
`MHA`随着*Attention Is All You Need*一起提出，主要只干一件事：把原来一个`attention`计算，拆分成多个小份的`attention`并行计算，最后再融合回原本维度：
$$
\begin{align}
MultiheadAttention(Q, K, V) = Concat(head_1, ..., head_h) \\
head_i = Attention(W_i^Q Q, W_i^K K, W_i^V V)
\end{align}
$$
原论文指出这样的做法将一个大的维度`d_model`拆分成了多个语义空间，让不同的注意力头可以从不同的角度来分析和理解输入信息，获得更好的效果。
# 解码中的`KV cache`
无论是什么结构，解码生成时都采用自回归`auto-regressive`方式，也就是说，解码时，先根据当前输入$input_{i-1}$生成下一个$token_i$，然后把新生成的$token_i$拼接在$input_{i-1}$后面，获得新的输入$input_i$，再用$input_i$生成$token_{i+1}$，依次迭代，直到生成结束。
代入`attention`计算：$\alpha_{i,j} = softmax(q_i k_j^T), o_i = \sum_{j=0}^i \alpha_{i, j} v_j$。
在`decode`的过程中，由于`mask attention`的存在，每个输入只能看到自己和前面的内容，而看不到后面的内容，即假设当前输入的长度是$3$，预测第$4$个字，那每层`attention`所做的计算有：
$$
\begin{align}
o_0 = \alpha_{0, 0} v_0 \\
o_1 = \alpha_{0, 0} v_0 + \alpha_{1, 1} v_1 \\
o_2 = \alpha_{0, 0} v_0 + \alpha_{1, 1} v_1 + \alpha_{2, 2} v_2
\end{align}
$$

预测完第$4$个字，放入输入中，继续预测第$5$个字，每层`attention`所做的计算有：
$$
\begin{align}
o_0 = \alpha_{0, 0} v_0 \\
o_1 = \alpha_{0, 0} v_0 + \alpha_{1, 1} v_1 \\
o_2 = \alpha_{0, 0} v_0 + \alpha_{1, 1} v_1 + \alpha_{2, 2} v_2 \\
o_3 = \alpha_{0, 0} v_0 + \alpha_{1, 1} v_1 + \alpha_{2, 2} v_2 + \alpha_{3, 3} v_3
\end{align}
$$

只有最后一步引入了新的计算，之前的计算全是重复的。但模型在推理时不管这些，无论是不是只要最后一个字的输出，它都把所有输入计算了一遍，给出所有输出结果。也就是说中间浪费了很多计算资源，并且随着生成序列的增长，浪费的计算资源成平方关系递增。
为了减少这些浪费，利用`KV cache`将上一个step中的计算结果缓存下来，其中每层的`k`和`v`就是要缓存的对象。
$$
cache_l = [(k_0, v_0), (k_1, v_1), ..., (k_m, v_m)]
$$
因为第$l$层的$o_i$本来就会经过`FFN`之后进入到$l+1$层，再经过新的投影变换，成为$l+1$层的`k`和`v`，但是$l+1$层的`k`和`v`我们已经缓存过了，只需要把本次新增的`k`和`v`计算出来存入缓存即可。这样的操作减少了大量的推理过程中`attention`和`FFN`的重复计算。
总体来说，`KV Cache`以空间换时间，通过快速的缓存存取，减少了重复计算，但使用了`KV Cache`也引发了新的问题，对于输入长度为$s$，层数为$L$，`hidden size`为$d$的模型，所需要缓存的参数量为：$2 \times L \times s \times d$，若使用半精度浮点数，所需要的空间还需要扩大$2$倍。
以`LLaMA 2 7B`为例，$L=32, hidden_size=4096$，每个`token`需要的缓存空间就是`524288bytes`，约为$52K$，当$s=1024$，缓存超过$500M$，这只是$batch_size = 1$的情况，批量增大是，这个值很容易超过$1G$。
对比显卡的`Cache`，这个数据量过于庞大，超出显卡`Cache`的部分只能走到现存中，而现存速度比`Cache`慢很多。
# `MQA`
`Google`在2019年在*Fast Transformer Decoding: One Write-Head is All You Need*中提出了`MQA`，做法很简单：在`MHA`中，输入分别经过$W_Q$、$W_K$、$W_V$的变换后，都切成$n$份，维度也从$d_model$下降到$d_head$，分别进行`attention`再拼接计算。而`MQA`中，在线性变换后，只对`Q`进行切分，而`K`、`V`则直接在线性变化后将维度降低到$d_head$（而非切分变小），这样$n$个`Query`头和同一份`K`、`V`进行`attention`计算，再将结果拼接起来。这样依赖，需要缓存的`K`、`V`值就从所有头变成了一个头的量。
由于共享了多个头的参数，限制了模型的表达能力，`MQA`虽然能很好支持推理加速，但是在效果上比`MHA`略差一些。
# `GQA`
既然`MQA`对效果有影响，`MHA`缓存又存不下，于是提出了一个这种的方案，既能减少`MQA`效果损失，又相比`MHA`需要更小的缓存。
*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*提出了`GQA`，在这种做法中，`Q`还是按照原来的`MHA/MQA`方案不变，因为一套`K`、`V`效果不好，那就多弄几套，但是限制数量，比$Q$的头数少一些。相当于把`Q`的多个头分组，同一个组内的`Q`共享同一套`K`、`V`，不同组的`K`、`V`不同。
`LLaMA 2`采用的就是`GQA`，的小的效果不错。