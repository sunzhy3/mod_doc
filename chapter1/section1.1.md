# Section1.1

Fast Optical Flow using Dense Inverse Search

作者: Till Kroeger,  Radu Timofte, Dengxin Dai, Luc Van Goo

发表年份: 2016年

发表会议(期刊): ECCV


[项目主页](http://www.vision.ee.ethz.ch/~kroegert/OFlow/)

# 1.Introduction
略


# 2. Proposed method

## 2.1 Fast inverse search for correspondences
核心的部分就是如何实现高效的进行块匹配的搜索, 

对于一个在参考图像 $$I_{t}$$ 中给定的模板块 $$T$$, 大小为 $$\theta_{ps} \times \theta_{ps}$$, 中心点为 $$x = (x, y)^{T}$$, 我们的目标是在下一帧图像 $$I_{t+1}$$ 中找到大小为 $$\theta_{ps} \times \theta_{ps}$$最匹配的块, 

转化为我们希望寻找一个匹配向量 $$u = (u, v)$$ 之后我们可以通过梯度下降最小化

注: 下面 $$u$$, $$x$$ 都指向量.

$$
u = argmin_{u^{'}} \sum_{x} [I_{t+1}(x + u^{'}) - T(x)]^{2}, \quad (1)
$$

我们可以利用 LuKas-Kanade 算法来进行优化, 直到上面的公式收敛, 一般分成两步.

$$
\Delta u = argmin_{\Delta u^{'}} \sum_{x} [I_{t+1}(x+u+\Delta u^{'}) - T(x)]^{2}, \quad (2)

u = u + \Delta u
$$
原始的 LuKas-Kanade 算法需要复杂的重复计算 Hessian 矩阵, 具体的讲, 

将 一个匹配记做 $$W(x; u)$$, 其中 $$u = (u, v)^{T}$$, $$x = (x, y)^{T}$$, 则
$$W(x; u) = (x+u, y+v)$$, 则我们需要优化的目标函数为

$$
\sum_{x} [I_{t+1}(W(x; u)) - T(x)]^{2}
$$

其中 $$u$$ 通过两步来寻找

$$
\sum_{x} [I_{t+1}(W(x; u + \Delta u)) - T(x)]^{2}, (6)

u = u + \Delta u
$$

采用 Gauss-Newton 梯度下降来最小化方程 (6), 将前一项进行一阶泰勒展开,

$$
\sum_{x} [I_{t+1}(W(x; u)) + \nabla I_{t+1}\frac{\partial W}{\partial u} \Delta u - T(x)]^{2}, \qquad (7)
$$

其中 $$\nabla I_{t+1}$$ 为 $$W(x; u)$$ 处的图像梯度, $$\frac{\partial W}{\partial u}$$ 为匹配的 Jacobian 矩阵 (对光流来讲就是 2\*2 的单位阵)

下面可以求最小化方程 (7) 的一个闭式解, 令 (7) 对于 $$\Delta u$$ 的偏导数为 0 可得

$$
\sum_{x} S^{T} [I_{t+1}(W(x; u)) + \nabla I_{t+1}\frac{\partial W}{\partial u} \Delta u - T(x)] = 0, \qquad (8)
$$

其中 $$S = [\nabla I_{t+1} \frac{\partial W}{\partial u}]$$, 于是我们可以解出 $$\Delta u$$

$$
\Delta u = H^{-1} \sum_{x} S^{T} [T(x) - I_{t+1}(W(x; u))], \qquad (9)
$$

其中 $$H = \sum_{x} S^{T}S$$ 为 $$n \times n$$ 的 Hessian 矩阵的估计.

由于 $$S$$ 依赖于 图像 $$I_{t+1}$$ 对 $$u$$ 的梯度, 所以 $$S, H$$ 每次迭代的过程都需要被重复计算. 接下来我们对需要优化的目标函数进行一下修改.

$$
\sum_{x} [T(W(x; \Delta u)) - I_{t+1}(W(x; u))]^{2}, (10)

W(x; u) = W(x; u) \circ W(x; \Delta u)^{-1} (u = u - \Delta u)
$$

此时将 (10) 进行一阶泰勒展开得到

$$
\sum_{x} [T(W(x;0)) + \nabla T \frac{\partial W}{\partial u} \Delta u - I_{t+1}(W(x; u))], \qquad (11)
$$

其中 $$W(x;0)$$ 就是不变匹配, 与 (9) 同理我们可以解 (11) 得到

$$
\Delta u = H^{'-1} \sum_{x} S^{'T} [I_{t+1}(W(x; u)) - T(x)], \qquad (12)
$$

其中 $$S^{'} = [\nabla T \frac{\partial W}{\partial u}]$$, $$H^{'} = \sum_{x} S^{'T}S^{'}$$, 并且 Jaconbian 矩阵  $$\frac{\partial W}{\partial u}$$ 是在 $$(x; 0)$$ 处计算的. 所以 $$S^{'}, H^{'}$$ 都是不依赖于 $$u$$ 可以提前计算出来的.

实际中, 本文对 patch 进行 mean-normalize

可能出现问题的部分是, 如果匹配超出 patch 的大小, 很难能够得到优化. 此外, 梯度下降最好是能在对应 patch 相似的情况下, 于是本文采用 corse-to-fine 的方式来进行匹配的寻找.

## 2.2 Fast Optical flow with multi-scale reasoning
$$\theta_{ss}$$: 分辨率最低层索引\
$$\theta_{sd}$$: 下采样率, 一般大于 1 \
$$\theta_{sf}$$: 分辨率最高层索引

1) Creation of grid

由参数 $$\theta_{ov} \in [0, 1)$$ 决定划分的 patch 数 $$N_{s}$$, $$\theta_{ov}$$ 表示patch 间重叠的比率, 例如 $$\theta_{ov} = 0$$ 表示两个邻近的 patch 没有重叠.

2) Initialization

对于第一次迭代($$s = \theta_{ss}$$), 我们将 flow 初始化为 0, 对于后面的
$$
u_{i, init} = U_{s+1}(\frac{x}{\theta_{sd}})\theta_{sd}
$$

3) Inverse search

as in 2.1

4) Densification

加一步来增加结果的鲁棒性, 基于步骤 (3) 我们获得更新后的匹配向量 $$u_{i}$$

首先

if $$||u_{i, init} - u_{i}||_{2} > \theta_{ps}$$ \
$$\quad$$ $$u_{i} = u_{i, init}$$

之后, 对每个像素 $$x$$ 上的匹配向量进行加权平均(由于上面的 patch 有重叠)

$$
U_{s}(x) = \frac{1}{Z} \sum_{i}^{N_{s}} \frac{\lambda_{i, x}}{max(1, ||d_{i}(x)||_{2})} u_{i}, \qquad (3)
$$

其中, 若在参考帧中 patch $$i$$ 包含像素 $$x$$, 则指示函数 $$\lambda_{i, x} = 1$$

$$d_{i}(x) = I_{t+1}(x + u_{i}) - T(x)$$ 表示对应像素之间的差别.

$$Z = \sum_{i} \lambda_{i, x} / max(1, \Vert d_{i}(x) \Vert_{2})$$, 为归一化系数.

5) Variational energy minimization
变分能量优化, 参见2.3

## 2.3 Fast Variational refinement

变分能量优化的能量函数为

$$
E(U) = \int_{\omega} \sigma \Phi(E_{I}) + \gamma \Phi(E_{G}) + \alpha \Phi(E_{S}) dx, \qquad (4)
$$

其中, $$E_{I}$$ 为 intensity data trem, $$E_{G}$$ 为 gradient data term, $$E_{S}$$ 为 smoothness term.

$$\Phi(a^{2}) = \sqrt(a^{2} + \epsilon^{2})$$, 其中 $$\epsilon = 0.001$$

令 $$\nabla_{3} = (\partial x, \partial y, \partial z)^{T}$$ , 
$$\nabla_{2} = (\partial x, \partial y)^{T}$$ 对于 intensity data term, 主要是基于相邻帧图像的光照强度(图像像素值)应该是近似不变的. 

$$
E_{I} = u^{T}J_{0}u

J_{0} = \beta_{0} (\nabla_{3}I)(\nabla_{3}^{T}I)

\beta_{0} = (||\nabla_{2}I||^{2} + 0.001)^{-1}
$$

对于 gradient data term, 主要是基于像素的邻域梯度不会有比较大的变化

$$
E_{G} = u^{T} J_{xy} u

J_{xy} = \beta_{x}(\nabla_{3}I_{dx})(\nabla_{3}^{T}I_{dx}) + 
\beta_{y}(\nabla_{3}I_{dy})(\nabla_{3}^{T}I_{dy})

\beta_{x} = (||\nabla_{2}I_{dx}||^{2} + 0.001)^{-1}

\beta_{y} = (||\nabla_{2}I_{dy}||^{2} + 0.001)^{-1}
$$

对于 smoothness term 

$$
E_{S} = ||\nabla u||^{2} + ||\nabla v||^{2}
$$

非凸能量函数 $$E(U)$$ 的迭代最小化通过 $$\theta_{vo}$$ 次不动点迭代, 以及 $$\theta_{vi}$$ 次 SOR (Successive Over Realaxation) 迭代来实现. 不动点迭代比较通用, SOR 迭代可以保证收敛比较稳定.

## 2.4 Extension

1) Parallelization

2) Using RGB

3) Merging forward-backward flow estimations

4) Robust error norms

5) Using DIS for stereo depth


# 3. Experiments

略























