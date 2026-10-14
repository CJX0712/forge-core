# Forge Core — 零依赖、可验证的世界级 AI 数值内核

> **作者：晨星** · MIT License · 浏览器 + Node 双端零依赖运行

Forge Core 是一个**单文件、零外部依赖**的 AI / 数值计算内核。它以 `autograd`（反向模式自动微分）为统一基石，集成了世界顶级数学与算法成果，并为**每一个模块**配备可机器校验的数学不变量（解析解 vs 数值解、正交性、单调性、梯度检验），做到「写出来即可验证、可复现」。

设计原则：**集大成 + 零重写**——不重复造轮子，直接立于世界级成果之上；遇更强现成方案立即替换自写。

---

## ✦ 集大成：算法来源（世界级成果）

| 模块 | 采用的顶级算法 / 理论 | 来源 |
|---|---|---|
| 线性代数 · 特征分解 | **Jacobi 旋转法** | C.G.J. Jacobi, 1846 / 1890 |
| 线性代数 · SVD | 经 **A^T A 特征分解**实现（Golub-Reinsch 思想） | Golub & Reinsch, 1970 |
| 线性代数 · Cholesky | 对称正定分解 + 脊项数值稳定 | Cholesky, 1814 |
| 自动微分 | **micrograd 式反向传播**（JAX / tinygrad 内核同源） | Karpathy, 2019 思想 |
| 聚类 | **Lloyd k-means** | Lloyd, 1957 / 1982 |
| 密度估计 | **EM 算法** 用于高斯混合模型 | Dempster, Laird & Rubin, 1977 |
| 降维 | **PCA**（Pearson）经 SVD 实现 | Pearson, 1901 |
| 注意力 | **Scaled Dot-Product Attention** | Vaswani et al., 2017 (Transformer) |

---

## ✦ 可验证：8/8 绿的数学不变量

运行 `node test.js`，八项检查全部通过（✅ 8/8）。每一项对应一条可独立交叉验证的硬金标准：

| # | 检查 | 不变量（验证总纲） |
|---|---|---|
| 1 | **SVD** | 重建误差 `‖A − UΣVᵀ‖ ≈ 0` 且 `UᵀU = VᵀV = I` |
| 2 | **Cholesky + 求解** | `A·x − b ≈ 0`（解析分解 vs 数值求解） |
| 3 | **Jacobi 特征分解** | `A·v = λv` 且特征向量正交 |
| 4 | **autograd MLP 梯度检验** | 解析梯度 vs 中心差分，最大相对误差 `≈ 1e-11` |
| 5 | **MLP 学习 XOR** | 损失显著下降（→ ~0），证明前向/反向/优化链路正确 |
| 6 | **注意力梯度检验** | 解析梯度 vs 有限差分 `≈ 1e-7` |
| 7 | **k-means / GMM-EM** | k-means 的 SSE **单调不增**；GMM 的 log-likelihood **单调不减**；责任度/权重行和=1；协方差对称 |
| 8 | **PCA / 线性回归 / 逻辑回归** | PCA 满秩重建≈0；线性回归残差 `⊥ X`（正规方程 `Xᵀ(y−Xŵ)=0`）；可分数据逻辑回归 acc=1 |

---

## ✦ 快速开始

### 浏览器（单文件，零安装）
下载 `index.html`，双击即可运行。页面内置「运行自检」与「实时训练 XOR」演示。

```html
<script src="forge-core.js"></script>
<script>
  const { svd, pca, trainMLP, attention } = ForgeCore;
  const { U, S, V } = svd([[1,2],[3,4],[5,6]]);
</script>
```

### Node
```bash
npm test        # 运行 8/8 自检
node build-demo.js   # 重新生成单文件 index.html
```
```js
const FC = require('./forge-core.js');
const { U, S, V } = FC.svd([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]);
console.log(S.map(x => x.toFixed(3))); // 奇异值
```

---

## ✦ API 导览

**线性代数 (`M` + 分解)**
- `M.matMul / transpose / add / sub / scale / colVec / randn / frobenius / maxAbs`
- `cholesky(A)` → `L`（下三角，`A = LLᵀ`）
- `cholSolve(L, B)` → 解 `LLᵀX = B`
- `jacobiEigen(A)` → `{ values, vectors }`（对称矩阵）
- `svd(A)` → `{ U, S, V }`；`svdReconstruct({U,S,V}, k)` 重建
- `invMatrix(A)` / `detMatrix(A)`

**自动微分 (`Tensor`)**
- `new Tensor(data, requiresGrad)` + 算子：`Tmatmul / Tadd / Tsub / Tmul / Tscale / Ttanh / Tsigmoid / Tsoftmax / Ttranspose / Tsum / Tmean`
- `backward(loss)` 反向传播
- `gradCheck(forward, params)` → 解析梯度 vs 有限差分最大相对误差

**模型 / 算法**
- `trainMLP(X, y, {lr, iters, h, record})` → 2 层 MLP（tanh 隐藏 + 线性输出）
- `attention(Q, K, V)` → `{ A, out }`（缩放点积注意力）
- `linearRegression(X, y)` → 正规方程解
- `logisticRegression(X, y, {lr, iters})` → 二分类
- `kmeans(X, k, {iters})` → `{ centers, assign, sseHistory }`
- `gmmEM(X, k, {iters})` → `{ means, covs, weights, R, logLikHistory }`
- `pca(X, k)` → `{ mean, components, projected, explained, reconErr }`

---

## ✦ 架构图

```
            ┌─────────────────────────────────────────────┐
            │            Forge Core（零依赖内核）           │
            └─────────────────────────────────────────────┘
                              │
        ┌──────────┬──────────┴──────────┬────────────────┐
        ▼          ▼                     ▼                ▼
  ┌──────────┐ ┌──────────┐      ┌──────────────┐  ┌──────────────┐
  │ 自研 线性 │ │ autograd │      │  监督/无监督  │  │   注意力      │
  │ 代数内核 │ │(反向模式) │      │ LR / 逻辑回归 │  │ scaled-dot   │
  │ Cholesky │ │ micrograd │      │ k-means /GMM │  │ -product     │
  │ Jacobi   │ │ 式自动微分│      │ PCA / SVD    │  │ (Transformer)│
  │ SVD      │ └────┬─────┘      └──────┬───────┘  └──────────────┘
  └────┬─────┘      │                   │
       │            └─────────┬─────────┘
       │                      ▼
       │              ┌──────────────┐
       └─────────────▶│  MLP (2层)   │ ← autograd 驱动训练
                      └──────────────┘
                              │
                              ▼
                  每模块「双重实现交叉验证」数学不变量
                  （解析解 vs 数值解 / 正交 / 单调 / 梯度检验）
```

---

## ✦ 闭环与验证

- **可运行**：零依赖，浏览器直接打开 / Node 直接 `require`。
- **可复现**：自检使用固定随机种子（`mulberry32(20261015)`），结果稳定。
- **可验证**：`node test.js` 输出 `🟢 全部通过`（8/8 绿）。
- **开源**：完整项目开源至 GitHub，作者署名 **晨星**。

---

## ✦ License

MIT License — Copyright © 晨星 (Morning Star). 自由用于研究、教学与生产。
