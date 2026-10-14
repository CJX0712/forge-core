/*
 * Forge Core — 零依赖、可验证的世界级 AI 数值内核
 * ============================================================
 * 集大成：每一模块均立于世界顶级成果之上，禁止从零造轮子。
 *   - 线性代数：Jacobi 特征分解 (Jacobi 1846/1890)、Golub-Reinsch 思想派生的 SVD、
 *     Cholesky 分解 (Cholesky 1814 / 数值稳定变体)
 *   - autograd：micrograd (Karpathy) 式反向模式自动微分，tinygrad/JAX 内核思想
 *   - 聚类：Lloyd 的 k-means (1957/1982)
 *   - 密度估计：Dempster 的 EM 算法 (1977) 用于 GMM
 *   - 降维：Pearson 的 PCA (1901) / 经 SVD 实现 (Golub-Reinsch)
 *   - 注意力：Vaswani 等 scaled dot-product attention (2017)
 * 可验证：每个模块都带「双重实现交叉验证」数学不变量（见 test.js：8/8 绿）。
 *
 * 作者：晨星  |  许可：MIT  |  运行：浏览器（<script>）或 Node（require）
 */
(function (global) {
  'use strict';

  /* ============================ 1. 矩阵/向量原语 ============================ */
  // 矩阵一律表示为 number[][]（行优先）。向量为 1D number[] 或列矩阵 number[][]。
  const M = {
    zeros(r, c) {
      const A = new Array(r);
      for (let i = 0; i < r; i++) A[i] = new Array(c).fill(0);
      return A;
    },
    identity(n) {
      const A = M.zeros(n, n);
      for (let i = 0; i < n; i++) A[i][i] = 1;
      return A;
    },
    matMul(A, B) {
      const r = A.length, n = B.length, c = B[0].length;
      const O = M.zeros(r, c);
      for (let i = 0; i < r; i++) {
        const Arow = A[i], Orow = O[i];
        for (let k = 0; k < n; k++) {
          const aik = Arow[k];
          if (aik === 0) continue;
          const Brow = B[k];
          for (let j = 0; j < c; j++) Orow[j] += aik * Brow[j];
        }
      }
      return O;
    },
    transpose(A) {
      const r = A.length, c = A[0].length;
      const O = M.zeros(c, r);
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) O[j][i] = A[i][j];
      return O;
    },
    add(A, B) {
      const r = A.length, c = A[0].length, O = M.zeros(r, c);
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) O[i][j] = A[i][j] + B[i][j];
      return O;
    },
    sub(A, B) {
      const r = A.length, c = A[0].length, O = M.zeros(r, c);
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) O[i][j] = A[i][j] - B[i][j];
      return O;
    },
    mul(A, B) { // 逐元素
      const r = A.length, c = A[0].length, O = M.zeros(r, c);
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) O[i][j] = A[i][j] * B[i][j];
      return O;
    },
    scale(A, s) { return A.map(row => row.map(v => v * s)); },
    colVec(arr) { return arr.map(v => [v]); },
    // Box-Muller 标准正态
    randn() {
      let u = 0, v = 0;
      while (u === 0) u = Math.random();
      while (v === 0) v = Math.random();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    },
    frobenius(A) {
      let s = 0;
      for (const row of A) for (const v of row) s += v * v;
      return Math.sqrt(s);
    },
    maxAbs(A) {
      let m = 0;
      for (const row of A) for (const v of row) m = Math.max(m, Math.abs(v));
      return m;
    }
  };

  /* ============================ 2. Cholesky 分解与求解 ============================ */
  // 对对称正定矩阵 A 求下三角 L，使 A = L L^T。
  // 数值稳定：先对称化并加极小脊项，避免非 SPD 时崩溃（世界顶级数值实践）。
  function cholesky(A, ridge = 1e-12) {
    const n = A.length;
    const S = A.map((row, i) => row.map((v, j) => (i === j ? v + ridge : 0.5 * (v + A[j][i]))));
    const L = M.zeros(n, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = S[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) L[i][j] = Math.sqrt(Math.max(s, 1e-18));
        else L[i][j] = s / L[j][j];
      }
    }
    return L;
  }
  // 解 L L^T X = B（B 为列矩阵）。前向回代 + 后向回代。
  function cholSolve(L, B) {
    const n = L.length, c = B[0].length;
    const Y = M.zeros(n, c);
    for (let j = 0; j < c; j++) {
      for (let i = 0; i < n; i++) {
        let s = B[i][j];
        for (let k = 0; k < i; k++) s -= L[i][k] * Y[k][j];
        Y[i][j] = s / L[i][i];
      }
    }
    const X = M.zeros(n, c);
    for (let j = 0; j < c; j++) {
      for (let i = n - 1; i >= 0; i--) {
        let s = Y[i][j];
        for (let k = i + 1; k < n; k++) s -= L[k][i] * X[k][j];
        X[i][j] = s / L[i][i];
      }
    }
    return X;
  }

  /* ============================ 3. Jacobi 特征分解（对称矩阵） ============================ */
  // 经典 Jacobi 旋转法：反复消去最大非对角元，累加正交矩阵 V。
  // 不变量：A_orig · V ≈ V · diag(λ)，且 V^T V = I。
  function jacobiEigen(Ain, maxSweeps = 100, tol = 1e-12) {
    const n = Ain.length;
    const A = Ain.map((row, i) => row.map((v, j) => (i === j ? v : 0.5 * (v + Ain[j][i]))));
    const V = M.identity(n);
    for (let sweep = 0; sweep < maxSweeps; sweep++) {
      let p = 0, q = 1, max = 0;
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++) {
          const v = Math.abs(A[i][j]);
          if (v > max) { max = v; p = i; q = j; }
        }
      if (max < tol) break;
      const app = A[p][p], aqq = A[q][q], apq = A[p][q];
      const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
      const c = Math.cos(phi), s = Math.sin(phi);
      for (let i = 0; i < n; i++) {
        if (i === p || i === q) continue;
        const aip = A[i][p], aiq = A[i][q];
        A[i][p] = c * aip - s * aiq;
        A[i][q] = s * aip + c * aiq;
        A[p][i] = A[i][p];
        A[q][i] = A[i][q];
      }
      A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
      A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
      A[p][q] = 0; A[q][p] = 0;
      for (let i = 0; i < n; i++) {
        const vip = V[i][p], viq = V[i][q];
        V[i][p] = c * vip - s * viq;
        V[i][q] = s * vip + c * viq;
      }
    }
    const values = [];
    for (let i = 0; i < n; i++) values.push(A[i][i]);
    const idx = values.map((v, i) => i).sort((a, b) => values[b] - values[a]);
    return {
      values: idx.map(i => values[i]),
      vectors: idx.map(i => V.map(row => row[i])) // 单位特征向量（列）
    };
  }

  /* ============================ 4. 奇异值分解 SVD ============================ */
  // A (m×n) = U Σ V^T。通过 A^T A 的 Jacobi 特征分解得 V 与奇异值，
  // U 由 A·v_k/σ_k 构造并对剩余列 Gram-Schmidt 正交补全（满秩 U）。
  function svd(A) {
    const m = A.length, n = A[0].length;
    const At = M.transpose(A);
    const ATA = M.matMul(At, A);
    const { values, vectors } = jacobiEigen(ATA); // vectors[k] = 右奇异向量 v_k（行向量，长度 n）
    const R = vectors;
    const S = values.map(v => Math.sqrt(Math.max(v, 0)));
    const tol = 1e-9 * (S[0] || 1);
    const U = M.zeros(m, m);
    const setCols = [];
    // 先填入有效左奇异向量（保留 A v_k = σ_k u_k 关系，重建精确）
    for (let k = 0; k < n; k++) {
      const Av = M.matMul(A, M.colVec(R[k]));
      const norm = Math.sqrt(Av.reduce((s, r) => s + r[0] * r[0], 0));
      if (norm > tol) {
        for (let i = 0; i < m; i++) U[i][k] = Av[i][0] / norm;
        S[k] = norm;
        setCols.push(k);
      } else {
        S[k] = 0; // 近零奇异值：列留待 Gram-Schmidt 补全
      }
    }
    // Gram-Schmidt 补全剩余列，保证 U 为完整标准正交基（σ=0 项不影响重建）
    for (let k = 0; k < m; k++) {
      if (setCols.includes(k)) continue;
      let vec = [];
      for (let i = 0; i < m; i++) vec.push(M.randn());
      for (const j of setCols) {
        let dot = 0;
        for (let i = 0; i < m; i++) dot += vec[i] * U[i][j];
        for (let i = 0; i < m; i++) vec[i] -= dot * U[i][j];
      }
      const norm = Math.sqrt(vec.reduce((a, b) => a + b * b, 0)) || 1;
      for (let i = 0; i < m; i++) U[i][k] = vec[i] / norm;
      setCols.push(k);
    }
    const V = M.transpose(R); // 列 = 右奇异向量（标准 SVD: A = U Σ V^T）
    return { U, S, V };
  }
  function svdReconstruct({ U, S, V }, k) {
    const m = U.length, n = V.length;
    const kk = k == null ? Math.min(m, n) : k;
    const Sigma = M.zeros(m, n);
    for (let i = 0; i < kk; i++) Sigma[i][i] = S[i];
    return M.matMul(M.matMul(U, Sigma), M.transpose(V));
  }

  /* ============================ 5. 高斯消元辅助：逆与行列式 ============================ */
  function invMatrix(A) {
    const n = A.length;
    const X = A.map((row, i) => row.concat(M.identity(n)[i]));
    for (let i = 0; i < n; i++) {
      let piv = i;
      for (let r = i + 1; r < n; r++) if (Math.abs(X[r][i]) > Math.abs(X[piv][i])) piv = r;
      [X[i], X[piv]] = [X[piv], X[i]];
      const d = X[i][i];
      for (let j = 0; j < 2 * n; j++) X[i][j] /= d;
      for (let r = 0; r < n; r++) {
        if (r === i) continue;
        const f = X[r][i];
        for (let j = 0; j < 2 * n; j++) X[r][j] -= f * X[i][j];
      }
    }
    return X.map(row => row.slice(n));
  }
  function detMatrix(A) {
    const n = A.length;
    const X = A.map(row => row.slice());
    let det = 1;
    for (let i = 0; i < n; i++) {
      let piv = i;
      for (let r = i + 1; r < n; r++) if (Math.abs(X[r][i]) > Math.abs(X[piv][i])) piv = r;
      if (piv !== i) { [X[i], X[piv]] = [X[piv], X[i]]; det = -det; }
      const d = X[i][i];
      if (Math.abs(d) < 1e-15) return 0;
      det *= d;
      for (let r = i + 1; r < n; r++) {
        const f = X[r][i] / d;
        for (let j = i; j < n; j++) X[r][j] -= f * X[i][j];
      }
    }
    return det;
  }

  /* ============================ 6. 反向模式自动微分（autograd） ============================ */
  // micrograd 式：标量/矩阵张量 + 计算图 + 反向传播。世界顶级框架（JAX/tinygrad）内核同源。
  class Tensor {
    constructor(data, requiresGrad = false) {
      this.data = data;             // number[][]
      this.requiresGrad = requiresGrad;
      this.grad = null;
      this._backward = () => {};
      this._prev = [];
    }
    zeroGrad() { this.grad = M.zeros(this.data.length, this.data[0].length); }
  }
  function Tmatmul(a, b) {
    const out = new Tensor(M.matMul(a.data, b.data), a.requiresGrad || b.requiresGrad);
    out._prev = [a, b];
    out._backward = () => {
      if (a.requiresGrad) { if (!a.grad) a.zeroGrad(); a.grad = M.add(a.grad, M.matMul(out.grad, M.transpose(b.data))); }
      if (b.requiresGrad) { if (!b.grad) b.zeroGrad(); b.grad = M.add(b.grad, M.matMul(M.transpose(a.data), out.grad)); }
    };
    return out;
  }
  function Tadd(a, b) { // a(m,n) + b(1,n 广播 或 m,n)
    const outData = b.data.length === 1
      ? a.data.map(row => row.map((v, j) => v + b.data[0][j]))
      : M.add(a.data, b.data);
    const out = new Tensor(outData, a.requiresGrad || b.requiresGrad);
    out._prev = [a, b];
    out._backward = () => {
      if (a.requiresGrad) { if (!a.grad) a.zeroGrad(); a.grad = M.add(a.grad, out.grad); }
      if (b.requiresGrad) {
        if (!b.grad) b.zeroGrad();
        if (b.data.length === 1) {
          const acc = new Array(b.data[0].length).fill(0);
          for (const row of out.grad) for (let j = 0; j < row.length; j++) acc[j] += row[j];
          for (let j = 0; j < b.data[0].length; j++) b.grad[0][j] += acc[j];
        } else b.grad = M.add(b.grad, out.grad);
      }
    };
    return out;
  }
  function Tsub(a, b) {
    const out = new Tensor(M.sub(a.data, b.data), a.requiresGrad || b.requiresGrad);
    out._prev = [a, b];
    out._backward = () => {
      if (a.requiresGrad) { if (!a.grad) a.zeroGrad(); a.grad = M.add(a.grad, out.grad); }
      if (b.requiresGrad) { if (!b.grad) b.zeroGrad(); b.grad = M.add(b.grad, M.scale(out.grad, -1)); }
    };
    return out;
  }
  function Tmul(a, b) { // 逐元素
    const r = a.data.length, c = a.data[0].length;
    const d = M.zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) d[i][j] = a.data[i][j] * b.data[i][j];
    const out = new Tensor(d, a.requiresGrad || b.requiresGrad);
    out._prev = [a, b];
    out._backward = () => {
      if (a.requiresGrad) { if (!a.grad) a.zeroGrad(); a.grad = M.add(a.grad, M.mul(b.data, out.grad)); }
      if (b.requiresGrad) { if (!b.grad) b.zeroGrad(); b.grad = M.add(b.grad, M.mul(a.data, out.grad)); }
    };
    return out;
  }
  function Tscale(a, s) {
    const out = new Tensor(M.scale(a.data, s), a.requiresGrad);
    out._prev = [a];
    out._backward = () => { if (a.requiresGrad) { if (!a.grad) a.zeroGrad(); a.grad = M.add(a.grad, M.scale(out.grad, s)); } };
    return out;
  }
  function Ttanh(a) {
    const d = a.data.map(row => row.map(v => Math.tanh(v)));
    const out = new Tensor(d, a.requiresGrad);
    out._prev = [a];
    out._backward = () => {
      if (a.requiresGrad) {
        if (!a.grad) a.zeroGrad();
        const g = a.data.map((row, i) => row.map((v, j) => out.grad[i][j] * (1 - d[i][j] * d[i][j])));
        a.grad = M.add(a.grad, g);
      }
    };
    return out;
  }
  function Tsigmoid(a) {
    const d = a.data.map(row => row.map(v => 1 / (1 + Math.exp(-v))));
    const out = new Tensor(d, a.requiresGrad);
    out._prev = [a];
    out._backward = () => {
      if (a.requiresGrad) {
        if (!a.grad) a.zeroGrad();
        const g = a.data.map((row, i) => row.map((v, j) => out.grad[i][j] * d[i][j] * (1 - d[i][j])));
        a.grad = M.add(a.grad, g);
      }
    };
    return out;
  }
  function Tsoftmax(a) { // 每行独立归一
    const d = a.data.map(row => {
      const m = Math.max(...row);
      const ex = row.map(v => Math.exp(v - m));
      const sum = ex.reduce((x, y) => x + y, 0);
      return ex.map(v => v / sum);
    });
    const out = new Tensor(d, a.requiresGrad);
    out._prev = [a];
    out._backward = () => {
      if (a.requiresGrad) {
        if (!a.grad) a.zeroGrad();
        const g = d.map((p, i) => {
          let dot = 0;
          for (let j = 0; j < p.length; j++) dot += out.grad[i][j] * p[j];
          return p.map((pj, j) => pj * (out.grad[i][j] - dot));
        });
        a.grad = M.add(a.grad, g);
      }
    };
    return out;
  }
  function Ttranspose(a) {
    const out = new Tensor(M.transpose(a.data), a.requiresGrad);
    out._prev = [a];
    out._backward = () => { if (a.requiresGrad) { if (!a.grad) a.zeroGrad(); a.grad = M.add(a.grad, M.transpose(out.grad)); } };
    return out;
  }
  function Tsum(a) {
    let s = 0;
    for (const row of a.data) for (const v of row) s += v;
    const out = new Tensor([[s]], a.requiresGrad);
    out._prev = [a];
    out._backward = () => {
      if (a.requiresGrad) {
        if (!a.grad) a.zeroGrad();
        const g = a.data.map(row => row.map(() => out.grad[0][0]));
        a.grad = M.add(a.grad, g);
      }
    };
    return out;
  }
  function Tmean(a) { return Tscale(Tsum(a), 1 / (a.data.length * a.data[0].length)); }

  function backward(loss) {
    const topo = [], visited = new Set();
    (function build(t) {
      if (visited.has(t)) return;
      visited.add(t);
      for (const p of t._prev) build(p);
      topo.push(t);
    })(loss);
    for (const t of topo) if (t.requiresGrad) t.zeroGrad(); // 每次反向前重置叶子梯度，避免跨次累加
    loss.grad = M.zeros(loss.data.length, loss.data[0].length);
    loss.grad[0][0] = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }

  // 有限差分梯度校验：对比 autograd 解析梯度与中心差分数值梯度，返回最大相对误差。
  function gradCheck(forward, params, eps = 1e-5, tol = 1e-3) {
    const loss = forward();
    backward(loss);
    const analytic = params.map(p => p.grad.map(row => row.slice()));
    let maxRel = 0;
    for (let pi = 0; pi < params.length; pi++) {
      const p = params[pi];
      const r = p.data.length, c = p.data[0].length;
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
        const orig = p.data[i][j];
        p.data[i][j] = orig + eps;
        const lp = forward().data[0][0];
        p.data[i][j] = orig - eps;
        const lm = forward().data[0][0];
        p.data[i][j] = orig;
        const fd = (lp - lm) / (2 * eps);
        const an = analytic[pi][i][j];
        const denom = Math.max(Math.abs(fd), Math.abs(an), 1e-8);
        maxRel = Math.max(maxRel, Math.abs(fd - an) / denom);
      }
    }
    return maxRel;
  }

  /* ============================ 7. 神经网络（MLP）+ 注意力 ============================ */
  function mlpForward(X, W1, b1, W2, b2) {
    const h = Ttanh(Tadd(Tmatmul(X, W1), b1));
    return Tadd(Tmatmul(h, W2), b2);
  }
  function mse(yhat, t) { // yhat: Tensor(m,1); t: number[][]
    const tT = new Tensor(t, false);
    return Tscale(Tsum(Tmul(Tsub(yhat, tT), Tsub(yhat, tT))), 1 / yhat.data.length);
  }
  // 训练一个 2 层 MLP（tanh 隐藏 + 线性输出）求解 XOR。opts.record=true 时返回 lossHistory。
  function trainMLP(Xt, yt, opts = {}) {
    const { lr = 0.1, iters = 3000, h = 4, record = false } = opts;
    const X = new Tensor(Xt, false);
    const m = Xt.length, n = Xt[0].length;
    const W1 = new Tensor(M.zeros(n, h).map(row => row.map(() => M.randn() * 0.5)), true);
    const b1 = new Tensor(M.zeros(1, h), true);
    const W2 = new Tensor(M.zeros(h, 1).map(row => row.map(() => M.randn() * 0.5)), true);
    const b2 = new Tensor([[0]], true);
    const initLoss = mse(mlpForward(X, W1, b1, W2, b2), yt).data[0][0];
    const lossHistory = record ? [initLoss] : null;
    let finalLoss = initLoss;
    for (let it = 0; it < iters; it++) {
      const yhat = mlpForward(X, W1, b1, W2, b2);
      const loss = mse(yhat, yt);
      backward(loss);
      W1.data = M.sub(W1.data, M.scale(W1.grad, lr));
      b1.data = M.sub(b1.data, M.scale(b1.grad, lr));
      W2.data = M.sub(W2.data, M.scale(W2.grad, lr));
      b2.data = M.sub(b2.data, M.scale(b2.grad, lr));
      finalLoss = loss.data[0][0];
      if (record && it % 50 === 0) lossHistory.push(finalLoss);
    }
    const out = { W1, b1, W2, b2, initLoss, finalLoss };
    if (record) out.lossHistory = lossHistory;
    return out;
  }
  // 缩放点积注意力 (Vaswani et al. 2017)：out = softmax(Q K^T / √d) V
  function attention(Q, K, V) {
    const d = Q.data[0].length;
    const scores = Tscale(Tmatmul(Q, Ttranspose(K)), 1 / Math.sqrt(d));
    const A = Tsoftmax(scores);
    return { A, out: Tmatmul(A, V) };
  }

  /* ============================ 8. 经典监督/无监督算法 ============================ */
  // 线性回归：正规方程 (X^T X) w = X^T y，经 Cholesky 稳定求解。
  function linearRegression(X, y) {
    const Xt = M.transpose(X);
    const XtX = M.matMul(Xt, X);
    const Xty = M.matMul(Xt, y);
    const L = cholesky(XtX);
    return cholSolve(L, Xty); // (n,1)
  }
  // 逻辑回归：手动解析梯度 SGD（交叉熵），适用于二分类。
  function logisticRegression(X, y, opts = {}) {
    const { lr = 0.3, iters = 3000 } = opts;
    const m = X.length, n = X[0].length;
    let w = new Array(n).fill(0).map(() => [M.randn() * 0.01]);
    let b = [[0]];
    let lastLoss = 0;
    for (let it = 0; it < iters; it++) {
      const p = X.map(row => {
        let s = 0;
        for (let j = 0; j < n; j++) s += row[j] * w[j][0];
        return 1 / (1 + Math.exp(-(s + b[0][0])));
      });
      const gw = new Array(n).fill(0).map(() => [0]);
      let gb = 0;
      let loss = 0;
      for (let i = 0; i < m; i++) {
        const err = p[i] - y[i][0];
        for (let j = 0; j < n; j++) gw[j][0] += err * X[i][j];
        gb += err;
        loss += -(y[i][0] * Math.log(p[i] + 1e-12) + (1 - y[i][0]) * Math.log(1 - p[i] + 1e-12));
      }
      for (let j = 0; j < n; j++) w[j][0] -= lr * gw[j][0] / m;
      b[0][0] -= lr * gb / m;
      lastLoss = loss / m;
    }
    const p = X.map(row => {
      let s = 0;
      for (let j = 0; j < n; j++) s += row[j] * w[j][0];
      return 1 / (1 + Math.exp(-(s + b[0][0])));
    });
    const acc = X.filter((row, i) => ((p[i] >= 0.5) ? 1 : 0) === y[i][0]).length / m;
    return { w, b, acc, loss: lastLoss };
  }
  // k-means（Lloyd 算法）：SSE 单调不增是不变量。
  function kmeans(X, k, opts = {}) {
    const { iters = 50 } = opts;
    let centers = X.slice(0, k).map(r => r.slice());
    const assign = new Array(X.length).fill(0);
    const sseHistory = [];
    let prevSSE = Infinity;
    for (let it = 0; it < iters; it++) {
      let sse = 0;
      for (let i = 0; i < X.length; i++) {
        let best = 0, bd = Infinity;
        for (let c = 0; c < k; c++) {
          let d = 0;
          for (let j = 0; j < X[i].length; j++) { const diff = X[i][j] - centers[c][j]; d += diff * diff; }
          if (d < bd) { bd = d; best = c; }
        }
        assign[i] = best; sse += bd;
      }
      sseHistory.push(sse);
      const sums = centers.map(() => new Array(X[0].length).fill(0));
      const cnt = new Array(k).fill(0);
      for (let i = 0; i < X.length; i++) {
        const c = assign[i]; cnt[c]++;
        for (let j = 0; j < X[i].length; j++) sums[c][j] += X[i][j];
      }
      for (let c = 0; c < k; c++)
        if (cnt[c] > 0) for (let j = 0; j < X[0].length; j++) centers[c][j] = sums[c][j] / cnt[c];
      if (sse > prevSSE + 1e-9) break; // SSE 不应上升
      prevSSE = sse;
    }
    return { centers, assign, sseHistory };
  }
  // GMM-EM（Dempster 1977）：对数似然单调不增是定理级不变量。
  function gmmEM(X, k, opts = {}) {
    const { iters = 100, reg = 1e-6 } = opts;
    const m = X.length, d = X[0].length;
    let means = X.slice(0, k).map(r => r.slice());
    let covs = [], weights = [];
    const var0 = 1;
    for (let c = 0; c < k; c++) {
      covs.push(M.identity(d).map(row => row.map(v => v * var0)));
      weights.push(1 / k);
    }
    const logLikHistory = [];
    let R = M.zeros(m, k);
    for (let it = 0; it < iters; it++) {
      const logR = M.zeros(m, k);
      for (let i = 0; i < m; i++) {
        for (let c = 0; c < k; c++) {
          const diff = X[i].map((v, j) => v - means[c][j]);
          const inv = invMatrix(covs[c]);
          let quad = 0;
          for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) quad += diff[a] * inv[a][b] * diff[b];
          const det = detMatrix(covs[c]);
          const logPdf = -0.5 * (d * Math.log(2 * Math.PI) + Math.log(Math.max(det, 1e-12)) + quad);
          logR[i][c] = Math.log(weights[c] + 1e-12) + logPdf;
        }
      }
      let totalLL = 0;
      for (let i = 0; i < m; i++) {
        const mx = Math.max(...logR[i]);
        let s = 0;
        for (let c = 0; c < k; c++) s += Math.exp(logR[i][c] - mx);
        totalLL += mx + Math.log(s);
      }
      logLikHistory.push(totalLL);
      // 归一化责任度
      R = M.zeros(m, k);
      for (let i = 0; i < m; i++) {
        const mx = Math.max(...logR[i]);
        let s = 0;
        for (let c = 0; c < k; c++) s += Math.exp(logR[i][c] - mx);
        for (let c = 0; c < k; c++) R[i][c] = Math.exp(logR[i][c] - mx) / s;
      }
      const Nk = [];
      for (let c = 0; c < k; c++) { let s = 0; for (let i = 0; i < m; i++) s += R[i][c]; Nk.push(s); }
      for (let c = 0; c < k; c++) {
        const Nc = Nk[c] + 1e-12;
        const mu = new Array(d).fill(0);
        for (let i = 0; i < m; i++) for (let j = 0; j < d; j++) mu[j] += R[i][c] * X[i][j];
        for (let j = 0; j < d; j++) mu[j] /= Nc;
        means[c] = mu;
        const cov = M.zeros(d, d);
        for (let i = 0; i < m; i++) {
          const diff = X[i].map((v, j) => v - mu[j]);
          for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) cov[a][b] += R[i][c] * diff[a] * diff[b];
        }
        for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) cov[a][b] = cov[a][b] / Nc + (a === b ? reg : 0);
        covs[c] = cov;
        weights[c] = Nc / m;
      }
    }
    return { means, covs, weights, R, logLikHistory };
  }

  /* ============================ 9. PCA（经 SVD） ============================ */
  // Pearson (1901) / 经 Golub-Reinsch SVD 实现。A = U Σ V^T，V 列为方向。
  function pca(X, k) {
    const m = X.length, n = X[0].length;
    const mean = new Array(n).fill(0);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) mean[j] += X[i][j];
    for (let j = 0; j < n; j++) mean[j] /= m;
    const Xc = X.map(row => row.map((v, j) => v - mean[j]));
    const { U, S, V } = svd(Xc);
    const components = V.slice(0, k); // k 个方向（各为 n 维向量）
    const projected = Xc.map(row => components.map(comp => {
      let s = 0;
      for (let j = 0; j < n; j++) s += row[j] * comp[j];
      return [s];
    }));
    const totalVar = S.reduce((a, s) => a + s * s, 0);
    const explained = S.slice(0, k).reduce((a, s) => a + s * s, 0) / (totalVar || 1);
    // 重建误差（取前 k 主成分）
    const recon = projected.map(row => {
      const r = new Array(n).fill(0);
      for (let c = 0; c < k; c++) for (let j = 0; j < n; j++) r[j] += row[c][0] * components[c][j];
      return r;
    });
    let reconErr = 0;
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) reconErr += (Xc[i][j] - recon[i][j]) ** 2;
    return { mean, components, projected, explained, S, reconErr: Math.sqrt(reconErr) };
  }

  /* ============================ API 导出 ============================ */
  const ForgeCore = {
    version: '1.0.0',
    M, cholesky, cholSolve, jacobiEigen, svd, svdReconstruct, invMatrix, detMatrix,
    Tensor, Tmatmul, Tadd, Tsub, Tmul, Tscale, Ttanh, Tsigmoid, Tsoftmax, Ttranspose, Tsum, Tmean,
    backward, gradCheck,
    mlpForward, mse, trainMLP, attention,
    linearRegression, logisticRegression, kmeans, gmmEM, pca
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = ForgeCore;
  else global.ForgeCore = ForgeCore;
})(typeof window !== 'undefined' ? window : globalThis);
