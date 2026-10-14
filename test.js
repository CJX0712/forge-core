/*
 * Forge Core — 自检（8/8 绿）
 * 每个检查对应一个「可验证不变量」：解析解 vs 数值解、正交性、单调不增/不减、梯度检验等。
 * 运行：node test.js
 */
const FC = require('./forge-core.js');

// 可复现随机（mulberry32）
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
Math.random = mulberry32(20261015);

const M = FC.M;
function makeBlobs(n, k, d, spread = 0.3) {
  const centers = [];
  for (let c = 0; c < k; c++) centers.push(Array.from({ length: d }, () => (Math.random() * 2 - 1) * 3));
  const X = [];
  for (let i = 0; i < n; i++) {
    const c = i % k;
    X.push(centers[c].map(v => v + M.randn() * spread));
  }
  return X;
}

/* ---------- 8 项检查 ---------- */
const checks = [];

// 1. SVD：重建误差 + 正交性
checks.push({
  name: 'SVD 重建 + 正交不变量',
  run() {
    const A = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]];
    const { U, S, V } = FC.svd(A);
    const R = FC.svdReconstruct({ U, S, V });
    const err = M.maxAbs(M.sub(A, R));
    const Iu = M.identity(U.length), Iv = M.identity(V.length);
    const ortho = Math.max(M.maxAbs(M.sub(M.matMul(M.transpose(U), U), Iu)),
                           M.maxAbs(M.sub(M.matMul(M.transpose(V), V), Iv)));
    return { pass: err < 1e-6 && ortho < 1e-6, detail: `reconErr=${err.toExponential(2)} orthoErr=${ortho.toExponential(2)}` };
  }
});

// 2. Cholesky + 求解
checks.push({
  name: 'Cholesky 分解与线性求解',
  run() {
    const A = [[4, 1, 0], [1, 3, 1], [0, 1, 2]];
    const L = FC.cholesky(A);
    const b = [[1], [2], [3]];
    const x = FC.cholSolve(L, b);
    const err = M.maxAbs(M.sub(M.matMul(A, x), b));
    return { pass: err < 1e-8, detail: `Ax-b err=${err.toExponential(2)}` };
  }
});

// 3. Jacobi 特征分解：A·v = λ·v 且 V 正交
checks.push({
  name: 'Jacobi 特征值不变量',
  run() {
    const A = [[2, 1, 0], [1, 2, 1], [0, 1, 2]];
    const { values, vectors } = FC.jacobiEigen(A);
    let err = 0;
    for (let i = 0; i < 3; i++) {
      const Av = M.matMul(A, M.colVec(vectors[i]));
      const lv = M.scale(M.colVec(vectors[i]), values[i]);
      err = Math.max(err, M.maxAbs(M.sub(Av, lv)));
    }
    const VtV = M.matMul(M.transpose(vectors), vectors);
    const ortho = M.maxAbs(M.sub(VtV, M.identity(3)));
    return { pass: err < 1e-6 && ortho < 1e-6, detail: `Av-λv=${err.toExponential(2)} ortho=${ortho.toExponential(2)}` };
  }
});

// 4. autograd：MLP 梯度检验（解析 vs 有限差分）
checks.push({
  name: 'autograd MLP 梯度检验',
  run() {
    const Xt = [[0, 0], [0, 1], [1, 0], [1, 1]];
    const yt = [[0], [1], [1], [0]];
    const W1 = new FC.Tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], true);
    const b1 = new FC.Tensor([[0, 0, 0]], true);
    const W2 = new FC.Tensor([[0.2], [-0.1], [0.3]], true);
    const b2 = new FC.Tensor([[0]], true);
    const fwd = () => FC.mse(FC.mlpForward(new FC.Tensor(Xt, false), W1, b1, W2, b2), yt);
    const maxRel = FC.gradCheck(fwd, [W1, b1, W2, b2]);
    return { pass: maxRel < 1e-3, detail: `maxRelErr=${maxRel.toExponential(2)}` };
  }
});

// 5. MLP 训练 XOR：损失显著下降
checks.push({
  name: 'MLP 学习 XOR',
  run() {
    const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
    const y = [[0], [1], [1], [0]];
    const { initLoss, finalLoss } = FC.trainMLP(X, y, { lr: 0.1, iters: 3000, h: 4 });
    const ok = finalLoss < initLoss / 5 && finalLoss < 0.05;
    return { pass: ok, detail: `init=${initLoss.toFixed(3)} final=${finalLoss.toFixed(4)}` };
  }
});

// 6. 注意力梯度检验
checks.push({
  name: 'scaled-dot-product 注意力梯度检验',
  run() {
    const Q = new FC.Tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]], true);
    const K = new FC.Tensor([[0.2, 0.1, 0.4, 0.3], [0.5, 0.6, 0.7, 0.8], [0.1, 0.9, 0.2, 0.4], [0.3, 0.3, 0.3, 0.3], [0.8, 0.2, 0.6, 0.4]], true);
    const V = new FC.Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]], true);
    const fwd = () => FC.Tsum(FC.attention(Q, K, V).out);
    const maxRel = FC.gradCheck(fwd, [Q, K, V]);
    return { pass: maxRel < 1e-3, detail: `maxRelErr=${maxRel.toExponential(2)}` };
  }
});

// 7. k-means(SSE 单调不增) + GMM-EM(对数似然单调不减 + 责任度/权重/协方差不变量)
checks.push({
  name: 'k-means / GMM-EM 单调不变量',
  run() {
    const X = makeBlobs(60, 3, 2, 0.25);
    const km = FC.kmeans(X, 3, { iters: 50 });
    let nonInc = true;
    for (let i = 0; i < km.sseHistory.length - 1; i++)
      if (km.sseHistory[i] < km.sseHistory[i + 1] - 1e-6) nonInc = false;
    const gmm = FC.gmmEM(X, 3, { iters: 60 });
    let nonDec = true;
    for (let i = 0; i < gmm.logLikHistory.length - 1; i++)
      if (gmm.logLikHistory[i] > gmm.logLikHistory[i + 1] + 1e-6) nonDec = false;
    let rSumOK = true;
    for (const row of gmm.R) if (Math.abs(row.reduce((a, b) => a + b, 0) - 1) > 1e-9) rSumOK = false;
    const wSum = Math.abs(gmm.weights.reduce((a, b) => a + b, 0) - 1) < 1e-9;
    let symOK = true;
    for (const cov of gmm.covs)
      for (let a = 0; a < cov.length; a++)
        for (let b = 0; b < cov.length; b++)
          if (Math.abs(cov[a][b] - cov[b][a]) > 1e-12) symOK = false;
    return {
      pass: nonInc && nonDec && rSumOK && wSum && symOK,
      detail: `kmeansSSE↓=${nonInc} gmmLL↑=${nonDec} RΣ=1:${rSumOK} wΣ=1:${wSum} covSym:${symOK}`
    };
  }
});

// 8. PCA(满秩重建≈0) + 线性回归(残差⊥X) + 逻辑回归(可分数据 acc=1)
checks.push({
  name: 'PCA / 线性回归 / 逻辑回归 不变量',
  run() {
    const X = Array.from({ length: 50 }, () => [M.randn(), M.randn()]);
    const p = FC.pca(X, 2);
    // 线性回归：构造 y = X w* + 极小噪声
    const Xl = Array.from({ length: 40 }, () => [M.randn(), M.randn()]);
    const wTrue = [[1.5], [-2.0]];
    const yl = Xl.map(row => [[row[0] * wTrue[0][0] + row[1] * wTrue[1][0] + M.randn() * 1e-9]]);
    const w = FC.linearRegression(Xl, yl);
    const resid = M.sub(yl, M.matMul(Xl, w));
    const perp = M.maxAbs(M.matMul(M.transpose(Xl), resid));
    // 逻辑回归：可分数据 x1 + x2 > 0 → 1
    const Xlog = Array.from({ length: 40 }, () => [M.randn() * 1.5, M.randn() * 1.5]);
    const ylog = Xlog.map(r => [r[0] + r[1] > 0 ? 1 : 0]);
    const lr = FC.logisticRegression(Xlog, ylog, { lr: 0.3, iters: 3000 });
    return {
      pass: p.reconErr < 1e-6 && perp < 1e-6 && lr.acc === 1,
      detail: `pcaRecon=${p.reconErr.toExponential(2)} resid⊥X=${perp.toExponential(2)} logisticAcc=${lr.acc}`
    };
  }
});

/* ---------- 运行 + 汇总 ---------- */
let passN = 0;
console.log('════════ Forge Core 自检 ════════');
for (const c of checks) {
  try {
    const r = c.run();
    if (r.pass) { passN++; console.log(`✅ ${c.name} — ${r.detail}`); }
    else console.log(`⚠️  ${c.name} — ${r.detail}`);
  } catch (e) {
    console.log(`⚠️  ${c.name} — ERROR: ${e.message}`);
  }
}
console.log('────────────────────────────────');
console.log(`结果：${passN}/${checks.length} 绿`);
console.log(passN === checks.length ? '🟢 全部通过' : '🔴 存在失败');
process.exit(passN === checks.length ? 0 : 1);
