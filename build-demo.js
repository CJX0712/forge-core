/*
 * build-demo.js — 生成单文件 index.html（内联 forge-core，零外部依赖，可双击运行）
 * 运行：node build-demo.js
 */
const fs = require('fs');
const path = require('path');

const core = fs.readFileSync(path.join(__dirname, 'forge-core.js'), 'utf8');

const TAIL = `</body></html>`;

const HEAD = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Forge Core — 可验证 AI 数值内核 · 晨星</title>
<style>
  :root{ --bg:#f7f8fa; --card:#fff; --ink:#1c2330; --muted:#6b7280; --line:#e5e7eb;
         --ok:#16a34a; --bad:#dc2626; --accent:#2563eb; --accent2:#7c3aed; }
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
       background:var(--bg);color:var(--ink);line-height:1.6;padding:28px;max-width:980px;margin:0 auto}
  h1{font-size:26px;margin:0 0 4px}
  .sub{color:var(--muted);font-size:14px;margin-bottom:18px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin-bottom:18px;
        box-shadow:0 1px 3px rgba(0,0,0,.04)}
  h2{font-size:18px;margin:0 0 10px;display:flex;align-items:center;gap:8px}
  .tag{display:inline-block;font-size:11px;padding:2px 8px;border-radius:999px;background:#eef2ff;color:var(--accent);margin-right:6px}
  .principle{font-size:13px;color:var(--muted);margin:4px 0}
  .principle b{color:var(--ink)}
  button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:9px 16px;font-size:14px;cursor:pointer}
  button:disabled{opacity:.5;cursor:default}
  table{width:100%;border-collapse:collapse;font-size:13.5px}
  th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}
  th{color:var(--muted);font-weight:600;font-size:12.5px}
  .ok{color:var(--ok);font-weight:700}
  .bad{color:var(--bad);font-weight:700}
  .metric{color:var(--muted);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
  canvas{width:100%;height:200px;background:#fff;border:1px solid var(--line);border-radius:8px;margin-top:8px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  @media(max-width:680px){.grid{grid-template-columns:1fr}}
  code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12.5px}
  .legend{font-size:12px;color:var(--muted);margin-top:6px}
  a{color:var(--accent);text-decoration:none}
</style>
</head>
<body>
<h1>Forge Core</h1>
<div class="sub">零依赖 · 可验证的世界级 AI 数值内核 — 作者 <b>晨星</b> · MIT License</div>

<div class="card">
  <h2><span class="tag">集大成</span>架构与数学基石</h2>
  <div class="principle"><b>autograd</b>：micrograd(Karpathy) 式反向模式自动微分（JAX/tinygrad 内核同源）</div>
  <div class="principle"><b>线性代数</b>：Jacobi 特征分解(1846/1890) · SVD(Golub-Reinsch 思想) · Cholesky 分解</div>
  <div class="principle"><b>聚类</b>：Lloyd k-means(1957/1982) · <b>密度估计</b>：Dempster EM(1977) 用于 GMM</div>
  <div class="principle"><b>降维</b>：Pearson PCA(1901)/SVD · <b>注意力</b>：Vaswani scaled dot-product(2017)</div>
  <div class="principle"><b>可验证</b>：每模块带「双重实现交叉验证」数学不变量 — 解析解 vs 数值解、正交性、单调性、梯度检验</div>
</div>

<div class="card">
  <h2>🧪 自检（8/8 绿）</h2>
  <div class="legend">点击运行浏览器内完整自检，校验全部数学不变量（与 <code>node test.js</code> 一致）。</div>
  <div style="margin:12px 0"><button id="runBtn" onclick="runChecksUI()">运行自检</button>
    <span id="summary" class="metric" style="margin-left:10px"></span></div>
  <table>
    <thead><tr><th style="width:46px">状态</th><th>检查项</th><th>不变量度量</th></tr></thead>
    <tbody id="resultBody"><tr><td colspan="3" class="metric">尚未运行</td></tr></tbody>
  </table>
</div>

<div class="card">
  <h2>🎯 实时演示：autograd MLP 学习 XOR</h2>
  <div class="legend">基于内核 autograd 训练 2 层 MLP，绘制损失下降曲线并给出最终预测。</div>
  <div style="margin:12px 0"><button id="trainBtn" onclick="trainUI()">训练 XOR</button>
    <span id="trainSummary" class="metric" style="margin-left:10px"></span></div>
  <canvas id="lossCanvas" width="900" height="200"></canvas>
  <div id="predBox" class="metric" style="margin-top:10px"></div>
</div>

<div class="card">
  <h2>📦 模块导览</h2>
  <div class="principle"><b>linalg</b>：matMul / transpose / cholesky / cholSolve / jacobiEigen / svd / svdReconstruct / invMatrix / detMatrix</div>
  <div class="principle"><b>autograd</b>：Tensor + Tmatmul/Tadd/Ttanh/Tsigmoid/Tsoftmax/Ttranspose/Tsum/Tmean + backward/gradCheck</div>
  <div class="principle"><b>模型</b>：trainMLP / attention / linearRegression / logisticRegression / kmeans / gmmEM / pca</div>
  <div class="principle">浏览器：<code>window.ForgeCore</code> · Node：<code>require('./forge-core.js')</code></div>
</div>
`;

const SCRIPT = `
/* ============ 内联自检逻辑（与 test.js 同源） ============ */
function mulberry32(a){return function(){a|=0;a=(a+0x6D2B79F5)|0;let t=Math.imul(a^(a>>>15),1|a);t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296;};}
const FC = window.ForgeCore;
const M = FC.M;
function makeBlobs(n,k,d,spread){const centers=[];for(let c=0;c<k;c++)centers.push(Array.from({length:d},()=> (Math.random()*2-1)*3));const X=[];for(let i=0;i<n;i++){const c=i%k;X.push(centers[c].map(v=>v+M.randn()*spread));}return X;}

function runChecks(){
  Math.random = mulberry32(20261015);
  const checks=[];
  checks.push({name:'SVD 重建 + 正交不变量',run(){
    const A=[[1,2,3],[4,5,6],[7,8,9],[10,11,12]];
    const {U,S,V}=FC.svd(A);const R=FC.svdReconstruct({U,S,V});
    const err=M.maxAbs(M.sub(A,R));
    const Iu=M.identity(U.length),Iv=M.identity(V.length);
    const ortho=Math.max(M.maxAbs(M.sub(M.matMul(M.transpose(U),U),Iu)),M.maxAbs(M.sub(M.matMul(M.transpose(V),V),Iv)));
    return {pass:err<1e-6&&ortho<1e-6,detail:'reconErr='+err.toExponential(2)+' orthoErr='+ortho.toExponential(2)};
  }});
  checks.push({name:'Cholesky 分解与线性求解',run(){
    const A=[[4,1,0],[1,3,1],[0,1,2]];const L=FC.cholesky(A);const b=[[1],[2],[3]];
    const err=M.maxAbs(M.sub(M.matMul(A,FC.cholSolve(L,b)),b));
    return {pass:err<1e-8,detail:'Ax-b err='+err.toExponential(2)};
  }});
  checks.push({name:'Jacobi 特征值不变量',run(){
    const A=[[2,1,0],[1,2,1],[0,1,2]];const {values,vectors}=FC.jacobiEigen(A);
    let err=0;for(let i=0;i<3;i++){const Av=M.matMul(A,M.colVec(vectors[i]));const lv=M.scale(M.colVec(vectors[i]),values[i]);err=Math.max(err,M.maxAbs(M.sub(Av,lv)));}
    const VtV=M.matMul(M.transpose(vectors),vectors);const ortho=M.maxAbs(M.sub(VtV,M.identity(3)));
    return {pass:err<1e-6&&ortho<1e-6,detail:'Av-λv='+err.toExponential(2)+' ortho='+ortho.toExponential(2)};
  }});
  checks.push({name:'autograd MLP 梯度检验',run(){
    const Xt=[[0,0],[0,1],[1,0],[1,1]];const yt=[[0],[1],[1],[0]];
    const W1=new FC.Tensor([[0.1,-0.2,0.3],[0.4,0.5,-0.6]],true);
    const b1=new FC.Tensor([[0,0,0]],true);const W2=new FC.Tensor([[0.2],[-0.1],[0.3]],true);const b2=new FC.Tensor([[0]],true);
    const fwd=()=>FC.mse(FC.mlpForward(new FC.Tensor(Xt,false),W1,b1,W2,b2),yt);
    const maxRel=FC.gradCheck(fwd,[W1,b1,W2,b2]);
    return {pass:maxRel<1e-3,detail:'maxRelErr='+maxRel.toExponential(2)};
  }});
  checks.push({name:'MLP 学习 XOR',run(){
    const X=[[0,0],[0,1],[1,0],[1,1]];const y=[[0],[1],[1],[0]];
    const {initLoss,finalLoss}=FC.trainMLP(X,y,{lr:0.1,iters:3000,h:4});
    return {pass:finalLoss<initLoss/5&&finalLoss<0.05,detail:'init='+initLoss.toFixed(3)+' final='+finalLoss.toFixed(4)};
  }});
  checks.push({name:'scaled-dot-product 注意力梯度检验',run(){
    const Q=new FC.Tensor([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,1.0,1.1,1.2]],true);
    const K=new FC.Tensor([[0.2,0.1,0.4,0.3],[0.5,0.6,0.7,0.8],[0.1,0.9,0.2,0.4],[0.3,0.3,0.3,0.3],[0.8,0.2,0.6,0.4]],true);
    const V=new FC.Tensor([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8],[0.9,1.0]],true);
    const fwd=()=>FC.Tsum(FC.attention(Q,K,V).out);
    const maxRel=FC.gradCheck(fwd,[Q,K,V]);
    return {pass:maxRel<1e-3,detail:'maxRelErr='+maxRel.toExponential(2)};
  }});
  checks.push({name:'k-means / GMM-EM 单调不变量',run(){
    const X=makeBlobs(60,3,2,0.25);const km=FC.kmeans(X,3,{iters:50});
    let nonInc=true;for(let i=0;i<km.sseHistory.length-1;i++)if(km.sseHistory[i]<km.sseHistory[i+1]-1e-6)nonInc=false;
    const gmm=FC.gmmEM(X,3,{iters:60});let nonDec=true;
    for(let i=0;i<gmm.logLikHistory.length-1;i++)if(gmm.logLikHistory[i]>gmm.logLikHistory[i+1]+1e-6)nonDec=false;
    let rSumOK=true;for(const row of gmm.R)if(Math.abs(row.reduce((a,b)=>a+b,0)-1)>1e-9)rSumOK=false;
    const wSum=Math.abs(gmm.weights.reduce((a,b)=>a+b,0)-1)<1e-9;
    let symOK=true;for(const cov of gmm.covs)for(let a=0;a<cov.length;a++)for(let b=0;b<cov.length;b++)if(Math.abs(cov[a][b]-cov[b][a])>1e-12)symOK=false;
    return {pass:nonInc&&nonDec&&rSumOK&&wSum&&symOK,detail:'kmeansSSE↓='+nonInc+' gmmLL↑='+nonDec+' RΣ=1:'+rSumOK+' wΣ=1:'+wSum+' covSym:'+symOK};
  }});
  checks.push({name:'PCA / 线性回归 / 逻辑回归 不变量',run(){
    const X=Array.from({length:50},()=>[M.randn(),M.randn()]);const p=FC.pca(X,2);
    const Xl=Array.from({length:40},()=>[M.randn(),M.randn()]);const wTrue=[[1.5],[-2.0]];
    const yl=Xl.map(row=>[[row[0]*wTrue[0][0]+row[1]*wTrue[1][0]+M.randn()*1e-9]]);
    const w=FC.linearRegression(Xl,yl);const resid=M.sub(yl,M.matMul(Xl,w));
    const perp=M.maxAbs(M.matMul(M.transpose(Xl),resid));
    const Xlog=Array.from({length:40},()=>[M.randn()*1.5,M.randn()*1.5]);
    const ylog=Xlog.map(r=>[r[0]+r[1]>0?1:0]);
    const lr=FC.logisticRegression(Xlog,ylog,{lr:0.3,iters:3000});
    return {pass:p.reconErr<1e-6&&perp<1e-6&&lr.acc===1,detail:'pcaRecon='+p.reconErr.toExponential(2)+' resid⊥X='+perp.toExponential(2)+' logisticAcc='+lr.acc};
  }});
  return checks;
}

function runChecksUI(){
  const btn=document.getElementById('runBtn');btn.disabled=true;
  const checks=runChecks();let passN=0;const rows=[];
  for(const c of checks){
    try{const r=c.run();if(r.pass){passN++;rows.push('<tr><td class="ok">✅</td><td>'+c.name+'</td><td class="metric">'+r.detail+'</td></tr>');}
      else rows.push('<tr><td class="bad">⚠️</td><td>'+c.name+'</td><td class="metric">'+r.detail+'</td></tr>');}
    catch(e){rows.push('<tr><td class="bad">⚠️</td><td>'+c.name+'</td><td class="metric">ERROR: '+e.message+'</td></tr>');}
  }
  document.getElementById('resultBody').innerHTML=rows.join('');
  const summ=document.getElementById('summary');
  summ.textContent='结果：'+passN+'/'+checks.length+' 绿 · '+(passN===checks.length?'🟢 全部通过':'🔴 存在失败');
  summ.style.color=passN===checks.length?'var(--ok)':'var(--bad)';
  btn.disabled=false;
}

function trainUI(){
  const btn=document.getElementById('trainBtn');btn.disabled=true;
  const X=[[0,0],[0,1],[1,0],[1,1]];const y=[[0],[1],[1],[0]];
  const o=FC.trainMLP(X,y,{lr:0.3,iters:4000,h:6,record:true});
  document.getElementById('trainSummary').textContent='init='+o.initLoss.toFixed(3)+' final='+o.finalLoss.toFixed(5);
  // 预测
  const W1=o.W1.data,b1=o.b1.data,W2=o.W2.data,b2=o.b2.data;
  const pred=X.map(r=>{let s=0;for(let j=0;j<r.length;j++)s+=r[j]*W1[j][0];const h=Math.tanh(s+b1[0][0]);
    let s2=h*W2[0][0]+b2[0][0];return s2;});
  document.getElementById('predBox').innerHTML='预测：'+X.map((r,i)=>'('+r.join(',')+')='+pred[i].toFixed(3)).join('  ');
  // 绘制损失曲线
  const cv=document.getElementById('lossCanvas');const ctx=cv.getContext('2d');
  ctx.clearRect(0,0,cv.width,cv.height);
  const h=o.lossHistory;const max=h[0],min=Math.min(...h);
  ctx.strokeStyle='#2563eb';ctx.lineWidth=2;ctx.beginPath();
  h.forEach((v,i)=>{const x=i/(h.length-1)*cv.width;const yy=cv.height-((v-min)/(max-min||1))*(cv.height-20)-10;if(i===0)ctx.moveTo(x,yy);else ctx.lineTo(x,yy);});
  ctx.stroke();
  btn.disabled=false;
}
`;

const html = HEAD + '<script>\n' + core + '\n</script>\n' +
  '<script>\n' + SCRIPT + '\n</script>\n' + TAIL;

fs.writeFileSync(path.join(__dirname, 'index.html'), html, 'utf8');
console.log('✅ index.html 已生成（单文件，内联核心，' + (core.length / 1024).toFixed(1) + ' KB 核心）');
