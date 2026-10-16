/* Transcript reader ONNX Web demo. The image pipeline mirrors runtime.py. */
'use strict';
const $ = id => document.getElementById(id);
const state = { image:null, stream:null, sessions:null, config:null, busy:false };
const FILES = { roi:'./roi_detector.onnx', grid:'./grid_detector.onnx', reader:'./row_reader.onnx' };
const status = (msg,kind='') => { $('status').textContent=msg; $('status').className='status '+kind; };
function setImage(img){ state.image=img; $('preview').src=img.src || img; $('preview').hidden=false; $('runBtn').disabled=false; $('runBtn').textContent='Chạy nhận diện'; $('resultsPanel').hidden=true; status('Đã nhận ảnh. Nhấn “Chạy nhận diện”.'); }
$('fileInput').addEventListener('change', e=>{const f=e.target.files&&e.target.files[0];if(!f)return;const url=URL.createObjectURL(f);const img=new Image();img.onload=()=>setImage(img);img.onerror=()=>status('Không đọc được ảnh này. Hãy thử JPG hoặc PNG.','error');img.src=url;});
$('cameraBtn').onclick=async()=>{try{state.stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}},audio:false});$('video').srcObject=state.stream;$('cameraArea').hidden=false;status('Camera đã mở. Canh bảng điểm rồi bấm Chụp ảnh.');}catch(e){status('Không mở được camera. Hãy cho phép quyền Camera trong Safari và mở trang qua HTTPS. Chi tiết: '+e.message,'error');}};
$('closeCameraBtn').onclick=()=>stopCamera();function stopCamera(){if(state.stream){state.stream.getTracks().forEach(t=>t.stop());state.stream=null;}$('cameraArea').hidden=true;}
$('captureBtn').onclick=()=>{const v=$('video'),c=$('workCanvas');if(!v.videoWidth){status('Camera chưa sẵn sàng, thử lại sau một chút.','error');return;}c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0);const img=new Image();img.onload=()=>setImage(img);img.src=c.toDataURL('image/jpeg',.94);stopCamera();};
window.addEventListener('pagehide',stopCamera);
async function loadModels(){if(state.sessions)return state.sessions;if(!window.ort)throw new Error('Không thể tải dữ liệu cần thiết. Vui lòng kiểm tra kết nối mạng và thử lại.');ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;status('Đang chuẩn bị, vui lòng đợi…');const opts={executionProviders:['wasm'],graphOptimizationLevel:'all'};const s={};for(const [k,path] of Object.entries(FILES)){ $('progress').textContent='Đang tải dữ liệu…';s[k]=await ort.InferenceSession.create(path,opts); }state.sessions=s;$('progress').textContent='Đã chuẩn bị xong.';return s;}
function canvasFor(img,w,h,fit=false){const c=document.createElement('canvas');c.width=w;c.height=h;const ctx=c.getContext('2d',{willReadFrequently:true});ctx.fillStyle='rgb(238,238,238)';ctx.fillRect(0,0,w,h);if(fit){const scale=Math.min((w-8)/Math.max(img.width,1),(h-8)/Math.max(img.height,1));const rw=Math.max(1,Math.round(img.width*scale)),rh=Math.max(1,Math.round(img.height*scale));ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality=scale>1?'high':'low';ctx.drawImage(img,Math.floor((w-rw)/2),Math.floor((h-rh)/2),rw,rh);}else{ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';ctx.drawImage(img,0,0,w,h);}return c;}
function tensorFromCanvas(c){const d=c.getContext('2d',{willReadFrequently:true}).getImageData(0,0,c.width,c.height).data;const plane=c.width*c.height,out=new Float32Array(3*plane);for(let i=0;i<plane;i++){out[i]=(d[i*4]/255-.5)/.5;out[plane+i]=(d[i*4+1]/255-.5)/.5;out[2*plane+i]=(d[i*4+2]/255-.5)/.5;}return new ort.Tensor('float32',out,[1,3,c.height,c.width]);}
function prep(img,w,h,fit=false){return tensorFromCanvas(canvasFor(img,w,h,fit));}
function sigmoid(x){return 1/(1+Math.exp(-Math.max(-30,Math.min(30,x))));}
function findComponents(mask,w,h){const parent=[],runs=[],prev=[];let previous=[];for(let y=0;y<h;y++){let xs=[];for(let x=0;x<w;x++)if(mask[y*w+x])xs.push(x);const cur=[];if(xs.length){let groups=[],g=[xs[0]];for(let i=1;i<xs.length;i++){if(xs[i]===xs[i-1]+1)g.push(xs[i]);else{groups.push(g);g=[xs[i]];}}groups.push(g);for(const gr of groups){const idx=parent.length;parent.push(idx);const r=[gr[0],y,gr[gr.length-1],idx];runs.push(r);cur.push(r);for(const old of previous){if(old[2]+1<r[0])continue;if(r[2]+1<old[0])break;let a=idx,b=old[3];while(parent[a]!==a)a=parent[a];while(parent[b]!==b)b=parent[b];if(a!==b)parent[b]=a;}}}previous=cur;}
const found=new Map();for(const [x0,y,x1,idx] of runs){let root=idx;while(parent[root]!==root)root=parent[root];parent[idx]=root;if(!found.has(root))found.set(root,[x0,y,x1+1,y+1]);const b=found.get(root);b[0]=Math.min(b[0],x0);b[1]=Math.min(b[1],y);b[2]=Math.max(b[2],x1+1);b[3]=Math.max(b[3],y+1);}return [...found.values()];}
function roiMorphology(binary,w,h){ // Approximate PIL MaxFilter(3) then MinFilter(3) (closing)
 const dil=new Uint8Array(binary.length),out=new Uint8Array(binary.length);for(let y=0;y<h;y++)for(let x=0;x<w;x++){let v=0;for(let yy=Math.max(0,y-1);yy<=Math.min(h-1,y+1);yy++)for(let xx=Math.max(0,x-1);xx<=Math.min(w-1,x+1);xx++)if(binary[yy*w+xx])v=1;dil[y*w+x]=v;}for(let y=0;y<h;y++)for(let x=0;x<w;x++){let v=1;for(let yy=Math.max(0,y-1);yy<=Math.min(h-1,y+1);yy++)for(let xx=Math.max(0,x-1);xx<=Math.min(w-1,x+1);xx++)if(!dil[yy*w+xx])v=0;out[y*w+x]=v;}return out;}
function outputByName(result,preferred){if(result[preferred])return result[preferred];return result[Object.keys(result)[0]];}
async function findRois(img,s){const t=prep(img,384,512);const res=await s.roi.run({image:t});const ten=outputByName(res,'roi_logits'),a=ten.data,shape=ten.dims;const h=shape[2],w=shape[3],mask0=new Uint8Array(w*h);for(let i=0;i<w*h;i++)mask0[i]=sigmoid(a[i])>=.5?1:0;const mask=roiMorphology(mask0,w,h),components=findComponents(mask,w,h),sx=img.width/w,sy=img.height/h;return components.filter(b=>(b[2]-b[0])*(b[3]-b[1])>=.008*w*h).map(b=>({bbox:[Math.round(b[0]*sx),Math.round(b[1]*sy),Math.round(b[2]*sx),Math.round(b[3]*sy)]})).sort((a,b)=>a.bbox[1]-b.bbox[1]||a.bbox[0]-b.bbox[0]);}
function peaks(values,threshold,minGap){if(!values.length||Math.max(...values)<threshold)return[];const p=[];for(let i=0;i<values.length;i++){const get=j=>values[Math.max(0,Math.min(values.length-1,j))];p.push((get(i-2)+4*get(i-1)+6*get(i)+4*get(i+1)+get(i+2))/16);}const max=Math.max(...p),cand=[];for(let i=1;i<p.length-1;i++)if(p[i]>=p[i-1]&&p[i]>=p[i+1]&&p[i]>=Math.max(threshold,max*.28))cand.push(i);const chosen=[];cand.sort((a,b)=>p[b]-p[a]);for(const x of cand)if(chosen.every(y=>Math.abs(x-y)>=minGap))chosen.push(x);return chosen.sort((a,b)=>a-b);}

function orderedColumns(scores, minGap) {
  if (!Array.isArray(scores) || scores.length !== 5) {
    throw new Error(
      `Model lưới phải trả về 5 đường biên dọc, nhưng nhận được ${scores?.length ?? 0}.`
    );
  }

  const width = scores[0]?.length ?? 0;

  if (width < scores.length) {
    throw new Error(
      `Đầu ra model quá nhỏ để tìm 5 cột (chiều rộng=${width}).`
    );
  }

  if (scores.some(row => !row || row.length !== width)) {
    throw new Error("Kích thước các đường biên dọc không đồng nhất.");
  }

  // Thay NaN và Infinity bằng điểm thấp để tránh làm hỏng thuật toán.
  const clean = scores.map(row =>
    Array.from(row, v => Number.isFinite(v) ? v : -1e6)
  );

  // Không ép khoảng cách quá lớn khiến không thể tìm đủ 5 đường.
  const gap = Math.max(
    1,
    Math.min(
      Math.floor(Number.isFinite(minGap) ? minGap : 1),
      Math.floor((width - 1) / (scores.length - 1))
    )
  );

  let dp = Float64Array.from(clean[0]);
  const backs = [];

  for (let r = 1; r < clean.length; r++) {
    const cur = new Float64Array(width);
    cur.fill(-Infinity);

    const par = new Int32Array(width);
    par.fill(-1);

    let best = -Infinity;
    let bestIndex = -1;

    for (let x = 0; x < width; x++) {
      const j = x - gap;

      if (j >= 0 && dp[j] > best) {
        best = dp[j];
        bestIndex = j;
      }

      if (bestIndex >= 0 && Number.isFinite(best)) {
        cur[x] = best + clean[r][x];
        par[x] = bestIndex;
      }
    }

    if (!Array.from(cur).some(Number.isFinite)) {
      throw new Error(
        "Không xác định được các cột điểm. Hãy thử ảnh rõ hơn."
      );
    }

    backs.push(par);
    dp = cur;
  }

  let x = 0;
  for (let i = 1; i < width; i++) {
    if (dp[i] > dp[x]) x = i;
  }

  if (!Number.isFinite(dp[x])) {
    throw new Error("Model không tạo được điểm số hợp lệ cho ranh giới cột.");
  }

  const positions = [x];

  for (let k = backs.length - 1; k >= 0; k--) {
    const previous = backs[k][positions[positions.length - 1]];

    if (previous < 0) {
      throw new Error("Không truy ngược được vị trí các cột.");
    }

    positions.push(previous);
  }

  return positions.reverse();
}
function parseGrid(logits,shape,iw,ih){const [n,ch,mh,mw]=shape,raw=logits;const prob=(c,y,x)=>sigmoid(raw[c*mh*mw+y*mw+x]);const yp=[];for(let y=0;y<mh;y++){let sum=0;for(let x=0;x<mw;x++)sum+=prob(0,y,x);yp.push(sum/mw);}let ys=[];for(const th of [.55,.40,.30,.22,.16]){ys=peaks(yp,th,Math.max(1,Math.floor(mh*.008)));if(ys.length>=4)break;}if(ys.length<4)throw Error(`Mô hình lưới chỉ tìm được ${ys.length} đường ngang.`);const profiles=[];for(let c=1;c<6;c++){const row=[];for(let x=0;x<mw;x++){let sum=0;for(let y=ys[0];y<=ys[ys.length-1];y++)sum+=raw[c*mh*mw+y*mw+x];row.push(sum/(ys[ys.length]-ys[0]+1));}profiles.push(row);}const xs=orderedColumns(profiles,Math.max(1,Math.floor(mw*.015))).map(x=>Math.max(0,Math.min(iw-1,Math.round(x*iw/mw))));const rows=[];for(let i=0;i<ys.length-1;i++){const ya=Math.max(0,Math.round(ys[i]*ih/mh)),yb=Math.min(ih,Math.round(ys[i+1]*ih/mh));if(yb<=ya)continue;rows.push({bbox:[xs[0],ya,xs[4],yb],cell_boxes:[0,1,2,3].map(j=>[xs[j],ya,xs[j+1],yb])});}return rows;}
function cropToImage(img,b){const c=document.createElement('canvas');c.width=Math.max(1,b[2]-b[0]);c.height=Math.max(1,b[3]-b[1]);c.getContext('2d').drawImage(img,b[0],b[1],c.width,c.height,0,0,c.width,c.height);const out=new Image();out.src=c.toDataURL('image/png');return new Promise((resolve,reject)=>{out.onload=()=>resolve(out);out.onerror=reject;});}
async function findRows(img,rois,s){const all=[];for(let ri=0;ri<rois.length;ri++){const [x0,y0,x1,y1]=rois[ri].bbox;const cx0=x0,cy0=y0,cx1=x1,cy1=y1;const crop=await cropToImage(img,[cx0,cy0,cx1,cy1]);const t=prep(crop,384,512);const res=await s.grid.run({roi:t});const ten=outputByName(res,'grid_logits');const rows=parseGrid(ten.data,ten.dims,crop.width,crop.height);for(const r of rows){const boxes=r.cell_boxes.map(b=>[b[0]+cx0,b[1]+cy0,b[2]+cx0,b[3]+cy0]);boxes[0][0]=x0;boxes[3][2]=x1;for(const b of boxes){b[0]=Math.max(x0,Math.min(x1,b[0]));b[2]=Math.max(x0,Math.min(x1,b[2]));b[1]=Math.max(y0,Math.min(y1,b[1]));b[3]=Math.max(y0,Math.min(y1,b[3]));
if (valid) {
  all.push({
    bbox: [
      x0,
      Math.max(y0, r.bbox[1] + cy0),
      x1,
      Math.min(y1, r.bbox[3] + cy0)
    ],
    cell_boxes: boxes,
    roi_index: ri
  });
};}}return all.sort((a,b)=>a.bbox[1]-b.bbox[1]||a.bbox[0]-b.bbox[0]);}
function softmax(arr){let max=-Infinity;for(const x of arr)if(x>max)max=x;const e=Array.from(arr,x=>Math.exp(x-max)),sum=e.reduce((a,b)=>a+b,0);return e.map(x=>x/sum);}
function top3(p,names){return p.map((probability,i)=>({label:names[i]??String(i),probability})).sort((a,b)=>b.probability-a.probability).slice(0,3);}
function decode(outputs,idx,cfg){const names=[cfg.subject_classes,cfg.score_classes,cfg.score_classes,cfg.score_classes];const probs=outputs.map(o=>{const per=o.dims.slice(1).reduce((a,b)=>a*b,1);return softmax(o.data.slice(idx*per,(idx+1)*per));});const raw=probs.map(p=>p.indexOf(Math.max(...p))),chosen=raw.slice();let constrained=false;if(raw.slice(1).every(i=>i<101)){let best=-Infinity,ia=0,ib=0,ic=0;for(let a=0;a<101;a++)for(let b=0;b<101;b++){const c=Math.floor((a+2*b+1)/3);const score=Math.log(Math.max(probs[1][a],1e-12))+Math.log(Math.max(probs[2][b],1e-12))+Math.log(Math.max(probs[3][c],1e-12));if(score>best){best=score;ia=a;ib=b;ic=c;}}chosen.splice(1,3,ia,ib,ic);constrained=true;}const out={};['subject','HK1','HK2','CN'].forEach((key,k)=>{const i=chosen[k];out[key]={prediction:names[k][i]??String(i),raw_prediction:names[k][raw[k]]??String(raw[k]),confidence:probs[k][i],constraint_applied:constrained&&k>0,top3:top3(probs[k],names[k])};});return out;}
async function classifyRows(img,rows,s,cfg){const tensors=[];for(const row of rows){const crop=await cropToImage(img,[Math.max(0,row.bbox[0]),Math.max(0,row.bbox[1]),Math.min(img.width,row.bbox[2]),Math.min(img.height,row.bbox[3])]);tensors.push(tensorFromCanvas(canvasFor(crop,384,96,true)).data);}const data=new Float32Array(tensors.length*3*96*384);tensors.forEach((t,i)=>data.set(t,i*t.length));const session=s.reader;const inputName=session.inputNames[0];const batch=new ort.Tensor('float32',data,[tensors.length,3,96,384]);const result=await session.run({[inputName]:batch});const outputs=session.outputNames.map(n=>result[n]);if(outputs.length<4)throw Error('Model reader không trả về đủ 4 đầu ra.');return rows.map((_,i)=>decode(outputs,i,cfg));}
async function runPrediction(){if(!state.image||state.busy)return;state.busy=true;$('runBtn').disabled=true;$('runBtn').textContent='Đang xử lý…';$('resultsPanel').hidden=true;const start=performance.now();try{const s=await loadModels();status('Đang kiểm tra ảnh…');const rois=await findRois(state.image,s);if(!rois.length)throw Error('Không tìm thấy bảng điểm trong ảnh. Hãy chọn ảnh rõ hơn và chụp trọn bảng.');status('Đang đọc các hàng điểm…');const rows=await findRows(state.image,rois,s);if(!rows.length)throw Error('Không đọc được các hàng điểm. Hãy thử ảnh rõ hơn, ít nghiêng hơn.');status('Đang hoàn tất kết quả…');const heads=await classifyRows(state.image,rows,s,state.config);const results=rows.map((r,i)=>({row:i+1,bbox:r.bbox,cell_boxes:r.cell_boxes,heads:heads[i]}));state.lastResult={filename:'browser-upload',runtime:'ONNX Runtime Web WASM (local)',roi_count:rois.length,rows_detected:results.length,elapsed_ms:Math.round(performance.now()-start),rows:results};renderResults(state.lastResult);status('Đã hoàn tất. Vui lòng kiểm tra lại kết quả bên dưới.','ok');$('progress').textContent='';}catch(e){console.error(e);status('Chưa chạy được: '+(e&&e.message?e.message:String(e)),'error');}finally{state.busy=false;$('runBtn').disabled=!state.image;$('runBtn').textContent='Chạy nhận diện';}}
function renderResults(r){$('resultsPanel').hidden=false;$('summary').textContent=`Số vùng bảng: ${r.roi_count} · Số hàng nhận diện: ${r.rows_detected} · Thời gian: ${(r.elapsed_ms/1000).toFixed(2)} giây`;const body=$('resultRows');body.replaceChildren();for(const row of r.rows){const tr=document.createElement('tr');const vals=[row.row,row.heads.subject.prediction,row.heads.HK1.prediction,row.heads.HK2.prediction,row.heads.CN.prediction];for(const v of vals){const td=document.createElement('td');td.textContent=v;tr.appendChild(td);}body.appendChild(tr);}$('rawResults').textContent=JSON.stringify(r,null,2);$('resultsPanel').scrollIntoView({behavior:'smooth',block:'start'});}
$('runBtn').onclick=runPrediction;$('downloadBtn').onclick=()=>{if(!state.lastResult)return;const blob=new Blob([JSON.stringify(state.lastResult,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='ket_qua_doc_diem.json';a.click();URL.revokeObjectURL(url);};
(async()=>{try{const r=await fetch('config.json');if(!r.ok)throw Error('Không tải được config.json');state.config=await r.json();status('Sẵn sàng. Chọn ảnh hoặc mở camera để bắt đầu.');}catch(e){status('Không tải được config.json. Kiểm tra đã upload file config.json ở thư mục gốc chưa.','error');}})();
