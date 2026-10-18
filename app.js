'use strict';

const $ = (id) => document.getElementById(id);
const MODEL_URL = "https://huggingface.co/nhphuc1411/hocba/resolve/main/unified_document_pipeline.onnx";
const DOC_CLASSES = ['Bằng tốt nghiệp', 'CCCD', 'ĐTUT', 'Giấy xác nhận', 'Học bạ lớp 10', 'Học bạ lớp 11', 'Học bạ lớp 12', 'Bìa học bạ', 'VSAT'];
const DOC_KEYS = ['bangtotnghiep', 'cccd', 'dtut', 'gxn', 'hocba10', 'hocba11', 'hocba12', 'hocbabia', 'vsat'];
const GRADE_KEYS = ['hocba10', 'hocba11', 'hocba12'];
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const SIZE = { doc: 320, page: 512, base: 192, detailW: 384, detailH: 192 };

let session = null;
let modelLoadMs = 0;
let currentImage = null;
let cameraStream = null;
let busy = false;

function setStatus(text, kind = '') {
  $('modelStatus').textContent = text;
  $('modelStatus').className = `status ${kind}`;
}
function showError(message) {
  $('error').textContent = message;
  $('error').hidden = false;
}
function clearError() { $('error').hidden = true; $('error').textContent = ''; }

async function loadModel() {
  try {
    if (!window.ort) throw new Error('Không tải được ONNX Runtime Web. Kiểm tra kết nối mạng/CDN.');
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
    // WASM CPU giúp so sánh tốc độ giữa các máy theo cùng một backend.
    ort.env.wasm.numThreads = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1));
    const start = performance.now();
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });
    modelLoadMs = performance.now() - start;
    $('loadTime').textContent = `${(modelLoadMs / 1000).toFixed(2)} giây`;
    setStatus('Model đã sẵn sàng', 'ready');
    $('runButton').disabled = !currentImage;
  } catch (e) {
    setStatus('Lỗi tải model', 'bad');
    showError(`Không tải được model: ${e.message}. Hãy kiểm tra file model và mở trang qua GitHub Pages/HTTPS.`);
  }
}

function makeCanvas(width, height) {
  const c = document.createElement('canvas');
  c.width = width; c.height = height;
  return c;
}
function letterbox(source, width, height = width) {
  const c = makeCanvas(width, height);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = 'rgb(255,255,255)'; ctx.fillRect(0, 0, width, height);
  const scale = Math.min(width / source.width, height / source.height);
  const rw = Math.max(1, Math.round(source.width * scale));
  const rh = Math.max(1, Math.round(source.height * scale));
  const left = Math.floor((width - rw) / 2);
  const top = Math.floor((height - rh) / 2);
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, source.width, source.height, left, top, rw, rh);
  return { canvas: c, left, top, scale };
}
function tensorFromCanvas(canvas) {
  const { data } = canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, canvas.width, canvas.height);
  const hw = canvas.width * canvas.height;
  const out = new Float32Array(3 * hw);
  for (let p = 0; p < hw; p++) {
    out[p] = (data[p * 4] / 255 - MEAN[0]) / STD[0];
    out[hw + p] = (data[p * 4 + 1] / 255 - MEAN[1]) / STD[1];
    out[2 * hw + p] = (data[p * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  return new ort.Tensor('float32', out, [1, 3, canvas.height, canvas.width]);
}
function zeroTensor(h, w = h) {
  return new ort.Tensor('float32', new Float32Array(3 * h * w), [1, 3, h, w]);
}
function getOutputByIndex(outputs, index) {
  const name = session.outputNames[index];
  if (!name || !outputs[name]) throw new Error(`Không tìm thấy output số ${index} (${name || 'unknown'}).`);
  return outputs[name].data;
}
async function runStage(feeds, indices) {
  const names = indices.map(i => session.outputNames[i]);
  const outputs = await session.run(feeds, names);
  return Object.fromEntries(indices.map(i => [i, getOutputByIndex(outputs, i)]));
}
function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}
function findMaskBox(maskLogits, threshold = 0.55) {
  let minX = SIZE.page, minY = SIZE.page, maxX = -1, maxY = -1;
  for (let y = 0; y < SIZE.page; y++) {
    for (let x = 0; x < SIZE.page; x++) {
      const v = maskLogits[y * SIZE.page + x];
      const sigmoid = v >= 0 ? 1 / (1 + Math.exp(-v)) : Math.exp(v) / (1 + Math.exp(v));
      if (sigmoid > threshold) {
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < minX || maxY < minY) return [0, 0, SIZE.page, SIZE.page];
  return [minX, minY, maxX + 1, maxY + 1];
}
function getRoiCanvas(sourceCanvas, pageInfo, box) {
  const [bx0, by0, bx1, by1] = box;
  const x0 = Math.max(0, Math.min(sourceCanvas.width, Math.round((bx0 - pageInfo.left) / pageInfo.scale)));
  const y0 = Math.max(0, Math.min(sourceCanvas.height, Math.round((by0 - pageInfo.top) / pageInfo.scale)));
  const x1 = Math.max(x0 + 1, Math.min(sourceCanvas.width, Math.round((bx1 - pageInfo.left) / pageInfo.scale)));
  const y1 = Math.max(y0 + 1, Math.min(sourceCanvas.height, Math.round((by1 - pageInfo.top) / pageInfo.scale)));
  const c = makeCanvas(x1 - x0, y1 - y0);
  c.getContext('2d').drawImage(sourceCanvas, x0, y0, x1 - x0, y1 - y0, 0, 0, c.width, c.height);
  return { canvas: c, box: [x0, y0, x1, y1] };
}

async function predict(sourceCanvas) {
  const start = performance.now();
  const docInfo = letterbox(sourceCanvas, SIZE.doc);
  const documentImage = tensorFromCanvas(docInfo.canvas);
  const pageDummy = zeroTensor(SIZE.page);
  const baseDummy = zeroTensor(SIZE.base);
  const detailDummy = zeroTensor(SIZE.detailH, SIZE.detailW);
  const commonFeeds = { document_image: documentImage, page: pageDummy, roi_base: baseDummy, roi_detail: detailDummy };

  // Stage 1: phân loại loại giấy tờ.
  const d = await runStage(commonFeeds, [1, 2, 3, 4]);
  const probs = d[1];
  const cosines = d[3];
  const docIndex = argmax(probs);
  const nearestIndex = argmax(cosines);
  const rejected = d[4][0] >= 0.5;
  let finalIndex = rejected ? 9 : docIndex;
  let stageInfo = 'Phân loại giấy tờ';

  if (!rejected && GRADE_KEYS.includes(DOC_KEYS[docIndex])) {
    // Stage 2: tìm vùng học bạ.
    const pageInfo = letterbox(sourceCanvas, SIZE.page);
    const pageTensor = tensorFromCanvas(pageInfo.canvas);
    const m = await runStage({ ...commonFeeds, page: pageTensor }, [5]);
    const box = findMaskBox(m[5]);
    const roi = getRoiCanvas(sourceCanvas, pageInfo, box);

    // Stage 3: phân loại lớp 10/11/12 trên ROI.
    const baseInfo = letterbox(roi.canvas, SIZE.base);
    const detailInfo = letterbox(roi.canvas, SIZE.detailW, SIZE.detailH);
    const grade = await runStage({
      ...commonFeeds,
      roi_base: tensorFromCanvas(baseInfo.canvas),
      roi_detail: tensorFromCanvas(detailInfo.canvas)
    }, [7, 8, 9]);
    const gradeIndex = argmax(grade[9]);
    finalIndex = DOC_KEYS.indexOf(GRADE_KEYS[gradeIndex]);
    stageInfo = 'Phân loại giấy tờ + tìm vùng + phân loại lớp';
  }

  return {
    label: finalIndex === 9 ? 'Không phải giấy tờ đã học' : DOC_CLASSES[finalIndex],
    rejected,
    confidence: finalIndex === 9 ? 0 : probs[docIndex],
    ms: performance.now() - start,
    stageInfo,
    docLabel: DOC_CLASSES[docIndex],
    docConfidence: probs[docIndex],
    nearestLabel: DOC_CLASSES[nearestIndex]
  };
}

function drawPreview(blobOrUrl) {
  $('preview').src = blobOrUrl;
  $('preview').hidden = false;
  $('placeholder').hidden = true;
  $('video').hidden = true;
  $('runButton').disabled = !session || busy;
  $('resultPanel').hidden = true;
}
async function loadImageFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  clearError();
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    currentImage = makeCanvas(img.naturalWidth, img.naturalHeight);
    currentImage.getContext('2d').drawImage(img, 0, 0);
    drawPreview(url);
  };
  img.onerror = () => showError('Không đọc được ảnh này. Hãy chọn ảnh JPG hoặc PNG.');
  img.src = url;
}

$('fileInput').addEventListener('change', e => loadImageFile(e.target.files[0]));
$('cameraButton').addEventListener('click', async () => {
  clearError();
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera cần HTTPS. Hãy mở trang qua GitHub Pages.');
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' } }, audio: false });
    $('video').srcObject = cameraStream;
    $('video').hidden = false; $('preview').hidden = true; $('placeholder').hidden = true;
    $('captureButton').disabled = false; $('stopCameraButton').disabled = false;
  } catch (e) { showError(`Không mở được camera: ${e.message}`); }
});
$('captureButton').addEventListener('click', () => {
  const video = $('video');
  if (!video.videoWidth) return;
  currentImage = makeCanvas(video.videoWidth, video.videoHeight);
  currentImage.getContext('2d').drawImage(video, 0, 0);
  drawPreview(currentImage.toDataURL('image/jpeg', 0.92));
});
$('stopCameraButton').addEventListener('click', () => {
  if (cameraStream) cameraStream.getTracks().forEach(t => t.stop());
  cameraStream = null; $('video').srcObject = null; $('video').hidden = true;
  $('captureButton').disabled = true; $('stopCameraButton').disabled = true;
  if (currentImage) $('preview').hidden = false;
  else $('placeholder').hidden = false;
});
$('runButton').addEventListener('click', async () => {
  if (!currentImage || !session || busy) return;
  clearError(); busy = true; $('runButton').disabled = true; $('runButton').textContent = 'Đang chạy…';
  try {
    const result = await predict(currentImage);
    $('resultPanel').hidden = false;
    $('resultName').textContent = result.label;
    $('inferenceTime').textContent = `${(result.ms / 1000).toFixed(2)} giây`;
    $('loadTime').textContent = `${(modelLoadMs / 1000).toFixed(2)} giây`;
    $('resultNote').textContent = `${result.stageInfo}. Lớp sơ bộ: ${result.docLabel} (${(result.docConfidence * 100).toFixed(1)}%).`;
  } catch (e) {
    showError(`Lỗi suy luận: ${e.message}`);
  } finally {
    busy = false; $('runButton').disabled = !currentImage || !session; $('runButton').textContent = 'Nhận diện';
  }
});

loadModel();
