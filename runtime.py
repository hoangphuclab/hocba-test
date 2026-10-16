"""CPU ONNX Runtime inference for the Vietnamese transcript reader."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFilter


class TranscriptONNX:
    def __init__(self, bundle: Path):
        self.bundle = Path(bundle)
        self.config = json.loads((self.bundle / "config.json").read_text(encoding="utf-8"))
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = max(1, min(8, __import__("os").cpu_count() or 2))
        options.inter_op_num_threads = 1
        self.roi_session = ort.InferenceSession(str(self.bundle / self.config["models"]["roi"]), options,
                                                providers=["CPUExecutionProvider"])
        self.grid_session = ort.InferenceSession(str(self.bundle / self.config["models"]["grid"]), options,
                                                 providers=["CPUExecutionProvider"])
        self.reader_session = ort.InferenceSession(str(self.bundle / self.config["models"]["reader"]), options,
                                                   providers=["CPUExecutionProvider"])
        self.subject_classes = self.config["subject_classes"]
        self.score_classes = self.config["score_classes"]

    @staticmethod
    def _normalized(image: Image.Image, size: tuple[int, int], fit=False) -> np.ndarray:
        image = image.convert("RGB")
        width, height = size
        if fit:
            scale = min((width - 8) / max(image.width, 1), (height - 8) / max(image.height, 1))
            rw, rh = max(1, round(image.width * scale)), max(1, round(image.height * scale))
            resized = image.resize((rw, rh), Image.Resampling.BICUBIC if scale > 1 else Image.Resampling.BOX)
            canvas = Image.new("RGB", (width, height), (238, 238, 238))
            canvas.paste(resized, ((width-rw)//2, (height-rh)//2))
        else:
            canvas = image.resize((width, height), Image.Resampling.BOX)
        array = np.asarray(canvas, dtype=np.float32) / 255.
        return ((array - .5) / .5).transpose(2, 0, 1)[None].astype(np.float32)

    @staticmethod
    def _components(mask: np.ndarray):
        h, w = mask.shape
        parents, runs, previous = [], [], []
        for y in range(h):
            xs = np.flatnonzero(mask[y]); current = []
            if xs.size:
                for group in np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1):
                    idx = len(parents); parents.append(idx)
                    run = [int(group[0]), y, int(group[-1]), idx]
                    runs.append(run); current.append(run)
                    for old in previous:
                        if old[2] + 1 < run[0]: continue
                        if run[2] + 1 < old[0]: break
                        a, b = idx, old[3]
                        while parents[a] != a: a = parents[a]
                        while parents[b] != b: b = parents[b]
                        if a != b: parents[b] = a
            previous = current
        found = {}
        for x0, y, x1, idx in runs:
            root = idx
            while parents[root] != root: root = parents[root]
            parents[idx] = root
            box = found.setdefault(root, [x0, y, x1+1, y+1])
            box[0] = min(box[0], x0); box[1] = min(box[1], y)
            box[2] = max(box[2], x1+1); box[3] = max(box[3], y+1)
        return list(found.values())

    def find_rois(self, image: Image.Image):
        w, h = image.size
        tensor = self._normalized(image, (384, 512))
        logits = self.roi_session.run(["roi_logits"], {"image": tensor})[0]
        probability = 1. / (1. + np.exp(-np.clip(logits[0, 0], -30, 30)))
        mask_image = Image.fromarray((probability >= .5).astype(np.uint8) * 255)
        k = 3
        mask_image = mask_image.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MinFilter(k))
        mask = np.asarray(mask_image) > 127
        mh, mw = mask.shape
        sx, sy = w/mw, h/mh
        result = []
        for x0, y0, x1, y1 in self._components(mask):
            if (x1-x0)*(y1-y0) < .008*mw*mh: continue
            result.append({"bbox": [round(x0*sx), round(y0*sy), round(x1*sx), round(y1*sy)],
                           "confidence": float(probability[y0:y1, x0:x1].mean())})
        return sorted(result, key=lambda r: (r["bbox"][1], r["bbox"][0]))

    @staticmethod
    def _peaks(values, threshold, min_gap):
        values = np.asarray(values, dtype=np.float32)
        if not len(values) or float(values.max()) < threshold: return []
        p = np.pad(values, (2, 2), mode="edge")
        smooth = (p[:-4] + 4*p[1:-3] + 6*p[2:-2] + 4*p[3:-1] + p[4:]) / 16.
        candidates = [i for i in range(1, len(smooth)-1)
                      if smooth[i] >= smooth[i-1] and smooth[i] >= smooth[i+1]
                      and smooth[i] >= max(threshold, float(smooth.max())*.28)]
        selected = []
        for point in sorted(candidates, key=lambda i: float(smooth[i]), reverse=True):
            if all(abs(point-old) >= min_gap for old in selected): selected.append(point)
        return sorted(selected)

    @staticmethod
    def _ordered_columns(scores, min_gap):
        width = len(scores[0])
        dp = np.asarray(scores[0], dtype=np.float64).copy()
        back = []
        for profile in scores[1:]:
            current = np.full(width, -np.inf, dtype=np.float64)
            parent = np.full(width, -1, dtype=np.int32)
            for x in range(width):
                last = x - min_gap
                if last >= 0:
                    prev = int(np.argmax(dp[:last+1]))
                    current[x] = dp[prev] + profile[x]
                    parent[x] = prev
            if not np.isfinite(current).any(): raise ValueError("Could not order predicted vertical boundaries")
            back.append(parent); dp = current
        positions = [int(np.argmax(dp))]
        for parent in reversed(back): positions.append(int(parent[positions[-1]]))
        return list(reversed(positions))

    @classmethod
    def _parse_grid(cls, logits, size):
        raw = logits[0]
        probs = 1. / (1. + np.exp(-np.clip(raw, -30, 30)))
        ih, iw = size; mh, mw = probs.shape[1:]
        yproj = probs[0].mean(axis=1)
        ys = []
        for threshold in (.55, .40, .30, .22, .16):
            ys = cls._peaks(yproj, threshold, max(1, int(mh*.008)))
            if len(ys) >= 4: break
        if len(ys) < 4: raise ValueError(f"Neural detector found only {len(ys)} row boundaries")
        profiles = []
        for ch in range(1, 6):
            profiles.append(raw[ch, ys[0]:ys[-1]+1].mean(axis=0))
        xs = cls._ordered_columns(profiles, max(1, int(mw*.015)))
        xb = [max(0, min(iw-1, round(x*iw/mw))) for x in xs]
        rows = []
        for top, bottom in zip(ys[:-1], ys[1:]):
            ya, yb = max(0, round(top*ih/mh)), min(ih, round(bottom*ih/mh))
            if yb <= ya: continue
            cells = [[xb[j], ya, xb[j+1], yb] for j in range(4)]
            rows.append({"bbox": [xb[0], ya, xb[-1], yb], "cell_boxes": cells})
        return rows

    def find_rows(self, image, rois):
        w, h = image.size
        rows = []
        for roi_index, item in enumerate(rois):
            x0, y0, x1, y1 = item["bbox"]
            px, py = round((x1-x0)*.025), round((y1-y0)*.025)
            cx0, cy0, cx1, cy1 = max(0,x0-px), max(0,y0-py), min(w,x1+px), min(h,y1+py)
            crop = image.crop((cx0, cy0, cx1, cy1))
            tensor = self._normalized(crop, (384, 512))
            logits = self.grid_session.run(["grid_logits"], {"roi": tensor})[0]
            for row in self._parse_grid(logits, (crop.height, crop.width)):
                boxes = [[b[0]+cx0, b[1]+cy0, b[2]+cx0, b[3]+cy0] for b in row["cell_boxes"]]
                boxes[0][0], boxes[-1][2] = x0, x1
                for b in boxes:
                    b[0]=max(x0,min(x1,b[0])); b[2]=max(x0,min(x1,b[2]))
                    b[1]=max(y0,min(y1,b[1])); b[3]=max(y0,min(y1,b[3]))
                if all(b[2]>b[0] and b[3]>b[1] for b in boxes):
                    rows.append({"bbox":[x0,max(y0,row["bbox"][1]+cy0),x1,min(y1,row["bbox"][3]+cy0)],
                                 "cell_boxes":boxes,"roi_index":roi_index})
        return sorted(rows, key=lambda r:(r["bbox"][1],r["bbox"][0]))

    def _classify_rows(self, image, rows):
        batch = []
        for row in rows:
            x0,y0,x1,y1 = row["bbox"]
            crop = image.crop((max(0,x0),max(0,y0),min(image.width,x1),min(image.height,y1)))
            batch.append(self._normalized(crop,(384,96),fit=True)[0])
        outputs = self.reader_session.run(None, {"rows":np.stack(batch).astype(np.float32)})
        return outputs

    @staticmethod
    def _softmax(logits):
        x = logits.astype(np.float64); x -= x.max(axis=-1, keepdims=True)
        e = np.exp(x); return e/e.sum(axis=-1, keepdims=True)

    def _decode(self, row_outputs, row_index):
        names = [self.subject_classes, self.score_classes, self.score_classes, self.score_classes]
        probs = [self._softmax(out[row_index]) for out in row_outputs]
        raw = [int(p.argmax()) for p in probs]; chosen = raw.copy(); constrained = False
        if all(i < 101 for i in raw[1:]):
            logs = [np.log(p[:101].clip(1e-12)) for p in probs[1:]]
            a,b=np.indices((101,101)); cn=(a+2*b+1)//3
            joint=logs[0][:,None]+logs[1][None,:]+logs[2][cn]
            ia,ib=np.unravel_index(int(np.argmax(joint)),joint.shape)
            chosen[1:]=[int(ia),int(ib),int(cn[ia,ib])]; constrained=True
        result={}
        for k,field in enumerate(("subject","HK1","HK2","CN")):
            i=chosen[k]; order=np.argsort(probs[k])[::-1][:3]
            result[field]={"prediction":names[k][i],"raw_prediction":names[k][raw[k]],
                           "confidence":float(probs[k][i]),"constraint_applied":bool(constrained and k>0),
                           "top3":[{"label":names[k][j],"probability":float(probs[k][j])} for j in order]}
        return result

    def predict(self, source: Path, output: Path | None=None):
        image=Image.open(source).convert("RGB")
        rois=self.find_rois(image)
        if not rois: raise RuntimeError("ROI model did not find the score table")
        rows=self.find_rows(image,rois)
        if not rows: raise RuntimeError("Grid model did not find score rows inside ROI")
        outs=self._classify_rows(image,rows)
        results=[]
        for i,row in enumerate(rows):
            results.append({"row":i+1,"bbox":row["bbox"],"cell_boxes":row["cell_boxes"],"heads":self._decode(outs,i)})
        preview=image.copy(); draw=ImageDraw.Draw(preview)
        colors=[(255,150,0),(0,155,255),(75,190,50),(220,55,125)]
        for roi in rois: draw.rectangle(tuple(roi["bbox"]),outline=(190,70,240),width=3)
        for i,row in enumerate(rows):
            for j,b in enumerate(row["cell_boxes"]): draw.rectangle(tuple(b),outline=colors[j],width=2)
            draw.text((row["bbox"][0]+2,row["bbox"][1]+2),str(i+1),fill=(20,20,20))
        if output:
            output.parent.mkdir(parents=True,exist_ok=True)
            preview.save(output,quality=92)
        return {"filename":source.name,"runtime":"ONNX Runtime CPU","roi_count":len(rois),
                "rows_detected":len(results),"rows":results,"preview":str(output) if output else None}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description="Predict transcript rows using ONNX Runtime")
    parser.add_argument("image",type=Path); parser.add_argument("--bundle",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--json",type=Path,default=None); parser.add_argument("--preview",type=Path,default=None)
    args=parser.parse_args(); runtime=TranscriptONNX(args.bundle)
    result=runtime.predict(args.image,args.preview)
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True); args.json.write_text(text,encoding="utf-8")
    print(text)


if __name__=="__main__": main()
