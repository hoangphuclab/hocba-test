# Transcript Reader Web — GitHub Pages

## Files to publish
Upload the contents of this folder to the **root** of a GitHub repository. Keep the `models/` folder and its three ONNX files in the same paths.

Required layout:
```
index.html
style.css
app.js
config.json
.nojekyll
models/roi_detector.onnx
models/grid_detector.onnx
models/row_reader.onnx
```

## Publish from an iPhone
1. Create a **Public** repository on https://github.com/new (e.g. `transcript-reader-web`).
2. Upload these files while preserving the `models/` folder structure. Do not upload only the ZIP itself.
3. In repository **Settings → Pages**, choose **Deploy from a branch**, branch `main`, folder `/(root)`, then Save.
4. Open the Pages URL shown there, e.g. `https://YOUR-USERNAME.github.io/transcript-reader-web/`.
5. Allow camera permission in Safari. If camera permission is unavailable, use “Choose image from library”.

## Notes
- This is a static site. Inference is performed in Safari with ONNX Runtime Web/WASM; the image is not uploaded to a prediction server by this app.
- Internet is needed to load the ONNX Runtime Web CDN and model files the first time.
- Model files are about 12.3 MB total. A public repository means anyone can download the models; do not publish if you lack permission.
- Results are experimental. Compare against the Python `runtime.py` on several identical test images before relying on outputs. Browser canvas resizing differs slightly from Pillow resizing and can affect model predictions.
