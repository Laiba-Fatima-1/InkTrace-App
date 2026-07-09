# InkTrace — Live Web App (Task 3: Handwritten Character Recognition)

An interactive web app where anyone can draw a digit or upload a photo of handwriting and get an instant CNN prediction.

## 📁 Files
- `streamlit_app.py` — the full app (draw + upload tabs, prediction, confidence bars). The drawing canvas is built with plain HTML5/JavaScript embedded directly in the app — no external canvas package — for maximum reliability on Streamlit Cloud.
- `digit_model.h5` — trained CNN (digits 0-9, 99.3% test accuracy)
- `requirements.txt` — pinned dependencies
- `runtime.txt` — forces Python 3.11 for deployment

## ⚠️ Important: Python version
Set this app's Python version to **3.11** (already forced via `runtime.txt`, but double check in
**Manage app → Settings → General** if you have issues). Newer Python versions (3.13+) don't yet have
pre-built TensorFlow packages, which causes builds to hang or fail.

## 🚀 How to deploy live (Streamlit Community Cloud — free)

1. Create a new GitHub repo (e.g., `InkTrace-App`) and upload all 4 files above.
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub.
3. Click **New app** → select the repo → branch `main` → main file `streamlit_app.py`.
4. Click **Deploy**. First build takes 3-5 minutes (TensorFlow is a large package).
5. You'll get a live URL like `https://inktrace-yourname.streamlit.app`.

## 🧪 Test locally first (optional)
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 🔄 Upgrading to full digits + letters (EMNIST)
Once you've trained the EMNIST model in Colab (from the `CodeAlpha_HandwrittenCharacterRecognition` repo),
just:
1. Save it as `digit_model.h5` (replacing this one)
2. Update `CLASS_NAMES` in `streamlit_app.py` to the full 47-class list
3. Push to GitHub — the live app auto-updates

## ⚠️ Note
Keep `digit_model.h5` in the **same folder** as `streamlit_app.py`.
