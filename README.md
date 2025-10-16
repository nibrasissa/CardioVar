# CardioVar — Working Demo (20250907_070933)

This bundle runs out of the box using a built-in *BaselineModel* if no trained model is provided.
You can still point to your own `*_best_model.joblib` later via env vars.

## 1) Create environment
### Windows PowerShell
```powershell
cd CardioVarApp_Working_20250907_070933
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

### macOS/Linux
```bash
cd CardioVarApp_Working_20250907_070933
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

## 2) (Optional) Use your trained model
```powershell
$env:MODEL_PATH="C:\path\to\rf_best_model.joblib"   # optional
$env:FEATURES_PATH="C:\path\to\features_pruned.txt" # optional
$env:THRESHOLD="0.5"
$env:LOW_THRESH="0.3"
$env:HIGH_THRESH="0.7"
$env:CLASS_MAP="0:not candidate,1:candidate"
```

If `MODEL_PATH` cannot be loaded, the server will automatically use a **BaselineModel** so you can test immediately.

## 3) Run the server
```powershell
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
# Then open http://localhost:8000   (API docs at http://localhost:8000/docs)
```

## 4) Test with the provided sample
- In the web UI, click Upload & Run CardioVar → choose `sample\example_variants.csv` → Run CardioVar.
- You will see a preview table and a Download results (CSV) button.
