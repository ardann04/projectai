# Copilot Instructions for this repo

Purpose
- Help AI coding agents understand this Django + local-ML project quickly and make safe, productive edits.

Big picture
- Single Django project: `djangoApp` (settings, urls, wsgi). The main app is `app/` which serves both website pages (`templates/`, `static/`) and two APIs: `POST /chat/` and `POST /predict/` (see `app/urls.py`).
- Chatbot: a local transformer-based model lives in `models/gym_final_merged/` (safetensors, tokenizer files). `app/views.py` loads it with `transformers` and runs generation. The code may use `optimum.BetterTransformer` when available for speed (optional).
- ML classifier: traditional ML model artifacts (joblib pickles) are in `app/models/` and loaded at module import in `app/views.py` (files: `model.pkl`, `encoders.pkl`, `binning.pkl`, `features.pkl`).

Where to look first
- `app/views.py` — core logic for both chatbot and ML prediction endpoints. Note: heavy model loads occur at import time here.
- `app/utils/chat_utils.py` — small utilities (greeting detection) used by the chat flow.
- `djangoApp/settings.py` — useful runtime settings (CORS enabled, static dirs, sqlite DB).
- `app/urls.py` — API routes.
- `models/gym_final_merged/` — local transformer model files and tokenizer.

Developer workflows & commands (PowerShell on Windows)
- Activate the repository venv (this repo contains a `.venv` folder):
```powershell
. .venv\Scripts\Activate.ps1
```
- Run migrations (uses sqlite):
```powershell
python manage.py migrate
```
- Start dev server:
```powershell
python manage.py runserver 0.0.0.0:8000
```
- Test chat endpoint (JSON POST):
```powershell
curl -X POST http://127.0.0.1:8000/chat/ -H "Content-Type: application/json" -d '{"message":"halo"}'
```
- Test predict endpoint: send JSON with features matching the pickled `feature_columns` loaded from `app/models/features.pkl`.

Project-specific patterns & gotchas
- Heavy imports at module scope: `app/views.py` loads joblib models and the transformer model at import time. This means Django will attempt to load models on process start — be cautious when editing or running tests; prefer lazy-loading if making frequent code changes.
- Optional `optimum` usage: the code attempts to import `optimum.bettertransformer` in a `try/except` block. Missing `optimum` will be skipped (safe fallback). If you need to enable BetterTransformer, install `optimum` in the venv (`pip install optimum`).
- GPU vs CPU: generation uses `device_map='auto'` and chooses `torch.float16` when CUDA is available. Ensure `torch` is installed with appropriate CUDA support before expecting GPU acceleration.
- CORS is enabled via `CORS_ALLOW_ALL_ORIGINS = True` in `djangoApp/settings.py` — useful for mobile/web frontends but be mindful for production.

Integration points
- Frontend: `templates/` + `static/` hold the website UI; `index.html` is rendered by `views.index`.
- Mobile clients: settings contain a comment indicating Flutter access; `/chat/` and `/predict/` are designed as JSON APIs.
- ML model files: `app/models/*.pkl` are required for `/predict/`; `models/gym_final_merged/` is required for `/chat/`.

Editing guidance for agents
- Avoid introducing top-level side effects that trigger heavy model loading. If you must change `app/views.py`, prefer function-scoped or lazy initialization for models.
- When adding tests or CI, mock or stub the model-loading paths so test startup is fast.
- Keep changes minimal and run `python manage.py runserver` locally to validate API behavior.

Examples (patterns found in repo)
- Greeting handler: `app/utils/chat_utils.py` contains `handle_greetings(user_input)` that returns a canned response for greetings.
- Chat generation: `app/views.py` uses `AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)` and `AutoModelForCausalLM.from_pretrained(..., device_map='auto')` — edits around tokenization or generation should consider device placement and dtype.

When in doubt
- Search `app/views.py` first — it is the single most important file to understand changes to chat/predict behavior.

Please review this draft and tell me if you want more details (examples of sample requests/responses, preferred test commands, or additional warnings about the local model files).