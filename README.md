# TrustLayer — working prediction API and dashboard

TrustLayer is a research application with a real server-side inference API, a trained scikit-learn model exported for portable JavaScript inference, model evaluation, CSV scoring, and a separate session-context lab. It does not connect to a bank or execute/block payments.

## Quick start on Windows
Install Node.js 22+; no npm dependencies are required for the dashboard or API.

```powershell
npm run build
npm test
npm start
```

Open http://localhost:8000. The trained model is included. The local server binds only to localhost. `PORT` can override the default. The hosted version is protected by Sites private access; local execution is a development server, not a public production deployment.

## APIs
- `GET /api/health`: loaded model and service status.
- `GET /api/model`: model version, dataset, candidate metrics, split details and limitations.
- `POST /api/predict`: score one transaction.
- `POST /api/predict/batch`: score 1–200 transactions. Entire batch is validated before predictions.
- `GET /api/openapi.json`: OpenAPI 3.1 schema.
- `GET /api/demo`: explicitly labelled synthetic held-out benchmark examples.

Example PowerShell:

```powershell
$body = @{transaction_id='TX-001'; amount=2500; type='TRANSFER'; hour=14} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/predict -Method Post -ContentType 'application/json' -Body $body
```

Amounts are PaySim local currency units, **not INR**. Types: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER. Hour: integer 0–23. Extra fields are rejected. No unknown UPI transaction type is silently mapped. API requests have a 150 KB limit. The browser CSV scorer accepts up to 1,000 rows and calls the batch API. This is inference, not automatic retraining.

The hosted service uses private Sites access. An external program needs authorized platform access; no anonymous public API or custom token has been configured. The browser playground uses the signed-in site session.

## Training and reproduction
Download the original dataset from https://www.kaggle.com/datasets/ealaxi/paysim1 (public download: https://www.kaggle.com/api/v1/datasets/download/ealaxi/paysim1). Keep the ZIP outside this repository.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r ml/requirements.txt
python ml/train.py --data C:\datasets\paysim.zip --sample-rate 1
npm run build
npm test
npm start
```

Training compares logistic regression and three histogram gradient boosting configurations. Model selection uses validation average precision; the selected configuration is refit on train+validation. Chronological steps separate train, validation, calibration and final test. No balancing or resampling changes test prevalence. Thresholds use calibration negatives for ~1% review and ~0.1% urgent review false-positive targets. Test precision, recall, ROC-AUC, average precision, confusion matrices and observed FPR are reported. Artifacts are `models/paysim.json`, `models/report.json`, `models/examples.json`, and `models/parity.json`. Rebuild and redeploy to update the hosted model; no unauthenticated online training endpoint exists.

Only log amount, transaction type one-hot encoding and cyclic hour are used. Excluded: all balance fields (per dataset author warning), post-transaction information, account identifiers, absolute simulation step, isFlaggedFraud, and target labels. With this intentionally limited signal, fraud detection is imperfect. No claim of universal best model or production accuracy is made.

The first exploratory run used a 15% sample. The final run uses all records with the same pre-specified candidate configurations and validation selection rule. Source ZIP SHA256, versions and final metrics are recorded in the model report. Python/JavaScript parity is checked against predictions from the final model.

## Uploaded notebook audit
The supplied `bank-transaction-fraud-detection-accuracy-95.ipynb` is an analysis notebook, not the underlying dataset. Its saved output shows 37,955 legitimate and 2,045 fraud cases in a 40,000-row test set: predicting all legitimate gives 94.8875% accuracy. Several displayed models report fraud recall 0.00 and ROC-AUC around 0.50.

Additional evaluation weaknesses: feature importance selection before train/test separation; SMOTE before GridSearchCV folds; default accuracy scoring for selection; plotted ROC based on hard predictions. The underlying full CSV and fitted model are not attached, so those results were audited from saved outputs, not independently rerun. Near-random performance does not by itself prove the original labels are random.

## Session-context lab
`/lab.html` preserves the original synthetic coercion simulator, class-conditional conformal intervention and browser-local labelled CSV training. It uses a different dataset/model and is explicitly isolated from the deployed PaySim transaction-fraud model. PaySim labels must not be repurposed as verified coercion labels. Real coercion training needs consented session telemetry and verified outcomes. Browser lab imports never leave that browser. Lab models are not promoted to the deployed API.

## Runtime and privacy
The server is stateless. It does not retain uploaded transactions, request histories or scoring results. The dashboard holds recent requests in browser memory (max 2,000); refresh clears them. Model artifacts are permanent source-controlled deployment assets. Curated replay deliberately includes 20 fraud and 40 legitimate held-out records and is not representative live traffic. Model metrics remain fixed while requests are replayed.

Real-time means on-demand scoring over HTTP, not a real-time bank feed. A genuine UPI integration would require an authorized bank/PSP sandbox and mobile telemetry integration. This project requests no account credentials or UPI PINs and implements no payment movement.

## Dataset attribution and license
PaySim: Edgar A. Lopez-Rojas, Ahmad Elmir and Stefan Axelsson, “PaySim: A financial mobile money simulator for fraud detection,” EMSS 2016. Dataset: https://www.kaggle.com/datasets/ealaxi/paysim1. Dataset license: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). Included examples are a selected/transformed subset of that synthetic dataset and remain under CC BY-SA 4.0. The simulator represents mobile money in an African setting, not actual Indian UPI transactions. Full raw data is not bundled.

## Testing
`npm test` verifies Python/JavaScript prediction parity, API validation, atomic batch rejection, size limits, cross-origin rejection and route availability. It does not prove production fraud effectiveness. The model report exposes chronological prevalence shift and errors.
