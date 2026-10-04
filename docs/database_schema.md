# SQLite database schema

The application creates a `prediction_history` table in `data/predictions.sqlite3` the first time the backend starts.

| Column | Type | Meaning |
|---|---|---|
| `id` | integer primary key | Local history row identifier |
| `created_at` | timestamp | Time the prediction request completed |
| `mode` | text | `ml`, `cnn`, or `compare` |
| `predictions_json` | text | Predicted class, top model scores, model version, and inference time |
| `latency_ms` | real | Total request inference time in milliseconds |

The app never stores the source image or its bytes. `DELETE /api/history` removes all rows. Dataset files, manifests, the SQLite database, and trained artifacts are ignored by Git.
