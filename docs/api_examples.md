# API examples

Start the backend with `uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000`. Interactive OpenAPI docs are served at `/docs`.

## `GET /api/health`

```json
{"status":"ok","models":{"ml":false,"cnn":false}}
```

Model availability becomes `true` after its artifact is trained and saved.

## `GET /api/classes`

```json
{"available":true,"classes":["Apple___healthy","Apple___scab"]}
```

## `POST /api/predict`

Send multipart form fields `file` (JPEG, PNG, WebP, or BMP) and `model` (`ml` or `cnn`).

Example response shape (values are illustrative schema examples, not project results):

```json
{
  "model":"ml",
  "predicted_class":"Apple___healthy",
  "display_name":"Apple — healthy",
  "class_score":0.42,
  "top_classes":[
    {"class_name":"Apple___healthy","display_name":"Apple — healthy","score":0.42},
    {"class_name":"Apple___scab","display_name":"Apple — scab","score":0.16}
  ],
  "inference_ms":4.8,
  "model_version":"ml-local-build"
}
```

Score values in this example only document the response format. Real values come from trained model artifacts. They are uncalibrated model scores, not probabilities.

## `POST /api/compare`

Send multipart field `file`. The response contains `ml`, `cnn`, and `total_inference_ms`, each prediction matching the `/api/predict` shape.

## `GET /api/metrics`

Returns `models.ml` and `models.cnn`, each with `available` and `metrics`. Before a model has been trained its metric value is `null`. `comparison` is also `null` until both metric files exist.

## `GET /api/history` and `DELETE /api/history`

History returns the newest result rows first, with prediction metadata and no image data. Delete returns `{"deleted": <row_count>}`.
