# TAB — OpenShift / Kubernetes Runtime Deployment

This directory contains minimal manifests to run `tab` as a GPU-enabled
batch workload on OpenShift or any Kubernetes cluster.

---

## Files

| File | Purpose |
|------|---------|
| `job.yaml` | Kubernetes `Job` template for running a single benchmark |
| `values.example.yaml` | Reference list of all knobs that must/can be set |

---

## Quick start

### 1. Create the S3 credentials secret

```bash
kubectl create secret generic tab-s3-credentials \
  --from-literal=DIH_S3_ENDPOINT=https://<your-s3-endpoint> \
  --from-literal=DIH_S3_TOKEN=<access-key-id> \
  --from-literal=DIH_S3_SECRET=<secret-access-key> \
  -n <your-namespace>
```

### 2. Edit the Job manifest

Open `job.yaml` and fill in or uncomment the cluster-specific sections:

```yaml
# Set your image reference
image: "<registry>/<namespace>/tab:<tag>"

# Uncomment and set the GPU node selector (ask your platform team):
# nodeSelector:
#   <gpu-node-label-key>: "<gpu-node-label-value>"

# Uncomment and set GPU tolerations if nodes are tainted:
# tolerations:
#   - key: "nvidia.com/gpu"
#     operator: "Exists"
#     effect: "NoSchedule"

# Optionally set the ServiceAccount:
# serviceAccountName: <your-serviceaccount>
```

### 3. Submit the Job

```bash
kubectl create -f deploy/openshift/job.yaml -n <your-namespace>
```

### 4. Follow logs

```bash
kubectl logs -f job/tab-benchmark -n <your-namespace>
```

---

## Running a different benchmark

Override the `BENCHMARK_SCRIPT` environment variable:

```bash
# Using kubectl patch / edit after creation, or set in job.yaml before creation:
BENCHMARK_SCRIPT=./scripts/multivariate/score/CATCH.sh
```

All available benchmark scripts are located under:
- `scripts/multivariate/score/` — multivariate anomaly-detection benchmarks
- `scripts/univariate/` — univariate benchmarks

---

## Environment variables reference

### S3 / DIH connection

| Variable | Source | Default | Description |
|----------|--------|---------|-------------|
| `DIH_S3_ENDPOINT` | Secret | — | S3 endpoint URL |
| `DIH_S3_TOKEN` | Secret | — | S3 access key ID |
| `DIH_S3_SECRET` | Secret | — | S3 secret access key |
| `DIH_S3_REGION` | Job env | `eu-central-1` | AWS region |
| `DIH_S3_BUCKET_NAME` | Job env | `ka-etu-dih-001-oktoplustim-001` | Bucket name |
| `DIH_S3_DATASET_PREFIX` | Job env | `data/tab/dataset` | S3 key prefix for datasets |
| `DIH_S3_CHECKPOINTS_PREFIX` | Job env | `data/tab/checkpoints` | S3 key prefix for model checkpoints |
| `DIH_S3_RESULTS_PREFIX` | Job env | `data/tab/results` | S3 key prefix for result uploads |
| `DIH_S3_VERIFY_SSL` | Job env | `true` | Verify TLS certificates |
| `TAB_USE_S3` | Job env | `true` | Enable S3 data/result backend |

### Runtime paths

| Variable | Default | Description |
|----------|---------|-------------|
| `TAB_RESULT_PATH` | `/tmp/result` | Local result output directory |
| `MPLCONFIGDIR` | `/tmp/matplotlib` | Matplotlib config/cache (writable) |
| `HF_HOME` | `/tmp/cache/huggingface` | HuggingFace home directory |
| `TRANSFORMERS_CACHE` | `/tmp/cache/huggingface/transformers` | Transformers model cache |

### GPU

| Resource | Default | Description |
|----------|---------|-------------|
| `nvidia.com/gpu` (request) | `1` | Number of GPUs requested |
| `nvidia.com/gpu` (limit) | `1` | Number of GPUs allowed |

Set the request/limit to `0` and remove the `nodeSelector`/`tolerations`
to run in CPU-only mode (see *CPU fallback* below).

---

## Cluster-specific values (supplied by the platform team)

The following cannot be derived from the repository alone and must be
provided by the OpenShift / Kubernetes platform team:

| Item | Description |
|------|-------------|
| GPU node label | Key/value pair that identifies GPU-capable nodes, e.g. `nvidia.com/gpu.present: "true"` |
| GPU node taint | Taint key/effect applied to GPU nodes, e.g. `nvidia.com/gpu:NoSchedule` |
| Container registry | Full registry URL where the built `tab` image is pushed |
| ServiceAccount | Name of the SA with image-pull-secret and required RBAC |
| Namespace | OpenShift project / k8s namespace to deploy into |

---

## CPU fallback

Several benchmark models contain hard `.cuda()` calls and will crash when
no GPU is present.  The following files have been patched in this
repository to be device-agnostic (using `register_buffer` /
`.to(device)`):

- `ts_benchmark/baselines/self_impl/Anomaly_trans/attn.py`
- `ts_benchmark/baselines/self_impl/DualTF/model/FrequencyTransformer.py`
- `ts_benchmark/baselines/self_impl/DualTF/model/TimeTransformer.py`
- `ts_benchmark/baselines/self_impl/DualTF/DualTF.py`
- `ts_benchmark/baselines/self_impl/CrossAD/models/CrossAD_model.py`
- `ts_benchmark/baselines/self_impl/TFAD/model/mixup.py`
- `ts_benchmark/baselines/time_series_library/layers/AutoCorrelation.py`

The following files inside `submodules/` still contain hard `.cuda()`
calls and are **not** patched (they are third-party code managed
separately):

- `ts_benchmark/baselines/LLM/submodules/LLMMixer/layers/AutoCorrelation.py`
- `ts_benchmark/baselines/pre_train/submodules/Timer/exp/exp_forecast.py`

For a pure CPU run, set `nvidia.com/gpu: "0"` in the Job resources and
remove `nodeSelector` / `tolerations`.  Benchmarks relying on patched
models will fall back to CPU automatically.  Benchmarks that use the
unpatched submodule code may still fail without a GPU.
