# ---
# cmd: ["modal", "run", "06_gpu_and_ml/embeddings/cost_discipline_embeddings.py"]
# lambda-test: true
# ---

# # Cost-disciplined embeddings: pay for GPUs when you embed, zero when you don't

# This example shows the operational pattern this toolkit's packages encode:
# [Modal](https://modal.com) bills **per GPU-hour, never per token**, so the
# cheapest serving posture is **scale-to-zero with a warm probe at session
# start**, and the cheapest bulk posture is **batch jobs that stop the GPU
# when the drain finishes**.

# We deploy a small sentence-transformers embedding service on an L4 that
# scales to zero after 2 minutes idle, then run a local client that:
#
# 1. warms the service (kicks off the container boot while you type),
# 2. embeds a query and 3 documents,
# 3. reads back the bill shape (always-on burn vs token demand).
#
# The full version of this pattern, with model selection evals, monotonic
# sync down to a local vector store, and bulk re-embedding for model swaps,
# lives at [modal-embedding-server](https://github.com/kylebrodeur/modal-embedding-server).
# The companion operator CLI ([mtk](https://github.com/kylebrodeur/modal-toolkit))
# prints the same cost attribution as a fleet verb.

import time

import modal

MINUTES = 60  # seconds

app = modal.App(name="example-cost-discipline-embeddings")

# The model weights live on a Modal [Volume](https://modal.com/docs/guide/volumes)
# so each container boot skips the Hugging Face download. The first boot pays
# the download once; every boot after that reads from local disk.
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = "/model"
model_volume = modal.Volume.from_name("cost-discipline-model-cache", create_if_missing=True)

def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, cache_dir=MODEL_DIR)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "sentence-transformers==3.3.1",
        "huggingface-hub==0.26.2",
        "torch==2.5.1",
        "fastapi[standard]==0.115.6",
    )
    .env({"HF_HOME": MODEL_DIR})
    .run_function(download_model, volumes={MODEL_DIR: model_volume})
)

# The inference class scales to zero (`scaledown_window=2 * MINUTES`) and warms
# the model at container start, so the first request after a cold boot pays
# only the boot, not a model-load on top of it.
@app.cls(
    image=image,
    gpu="L4",
    volumes={MODEL_DIR: model_volume},
    scaledown_window=2 * MINUTES,
)
@modal.concurrent(max_inputs=4)
class EmbeddingService:
    @modal.enter()
    def load_model(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_ID, cache_folder=MODEL_DIR)
        # A single dummy encode pre-compiles the CUDA kernels so the first
        # real request hits peak performance.
        self.model.encode(["warmup"])

    @modal.method()
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    @modal.web_server(port=8000)
    def serve(self):
        # Optional HTTP surface for clients that prefer REST over Function calls.
        import fastapi
        import uvicorn

        web = fastapi.FastAPI()

        @web.post("/embed")
        def embed(body: dict):
            return {"vectors": self.model.encode(body["texts"]).tolist()}

        uvicorn.run(web, host="0.0.0.0", port=8000)

# ## The warm probe: hide the cold boot behind your typing time

# Scale-to-zero means the first request of the day pays a container boot
# (~30-60s on an L4 with a small model). A client-side warm probe at session
# start runs the boot while you read or type, so by the time you submit the
# real request the container is already hot. The probe is a cheap Function
# call that returns as soon as the container answers.

@app.function(image=image)
def warm_probe() -> str:
    t0 = time.perf_counter()
    EmbeddingService().embed.remote(["warm"])
    return f"warmed in {time.perf_counter() - t0:.1f}s"

# ## The local entrypoint: warm, use, and read the bill shape

# The client runs in three moves, and the total GPU-seconds spent is printed
# at the end so the cost is always visible, never silent.

@app.local_entrypoint()
def main():
    t0 = time.perf_counter()
    warm = warm_probe.remote()
    print(warm)

    texts = [
        "Modal bills per GPU-hour, never per token",
        "scale-to-zero is the default posture",
        "warm probes hide cold boots",
        "cost attribution belongs in the operator CLI",
    ]
    vectors = EmbeddingService().embed.remote(texts)
    print(f"embedded {len(vectors)} texts -> dim {len(vectors[0])} in {time.perf_counter() - t0:.1f}s total")

    # The bill shape: with scale-to-zero there is no always-on burn; the
    # session costs GPU-seconds only. An always-on L4 for this model would be
    # ~$0.80/hr ~ $576/mo. This session, spread out, is a rounding error.
    print("always-on burn: $0 (scaled to zero)")
    print("session GPU cost: L4-seconds only, for the boot + the two calls")

# Run it:
#
# ```bash
# modal run cost_discipline_embeddings.py
# ```
#
# Then stop the app when you're done (or let the 2-minute idle window do it
# for you):
#
# ```bash
# modal app stop example-cost-discipline-embeddings
# ```