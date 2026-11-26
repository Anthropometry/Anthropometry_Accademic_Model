# S7c: serve the model as a web API (free hosting) and weigh laptop and phone options

**Status:** approved by Ryan, 26 Nov 2026 (packages approved, D-045; segmentation tool A by default, switchable,
D-046). Host: Ryan's Azure credit is used up; choosing between Oracle Always Free (recommended) and Google Cloud
Run. The code, Docker image and laptop route do not depend on the host.
**Branch:** `s7c-serve` (on top of `s7a-photo-silhouette`)

## Goal (one sentence)

Let the app send two photos and the person's details over the internet and get the `anthro-output/1` result
back, from a free host, using one Docker image that can also run on a laptop.

## The model today, by weight

| Part | What it is | Size (approx.) | Needed for |
| --- | --- | --- | --- |
| Photo model `runs/r4/best.pt` | ResNet-18 with the hybrid head: about 11 million numbers | ~45 MB | every request |
| Segmentation tool A (torchvision DeepLabV3-ResNet101) | cuts the person out of the photo | ~230 MB | photo requests |
| Segmentation tool B (rembg `u2net_human_seg`) | same job, other tool | ~170 MB | photo requests |
| ANSUR formulas, catalogue, routing | plain numbers (JSON) | < 0.1 MB | every request |
| Code + PyTorch (CPU build) + libraries | the `anthro` package and what it runs on | ~1-1.5 GB in a Docker image | every request |

So the server needs about **2-3 GB of memory** and a normal CPU; no GPU. The heavy parts are PyTorch and
the segmentation tool, not our model. Timing per request is **not measured yet**; this step measures it.

## Hosting: free routes checked (26 Nov 2026)

| Host | Free? | Fits our ~2-3 GB? | Card needed | Notes |
| --- | --- | --- | --- | --- |
| **Hugging Face Spaces (Docker)** | **no longer**: the docs now say a Space running on compute (Docker or Gradio) needs a paid plan; only static Spaces are free | yes (16 GB) | paid plan | ruled out for free use |
| **Azure for Students** (Azure Container Apps) | $100 credit for 12 months, renewable each year while a student | yes | **no** (school email) | best fit for an Ashesi student; credit runs out if left running carelessly |
| **Google Cloud Run** | always-free monthly allowance (2 million requests, 180,000 vCPU-seconds, 360,000 GiB-seconds) | yes | **yes** (billing account) | scales to zero; a cold start loads ~300 MB of model files first |
| **Oracle Cloud Always Free** (Ampere A1 VM) | 4 cores and 24 GB memory, never expires | yes | yes (identity check) | a full server you manage yourself (Docker, HTTPS, updates); ARM processor; free capacity is sometimes unavailable |
| **Laptop + tunnel** (e.g. Cloudflare Tunnel) | free | yes | no | for learning and a live demo only (see below) |

**Proposed: Azure for Students** for the class demo (free with the Ashesi email, no card), with **Google Cloud
Run** as the fallback if a card is available. The same Docker image runs on all of them, so the choice can
change later without code changes. Hugging Face was the original plan; it is dropped because it is no longer
free for this kind of app.

## What gets built

### 1. FastAPI service (`src/anthro/serve.py`)

| Endpoint | Does | Answers |
| --- | --- | --- |
| `GET /v1/health` | is the model loaded | `{"status": "ok", "schema": "anthro-output/1", "model_version", "catalogue"}`, or 503 while loading |
| `GET /v1/catalogue` | the measurement list for screens | codes, labels, which sex, adult ranges |
| `POST /v1/measure` | two photos + details → result | 200 `anthro-output/1`; 422 refusal `{"error": "refused", "reasons": [...]}`; 401; 413; 503 |

`POST /v1/measure` is a multipart form: `front` and `side` (image files), `height_cm`, `height_kind`, `sex`,
`age_range`, and optionally `weight_kg`, `weight_kind`, `clothing`, `front_pitch`, `front_roll`, `side_pitch`,
`side_roll`. It calls exactly `predict.predict_photos` (S7a): the server adds no logic of its own.

**Why FastAPI:** small, standard, automatic interactive docs at `/docs` (useful for the front-end team), and
already familiar to Ryan.

### 2. Safety built in
- **API key:** every request needs a header `X-API-Key`; the key is a host setting (a secret), never in code.
- **Limits:** each photo at most 15 MB; one request at a time per worker; a timeout.
- **Nothing kept:** photos are read from memory; the two silhouettes the model reads go to a temporary folder
  in memory (`/dev/shm` when the machine has it) that is deleted when the answer is sent. Logs hold only the
  time taken and the status (no photos, no measurements, no names).
- **CORS:** only the app's own web address may call it from a browser.
- **HTTPS** from the host.
- **Licence:** the model is trained on BodyM (CC BY-NC 4.0): the service is for the school project and
  demos only, never a paid product; the health answer says so.

### 3. Docker image (`Dockerfile`)
- `python:3.11-slim`, PyTorch **CPU** build (about a third of the GPU build's size), our package with the
  `serve` and `photos` extras.
- The model file and the segmentation weights are **baked into a private image** at build time (question 3),
  so the first request does not wait for downloads. The image is never public.
- Listens on port 8080 (`PORT` setting); one command to run it locally: `docker run -p 8080:8080 ...`.

### 4. Deployment guide (`docs/DEPLOY.md`)
Step by step for the chosen host, plus "run it on your laptop". Ryan does the account steps and keeps every
password and key; nothing secret goes in the repository.

### 5. Front-end doc: new "API" section
Endpoints, a sample request (curl), sample answers for success, refusal and errors, the API-key header, the
expected wait (from this step's timing).

## Laptop hosting (for Ryan's learning)

"Hosting on a laptop" means the laptop runs the same Docker image and answers requests; a **tunnel** (e.g.
Cloudflare Tunnel or ngrok) gives it a public HTTPS address, so phones outside the room can reach it.

| | |
| --- | --- |
| Good for | learning how serving works; a live demo in the room; zero cost |
| Not good for | anything others rely on: it stops when the laptop sleeps, closes or loses Wi-Fi; your home network and laptop are exposed to the internet through the tunnel; slow upload links make each request slower |

`docs/DEPLOY.md` will include this as its first, simplest route.

## Offline on the phone (later, for the long run): weighed

| | On the phone | On a server (this step) |
| --- | --- | --- |
| Works without internet | **yes** (field work, villages, low data) | no |
| Photos leave the phone | **no** (better privacy, easier ethics) | yes, over HTTPS |
| Cost per measurement | none | server costs once the free credit ends |
| App size | grows by the model files (see below) | unchanged |
| Speed | depends on the phone; older phones are slow | depends on the server |
| Updating the model | needs an app update | change the server, all users get it at once |
| Work needed | convert the models (e.g. to ONNX / TensorFlow Lite / Core ML), rewrite the outline geometry and formulas for the phone, test on low-end phones | this step |

**Size on the phone:** the photo model shrinks to about **11 MB** when stored in 8-bit form (a standard
step, usually with little accuracy loss, which must be checked). The segmentation tool is the problem: tools
A and B are 170-230 MB. A phone-sized tool (a few MB, e.g. Google's MediaPipe segmentation) would be needed,
and must be re-tested for accuracy, because edges move girths (0.8 cm per pixel at the waist).
**Verdict:** worth it for the long run (offline field use and privacy are strong reasons), but only after
the server version works and the segmentation tool is settled. It becomes its own step (S9).

## Acceptance criteria (testable)

Built 26 Nov 2026 (tests/test_s7c.py; CI job `docker` builds and starts the image). Notebook 08 prepares the
server folder and times the API in Colab. Host-specific deploy steps follow Ryan's host choice.


- [x] `GET /v1/health`, `GET /v1/catalogue`, `POST /v1/measure` answer as in the table, tested with FastAPI's
      test client and a fake model and segmenter (no downloads in CI).
- [x] A request's answer equals `predict-photos` on the same photos and details.
- [x] Refusals come back as 422 with the same reasons; wrong or missing key 401; photo over 15 MB 413.
- [x] Nothing is kept after a request and no photo or value appears in logs (tested).
- [x] `docker build` succeeds and the container answers `/v1/health` (checked in CI without the model files,
      which only the private build has).
- [ ] Deployed on the chosen host by Ryan following `docs/DEPLOY.md`; timing of 5 requests recorded.
- [ ] Front-end doc has the API section. Coverage stays ≥ 80%.

## Security, licence and privacy checks
- New packages `fastapi`, `uvicorn`, `python-multipart` (MIT / BSD / Apache-2.0), in a `serve` extra;
  `httpx` for tests (BSD). Approved first (question 1).
- The model file and image stay private (D-014 still holds: never in GitHub). Keys and passwords live in the
  host's secret settings, kept by Ryan.
- Non-commercial use only (BodyM licence).

## Ryan's answers (26 Nov 2026)

1. Approved, with the packages (D-045). 2. Host: Azure credit used up; Ryan asked about Cloud Run's always-free
allowance and is interested in Oracle; choice pending. 3. Model file: explained (it travels inside the image,
wherever the image runs); inside the image unless the host needs otherwise. 4. Tool A by default (D-046).

## Questions as asked

1. **Approve S7c**, including the new packages (`fastapi`, `uvicorn`, `python-multipart`, and `httpx` for tests)?
2. **Host:** Azure for Students (no card; recommended), Google Cloud Run (needs a card), Oracle Always Free
   (needs a card; you manage the server), or start with your laptop and a tunnel?
3. **Where the model file lives:** baked into a private Docker image (recommended: simplest, no download at
   start), or downloaded at start from private storage with a secret key?
4. **Which segmentation tool** the service uses until notebook 07 picks one: A (torchvision, no extra
   package) as the default, switchable by a setting?
