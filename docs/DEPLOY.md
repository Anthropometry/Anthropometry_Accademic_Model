# Deploying the model's web API (S7c)

specs/s7c-serve/spec.md. The same Docker image runs on a laptop or a cloud server. Start on the laptop to
learn the steps; then do the same on the chosen cloud host.

**Rules that never change**
- The model folder (`serving/`) and any image built with it stay **private**: never in GitHub, never a public
  image (D-014; BodyM licence CC BY-NC 4.0: non-commercial use only).
- The API key is chosen by Ryan, kept in a password manager or the host's secret settings, never in code,
  notebooks, chats or commits.

## 0. What Docker is, in one paragraph

An **image** is a packaged box: our code, Python, PyTorch, the model file and the photo-cutting tool. A
**container** is that box running. The same image behaves the same on any machine that has Docker, so what
works on the laptop works on the server. The box answers web requests on port 8080.

## 1. Get the model folder (once)

1. In Colab, run `notebooks/08_s7c_serve.ipynb` cell 3: it writes `MyDrive/anthro/serving/` with
   `model.pt` (~45 MB), `eval_report.json` and `MANIFEST.json`.
2. Download those three files and put them in the `serving/` folder of your copy of this repository
   (next to `serving/README.md`). Git ignores them.

## 2. On your laptop (learning and live demos)

**Install:** Docker Desktop (Windows or Mac) or Docker Engine (Linux), from docker.com.

**Build the image** (in the repository folder; the first build downloads PyTorch, about 10-20 minutes):

```bash
docker build -t anthro-api .
```

**Run it** (pick your own long key; this one is only an example):

```bash
docker run --rm -p 8080:8080 -e ANTHRO_API_KEY=replace-with-a-long-random-key anthro-api
```

**Check it** (in another terminal):

```bash
curl http://localhost:8080/v1/health
curl -H "X-API-Key: replace-with-a-long-random-key" \
     -F front=@front.jpg -F side=@side.jpg \
     -F height_cm=174 -F sex=female -F age_range=25-34 -F weight_kg=60 \
     http://localhost:8080/v1/measure
```

The interactive API page is at http://localhost:8080/docs (try requests from the browser).

**Reach it from a phone outside your Wi-Fi (optional):** a tunnel gives the laptop a temporary public HTTPS
address. With Cloudflare's free quick tunnel (install `cloudflared` first):

```bash
cloudflared tunnel --url http://localhost:8080
```

It prints an address like `https://something.trycloudflare.com`; the app uses that instead of localhost.

**Know the limits:** it works only while the laptop is on, awake and online; closing the lid stops it; the
address changes each time; anyone with the address can try it (the API key is what stops them). Close the
tunnel when you are done.

**Stop it:** press Ctrl+C in the terminal running `docker run` (and in the tunnel's terminal).

## 3. In the cloud (class demo)

The host is being chosen (Ryan, 26 Nov 2026): **Oracle Always Free** (recommended: free permanently, a full
server you manage) or **Google Cloud Run** (always-free monthly allowance after the trial; needs a billing
account with a card). The step-by-step section for the chosen host is added here once it is picked.

## Settings (environment variables)

| Variable | Meaning | Default |
| --- | --- | --- |
| `ANTHRO_API_KEY` | key every request sends as `X-API-Key` | required |
| `ANTHRO_TOOL` | photo-cutting tool: `torchvision` or `rembg` | `torchvision` (D-046) |
| `ANTHRO_ALLOWED_ORIGINS` | web addresses allowed to call from a browser, comma-separated | none |
| `ANTHRO_MAX_MB` | largest photo accepted (MB) | 15 |
| `PORT` | port inside the container | 8080 |
| `ANTHRO_ALLOW_NO_KEY` | `1` to run without a key: laptop practice only, never online | off |
