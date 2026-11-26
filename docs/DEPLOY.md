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

**Choose a key and keep it out of files and history.** In each terminal you use, type this; it asks for the
key without showing it, so it never lands in a file or your shell history:

```bash
read -rs ANTHRO_KEY && export ANTHRO_KEY
```

**Run it:**

```bash
docker run --rm -p 8080:8080 -e ANTHRO_API_KEY="$ANTHRO_KEY" anthro-api
```

**Check it** (in another terminal, after the same `read` line):

```bash
curl http://localhost:8080/v1/health
curl -H "X-API-Key: $ANTHRO_KEY" \
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

## 3. In the cloud: Oracle Cloud Always Free (D-048)

A free Arm server (Ampere A1) that you own and manage. Plan about two hours the first time. Every command
below runs either on **your laptop** or **on the server**, as marked.

### 3.1 Account and server (Oracle website)

1. Sign up at https://www.oracle.com/cloud/free/ . The card is for identity checking; stay on the free
   tier (do not upgrade to "Pay As You Go" unless you mean to). Home region: **South Africa Central
   (Johannesburg)** (D-049; it cannot be changed later).
2. Console → **Compute → Instances → Create instance**:
   - **Image:** Canonical Ubuntu 24.04 (the aarch64 / Arm version).
   - **Shape:** Ampere → `VM.Standard.A1.Flex`, **2 OCPUs, 12 GB** (inside the free 4 OCPUs / 24 GB, leaving
     room for a second server later). It must show "Always Free-eligible".
   - **SSH keys:** "Generate a key pair", **download the private key** and keep it safe (it is the only way in).
   - If it says "Out of capacity": Johannesburg has one availability domain, so try again a few hours later,
     or start at 1 OCPU / 6 GB and resize later (Instance → Edit → shape).
3. Note the instance's **public IP address** (for example `129.151.10.20`).
4. **Open the web ports:** instance → subnet → **Security List** → **Add ingress rules**: source `0.0.0.0/0`,
   TCP, destination port `80`; and again for `443`. (Port 22, for SSH, is already open.)

### 3.2 First login and firewall (laptop, then server)

```bash
# laptop: protect the key file, then log in
chmod 600 ~/Downloads/ssh-key.key
ssh -i ~/Downloads/ssh-key.key ubuntu@129.151.10.20
```

```bash
# server: Oracle's Ubuntu image also blocks ports inside the machine; open 80 and 443 there too
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
sudo apt update && sudo apt upgrade -y
```

### 3.3 Docker (server)

```bash
curl -fsSL https://get.docker.com | sudo sh     # Docker's official install script
sudo usermod -aG docker ubuntu && exit           # log out, then ssh in again so the group applies
```

### 3.4 Code, model folder and key

```bash
# server: get the code (public repository; no model inside)
git clone https://github.com/Anthropometry/Anthropometry_Accademic_Model.git
```

```bash
# laptop: copy the private model folder (from notebook 08) to the server
scp -i ~/Downloads/ssh-key.key serving/model.pt serving/eval_report.json serving/MANIFEST.json \
    ubuntu@129.151.10.20:~/Anthropometry_Accademic_Model/serving/
```

```bash
# server: check the copy is complete (compare with the sha256 values in MANIFEST.json)
cd ~/Anthropometry_Accademic_Model && sha256sum serving/model.pt serving/eval_report.json
# make a long random API key, stored only in a file only you can read
printf 'ANTHRO_API_KEY=%s\n' "$(openssl rand -hex 32)" > ~/anthro.env && chmod 600 ~/anthro.env
cat ~/anthro.env     # copy the key once into your password manager; give it to the app team privately
```

### 3.5 Build and run the API (server)

```bash
cd ~/Anthropometry_Accademic_Model
docker build -t anthro-api .                     # 20-40 minutes the first time on Arm
docker run -d --name anthro --restart unless-stopped --env-file ~/anthro.env \
    -p 127.0.0.1:8080:8080 anthro-api            # reachable only from the server itself (Caddy is the door)
docker logs -f anthro                            # wait for "Application startup complete", then Ctrl+C
```

### 3.6 HTTPS with Caddy and a free address (server)

Caddy gets a free HTTPS certificate automatically. For a free web address without buying a domain, use
`sslip.io`: your IP with dashes, e.g. `129-151-10-20.sslip.io` (it always points to that IP).

```bash
cd ~ && cat > Caddyfile <<'CADDY'
129-151-10-20.sslip.io {
    request_body {
        max_size 32MB
    }
    reverse_proxy 127.0.0.1:8080
}
CADDY
docker run -d --name caddy --restart unless-stopped --network host \
    -v ~/Caddyfile:/etc/caddy/Caddyfile -v caddy_data:/data caddy:2
```

**Check it from your laptop:**

```bash
curl https://129-151-10-20.sslip.io/v1/health
```

The app then uses `https://129-151-10-20.sslip.io` as the API address (and `/docs` shows the API page).
Optionally set `ANTHRO_ALLOWED_ORIGINS` in `~/anthro.env` to the app's web address and restart (3.7).

### 3.7 Updating (server)

```bash
cd ~/Anthropometry_Accademic_Model && git pull
docker build -t anthro-api . && docker rm -f anthro
docker run -d --name anthro --restart unless-stopped --env-file ~/anthro.env -p 127.0.0.1:8080:8080 anthro-api
```

A new model: replace the files in `serving/` (3.4, with a fresh `MANIFEST.json`) and run the three update lines.

### 3.8 Keeping it safe and free

- Only ports 22, 80 and 443 are open; the API itself listens only inside the server.
- Log in only with the SSH key; never share the key file or put it in a repository.
- Ubuntu installs security updates automatically; run `sudo apt update && sudo apt upgrade -y` now and then.
- Oracle can reclaim an Always Free server that stays almost idle for a long time; check its Free Tier page,
  and use the API now and then.
- Stay within 4 OCPUs and 24 GB in total, and do not upgrade the account unless you choose to pay.

## Settings (environment variables)

| Variable | Meaning | Default |
| --- | --- | --- |
| `ANTHRO_API_KEY` | key every request sends as `X-API-Key` | required |
| `ANTHRO_TOOL` | photo-cutting tool: `torchvision` or `rembg` | `torchvision` (D-046) |
| `ANTHRO_ALLOWED_ORIGINS` | web addresses allowed to call from a browser, comma-separated | none |
| `ANTHRO_MAX_MB` | largest photo accepted (MB) | 15 |
| `PORT` | port inside the container | 8080 |
| `ANTHRO_ALLOW_NO_KEY` | `1` to run without a key: laptop practice only, never online | off |
