# AWS Deployment Guide

This guide documents how to run the stack (FastAPI + Streamlit dashboard) on AWS.
The primary path is a single **EC2** instance running `docker compose`. An
**ECR + ECS** alternative is outlined at the end.

> Status: these steps are documented, not deployed live. No cloud spend is
> required to use this project.

## What gets deployed

| Service   | Container port | Default host port | Check                    |
|-----------|----------------|-------------------|--------------------------|
| API       | 8000           | `API_PORT` (8000) | `GET /health`, docs at `/docs` |
| Dashboard | 8501           | `DASHBOARD_PORT` (8501) | open in browser    |

The dashboard talks to the API over the Docker network (`http://api:8000`), and
starts only after the API healthcheck passes. The model, dataset and source are
baked into the images, so no S3 bucket or AWS credentials are needed at runtime.

## Option A: EC2 with Docker Compose

### 1. Launch an instance
- AMI: Ubuntu 24.04 LTS
- Type: `t3.small` or larger (CatBoost + SHAP need some memory; use `t3.medium` if builds are slow)
- Storage: 20 GB
- Key pair: create or select one to SSH in

### 2. Security group (inbound rules)
| Port | Source        | Purpose        |
|------|---------------|----------------|
| 22   | Your IP only  | SSH            |
| 8501 | Your IP (or 0.0.0.0/0 for a public demo) | Dashboard |
| 8000 | Your IP only (optional) | API / `/docs` |

Do not open ports to the world unless you intend a public demo.

### 3. Install Docker
```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
exit   # log in again so the group change applies
```

### 4. Get the code and start the stack
```bash
git clone https://github.com/vidityende727-code/purchase-intent-xai.git
cd purchase-intent-xai
cp .env.example .env          # optional: change ports here
docker compose up -d --build
docker compose ps             # api should become "healthy"
```

### 5. Verify
- API health: `curl http://localhost:8000/health`
- API docs: `http://<EC2_PUBLIC_IP>:8000/docs`
- Dashboard: `http://<EC2_PUBLIC_IP>:8501`

### 6. Operate
```bash
docker compose logs -f          # view logs
docker compose down             # stop the stack
git pull && docker compose up -d --build   # update
```

### 7. Cost control
Stop or terminate the instance when you are done. A stopped instance does not
bill for compute (storage still bills); the public IP changes on restart unless
you attach an Elastic IP.

## Configuration

All deployment settings are parameters, not hard-coded values:

| Variable         | Default | Where used                         |
|------------------|---------|------------------------------------|
| `API_PORT`       | 8000    | Host port for the API (compose)    |
| `DASHBOARD_PORT` | 8501    | Host port for the dashboard (compose) |
| `API_URL`        | `http://localhost:8000` | Dashboard, when run without Docker |

Never commit a real `.env`; only `.env.example` is tracked.

## Option B: ECR + ECS (outline)

1. Create two ECR repositories (e.g. `purchase-intent-api`, `purchase-intent-dashboard`).
2. Build, tag and push each image:
```bash
   export AWS_REGION=<region>
   export ACCOUNT_ID=<aws-account-id>
   aws ecr get-login-password --region $AWS_REGION | \
     docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

   docker build -f Dockerfile.api -t purchase-intent-api .
   docker tag purchase-intent-api:latest $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/purchase-intent-api:latest
   docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/purchase-intent-api:latest
```
   Repeat with `Dockerfile.dashboard` for the dashboard image.
3. Create an ECS service (Fargate) per image. API: container port 8000, health
   check `/health`. Dashboard: container port 8501, environment variable
   `API_URL` pointing to the API service's address.
4. Open the matching ports in the service security groups.

Prefer EC2 + Compose for a first deployment; it reuses `docker-compose.yml`
unchanged.