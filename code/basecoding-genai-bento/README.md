# Simple Python model via bentoml.picklable_model

### Setup
1. add .py files required for the model
2. update bentofile.yaml to include the model file and any required python packages
3. update names in service.py

### Run Locally and Deploy to Cloud Run
1. Install dependencies:

```bash
pip install -r ./requirements.txt
```

2. Save the simple python model:

```bash
python ./save_model.py
```

3. Run the service:

```bash
bentoml serve service.py:svc
```

4. Send test request:

```bash
curl -X POST -H "content-type: application/json" --data "[1,2,3,4,5]" http://127.0.0.1:3000/square
```

5. Build Bento

```bash
bentoml build
```

6. Build docker image

```bash
bentoml containerize simple_square_svc:latest
```

```bash
docker run -p 3000:3000 simple_square_svc:nmiekcgrywhoy6uf
```

7. Push to Artifact Registry

Set up:
```bash
# Authenticate with Google Cloud (Only done once as part of set up)
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Configure Docker to use gcloud as a credential helper (Only done once as part of set up)
gcloud auth configure-docker
```

Push the image to Google Artifact Registry:
```bash
# Tag the local Docker image
docker tag simple_square_svc:nmiekcgrywhoy6uf LOCATION-docker.pkg.dev/YOUR_PROJECT_ID/REPOSITORY/IMAGE:TAG

# Push the image to Google Artifact Registry
docker push LOCATION-docker.pkg.dev/YOUR_PROJECT_ID/REPOSITORY/IMAGE:TAG
```

Example:
```bash
# Tag the local Docker image
docker tag simple_square_svc:nmiekcgrywhoy6uf us-central1-docker.pkg.dev/spins-retail-solutions/demos/simple_square_svc:nmiekcgrywhoy6uf
# Push the image to Google Artifact Registry
docker push us-central1-docker.pkg.dev/spins-retail-solutions/demos/simple_square_svc:nmiekcgrywhoy6uf
```

8. Deploy to Cloud Run

#TODO: Add instructions for deploying to Cloud Run

Test the deployed service:
```bash
# Note that in this case the cloud 
curl -X POST -H "content-type: application/json" --data "[1,2,3,4,5]" CLOUD_RUN_URL/square 
```
example:
```bash
curl -X POST -H "content-type: application/json" --data "[1,2,3,4,5]" https://simple-square-22nj5y2lda-uc.a.run.app/square
```

View swagger docs by going to the same url (in this case: https://simple-square-22nj5y2lda-uc.a.run.app) 