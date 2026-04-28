# python3 -m venv venv
source venv/bin/activate

# pip install fastapi uvicorn

# uvicorn api.app:app --reload




# house predict Model
What happened:

CSV → pandas → sklearn → train → saved model.pkl

Now we move to the second phase:

Load saved model → Accept user input → Predict



## Docker

docker build -t house-price-api .

docker run -p 8000:8000 house-price-api

docker run -p 8001:8000 house-price-api

    # 1. docker run
    It creates and starts a container from your Docker image.
    your laptop : docker container
    8001        -> 8000

    Meaning:

Your app inside Docker is running on port 8000.

But you access it from your machine on:

http://localhost:8001