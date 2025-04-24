FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

RUN mkdir -p /mnist_classifier/data
WORKDIR /mnist_classifier

# copy the minimum required files:
## the uv files needed to recreate the environment
COPY pyproject.toml uv.lock ./
## the prediction script
COPY predict.py .
## the model weights
COPY data/mnist_cnn.pt data/

# recreate the environment
RUN uv sync --locked

CMD ["uv", "run", "predict.py"]
