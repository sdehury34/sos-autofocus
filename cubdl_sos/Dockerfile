# CUBDL speed-of-sound comparison (CPU).
#   docker build -t cubdl-sos .
#   docker run --rm -v /path/to/cubdl:/data:ro -v "$PWD/results":/work/results cubdl-sos --data /data
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 JAX_PLATFORMS=cpu
WORKDIR /work
COPY requirements.txt /tool/requirements.txt
RUN pip install -r /tool/requirements.txt
COPY *.py /tool/
ENTRYPOINT ["python", "/tool/run_sos_comparison.py"]
