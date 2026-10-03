# Verifier image: runs /tests/test.sh against the copied-out submission.
# numpy/scipy are pinned to the versions the verifier and the authoring
# evidence were validated with; keep them identical to environment/Dockerfile.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# pytest's own dependencies are pinned too, for reproducible rebuilds.
RUN pip install \
        numpy==2.4.6 \
        scipy==1.17.1 \
        pytest==8.4.1 \
        pytest-json-ctrf==0.3.5 \
        iniconfig==2.3.0 \
        packaging==26.3 \
        pluggy==1.6.0 \
        pygments==2.21.0

COPY . /tests
# /app/output: mount point for the artifact (/app/output/submission.json).
RUN chmod +x /tests/test.sh \
 && mkdir -p /app/output \
 && cd /tests \
 && python -c "import verifier.mesh_checks, verifier.fem_ref, verifier.error; verifier.mesh_checks.load_config()"

WORKDIR /tests
