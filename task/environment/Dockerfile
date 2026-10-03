# Agent environment: the provided P1 FEM package, the problem specification
# and the starting mesh in /app; the submission goes to /app/output.
# numpy/scipy are pinned to the same versions as tests/Dockerfile, so code the
# agent develops here behaves identically under the verifier.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install \
        numpy==2.4.6 \
        scipy==1.17.1

# /app/specs.txt, /app/initial_mesh.json, /app/fem/ (package + its unit tests)
COPY data/ /app/
RUN mkdir -p /app/output \
 && cd /app \
 && python -c "import fem.problem, fem.assembly, fem.solver, fem.error, fem.io; fem.io.load_mesh('initial_mesh.json')"

WORKDIR /app
