# Docker Setup for imsearch_benchmarks

This directory contains the required files to build and work in a dockerized environment with GPU support for the `imsearch_benchmarks` service.

## Contents

- **Dockerfile**: Instructions for building the Docker image.
- **docker-compose.yml**: Compose configuration to easily run the container with all necessary settings.
- **requirements.txt**: Python dependencies.
- **volumes/**: Directory for persistent data, configs, and input/output.

## Quick Start

### 1. Build the Docker Image

```bash
docker-compose build
```

### 2. Run the Container

```bash
docker-compose up
```

This will start the container and mount the following local directories:
- `./volumes/config` &rarr; `/app/config`
- `./volumes/inputs` &rarr; `/app/input`
- `./volumes/outputs` &rarr; `/app/output`

You can place configs, and input files into these folders as needed.

### 3. Environment Variables

You may set the following environment variables (either in a `.env` file or directly when running):

- `IMSEARCH_BENCHMAKER_CONFIG_PATH` (path to the config file, e.g. `~/imsearch_benchmarks/FireBenchMaker/config.toml`)
- `NVIDIA_VISIBLE_DEVICES` (default: `all`)
- adapter-specific environment variables (e.g. `OPENAI_API_KEY`, `HF_TOKEN`, `SAGE_USER`, `SAGE_PASSWORD`)

### 4. GPU Support

The container is configured to use NVIDIA GPUs via the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html).
- Ensure you have the NVIDIA drivers and the toolkit installed on your host machine.

### 5. Enter the Container

For an interactive session:

```bash
docker-compose run --rm imsearch_benchmarks /bin/bash
```

### 6. Exiting

To stop and remove containers:

```bash
docker-compose down
```

---

For more details, see the individual files or contact the repository maintainer.


