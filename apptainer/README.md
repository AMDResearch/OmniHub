# Apptainer

The Apptainer definition file in this directory is auto-generated from the
corresponding Dockerfile in the `docker` directory using `spython`. Modifications to the Dockerfile
will trigger a GitHub action, which generates the Apptainer definition file like below.
```
spython recipe docker/Dockerfile-ubuntu-dev > apptainer/omnihub-ubuntu-dev.def
```

## Image Organization
Pre-built Apptainer images are stored on the following clusters and the corresponding locations.

- `radha:/home/shared/projs/omnihub/apptainer/omnihub.sif`
- `hpcfund:/work1/amd/omnihub/apptainer/omnihub.sif`

Images were built using the command `apptainer build /path/to/omnihub.sif apptainer/omnihub-ubuntu-dev.def`

## Run container

To interactively run the container (e.g., on the HPC Fund cluster):
```
apptainer run /work1/amd/omnihub/apptainer/omnihub.sif
```
