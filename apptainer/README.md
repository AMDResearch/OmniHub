## Apptainer

The Apptainer image is based on the Docker image, automatically converted from
the Dockerfile using `spython`.

In the HPC Fund cluster, a pre-built apptainer image is available in the
shared directory: `/work1/amd/omnihub/apptainer/omnihub.sif`.

## Build images

To build your own image:

```
apptainer build $WORK/omnihub.sif apptainer/omnihub-ubuntu-dev.def
```

## Run container

```
apptainer run --rocm /work1/amd/omnihub/apptainer/omnihub.sif
```
