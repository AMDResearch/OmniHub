## Apptainer

The Apptainer image is based on the Docker image, automatically converted from
the Dockerfile using `spython`. For now, Omnitrace needs to be disabled in the
Apptainer builds.

Apptainer images are available in two flavors: `torchrun` and `mpi`. Both
images are very similar, but meant to be launched in slightly different
environments.

In the HPC Fund cluster, pre-built apptainer images for both variants are
available in the shared directory:
 - `/work1/amd/omnihub/apptainer/omnihub-torchrun.sif`.
 - `/work1/amd/omnihub/apptainer/omnihub-mpi.sif`.

## Build images

To build your own images:

```
apptainer build $WORK/omnihub-torchrun.sif apptainer/omnihub-ubuntu-dev-torchrun.def
apptainer build $WORK/omnihub-mpi.sif apptainer/omnihub-ubuntu-dev-mpi.def
```

## Run container

```
apptainer run --rocm /work1/amd/omnihub/apptainer/omnihub-torchrun.sif
apptainer run --rocm /work1/amd/omnihub/apptainer/omnihub-mpi.sif
```
