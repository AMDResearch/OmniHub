# Containerization
We have pre-built Docker and Apptainer OmniHub images on Radha and Apptainer images on HPC Fund. We highly
recommend using Apptainer for multi-node runs. 

## Docker

Pre-build Docker images are available on the below artifactories. On the specific compute node, pull one of 
the following before using the OmniHub scripts. In fact, for your convenience, it is very likely that we
may have already done a `docker pull` on these images on each Radha node.

- `docker pull docker-virtual.atlartifactory.amd.com/amd/omnihub/radha:gfx90a`
- `docker pull docker-virtual.atlartifactory.amd.com/amd/omnihub/radha:gfx942`

### Run container
To interactively run the docker container, run as usual like below (pick the image according to the GPU architecture):

```
docker run -it --rm --name omnihub \
  -v /home/shared/projs/omnihub:/share -v $HOME:/host-home -w /host-home \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device=/dev/kfd --device=/dev/dri \
  --network=host \
  --ipc=host --shm-size 8G \
  docker-virtual.atlartifactory.amd.com/amd/omnihub/radha:gfx90a
```

## Apptainer
Pre-built Apptainer images are stored on the following clusters at the corresponding shared locations.

- `radha:/home/shared/projs/omnihub/apptainer/omnihub.gfx90a.sif`
- `radha:/home/shared/projs/omnihub/apptainer/omnihub.gfx942.sif`
- `hpcfund:/work1/amd/omnihub/apptainer/omnihub.gfx90a.sif`
- `hpcfund:/work1/amd/omnihub/apptainer/omnihub.gfx942.sif`

### Run container

To interactively run the container on HPC Fund:
```
apptainer run /work1/amd/omnihub/apptainer/omnihub.gfx90a.sif
```

To interactively run the container on Radha:
```
apptainer run /home/shared/projs/omnihub/apptainer/omnihub.gfx90a.sif
```

### How we built the images
Apptainer Images were built based on the docker images by using the command 
`apptainer build /path/to/omnihub.<arch>.sif docker-virtual.atlartifactory.amd.com/amd/omnihub/radha:<arch>`
