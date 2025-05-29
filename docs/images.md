# Containerization

We have pre-built both Docker and Apptainer OmniHub images on Radha and Apptainer images on HPC Fund. The images include
all the supported performance measurement and tracing tools. We highly recommend using Apptainer for multi-node runs.

## Docker

Pre-built Docker images are available on the below artifactories. On the specific compute node, pull one of
the following before using the OmniHub scripts. In fact, for your convenience, it is very likely that we
may have already done a `docker pull` on these images on each Radha node.

- `docker pull docker-virtual.atlartifactory.amd.com/amd/omnihub/radha:gfx90a.631`
- `docker pull docker-virtual.atlartifactory.amd.com/amd/omnihub/radha:gfx942.631`

## Apptainer

Pre-built Apptainer images are stored on the following clusters at the corresponding shared locations.

- `radha:/home/shared/projs/omnihub/apptainer/omnihub.gfx90a.631.sif`
- `radha:/home/shared/projs/omnihub/apptainer/omnihub.gfx942.631.sif`
- `hpcfund:/work1/amd/omnihub/apptainer/omnihub.gfx90a.631.sif`
- `hpcfund:/work1/amd/omnihub/apptainer/omnihub.gfx942.631.sif`

### How we built the images

Apptainer Images were built based on the docker images by using the command:
`apptainer build /path/to/omnihub.<arch>.<rocmver>.sif docker-virtual.atlartifactory.amd.com/amd/omnihub/radha:<arch>.<rocmver>`
