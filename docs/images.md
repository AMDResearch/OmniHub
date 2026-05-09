# Containerization

OmniHub supports both Docker and Apptainer containers. The images include all supported
performance measurement and tracing tools. We highly recommend using Apptainer for
multi-node runs.

## Docker

To use Docker, add a `docker-image` key to your cluster config pointing to your registry
base path. OmniHub appends the tag automatically in the form `{variant}.{arch}.{rocmver}`
(e.g., `inference.gfx942.631`):

```yaml
# config/mycluster.yaml
cluster:
  container-platforms:
    - docker
  docker-image: "your-registry.example.com/omnihub/image-name"
```

Pull the image on each compute node before running:

```console
docker pull your-registry.example.com/omnihub/image-name:inference.gfx942.631
```

## Apptainer

Apptainer images are stored on the cluster at a shared path configured via `shared-dir`
in your cluster config. Images are split by workload type: `inference` for inference
workloads and `training` for training/fine-tuning workloads.

The expected path pattern is:

```
{shared-dir}/apptainer/omnihub.{variant}.{arch}.{rocmver}.sif
```

### Building Apptainer images from Docker

Apptainer images can be built from any compatible Docker image:

```console
apptainer build /path/to/omnihub.inference.<arch>.<rocmver>.sif docker://your-registry.example.com/omnihub/image-name:<arch>.<rocmver>
apptainer build /path/to/omnihub.training.<arch>.<rocmver>.sif docker://your-registry.example.com/omnihub/image-name:<arch>.<rocmver>
```
