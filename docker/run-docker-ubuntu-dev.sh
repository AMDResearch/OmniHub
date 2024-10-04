docker run -it --rm --name omnihub \
  -v /home/shared/projs/omnihub:/share -v $HOME:/host-home -w /host-home \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device=/dev/kfd --device=/dev/dri \
  --network=host \
  --ipc=host --shm-size 8G \
  docker-virtual.atlartifactory.amd.com/amd/omnihub/radha:latest
