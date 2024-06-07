docker run -it --rm --name pachyderm \
  -v $SHARED/aaji:/share -v $HOME:/host-home -w /host-home \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --shm-size 8G \
  pachyderm:latest
