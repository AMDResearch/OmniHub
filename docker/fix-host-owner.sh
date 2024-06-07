#!/bin/bash
#
# Fix permissions of files generated in the container
#
# Recursively changes the permissions of the given files/directories created
# inside of a container to match the owner and group of the user running the
# container in the host.

if [[ $# -lt 1 ]] || [[ ! -e /host-home ]]; then
    echo "Usage: $0 FILE..."
    exit 1
fi

host_uid=$(stat -c "%u" /host-home)
host_gid=$(stat -c "%g" /host-home)

chown -R $host_uid:$host_gid $@
