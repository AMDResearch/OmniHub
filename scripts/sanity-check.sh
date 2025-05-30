#!/bin/bash

set -e

# If the script runs without errors, it means ROCm and other system software are installed correctly.

rocminfo > /dev/null 2>&1
