# Model Organization

All ML models are stored on the following clusters and the corresponding locations.

- `radha:/shared/omnihub/models`
- `hpcfund:/work1/models/amd`

The models are organized as a flat structure with an approximate naming convention like
`<org>/<model>-<params>-<task>-<format>`. The naming convention comes from HuggingFace. Many of the model directories
also contain the original `.pth` model as well, which may be used by native PT libraries.
