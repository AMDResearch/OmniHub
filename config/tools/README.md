## Tools

This directory contains configuration files for the tools supported by
OmniHub.

Tools are defined in YAML files. Each tool definition includes a mandatory
unique string `name`, and several optional fields that describe how the tool
is executed:
- `pre-execute`: Commands to be executed in the host before launching the
  application.
- `post-execute`: Commands to be executed in the host after launching the
  application.
- `prefix`: String prepended to the application command.
- `args`: List of CLI arguments that need to be passed to the application.
- `env`: Map of environment variable names and values. In Apptainer
  environments, these variables are simply set in the host before launching
  the application. In Docker environments, these variables are passed as
  arguments to the `docker` command.

The `conflicts` keyword can be used to list other tools that aren't meant to
be executed in the same execution.
