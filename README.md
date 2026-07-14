# Osprey

[![CI](https://github.com/als-apg/osprey/actions/workflows/ci.yml/badge.svg)](https://github.com/als-apg/osprey/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/osprey-framework)](https://pypi.org/project/osprey-framework/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE.txt)
[![DOI](https://img.shields.io/badge/DOI-10.1063%2F5.0306302-blue)](https://doi.org/10.1063/5.0306302)

**An agentic interface and safety harness for safety-critical control systems.**

Osprey addresses control-specific challenges: semantic addressing across large channel
namespaces, protocol-agnostic integration with control stacks, intelligent logbook search
across facility electronic logbooks, and mandatory human oversight for every hardware write.

Built for particle accelerators, fusion experiments, beamlines, and large scientific facilities.

<p align="center">
  <img src="docs/source/_static/resources/architecture.png" width="100%"
       alt="Osprey system architecture, from operator to facility, with the safety gate and approval workflow in-line." />
</p>

## Quick start

```bash
# Install the framework as a standalone CLI tool (using uv, recommended)
uv tool install osprey-framework

# Create a minimal project to verify your setup
osprey build quickstart --preset hello-world
cd quickstart

# If API keys aren't already in your environment, copy and edit .env:
# cp .env.example .env

# Start an agent session
claude
```

For a project tailored to your detector, beamline, or accelerator subsystem, install the
guided build-interview skill and run it from your agent session:

```bash
osprey skills install osprey-build-interview
```

Then start the agent in an empty directory and type `/osprey-build-interview`. The skill
walks you through a guided conversation, produces a build profile, and
`osprey build profile.yml` generates a ready-to-use project.

## Key features

- **Agent-driven orchestration** — Skills, MCP tools, and explicit dependency declarations
  let the Osprey agent decompose operator requests into auditable steps with mandatory
  approval gates.
- **Control-system safety** — Pattern detection, channel boundary checking, and mandatory
  human approval for every hardware write.
- **Protocol-agnostic integration** — EPICS and Mock connectors ship in-tree; LabVIEW, Tango,
  and other stacks connect through the
  [connector interface](https://als-apg.github.io/osprey/how-to/add-connector.html).
- **Replaceable backends** — The agent harness, the underlying model, and the compute backend
  are each swappable by configuration, without changing what the operator sees.
- **Scalable capability management** — Dynamic classification prevents prompt explosion as
  toolsets grow.

## Documentation

**[Read the full documentation →](https://als-apg.github.io/osprey)**

Osprey follows CalVer (`vYYYY.M.P`). Public APIs may change between releases — pin a version
and check the [changelog](CHANGELOG.md) before upgrading.

## Contributing

Contributions are welcome. See the [Contributing Guide](CONTRIBUTING.md) for development
setup, coding standards, and the pull-request workflow. To report a security issue, please
follow the [security policy](SECURITY.md) rather than opening a public issue.

## Citation

If you use Osprey in your research, please cite the
[paper](https://doi.org/10.1063/5.0306302). GitHub's **Cite this repository** button, in the
sidebar, exports BibTeX and APA directly.

## License

BSD 3-Clause — see [LICENSE.txt](LICENSE.txt). Additional notices, including the
U.S. Department of Energy's retained rights and Berkeley Lab's Enhancements grant, are in
[NOTICE](NOTICE).

Copyright (c) 2025, The Regents of the University of California, through Lawrence Berkeley
National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of
Energy). All rights reserved.

Questions about your rights to use or distribute this software: contact Berkeley Lab's
Intellectual Property Office at IPO@lbl.gov.
