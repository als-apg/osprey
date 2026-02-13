# Osprey Framework

[![CI](https://github.com/als-apg/osprey/workflows/CI/badge.svg)](https://github.com/als-apg/osprey/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/osprey-framework/badge/?version=latest)](https://als-apg.github.io/osprey/)
[![codecov](https://codecov.io/gh/als-apg/osprey/branch/main/graph/badge.svg)](https://codecov.io/gh/als-apg/osprey)
[![PyPI version](https://badge.fury.io/py/osprey-framework.svg)](https://badge.fury.io/py/osprey-framework)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**🎉 Latest Release: v0.11.0** - Built-in Capabilities & ARIEL Logbook Search

> **🚧 Early Access Release**
> This is an early access version of the Osprey Framework. While the core functionality is stable and ready for experimentation, documentation and APIs may still evolve. We welcome feedback and contributions!

A production-ready framework for deploying agentic AI in large-scale, safety-critical control system environments—particle accelerators, fusion experiments, beamlines, and complex scientific facilities.

**📄 Research**
This work was presented as a contributed oral presentation at [ICALEPCS'25](https://indico.jacow.org/event/86/overview) and will be featured at the [Machine Learning and the Physical Sciences Workshop](https://ml4physicalsciences.github.io/2025/) at NeurIPS 2025.


## 🚀 Quick Start

```bash
# Install the framework
pip install osprey-framework

# Recommended: Interactive setup (guides you through everything!)
osprey

# The interactive menu will:
# - Help you choose a template with descriptions
# - Guide you through AI provider and model selection
# - Automatically detect and configure API keys from your environment
# - Create a ready-to-use project with smart defaults

# Alternative: Direct command if you know what you want
osprey init my-weather-agent --template hello_world_weather
cd my-weather-agent
# If API keys aren't in your environment, copy and edit .env:
# cp .env.example .env

# Start the command line chat interface
osprey chat
```


## 📚 Documentation

**[📖 Read the Full Documentation →](https://als-apg.github.io/osprey)**

### 🧪 Testing

```bash
# Run unit tests (fast, no API keys required)
pytest tests/ --ignore=tests/e2e -v

# Run e2e tests (slow, requires API keys)
# ⚠️ IMPORTANT: Use 'pytest tests/e2e/' NOT 'pytest -m e2e'
pytest tests/e2e/ -v
```

See [TESTING_GUIDE.md](TESTING_GUIDE.md) and [tests/e2e/README.md](tests/e2e/README.md) for details.


## Key Features

- **Plan-First Orchestration** - Complete execution plans with explicit dependencies before any hardware interaction
- **Control-System Safety** - Pattern detection, PV boundary checking, and mandatory approval for hardware writes
- **Protocol-Agnostic Integration** - Seamless connection to EPICS, LabVIEW, Tango, and mock environments
- **Scalable Capability Management** - Dynamic classification prevents prompt explosion as toolsets grow
- **Production-Proven** - Deployed at major facilities including LBNL's Advanced Light Source accelerator

---

## 📖 Citation

If you use the Osprey Framework in your research or projects, please cite our [paper](https://doi.org/10.1063/5.0306302):

```bibtex
@article{10.1063/5.0306302,
      author = {Hellert, Thorsten and Montenegro, João and Sulc, Antonin},
      title = {Osprey: Production-ready agentic AI for safety-critical control systems},
      journal = {APL Machine Learning},
      volume = {4},
      number = {1},
      pages = {016103},
      year = {2026},
      month = {02},
      doi = {10.1063/5.0306302},
      url = {https://doi.org/10.1063/5.0306302},
}
```

---

*For detailed installation instructions, tutorials, and API reference, please visit our [complete documentation](https://als-apg.github.io/osprey).*

---

**Copyright Notice**

Osprey Framework Copyright (c) 2025, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to do so.

---