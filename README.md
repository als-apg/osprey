# Osprey Framework

**🎉 Latest Release: v0.9.5** - Pluggable Code Generator System & Python Executor Refactoring

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

If you use the Alpha Berkeley Framework in your research or projects, please cite our [paper](https://arxiv.org/abs/2508.15066):

```bibtex
@misc{hellert2025osprey,
      title={Osprey: A Scalable Framework for the Orchestration of Agentic Systems},
      author={Thorsten Hellert and João Montenegro and Antonin Sulc},
      year={2025},
      eprint={2508.15066},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2508.15066},
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