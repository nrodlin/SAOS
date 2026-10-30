# SAOS: Solar Adaptive Optics Simulator 🌞🔭

**SAOS** is a new simulator designed to support Solar Adaptive Optics simulations. Its core structure and modular philosophy are based on **OOPAO** (Object Oriented Python Adaptive Optics), created by Cédric Taïssir Héritier. You can find the original project here: [OOPAO GitHub Repository](https://github.com/cheritier/OOPAO/tree/master).

SAOS modifies the internal architecture of OOPAO, which was heavily managed through the telescope class, to offer a framework centered on lines of sight, implemented via the **LightPath** class. This architectural shift enables easier and more powerful parallelization, essential for solar AO simulations where the Field of View (FoV) is divided into multiple sub-directions to physically introduce extended field effects into the simulation.

A key feature of SAOS is that it is a **Python-only** repository. All parallelization is managed via **joblib** and threads, leveraging the new **free-threaded Python 3.13** to enable extensive and straightforward parallelization. Pytorch is extensively used in the code for costly operations, being ready for an upgrade to a GPU version.

---

## 📖 Documentation

Full documentation — installation guides, pedagogical tutorials, and API reference — is available at:

**➡️ [https://nrodlin.github.io/SAOS/](https://nrodlin.github.io/SAOS/)**

To build the documentation locally:
```bash
pip install -r requirements.txt   # or: pip install mkdocs mkdocs-material mkdocs-jupyter mkdocstrings[python]
mkdocs serve
```

## ⚙️ Installation

For optimal performance, we strongly recommend using a free-threaded Python environment (e.g., Python 3.13.9t). 
A quick installation snippet is provided below. For detailed instructions, please check the [Installation Guide](docs/installation.md).

```bash
git clone https://github.com/nrodlin/SAOS.git
cd SAOS
git checkout develop
pip install -e .
```

