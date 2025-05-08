# SAOS: Solar Adaptive Optics Simulator 🌞🔭

**SAOS** is a new simulator designed to support Solar Adaptive Optics simulations. Its core structure and modular philosophy are based on **OOPAO** (Object Oriented Python Adaptive Optics), created by Cédric Taïssir Héritier. You can find the original project here: [OOPAO GitHub Repository](https://github.com/cheritier/OOPAO/tree/master).

SAOS modifies the internal architecture of OOPAO, which was heavily managed through the telescope class, to offer a framework centered on lines of sight, implemented via the **LightPath** class. This architectural shift enables easier and more powerful parallelization, essential for solar AO simulations where the Field of View (FoV) is divided into multiple sub-directions to physically introduce extended field effects into the simulation.

SAOS is currently in the testing phase before its first release. All development is ongoing in the **tech-development** branch. We recommend waiting a little longer before using SAOS in your work!

A key feature of SAOS is that it is a **Python-only** repository. All parallelization is managed via **joblib** and threads, leveraging the new **free-threaded Python 3.13** to enable extensive and straightforward parallelization.

# Guidelines

- If you are primarily a night-time AO user and do not require multi-conjugate AO (MCAO) or you need a pyramid WFS simulations, we recommend using **OOPAO** instead. See: [OOPAO AO4ELT7 Proceedings (Héritier et al. 2023)](https://hal.science/AO4ELT7/hal-04402878v1).

- If you need to simulate multiple lines of sight or perform Solar AO simulations, **SAOS** is the repository for you.  
  We have not yet published a full article describing SAOS, but it is in preparation —please keep an eye on the repository for updates!
