# Labs for Wheels

An isolated extension for wheeled robotics simulation and reinforcement learning, built on Isaac Lab.

## Overview

This project is an **adaptation of [WheeledLab](https://github.com/UWRobotLearning/WheeledLab)** by the University of Washington Robot Learning Lab, restructured as an isolated environment outside of the core Isaac Lab repository. This allows for independent development and extension of wheeled robotics capabilities while maintaining compatibility with the Isaac Lab ecosystem.

**Key Features:**

- **Isolation**: Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained
- **Flexibility**: This extension is set up to allow your code to be run as an extension in Omniverse
- **Modularity**: Clean separation of tasks, assets, and configurations for easy extension
- **RL-Ready**: Pre-configured environments for reinforcement learning with RSL-RL and skrl

**What can the Hot Wheels do? Well, they can perform:**

- Spatial reasoning for obstacle detection
- Visual semantic navigation
- Advanced waypoint following
- Drifting

## Prerequisites

Before installing this extension, ensure you have the following:

- **Ubuntu 22.04+** (or Windows with WSL2)
- **CUDA-capable GPU** (NVIDIA GPU recommended)
- **Python 3.10 or 3.11**
- **Isaac Sim 5.1.0** (latest recommended)
- **Isaac Lab** (latest version recommended)

## Installation

### Step 1: Install Isaac Sim and Isaac Lab

We recommend using the latest versions for the best compatibility and features:

**Install Isaac Sim 5.1.0:**

```bash
# Create a conda environment (you can name it anything, e.g., 'labs_for_wheels')
conda create -n wheeled_lab python=3.10
conda activate wheeled_lab

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121  # For CUDA 12.1
# OR
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8

# Install Isaac Sim 5.1.0
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
```

**Install Isaac Lab (Latest Version):**

```bash
# Clone Isaac Lab (use the latest version)
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Install Isaac Lab (make sure you have build dependencies: cmake, build-essential)
./isaaclab.sh -i
```

For detailed installation instructions, see the [Isaac Lab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

### Step 2: Install Labs for Wheels Extension

1. **Clone this repository** separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):

```bash
cd ~/Development  # or your preferred development directory
git clone <your-repo-url> labs-for-wheels
cd labs-for-wheels
```

2. **Install the extension** in editable mode:

```bash
# Activate your Isaac Lab conda environment
conda activate wheeled_lab  # or your Isaac Lab environment name

# Install the extension
python -m pip install -e source/wheeled_lab
```

**Note**: If Isaac Lab is not installed in a Python venv or conda environment, use `PATH_TO_isaaclab.sh -p` instead of `python`:

```bash
<IsaacLab>/isaaclab.sh -p -m pip install -e source/wheeled_lab
```

### Step 3: Verify Installation

Verify that the extension is correctly installed:

**List available environments:**

```bash
python scripts/list_envs.py
```

You should see environments with the prefix `Template-WheeledLab-` listed.



## IDE Setup (Optional but Recommended)

Setting up your IDE with proper IntelliSense is **strongly recommended** for development efficiency.

### VSCode Setup

1. **Run the setup task:**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Select `Tasks: Run Task`
   - Choose `setup_python_env`
   - Follow the prompts to provide the absolute path to your Isaac Sim installation

2. **Verify setup:**
   - A `.python.env` file should be created in the `.vscode` directory
   - This file contains Python paths to all Isaac Sim and Omniverse extensions
   - IntelliSense should now work for Isaac Lab and Isaac Sim modules

3. **Install Python extension:**
   - Make sure you have the Microsoft Python extension installed in VSCode

### Setup as Omniverse Extension (Optional)

To enable this extension in Omniverse:

1. **Add extension search paths:**
   - Open Omniverse → `Window` → `Extensions`
   - Click the **Hamburger Icon** → `Settings`
   - In `Extension Search Paths`, add:
     - Absolute path to this repository's `source` directory
     - Path to Isaac Lab's extension directory (`IsaacLab/source`)
   - Click **Hamburger Icon** → `Refresh`

2. **Enable the extension:**
   - Find `labs_for_wheels` (or `wheeled_lab` depending on package name) under the `Third Party` category
   - Toggle to enable

## Contributing

We welcome contributions! This project is actively being developed with plans for:

- Spatial reasoning for obstacle detection
- Visual semantic navigation
- Advanced waypoint following
- Additional robot platforms and tasks

### Development Guidelines

1. **Code Formatting**: We use pre-commit hooks for automatic code formatting

   ```bash
   pip install pre-commit
   pre-commit run --all-files
   ```

2. **Code Style**: Follow the Isaac Lab coding conventions and use type hints where appropriate

3. **Testing**: Before submitting, test your changes with:
   - Zero-action agent to verify environment setup
   - Random-action agent to test dynamics
   - Training scripts to ensure RL workflows function correctly

4. **Documentation**: Update relevant documentation when adding new features or tasks

### Getting Started with Development

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and test thoroughly
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## Code Formatting

We use pre-commit hooks for automatic code formatting. To set up:

```bash
pip install pre-commit
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

If IntelliSense is not working properly in VSCode:

1. Add the extension path to `.vscode/settings.json` under `"python.analysis.extraPaths"`:

```json
{
    "python.analysis.extraPaths": [
        "<path-to-this-repo>/source/wheeled_lab",
        "<path-to-isaaclab>/source/isaaclab",
        "<path-to-isaaclab>/source/isaaclab_assets",
        "<path-to-isaaclab>/source/isaaclab_tasks",
        "<path-to-isaaclab>/source/isaaclab_rl"
    ]
}
```

### Pylance Crash

If Pylance crashes due to memory issues, exclude unused Omniverse packages in `.vscode/settings.json`:

```json
{
    "python.analysis.exclude": [
        "<path-to-isaac-sim>/extscache/omni.anim.*",
        "<path-to-isaac-sim>/extscache/omni.kit.*",
        "<path-to-isaac-sim>/extscache/omni.graph.*",
        "<path-to-isaac-sim>/extscache/omni.services.*"
    ]
}
```

### Environment Not Found

If environments don't appear when running `scripts/list_envs.py`:

1. Ensure the extension is installed: `pip install -e source/wheeled_lab`
2. Check that the `config` module is being imported (see `source/wheeled_lab/wheeled_lab/tasks/drifting/__init__.py`)
3. Verify gymnasium registrations are correct

### Import Errors

If you encounter import errors:

1. Verify Isaac Lab is properly installed: `./isaaclab.sh -i`
2. Ensure you're using the correct Python environment
3. Check that all dependencies are installed: `pip install -r requirements.txt` (if available)

## Acknowledgments

This project is an adaptation of [WheeledLab](https://github.com/UWRobotLearning/WheeledLab) by the University of Washington Robot Learning Lab. We extend our gratitude to the original authors for their excellent work on wheeled robotics simulation.

## License

This project follows the same license as Isaac Lab. See the LICENSE file for details.

## References

### Original WheeledLab

- **Repository**: [UWRobotLearning/WheeledLab](https://github.com/UWRobotLearning/WheeledLab)
- **Paper**: [Demonstrating WheeledLab: Modern Sim2Real for Low-cost, Open-source Wheeled Robotics](https://arxiv.org/abs/2502.07380)

### Robot Platforms

- **MuSHR**: [MuSHR: A Low-Cost, Open-Source Robotic Racecar](https://arxiv.org/abs/1908.08031)
- **F1Tenth**: [F1TENTH: An Open-source Evaluation Environment](https://proceedings.mlr.press/v123/o-kelly20a.html)
- **HOUND**: [Demonstrating HOUND: A Low-cost Research Platform](https://arxiv.org/abs/2311.11199)
