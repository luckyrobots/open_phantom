# Open Phantom: Collecting Data From Robots Using Only Human Videos

## Overview

Open Phantom is a fully open-source implementation of the approach described in the paper "[Phantom: Training Robots Without Robots Using Only Human Videos.](https://phantom-human-videos.github.io/)" This project focuses on the data collection component of the Phantom pipeline, enabling anyone with a standard RGB camera to generate training data for robot learning without requiring actual robot hardware.

## Key Features

* **Camera-Only Data Collection** : Capture hand movements using any standard RGB camera
* **3D Hand Tracking** : Convert 2D video to 3D hand poses using MediaPipe landmarks
* **Advanced Depth Estimation** : Generate depth maps from monocular RGB input using MiDaS
* **Hand Segmentation** : Precisely isolate hand regions with Meta's SAM2 for better depth estimation
* **ICP Registration** : Align hand mesh with depth point cloud for improved 3D accuracy
* **Anatomical Constraints** : Apply natural hand constraints to ensure realistic movements
* **Robot Action Extraction** : Transform hand poses into robot control parameters (position, orientation, gripper width)
* **Visualization Pipeline** : Debug-friendly visualization of each processing stage
* **Commercial-Friendly** : Built entirely with open-source, commercially usable components

## Project Status

⚠️  **Work in Progress** : Open Phantom is currently under active development.

Known limitations:

* Camera calibrations
* Transforming robot to human reference frame
* In-painting human hand
* Overlaying robot arm
* Performance optimizations needed for real-time processing

## Background

The original Phantom paper demonstrated that robots could learn tasks from human demonstrations without any robot-specific data collection. By capturing hand movements in diverse environments and converting them to robot action parameters, it's possible to train robot policies that perform effectively during zero-shot deployment.

Unlike the original implementation which relies on [MANO](https://mano.is.tue.mpg.de/index.html) (a hand model not available for commercial use), Open Phantom is built entirely with open-source components that can be used in commercial applications.

## How It Works

1. **Video Capture** : Record video of your hand performing a task using a standard RGB camera
2. **Hand Tracking** : Track hand landmarks in the video
3. **Depth Estimation** : Estimate depth information from the monocular RGB input
4. **Segmentation** : Segment the hand using SAM2 (Segment Anything Model 2)
5. **3D Reconstruction** : Create a 3D hand model from the landmarks and depth information
6. **Robot Parameters** : Extract position, orientation, and gripper parameters for robot control

## Installation

```bash
# Clone the repository with submodules
git clone https://github.com/luckyrobots/open_phantom.git
cd open-phantom

# Create and activate conda environment
conda env create -f environment.yml
conda activate open_phantom

# Initialize and update submodules
git submodule update --init --recursive

# Install dependencies for SAM2
cd external/sam2
pip install -e .
cd ../..

# Verify installation
python -c "import mediapipe; import open3d; print('Dependencies successfully installed!')"
```

## Usage

```bash
# Run the main script to record and process a video
python open_phantom/main.py
```

## Contributing

### Find an Issue

* Browse [GitHub Issues](https://github.com/luckyrobots/open_phantom/issues) or create a new one
* Comment on issues you plan to work on

### Setup & Branch

```bash
# Fork and clone the repository
git clone https://github.com/luckyrobots/open_phantom.git
cd open-phantom

# Add upstream remote and stay updated
git remote add upstream https://github.com/luckyrobots/open_phantom.git
git pull upstream main

# Create a dedicated branch for your work
git checkout -b fix-issue-42  # Use descriptive names
```

### Set Up Pre-commit Hooks

```bash
# Install pre-commit if you don't have it
pip install pre-commit

# Set up the git hooks
pre-commit install
```

Pre-commit will automatically check your code style, formatting, and other quality standards before each commit. If any checks fail, it will prevent the commit and show what needs to be fixed.

### Make Changes

* Write clean, commented code
* Keep commits focused and with clear messages
* Test your changes thoroughly
* When you commit, pre-commit hooks will run automatically

### Submit a Pull Request

* Push your branch: `git push origin fix-issue-42`
* Create PR on GitHub pointing to `main` branch
* Include clear title (e.g., "Fix #42: Improve Hand Tracking")
* Briefly describe changes and reference related issues
* Respond to reviewer feedback

### Guidelines

* Document code with docstrings
* Be respectful in all interactions
* Give credit where due

## Citation

If you use Open Phantom in your research, please cite the original Phantom paper:

```
@article{lepert2025phantom,
  title={Phantom: Training Robots Without Robots Using Only Human Videos},
  author={Lepert, Marion and Fang, Jiaying and Bohg, Jeannette},
  journal={arXiv preprint arXiv:2503.00779},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* This project is based on the research presented in "Phantom: Training Robots Without Robots Using Only Human Videos"
* We use Meta's SAM2 (Segment Anything Model 2) for hand segmentation

## Disclaimer

Open Phantom is a community implementation focused on the data collection aspects of the Phantom approach. The original paper authors are not affiliated with this specific implementation.
