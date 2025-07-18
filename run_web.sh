#!/bin/bash

# STEP 1: Source ROS 2 Humble
source /opt/ros/humble/setup.bash

# STEP 2: Source your workspace
source ~/rn_prototype/install/setup.bash

# STEP 3: Move to the web app folder (so Flask works from correct place)
cd "$(dirname "$0")"

# STEP 4: Launch Flask web app
python3 app.py
