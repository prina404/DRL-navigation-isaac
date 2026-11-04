#!/bin/bash

ros_ws_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P )

source /opt/ros/jazzy/setup.bash
source "$ros_ws_path/install/local_setup.bash"
