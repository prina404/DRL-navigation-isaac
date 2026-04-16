#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ;)
cd "$parent_path"

source /opt/ros/jazzy/setup.bash
source ./install/local_setup.bash
