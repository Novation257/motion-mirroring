#!/bin/bash
set -e

cd ~/open_manipulator

sudo ./docker/container.sh start

sudo docker exec -it open_manipulator \
    bash -ic "ros2 launch open_manipulator_bringup open_manipulator_x.launch.py"
