sudo ./docker/container.sh start

sudo docker exec -it open_manipulator \
    bash -ic "ros2 run open_manipulator_teleop open_manipulator_x_teleop"

