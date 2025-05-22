#!/usr/bin/env python3
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from px4_msgs.msg import VehicleGlobalPosition
from sensor_msgs.msg import Range

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout
)
from PyQt5.QtCore import QTimer

# Matplotlib imports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class GuiNode(Node):
    def __init__(self):
        super().__init__('sensor_gui')
        self.lat = 0.0
        self.lon = 0.0
        self.alt = 0.0
        self.dist = 0.0
        # history
        self.path = []       # list of (lat, lon)
        self.times = []      # relative timestamps
        self.dist_hist = []  # distance history

        self.create_subscription(
            VehicleGlobalPosition,
            '/fmu/out/vehicle_global_position',
            self.gps_cb, 10)
        self.create_subscription(
            Range,
            '/distance_sensor',
            self.range_cb, 10)

        self.start_time = time.time()

    def gps_cb(self, msg):
        self.lat = msg.lat
        self.lon = msg.lon
        self.alt = msg.alt
        # record path
        self.path.append((self.lat, self.lon))

    def range_cb(self, msg):
        self.dist = msg.range
        # record distance vs time
        t = time.time() - self.start_time
        self.times.append(t)
        self.dist_hist.append(self.dist)


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self):
        fig = Figure(figsize=(5, 4), tight_layout=True)
        super().__init__(fig)
        self.ax1 = fig.add_subplot(121)
        self.ax2 = fig.add_subplot(122)
        # initial labels
        self.ax1.set_title("GPS Path")
        self.ax1.set_xlabel("Longitude")
        self.ax1.set_ylabel("Latitude")
        self.ax2.set_title("Distance vs Time")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Distance (m)")

    def update(self, path, times, dist_hist):
        # clear and redraw
        self.ax1.cla()
        self.ax1.set_title("GPS Path")
        if path:
            lats, lons = zip(*path)
            self.ax1.plot(lons, lats, '-o', markersize=2)
        self.ax1.set_xlabel("Lon")
        self.ax1.set_ylabel("Lat")

        self.ax2.cla()
        self.ax2.set_title("Distance vs Time")
        if times:
            self.ax2.plot(times, dist_hist, '-')
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Dist (m)")

        self.draw()


class MainWindow(QWidget):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node

        self.setWindowTitle('Drone Sensors')
        main_layout = QVBoxLayout(self)

        # add the plot canvas
        self.canvas = PlotCanvas()
        main_layout.addWidget(self.canvas)

        # labels (optional)
        self.gps_label = QWidget(self)  # placeholder if you still want labels

        # timer to update plot
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(200)  # every 200 ms

    def update_display(self):
        # repaint with new data
        self.canvas.update(
            self.ros_node.path,
            self.ros_node.times,
            self.ros_node.dist_hist
        )


def main():
    rclpy.init()
    ros_node = GuiNode()

    # spin ROS in background
    executor = SingleThreadedExecutor()
    executor.add_node(ros_node)
    threading.Thread(target=executor.spin, daemon=True).start()

    # start Qt
    app = QApplication(sys.argv)
    win = MainWindow(ros_node)
    win.show()
    app.exec_()

    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
