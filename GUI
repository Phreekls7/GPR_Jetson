#!/usr/bin/env python3
import sys
import threading
import math
import collections

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from px4_msgs.msg import VehicleOdometry, DistanceSensor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel
)
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg
import numpy as np


class TelemetryGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Drone Telemetry')

        # ROS2 setup
        rclpy.init()
        self.node = Node('telemetry_gui')
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, daemon=True).start()

        # bridge
        self.bridge = CvBridge()

        # flight path
        self.path = []
        self.origin = None

        # GPR & LiDAR
        self.latest_gpr    = None      # raw mono8 image (2D)
        self.latest_range  = None      # float
        self.base_range    = None
        self.mount_offset  = 0.40      # m: lidar minus GPR
        self.pix_per_m     = 50        # px per meter

        # history of range readings per-trace
        self.range_hist = None         # will become deque(maxlen=width)

        # QoS
        qos = QoSProfile(depth=10)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos.durability   = QoSDurabilityPolicy.VOLATILE

        # subs
        self.node.create_subscription(Image,
            '/gpr/bscan', self.gpr_cb, qos)
        self.node.create_subscription(DistanceSensor,
            '/fmu/out/distance_sensor', self.range_cb, qos)
        self.node.create_subscription(VehicleOdometry,
            '/fmu/out/vehicle_odometry', self.odom_cb, qos)

        # UI
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # flight path + range
        top = QHBoxLayout()
        self.plot = pg.PlotWidget(title='Flight Path')
        self.plot.setBackground('w')
        self.plot.showGrid(True, True)
        self.path_curve = self.plot.plot([], [], pen=pg.mkPen('b', width=2))
        top.addWidget(self.plot, stretch=3)
        self.range_label = QLabel('Range: -- m')
        self.range_label.setAlignment(Qt.AlignCenter)
        top.addWidget(self.range_label, stretch=1)
        layout.addLayout(top, stretch=1)

        # GPR display
        self.img_view = pg.ImageView()
        for btn in (self.img_view.ui.histogram,
                    self.img_view.ui.roiBtn,
                    self.img_view.ui.menuBtn):
            btn.hide()
        v = self.img_view.getView()
        v.enableAutoRange(False)
        v.setAspectLocked(False)
        layout.addWidget(self.img_view, stretch=1)

        # timer ~30Hz
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(33)

    def gpr_cb(self, msg: Image):
        # convert to mono8 numpy
        img = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        h, w = img.shape

        # initialize history on first frame
        if self.range_hist is None:
            # baseline fill with base_range if known, else zeros
            init = self.base_range if self.base_range is not None else 0.0
            self.range_hist = collections.deque([init]*w, maxlen=w)

        # append current_range for this new trace (column)
        self.range_hist.append(self.latest_range or self.base_range or 0.0)

        self.latest_gpr = img

    def range_cb(self, msg: DistanceSensor):
        self.latest_range = msg.current_distance
        if self.base_range is None:
            self.base_range = msg.current_distance

    def odom_cb(self, msg: VehicleOdometry):
        if msg.pose_frame == VehicleOdometry.POSE_FRAME_FRD:
            x, y, _ = msg.position
        else:
            xn, yn, _ = msg.position; x, y = yn, xn
        if self.origin is None:
            self.origin = (x, y)
        dx, dy = x - self.origin[0], y - self.origin[1]
        w, xq, yq, zq = msg.q
        yaw = math.atan2(2*(w*zq + xq*yq),
                          1 - 2*(yq*yq + zq*zq))
        c, s = math.cos(-yaw), math.sin(-yaw)
        fx = dx*c - dy*s
        fy = dx*s + dy*c
        self.path.append((fx, fy))

    def update_ui(self):
        # flight path
        if self.path:
            xs, ys = zip(*self.path)
            self.path_curve.setData(xs, ys)

        # range label
        if self.latest_range is not None:
            self.range_label.setText(f'Range: {self.latest_range:.2f} m')

        # gpr data shifting
        if self.latest_gpr is not None and self.range_hist is not None:
            img = self.latest_gpr.copy()
            h, w = img.shape
            # build a new image by shifting each column
            shifted = np.zeros_like(img)
            for j in range(w):
                # true GPR-to-ground distance at that trace
                d_gpr = (self.range_hist[j] or 0.0) - self.mount_offset
                delta  = d_gpr - ((self.base_range or 0.0) - self.mount_offset)
                off_px = int(delta * self.pix_per_m)
                shifted[:, j] = np.roll(img[:, j], -off_px, axis=0)

            disp = shifted.T
            self.img_view.setImage(disp, autoLevels=True)
            self.img_view.getView().setRange(
                xRange=(0, w), yRange=(0, h), padding=0)

    def closeEvent(self, event):
        self.executor.shutdown()
        self.node.destroy_node()
        rclpy.shutdown()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = TelemetryGUI()
    gui.showMaximized()
    sys.exit(app.exec_())
