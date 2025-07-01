#!/usr/bin/env python3
import os
import math
import io
import rclpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import requests
from geographiclib.geodesic import Geodesic
from rclpy.node import Node
from px4_msgs.msg import VehicleOdometry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class FlightPathPost(Node):
    def __init__(self):
        super().__init__('flight_path_post')

        # static home GPS position (latitude, longitude)
        self.home_lat = 52.0100  # replace with your take-off latitude
        self.home_lon = 4.3600   # replace with your take-off longitude
        self.zoom = 15           # OSM tile zoom level

        # output directory for CSV
        self.declare_parameter('output_dir', '/tmp')
        out_dir = self.get_parameter('output_dir').value
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, 'flight_odometry.csv')

        # buffers for odometry and geolocation
        self.times = []
        self.xs = []
        self.ys = []
        self.zs = []
        self.lats = []
        self.lons = []

        # configure QoS matching PX4 publisher
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_cb,
            qos
        )

        # geodetic converter
        self.geod = Geodesic.WGS84

        # fetch a single OSM tile around home
        self.fetch_basemap()

        # set up interactive Matplotlib plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        # show the tile image as background
        self.ax.imshow(self.tile_img, extent=(self.lon_min, self.lon_max, self.lat_min, self.lat_max), origin='upper')
        # prepare the flight path line (lon vs lat)
        self.line, = self.ax.plot([], [], '-o', markersize=4, color='red')
        self.ax.set_xlim(self.lon_min, self.lon_max)
        self.ax.set_ylim(self.lat_min, self.lat_max)
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.set_title('Real-time Flight Path on OSM Basemap')
        plt.show()

        # timer to refresh at 10 Hz
        self.create_timer(0.1, self.update_plot)
        self.get_logger().info("Node started: plotting real-time flight path on map.")

    def deg2tile(self, lat_deg, lon_deg, zoom):
        # convert lat/lon to OSM tile indices
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xt = int((lon_deg + 180.0) / 360.0 * n)
        yt = int((1.0 - math.log(math.tan(lat_rad) + 1/math.cos(lat_rad)) / math.pi) / 2.0 * n)
        return xt, yt

    def tile2lat(self, y, zoom):
        # convert tile y index to latitude
        n = 2.0 ** zoom
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        return math.degrees(lat_rad)

    def fetch_basemap(self):
        # get tile coords
        xt, yt = self.deg2tile(self.home_lat, self.home_lon, self.zoom)
        # compute bounding lon/lat
        n = 2.0 ** self.zoom
        self.lon_min = xt / n * 360.0 - 180.0
        self.lon_max = (xt + 1) / n * 360.0 - 180.0
        self.lat_max = self.tile2lat(yt, self.zoom)
        self.lat_min = self.tile2lat(yt + 1, self.zoom)
        # download tile PNG
        url = f"https://tile.openstreetmap.org/{self.zoom}/{xt}/{yt}.png"
        resp = requests.get(url)
        img = Image.open(io.BytesIO(resp.content))
        self.tile_img = np.array(img)

    def odom_cb(self, msg: VehicleOdometry):
        # only record NED frame
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            return
        # skip NaNs
        if any(math.isnan(v) for v in msg.position):
            return
        # timestamp in seconds
        t = msg.timestamp * 1e-6
        x, y, z = msg.position
        # store local NED and convert to geo
        self.times.append(t)
        self.xs.append(x)
        self.ys.append(y)
        self.zs.append(z)
        # convert offsets to lat/lon
        az = math.degrees(math.atan2(y, x))
        dist = math.hypot(x, y)
        geo = self.geod.Direct(self.home_lat, self.home_lon, az, dist)
        self.lats.append(geo['lat2'])
        self.lons.append(geo['lon2'])

    def update_plot(self):
        if not self.lats:
            return
        # update line data in geographic coords
        self.line.set_data(self.lons, self.lats)
        self.fig.canvas.draw()
        plt.pause(0.001)

    def save_csv(self):
        df = pd.DataFrame({
            'timestamp_s': self.times,
            'x_m': self.xs,
            'y_m': self.ys,
            'z_m': self.zs,
            'lat': self.lats,
            'lon': self.lons
        })
        df.to_csv(self.csv_path, index=False)
        self.get_logger().info(f"Saved CSV → {self.csv_path}")
        return df

    def on_shutdown(self):
        df = self.save_csv()
        if df.empty:
            self.get_logger().warning("No data recorded.")


def main(args=None):
    rclpy.init(args=args)
    node = FlightPathPost()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.on_shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
