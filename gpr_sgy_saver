#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpr_sgy_saver.py

Subscribes to /gpr/bscan (sensor_msgs/Image, mono8), takes each new column
as an int16 GPR trace, accumulates them, and on shutdown writes a
GeoLitix-compatible SEG-Y file.
"""

import os
import datetime
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import numpy as np
import segyio
from cv_bridge import CvBridge, CvBridgeError


class GPRSegySaver(Node):
    def __init__(self):
        super().__init__("gpr_segy_saver")
        self.bridge = CvBridge()
        self.accumulated_traces = []     # list of 1D np.int16 arrays
        self.lock = threading.Lock()
        self.trace_count = 0

        # subscribe to the rolling B-scan topic
        topic = "/gpr/bscan"
        qos = rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Image, topic, self.image_callback, qos)
        self.get_logger().info(f"Subscribed to {topic}")

    def image_callback(self, msg: Image):
        # Convert to CV image (uint8)
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except CvBridgeError as e:
            self.get_logger().error(f"CVBridge error: {e}")
            return

        # grab rightmost column
        if cv_img.ndim != 2:
            return
        col = cv_img[:, -1]  # shape (n_samples,), dtype=uint8

        # uint8->int16 inverse of ((x+32768)*(255/65535))->uint8
        val32 = col.astype(np.int32) * (65535.0/255.0)
        trace = np.round(val32 - 32768.0).astype(np.int16)

        with self.lock:
            self.accumulated_traces.append(trace)
            self.trace_count += 1
            if self.trace_count % 100 == 0:
                self.get_logger().info(f"Collected {self.trace_count} traces")

    def on_shutdown(self):
        # copy & clear
        with self.lock:
            traces = self.accumulated_traces[:]
            self.accumulated_traces.clear()

        n_traces = len(traces)
        if not n_traces:
            self.get_logger().warn("No traces to write; exiting")
            return
        n_samples = traces[0].size

        # build data matrix
        data = np.stack(traces, axis=1).astype(np.int16)

        # build SEG-Y spec
        spec = segyio.spec()
        spec.format  = 3                  # 2-byte int
        spec.samples = np.arange(n_samples)
        spec.ilines = np.zeros(n_traces, dtype=np.int32)
        spec.xlines = np.zeros(n_traces, dtype=np.int32)

        # output path & name
        cwd = os.getcwd()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(cwd, f"gpr_output_{ts}.sgy")

        with segyio.create(out, spec) as f:
            # EBCDIC header
            txt = (
                "C 1 SEG-Y REV1.0\n"
                "C 2 GPR B-Scan GeoLitix-Compatible Export\n"
                f"C 3 Samples/Trace: {n_samples}\n"
                f"C 4 Total Traces: {n_traces}\n"
                "C 5 Sample Interval (µs): 1\n"
            ).ljust(3200)[:3200]
            f.text[0] = txt.encode("cp500", "replace")

            # binary header
            binh = f.bin
            binh[segyio.BinField.JobID]      = 1
            binh[segyio.BinField.LineNumber] = 1
            binh[segyio.BinField.ReelNumber] = 1
            binh[segyio.BinField.Samples]    = n_samples
            binh[segyio.BinField.Interval]   = 1
            binh[segyio.BinField.Format]     = 3

            # write traces
            for i in range(n_traces):
                hdr = f.header[i]
                hdr[segyio.TraceField.TraceSequenceNumber] = i+1
                hdr[segyio.TraceField.FieldRecordNumber]   = 1
                hdr[segyio.TraceField.TraceNumber]         = i+1
                # use CDP_X as a 32-bit index
                hdr[segyio.TraceField.CDP_X] = i
                hdr[segyio.TraceField.CDP_Y] = 0
                f.trace[i] = data[:, i].astype(">i2")

        self.get_logger().info(f"SEG-Y written: {out}")

def main():
    rclpy.init()
    node = GPRSegySaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down, writing SEG-Y…")
    finally:
        node.on_shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
