#!/usr/bin/env python3
import socket
import sys
import binascii
import gc
import threading
import queue

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

ACK_HEX = b'007f007f'

def create_setup_message(q, rng):
    """
    Build the COBRA GPR setup string exactly as before.
    """
    m_N, m_00, m_01 = ' ', '1', '1'
    m_07, m_08_10, m_11_12, m_15 = '0','000','00','0'
    m_16_19, m_20_21, m_22_31 = '1010','00','1010110010'

    # Bits 05–06 select quantity
    if q == 128:
        m_05_06 = '00'
    elif q == 256:
        m_05_06 = '10'
    elif q == 512:
        m_05_06 = '01'
    elif q == 1024:
        m_05_06 = '11'
    else:
        m_05_06 = '01'

    # Bits 02–04 and 13–14 select range (ns)
    if rng == 25:
        m_02_04, m_13_14 = '000', '10'
    elif rng == 50:
        m_02_04, m_13_14 = '000', '00'
    elif rng == 100:
        m_02_04, m_13_14 = '100', '00'
    elif rng == 200:
        m_02_04, m_13_14 = '010', '00'
    elif rng == 300:
        m_02_04, m_13_14 = '110', '00'
    elif rng == 2000:
        m_02_04, m_13_14 = '111', '00'
    else:
        m_02_04, m_13_14 = '000', '00'

    return (
        'T' + m_N + m_00 + m_01 +
        m_02_04 + m_05_06 + m_07 +
        m_08_10 + m_11_12 + m_13_14 +
        m_15 + m_16_19 + m_20_21 + m_22_31
    )

def read_one_trace(sock, q):
    """
    Read exactly q samples (2 bytes each) from the socket,
    then convert to a NumPy array of length (q - q/16) as big‐endian int16.
    """
    total = q * 2
    buf = b''
    while len(buf) < total:
        chunk = sock.recv(total - len(buf))
        if not chunk:
            raise IOError("Socket closed by GPR")
        buf += chunk
    svc = q // 16
    main_n = q - svc
    return np.frombuffer(buf[: main_n * 2], dtype='>i2')

class GPRStreamer(Node):
    """
    A ROS 2 node that:
      • connects to the COBRA GPR at 192.168.0.10:23
      • continuously reads raw traces (quantity=1024, range=100)
      • maintains a rolling 2D array of 1000 columns (window=1000)
      • converts to a mono8 image and publishes on /gpr/bscan
    """
    def __init__(self):
        super().__init__('gpr_streamer')

        # ------------------------------------------------------------
        # 1) Publisher + CvBridge
        # ------------------------------------------------------------
        self.img_pub = self.create_publisher(Image, '/gpr/bscan', 10)
        self.bridge = CvBridge()

        # ------------------------------------------------------------
        # 2) Open socket and handshake exactly as in the original script
        # ------------------------------------------------------------
        host = '192.168.0.10'
        port = 23
        quantity = 1024
        gpr_range = 100
        window_cols = 1000

        setup_msg = create_setup_message(quantity, gpr_range)
        try:
            self.sock = socket.create_connection((host, port), timeout=5)
            # Disable Nagle's algorithm for minimal latency
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception as e:
            self.get_logger().error(f"Cannot connect to GPR at {host}:{port} → {e}")
            sys.exit(1)

        # Disable Python GC (same as original) while streaming
        gc.disable()
        self.sock.sendall((setup_msg + "\n").encode('ascii'))
        self.sock.sendall(b"P1\n")
        ack = self.sock.recv(4)
        if binascii.hexlify(ack) != ACK_HEX:
            self.get_logger().error(f"Bad ACK from GPR: {binascii.hexlify(ack)}")
            sys.exit(1)
        # Consume that trailing byte (per COBRA protocol)
        self.sock.recv(1)

        # ------------------------------------------------------------
        # 3) Preallocate rolling-window data buffer (int16)
        # ------------------------------------------------------------
        svc = quantity // 16
        main_n = quantity - svc
        self.data = np.zeros((main_n, window_cols), dtype=np.int16)

        # ------------------------------------------------------------
        # 4) Reader thread + queue for passing images safely
        # ------------------------------------------------------------
        self.img_q = queue.Queue(maxsize=1)
        self.stop_evt = threading.Event()
        self.reader_thread = threading.Thread(target=self._reader_loop,
                                              args=(quantity, window_cols),
                                              daemon=True)
        self.reader_thread.start()

        # ------------------------------------------------------------
        # 5) Timer to publish at ~30 Hz
        # ------------------------------------------------------------
        timer_period = 1.0 / 30.0  # seconds
        self.timer = self.create_timer(timer_period, self._timer_callback)

        self.get_logger().info(
            f"GPR streamer ready → publishing /gpr/bscan "
            f"({window_cols}×{main_n} px) @ ~{1/timer_period:.1f} Hz"
        )

    def _reader_loop(self, quantity, window_cols):
        """
        Background thread: read one trace at a time, roll the 2D array,
        convert to uint8 image, and put into self.img_q.
        """
        filled = 0
        while not self.stop_evt.is_set():
            try:
                trace = read_one_trace(self.sock, quantity)
            except Exception:
                continue

            if filled < window_cols:
                self.data[:, filled] = trace
                filled += 1
            else:
                np.roll(self.data, -1, axis=1, out=self.data)
                self.data[:, -1] = trace

            # Scale int16 (−32768..+32767) → uint8 (0..255)
            img = ((self.data.astype(np.int32) + 32768) * (255.0 / 65535.0))\
                  .astype(np.uint8)

            # Only keep the latest image in the queue
            if not self.img_q.empty():
                try:
                    self.img_q.get_nowait()
                except queue.Empty:
                    pass
            self.img_q.put(img)

    def _timer_callback(self):
        """
        Called at ~30 Hz by ROS 2. Pop the latest image (if any), convert
        to sensor_msgs/Image via CvBridge, and publish.
        """
        try:
            img = self.img_q.get_nowait()
        except queue.Empty:
            return

        ros_img = self.bridge.cv2_to_imgmsg(img, encoding='mono8')
        ros_img.header.stamp = self.get_clock().now().to_msg()
        ros_img.header.frame_id = 'gpr_frame'
        self.img_pub.publish(ros_img)

    def destroy_node(self):
        """
        Clean up: signal the reader thread to stop, join, close socket,
        re-enable GC, then destroy base class.
        """
        self.stop_evt.set()
        try:
            self.reader_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass
        gc.enable()
        super().destroy_node()

def main():
    rclpy.init()
    node = GPRStreamer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down GPR streamer...")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
