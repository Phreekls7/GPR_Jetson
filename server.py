#!/usr/bin/env python3
import argparse
import time
from pymavlink import mavutil

def main():
    p = argparse.ArgumentParser(
        description="Read Pixhawk 6X rangefinder via MAVLink"
    )
    p.add_argument('--connect', required=True,
                   help="MAVLink connection string, e.g. /dev/ttyTHS1,57600 or udp:0.0.0.0:14550")
    p.add_argument('--rate', type=int, default=10,
                   help="Desired update rate in Hz for DISTANCE_SENSOR")
    args = p.parse_args()

    print(f"[+] Connecting to {args.connect}…")
    m = mavutil.mavlink_connection(args.connect)
    # wait for heartbeat
    m.wait_heartbeat()
    print("[+] Heartbeat received. Autopilot system %u component %u" %
          (m.target_system, m.target_component))

    # Request DISTANCE_SENSOR at args.rate Hz
    # MAV_CMD_SET_MESSAGE_INTERVAL (180), param1 = msg_id, param2 = interval in µs
    MSG_ID = mavutil.mavlink.MAVLINK_MSG_ID_DISTANCE_SENSOR  # 132
    interval_us = int(1e6 / args.rate)
    m.mav.command_long_send(
        m.target_system,
        m.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        MSG_ID,
        interval_us,
        0,0,0,0,0
    )
    print(f"[+] Requested DISTANCE_SENSOR @ {args.rate} Hz")

    try:
        while True:
            # Blocking wait for next DISTANCE_SENSOR
            msg = m.recv_match(type='DISTANCE_SENSOR', blocking=True, timeout=5)
            if msg is None:
                print("[!] No DISTANCE_SENSOR in 5 s, retrying…")
                continue
            # Fields per MAVLink common spec :contentReference[oaicite:0]{index=0}
            dist_cm = msg.distance        # uint16_t in cm
            min_cm  = msg.min_distance    # minimum sensor range
            max_cm  = msg.max_distance    # maximum sensor range
            ori     = msg.orientation     # sensor orientation :contentReference[oaicite:1]{index=1}
            print(f"Distance: {dist_cm} cm  (range: {min_cm}–{max_cm} cm)  orientation: {ori}")
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user, exiting.")
    finally:
        m.close()

if __name__ == "__main__":
    main()
