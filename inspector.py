#!/usr/bin/env python3
import argparse
import socket
import sys
import binascii

import numpy as np
import cv2

ACK_HEX = b'007f007f'  # Cobra’s 4-byte ACK

def create_setup_message(sample_quantity: int, time_range: int) -> str:
    # Build the Cobra “T…” setup string
    m_N, m_00, m_01 = ' ', '1', '1'
    m_07, m_08_10, m_11_12, m_15 = '0', '000', '00', '0'
    m_16_19, m_20_21, m_22_31 = '1010', '00', '1010110010'
    if sample_quantity == 128:   m_05_06 = '00'
    elif sample_quantity == 256: m_05_06 = '10'
    elif sample_quantity == 512: m_05_06 = '01'
    elif sample_quantity == 1024: m_05_06 = '11'
    else:                         m_05_06 = '01'
    if time_range == 25:     m_02_04, m_13_14 = '000','10'
    elif time_range == 50:   m_02_04, m_13_14 = '000','00'
    elif time_range == 100:  m_02_04, m_13_14 = '100','00'
    elif time_range == 200:  m_02_04, m_13_14 = '010','00'
    elif time_range == 300:  m_02_04, m_13_14 = '110','00'
    elif time_range == 2000: m_02_04, m_13_14 = '111','00'
    else:                    m_02_04, m_13_14 = '000','00'

    return (
        'T'+m_N+m_00+m_01+
        m_02_04+m_05_06+m_07+
        m_08_10+m_11_12+m_13_14+
        m_15+m_16_19+m_20_21+m_22_31
    )

def read_one_trace(sock, sample_quantity):
    # Read main samples, then drop the service samples
    service = sample_quantity // 16
    main_n  = sample_quantity - service
    buf = np.empty(main_n, dtype=np.int16)
    for i in range(main_n):
        b = sock.recv(2)
        if len(b) < 2:
            raise IOError("Socket closed")
        buf[i] = int.from_bytes(b, byteorder='big', signed=True)
    sock.recv(service * 2)
    return buf

def main():
    p = argparse.ArgumentParser("Live GPR B-scan via OpenCV")
    p.add_argument('--host',     required=True, help="GPR IP (e.g. 192.168.0.10)")
    p.add_argument('--port',     type=int, default=23, help="GPR port (default 23)")
    p.add_argument('--quantity', type=int, default=512, help="sampleQuantity")
    p.add_argument('--range',    type=int, default=100, help="timeRange in ns")
    p.add_argument('--window',   type=int, default=200, help="number of traces on screen")
    args = p.parse_args()

    # 1) Connect & SETUP
    setup = create_setup_message(args.quantity, args.range)
    try:
        sock = socket.create_connection((args.host, args.port), timeout=5)
        sock.sendall((setup + "\n").encode('ascii'))
        sock.sendall(b"P1\n")
        ack = sock.recv(4)
        if binascii.hexlify(ack) != ACK_HEX:
            print("Bad ACK:", ack, file=sys.stderr)
            sys.exit(1)
        sock.recv(1)
    except Exception as e:
        print("Setup failed:", e, file=sys.stderr)
        sys.exit(1)

    # 2) Prepare display buffer
    service = args.quantity // 16
    main_n  = args.quantity - service
    data    = np.zeros((main_n, args.window), dtype=np.int16)

    cv2.namedWindow("GPR B-scan", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("GPR B-scan", 800, 600)

    # 3) Streaming loop
    while True:
        try:
            trace = read_one_trace(sock, args.quantity)
        except Exception as e:
            print("Read error:", e, file=sys.stderr)
            break

        # shift old data right, insert new on left
        data = np.roll(data, 1, axis=1)
        data[:, 0] = trace

        # normalize to 0–255 for display
        img = ((data.astype(np.int32) + 32768) * (255/65535)).astype(np.uint8)

        cv2.imshow("GPR B-scan", img)
        # ESC key to quit
        if cv2.waitKey(1) == 27:
            break

    sock.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
