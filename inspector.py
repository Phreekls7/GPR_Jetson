#!/usr/bin/env python3
import argparse
import socket
import sys
import binascii
import gc
import threading
import queue

import numpy as np
import cv2

ACK_HEX = b'007f007f'

def create_setup_message(q, rng):
    m_N, m_00, m_01 = ' ', '1', '1'
    m_07, m_08_10, m_11_12, m_15 = '0','000','00','0'
    m_16_19, m_20_21, m_22_31 = '1010','00','1010110010'
    if q==128:   m_05_06='00'
    elif q==256: m_05_06='10'
    elif q==512: m_05_06='01'
    elif q==1024:m_05_06='11'
    else:        m_05_06='01'
    if rng==25:   m_02_04,m_13_14='000','10'
    elif rng==50: m_02_04,m_13_14='000','00'
    elif rng==100:m_02_04,m_13_14='100','00'
    elif rng==200:m_02_04,m_13_14='010','00'
    elif rng==300:m_02_04,m_13_14='110','00'
    elif rng==2000:m_02_04,m_13_14='111','00'
    else:         m_02_04,m_13_14='000','00'
    return (
        'T'+m_N+m_00+m_01+
        m_02_04+m_05_06+m_07+
        m_08_10+m_11_12+m_13_14+
        m_15+m_16_19+m_20_21+m_22_31
    )

def read_one_trace(sock, q):
    total = q * 2
    buf = b''
    while len(buf) < total:
        chunk = sock.recv(total - len(buf))
        if not chunk:
            raise IOError("Socket closed")
        buf += chunk
    svc = q // 16
    main_n = q - svc
    return np.frombuffer(buf[: main_n*2], dtype='>i2')

def reader(sock, args, data, img_q, stop_evt):
    filled = 0
    while not stop_evt.is_set():
        try:
            trace = read_one_trace(sock, args.quantity)
        except Exception:
            continue
        if filled < args.window:
            data[:, filled] = trace
            filled += 1
        else:
            np.roll(data, -1, axis=1, out=data)
            data[:, -1] = trace

        img = ((data.astype(np.int32) + 32768) * (255/65535)).astype(np.uint8)
        if not img_q.empty():
            try: img_q.get_nowait()
            except: pass
        img_q.put(img)

def main():
    p = argparse.ArgumentParser("GPR B-scan fill→scroll windowed")
    p.add_argument('--host',     required=True, help="GPR IP")
    p.add_argument('--port',     type=int, default=23, help="GPR port")
    p.add_argument('--quantity', type=int, default=1024, help="samples per trace")
    p.add_argument('--range',    type=int, default=100,  help="timeRange (ns)")
    p.add_argument('--window',   type=int, default=1000, help="columns on screen")
    args = p.parse_args()

    setup = create_setup_message(args.quantity, args.range)
    sock  = socket.create_connection((args.host, args.port), timeout=5)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    gc.disable()
    sock.sendall((setup + "\n").encode('ascii'))
    sock.sendall(b"P1\n")
    if binascii.hexlify(sock.recv(4)) != ACK_HEX:
        print("Bad ACK", file=sys.stderr)
        sys.exit(1)
    sock.recv(1)

    svc    = args.quantity // 16
    main_n = args.quantity - svc
    data   = np.zeros((main_n, args.window), dtype=np.int16)
    img_q  = queue.Queue(maxsize=1)
    stop_evt = threading.Event()

    t = threading.Thread(target=reader,
                         args=(sock, args, data, img_q, stop_evt),
                         daemon=True)
    t.start()

    # Create a 800×600 window instead of fullscreen
    cv2.namedWindow("GPR B-scan", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("GPR B-scan", 800, 600)

    while True:
        try:
            img = img_q.get(timeout=0.1)
            cv2.imshow("GPR B-scan", img)
        except queue.Empty:
            cv2.waitKey(1)

        if cv2.waitKey(1) == 27:
            break

    stop_evt.set()
    sock.close()
    cv2.destroyAllWindows()
    gc.enable()

if __name__ == '__main__':
    main()
