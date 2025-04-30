#!/usr/bin/env python3
import argparse
import socket
import sys
import binascii

import numpy as np
import matplotlib.pyplot as plt

ACK_HEX = b'007f007f'

def create_setup_message(sample_quantity: int, time_range: int) -> str:
    # same builder as before...
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
    service = sample_quantity//16
    main_n  = sample_quantity - service
    buf = np.empty(main_n, dtype=np.int16)
    for i in range(main_n):
        b = sock.recv(2)
        if len(b)<2: raise IOError("Socket closed")
        buf[i] = int.from_bytes(b, byteorder='big', signed=True)
    sock.recv(service*2)
    return buf

def main():
    p = argparse.ArgumentParser("Live GPR B-scan")
    p.add_argument('--host',     required=True, help="GPR IP")
    p.add_argument('--port',     type=int, default=23, help="GPR port")
    p.add_argument('--quantity', type=int, default=512, help="sampleQuantity")
    p.add_argument('--range',    type=int, default=100, help="timeRange (ns)")
    p.add_argument('--window',   type=int, default=100,
                   help="number of traces across")
    args = p.parse_args()

    # connect & setup
    setup = create_setup_message(args.quantity, args.range)
    try:
        sock = socket.create_connection((args.host,args.port), timeout=5)
        sock.sendall((setup+"\n").encode('ascii'))
        sock.sendall(b"P1\n")
        ack = sock.recv(4)
        if binascii.hexlify(ack)!=ACK_HEX:
            print("Bad ACK",ack,file=sys.stderr); sys.exit(1)
        sock.recv(1)
    except Exception as e:
        print("Setup failed:",e,file=sys.stderr); sys.exit(1)

    # prep buffer & plot
    service = args.quantity//16
    main_n  = args.quantity - service
    data    = np.zeros((main_n, args.window), dtype=np.int16)
    col     = 0
    trace_idx = 0

    plt.ion()
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(
        data,
        cmap='gray',
        aspect='auto',
        origin='upper',
        vmin=-32768,
        vmax=32767
    )
    ax.set_xlabel("Trace #")
    ax.set_ylabel("Sample (time→depth)")
    ax.set_title("Live GPR B-scan (close to stop)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Amplitude")

    # set x-ticks at left, mid, right
    ticks = [0, args.window//2, args.window-1]
    ax.set_xticks(ticks)

    try:
        while plt.fignum_exists(fig.number):
            trace = read_one_trace(sock, args.quantity)
            data[:, col] = trace
            trace_idx += 1
            col = (col+1) % args.window

            im.set_data(data)
            # label x-ticks with absolute trace numbers
            start = trace_idx - args.window + 1
            labels = [max(0, start + t) for t in ticks]
            ax.set_xticklabels(labels)

            fig.canvas.draw()
            fig.canvas.flush_events()

    except Exception as e:
        print("Error:",e,file=sys.stderr)
    finally:
        sock.close()

if __name__=='__main__':
    main()
