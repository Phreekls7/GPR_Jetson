#!/usr/bin/env python3
import argparse
import socket
import sys
import time
import binascii

# Acknowledge message (hex) from the GPR after P1
ACK_HEX = b'007f007f'

def create_setup_message(sample_quantity: int, time_range: int) -> str:
    """
    Build the Cobra Zond-12e setup string (protocol T...).
    Mirrors hidden_graves.factory.sensor.sensor_gpr.GPR.create_setup_message.
    """
    # fixed bits
    m_N       = ' '
    m_00      = '1'   # Tx off
    m_01      = '1'   # cables combined
    m_07      = '0'
    m_08_10   = '000'
    m_11_12   = '00'
    m_15      = '0'
    m_16_19   = '1010'  # sounding regime
    m_20_21   = '00'    # single channel
    m_22_31   = '1010110010'

    # sample-quality bits:
    if sample_quantity == 128:
        m_05_06 = '00'
    elif sample_quantity == 256:
        m_05_06 = '10'
    elif sample_quantity == 512:
        m_05_06 = '01'
    elif sample_quantity == 1024:
        m_05_06 = '11'
    else:
        # pick nearest lower
        m_05_06 = '01'  # default to 512
    # time-range bits:
    if time_range == 25:
        m_02_04, m_13_14 = '000', '10'
    elif time_range == 50:
        m_02_04, m_13_14 = '000', '00'
    elif time_range == 100:
        m_02_04, m_13_14 = '100', '00'
    elif time_range == 200:
        m_02_04, m_13_14 = '010', '00'
    elif time_range == 300:
        m_02_04, m_13_14 = '110', '00'
    elif time_range == 2000:
        m_02_04, m_13_14 = '111', '00'
    else:
        m_02_04, m_13_14 = '000', '00'

    return (
        'T' + m_N + m_00 + m_01 +
        m_02_04 + m_05_06 + m_07 +
        m_08_10 + m_11_12 + m_13_14 +
        m_15 + m_16_19 + m_20_21 + m_22_31
    )

def read_traces(sock: socket.socket, sample_quantity: int, time_range: int, num_traces: int):
    """
    After setup & P1, pull num_traces traces.
    Each trace = sample_size signed-ints, then skip service bytes.
    """
    # sample_size = total – service; service = total/16
    service_sz = int(sample_quantity // 16)
    sample_sz  = sample_quantity - service_sz

    for t in range(num_traces):
        trace = []
        # read the main samples
        for _ in range(sample_sz):
            data = sock.recv(2)
            if len(data) < 2:
                raise IOError("Socket closed mid-trace")
            val = int.from_bytes(data, byteorder='big', signed=True)
            trace.append(val)
        # discard the service samples
        to_skip = service_sz * 2
        _ = sock.recv(to_skip)
        yield trace

def main():
    p = argparse.ArgumentParser(
        description="Fetch raw GPR traces from Cobra Zond-12e over TCP"
    )
    p.add_argument('--host',    required=True, help="GPR IP (e.g. 192.168.0.10)")
    p.add_argument('--port',    type=int, default=23, help="GPR port (default 23)")
    p.add_argument('--quantity',type=int, default=512,
                   help="sampleQuantity (128/256/512/1024)")
    p.add_argument('--range',   type=int, default=100,
                   help="timeRange in ns (25/50/100/200/300/2000)")
    p.add_argument('--traces',  type=int, default=5,
                   help="How many full traces to read")
    args = p.parse_args()

    setup_msg = create_setup_message(args.quantity, args.range)
    print(f"[+] Connecting to {args.host}:{args.port}")
    try:
        sock = socket.create_connection((args.host, args.port), timeout=5)
    except Exception as e:
        print(f"[!] Connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # send SETUP
        sock.sendall((setup_msg + "\n").encode('ascii'))
        # send P1 = start Tx
        sock.sendall(b"P1\n")
        # wait for 4-byte ACK
        ack = sock.recv(4)
        if binascii.hexlify(ack) != ACK_HEX:
            print(f"[!] Bad ACK: {ack!r}", file=sys.stderr)
            sys.exit(1)
        # discard dummy byte
        sock.recv(1)
        print("[+] Setup OK, reading traces…")
    except Exception as e:
        print(f"[!] Setup error: {e}", file=sys.stderr)
        sock.close()
        sys.exit(1)

    # fetch and print
    for idx, trace in enumerate(read_traces(sock, args.quantity, args.range, args.traces), 1):
        print(f"Trace {idx}: {len(trace)} samples")
        print(trace[:10], "…", trace[-10:])  # head & tail
    sock.close()
    print("[+] Done.")

if __name__ == '__main__':
    main()
