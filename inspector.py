#!/usr/bin/env python3
import argparse, socket, sys, binascii, gc, threading, queue
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

ACK_HEX = b'007f007f'

def create_setup_message(q, rng):
    # same T-command builder…
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
    return ('T'+m_N+m_00+m_01+
            m_02_04+m_05_06+m_07+
            m_08_10+m_11_12+m_13_14+
            m_15+m_16_19+m_20_21+m_22_31)

def read_one_trace(sock, q):
    total = q*2
    buf = b''
    while len(buf)<total:
        chunk = sock.recv(total-len(buf))
        if not chunk: raise IOError("Socket closed")
        buf += chunk
    svc = q//16
    main_n = q-svc
    return np.frombuffer(buf[:main_n*2], dtype='>i2')

def reader(sock, args, data, img_q, stop_evt):
    while not stop_evt.is_set():
        try:
            trace = read_one_trace(sock, args.quantity)
        except Exception:
            continue
        # roll right, insert new at col 0
        np.roll(data, 1, axis=1, out=data)
        data[:,0] = trace
        # map to 0–255 uint8
        img = ((data.astype(np.int32)+32768)*(255/65535)).astype(np.uint8)
        if not img_q.empty():
            _ = img_q.get_nowait()
        img_q.put(img)

def main():
    p = argparse.ArgumentParser("GPR B-scan with PyQtGraph")
    p.add_argument('--host',    required=True)
    p.add_argument('--port',    type=int, default=23)
    p.add_argument('--quantity',type=int, default=1024)
    p.add_argument('--range',   type=int, default=100)
    p.add_argument('--window',  type=int, default=1000)
    args = p.parse_args()

    # connect & setup
    setup = create_setup_message(args.quantity, args.range)
    sock  = socket.create_connection((args.host,args.port),timeout=5)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    gc.disable()
    sock.sendall((setup+"\n").encode()); sock.sendall(b"P1\n")
    if binascii.hexlify(sock.recv(4))!=ACK_HEX:
        print("Bad ACK"); sys.exit(1)
    sock.recv(1)

    # shared buffer & queue
    svc = args.quantity//16
    main_n = args.quantity - svc
    data = np.zeros((main_n,args.window), dtype=np.int16)
    img_q = queue.Queue(maxsize=1)
    stop_evt = threading.Event()

    # start reader thread
    t = threading.Thread(target=reader,
                         args=(sock,args,data,img_q,stop_evt),
                         daemon=True)
    t.start()

    # Qt application + window
    app = QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, 
           title="Real-time GPR B-scan")
    plot = win.addPlot()
    img_item = pg.ImageItem(border='w')
    plot.addItem(img_item)
    plot.invertY(True)      # origin at top
    plot.getViewBox().setAspectLocked(False)
    plot.getAxis('left').setLabel('Sample (time→depth)')
    plot.getAxis('bottom').setLabel('Trace #')

    # update function, called every 0 ms if possible
    def update():
        if not img_q.empty():
            frame = img_q.get()
            img_item.setImage(frame, levels=(0,255),
                              autoRange=False,
                              autoLevels=False)

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(0)  # as fast as Qt can

    # clean exit on window close
    win.closeEvent = lambda ev: (stop_evt.set(), sock.close(), gc.enable(), ev.accept())

    QtGui.QApplication.instance().exec_()

if __name__=='__main__':
    main()

