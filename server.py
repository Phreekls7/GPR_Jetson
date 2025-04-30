import time
import gc
import threading

from flask import Flask
from flask_socketio import SocketIO
from inspector import Inspector  # local file

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')

# init Inspector from inspector.py
inspector = Inspector(socketio=socketio, debug=True)
inspector.socketio_emitter = lambda data: socketio.emit(data['event'], data['payload'])
inspector.create_project('Default')
inspector.set_project('Default')
gc.enable()

@socketio.on('start_gpr')
def handle_start_gpr(msg):
    # start GPR scan on the Cobra unit
    inspector.is_gpr_logging = True
    inspector.gpr.run(
        sample_quantity=msg['sampleQuantity'],
        depth_index=msg['depthIndex'],
        frequency=msg['scanningFrequency'],
        gps_latest_message=inspector.gps.latest_message,
        gps_lock=inspector.gps.gps_lock,
        lidar_latest_message=inspector.lidar.latest_message,
        lidar_lock=inspector.lidar.lidar_lock
    )
    inspector.stop_gpr = False

    # collect scans in a background thread
    def collect():
        while not inspector.stop_gpr:
            with inspector.gpr_logger_last_700_traces_lock:
                m = inspector.gpr.queue.get()
                inspector.gpr_logger_last_700_traces = m['stream']
                inspector.gpr_logger = m['logger']
            time.sleep(0.5)
        # ensure logger is set before exit
        while inspector.gpr_logger is None:
            m = inspector.gpr.queue.get()
            inspector.gpr_logger = m['logger']

    t = threading.Thread(target=collect)
    t.daemon = True
    t.start()

@socketio.on('get_raw_gpr')
def handle_get_raw(msg):
    # wait up to 5s for data
    deadline = time.time() + 5
    while not inspector.gpr_logger_last_700_traces and time.time() < deadline:
        time.sleep(0.1)
    count = msg.get('count', len(inspector.gpr_logger_last_700_traces))
    data = inspector.gpr_logger_last_700_traces[-count:]
    socketio.emit('raw_gpr_data', data)

if __name__ == '__main__':
    # listens on port 5000
    socketio.run(app, host='0.0.0.0', port=23, debug=False)
