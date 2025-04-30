#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This describes methods and attributes of class Inspector."""
__version__ = '1.0'
__author__ = 'Andrej Pistek'

import csv
from threading import Lock
from typing import Optional
import numpy as np
from PIL import Image
import os
from obspy.io.segy.segy import SEGYFile
from factory.abstract.camera import Camera
from factory.abstract.sensor import Sensor
from factory.sensor.sensor_gpr import GPR
from factory.dependencies.plotter import ColorMapPlotter, LinePlotter
from factory.dependencies.task_manager import TaskManager
from factory.inspector_hg import \
    InspectorHG, \
    ClassifierTensorflow, \
    ClassifierScikit, \
    ClassifierGPR, \
    Lidar, \
    GPS, \
    Spectrometer
import logging.handlers
import time
from io import BytesIO
from flask_socketio import emit
import platform

SYS_PATH = os.path.dirname(os.path.realpath(__file__))
SYS_PATH = SYS_PATH[:SYS_PATH.find('hidden_graves')]

if platform.system() != 'Linux':
    SYS_PATH = ''


class Inspector:
    inspector_hg: InspectorHG = None

    clf_tensorflow: ClassifierTensorflow = None
    clf_scikit: ClassifierScikit = None
    clf_gpr: ClassifierGPR = None

    gpr: GPR = None
    gps: GPS = None
    lidar: Lidar = None
    spectrometer: Spectrometer = None
    camera: Camera

    socketio_emit: emit = None

    stop_gps_linker = False
    is_gpr_logging = False

    # frame coordinates
    x = 0
    y = 0
    y2 = 1
    x2 = 1

    gpr_logger: Optional[SEGYFile] = None
    gpr_logger_last_700_traces: Optional[SEGYFile] = None
    gpr_logger_last_700_traces_lock = Lock()
    stop_gps = False
    stop_gpr = True

    backgound_gps = False
    __mutex = Lock()
    freeze_rgb = False

    gpr_log = None
    running_gps_linker = False
    _frame = None

    gpr_settings = {'typeOfSoil': 4,
                    'depth': 1}

    def __init__(self, socketio, debug=False) -> None:
        if debug:
            self.enable_debug_mode()
        if platform.system() == "Linux":
            from xvfbwrapper import Xvfb
            self.vdisplay = Xvfb(width=700, height=500, colordepth=16)
            self.vdisplay.start()
        self.socketio = socketio
        self.inspector_hg = InspectorHG()

        self.camera = self.inspector_hg.create_camera()
        self.gpr, self.gps, self.lidar, self.spectrometer = self.inspector_hg.create_sensors()
        self.clf_scikit, self.clf_gpr, self.clf_tensorflow = self.inspector_hg.create_models()

        self.task_manager = TaskManager()
        self.color_map_plotter = ColorMapPlotter()
        self.line_plotter = LinePlotter()

        self.static_gpr_image = open(SYS_PATH + 'resources/static_gpr_image.png', 'rb').read()
        self.static_hyperspectral_image = open(SYS_PATH + 'resources/static_hyperspectral_image.png', 'rb').read()

        self.gps.connect()
        # self.gps.run_()
        self.lidar.connect()
        self.lidar.run_()

    def get_lidar(self) -> Optional[Lidar]:
        """Lidar getter"""
        return self._get_sensor(self.lidar)

    def get_gps(self) -> Optional[GPS]:
        """GPS getter"""
        return self._get_sensor(self.gps)

    def get_spectrometer(self) -> Optional[Spectrometer]:
        """Spectrometer getter"""
        return self._get_sensor(self.spectrometer)

    def get_drone_height(self):
        """Drone height getter.
        :returns int: height of the drone
        """
        return self.lidar.get_data()

    def get_drone_gps(self):
        """Drone location getter.
        :returns NMVMA : GNGGA message
        """
        return self.gps.get_data()

    def get_gpr_output(self) -> bytes:
        """GPR B-scan plot generator. Emits LIDAR data on socketio event "lidar_update" as well.
                :returns bytes: GPR B-scan plot"""
        _byte_buffer = BytesIO()
        counter = 0

        file_path = self.task_manager.sys_path + self.task_manager.projects_path + self.task_manager.project_name + "/" + self.task_manager.gpr_path
        try:
            os.makedirs(file_path)
        except:
            pass
        file_path += '/'

        self.gpr_predictions_file_path = file_path

        with open(self.gpr_predictions_file_path + 'gpr_hyperbola_predictions.csv', 'w+') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerows([('Batch', 'False probability', 'True probability', 'Latitude', 'Longitude')])

        while True:
            if not self.stop_gpr:
                self.emit_('lidar_update', '')
                if self.gpr_logger_last_700_traces is not None:
                    with self.gpr_logger_last_700_traces_lock:
                        batch = self.gpr_logger_last_700_traces
                        self.gpr_logger_last_700_traces = None
                    if batch is not None:
                        self.color_map_plotter.resize_plot(self.gpr.plot_size + 1, self.gpr.sample_size + 1)
                        self.color_map_plotter.plot_(np.stack(t.data for t in batch.traces))
                        _bytes = self.color_map_plotter.export_scene_to_bytes()
                        results = self.clf_gpr.detect_hyperbola(batch)
                        processed_results = []
                        i = 0
                        for prediction in results[0]:
                            lat = round(batch.traces[i].header.y_source_coordinate, 7)
                            lon = round(batch.traces[i].header.x_source_coordinate, 7)
                            processed_results.append([counter, prediction[0], prediction[1], lat, lon])
                            i += 1
                            counter += 20
                        self.write_down_gpr_predictions(processed_results)
                else:
                    time.sleep(.01)
                    continue
            else:
                time.sleep(.08)
                self.emit_lidar_data()
                _bytes = self.static_gpr_image
            yield b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + _bytes + b'\r\n\r\n'

    def write_down_gpr_predictions(self, predictions):
        with open(self.gpr_predictions_file_path + 'gpr_hyperbola_predictions.csv', 'a') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerows(predictions)

    def get_spectrometer_output(self) -> bytes:
        """Spectrogram plot generator.
                :returns bytes: spectrogram plot with raw spectras and if calibrated also with ratio to white reference"""
        while True:
            if self.check_sensor_status(self.spectrometer):
                reflectance_ratio, raw_spectra, wavelengths = self.spectrometer.get_data()
                if raw_spectra is not None:
                    self.emit_('notificationMlTensorflow', self.clf_tensorflow.predict_spectra(raw_spectra))
                    self.line_plotter.plot_(raw_spectra, wavelengths, curve=0)
                    if reflectance_ratio is not None:
                        self.line_plotter.plot_(reflectance_ratio, wavelengths, curve=1)
                    _bytes = self.line_plotter.export_scene_to_bytes()
                else:
                    time.sleep(.01)
                    continue
            else:
                time.sleep(.5)
                _bytes = self.static_hyperspectral_image
            yield b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + _bytes + b'\r\n\r\n'

    def get_scikit_classifier_output(self) -> bytes:
        """Scikit-learn.LinearClassifier image classification's output generator.
                :returns bytes: frame from rgb camera with colored pixels from image classification"""
        _byte_buffer = BytesIO()
        self._frame = None
        if not self.camera.is_running_:
            self.camera.start_()
            self.reset_coordinates()
        while True:
            if not self.freeze_rgb:
                self._frame = self.camera.get_frame()
                if self.clf_scikit.is_trained() and not self.clf_scikit.is_training():
                    self._frame = self.clf_scikit.predict_frame(self._frame, self.y, self.y2, self.x, self.x2)
                Image.fromarray(self._frame, 'RGB').save(_byte_buffer, format='jpeg')
                _byte_buffer.seek(0)
                yield b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + _byte_buffer.getvalue() + b'\r\n\r\n'
            else:
                time.sleep(0.1)

    def train_scikit_model(self) -> None:
        """Train Sci-kit learn models based on the ROI."""
        self.clf_scikit.train(self._frame, self.x, self.x2, self.y, self.y2)

    def check_sensor_status(self, _sensor: Sensor) -> bool:
        """Verify provided instance of type Sensor. Method is taking every step possible to make sensor working.
        :parameter _sensor: instance which will be verified
        :returns status: True if sensor is ready False otherwise """
        if _sensor.is_connected():
            return self.reactivate_connected_sensor(_sensor)
        else:
            return self.reconnect_sensor(_sensor)

    def reconnect_sensor(self, _sensor: Sensor) -> bool:
        if isinstance(_sensor, Spectrometer):
            _sensor.connect(
                path=self.task_manager.get_current_project_path() + self.task_manager.spectral_reference_path)
        else:
            _sensor.connect()
        if _sensor.is_connected():
            if not _sensor.is_running():
                _sensor.start()
                return True
            else:
                return True
        else:
            return False

    @staticmethod
    def reactivate_connected_sensor(_sensor: Sensor) -> bool:
        try:
            if _sensor.is_sleeping():
                _sensor.wake_up()
                return True
            if not _sensor.is_running():
                if not _sensor._started.is_set():
                    _sensor.start()
                return True
        except Exception:
            return False

    def reset_coordinates(self):
        self.x = 0
        self.y = 0
        self.y2 = 1
        self.x2 = 1

    def write_gpr_data(self, filename):
        self.task_manager.write_gpr_data(self.gpr_logger, filename, self.gpr.time_range)

    def connect_spectrometer(self):
        self.spectrometer.connect(
            self.task_manager.get_current_project_path() + self.task_manager.spectral_reference_path)

    def emit_lidar_data(self) -> None:
        """LIDAR data emitter. Emits readings of the sensor on the "lidar_update" socketio event."""
        if self.lidar is not None:
            _height = self.lidar.get_data()
            if _height is not None and float(_height) > 0.30:
                _height = round(float(_height) - 0.30, 2)
            message = '{0:.2f}'.format(float(_height))
            self.emit_('lidar_update', message)

    def set_roi(self, y_start, y_end, x_start, x_end) -> None:
        """Region of interest setter."""
        with self.__mutex:
            self.x = x_start
            self.y = y_start
            self.x2 = x_end
            self.y2 = y_end

    def rgb_camera_is_alive(self) -> bool:
        """Life Control of Camera.
                :returns status: True if sensor passed check_sensor_status False otherwise"""
        return self.camera is not None

    def update_black_reference(self) -> None:
        """Request sender for updating black reference of Spectrometer"""
        self.spectrometer.update_black_reference(
            path=self.task_manager.get_current_project_path() + self.task_manager.spectral_reference_path)

    def update_white_reference(self) -> None:
        """Request sender for updating white reference of Spectrometer"""
        self.spectrometer.update_white_reference(
            path=self.task_manager.get_current_project_path() + self.task_manager.spectral_reference_path)

    def create_project(self, project_name) -> None:
        """Creates project folder with init template based on provided name."""
        spectrometer_type = 'HyperSlit-H2'
        gpr_type = 'Cobra_Zond-12e'
        gps_type = 'Reach_M_RTK'
        lidar_type = 'Lightware_rf_finder'
        camera_type = 'RaspberryPi_v2'
        self.task_manager.create_requested_project_name(project_name, spectrometer_type, gpr_type, gps_type, lidar_type,
                                                        camera_type)
        self.spectrometer.fetch_old_reference(
            self.task_manager.get_current_project_path() + self.task_manager.spectral_reference_path)

    def set_project(self, _project_name) -> None:
        """Set requested project name as current project."""
        self.task_manager.set_project(_project_name)
        self.spectrometer.fetch_old_reference(
            self.task_manager.get_current_project_path() + self.task_manager.spectral_reference_path)

    def emit_(self, event, data):
        """SocketIO emitter.
        :param data: any data needed to be broadcasted to provided event
        :type event: name of even on which the data will be broadcasted
        """
        self.socketio_emit({'event': event, 'payload': {'payload': data}})
        # self.socketio_emit('alert', 10)

    @staticmethod
    def enable_debug_mode() -> None:
        """Debug enabler for instances of type Sensor, Model and Camera."""
        logging.basicConfig(level=logging.CRITICAL)
        logging.getLogger('GPS').setLevel(logging.DEBUG)
        logging.getLogger('GPR').setLevel(logging.DEBUG)
        logging.getLogger('Lidar').setLevel(logging.DEBUG)
        logging.getLogger('Spectrometer').setLevel(logging.DEBUG)
        logging.getLogger('Camera').setLevel(logging.DEBUG)
        logging.getLogger('Sci-Kit').setLevel(logging.DEBUG)

    @staticmethod
    def _get_sensor(_sensor: Sensor) -> Optional[Sensor]:
        """Safe getter for instance of type Sensor.
        :parameter _sensor: requested instance
        :returns object: _sensor if it is connected None otherwise"""
        if _sensor.is_connected():
            return _sensor
        else:
            return None

    @property
    def socketio_emitter(self):
        assert self.socketio_emit is not None
        return self.socketio_emit

    @socketio_emitter.setter
    def socketio_emitter(self, emitter):
        assert emitter is not None
        self.socketio_emit = emitter
