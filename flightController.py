#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python code for the Raspberry Pi in XT-S.A.L.O. project for NASA SpaceApps 2019
Needs:
    - Servos attached on pins 13 (left) and 18 (right). 
    - Internet connection to the the API-client of the TCP-IP protocol
    - MPU 6050 IMU connected via I2C

This code reads data form the IMU, applies a basic Kalman filter to the data, 
receives the reference points from the API and applies PID control for roll and
pitch, moving the servomotors. 

DISCLAIMER
This is a prototyping code and is in no way intended to be more
than part of a prototype. No data response to the ground station has been 
implemented. 

REFECENCES
1. MPU-6050 IMU reference: 
    https://www.electronicwings.com/raspberry-pi/mpu6050-accelerometergyroscope-interfacing-with-raspberry-pi

2. Asynchronous socketserver protocol:
    https://docs.python.org/3/library/socketserver.html#socketserver.ThreadingTCPServer

3. Basic Kalman filter (translated from Arduino):
    http://blog.tkjelectronics.dk/2012/09/a-practical-approach-to-kalman-filter-and-how-to-implement-it/
    
    
@author: Jose Javier Aguilar
"""
# Packages
import socket
import threading
import socketserver
import smbus
import time
import math
import numpy
import RPi.GPIO as GPIO
import socketserver
import socket
import re

# Constants
# some MPU6050 Registers and their Address
PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
INT_ENABLE   = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H  = 0x43
GYRO_YOUT_H  = 0x45
GYRO_ZOUT_H  = 0x47

# Constants for Kalman Filters
qa = 0.001;
qb = 0.003;
rm = 0.03;
P0 = [[0.0,0.0],[0.0,0.0]];
init_angle = 0.0
init_rate = 0.0
init_bias  = 0.0


# PID control constants
# Yet to be calibrated for flight
kp_pitch = 10    # pitch - proportional
ki_pitch = 0.1   # pitch - integral
kd_pitch = 0.01  # pitch - derivative

kp_roll = 10    # roll - proportional
ki_roll = 0.1   # roll - integral
kd_roll = 0.01  # roll - derivative




# Global variable for reference values
refs = [0,0]


# Socketserver definition
class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.data = self.request.recv(1024).strip()
        re.findall(r"[-+]?\d*\.\d+|\d+",self.data)
        refs[0] = float(re.findall(r"[-+]?\d*\.\d+|\d+",self.data)[0])
        refs[1] = float(re.findall(r"[-+]?\d*\.\d+|\d+",self.data)[1])

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

# MPU functions definition
def MPU_Init():
    #write to sample rate register
    bus.write_byte_data(Device_Address, SMPLRT_DIV, 7)
    
    #Write to power management register
    bus.write_byte_data(Device_Address, PWR_MGMT_1, 1)
    
    #Write to Configuration register
    bus.write_byte_data(Device_Address, CONFIG, 0)
    
    #Write to Gyro configuration register
    bus.write_byte_data(Device_Address, GYRO_CONFIG, 24)
    
    #Write to interrupt enable register
    bus.write_byte_data(Device_Address, INT_ENABLE, 1)

def read_raw_data(addr):
    #Accelero and Gyro value are 16-bit
        high = bus.read_byte_data(Device_Address, addr)
        low = bus.read_byte_data(Device_Address, addr+1)
    
        #concatenate higher and lower value
        value = ((high << 8) | low)
        
        #to get signed value from mpu6050
        if(value > 32768):
                value = value - 65536
        return value

def read_accel_gyro():
    # Read Accelerometer raw value
    acc_x = read_raw_data(ACCEL_XOUT_H)
    acc_y = read_raw_data(ACCEL_YOUT_H)
    acc_z = read_raw_data(ACCEL_ZOUT_H)
    
    # Read Gyroscope raw value
    gyro_x = read_raw_data(GYRO_XOUT_H)
    gyro_y = read_raw_data(GYRO_YOUT_H)
    
    # Careful with IMU orientation
    th = math.atan2(acc_x,acc_z)
    ph = math.atan2(acc_y,acc_x)
    q = gyro_y*0.00026646248122050834
    p = gyro_x*0.00026646248122050834
    return (th, ph, q, p)
    
# Kalman filters definition
class KalmanAngle:
    # basic kalman filter
    def __init__(self, angle, rate, bias, pmat, qangle, qbias, rmeasure):
        self.Angle = angle
        self.Bias  = bias
        self.Rate  = rate
        self.Pmat  = pmat
        self.Qangle = qangle
        self.Qbias  = qbias
        self.Rmeasure = rmeasure
        self.Tcurr = time.time()
        self.Tprev = 0.0
    
    def updateAngle(self, newAngle, newRate, t):
        # time step
        self.Tprev = self.Tcurr
        self.Tcurr = t
        dt = self.Tcurr - self.Tprev;
        # predictions
        self.Rate  = newRate - self.Bias
        self.Angle = self.Angle + dt*self.Rate
        
        # kalman filtering
        P = self.Pmat
        P[0][0] = P[0][0] + dt * (dt*P[1][1] - P[0][1] - P[1][0] + self.Qangle)
        P[0][1] = P[0][1] - dt * P[1][1]
        P[1][0] = P[1][0] - dt * P[1][1]
        P[1][1] = P[1][1] + self.Qbias * dt
        
        S = P[0][0] + self.Rmeasure
        K = [P[0][0] / S, P[1][0] / S]
        y = newAngle - self.Angle
        self.Angle = self.Angle + K[0] * y
        self.Bias  = self.Bias  + K[1] * y
        
        P00_temp = P[0][0]
        P01_temp = P[0][1]
        
        P[0][0] = P[0][0] - K[0] * P00_temp
        P[0][1] = P[0][1] - K[0] * P01_temp
        P[1][0] = P[1][0] - K[1] * P00_temp
        P[1][1] = P[1][1] - K[1] * P01_temp
        self.Pmat = P

# PID controls definitions    
class PIDControl:
    def __init__(self, kp, ki, kd, ref, npts):
        self.Kp  = kp
        self.Ki  = ki
        self.Kd  = kd
        self.Ref = ref
        self.Npts= npts
        self.Errval = numpy.zeros([npts, 2])
        self.Errint = 0.0;
        self.Errdif = 0.0;
        self.control = 0.0;
        
    def updateControl(self, measure, t):
        error = self.Ref - measure
        self.Errval[1:,:] = self.Errval[:-1,:]
        self.Errval[0, 0] = error
        self.Errval[0, 1] = t
        # integral error
        self.Errint = self.Errint + 0.5 * (self.Errval[0, 0] + self.Errval[1, 0]) * (self.Errval[0, 0] - self.Errval[1, 0]) 
        # differential error
        # Cubic fit of Npts points
        deg = 3
        A = numpy.zeros([self.Npts, deg + 1])
        B = numpy.zeros([self.Npts])
        for i in range(0, self.Npts):
            A[i,0] = 1
            B[i] = self.Errval[i,0]
            for j in range(1, deg+1):
                A[i,j] = self.Errval[i,1]**j
        coeffs = numpy.linalg.lstsq(A,B)[0]
        # derivative of the interpolation
        self.Errdif = 0.0;
        for i in range(0, deg):
            self.Errdif = self.Errdif + (deg - i) * coeffs[i] * (t**(deg - i - 1))
        # control inpout calculation
        self.Control = error*self.Kp + self.Errint*self.Ki + self.Errdif*self.Kd

## MAIN PROGRAM BODY ##

# Init MPU 6050
bus = smbus.SMBus(1)    # or bus = smbus.SMBus(0) for older version boards
Device_Address = 0x68   # MPU6050 device address
MPU_Init()

# Init kalmans
the = KalmanAngle(init_angle, init_rate, init_bias, P0, qa, qb, rm)
phi = KalmanAngle(init_angle, init_rate, init_bias, P0, qa, qb, rm)

pitch_control = PIDControl(kp_pitch, ki_pitch, kd_pitch, 0.0, 5)
roll_control  = PIDControl(kp_roll , ki_roll , kd_roll , 0.0, 5)

# Init servos
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)
GPIO.setup(13, GPIO.OUT)

servo0 = GPIO.PWM(18, 500)
servo1 = GPIO.PWM(13, 500)

servo0.start(0)
servo1.start(0)
    
if __name__ == "__main__":
    # Start the socketserver
    HOST, PORT = socket.gethostname(), 9182
    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    with server:
        ip, port = server.server_address
        # Start a thread with the server -- that thread will then start one
        # more thread for each request
        server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()
        print("Server loop running in thread:", server_thread.name)
        
        # MAIN LOOP
        try:
            while True:
                # Read the data from the IMU
                (th, ph, q, p) = read_accel_gyro()
                # Update refetence values
                pitch_control.Ref = refs[0];
                roll_control.Ref = refs[1];
                # Update Kalman Filters
                t = time.time()
                the.updateAngle(th, q, t)
                phi.updateAngle(ph, p, t)
                # Apply PID controls
                pitch_control.updateControl(the.Angle, t);
                roll_control.updateControl(the.Angle, t);
                
                # Send signals to servos - between 1 and 99
                sig0 = pitch_control.Control + 0.5*roll_control.Control + 50
                if sig0 > 99:
                    sig0 = 99
                elif sig0 < 1:
                    sig0 = 1
                    
                sig1 = roll_control.Control - 0.5*roll_control.Control + 50
                if sig1 > 99:
                    sig1 = 99
                elif sig1 < 1:
                    sig1 = 1
                
                servo0.ChangeDutyCycle(sig0)
                servo1.ChangeDutyCycle(sig0)
                
        except KeyboardInterrupt:
            servo0.stop()
            servo1.stop()
            GPIO.cleanup()
            server.shutdown()
        