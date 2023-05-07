# Copyright 2023 Toby Johnson. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import cv2 as cv
import numpy as np
import sys

if sys.version_info[0] != 3:
    print("This module requires Python 3")
    sys.exit(1)

'''transitionMatrix: 
    Each row of the transition matrix corresponds to an element of the state vector at time t, 
    and each column corresponds to the same element at time t+1. The diagonal elements represent the current state of 
    each variable, and the off-diagonal elements represent how the values of these variables evolve over time.

    In particular, the non-zero elements of the transition matrix indicate how the state variables are expected to 
    change over time. For example, the first row of the matrix specifies that the position in the x direction is 
    expected to increase by an amount equal to the velocity in the x direction multiplied by the elapsed time between 
    updates. Similarly, the fifth and sixth rows indicate that the width and height of the bounding box are expected 
    to remain constant, while the seventh and eighth rows indicate that the velocity in the width and height 
    directions are expected to remain constant as well.

    The identity matrix on the right-hand side of the assignment sets the initial value of the transition matrix to an 
    identity matrix, which assumes that the state variables are uncorrelated and evolve independently over time. The 
    subsequent assignments update the elements of the transition matrix corresponding to the position, width, and height 
    variables to reflect the expected changes in these variables due to the elapsed time between updates.

    .correct() is used to update the state of the kalman filter

'''


class KalmanBB:
    def __init__(self):
        self.nstates = 8
        self.measurements = 4
        self.kfilt = cv.KalmanFilter(self.nstates, self.measurements, 0)
        self.ticks = 0
        self.lastTicks = 0

        # state variable indices
        self.SBBX, self.SBBY, self.SV_X, self.SV_Y, self.SBBW, self.SBBH, self.SV_W, self.SV_H = range(self.nstates)
        # measurement variable indices
        self.MBBX, self.MBBY, self.MBBW, self.MBBH = range(self.measurements)

        # A: Transition State Matrix
        #   x  y vx vt  w  h vw vh
        # [ 1  0 dT  0  0  0  0  0 ] => dT at TDTX
        # [ 0  1  0 dT  0  0  0  0 ] => dT at TDTY
        # [ 0  0  1  0  0  0  0  0 ]
        # [ 0  0  0  1  0  0  0  0 ]
        # [ 0  0  0  0  1  0 dT  0 ] => dT at TDTW
        # [ 0  0  0  0  0  1  0 dT ] => dT at TDTH
        # [ 0  0  0  0  0  0  1  0 ]
        # [ 0  0  0  0  0  0  0  1 ]
        self.kfilt.transitionMatrix = np.eye(self.nstates, dtype=np.float32)
        self.kfilt.transitionMatrix[self.SBBX, self.SV_X] = 1.0
        self.kfilt.transitionMatrix[self.SBBY, self.SV_Y] = 1.0
        self.kfilt.transitionMatrix[self.SBBW, self.SV_W] = 1.0
        self.kfilt.transitionMatrix[self.SBBH, self.SV_H] = 1.0

        # H: Measurement Matrix
        #   x  y vx vt  w  h vw vh
        # [ 1  0  0  0  0  0  0  0 ] x
        # [ 0  1  0  0  0  0  0  0 ] y
        # [ 0  0  0  0  1  0  0  0 ] w
        # [ 0  0  0  0  0  1  0  0 ] h
        self.kfilt.measurementMatrix = np.zeros((self.measurements, self.nstates), dtype=np.float32)
        self.kfilt.measurementMatrix[self.MBBX, self.SBBX] = 1.0
        self.kfilt.measurementMatrix[self.MBBY, self.SBBY] = 1.0
        self.kfilt.measurementMatrix[self.MBBW, self.SBBW] = 1.0
        self.kfilt.measurementMatrix[self.MBBH, self.SBBH] = 1.0

        # Q: Process Noise Covariance Matrix
        #   x    y   vx   vt    w    h    vw   vh
        # [ Ebbx 0    0    0    0    0    0    0   ]
        # [ 0    Ebby 0    0    0    0    0    0   ]
        # [ 0    0    Ev_x 0    0    0    0    0   ]
        # [ 0    0    0    Ev_y 0    0    0    0   ]
        # [ 0    0    0    0    Ebbw 0    0    0   ]
        # [ 0    0    0    0    0    Ebbh 0    0   ]
        # [ 0    0    0    0    0    0    Ev_w 0   ]
        # [ 0    0    0    0    0    0    0    Ev_h]
        self.kfilt.processNoiseCov = np.eye(self.nstates, dtype=np.float32) * (1e-2)
        # Override velocity errors
        self.kfilt.processNoiseCov[self.SV_X, self.SV_X] = 2.0
        self.kfilt.processNoiseCov[self.SV_Y, self.SV_Y] = 2.0
        self.kfilt.processNoiseCov[self.SV_W, self.SV_W] = 2.0
        self.kfilt.processNoiseCov[self.SV_H, self.SV_H] = 2.0

        # R: Measurement Noise Covariance Matrix
        # Higher value increases uncertainty to give more weight to prediction
        # Lower value decreases uncertainty to give more weight to measurement
        self.kfilt.measurementNoiseCov = np.eye(self.measurements, dtype=np.float32) * 0.1

    def init(self, meas):
        state = np.zeros(self.kfilt.statePost.shape, np.float32)
        state[self.SBBX] = meas[0]
        state[self.SBBY] = meas[1]
        state[self.SV_X] = 0
        state[self.SV_Y] = 0
        state[self.SBBW] = meas[2]
        state[self.SBBH] = meas[3]
        state[self.SV_W] = 0
        state[self.SV_H] = 0
        self.kfilt.statePost = state
        self.kfilt.statePre = state
        self.lastTicks = self.ticks
        self.ticks = cv.getTickCount()
        return meas

    def make_prediction(self, dT=-1):  # get predicted state
        # Updating the time
        self.lastTicks = self.ticks
        self.ticks = cv.getTickCount()

        # Keeping track of the passage of time in seconds
        if dT == -1:
            dT = 1.0 * (self.ticks - self.lastTicks) / cv.getTickFrequency()

        # Update the transition Matrix A with dT for this stamp
        self.kfilt.transitionMatrix[self.SBBX, self.SV_X] = dT
        self.kfilt.transitionMatrix[self.SBBY, self.SV_Y] = dT
        self.kfilt.transitionMatrix[self.SBBW, self.SV_W] = dT
        self.kfilt.transitionMatrix[self.SBBH, self.SV_H] = dT

        # Step the prediction, we now need to normalise our new coordinates
        self.kfilt.predict()

        # Keeping the values greater than or equal to 0
        self.kfilt.statePre[self.SBBX] = max(0.0, self.kfilt.statePre[self.SBBX])
        self.kfilt.statePre[self.SBBY] = max(0.0, self.kfilt.statePre[self.SBBY])
        self.kfilt.statePre[self.SBBW] = max(0.0, self.kfilt.statePre[self.SBBW])
        self.kfilt.statePre[self.SBBH] = max(0.0, self.kfilt.statePre[self.SBBH])

        # Returning the predicted values for bbx, bby, bbw, and bbh
        return np.float32(
            [self.kfilt.statePre[self.SBBX], self.kfilt.statePre[self.SBBY], self.kfilt.statePre[self.SBBW],
             self.kfilt.statePre[self.SBBH]]).squeeze()

    def correct(self, bbx, bby, bbw, bbh, restart=False):  # state correction using measurement matrix with BB

        if restart:
            self.ticks = cv.getTickCount()
            cv.setIdentity(self.kfilt.errorCovPre, 1.0)

            # Updating statePost with bbx, bby, bbw, and bbh
            self.kfilt.statePost[self.SBBX] = bbx
            self.kfilt.statePost[self.SBBX] = bby
            self.kfilt.statePost[self.SV_X] = 0
            self.kfilt.statePost[self.SV_Y] = 0
            self.kfilt.statePost[self.SBBW] = bbw
            self.kfilt.statePost[self.SBBH] = bbh
            self.kfilt.statePost[self.SV_W] = 0
            self.kfilt.statePost[self.SV_H] = 0

        else:
            # Running correct with bbx, bby, bbw, and bbh
            self.kfilt.correct(np.float32([bbx, bby, bbw, bbh]).squeeze())

        # Keeping the values greater than or equal to 0
        self.kfilt.statePost[self.SBBX] = max(0.0, self.kfilt.statePost[self.SBBX])
        self.kfilt.statePost[self.SBBY] = max(0.0, self.kfilt.statePost[self.SBBY])
        self.kfilt.statePost[self.SBBW] = max(0.0, self.kfilt.statePost[self.SBBW])
        self.kfilt.statePost[self.SBBH] = max(0.0, self.kfilt.statePost[self.SBBH])

        # Return a floating point array with the corrected values for bbx, bby, bbw, bbh
        return np.float32(
            [self.kfilt.statePost[self.SBBX], self.kfilt.statePost[self.SBBY], self.kfilt.statePost[self.SBBW],
             self.kfilt.statePost[self.SBBH]]).squeeze()

    def getPostState(self):  # get the state after correction
        # Return a floating point array with the correct values for bbx, bby, bbw, bbh
        # Use if you want to get the state after correction without actually running correction again
        return np.float32(
            [self.kfilt.statePost[self.SBBX], self.kfilt.statePost[self.SBBY], self.kfilt.statePost[self.SBBW],
             self.kfilt.statePost[self.SBBH]]).squeeze()


