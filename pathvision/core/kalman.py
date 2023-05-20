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


class KalmanTracker:
    global nstates, nmeasures, kfilt, SBBX, SBBY, SV_X, SV_Y, SBBW, SBBH, SV_W, SV_H, MBBX, MBBY, MBBW, MBBH
    nstates = 8  # bbx, bby, vx, vy, bbw, bbh, vw, bh (8x8 matrix)
    nmeasures = 4  # bbx, bby, bbw, bbh (4x4 matrix)
    kfilt = cv.KalmanFilter(nstates, nmeasures, 0)
    # state variable indices
    SBBX = 0
    SBBY = 1
    SV_X = 2
    SV_Y = 3
    SBBW = 4
    SBBH = 5
    SV_W = 6
    SV_H = 7
    # measurement variable indices
    MBBX = 0
    MBBY = 1
    MBBW = 2
    MBBH = 3

    def __init__(self):
        # For keeping track of time
        self.ticks = 0
        self.lastTicks = 0

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
        kfilt.transitionMatrix = np.eye(nstates, dtype=np.float32)

        # H: Measurement Matrix
        #   x  y vx vt  w  h vw vh
        # [ 1  0  0  0  0  0  0  0 ] x
        # [ 0  1  0  0  0  0  0  0 ] y
        # [ 0  0  0  0  1  0  0  0 ] w
        # [ 0  0  0  0  0  1  0  0 ] h
        kfilt.measurementMatrix = np.zeros((nmeasures, nstates), dtype=np.float32)
        kfilt.measurementMatrix[MBBX, SBBX] = 1.0
        kfilt.measurementMatrix[MBBY, SBBY] = 1.0
        kfilt.measurementMatrix[MBBW, SBBW] = 1.0
        kfilt.measurementMatrix[MBBH, SBBH] = 1.0

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

        kfilt.processNoiseCov = np.eye(nstates, dtype=np.float32) * (1e-2)
        # Override velocity errors
        kfilt.processNoiseCov[SV_X, SV_X] = 2.0
        kfilt.processNoiseCov[SV_Y, SV_Y] = 2.0
        kfilt.processNoiseCov[SV_W, SV_W] = 2.0
        kfilt.processNoiseCov[SV_H, SV_H] = 2.0

        # R: Measurement Noise Covariance Matrix
        # Higher value increases uncertainty to give more weight to prediction
        # Lower value decreases uncertainty to give more weight to measurement
        kfilt.measurementNoiseCov = np.eye(nmeasures, dtype=np.float32) * (0.08)

    def init(self, meas):
        state = np.zeros(kfilt.statePost.shape, np.float32)
        state[SBBX] = meas[0]
        state[SBBY] = meas[1]
        state[SV_X] = 0
        state[SV_Y] = 0
        state[SBBW] = meas[2]
        state[SBBH] = meas[3]
        state[SV_W] = 0
        state[SV_H] = 0
        kfilt.statePost = state
        kfilt.statePre = state
        self.lastTicks = self.ticks
        self.ticks = cv.getTickCount()
        return meas

    def predict(self, dT=-1):  # get predicted state
        # Updating the time
        self.lastTicks = self.ticks
        self.ticks = cv.getTickCount()

        # Keeping track of the passage of time in seconds
        if dT == -1:
            dT = 1.0 * (self.ticks - self.lastTicks) / cv.getTickFrequency()

        # Update the transition Matrix A with dT for this stamp
        kfilt.transitionMatrix[SBBX, SV_X] = dT
        kfilt.transitionMatrix[SBBY, SV_Y] = dT
        kfilt.transitionMatrix[SBBW, SV_W] = dT
        kfilt.transitionMatrix[SBBH, SV_H] = dT

        kfilt.predict()

        # Keeping the values greater than or equal to 0
        kfilt.statePre[SBBX] = max(0.0, kfilt.statePre[SBBX])
        kfilt.statePre[SBBY] = max(0.0, kfilt.statePre[SBBY])
        kfilt.statePre[SBBW] = max(0.0, kfilt.statePre[SBBW])
        kfilt.statePre[SBBH] = max(0.0, kfilt.statePre[SBBH])

        # Returning the predicted values for bbx, bby, bbw, and bbh
        return np.float32(
            [kfilt.statePre[SBBX], kfilt.statePre[SBBY], kfilt.statePre[SBBW], kfilt.statePre[SBBH]]).squeeze()

    def correct(self, bbx, bby, bbw, bbh, restart=False):  # state correction using measurement matrix with BB

        if restart:
            self.ticks = cv.getTickCount()
            cv.setIdentity(kfilt.errorCovPre, 1.0)

            # Updating statePost with bbx, bby, bbw, and bbh
            kfilt.statePost[SBBX] = bbx
            kfilt.statePost[SBBX] = bby
            kfilt.statePost[SV_X] = 0
            kfilt.statePost[SV_Y] = 0
            kfilt.statePost[SBBW] = bbw
            kfilt.statePost[SBBH] = bbh
            kfilt.statePost[SV_W] = 0
            kfilt.statePost[SV_H] = 0

        else:
            # Running correct with bbx, bby, bbw, and bbh
            kfilt.correct(np.float32([bbx, bby, bbw, bbh]).squeeze())

        # Keeping the values greater than or equal to 0
        kfilt.statePost[SBBX] = max(0.0, kfilt.statePost[SBBX])
        kfilt.statePost[SBBY] = max(0.0, kfilt.statePost[SBBY])
        kfilt.statePost[SBBW] = max(0.0, kfilt.statePost[SBBW])
        kfilt.statePost[SBBH] = max(0.0, kfilt.statePost[SBBH])

        # Return a floating point array with the corrected values for bbx, bby, bbw, bbh
        return np.float32(
            [kfilt.statePost[SBBX], kfilt.statePost[SBBY], kfilt.statePost[SBBW], kfilt.statePost[SBBH]]).squeeze()

    def getPostState(self):  # get the state after correction
        # Return a floating point array with the correct values for bbx, bby, bbw, bbh
        # Use if you want to get the state after correction without actually running correction again
        return np.float32(
            [kfilt.statePost[SBBX], kfilt.statePost[SBBY], kfilt.statePost[SBBW], kfilt.statePost[SBBH]]).squeeze()

    def track(self, bbx, bby, bbw, bbh, dT=-1, onlyPred=False):
        if onlyPred:
            pred = self.predict(dT)
            cpred = self.correct(pred[0], pred[1], pred[2], pred[3], False)
        else:
            pred = self.predict(dT)
            cpred = self.correct(bbx, bby, bbw, bbh, False)
        return cpred



