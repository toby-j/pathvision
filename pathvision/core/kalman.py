import argparse
import cv2 as cv
import numpy as np
import sys

if sys.version_info[0] != 3:
    print("This module requires Python 3")
    sys.exit(1)


class KalmanBB:
    def __init__(self):
        self.nstates = 8
        self.nmeasures = 4
        self.kfilt = cv.KalmanFilter(self.nstates, self.nmeasures, 0)
        self.ticks = 0
        self.lastTicks = 0

        # state variable indices
        self.SBBX, self.SBBY, self.SV_X, self.SV_Y, self.SBBW, self.SBBH, self.SV_W, self.SV_H = range(self.nstates)
        # measurement variable indices
        self.MBBX, self.MBBY, self.MBBW, self.MBBH = range(self.nmeasures)

        # A: Transition State Matrix
        self.kfilt.transitionMatrix = np.eye(self.nstates, dtype=np.float32)
        self.kfilt.transitionMatrix[self.SBBX, self.SV_X] = 1.0
        self.kfilt.transitionMatrix[self.SBBY, self.SV_Y] = 1.0
        self.kfilt.transitionMatrix[self.SBBW, self.SV_W] = 1.0
        self.kfilt.transitionMatrix[self.SBBH, self.SV_H] = 1.0

        # H: Measurement Matrix
        self.kfilt.measurementMatrix = np.zeros((self.nmeasures, self.nstates), dtype=np.float32)
        self.kfilt.measurementMatrix[self.MBBX, self.SBBX] = 1.0
        self.kfilt.measurementMatrix[self.MBBY, self.SBBY] = 1.0
        self.kfilt.measurementMatrix[self.MBBW, self.SBBW] = 1.0
        self.kfilt.measurementMatrix[self.MBBH, self.SBBH] = 1.0

        # Q: Process Noise Covariance Matrix
        self.kfilt.processNoiseCov = np.eye(self.nstates, dtype=np.float32) * (1e-2)
        self.kfilt.processNoiseCov[self.SV_X, self.SV_X] = 2.0
        self.kfilt.processNoiseCov[self.SV_Y, self.SV_Y] = 2.0
        self.kfilt.processNoiseCov[self.SV_W, self.SV_W] = 2.0
        self.kfilt.processNoiseCov[self.SV_H, self.SV_H] = 2.0

        # R: Measurement Noise Covariance Matrix
        self.kfilt.measurementNoiseCov = np.eye(self.nmeasures, dtype=np.float32) * 0.1

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

    def predict(self, dT=-1):  # get predicted state
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

        self.kfilt.predict()

        # Keeping the values greater than or equal to 0
        self.kfilt.statePre[self.SBBX] = max(0.0, self.kfilt.statePre[self.SBBX])
        self.kfilt.statePre[self.SBBY] = max(0.0, self.kfilt.statePre[self.SBBY])
        self.kfilt.statePre[self.SBBW] = max(0.0, self.kfilt.statePre[self.SBBW])
        self.kfilt.statePre[self.SBBH] = max(0.0, self.kfilt.statePre[self.SBBH])

        # Returning the predicted values for bbx, bby, bbw, and bbh
        return np.float32(
            [self.kfilt.statePre[self.SBBX], self.kfilt.statePre[self.SBBY], self.kfilt.statePre[self.SBBW], self.kfilt.statePre[self.SBBH]]).squeeze()

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
            [self.kfilt.statePost[self.SBBX], self.kfilt.statePost[self.SBBY], self.kfilt.statePost[self.SBBW], self.kfilt.statePost[self.SBBH]]).squeeze()

    def getPostState(self):  # get the state after correction
        # Return a floating point array with the correct values for bbx, bby, bbw, bbh
        # Use if you want to get the state after correction without actually running correction again
        return np.float32(
            [self.kfilt.statePost[self.SBBX], self.kfilt.statePost[self.SBBY], self.kfilt.statePost[self.SBBW], self.kfilt.statePost[self.SBBH]]).squeeze()

    def track(self, bbx, bby, bbw, bbh, dT=-1, onlyPred=False):
        if (onlyPred):
            pred = self.predict(dT)
            cpred = self.correct(pred[0], pred[1], pred[2], pred[3], False)
        else:
            pred = self.predict(dT)
            cpred = self.correct(bbx, bby, bbw, bbh, False)
        return cpred



