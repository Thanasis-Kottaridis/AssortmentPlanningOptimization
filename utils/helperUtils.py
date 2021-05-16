import numpy as np
import pandas as pd

def calculateRMSE_PersentagePresision(y_test, predictionRMSE):
    return 100 - ((100 * predictionRMSE) / np.array(y_test).mean())