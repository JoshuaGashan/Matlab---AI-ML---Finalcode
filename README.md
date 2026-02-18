Accelerometer Time & Frequency Domain Dataset
This repository contains MATLAB data files (.mat) representing accelerometer sensor readings collected from multiple subjects. The data is organized into time-domain and frequency-domain representations to support machine learning and signal processing tasks.

File Structure
The dataset follows a specific naming convention: U[User]_Acc_[Domain]_[Session].mat.
User ID: U01 or U02 indicates the subject from whom the data was collected.
Sensor: Acc stands for Accelerometer data.

Domain:
TimeD: Raw or filtered signals in the Time Domain.
FreqD: Signals transformed into the Frequency Domain (e.g., via FFT).
Session: FDay (First Day) or MDay (Main/Subsequent Day) sessions.
Technical DetailsPlatform: Created on PCWIN64.Format: MATLAB 5.0 MAT-file.Data Type: Likely numerical arrays or structures containing 3-axis ($x, y, z$) acceleration values.


How to Use
In MATLAB
To load a specific file into your workspace, use the load command in the MATLAB Command Window:
"load('U01_Acc_TimeD_FDay.mat')
whos % To view the variables loaded"


In Python
You can access this data using the scipy.io library:
import scipy.io
"data = scipy.io.loadmat('U01_Acc_TimeD_FDay.mat')
print(data.keys())"
