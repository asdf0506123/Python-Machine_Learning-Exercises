import mne 
import numpy as np

#file path for the EEG data
file_path = 'data/file.edf'

#Load the EEG data using MNE
raw = mne.io.read_raw_edf(file_path, preload=True)

#Get the number of channels and samples
n_signals = raw.info['nchan']
sampling_rate = raw.info['sfreq']

#Retrieve the channels names
channels_names = []
for i in range(n_signals):
    channels_names.append(raw.ch_names[i])

#Read signal from all channels
signals = np.zeros((n_signals, raw.times))
for i in range(n_signals):
    signals[i, :] = raw.get_data(picks=i)

#Close the raw object to free memory
raw.close()

print(f"Number of channels: {n_signals}")
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Channel names: {channels_names}")
print(f"Signals shape: {signals.shape}")