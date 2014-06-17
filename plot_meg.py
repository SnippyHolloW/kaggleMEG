import numpy as np
from scipy.io import loadmat
from pylab import imshow, show
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 15, 10
import glob
subject_files = glob.glob("data/train*.mat")
print subject_files

d = loadmat(subject_files[0])
print d.keys()
time = np.linspace(float(d['tmin']), float(d['tmax']), num=d['X'].shape[2])
# d['X'].shape[2]/float(d['sfreq'][0])

# MAKE SURE you have memory before loading all the data!! (next cell)
data = d['X']
labels = d['y'].ravel()
'Aprox Mem needed : %.4f Gb' % (len(subject_files) * data.nbytes / float(2**30))

from mne.layouts import read_layout
layout = read_layout('Vectorview-all')
len(layout.names)

from mne.epochs import EpochsArray
from mne.io.meas_info import create_info
print data.shape

sfreq = d['sfreq']
ch_names = layout.names
print ch_names
types = ['mag'] * len(ch_names)
info = create_info(ch_names, sfreq, types)
events = np.vstack(zip([125 for _ in xrange(len(labels))], [0 for _ in xrange(len(labels))], labels))
print events.shape

#help(EpochsArray)
ea = EpochsArray(data, info, events, tmin=d['tmin'])

from mne.evoked import EvokedArray
evoked = EvokedArray(data[labels==1].mean(axis=0), info, tmin=d['tmin'])
print evoked
print info
times = np.arange(0.05, 0.15, 0.01)
print len(info['ch_names'])
evoked.plot_topomap(0.15, ch_type='mag', layout=layout, size=10, colorbar=False)
#evoked.plot_topomap(times, ch_type='mag', layout=layout, size=10, colorbar=False)

evoked = EvokedArray(data[labels==0].mean(axis=0), info, tmin=d['tmin'])
evoked.plot_topomap(0.15, ch_type='mag', layout=layout, size=10, colorbar=False)
#evoked.plot_topomap(times, ch_type='mag', layout=layout, size=10, colorbar=False)

evoked = EvokedArray(data[labels==1].mean(axis=0) - data[labels==0].mean(axis=0), info, tmin=d['tmin'])
evoked.plot_topomap(0.15, ch_type='mag', layout=layout, size=10, colorbar=False)


# TODO use the EpochsArray
