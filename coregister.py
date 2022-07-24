import os.path as op
import os
import mne
import numpy as np

subjects_dir = 'data/freesurfer/subjects'
subject = 'fsaverage'
eeg_locs = False #"data/ASP-64.bvef"
subject_path = f'data/subjects/{subject}'
eeg_path = os.path.join(subject_path,f"2022-2478_T1_P1_{subject}.vhdr")
TMAX=3
raw = mne.io.read_raw_brainvision(eeg_path, preload=True,eog=('EOG1','HEOGL', 'HEOGR', 'VEOGb'))
raw = raw.pick_types(eeg=True) # Remove non-eeg channels
raw = raw.crop(tmax=TMAX)
if eeg_locs:
    montage = mne.channels.read_custom_montage(eeg_locs)

    raw = raw.set_montage(montage)

raw_fpath=eeg_path.replace('.vhdr','_raw.fif')
raw.save(raw_fpath,overwrite=True) #N seconds to make test fast, should be None in real case

mne.gui.coregistration(subjects_dir=subjects_dir)