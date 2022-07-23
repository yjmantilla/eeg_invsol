import os.path as op
import mne

subjects_dir = 'data/freesurfer/subjects'
subject = 'FS_P005'
eeg_path = "data/subjects/P005/2022-2478_T1_P1_P005.vhdr"
eeg_locs = "data/ASP-64.bvef"
SPACING = 'oct5'
#%% Save a fif file of eeg + location info
raw = mne.io.read_raw_brainvision(eeg_path, preload=True,eog=('EOG1','HEOGL', 'HEOGR', 'VEOGb'))
raw = raw.pick_types(eeg=True) # Remove non-eeg channels

montage = mne.channels.read_custom_montage(eeg_locs)

raw = raw.set_montage(montage)

raw.save(eeg_path.replace('.vhdr','_raw.fif'),tmin=0,tmax=10) #10 seconds to make test fast, should be None in real case

#%% Make source space
src = mne.setup_source_space(subject, spacing=SPACING,subjects_dir=subjects_dir)
src_path = op.join(op.dirname(eeg_path),f'{subject}-{SPACING}-src.fif')
mne.write_source_spaces(src_path, src,overwrite=True)  

#%% Make watershed BEM surfaces
mne.bem.make_watershed_bem(subject, subjects_dir, overwrite=True)
# model = make_bem_model('sample') 
# #%% Do coregistration

# mne.gui.coregistration(subjects_dir=subjects_dir)
