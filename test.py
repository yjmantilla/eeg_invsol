import os.path as op
import os
import mne
import numpy as np

subjects_dir = 'data/freesurfer/subjects'
subject = 'FS_P005'
eeg_path = "data/subjects/P005/2022-2478_T1_P1_P005.vhdr"
eeg_locs = False #"data/ASP-64.bvef"
SPACING = 'oct5'
PREFLOOD=25
#%% Save a fif file of eeg + location info
raw = mne.io.read_raw_brainvision(eeg_path, preload=True,eog=('EOG1','HEOGL', 'HEOGR', 'VEOGb'))
raw = raw.pick_types(eeg=True) # Remove non-eeg channels

if eeg_locs:
    montage = mne.channels.read_custom_montage(eeg_locs)

    raw = raw.set_montage(montage)

raw.save(eeg_path.replace('.vhdr','_raw.fif'),tmin=0,tmax=10,overwrite=True) #10 seconds to make test fast, should be None in real case

#%% Make source space
src_path = op.join(op.dirname(eeg_path),f'{subject}-{SPACING}-src.fif')

if not os.path.exists(src_path):
    src = mne.setup_source_space(subject, spacing=SPACING,subjects_dir=subjects_dir)
    mne.write_source_spaces(src_path, src,overwrite=False)
else:
    src = mne.read_source_spaces(src_path)

bem_path = os.path.join(subjects_dir,subject,'bem')
bem_files = [os.path.join(bem_path,x) for x in ['inner_skull.surf','outer_skin.surf','outer_skull.surf']]

#%% Make watershed BEM surfaces
if not np.all([os.path.exists(x) for x in bem_files]):
    mne.bem.make_watershed_bem(subject, subjects_dir,preflood=PREFLOOD, overwrite=False)

#%% Revisar si las capas quedaron bien
mne.viz.plot_bem(subject, subjects_dir=subjects_dir)

#%% Do coregistration
mne.gui.coregistration(subjects_dir=subjects_dir)

#%% Make BEM
mne.viz.plot_bem(subject, subjects_dir=subjects_dir)

#https://mne.tools/stable/overview/faq.html#faq-watershed-bem-meshes
# Create a BEM model for a subject
surfaces = mne.make_bem_model(subject, ico=3,subjects_dir=subjects_dir)#,conductivity=[0.3])

# Write BEM surfaces to a fiff file
# mne.write_bem_surfaces(model_fname, surfaces)

# Create a BEM solution using the linear collocation approach
# bem = mne.make_bem_solution(surfaces)
# mne.write_bem_solution(bem_fname, bem)

# print(('\n*** BEM solution file {} written ***\n'.format(bem_fname)))