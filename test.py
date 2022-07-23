import os.path as op
import os
import mne
import numpy as np

def split_f(p):
    data_path,fname = os.path.split(p)
    fname,ext = os.path.splitext(fname)
    return data_path,fname,ext

def compute_cov_identity(raw_filename):
    "Compute Identity Noise Covariance matrix."
    raw = mne.io.read_raw_fif(raw_filename)

    data_path, basename, ext = split_f(raw_filename)
    cov_fname = op.join(data_path, 'identity_noise-cov.fif')

    if not op.isfile(cov_fname):
        picks = mne.pick_types(raw.info, eeg=True)

        ch_names = [raw.info['ch_names'][k] for k in picks]
        bads = [b for b in raw.info['bads'] if b in ch_names]
        noise_cov = mne.Covariance(np.identity(len(picks)), ch_names, bads,
                                   raw.info['projs'], nfree=0)

        mne.write_cov(cov_fname, noise_cov)

    return cov_fname

subjects_dir = 'data/freesurfer/subjects'
subject = 'P005'
eeg_locs = False #"data/ASP-64.bvef"
subject_path = f'data/subjects/{subject}'
eeg_path = os.path.join(subject_path,"2022-2478_T1_P1_P005.vhdr")
SNR = 10
SPACING = 'oct5'
TMAX=3
PREFLOOD=25 #https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/SkullStripFix_freeview
"""
the default preflooding height is 25, which produces the cleanest skull strip for most cases. There aren't any hard and fast rules about how to select your height value, but as a general rule of thumb, if part of the brain is missing, you should start with a watershed threshold around 35, and if too much skull is remaining, you should start with a threshold of around 15.
"""

#%% Save a fif file of eeg + location info
raw = mne.io.read_raw_brainvision(eeg_path, preload=True,eog=('EOG1','HEOGL', 'HEOGR', 'VEOGb'))
raw = raw.pick_types(eeg=True) # Remove non-eeg channels
raw = raw.crop(tmax=TMAX)
if eeg_locs:
    montage = mne.channels.read_custom_montage(eeg_locs)

    raw = raw.set_montage(montage)

raw_fpath=eeg_path.replace('.vhdr','_raw.fif')
raw.save(raw_fpath,overwrite=True) #N seconds to make test fast, should be None in real case

#%% Make source space
src_path = op.join(op.dirname(eeg_path),f'{subject}-{SPACING}-src.fif')

if not os.path.exists(src_path):
    src = mne.setup_source_space(subject, spacing=SPACING,subjects_dir=subjects_dir)
    mne.write_source_spaces(src_path, src,overwrite=False)
else:
    src = mne.read_source_spaces(src_path)

bem_path = os.path.join(subjects_dir,subject,'bem')
bem_files = [os.path.join(bem_path,x) for x in ['inner_skull.surf','outer_skin.surf','outer_skull.surf']]
model_fname = op.join(bem_path, f'{subject}-5120-bem.fif')
bem_fname = op.join(bem_path, f'{subject}-5120-bem-sol.fif')

#%% Make watershed BEM surfaces
if not np.all([os.path.exists(x) for x in bem_files]):
    mne.bem.make_watershed_bem(subject, subjects_dir,preflood=PREFLOOD, overwrite=False)

#%% Revisar si las capas quedaron bien
#mne.viz.plot_bem(subject, subjects_dir=subjects_dir)

#%% Do coregistration
#mne.gui.coregistration(subjects_dir=subjects_dir)

trans_fname = os.path.join(subject_path,f'{subject}_trans.fif')
#https://mne.tools/stable/overview/faq.html#faq-watershed-bem-meshes

if not os.path.exists(model_fname):
    # Create a BEM model for a subject
    surfaces = mne.make_bem_model(subject, ico=4,subjects_dir=subjects_dir)#,conductivity=[0.3])
    # Write BEM surfaces to a fiff file
    mne.write_bem_surfaces(model_fname, surfaces)
else:
    surfaces = mne.read_bem_surfaces(model_fname)

# Create a BEM solution using the linear collocation approach
if not os.path.exists(bem_fname):
    bem = mne.make_bem_solution(surfaces)
    mne.write_bem_solution(bem_fname, bem)
else:
    bem = mne.read_bem_solution(bem_fname)

"""Compute leadfield matrix by BEM."""
data_path, raw_fname = os.path.split(raw_fpath)
raw_fname = os.path.splitext(raw_fname)[0]
fwd_filename = raw_fname + '-' + SPACING

if not os.path.exists(fwd_filename):

    mindist = 5.  # ignore sources <= 0mm from inner skull
    fwd = mne.make_forward_solution(raw_fpath, trans_fname, src, bem,
                                mindist=mindist, meg=False, eeg=True,
                                n_jobs=2)


    fwd_filename = op.join(data_path, fwd_filename + '-fwd.fif')
    mne.write_forward_solution(fwd_filename, fwd, overwrite=True)
else:
    print(('\n*** READ FWD SOL %s ***\n' % fwd_filename))
    forward = mne.read_forward_solution(fwd_filename)

cov_fname = op.join(data_path, 'identity_noise-cov.fif')
if not os.path.exists(cov_fname):
    compute_cov_identity(raw_fpath)
    noise_cov = mne.read_cov(cov_fname)
else:
    noise_cov = mne.read_cov(cov_fname)

lambda2 = 1.0 / SNR ** 2

src_name = os.path.join(data_path,raw_fname +'-src')
if not os.path.exists(src_name+'-lh.stc'):
    inv = mne.minimum_norm.make_inverse_operator(
    raw.info, fwd, noise_cov, verbose=True)
    #raw,_ = mne.set_eeg_reference(raw, ref_channels='average')
    raw.set_eeg_reference('average', projection=True)
    stc = mne.minimum_norm.apply_inverse_raw(raw, inv,lambda2=lambda2,method='MNE')
    stc.save(src_name)
else:
    stc = mne.read_source_estimate(src_name)
brain = stc.plot(subject=subject,subjects_dir=subjects_dir,hemi='split',brain_kwargs=dict(block=True))