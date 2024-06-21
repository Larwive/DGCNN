from typing import Any
from os.path import join
from numpy import uint8
from scipy.io import loadmat
import matplotlib.pyplot as plt
import mne

# The path for the dataset (should be 'DREAMER').
dataset_path = 'DREAMER'

sampling_rate = 128  # From the DREAMER dataset
channel_labels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8',
                  'AF4']  # From the DREAMER dataset
info = mne.create_info(channel_labels, sfreq=sampling_rate, ch_types='eeg')  # Create raw data metadata to be used later


def read_dataset():
    """
    Read the dataset and save the data as `fif` files in the same directory as the dataset.
    :return: None
    """
    data = loadmat(join(dataset_path, 'DREAMER.mat'))
    dreamerdata = data['DREAMER'][0, 0]['Data']
    for patient in range(23):
        for type in ['baseline', 'stimuli']:
            for movie in range(18):
                path = join(dataset_path, 'p{}_{}_m{}_raw.fif'.format(patient, type, movie))
                print("Saving {}...".format(path))
                raw = mne.io.RawArray(dreamerdata[0, patient]['EEG'][0][0][type][0, 0][movie][0].T, info)
                raw.save(path, overwrite=True)


def read_valence_arousal_dominance():
    """
    Read the valence, arousal, and dominance values from the dataset in order (patient first then movie).
    :return: Lists of `numpy.uint8` corresponding to valence, arousal, and dominance values.
    """
    valence: list[uint8] = []
    arousal: list[uint8] = []
    dominance: list[uint8] = []
    data = loadmat(join(dataset_path, 'DREAMER.mat'))
    dreamerdata = data['DREAMER'][0, 0]['Data']
    for patient in range(23):
        val = dreamerdata['Data'][0, patient]['ScoreValence'][0, :][0]
        for value in val:
            valence.append(value[0])
        aro = dreamerdata['Data'][0, patient]['ScoreArousal'][0, :][0]
        for value in aro:
            arousal.append(value[0])
        dom = dreamerdata['Data'][0, patient]['ScoreDominance'][0, :][0]
        for value in dom:
            dominance.append(value[0])
    return valence, arousal, dominance

"""
raw = mne.io.read_raw_fif('DREAMER/p0_baseline_m0_raw.fif')
raw.plot(scalings='auto')
raw.crop(tmin=0, tmax=1)

raw2 = mne.io.read_raw_fif('DREAMER/p0_stimuli_m0_raw.fif')
raw2.plot(scalings='auto')

plt.show()"""


def read_raw(n_patient: int, type: str, n_movie: int, allow_maxshield: Any = False,
            preload: bool = False,
            on_split_missing: str = "raise",
            verbose: Any = None):
    """
    A `fif` file opener wrapper with mne.io.read_raw_fif for the DREAMER dataset.
    :param n_patient: The patient number (from 0 to 22 for the DREAMER dataset).
    :param type: Either 'baseline' or 'stimuli'.
    :param n_movie: The movie number (from 0 to 17 for the DREAMER dataset).
    :param allow_maxshield: See mne.io.read_raw_fif
    :param preload: See mne.io.read_raw_fif
    :param on_split_missing: See mne.io.read_raw_fif
    :param verbose: See mne.io.read_raw_fif
    :return: A Raw object containing FIF data.
    """
    return mne.io.read_raw_fif(join(dataset_path, '/p{}_{}_m{}_raw.fif'.format(n_patient, type, n_movie)),
                               allow_maxshield=allow_maxshield, preload=preload, on_split_missing=on_split_missing,
                               verbose=verbose)


def get_features(raw):
    """
    Extracts features from raw data as described in the paper:
    -Keep the last 60 seconds of data (for stimuli)
    -Filter the Theta, Alpha, Beta frequency bands
    -Compute PSD

    :param raw: Raw EEG data extracted from the DREAMER dataset (use read_dataset and get_raw).
    :return: PSD features for a stimuli.
    """
    last_index = raw.n_times - 1
    cropped = raw.get_data()[:, last_index - 60 * 128:last_index]
    orig_ch_names = raw.info['ch_names']
    orig_sfreq = raw.info['sfreq']

    info_cropped = mne.create_info(orig_ch_names, orig_sfreq, ch_types=['eeg'] * len(raw.info['ch_names']))
    info_cropped['meas_date'] = raw.info['meas_date']
    info_cropped['description'] = "Cropped EEG data (last 60 seconds)"

    raw_cropped = mne.io.RawArray(cropped, info_cropped)

    theta = raw_cropped.copy().filter(4, 8)
    alpha = raw_cropped.copy().filter(8, 13)
    beta = raw_cropped.copy().filter(13, 20)

    for i in range(59):
        theta_cropped = theta.get_data()[:, i * 128:(i + 1) * 128]
        alpha_cropped = alpha.get_data()[:, i * 128:(i + 1) * 128]
        beta_cropped = beta.get_data()[:, i * 128:(i + 1) * 128]
        raw_theta_cropped = mne.io.RawArray(theta_cropped, info_cropped)
        raw_alpha_cropped = mne.io.RawArray(alpha_cropped, info_cropped)
        raw_beta_cropped = mne.io.RawArray(beta_cropped, info_cropped)

        # PSD
        theta_frequencies, theta_psd = mne.time_frequency.psd_array_multitaper(raw_theta_cropped, sfreq=128, fmin=4,
                                                                               fmax=8, n_fft=500)
        alpha_frequencies, alpha_psd = mne.time_frequency.psd_array_multitaper(raw_alpha_cropped, sfreq=128, fmin=8,
                                                                               fmax=13, n_fft=500)
        beta_frequencies, beta_psd = mne.time_frequency.psd_array_multitaper(raw_beta_cropped, sfreq=128, fmin=13,
                                                                             fmax=20, n_fft=500)


if __name__ == '__main__':
    #read_dataset()

    pass