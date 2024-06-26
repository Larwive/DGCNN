from typing import Any
from os.path import join
from numpy import uint8, array
from scipy.io import loadmat
import matplotlib.pyplot as plt
import mne
from scipy.integrate import simps

# The path for the dataset.
dataset_path = 'DREAMER'

sampling_rate = 128  # From the DREAMER dataset
channel_labels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8',
                  'AF4']  # From the DREAMER dataset
# TODO: Remove AF, T, P, O, do clustering
info = mne.create_info(channel_labels, sfreq=sampling_rate, ch_types='eeg')  # Create raw data metadata to be used later


def read_dataset():
    """
    Read the dataset and save the data as `fif` files in the same directory as the dataset.
    :return: None
    """
    data = loadmat(join(dataset_path, 'DREAMER.mat'))
    dreamer_data = data['DREAMER'][0, 0]['Data']
    for patient in range(23):
        for type in ['baseline', 'stimuli']:
            for movie in range(18):
                path = join(dataset_path, 'p{}_{}_m{}_raw.fif'.format(patient, type, movie))
                print("Saving {}...".format(path))
                raw = mne.io.RawArray(dreamer_data[0, patient]['EEG'][0][0][type][0, 0][movie][0].T, info)
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
    dreamer_data = data['DREAMER'][0, 0]['Data']
    for patient in range(23):
        val = dreamer_data[0, patient]['ScoreValence'][0, :][0]
        for value in val:
            valence.append(value[0])
        aro = dreamer_data[0, patient]['ScoreArousal'][0, :][0]
        for value in aro:
            arousal.append(value[0])
        dom = dreamer_data[0, patient]['ScoreDominance'][0, :][0]
        for value in dom:
            dominance.append(value[0])
    return valence, arousal, dominance


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
    return mne.io.read_raw_fif(join(dataset_path, 'p{}_{}_m{}_raw.fif'.format(n_patient, type, n_movie)),
                               allow_maxshield=allow_maxshield, preload=preload, on_split_missing=on_split_missing,
                               verbose=verbose)


def get_features(raw):
    """
    Extracts features from raw data as described in the paper:
    -Keep the last 60 seconds of data (for stimuli)
    -Filter the Theta, Alpha, Beta frequency bands
    -Compute PSD for every 2 seconds (with 1-second overlapping window between each time frame, resulting in 59 frames)

    :param raw: Raw EEG data extracted from the DREAMER dataset (use read_dataset and get_raw).
    :return: PSD features for a stimuli.
    """
    last_index = raw.n_times - 1
    cropped = raw.get_data()[:, last_index - 60 * 128:last_index]
    orig_ch_names = raw.info['ch_names']
    orig_sfreq = raw.info['sfreq']

    info_cropped = mne.create_info(orig_ch_names, orig_sfreq, ch_types=['eeg'] * len(raw.info['ch_names']))
    info_cropped['description'] = "Cropped EEG data"

    raw_cropped = mne.io.RawArray(cropped, info_cropped, verbose=0)
    raw_cropped.set_meas_date(raw.info['meas_date'])

    # Filter to keep the frequency bands
    theta = raw_cropped.copy().filter(4, 8, verbose=0).get_data()
    alpha = raw_cropped.copy().filter(8, 13, verbose=0).get_data()
    beta = raw_cropped.copy().filter(13, 20, verbose=0).get_data()

    for i in range(59):
        theta_cropped = theta[:, i * 128:(i + 2) * 128]
        alpha_cropped = alpha[:, i * 128:(i + 2) * 128]
        beta_cropped = beta[:, i * 128:(i + 2) * 128]
        raw_theta_cropped = mne.io.RawArray(theta_cropped, info_cropped, verbose=0)
        raw_alpha_cropped = mne.io.RawArray(alpha_cropped, info_cropped, verbose=0)
        raw_beta_cropped = mne.io.RawArray(beta_cropped, info_cropped, verbose=0)

        # PSD
        theta_frequencies, theta_psd = mne.time_frequency.psd_array_multitaper(raw_theta_cropped.get_data(), sfreq=128,
                                                                               fmin=4,
                                                                               fmax=8, verbose=0)
        alpha_frequencies, alpha_psd = mne.time_frequency.psd_array_multitaper(raw_alpha_cropped.get_data(), sfreq=128,
                                                                               fmin=8,
                                                                               fmax=13, verbose=0)
        beta_frequencies, beta_psd = mne.time_frequency.psd_array_multitaper(raw_beta_cropped.get_data(), sfreq=128,
                                                                             fmin=13,
                                                                             fmax=20, verbose=0)

        #yield theta_psd, theta_frequencies, alpha_psd, alpha_frequencies, beta_psd, beta_frequencies
        #yield [[array(lis).mean(0)] for lis in theta_frequencies], [[array(lis).mean(0)] for lis in alpha_frequencies], [[array(lis).mean(0)] for lis in beta_frequencies]
        yield [[simps(lis,dx=0.5)] for lis in theta_frequencies], [[simps(lis,dx=0.5)] for lis in
                                                                    alpha_frequencies], [[simps(lis,dx=0.5)] for lis in
                                                                                         beta_frequencies]


if __name__ == '__main__':
    # read_dataset() # Only need to run once to save as `fif` files.

    t_channel_labels = ['AF3 (theta)', 'F7 (theta)', 'F3 (theta)', 'FC5 (theta)', 'T7 (theta)', 'P7 (theta)',
                        'O1 (theta)', 'O2 (theta)', 'P8 (theta)', 'T8 (theta)', 'FC6 (theta)', 'F4 (theta)', 'F8',
                        'AF4 (theta)']
    a_channel_labels = ['AF3 (alpha)', 'F7 (alpha)', 'F3 (alpha)', 'FC5 (alpha)', 'T7 (alpha)', 'P7 (alpha)',
                        'O1 (alpha)', 'O2 (alpha)', 'P8 (alpha)', 'T8 (alpha)', 'FC6 (alpha)', 'F4 (alpha)', 'F8',
                        'AF4 (alpha)']
    b_channel_labels = ['AF3 (beta)', 'F7 (beta)', 'F3 (beta)', 'FC5 (beta)', 'T7 (beta)', 'P7 (beta)',
                        'O1 (beta)', 'O2 (beta)', 'P8 (beta)', 'T8 (beta)', 'FC6 (beta)', 'F4 (beta)', 'F8',
                        'AF4 (beta)']
    test_raw = read_raw(n_patient=1, type='stimuli', n_movie=10)
    for theta__psd, theta__frequencies, alpha__psd, alpha__frequencies, beta__psd, beta__frequencies in get_features(
            test_raw):
        plt.figure()
        plt.plot(theta__psd, theta__frequencies.T, label=t_channel_labels)
        plt.plot(alpha__psd, alpha__frequencies.T, label=a_channel_labels)
        plt.plot(beta__psd, beta__frequencies.T, label=b_channel_labels)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (µV²/Hz)')
        plt.title('Power Spectral Density')
        plt.legend()
    plt.show()
