from scipy.io import loadmat
import matplotlib.pyplot as plt
import mne

data = loadmat('DREAMER/DREAMER.mat')
# print(data.keys())
dreamerdata = data['DREAMER'][0, 0]

"""
print(dreamerdata)
print(dreamerdata.dtype.names)
print(dreamerdata['EEG_SamplingRate'])
print(dreamerdata['ECG_SamplingRate'])
print(dreamerdata['EEG_Electrodes'])
print(dreamerdata['noOfSubjects'])
print(dreamerdata['Data'].shape)
print(dreamerdata['Data'][0, 0].shape)
print(len(dreamerdata['Data'][0, 0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0]))"""

sampling_rate = 128
channel_labels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

info = mne.create_info(channel_labels, sfreq=sampling_rate, ch_types='eeg')
valence = []
arousal = []
dominance = []

# print(dreamerdata['Data'][0, 0]['EEG'][0][0]['baseline'][0, 0][0][0].T)
for patient in range(23):
    for type in ['baseline', 'stimuli']:
        for movie in range(18):
            # title = 'DREAMER/p{}_{}_m{}raw.fif'.format(patient, type, movie)
            # print("Saving {}...".format(title))
            # raw = mne.io.RawArray(dreamerdata['Data'][0, patient]['EEG'][0][0][type][0, 0][movie][0].T, info)

            # raw.save(title, overwrite=True)
            pass
    val = dreamerdata['Data'][0, patient]['ScoreValence'][0, :][0]
    for value in val:
        valence.append(value[0])
    aro = dreamerdata['Data'][0, patient]['ScoreArousal'][0, :][0]
    for value in aro:
        arousal.append(value[0])
    dom = dreamerdata['Data'][0, patient]['ScoreDominance'][0, :][0]
    for value in dom:
        dominance.append(value[0])

print(valence, arousal, dominance)
exit(0)

print(dreamerdata['Data'][0, 0]['ScoreValence'][0, :][0])

raw = mne.io.read_raw_fif('DREAMER/p0_baseline_m0raw.fif')
raw.plot(scalings='auto')
raw.crop(tmin=0, tmax=1)

raw2 = mne.io.read_raw_fif('DREAMER/p0_stimuli_m0raw.fif')
raw2.plot(scalings='auto')

plt.show()


def get_features(raw):
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
