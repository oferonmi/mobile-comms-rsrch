# -*- coding: utf-8 -*-
"""
OFDM MIMO Detection - Implements OFDM MIMO Channel Estimation and Detection
Created on Wed Dec 28 16:37:02 2022

"""
#%% GPU Configuration and Imports
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0 # Number of the GPU to be used
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)
        
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pickle

from tensorflow.keras import Model

# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna

from sionna.mimo import StreamManagement
from sionna.utils import QAMSource, compute_ser, BinarySource, sim_ber, ebnodb2no, QAMSource
from sionna.mapping import Mapper
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEInterpolator, LinearDetector, KBestDetector, EPDetector, MMSEPICDetector
from sionna.channel import GenerateOFDMChannel, OFDMChannel, gen_single_sector_topology
from sionna.channel.tr38901 import UMi, Antenna, PanelArray
from sionna.fec.ldpc import LDPC5GEncoder
from sionna.fec.ldpc import LDPC5GDecoder

#%% Simualtion parameters

NUM_OFDM_SYMBOLS = 14
FFT_SIZE = 12*4 # 4 PRBs
SUBCARRIER_SPACING = 30e3 # Hz
CARRIER_FREQUENCY = 3.5e9 # Hz
SPEED = 3. # m/s

# The user terminals (UTs) are equipped with a single antenna
# with vertial polarization.
UT_ANTENNA = Antenna(polarization='single',
                     polarization_type='V',
                     antenna_pattern='omni', # Omnidirectional antenna pattern
                     carrier_frequency=CARRIER_FREQUENCY)

# The base station is equipped with an antenna
# array of 8 cross-polarized antennas,
# resulting in a total of 16 antenna elements.
NUM_RX_ANT = 16
BS_ARRAY = PanelArray(num_rows_per_panel=4,
                      num_cols_per_panel=2,
                      polarization='dual',
                      polarization_type='cross',
                      antenna_pattern='38.901', # 3GPP 38.901 antenna pattern
                      carrier_frequency=CARRIER_FREQUENCY)

# 3GPP UMi channel model is considered
CHANNEL_MODEL = UMi(carrier_frequency=CARRIER_FREQUENCY,
                    o2i_model='low',
                    ut_array=UT_ANTENNA,
                    bs_array=BS_ARRAY,
                    direction='uplink',
                    enable_shadow_fading=False,
                    enable_pathloss=False)

#%%
rg = ResourceGrid(num_ofdm_symbols=NUM_OFDM_SYMBOLS,
                  fft_size=FFT_SIZE,
                  subcarrier_spacing=SUBCARRIER_SPACING)
channel_sampler = GenerateOFDMChannel(CHANNEL_MODEL, rg)

def sample_channel(batch_size):
    # Sample random topologies
    topology = gen_single_sector_topology(batch_size, 1, 'umi', min_ut_velocity=SPEED, max_ut_velocity=SPEED)
    CHANNEL_MODEL.set_topology(*topology)

    # Sample channel frequency responses
    # [batch size, 1, num_rx_ant, 1, 1, num_ofdm_symbols, fft_size]
    h_freq = channel_sampler(batch_size)
    # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
    h_freq = h_freq[:,0,:,0,0]

    return h_freq

@tf.function(jit_compile=True) # Use XLA for speed-up
def estimate_covariance_matrices(num_it, batch_size):
    freq_cov_mat = tf.zeros([FFT_SIZE, FFT_SIZE], tf.complex64)
    time_cov_mat = tf.zeros([NUM_OFDM_SYMBOLS, NUM_OFDM_SYMBOLS], tf.complex64)
    space_cov_mat = tf.zeros([NUM_RX_ANT, NUM_RX_ANT], tf.complex64)
    for _ in tf.range(num_it):
        # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
        h_samples = sample_channel(batch_size)

        #################################
        # Estimate frequency covariance
        #################################
        # [batch size, num_rx_ant, fft_size, num_ofdm_symbols]
        h_samples_ = tf.transpose(h_samples, [0,1,3,2])
        # [batch size, num_rx_ant, fft_size, fft_size]
        freq_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [fft_size, fft_size]
        freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=(0,1))
        # [fft_size, fft_size]
        freq_cov_mat += freq_cov_mat_

        ################################
        # Estimate time covariance
        ################################
        # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
        time_cov_mat_ = tf.matmul(h_samples, h_samples, adjoint_b=True)
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat_ = tf.reduce_mean(time_cov_mat_, axis=(0,1))
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat += time_cov_mat_

        ###############################
        # Estimate spatial covariance
        ###############################
        # [batch size, num_ofdm_symbols, num_rx_ant, fft_size]
        h_samples_ = tf.transpose(h_samples, [0,2,1,3])
        # [batch size, num_ofdm_symbols, num_rx_ant, num_rx_ant]
        space_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [num_rx_ant, num_rx_ant]
        space_cov_mat_ = tf.reduce_mean(space_cov_mat_, axis=(0,1))
        # [num_rx_ant, num_rx_ant]
        space_cov_mat += space_cov_mat_

    freq_cov_mat /= tf.complex(tf.cast(NUM_OFDM_SYMBOLS*num_it, tf.float32), 0.0)
    time_cov_mat /= tf.complex(tf.cast(FFT_SIZE*num_it, tf.float32), 0.0)
    space_cov_mat /= tf.complex(tf.cast(FFT_SIZE*num_it, tf.float32), 0.0)

    return freq_cov_mat, time_cov_mat, space_cov_mat

#%%
#number of samples = batch_size x num_iterations

batch_size = 1000
num_iterations = 100

sionna.Config.xla_compat = True # Enable Sionna's support of XLA
FREQ_COV_MAT, TIME_COV_MAT, SPACE_COV_MAT = estimate_covariance_matrices(batch_size, num_iterations)
sionna.Config.xla_compat = False # Disable Sionna's support of XLA

# FREQ_COV_MAT : [fft_size, fft_size]
# TIME_COV_MAT : [num_ofdm_symbols, num_ofdm_symbols]
# SPACE_COV_MAT : [num_rx_ant, num_rx_ant]

np.save('freq_cov_mat', FREQ_COV_MAT.numpy())
np.save('time_cov_mat', TIME_COV_MAT.numpy())
np.save('space_cov_mat', SPACE_COV_MAT.numpy())

FREQ_COV_MAT = np.load('freq_cov_mat.npy')
TIME_COV_MAT = np.load('time_cov_mat.npy')
SPACE_COV_MAT = np.load('space_cov_mat.npy')

#%% Comparison of OFDM estimators
class MIMOOFDMLink(Model):

    def __init__(self, int_method, lmmse_order=None, **kwargs):
        super().__init__(kwargs)

        assert int_method in ('nn', 'lin', 'lmmse')


        # Configure the resource grid
        rg = ResourceGrid(num_ofdm_symbols=NUM_OFDM_SYMBOLS,
                          fft_size=FFT_SIZE,
                          subcarrier_spacing=SUBCARRIER_SPACING,
                          num_tx=1,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2,11])
        self.rg = rg

        # Stream management
        # Only a sinlge UT is considered for channel estimation
        sm = StreamManagement([[1]], 1)

        ##################################
        # Transmitter
        ##################################

        self.qam_source = QAMSource(num_bits_per_symbol=2) # Modulation order does not impact the channel estimation. Set to QPSK
        self.rg_mapper = ResourceGridMapper(rg)

        ##################################
        # Channel
        ##################################

        self.channel = OFDMChannel(CHANNEL_MODEL, rg, return_channel=True)

        ###################################
        # Receiver
        ###################################

        # Channel estimation
        freq_cov_mat = tf.constant(FREQ_COV_MAT, tf.complex64)
        time_cov_mat = tf.constant(TIME_COV_MAT, tf.complex64)
        space_cov_mat = tf.constant(SPACE_COV_MAT, tf.complex64)
        
        if int_method == 'nn':
            self.channel_estimator = LSChannelEstimator(rg, interpolation_type='nn')
        elif int_method == 'lin':
            self.channel_estimator = LSChannelEstimator(rg, interpolation_type='lin')
        elif int_method == 'lmmse':
            lmmse_int_freq_first = LMMSEInterpolator(rg.pilot_pattern, time_cov_mat, freq_cov_mat, space_cov_mat, order=lmmse_order)
            self.channel_estimator = LSChannelEstimator(rg, interpolator=lmmse_int_freq_first)

    @tf.function
    def call(self, batch_size, snr_db):


        ##################################
        # Transmitter
        ##################################

        x = self.qam_source([batch_size, 1, 1, self.rg.num_data_symbols])
        x_rg = self.rg_mapper(x)

        ##################################
        # Channel
        ##################################

        no = tf.pow(10.0, -snr_db/10.0)
        topology = gen_single_sector_topology(batch_size, 1, 'umi', min_ut_velocity=SPEED, max_ut_velocity=SPEED)
        CHANNEL_MODEL.set_topology(*topology)
        y_rg, h_freq = self.channel((x_rg, no))

        ###################################
        # Channel estimation
        ###################################

        h_hat,_ = self.channel_estimator((y_rg,no))

        ###################################
        # MSE
        ###################################

        mse = tf.reduce_mean(tf.square(tf.abs(h_freq-h_hat)))

        return mse
    
def evaluate_mse(model, snr_dbs, batch_size, num_it):

    # Casting model inputs to TensorFlow types to avoid
    # re-building of the graph
    snr_dbs = tf.cast(snr_dbs, tf.float32)
    batch_size = tf.cast(batch_size, tf.int32)

    mses = []
    for snr_db in snr_dbs:

        mse_ = 0.0
        for _ in range(num_it):
            mse_ += model(batch_size, snr_db).numpy()
        # Averaging over the number of iterations
        mse_ /= float(num_it)
        mses.append(mse_)

    return mses

# Range of SNR (in dB)
SNR_DBs = np.linspace(-10.0, 20.0, 20)

# Number of iterations and batch size.
# These parameters control the number of samples used to compute each SNR value.
# The higher the number of samples is, the more accurate the MSE estimation is, at
# the cost of longer compute time.
BATCH_SIZE = 512
NUM_IT = 10

# Interpolation/filtering order for the LMMSE interpolator.
# All valid configurations are listed.
# Some are commented to speed-up simulations.
# Uncomment configurations to evaluate them!
ORDERS = ['s-t-f', # Space - time - frequency
          #'s-f-t', # Space - frequency - time
          #'t-s-f', # Time - space - frequency
          't-f-s', # Time - frequency - space
          #'f-t-s', # Frequency - time - space
          #'f-s-t', # Frequency - space- time
          #'f-t',   # Frequency - time (no spatial smoothing)
          't-f'   # Time - frequency (no spatial smoothing)
          ]

MSES = {}

# Nearest-neighbor interpolation
e2e = MIMOOFDMLink("nn")
MSES['nn'] = evaluate_mse(e2e, SNR_DBs, BATCH_SIZE, NUM_IT)

# Linear interpolation
e2e = MIMOOFDMLink("lin")
MSES['lin'] = evaluate_mse(e2e, SNR_DBs, BATCH_SIZE, NUM_IT)

# LMMSE
for order in ORDERS:
    e2e = MIMOOFDMLink("lmmse", order)
    MSES[f"lmmse: {order}"] = evaluate_mse(e2e, SNR_DBs, BATCH_SIZE, NUM_IT)

# Plot MSES
plt.figure(figsize=(8,6))

for est_label in MSES:
    plt.semilogy(SNR_DBs, MSES[est_label], label=est_label)

plt.xlabel(r"SNR (dB)")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)

#%% Comparison of MIMO detectors

class MIMOOFDMLink(Model):

    def __init__(self, output, det_method, perf_csi, num_tx, num_bits_per_symbol, det_param=None, coderate=0.5, **kwargs):
        super().__init__(kwargs)

        assert det_method in ('lmmse', 'k-best', 'ep', 'mmse-pic'), "Unknown detection method"

        self._output = output
        self.num_tx = num_tx
        self.num_bits_per_symbol = num_bits_per_symbol
        self.coderate = coderate
        self.det_method = det_method
        self.perf_csi = perf_csi

        # Configure the resource grid
        rg = ResourceGrid(num_ofdm_symbols=NUM_OFDM_SYMBOLS,
                          fft_size=FFT_SIZE,
                          subcarrier_spacing=SUBCARRIER_SPACING,
                          num_tx=num_tx,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2,11])
        self.rg = rg

        # Stream management
        sm = StreamManagement(np.ones([1,num_tx], int), 1)

        # Codeword length and number of information bits per codeword
        n = int(rg.num_data_symbols*num_bits_per_symbol)
        k = int(coderate*n)
        self.n = n
        self.k = k

        # If output is symbol, then no FEC is used and hard decision are output
        hard_out = (output == "symbol")
        coded = (output == "bit")
        self.hard_out = hard_out
        self.coded = coded

        ##################################
        # Transmitter
        ##################################

        self.binary_source = BinarySource()
        self.mapper = Mapper(constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, return_indices=True)
        self.rg_mapper = ResourceGridMapper(rg)
        if coded:
            self.encoder = LDPC5GEncoder(k, n, num_bits_per_symbol=num_bits_per_symbol)

        ##################################
        # Channel
        ##################################

        self.channel = OFDMChannel(CHANNEL_MODEL, rg, return_channel=True)

        ###################################
        # Receiver
        ###################################

        # Channel estimation
        if not self.perf_csi:
            freq_cov_mat = tf.constant(FREQ_COV_MAT, tf.complex64)
            time_cov_mat = tf.constant(TIME_COV_MAT, tf.complex64)
            space_cov_mat = tf.constant(SPACE_COV_MAT, tf.complex64)
            lmmse_int_time_first = LMMSEInterpolator(rg.pilot_pattern, time_cov_mat, freq_cov_mat, space_cov_mat, order='t-f-s')
            self.channel_estimator = LSChannelEstimator(rg, interpolator=lmmse_int_time_first)

        # Detection
        if det_method == "lmmse":
            self.detector = LinearDetector("lmmse", output, "app", rg, sm, constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, hard_out=hard_out)
        elif det_method == 'k-best':
            if det_param is None:
                k = 64
            else:
                k = det_param
            self.detector = KBestDetector(output, num_tx, k, rg, sm, constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, hard_out=hard_out)
        elif det_method == "ep":
            if det_param is None:
                l = 10
            else:
                l = det_param
            self.detector = EPDetector(output, rg, sm, num_bits_per_symbol, l=l, hard_out=hard_out)
        elif det_method == 'mmse-pic':
            if det_param is None:
                l = 4
            else:
                l = det_param
            self.detector = MMSEPICDetector(output, rg, sm, 'app', num_iter=l, constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, hard_out=hard_out)

        if coded:
            self.decoder = LDPC5GDecoder(self.encoder, hard_out=False)

    @tf.function
    def call(self, batch_size, ebno_db):


        ##################################
        # Transmitter
        ##################################

        if self.coded:
            b = self.binary_source([batch_size, self.num_tx, 1, self.k])
            c = self.encoder(b)
        else:
            c = self.binary_source([batch_size, self.num_tx, 1, self.n])
        bits_shape = tf.shape(c)
        x,x_ind = self.mapper(c)
        x_rg = self.rg_mapper(x)

        ##################################
        # Channel
        ##################################

        no = ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate, resource_grid=self.rg)
        topology = gen_single_sector_topology(batch_size, self.num_tx, 'umi', min_ut_velocity=SPEED, max_ut_velocity=SPEED)
        CHANNEL_MODEL.set_topology(*topology)
        y_rg, h_freq = self.channel((x_rg, no))

        ###################################
        # Receiver
        ###################################

        # Channel estimation
        if self.perf_csi:
            h_hat = h_freq
            err_var = 0.0
        else:
            h_hat,err_var = self.channel_estimator((y_rg,no))

        # Detection
        if self.det_method == "mmse-pic":
            if self._output == "bit":
                prior_shape = bits_shape
            elif self._output == "symbol":
                prior_shape = tf.concat([tf.shape(x), [self.num_bits_per_symbol]], axis=0)
            prior = tf.zeros(prior_shape)
            det_out = self.detector((y_rg,h_hat,prior,err_var,no))
        else:
            det_out = self.detector((y_rg,h_hat,err_var,no))

        # (Decoding) and output
        if self._output == "bit":
            llr = tf.reshape(det_out, bits_shape)
            b_hat = self.decoder(llr)
            return b, b_hat
        elif self._output == "symbol":
            x_hat = tf.reshape(det_out, tf.shape(x_ind))
            return x_ind, x_hat
        
def run_sim(num_tx, num_bits_per_symbol, output, ebno_dbs, perf_csi, det_param=None):

    lmmse = MIMOOFDMLink(output, "lmmse", perf_csi, num_tx, num_bits_per_symbol, det_param)
    k_best = MIMOOFDMLink(output, "k-best", perf_csi, num_tx, num_bits_per_symbol, det_param)
    ep = MIMOOFDMLink(output, "ep", perf_csi, num_tx, num_bits_per_symbol, det_param)
    mmse_pic = MIMOOFDMLink(output, "mmse-pic", perf_csi, num_tx, num_bits_per_symbol, det_param)

    if output == "symbol":
        soft_estimates = False
        ylabel = "Uncoded SER"
    else:
        soft_estimates = True
        ylabel = "Coded BER"

    er_lmmse,_ = sim_ber(lmmse,
        ebno_dbs,
        batch_size=64,
        max_mc_iter=200,
        num_target_block_errors=200,
        soft_estimates=soft_estimates);

    er_ep,_ = sim_ber(ep,
        ebno_dbs,
        batch_size=64,
        max_mc_iter=200,
        num_target_block_errors=200,
        soft_estimates=soft_estimates);

    er_kbest,_ = sim_ber(k_best,
       ebno_dbs,
       batch_size=64,
       max_mc_iter=200,
       num_target_block_errors=200,
       soft_estimates=soft_estimates);

    er_mmse_pic,_ = sim_ber(mmse_pic,
       ebno_dbs,
       batch_size=64,
       max_mc_iter=200,
       num_target_block_errors=200,
       soft_estimates=soft_estimates);

    return er_lmmse, er_ep, er_kbest, er_mmse_pic

# Range of SNR (dB)
EBN0_DBs = np.linspace(-10., 20.0, 10)

# Number of transmitters
NUM_TX = 4

# Modulation order (number of bits per symbol)
NUM_BITS_PER_SYMBOL = 4 # 16-QAM

SER = {} # Store the results

# Perfect CSI
ser_lmmse, ser_ep, ser_kbest, ser_mmse_pic = run_sim(NUM_TX, NUM_BITS_PER_SYMBOL, "symbol", EBN0_DBs, True)
SER['Perf. CSI / LMMSE'] = ser_lmmse
SER['Perf. CSI / EP'] = ser_ep
SER['Perf. CSI / K-Best'] = ser_kbest
SER['Perf. CSI / MMSE-PIC'] = ser_mmse_pic

# Imperfect CSI
ser_lmmse, ser_ep, ser_kbest, ser_mmse_pic = run_sim(NUM_TX, NUM_BITS_PER_SYMBOL, "symbol", EBN0_DBs, False)
SER['Ch. Est. / LMMSE'] = ser_lmmse
SER['Ch. Est. / EP'] = ser_ep
SER['Ch. Est. / K-Best'] = ser_kbest
SER['Ch. Est. / MMSE-PIC'] = ser_mmse_pic

BER = {} # Store the results

# Perfect CSI
ber_lmmse, ber_ep, ber_kbest, ber_mmse_pic = run_sim(NUM_TX, NUM_BITS_PER_SYMBOL, "bit", EBN0_DBs, True)
BER['Perf. CSI / LMMSE'] = ber_lmmse
BER['Perf. CSI / EP'] = ber_ep
BER['Perf. CSI / K-Best'] = ber_kbest
BER['Perf. CSI / MMSE-PIC'] = ber_mmse_pic

# Imperfect CSI
ber_lmmse, ber_ep, ber_kbest, ber_mmse_pic = run_sim(NUM_TX, NUM_BITS_PER_SYMBOL, "bit", EBN0_DBs, False)
BER['Ch. Est. / LMMSE'] = ber_lmmse
BER['Ch. Est. / EP'] = ber_ep
BER['Ch. Est. / K-Best'] = ber_kbest
BER['Ch. Est. / MMSE-PIC'] = ber_mmse_pic

fig, ax = plt.subplots(1,2, figsize=(16,7))
fig.suptitle(f"{NUM_TX}x{NUM_RX_ANT} UMi | {2**NUM_BITS_PER_SYMBOL}-QAM")

## SER

ax[0].set_title("Symbol error rate")
# Perfect CSI
ax[0].semilogy(EBN0_DBs, SER['Perf. CSI / LMMSE'], 'x-', label='Perf. CSI / LMMSE', c='C0')
ax[0].semilogy(EBN0_DBs, SER['Perf. CSI / EP'], 'o--', label='Perf. CSI / EP', c='C0')
ax[0].semilogy(EBN0_DBs, SER['Perf. CSI / K-Best'], 's-.', label='Perf. CSI / K-Best', c='C0')
ax[0].semilogy(EBN0_DBs, SER['Perf. CSI / MMSE-PIC'], 'd:', label='Perf. CSI / MMSE-PIC', c='C0')

# Imperfect CSI
ax[0].semilogy(EBN0_DBs, SER['Ch. Est. / LMMSE'], 'x-', label='Ch. Est. / LMMSE', c='C1')
ax[0].semilogy(EBN0_DBs, SER['Ch. Est. / EP'], 'o--', label='Ch. Est. / EP', c='C1')
ax[0].semilogy(EBN0_DBs, SER['Ch. Est. / K-Best'], 's-.', label='Ch. Est. / K-Best', c='C1')
ax[0].semilogy(EBN0_DBs, SER['Ch. Est. / MMSE-PIC'], 'd:', label='Ch. Est. / MMSE-PIC', c='C1')

ax[0].set_xlabel(r"$E_b/N0$")
ax[0].set_ylabel("SER")
ax[0].set_ylim((1e-4, 1.0))
ax[0].legend()
ax[0].grid(True)

## SER

ax[1].set_title("Bit error rate")
# Perfect CSI
ax[1].semilogy(EBN0_DBs, BER['Perf. CSI / LMMSE'], 'x-', label='Perf. CSI / LMMSE', c='C0')
ax[1].semilogy(EBN0_DBs, BER['Perf. CSI / EP'], 'o--', label='Perf. CSI / EP', c='C0')
ax[1].semilogy(EBN0_DBs, BER['Perf. CSI / K-Best'], 's-.', label='Perf. CSI / K-Best', c='C0')
ax[1].semilogy(EBN0_DBs, BER['Perf. CSI / MMSE-PIC'], 'd:', label='Perf. CSI / MMSE-PIC', c='C0')

# Imperfect CSI
ax[1].semilogy(EBN0_DBs, BER['Ch. Est. / LMMSE'], 'x-', label='Ch. Est. / LMMSE', c='C1')
ax[1].semilogy(EBN0_DBs, BER['Ch. Est. / EP'], 'o--', label='Ch. Est. / EP', c='C1')
ax[1].semilogy(EBN0_DBs, BER['Ch. Est. / K-Best'], 's-.', label='Ch. Est. / K-Best', c='C1')
ax[1].semilogy(EBN0_DBs, BER['Ch. Est. / MMSE-PIC'], 'd:', label='Ch. Est. / MMSE-PIC', c='C1')

ax[1].set_xlabel(r"$E_b/N0$")
ax[1].set_ylabel("BER")
ax[1].set_ylim((1e-4, 1.0))
ax[1].legend()
ax[1].grid(True)