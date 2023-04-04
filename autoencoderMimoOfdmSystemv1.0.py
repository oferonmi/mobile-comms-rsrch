"""
Created on Sat Oct 22 18:01:37 2022
@Title: learning based end-to-end MIMO OFDM system using autoencoders
"""

#%%
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0 # Index of the GPU to use
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
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization, Dense
from tensorflow.nn import relu

# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna

from sionna.channel.tr38901 import AntennaArray, CDL, Antenna
from sionna.channel import OFDMChannel

from sionna.mimo import StreamManagement

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, RemoveNulledSubcarriers, ResourceGridDemapper
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder

from sionna.mapping import Mapper, Demapper, Constellation

from sionna.utils import BinarySource, ebnodb2no, insert_dims, flatten_last_dims, log10, expand_to_rank, sim_ber

# We need to enable sionna.config.xla_compat before we can use
# tf.function with jit_compile=True.
# See https://nvlabs.github.io/sionna/api/config.html#sionna.Config.xla_compat
#sionna.config.xla_compat=True

#%%
###############################################
## Channel configuration
###############################################
carrier_frequency = 28e9 # Hz
delay_spread = 300e-9 # s.
cdl_model = "C" # CDL model to use
direction = "uplink" # suitable values 'uplink' or 'downlink'

###############################################
# SNR range for evaluation and training [dB]
###############################################
ebno_db_min = 0.0
ebno_db_max = 10.0

##############################################
# Antenna configuration
##############################################
# Define the number of UT and BS antennas.
# For the CDL model, that will be used in this notebook, only
# a single UT and BS are supported???.
num_ut = 1 #has to be equal to 1 for CDL channels
num_bs = 1 #has to be equal to 1 for CDL channels
num_ut_ant = 1 #4
num_bs_ant = 1 #8

num_tx = num_ut # for uplink. num_bs otherwise

ut_array = AntennaArray(num_rows=1,
                        num_cols=1, #int(num_ut_ant/2),
                        polarization="single",
                        polarization_type="V",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)

bs_array = AntennaArray(num_rows=1,
                        num_cols=1, #int(num_bs_ant/2),
                        polarization="dual",
                        polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)

##############################################
# Stream managment configuration
##############################################
# The number of transmitted streams is equal to the number of UT antennas
# in both uplink and downlink
num_streams_per_tx = num_ut_ant

# Create an RX-TX association matrix
# rx_tx_association[i,j]=1 means that receiver i gets at least one stream
# from transmitter j. Depending on the transmission direction (uplink or downlink),
# the role of UT and BS can change. However, as we have only a single
# transmitter and receiver, this does not matter:
# for CDL
rx_tx_association = np.array([[1]]) 
# for other 3GPP 38.901 models
#rx_tx_association = np.zeros([1, num_ut])
#rx_tx_association[0, :] = 1


# Instantiate a StreamManagement object
# This determines which data streams are determined for which receiver.
# In this simple setup, this is fairly easy. However, it can get more involved
# for simulations with many transmitters and receivers.
stream_manager = StreamManagement(rx_tx_association, num_streams_per_tx)

##############################################
## OFDM waveform configuration
##############################################
subcarrier_spacing = 30e3 # Hz
fft_size = 76 #128 # Number of subcarriers forming the resource grid, including the null-subcarrier and the guard bands
num_ofdm_symbols = 14 # Number of OFDM symbols forming the resource grid
dc_null = True # Null the DC subcarrier
num_guard_carriers = [5, 6] # Number of guard carriers on each side
pilot_pattern = "kronecker" # Pilot pattern
pilot_ofdm_symbol_indices = [2, 11] # Index of OFDM symbols carrying pilots
cyclic_prefix_length = 6 #0 # Simulation in frequency domain. This is useless

resource_grid = ResourceGrid(num_ofdm_symbols = num_ofdm_symbols,
                             fft_size = fft_size,
                             subcarrier_spacing = subcarrier_spacing,
                             num_tx = num_tx,
                             num_streams_per_tx = num_streams_per_tx,
                             cyclic_prefix_length = cyclic_prefix_length,
                             dc_null = dc_null,
                             pilot_pattern = pilot_pattern,
                             pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices,
                             num_guard_carriers = num_guard_carriers)

###############################################
# Modulation and coding configuration
###############################################
num_bits_per_symbol = 6 # Baseline is 64-QAM
modulation_order = 2**num_bits_per_symbol
coderate = 0.5 # Coderate for the outer code
#n = 1500 # Codeword length [bit]. Must be a multiple of num_bits_per_symbol
n = int(resource_grid.num_data_symbols*num_bits_per_symbol) # Codeword length. It is calculated from the total number of databits carried by the resource grid, and the number of bits transmitted per resource element
num_symbols_per_codeword = n//num_bits_per_symbol # Number of modulated baseband symbols per codeword
k = int(n*coderate) # Number of information bits per codeword

#############################################
## Neural receiver configuration
##############################################
num_conv_channels = 76 #128 # Number of convolutional channels for the convolutional layers forming the neural receiver


###############################################
# Training configuration
###############################################
num_training_iterations_conventional = 10000 # Number of training iterations for conventional training
# Number of training iterations with RL-based training for the alternating training phase and fine-tuning of the receiver phase
num_training_iterations_rl_alt = 7000
num_training_iterations_rl_finetuning = 3000
training_batch_size = tf.constant(76, tf.int32) #tf.constant(128, tf.int32) # Training batch size
rl_perturbation_var = 0.01 # Variance of the perturbation used for RL-based training of the transmitter
model_weights_path_conventional_training = "ofdm_autoencoder_weights_conventional_training" # Filename to save the autoencoder weights once conventional training is done
model_weights_path_rl_training = "ofdm_autoencoder_weights_rl_training" # Filename to save the autoencoder weights once RL-based training is done

###############################################
# Evaluation configuration
###############################################
results_filename = "ofdm_autoencoder_results" # Location to save the results

#%%
##############################################
# Neural network based receiver model specification
# -substitutes channel estimation, equalization and demapping functions in classical receivers
##############################################

class ResidualBlock(Layer):
    r"""
    This Keras layer implements a convolutional residual block made of two convolutional layers with ReLU activation, layer normalization, and a skip connection.
    The number of convolutional channels of the input must match the number of kernel of the convolutional layers ``num_conv_channel`` for the skip connection to work.

    Input
    ------
    : [batch size, num time samples, num subcarriers, num_conv_channel], tf.float
        Input of the layer

    Output
    -------
    : [batch size, num time samples, num subcarriers, num_conv_channel], tf.float
        Output of the layer
    """

    def build(self, input_shape):

        # Layer normalization is done over the last three dimensions: time, frequency, conv 'channels'
        self._layer_norm_1 = LayerNormalization(axis=(-1, -2, -3))
        self._conv_1 = Conv2D(filters=num_conv_channels,
                              kernel_size=[3,3],
                              padding='same',
                              activation=None)
        # Layer normalization is done over the last three dimensions: time, frequency, conv 'channels'
        self._layer_norm_2 = LayerNormalization(axis=(-1, -2, -3))
        self._conv_2 = Conv2D(filters=num_conv_channels,
                              kernel_size=[3,3],
                              padding='same',
                              activation=None)

    def call(self, inputs):
        z = self._layer_norm_1(inputs)
        z = relu(z)
        z = self._conv_1(z)
        z = self._layer_norm_2(z)
        z = relu(z)
        z = self._conv_2(z) # [batch size, num time samples, num subcarriers, num_channels]
        # Skip connection
        z = z + inputs

        return z

class NeuralReceiver(Layer):
    r"""
    Keras layer implementing a residual convolutional neural receiver.

    This neural receiver is fed with the post-DFT received samples, forming a resource grid of size num_of_symbols x fft_size, and computes LLRs on the transmitted coded bits.
    These LLRs can then be fed to an outer decoder to reconstruct the information bits.

    As the neural receiver is fed with the entire resource grid, including the guard bands and pilots, it also computes LLRs for these resource elements.
    They must be discarded to only keep the LLRs corresponding to the data-carrying resource elements.

    Input
    ------
    y : [batch size, num rx antenna, num ofdm symbols, num subcarriers], tf.complex
        Received post-DFT samples.

    no : [batch size], tf.float32
        Noise variance. At training, a different noise variance value is sampled for each batch example.

    Output
    -------
    : [batch size, num ofdm symbols, num subcarriers, num_bits_per_symbol]
        LLRs on the transmitted bits.
        LLRs computed for resource elements not carrying data (pilots, guard bands...) must be discarded.
    """

    def build(self, input_shape):

        # Input convolution
        self._input_conv = Conv2D(filters=num_conv_channels,
                                  kernel_size=[3,3],
                                  padding='same',
                                  activation=None)
        # Residual blocks
        self._res_block_1 = ResidualBlock()
        self._res_block_2 = ResidualBlock()
        self._res_block_3 = ResidualBlock()
        self._res_block_4 = ResidualBlock()
        # Output conv
        self._output_conv = Conv2D(filters=num_bits_per_symbol,
                                   kernel_size=[3,3],
                                   padding='same',
                                   activation=None)

    def call(self, inputs):
        y, no = inputs

        # Feeding the noise power in log10 scale helps with the performance
        no = log10(no)

        # Stacking the real and imaginary components of the different antennas along the 'channel' dimension
        y = tf.transpose(y, [0, 2, 3, 1]) # Putting antenna dimension last
        no = insert_dims(no, 3, 1)
        no = tf.tile(no, [1, y.shape[1], y.shape[2], 1])
        # z : [batch size, num ofdm symbols, num subcarriers, 2*num rx antenna + 1]
        z = tf.concat([tf.math.real(y),
                       tf.math.imag(y),
                       no], axis=-1)
        # Input conv
        z = self._input_conv(z)
        # Residual blocks
        z = self._res_block_1(z)
        z = self._res_block_2(z)
        z = self._res_block_3(z)
        z = self._res_block_4(z)
        # Output conv
        z = self._output_conv(z)

        return z

#%%
###############################################################
# Supervised learning autoencoder based MIMO OFDM System model
###############################################################
class MimoOfdmE2ESystemConventionalTraining(Model):

    def __init__(self, speed, training=False):
        super().__init__()

        self._training = training

        ################
        ## Transmitter
        ################
        self._binary_source = BinarySource()
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not self._training:
            self._encoder = LDPC5GEncoder(k, n)
        # Trainable constellation
        constellation = Constellation("qam", num_bits_per_symbol, trainable=True)
        self.constellation = constellation
        self._mapper = Mapper(constellation=constellation)
        self._rg_mapper = ResourceGridMapper(resource_grid) # added for OFDM

        ################
        ## Channel
        ################
        #self._channel = AWGN()
        # A 3GPP CDL channel model is used
        cdl = CDL(cdl_model, delay_spread, carrier_frequency,
                  ut_array, bs_array, direction, min_speed=speed)
        self._channel = OFDMChannel(cdl, 
                                    resource_grid, 
                                    add_awgn=True, 
                                    normalize_channel=True, 
                                    return_channel=True)

        ################
        ## Receiver
        ################
        # We use the previously defined neural network for demapping
        #self._demapper = NeuralDemapper()
        self._neural_receiver = NeuralReceiver()
        self._rg_demapper = ResourceGridDemapper(resource_grid, stream_manager) # Used to extract data-carrying resource elements
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not self._training:
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

        #################
        # Loss function
        #################
        if self._training:
            self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            #self._bce = tf.nn.sigmoid_cross_entropy_with_logits()

    #@tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):

        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)
            
        #no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        #no = expand_to_rank(no, 2)
        
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, resource_grid)
        
        ################
        ## Transmitter
        ################
        # Outer coding is only performed if not training
        if self._training:
            #c = self._binary_source([batch_size, n])
            c = self._binary_source([batch_size, num_ut, resource_grid.num_streams_per_tx, n])
        else:
            #b = self._binary_source([batch_size, k])
            b = self._binary_source([batch_size, num_ut, resource_grid.num_streams_per_tx, k])
            c = self._encoder(b)
        # Modulation
        x = self._mapper(c) # x [batch size, num_symbols_per_codeword]
        x_rg = self._rg_mapper(x) # added for OFDM

        ################
        ## Channel
        ################
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y,h = self._channel([x_rg, no_])
        
        #y,h = self._channel([x_rg, no])

        ################
        ## Receiver
        ################
        #llr = self._demapper([y, no])
        #llr = tf.reshape(llr, [batch_size, n])
        # The neural receover computes LLRs from the frequency domain received symbols and N0
        y = tf.squeeze(y, axis=1)
        llr = self._neural_receiver([y, no])
        llr = insert_dims(llr, 2, 1) # Reshape the input to fit what the resource grid demapper is expected
        llr = self._rg_demapper(llr) # Extract data-carrying resource elements. The other LLrs are discarded
        llr = tf.reshape(llr, [batch_size, num_ut, resource_grid.num_streams_per_tx, n]) # Reshape the LLRs to fit what the outer decoder is expected
        
        # If training, outer decoding is not performed and the BCE is returned
        if self._training:
            #loss = self._bce(c, llr)
            #return loss
            bce = self._bce(c, llr)
            bce = tf.reduce_mean(bce)
            rate = tf.constant(1.0, tf.float32) - bce/tf.math.log(2.)
            loss = -rate
            return loss
        else:
            # Outer decoding
            b_hat = self._decoder(llr)
            return b,b_hat # Ground truth and reconstructed information bits returned for BER/BLER computation

#%%        
##############################################################################
# Functions for supervised learning based model training tasks
##############################################################################

# Function to execute training of model
def conventional_training(model):
    # Optimizer used to apply gradients
    optimizer = tf.keras.optimizers.Adam()

    for i in range(num_training_iterations_conventional):
        # Sampling a batch of SNRs
        ebno_db = tf.random.uniform(shape=[training_batch_size], minval=ebno_db_min, maxval=ebno_db_max)
        # Forward pass
        with tf.GradientTape() as tape:
            loss = model(training_batch_size, ebno_db) # The model is assumed to return the BMD rate
        # Computing and applying gradients
        weights = model.trainable_weights
        grads = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        # Printing periodically the progress
        if i % 100 == 0:
            print('Iteration {}/{}  BCE: {:.4f}'.format(i, num_training_iterations_conventional, loss.numpy()), end='\r')

# Function to save weights from trained model
def save_weights(model, model_weights_path):
    weights = model.get_weights()
    with open(model_weights_path, 'wb') as f:
        pickle.dump(weights, f)

# Utility function to load and set weights of a model
def load_weights(model, model_weights_path):
    model(1, tf.constant(10.0, tf.float32))
    with open(model_weights_path, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)

#%%
##########################################################
# Train supervised learning based MIMO OFDM model
##########################################################
# # Fix the seed for reproducible trainings
# tf.random.set_seed(1)
# # Instantiate and train the end-to-end system
# model = MimoOfdmE2ESystemConventionalTraining(speed, training=True)
# conventional_training(model)
# # Save weights
# save_weights(model, model_weights_path_conventional_training)

#%%
#########################################################
# Classical MIMO OFDM System model
#########################################################

class MimoOfdmE2ESystemBaseline(Model):

    def __init__(self, system, speed):
        super().__init__()
        
        self._system = system

        ################
        ## Transmitter
        ################
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(k, n)
        constellation = Constellation("qam", num_bits_per_symbol, trainable=False)
        self.constellation = constellation
        self._mapper = Mapper(constellation=constellation)
        self._rg_mapper = ResourceGridMapper(resource_grid) # added for OFDM

        ################
        ## Channel
        ################
        #self._channel = AWGN()
        # A 3GPP CDL channel model is used
        cdl = CDL(cdl_model, delay_spread, carrier_frequency,
                  ut_array, bs_array, direction, min_speed=speed)
        self._channel = OFDMChannel(cdl, 
                                    resource_grid, 
                                    add_awgn=True, 
                                    normalize_channel=True, 
                                    return_channel=True)

        ################
        ## Receiver
        ################
        if system == 'baseline-perfect-csi': # Perfect CSI
            self._removed_null_subc = RemoveNulledSubcarriers(resource_grid)
        elif system == 'baseline-ls-estimation': # LS estimation
            self._ls_est = LSChannelEstimator(resource_grid, interpolation_type="nn")

        self._lmmse_equ = LMMSEEqualizer(resource_grid, stream_manager)
        self._demapper = Demapper("app", constellation=constellation)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

    #@tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):

        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        #if len(ebno_db.shape) == 0:
        #    ebno_db = tf.fill([batch_size], ebno_db)

        #no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        #no = expand_to_rank(no, 2)
        
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, resource_grid)

        ################
        ## Transmitter
        ################
        #b = self._binary_source([batch_size, k])
        b = self._binary_source([batch_size, num_ut, resource_grid.num_streams_per_tx, self._encoder.k])
        c = self._encoder(b)

        # Modulation
        x = self._mapper(c) # x [batch size, num_symbols_per_codeword]
        x_rg = self._rg_mapper(x)

        ################
        ## Channel
        ################
        #no_ = expand_to_rank(no, tf.rank(x_rg))
        #y,h = self._channel([x_rg, no_])
        
        y,h = self._channel([x_rg, no])


        ################
        ## Receiver
        ################
        #llr = self._demapper([y, no])
        if self._system == 'baseline-perfect-csi':
            h_hat = self._removed_null_subc(h) # Extract non-null subcarriers
            err_var = 0.0 # No channel estimation error when perfect CSI knowledge is assumed
        elif self._system == 'baseline-ls-estimation':
            h_hat, err_var = self._ls_est([y, no]) # LS channel estimation with nearest-neighbor

        x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no]) # LMMSE equalization
        no_eff_= expand_to_rank(no_eff, tf.rank(x_hat))
        llr = self._demapper([x_hat, no_eff_]) # Demapping

        b_hat = self._decoder(llr) # decoding
        return b,b_hat # Ground truth and reconstructed information bits returned for BER/BLER computation
    
#%%
####################################################################
# Performance Evaluation
####################################################################
# Range of SNRs over which the systems are evaluated
ebno_dbs = np.arange(ebno_db_min, # Min SNR for evaluation
                     ebno_db_max, # Max SNR for evaluation
                     0.5) # Step

# Dictionnary storing the evaluation results
BER = {"baseline-perfect-csi": [],
       "baseline-ls-estimation": [],
       "autoencoder-conv": []}

BLER = {"baseline-perfect-csi": [],
       "baseline-ls-estimation": [],
       "autoencoder-conv": []}

MOBILITY_SIMS = {
    "scenario" : ["C"], #["umi", "uma", "rma"],
    "direction" : ["uplink", "downlink"],
    "carrier_frequency" : [3.5e9, 28e9, 60e9],
    "speed" : [0.0, 15.0]
}

speed = MOBILITY_SIMS["speed"][0]

#%%
model_perf_csi = MimoOfdmE2ESystemBaseline('baseline-perfect-csi', speed)

ber, bler = sim_ber(mc_fun=model_perf_csi, 
                  ebno_dbs=ebno_dbs, 
                  batch_size=76, #128, 
                  max_mc_iter=100, 
                  num_target_block_errors=1000)

BLER['baseline-perfect-csi'].append(list(bler.numpy()))
BER['baseline-perfect-csi'].append(list(ber.numpy()))

#%%
model_ls_baseline = MimoOfdmE2ESystemBaseline('baseline-ls-estimation', speed)

ber, bler = sim_ber(mc_fun=model_ls_baseline, 
                  ebno_dbs=ebno_dbs, 
                  batch_size=76, #128, 
                  max_mc_iter=100,
                  num_target_block_errors=1000)

BLER['baseline-ls-estimation'].append(list(bler.numpy()))
BER['baseline-ls-estimation'] .append(list(ber.numpy()))

#%%
# Fix the seed for reproducible trainings
tf.random.set_seed(1)
# Instantiate and train the end-to-end system
model = MimoOfdmE2ESystemConventionalTraining(speed, training=True)
conventional_training(model)
# Save weights
save_weights(model, model_weights_path_conventional_training)

#%%
# use trained model
model_conventional = MimoOfdmE2ESystemConventionalTraining(speed, training=False)

load_weights(model_conventional, model_weights_path_conventional_training)

ber, bler = sim_ber(mc_fun=model_conventional, 
                    ebno_dbs=ebno_dbs, 
                    batch_size=76, #128, 
                    max_mc_iter=100, 
                    num_target_block_errors=1000) 

BLER['autoencoder-conv'].append(list(bler.numpy()))
BER['autoencoder-conv'].append(list(ber.numpy()))

#%% save result to file
with open(results_filename, 'wb') as f:
    pickle.dump((ebno_dbs, BLER, BER), f)
    
#%%
# BLER Analysis
plt.figure(figsize=(10,6))
# Baseline - Perfect CSI
plt.semilogy(ebno_dbs, BLER['baseline-perfect-csi'][0], 'o-', c=f'C0', label=f'Baseline - Perfect CSI')
# Baseline - LS Estimation
plt.semilogy(ebno_dbs, BLER['baseline-ls-estimation'][0], 'x--', c=f'C1', label=f'Baseline - LS Estimation')
# Neural receiver
plt.semilogy(ebno_dbs, BLER['autoencoder-conv'][0], 's-.', c=f'C2', label=f'Autoencoder - Supervised')

plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.ylim((1e-4, 1.0))
plt.legend()
plt.tight_layout()

#%%
# BER Analysis
plt.figure(figsize=(10,6))
# # Baseline - Perfect CSI
plt.semilogy(ebno_dbs, BER['baseline-perfect-csi'][0], 'o-', c=f'C0', label=f'Baseline - Perfect CSI')
# # Baseline - LS Estimation
plt.semilogy(ebno_dbs, BER['baseline-ls-estimation'][0], 'x--', c=f'C1', label=f'Baseline - LS Estimation')
# # Neural receiver
plt.semilogy(ebno_dbs, BER['autoencoder-conv'][0], 's-.', c=f'C2', label=f'Autoencoder - Supervised')

plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BER")
plt.grid(which="both")
plt.ylim((1e-4, 1.0))
plt.legend()
plt.tight_layout()