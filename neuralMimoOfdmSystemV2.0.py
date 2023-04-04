# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 04:10:59 2022
@Title: Simulation of Multiuser MIMO OFDM sytem with neural receiver
@author: voche
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
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import relu

# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna

from sionna.channel.tr38901 import Antenna, AntennaArray, CDL, UMi, UMa, RMa
from sionna.channel import gen_single_sector_topology as gen_topology
from sionna.channel import OFDMChannel

from sionna.mimo import StreamManagement

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, RemoveNulledSubcarriers, ResourceGridDemapper

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder

from sionna.mapping import Mapper, Demapper

from sionna.utils import BinarySource, ebnodb2no, insert_dims, flatten_last_dims, log10, expand_to_rank
from sionna.utils.metrics import compute_ber
from sionna.utils import sim_ber

# We need to enable sionna.config.xla_compat before we can use
# tf.function with jit_compile=True.
# See https://nvlabs.github.io/sionna/api/config.html#sionna.Config.xla_compat
sionna.config.xla_compat=True

#%%
############################################
## Scenario basic specs
num_ut = 1
num_bs = 1
num_ut_ant = 1
num_bs_ant = 1

#UT antenna configuration parameters
ut_num_rows=1
ut_num_cols=1 #int(num_ut_ant/2),
ut_polarization="single" #"dual"
ut_polarization_type="V" #"VH"
ut_antenna_pattern="38.901"

#UT antenna configuration parameters
bs_num_rows=1
bs_num_cols=1 #int(num_ut_ant/2)
bs_polarization="dual"
bs_polarization_type="VH"
bs_antenna_pattern="38.901"

# for channel configuration
delay_spread = 300e-9 #100e-9 # s.
direction = "uplink" # suitable values 'uplink' or 'downlink'

# SNR range for evaluation and training [dB]
ebno_db_min = -5.0
#ebno_db_min = 0.0
ebno_db_max = 10.0

############################################
## OFDM waveform configuration
subcarrier_spacing = 30e3 # Hz
fft_size = 76 #128 # Number of subcarriers forming the resource grid, including the null-subcarrier and the guard bands
num_ofdm_symbols = 14 # Number of OFDM symbols forming the resource grid
dc_null = True # Null the DC subcarrier
num_guard_carriers = [5, 6] # Number of guard carriers on each side
pilot_pattern = "kronecker" # Pilot pattern
pilot_ofdm_symbol_indices = [2, 11] # Index of OFDM symbols carrying pilots
cyclic_prefix_length = 0 # Simulation in frequency domain. This is useless

############################################
## Modulation and coding configuration
num_bits_per_symbol = 2 #6 # QPSK
coderate = 0.5 # Coderate for LDPC code

############################################
## Neural receiver configuration
num_conv_channels = 76 #128 # Number of convolutional channels for the convolutional layers forming the neural receiver

############################################
## Training configuration
num_training_iterations = 15000 #30000 # Number of training iterations
training_batch_size = 76 #128 # Training batch size
model_weights_path = "neural_receiver_weights" # Location to save the neural receiver weights once training is done

############################################
## Evaluation configuration
results_filename = "neural_receiver_results" # Location to save the results
############################################

# The number of transmitted streams is equal to the number of UT antennas
num_streams_per_tx = num_ut_ant

# Create an RX-TX association matrix
# rx_tx_association[i,j]=1 means that receiver i gets at least one stream
# from transmitter j. Depending on the transmission direction (uplink or downlink),
# the role of UT and BS can change. However, as we have only a single
# transmitter and receiver, this does not matter:
rx_tx_association = np.zeros([1, num_ut])
rx_tx_association[0, :] = 1

# Instantiate a StreamManagement object
# This determines which data streams are determined for which receiver.
# In this simple setup, this is fairly simple. However, it can get complicated
# for simulations with many transmitters and receivers.
#stream_manager = StreamManagement(np.array([[1]]),1) # Receiver-transmitter association matrix. One stream per transmitter
stream_manager = StreamManagement(rx_tx_association, num_streams_per_tx) # Receiver-transmitter association matrix. One stream per transmitter

resource_grid = ResourceGrid(num_ofdm_symbols = num_ofdm_symbols,
                             fft_size = fft_size,
                             subcarrier_spacing = subcarrier_spacing,
                             num_tx = num_ut,
                             num_streams_per_tx = num_streams_per_tx,
                             cyclic_prefix_length = cyclic_prefix_length,
                             dc_null = dc_null,
                             pilot_pattern = pilot_pattern,
                             pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices,
                             num_guard_carriers = num_guard_carriers)

# Codeword length. It is calculated from the total number of databits carried by the resource grid, and the number of bits transmitted per resource element
n = int(resource_grid.num_data_symbols*num_bits_per_symbol)
# Number of information bits per codeword
k = int(n*coderate)

#%%
# Helper functions
def getScenarioSpecificChannel(scenario, carrier_frequency, delay_spread, 
                               ut_array, bs_array, direction, speed):
    if scenario == "umi":
        channel_model = UMi(carrier_frequency=carrier_frequency,
                            o2i_model="low",
                            ut_array=ut_array,
                            bs_array=bs_array,
                            direction=direction,
                            enable_pathloss=False,
                            enable_shadow_fading=False)
    elif scenario == "uma":
        channel_model = UMa(carrier_frequency=carrier_frequency,
                            o2i_model="low",
                            ut_array=ut_array,
                            bs_array=bs_array,
                            direction=direction,
                            enable_pathloss=False,
                            enable_shadow_fading=False)
    elif scenario == "rma":
        channel_model = RMa(carrier_frequency=carrier_frequency,
                            ut_array=ut_array,
                            bs_array=bs_array,
                            direction=direction,
                            enable_pathloss=False,
                            enable_shadow_fading=False)
    # A 3GPP channel model is used
    elif scenario == "A" or scenario == "B" or scenario == "C" or scenario == "D"  or scenario == "E":
        channel_model = CDL(model=scenario, 
                            delay_spread=delay_spread, 
                            carrier_frequency=carrier_frequency,
                            ut_array=ut_array, 
                            bs_array=bs_array, 
                            direction=direction, 
                            min_speed=speed)
        
    return channel_model

# set network topology
def getNeworkTopology(batch_size, num_ut, scenario, speed):
    """Set new network topology"""
    topology = gen_topology(batch_size,
                            num_ut,
                            scenario,
                            min_ut_velocity=speed,
                            max_ut_velocity=speed)
    return topology

#%%
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
  
class neuralEnabledMimoOfdmE2ESystem(Model):
    r"""
    Keras model that implements the end-to-end systems.

    As the three considered end-to-end systems (perfect CSI baseline, LS estimation baseline, and neural receiver) share most of
    the link components (transmitter, channel model, outer code...), they are implemented using the same Keras model.

    When instantiating the Keras model, the parameter ``system`` is used to specify the system to setup,
    and the parameter ``training`` is used to specified if the system is instantiated to be trained or to be evaluated.
    The ``training`` parameter is only relevant when the neural

    At each call of this model:
    * A batch of codewords is randomly sampled, modulated, and mapped to resource grids to form the channel inputs
    * A batch of channel realizations is randomly sampled and applied to the channel inputs
    * The receiver is executed on the post-DFT received samples to compute LLRs on the coded bits.
      Which receiver is executed (baseline with perfect CSI knowledge, baseline with LS estimation, or neural receiver) depends
      on the specified ``system`` parameter.
    * If not training, the outer decoder is applied to reconstruct the information bits
    * If training, the BMD rate is estimated over the batch from the LLRs and the transmitted bits

    Parameters
    -----------
    system : str
        Specify the receiver to use. Should be one of 'baseline-perfect-csi', 'baseline-ls-estimation' or 'neural-receiver'

    training : bool
        Set to `True` if the system is instantiated to be trained. Set to `False` otherwise. Defaults to `False`.
        If the system is instantiated to be trained, the outer encoder and decoder are not instantiated as they are not required for training.
        This significantly reduces the computational complexity of training.
        If training, the bit-metric decoding (BMD) rate is computed from the transmitted bits and the LLRs. The BMD rate is known to be
        an achievable information rate for BICM systems, and therefore training of the neural receiver aims at maximizing this rate.

    Input
    ------
    batch_size : int
        Batch size

    no : scalar or [batch_size], tf.float
        Noise variance.
        At training, a different noise variance should be sampled for each batch example.

    Output
    -------
    If ``training`` is set to `True`, then the output is a single scalar, which is an estimation of the BMD rate computed over the batch. It
    should be used as objective for training.
    If ``training`` is set to `False`, the transmitted information bits and their reconstruction on the receiver side are returned to
    compute the block/bit error rate.
    """

    def __init__(self,  
                 system,
                 scenario,
                 speed,
                 carrier_frequency,
                 training=False):
        
        super().__init__()
        
        # provided parameters
        self._system = system # system mode
        self._scenario = scenario # for channel model configuration
        self._speed = speed # s
        self._carrier_frequency = carrier_frequency # Hz
        self._training = training # True or False
        
        #other parameters
        self._direction = direction # values of "uplink" or "downlink"
        self._delay_spread = delay_spread # s
        self._num_ut = num_ut
        self._num_bs = num_bs
        self._num_ut_ant = num_ut_ant
        self._num_bs_ant = num_bs_ant
        
        self._num_tx = self._num_ut
        self._num_bits_per_symbol = num_bits_per_symbol
        self._num_streams_per_tx = num_streams_per_tx
        
        # Configure antenna array
        self._ut_array  = AntennaArray(num_rows=ut_num_rows,
                                num_cols=ut_num_cols,
                                polarization=ut_polarization,
                                polarization_type=ut_polarization_type,
                                antenna_pattern=ut_antenna_pattern,
                                carrier_frequency=self._carrier_frequency)

        self._bs_array = AntennaArray(num_rows=bs_num_rows,
                                num_cols=bs_num_cols,
                                polarization=bs_polarization,
                                polarization_type=bs_polarization_type,
                                antenna_pattern=bs_antenna_pattern,
                                carrier_frequency=self._carrier_frequency)

        ######################################
        ## Transmitter
        self._binary_source = BinarySource()
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not training:
            self._encoder = LDPC5GEncoder(k, n)
        self._mapper = Mapper("qam", num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(resource_grid)

        ######################################
        # Configure the channel model
        
        self._channel_model = getScenarioSpecificChannel(scenario=self._scenario, 
                                                         carrier_frequency=self._carrier_frequency, 
                                                         delay_spread=self._delay_spread, 
                                                         ut_array=self._ut_array, 
                                                         bs_array=self._bs_array, 
                                                         direction=self._direction, 
                                                         speed=self._speed)
        # OFDM Channel
        self._ofdm_channel = OFDMChannel(self._channel_model, 
                                         resource_grid,
                                         add_awgn=True,
                                         normalize_channel=True, 
                                         return_channel=True)

        ######################################
        ## Receiver
        # Three options for the receiver depending on the value of `system`
        if "baseline" in system:
            if system == 'baseline-perfect-csi': # Perfect CSI
                self._removed_null_subc = RemoveNulledSubcarriers(resource_grid)
            elif system == 'baseline-ls-estimation': # LS estimation
                self._ls_est = LSChannelEstimator(resource_grid, interpolation_type="nn")
            # Components required by both baselines
            self._lmmse_equ = LMMSEEqualizer(resource_grid, stream_manager)
            self._demapper = Demapper("app", "qam", num_bits_per_symbol)
        elif system == "neural-receiver": # Neural receiver
            self._neural_receiver = NeuralReceiver()
            self._rg_demapper = ResourceGridDemapper(resource_grid, stream_manager) # Used to extract data-carrying resource elements
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not training:
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
        
    @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):
        
        if self._scenario == "umi" or self._scenario == "uma" or self._scenario == "rma":
            #self.new_topology(batch_size)
            self._topology = getNeworkTopology(batch_size, self._num_ut, self._scenario, self._speed)
            self._channel_model.set_topology(*self._topology)

        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        #no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, resource_grid)
        # Outer coding is only performed if not training
        if self._training:
            c = self._binary_source([batch_size, self._num_tx, self._num_streams_per_tx , n])
        else:
            b = self._binary_source([batch_size, self._num_tx, self._num_streams_per_tx, k])
            c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y,h = self._ofdm_channel([x_rg, no_])

        ######################################
        ## Receiver
        # Three options for the receiver depending on the value of ``system``
        if "baseline" in self._system:
            if self._system == 'baseline-perfect-csi':
                h_hat = self._removed_null_subc(h) # Extract non-null subcarriers
                err_var = 0.0 # No channel estimation error when perfect CSI knowledge is assumed
            elif self._system == 'baseline-ls-estimation':
                h_hat, err_var = self._ls_est([y, no]) # LS channel estimation with nearest-neighbor
            x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no]) # LMMSE equalization
            no_eff_= expand_to_rank(no_eff, tf.rank(x_hat))
            llr = self._demapper([x_hat, no_eff_]) # Demapping
        elif self._system == "neural-receiver":
            # The neural receover computes LLRs from the frequency domain received symbols and N0
            y = tf.squeeze(y, axis=1)
            llr = self._neural_receiver([y, no])
            llr = insert_dims(llr, 2, 1) # Reshape the input to fit what the resource grid demapper is expected
            llr = self._rg_demapper(llr) # Extract data-carrying resource elements. The other LLrs are discarded
            llr = tf.reshape(llr, [batch_size, self._num_tx, self._num_streams_per_tx, n]) # Reshape the LLRs to fit what the outer decoder is expected

        # Outer coding is not needed if the information rate is returned
        if self._training:
            # Compute and return BMD rate (in bit), which is known to be an achievable
            # information rate for BICM systems.
            # Training aims at maximizing the BMD rate
            bce = tf.nn.sigmoid_cross_entropy_with_logits(c, llr)
            bce = tf.reduce_mean(bce)
            rate = tf.constant(1.0, tf.float32) - bce/tf.math.log(2.)
            return rate
        else:
            # Outer decoding
            b_hat = self._decoder(llr)
            return b,b_hat # Ground truth and reconstructed information bits returned for BER/BLER computation
        
#%%
# Range of SNRs over which the systems are evaluated
ebno_dbs = np.arange(ebno_db_min, # Min SNR for evaluation
                     ebno_db_max, # Max SNR for evaluation
                     0.5) # Step

# STUDIES    
# mobility studies
# Dictionnary storing the evaluation results
BER = {"baseline-perfect-csi": [],
       "baseline-ls-estimation": [],
       "neural-receiver": []
}

BLER = {
    "baseline-perfect-csi": [],
    "baseline-ls-estimation": [],
    "neural-receiver": []        
}


MOBILITY_SIMS = {
    "scenario" : ["umi", "uma", "rma", "A", "B", "C", "D", "E"],
    "speed" : [0.0, 15.0], #[0.0, 15.0, 30.0]
    "carrier_frequency" : [3.5e9, 28e9, 60e9]
}

scenario =  MOBILITY_SIMS["scenario"][0]
c_frequency =  MOBILITY_SIMS["carrier_frequency"][1]

#%% Train system for neural receiver mode

for speed in MOBILITY_SIMS["speed"]:

    # Train neural receiver
    model = neuralEnabledMimoOfdmE2ESystem(system='neural-receiver', 
                                           scenario=scenario,  
                                           speed=speed, 
                                           carrier_frequency=c_frequency, 
                                           training=True)
    
    optimizer = tf.keras.optimizers.Adam()
    
    for i in range(num_training_iterations):
        # Sampling a batch of SNRs
        ebno_db = tf.random.uniform(shape=[], minval=ebno_db_min, maxval=ebno_db_max)
        # Forward pass
        with tf.GradientTape() as tape:
            rate = model(training_batch_size, ebno_db)
            # Tensorflow optimizers only know how to minimize loss function.
            # Therefore, a loss function is defined as the additive inverse of the BMD rate
            loss = -rate
        # Computing and applying gradients
        weights = model.trainable_weights
        grads = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        # Periodically printing the progress
        if i % 100 == 0:
            print('Iteration {}/{}  Rate: {:.4f} bit'.format(i, num_training_iterations, rate.numpy()), end='\r')
    
    # Save the weights in a file
    weights = model.get_weights()
    with open(model_weights_path, 'wb') as f:
        pickle.dump(weights, f)
    
    # deploy/use trained neural receiver    
    model_neural_rx = neuralEnabledMimoOfdmE2ESystem('neural-receiver', 
                                           scenario=scenario,  
                                           speed=speed, 
                                           carrier_frequency=c_frequency)
    
    # Run one inference to build the layers and loading the weights
    model_neural_rx(1, tf.constant(10.0, tf.float32))
    with open(model_weights_path, 'rb') as f:
        weights = pickle.load(f)
    model_neural_rx.set_weights(weights)
    
    # Evaluations
    ber, bler = sim_ber(model_neural_rx, ebno_dbs, batch_size=76, num_target_block_errors=1000, max_mc_iter=100)
    BLER['neural-receiver'].append(list(bler.numpy()))
    BER['neural-receiver'].append(list(ber.numpy()))

#%% BLER Analysis       
for speed in MOBILITY_SIMS["speed"]:    
    
    model_perfect_csi = neuralEnabledMimoOfdmE2ESystem(system='baseline-perfect-csi', 
                                           scenario=scenario,  
                                           speed=speed, 
                                           carrier_frequency=c_frequency)
    
    ber, bler = sim_ber(model_perfect_csi, ebno_dbs, batch_size=76, num_target_block_errors=1000, max_mc_iter=100)
    BLER['baseline-perfect-csi'].append(list(bler.numpy()))
    BER['baseline-perfect-csi'].append(list(ber.numpy()))
    
#%%
for speed in MOBILITY_SIMS["speed"]:     
    model_ls_est = neuralEnabledMimoOfdmE2ESystem('baseline-ls-estimation', 
                                           scenario=scenario,  
                                           speed=speed, 
                                           carrier_frequency=c_frequency)
    
    ber, bler = sim_ber(model_ls_est, ebno_dbs, batch_size=76, num_target_block_errors=1000, max_mc_iter=100)
    BLER['baseline-ls-estimation'].append(list(bler.numpy()))
    BER['baseline-ls-estimation'].append(list(ber.numpy()))

#%% Save result to file
with open(results_filename, 'wb') as f:
    pickle.dump((ebno_dbs, BLER, BER), f)
    
#%% 
# BLER Analysis
# Plot BLER graphs
plt.figure(figsize=(10,6))
#plt.title("Different 3GPP {} Models Uplink - Impact of UT mobility".format(scenario));
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")

i=0
legend = []
for speed in MOBILITY_SIMS["speed"]:
    # Baseline - Perfect CSI
    plt.semilogy(ebno_dbs, BLER['baseline-perfect-csi'][i], 'o-', label="Perfect CSI - {}[m/s]".format(speed))
    # Baseline - LS Estimation
    plt.semilogy(ebno_dbs, BLER['baseline-ls-estimation'][i], 'x--', label="LS Estimation - {}[m/s]".format(speed))
    # Neural receiver
    plt.semilogy(ebno_dbs, BLER['neural-receiver'][i], 's-.', label="Neural receiver - {}[m/s]".format(speed))
    
    i+= 1
    
plt.legend()
plt.ylim((1e-4, 1.0))
plt.tight_layout()
plt.show()

#%%
# BER Analysis
# Plot BER graphs
plt.figure(figsize=(10,6))
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BER")
plt.grid(which="both")

i=0
legend = []
for speed in MOBILITY_SIMS["speed"]:
    # Baseline - Perfect CSI
    plt.semilogy(ebno_dbs, BER['baseline-perfect-csi'][i], 'o-', label="Perfect CSI - {}[m/s]".format(speed))
    # Baseline - LS Estimation
    plt.semilogy(ebno_dbs, BER['baseline-ls-estimation'][i], 'x--', label="LS Estimation - {}[m/s]".format(speed))
    # Neural receiver
    plt.semilogy(ebno_dbs, BER['neural-receiver'][i], 's-.', label="Neural receiver - {}[m/s]".format(speed))
    
    i+= 1
    
plt.legend()
plt.ylim((1e-4, 1.0))
plt.tight_layout()
plt.show()