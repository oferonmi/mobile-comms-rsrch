"""
Created on Mon Oct 17 22:43:09 2022
@Title: Mutliuser MIMO OFDM simualtions
"""
#%%
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
import time
import pickle

# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna

from sionna.mimo import StreamManagement

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers
from sionna.ofdm import LMMSEInterpolator, LinearDetector, KBestDetector, EPDetector, MMSEPICDetector

from sionna.channel.tr38901 import Antenna, AntennaArray, CDL, UMi, UMa, RMa
from sionna.channel import gen_single_sector_topology as gen_topology
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder

from sionna.mapping import Mapper, Demapper

from sionna.utils import BinarySource, ebnodb2no, sim_ber, QAMSource
from sionna.utils.metrics import compute_ber

# We need to enable sionna.config.xla_compat before we can use
# tf.function with jit_compile=True.
# See https://nvlabs.github.io/sionna/api/config.html#sionna.Config.xla_compat
sionna.config.xla_compat=True

#%%

#Parameters
delay_spread = 300e-9 # Nominal delay spread in [s]. Please see the CDL documentation about how to choose this value.
fft_size = 128 #76
subcarrier_spacing = 30e3
num_ofdm_symbols = 14
cyclic_prefix_length = 20
pilot_ofdm_symbol_indices = [2, 11]
num_bs = 1
num_ut = 1 #4
num_bits_per_symbol = 6 #2
coderate = 0.5

# SYSTEM KERAS MODEL FOR BER SIMULATIONS
class MimoOfdmSystemModel(tf.keras.Model):
    """Simulate OFDM MIMO transmissions over a 3GPP 38.901 model.
    """
    def __init__(self, 
                 scenario, 
                 perfect_csi, 
                 direction, 
                 speed, 
                 carrier_frequency, 
                 num_bs_ant, 
                 num_ut_ant):
        
        super().__init__()

        # Provided parameters
        self._scenario = scenario
        self._perfect_csi = perfect_csi
        self._direction = direction
        self._speed = speed
        self._carrier_frequency = carrier_frequency #3.5e9
        
        # Internally set parameters
        self._delay_spread = delay_spread 
        self._fft_size = fft_size
        self._subcarrier_spacing = subcarrier_spacing
        self._num_ofdm_symbols = num_ofdm_symbols
        self._cyclic_prefix_length = cyclic_prefix_length
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices
        self._num_bs = num_bs
        self._num_bs_ant = num_bs_ant #8
        self._num_ut = num_ut # number of active or served UT
        self._num_ut_ant = num_ut_ant #4
        self._num_bits_per_symbol = num_bits_per_symbol
        self._coderate = coderate

        # Create an RX-TX association matrix
        # rx_tx_association[i,j]=1 means that receiver i gets at least one stream
        # from transmitter j. Depending on the transmission direction (uplink or downlink),
        # the role of UT and BS can change.
        #bs_ut_association = np.zeros([1, self._num_ut])
        #bs_ut_association[0, :] = 1
        bs_ut_association = np.zeros([self._num_bs, self._num_ut])
        bs_ut_association[:, :] = 1
        #bs_ut_association = np.array([[1]])
        self._rx_tx_association = bs_ut_association
        
        self._num_tx = self._num_ut # for uplink
        self._num_streams_per_tx = self._num_ut_ant # for uplink


        # Setup an OFDM Resource Grid
        self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                fft_size=self._fft_size,
                                subcarrier_spacing=self._subcarrier_spacing,
                                num_tx=self._num_tx,
                                num_streams_per_tx=self._num_streams_per_tx,
                                cyclic_prefix_length=self._cyclic_prefix_length,
                                pilot_pattern="kronecker",
                                pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)

        # Setup StreamManagement
        self._sm = StreamManagement(self._rx_tx_association, self._num_streams_per_tx)

        # Configure antenna arrays
        self._ut_array = AntennaArray(
                                     num_rows=1,
                                     num_cols=int(self._num_ut_ant/2),
                                     polarization="dual",
                                     polarization_type="cross",
                                     antenna_pattern="38.901",
                                     carrier_frequency=self._carrier_frequency)

        self._bs_array = AntennaArray(num_rows=1,
                                      num_cols=int(self._num_bs_ant/2),
                                      polarization="dual",
                                      polarization_type="cross",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency)

        # Configure the channel model
        if self._scenario == "umi":
            self._channel_model = UMi(carrier_frequency=self._carrier_frequency,
                                      o2i_model="low",
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction=self._direction,
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
        elif self._scenario == "uma":
            self._channel_model = UMa(carrier_frequency=self._carrier_frequency,
                                      o2i_model="low",
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction=self._direction,
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
        elif self._scenario == "rma":
            self._channel_model = RMa(carrier_frequency=self._carrier_frequency,
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction=self._direction,
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
            
        # For 3GPP CDL channel models
        elif self._scenario == "A" or self._scenario == "B" or self._scenario == "C" or self._scenario == "D"  or self._scenario == "E":
            self._channel_model = CDL(model=self._scenario, 
                                      delay_spread=self._delay_spread, 
                                      carrier_frequency=self._carrier_frequency,
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction=self._direction, 
                                      min_speed=self._speed)

        # Instantiate other building blocks
        ##################################
        # Transmitter
        ##################################
        self._binary_source = BinarySource()
        self._qam_source = QAMSource(self._num_bits_per_symbol)

        self._n = int(self._rg.num_data_symbols*self._num_bits_per_symbol) # Number of coded bits
        self._k = int(self._n*self._coderate)                              # Number of information bits
        
        self._encoder = LDPC5GEncoder(self._k, self._n)
        self._decoder = LDPC5GDecoder(self._encoder)
        
        self._mapper = Mapper("pam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)
        
        if self._direction == "downlink":
            self._zf_precoder = ZFPrecoder(self._rg, self._sm, return_effective_channel=True)
        
        ##################################
        # Channel
        ##################################
        #self._ofdm_channel = OFDMChannel(self._channel_model, self._rg, add_awgn=True,
        #                                 normalize_channel=True, return_channel=True)
        self._frequencies = subcarrier_frequencies(self._rg.fft_size, self._rg.subcarrier_spacing)
        self._channel_freq = ApplyOFDMChannel(add_awgn=True)
        
        
        ###################################
        # Receiver
        ###################################
        self._remove_nulled_subcarriers = RemoveNulledSubcarriers(self._rg)
        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = Demapper("app", "pam", self._num_bits_per_symbol)

    def new_topology(self, batch_size):
        """Set new network topology"""
        topology = gen_topology(batch_size,
                                self._num_ut,
                                self._scenario,
                                min_ut_velocity=self._speed,
                                max_ut_velocity=self._speed)
        """Set topology"""
        self._channel_model.set_topology(*topology)
        
        """Visualize topology"""
        #self._channel_model.show_topology()


    #@tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):
        
        if self._scenario == "umi" or self._scenario == "uma" or self._scenario == "rma":
            self.new_topology(batch_size)
            
        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        b = self._binary_source([batch_size, self._num_tx, self._num_streams_per_tx, self._k])
        c = self._encoder(b)
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)
        
        #y, h_freq = self._ofdm_channel([x_rg, no])
        cir = self._channel_model(batch_size, self._rg.num_ofdm_symbols, 1/self._rg.ofdm_symbol_duration)
        h_freq = cir_to_ofdm_channel(self._frequencies, *cir, normalize=True)
        if self._direction == "downlink":
            x_rg, g = self._zf_precoder([x_rg, h_freq])
        
        y = self._channel_freq([x_rg, h_freq, no])
        
        if self._perfect_csi:
            if self._direction == "uplink":
                h_hat = self._remove_nulled_subcarriers(h_freq)
            elif self._direction =="downlink":
                h_hat = g
            err_var = 0.0
        else:
            h_hat, err_var = self._ls_est ([y, no])
            
        x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no])
        
        llr = self._demapper([x_hat, no_eff])
        b_hat = self._decoder(llr)
        return b, b_hat
    

#%%
# STUDIES    
# mobility studies

MOBILITY_SIMS = {
    "ebno_db" : list(np.arange(0, 15, 1.0)), #"ebno_db" : list(np.arange(-5, 15, 1.0))
    "scenario" : ["umi"], #["umi", "uma", "rma", "A", "B", "C", "D", "E"],
    "perfect_csi" : [True, False],
    "direction" : ["uplink", "downlink"],
    "num_bs_antenna" : [16], #[4, 8, 16],
    "num_ut_antenna" : [4], #[8], #[4],
    "carrier_frequency" : [28e9], #[3.5e9, 28e9, 60e9],
    "speed" : [0.0], #[0.0, 15.0],
    "ber" : [],
    "bler" : [],
    "duration" : None
}

speed = MOBILITY_SIMS["speed"][0] # remove if whole speed array used
#c_freq = MOBILITY_SIMS["carrier_frequency"][1]
num_ut_ant = MOBILITY_SIMS["num_ut_antenna"][0]

start = time.time()

for scenario in MOBILITY_SIMS["scenario"]:
    for perfect_csi_state in MOBILITY_SIMS["perfect_csi"]:
        for num_bs_ant in MOBILITY_SIMS["num_bs_antenna"]:
        #for speed in MOBILITY_SIMS["speed"]:
            for c_freq in MOBILITY_SIMS["carrier_frequency"]:
                model = MimoOfdmSystemModel(scenario=scenario,
                                        perfect_csi=perfect_csi_state,
                                        direction=MOBILITY_SIMS["direction"][0],
                                        speed=speed,
                                        carrier_frequency=c_freq, 
                                        num_bs_ant=num_bs_ant, 
                                        num_ut_ant=num_ut_ant)
    
                ber, bler = sim_ber(model,
                                MOBILITY_SIMS["ebno_db"],
                                batch_size=128, #76,
                                max_mc_iter=100,
                                num_target_block_errors=1000)
    
                MOBILITY_SIMS["ber"].append(list(ber.numpy()))
                MOBILITY_SIMS["bler"].append(list(bler.numpy()))

MOBILITY_SIMS["duration"] = time.time() - start

#%% Save result to file
results_filename = "mimo_ofdm_results"
with open(results_filename, 'wb') as f:
    pickle.dump((MOBILITY_SIMS["ebno_db"], MOBILITY_SIMS["bler"], MOBILITY_SIMS["ber"]), f)

#%%
#plt.figure()
plt.figure(figsize=(10,6))
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")

i=0
legend = []
for scenario in MOBILITY_SIMS["scenario"]:
    for perfect_csi_state in MOBILITY_SIMS["perfect_csi"]:
        for num_bs_ant in MOBILITY_SIMS["num_bs_antenna"]:
            #for speed in MOBILITY_SIMS["speed"]:
            for c_freq in MOBILITY_SIMS["carrier_frequency"]:
                if scenario=="umi":
                    t = "UMi"
                elif scenario=="uma":
                    t = "UMa"
                elif scenario=="rma":
                    t = "RMa"
                elif scenario=="A":
                    t = "CDL-A"
                elif scenario=="B":
                    t = "CDL-B"
                elif scenario=="C":
                    t = "CDL-C"
                elif scenario=="D":
                    t = "CDL-D"
                elif scenario=="E":
                    t = "CDL-E"
        
                if perfect_csi_state:
                    plt.semilogy(MOBILITY_SIMS["ebno_db"], MOBILITY_SIMS["bler"][i], 'o-');
                    #s = "{}x{} {}-Perfect CSI at {}[m/s]".format(num_bs_ant, num_ut_ant, t, speed)
                    s = "{}x{}-Perfect CSI".format(num_bs_ant, num_ut_ant)
                    #s = "Perfect CSI at {}[GHz]".format(c_freq/1e9)
                else:
                    plt.semilogy(MOBILITY_SIMS["ebno_db"], MOBILITY_SIMS["bler"][i], 'x--');
                    #s = "{}x{} {}-LS Estimation at {}[m/s]".format(num_bs_ant, num_ut_ant, t, speed)
                    s = "{}x{}-LS Estimation".format(num_bs_ant, num_ut_ant)
                    #s = "LS Estimation at {}[GHz]".format(c_freq/1e9)
                
                legend.append(s)
                
                i += 1
        
plt.legend(legend)
plt.ylim([1e-4, 1])
plt.tight_layout()
plt.show()

#plt.title("Different 3GPP 38.901 Models Multiuser 4x8 MIMO Uplink - Impact of UT mobility ");
#plt.title("3GPP 38.901 ModelsCDL Multi-User 4x8 MIMO Uplink - Impact of Carrier Frequency at speed {}".format(speed));

#%%
# BER analysis
#plt.figure()
plt.figure(figsize=(10,6))
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BER")
plt.grid(which="both")

i=0
legend = []
for scenario in MOBILITY_SIMS["scenario"]:
    for perfect_csi_state in MOBILITY_SIMS["perfect_csi"]:
        for num_bs_ant in MOBILITY_SIMS["num_bs_antenna"]:
            #for speed in MOBILITY_SIMS["speed"]:
            for c_freq in MOBILITY_SIMS["carrier_frequency"]:
                if scenario=="umi":
                    t = "UMi"
                elif scenario=="uma":
                    t = "UMa"
                elif scenario=="rma":
                    t = "RMa"
                elif scenario=="A":
                    t = "CDL-A"
                elif scenario=="B":
                    t = "CDL-B"
                elif scenario=="C":
                    t = "CDL-C"
                elif scenario=="D":
                    t = "CDL-D"
                elif scenario=="E":
                    t = "CDL-E"
        
                if perfect_csi_state:
                    plt.semilogy(MOBILITY_SIMS["ebno_db"], MOBILITY_SIMS["ber"][i], 'o-');
                    #s = "{}x{} {}-Perfect CSI at {}[m/s]".format(num_bs_ant, num_ut_ant, t, speed)
                    s = "{}x{}-Perfect CSI".format(num_bs_ant, num_ut_ant)
                    #s = "Perfect CSI at {}[GHz]".format(c_freq/1e9)
                else:
                    plt.semilogy(MOBILITY_SIMS["ebno_db"], MOBILITY_SIMS["ber"][i], 'x--');
                    #s = "{}x{} {}-LS Estimation at {}[m/s]".format(num_bs_ant, num_ut_ant, t, speed)
                    s = "{}x{}-LS Estimation".format(num_bs_ant, num_ut_ant)
                    #s = "LS Estimation at {}[GHz]".format(c_freq/1e9)
                
                legend.append(s)
                
                i += 1
        
plt.legend(legend)
plt.ylim([1e-4, 1])
plt.tight_layout()
plt.show()

#plt.title("Different 3GPP 38.901 Models Multiuser 4x8 MIMO Uplink - Impact of UT mobility ");
#plt.title("3GPP 38.901 CDL-{} Model Multi-User 4x8 MIMO Uplink - Impact of Carrier Frequencyon BER  at speed {}".format(scenario, speed));