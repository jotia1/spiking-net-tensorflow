import numpy as np
import tensorflow as tf

NEURON_DEFAULT_PARAMS = {
    'fgi':65,
    'tau':20,
    'v_rest':-65.0,
    'v_thresh':-55,
    'v_reset':-70,
    'taupre':20,
    'taupost':20,
    'Apre':0.01,
    'Apost':0.01,
    'gaussian_synapses':True,
    'W_MAX': 10,
    'a1':3,
    'a2':2,
    'b1':5,
    'b2':5,
    'nu':0.03,
    'nv':0.01,
    'VARIANCE_MAX':10,
    'VARIANCE_MIN':0.1,
    }
  

class DelayedLIFNeurons(object):

  def __init__(self, num_neurons, num_inputs, delay_max=20, timestep=0.5, **kwargs):
    self.dt = timestep
    self.num_neurons = num_neurons
    self.num_inputs = num_inputs
    self.set_neuron_attributes(**kwargs) 
    self._steps = 0
    
    self.v = tf.Variable(tf.ones([self.num_neurons]) * self.v_rest)
    self.fired = tf.Variable(tf.zeros(self.v.shape, dtype=tf.bool))
    
    # Set up synapse variables
    self.g = tf.Variable(np.zeros([self.num_inputs]), dtype=tf.float32)
    self.E = tf.Variable(np.zeros(self.g.shape), dtype=tf.float32)
    self.w = tf.Variable(1 * tf.ones([self.num_inputs, self.num_neurons]), dtype=tf.float32)
    #self.w.assign(tf.where(tf.random.uniform(self.w.shape) > 0.1, tf.zeros(self.w.shape), self.w))
    
    # Dealys variables
    self.delay_max = delay_max
    self.input_last_spike_times = tf.Variable(tf.ones([self.num_inputs, 1]) * -np.inf)
    self.neurons_last_spike_times = tf.Variable(tf.ones([self.num_neurons]) * -np.inf)
    self.delays = tf.Variable(tf.random.uniform([self.num_inputs, self.num_neurons], 1, self.delay_max))
    self.variance = tf.Variable(tf.random.truncated_normal(self.delays.shape, mean=5.0))
    #self.variance = tf.Variable(tf.random.uniform([self.num_inputs, self.num_neurons], 0.1, 10))
    
    # STDP variables
    #self.ms_to_step = lambda x: int(x / dt)
    self._delay_max_steps = int(delay_max / self.dt)
    # self.active_spikes will tell us when spikes 'arrive' since the gaussians
    #   just have a peak
    self.active_spikes = tf.Variable(tf.zeros([self.num_inputs, self.num_neurons, self._delay_max_steps]))
    self.dApre = tf.Variable(tf.zeros(self.w.shape))
    self.dApost = tf.Variable(tf.zeros(self.w.shape))
    self.STDPdecaypre = tf.constant(tf.exp(-1 / self.taupre))
    self.STDPdecaypost = tf.constant(tf.exp(-1 / self.taupost))

    self.dvar = tf.Variable(tf.zeros(self.variance.shape))
    self.du = tf.Variable(tf.zeros(self.delays.shape))  # (num_inputs, num_neruons)
    
    self.run_debugging = False
    self.debug = []
    self.dw = []
    self.ddapre = []
    self.ddapost = []
    
    self.ddu = []
    self.ddvar = []


  def __call__(self, spikes):
    """ The computation step for the neurons, called every timestep this function 
    updates the membrane voltage and keeps track of when neurons have fired.
    
    Args:
      spikes (array like) - An array of spikes of size [self.num_inputs, 1]
      
    Returns:
      tf.tensor - a logical tenors where True are the neurons that spiked in this
        timestep
    """

    iapp = self.spikes_to_current(spikes)
    
    # Calculate and update the value of the neuron membrane 
    dv = tf.divide(tf.subtract(tf.add(self.v_rest, iapp), self.v), self.tau) * self.dt
    v_updated = tf.add(self.v, dv)

    # Set any that have fired back to reset value then save result back to self.v
    self.fired.assign(tf.greater_equal(v_updated, tf.ones(self.v.shape) * self.v_thresh))
    v_out = tf.where(self.fired, tf.ones(self.v.shape) * self.v_reset, v_updated) 
    self.v.assign(v_out)
    
    # update spike times
    self.input_last_spike_times = tf.where(tf.cast(spikes, tf.bool), 
      tf.ones(self.input_last_spike_times.shape) * self.get_current_time(),
      self.input_last_spike_times)

    self.update_active_spikes(spikes)
    self.apply_stdp(spikes)
    self.apply_sdvl()
    
    self.clear_current_active_spikes()

    self._steps += 1
    return self.fired
  
  def apply_sdvl(self):
    
    #arriving_spikes = self.get_arriving_spikes()
    
    time = self._steps * self.dt
    fired_mask = tf.broadcast_to(self.fired, self.delays.shape)   # (num_inputs, num_neurons)
    t0 = time - self.neurons_last_spike_times   # (num_neurons)
    t0_negu = t0 - self.delays         # (num_inputs, num_neurons)
    k = tf.pow(self.variance + 0.9 ,2)  # (num_inputs, num_neurons)
    
    sgn = tf.sign(t0_negu)             # (num_inputs, num_neurons)
    # t0 >= a2 
    knu = k * self.nu               # (num_fired, num_neurons)
    a2_cond = tf.broadcast_to(tf.greater_equal(t0, self.a2), fired_mask.shape)  # (num_neurons)
    self.du.assign(tf.where(tf.equal(fired_mask, a2_cond), -knu, self.du ))  
    
    # |t0 - u| >= a1
    a1_cond = tf.broadcast_to(tf.greater_equal(t0_negu, self.a1), fired_mask.shape)
    self.du.assign(tf.where(tf.equal(fired_mask, a1_cond), sgn * knu, self.du ))

    self.delays.assign(tf.clip_by_value(tf.add(self.du, self.delays), 1, self.delay_max))
    #self.delays = tf.clip_by_value(self.delays, 1, self.delay_max)

    if self.run_debugging:
      self.ddu.append(self.delays[0, 0])

    knv = k * self.nv
    # | t0 - u | < b2
    b2_cond = tf.broadcast_to(tf.less(t0_negu, self.b2), fired_mask.shape)
    self.dvar.assign(tf.where(tf.equal(fired_mask, b2_cond), -knv, self.dvar))
    # | t0 - u | >= b1
    b1_cond = tf.broadcast_to(tf.greater_equal(t0_negu, self.b1), fired_mask.shape)
    self.dvar.assign(tf.where(tf.equal(fired_mask, b1_cond), knv, self.dvar))
    
    self.variance.assign(tf.clip_by_value(tf.add(self.dvar, self.variance), self.VARIANCE_MIN, self.VARIANCE_MAX))
    #self.variance = tf.clip_by_value(self.variance, self.VARIANCE_MIN, self.VARIANCE_MAX)
    
    if self.run_debugging:
      self.ddvar.append(self.variance[0,0])

    # Reset self.du and self.dvar to zeros
    self.du.assign(tf.zeros(self.delays.shape))
    self.dvar.assign(tf.zeros(self.variance.shape))
      
    # TODO: need to do max and min to constrain boundries.
        
  def clear_current_active_spikes(self):
    """ Remove any spikes that arrived at the current time step
    
    Parameters:
      None
      
    Returns:
      None
    """
    # Fill in any 1's with zeros
    spike_idxs = tf.where(tf.not_equal(self.active_spikes[:, :, self.get_active_spike_idx()], 0) )
    full_idxs = tf.concat([spike_idxs, tf.ones((spike_idxs.shape[0], 1), dtype=tf.int64) * self.get_active_spike_idx()], axis=1)
    self.active_spikes = tf.tensor_scatter_nd_update(self.active_spikes, full_idxs, tf.zeros(full_idxs.shape[0]))
  
  def update_active_spikes(self, spikes):
    """ Given some spikes, add them to active spikes with the appropraite delays
    
    Parameters:
      spikes (array like): The spikes that have just occured
      
    Returns:
      None
    """
    delays_some_hot = spikes * self.delays  # (100, 2)
    idxs = tf.where(tf.not_equal(delays_some_hot, 0))  # Will give indices of delays    (num_spikes * num_neurns, 2) elements are indices into delays_some_hot that are not 0
    just_delays = tf.gather_nd(delays_some_hot, idxs)  # These become the idx's in delay dimension? (after correction)  (num_spikes * num_neurns, 2) elements are delays (floats)
    
    # adjust for variable step size and circular array
    delay_dim_idxs = tf.reshape(self.spike_arrival_step(just_delays), [-1, 1])  # Okay now is the arrival index  (num_spikes * num_neurns, 1) elements are the correction step at which this spike will arrive
    
    full_idxs = tf.concat([idxs, delay_dim_idxs], axis=1)  # add delay indices as a column since they are an index and not more examples

    self.active_spikes = tf.tensor_scatter_nd_update(self.active_spikes, full_idxs, tf.ones(full_idxs.shape[0]))
    
    
  def get_arriving_spikes(self):
    return self.active_spikes[:, :, self.get_active_spike_idx()]
    
  
  def apply_stdp(self, spikes):
    """ Given the values in fired, update weights according to STDP rules
    
    Paramers:
      spikes (array like) - An array of spikes of size [self.num_inputs, 1]
      
    Returns:
      None
    """
    arriving_spikes = self.get_arriving_spikes()
    #if tf.math.count_nonzero(arriving_spikes) > 0:
    #  print(arriving_spikes)
    # pretsynaptic spike occurs, add dApost (which is -ve) to weight
    self.w = tf.where(tf.equal(tf.broadcast_to(arriving_spikes, self.w.shape), 1), 
                      self.w + self.dApost, 
                      self.w)
    # postynaptic spike occurs, add dApre (which is +ve) to weight
    self.w = tf.where(tf.broadcast_to(self.fired, self.w.shape), 
                      self.w + self.dApre, 
                      self.w)
    
    # presynaptic neuron fires, update the activity trace of presynaptics
    self.dApre = tf.where(tf.equal(tf.broadcast_to(arriving_spikes, self.dApre.shape), 1), 
                          self.dApre + self.Apre, 
                          self.dApre)
    # postsynaptic neuron fires, update activity trace of postsynaptics
    self.dApost = tf.where(tf.broadcast_to(self.fired, self.dApost.shape), 
                           self.dApost + self.Apost, 
                           self.dApost)
    
    self.w = tf.clip_by_value(self.w, 0, self.W_MAX)
    
    
    # Decay the trace of pre/post synaptic activity
    self.dApre = self.dApre * self.STDPdecaypre
    self.dApost = self.dApost * self.STDPdecaypost
    
    # Debugging logs for plotting...
    if self.run_debugging:
      self.dw.append(self.w[0,0])
      self.ddapre.append(self.dApre[0, 0])
      self.ddapost.append(self.dApost[0, 0])
    
  def spike_arrival_step(self, delays):
    """ Calculate the spike arrival step index into active spikes for the given
    delay from the current time.
    
    Parameters:
      delays (Tensor) - The delay of this connection
      
    Returns:
      (int) the index into active spikes
    """
    delay_steps = tf.cast(tf.divide(delays, self.dt), dtype=tf.int64)
    idx = tf.mod((self.get_active_spike_idx() + delay_steps), self._delay_max_steps)
    return idx    
    

  def get_active_spike_idx(self):
    """ Get the current index of the arriving spikes from the active_spikes 
    datastructure. 
    
    Used to keep track of when to update learning rules based off spike arrival 
    times (which might be some delay afer the presynaptic neuron spiked).
    
    Returns:
      (int) - current index of which spikes are arriving
    """
    return self._steps % self._delay_max_steps
  
  
  def spikes_to_current(self, spikes):
    """ Given an array of spikes calculate the current (iapp) applied to the 
    layer of neurons.
    
    Params:
      spikes (array like) - An arry of spikes of size [self.num_inputs, 1]
      
     Returns:
      tf.tensor - the current that should be applied.
    """
    iapp = tf.zeros(self.v.shape)
    if self.gaussian_synapses:
      t0 = self.get_current_time() - self.input_last_spike_times
      t0_negu = t0 - self.delays
      p = tf.divide(self.fgi, tf.sqrt(2 * np.pi * self.variance))
      g = tf.multiply(p, tf.exp(- tf.divide(tf.pow(t0_negu, 2), 2 * self.variance)))
      
      gaussian_values = tf.multiply(self.w, g)
      iapp = tf.reduce_sum(gaussian_values, 0)           
    else:
      # Adjust input applied based on what has spiked now and previously
      self.g.assign_add(spikes)
      iapp = tf.subtract(tf.matmul(self.w, tf.multiply(self.g, self.E)), tf.multiply(tf.matmul(self.w, self.g), self.v))
      self.g.assign((1 - self.dt / self.tau) * self.g)
    return iapp
  
  
  def get_current_time(self):
    """ Return how much time (in milliseconds) this neuron has simulated
    
    Returns:
      float - how many milliseconds this neuron has simulated
    """
    return self._steps * self.dt
  
  def set_neuron_attributes(self, **kwargs):
    """ Setup a few variables, removed from __init__ to help keep it clear
    """
    # Variables left in because I don't want to see pylint errors everywhere...
    # Bottom of method will overwrite all these values with defaults.
    self.tau = None
    self.v_rest = None
    self.v_thresh = None
    self.v_reset = None
    self.fgi = None
    self.taupre = None
    self.taupost = None
    self.Apre = None
    self.Apost = None
    self.gaussian_synapses = None
    self.W_MAX = None
    
    # SDVL params
    self.a1 = None
    self.a2 = None
    self.b1 = None
    self.b2 = None
    self.nu = None
    self.nv = None
    self.VARIANCE_MAX = None  # TODO : Why is this defined as a constant?
    self.VARIANCE_MIN = None

    default_params = {
    'fgi':65,
    'tau':20,
    'v_rest':-65.0,
    'v_thresh':-55,
    'v_reset':-70,
    'taupre':20,
    'taupost':20,
    'Apre':0.01,
    'Apost':0.01,
    'gaussian_synapses':True,
    'W_MAX': 10,
    'a1':3,
    'a2':2,
    'b1':5,
    'b2':5,
    'nu':0.03,
    'nv':0.01,
    'VARIANCE_MAX':10,
    'VARIANCE_MIN':0.1,
    }
    default_params.update(kwargs)

    for key, val in default_params.items():
      setattr(self, key, val)
  