
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from delayedmodels import *



def train_model(model, inp_spikes, n_in, sim_time=None, dt=1):
  """ Train a given model using supplied data
  
  Args:
    model (a callable tf model) - The model to train
    inp_spikes (np.ndarray) - a (2, s) array where s is the number of spikes with first 
      row being spike times and the second row being spike index's
    n_in (int) - Number of inputs to the model
    sim_time (int) - the number of ms' to simulate, if non find max of spikes
    dt (float) - the step size to use
    
  returns:
    data (np.ndarray) - Debugging data
    spike_times (np.ndarray) - a (2, s) matrix as per data input
  """
  
  # Sort input spikes by spike time
  inp_spikes = inp_spikes[:, inp_spikes[0, :].argsort()]
  
  if not sim_time:
    sim_time = inp_spikes[0, -1]
  
  num_steps = int(sim_time / dt)
  debug = []
  spike_times = np.array([]).reshape([0, 2])
  
  for step in range(num_steps):
    time = step * dt

    spike_idxs = inp_spikes[1, inp_spikes[0, :] == time]
    step_spikes = np.zeros((n_in, 1))
    step_spikes[spike_idxs] = 1
    
    fired = model(step_spikes).numpy()
    
    idxs = np.array(np.nonzero(np.squeeze(fired))).transpose()  ## there has got to be a neater way to do this...
    ts = np.ones(idxs.shape) * time
    if idxs.size > 0:
      spike_times = np.concatenate((spike_times, np.concatenate((ts, idxs), axis=1)))
    
    debug.append(model.v.numpy())
    
  data = np.array(tf.squeeze(debug))
  # Only plot the first 5 neurons
  fig, axs = plt.subplots(3, 2, figsize = (12, 12))
  #print(axs)
  for i in range(2):
    #print(sim_time, np.linspace(0, sim_time, int(sim_time//dt)).shape, data[:, i].shape)
    axs[0, 0].plot(np.linspace(0, sim_time, int(sim_time/dt)), data[:, i], label=f'neuron #{i}')
  axs[0, 0].legend()
  axs[0, 0].axhline(y=-55.0)
  axs[0,0].axvline(x=30)
  axs[0, 1].plot(spike_times[:, 0], spike_times[:, 1], '.k')
  
  # Plot STDP params
  axs[1, 0].plot(model.dw, label='w')
  axs[1, 0].legend()
  axs[2, 0].plot([x for x in model.ddapre], label='dApre')
  axs[2, 0].plot([x for x in model.ddapost], label='dApost')
  axs[2, 0].legend()
  axs[2, 0].axvline(x=30)
  
  # Plots SDVL variables
  axs[1, 1].plot([x for x in model.ddu], label='Mean of input 0')
  axs[2, 1].plot([x for x in model.ddvar], label='Variance of input 0')
  axs[1, 1].legend()
  axs[2, 1].legend()
  axs[1, 1].axvline(x=30)
  axs[2, 1].axvline(x=30)
  plt.show()
  
  return data#, spike_times

def main():
    dt = 1
    n_in = 100
    N = 2
    # Lets get something deterministic for testing
    tf.random.set_seed(1)
    np.random.seed(0)
    model = DelayedLIFNeurons(N, n_in, timestep=dt)

    num_inp_spikes = 100
    #inp_data = np.array([np.random.randint(1, 99, [num_inp_spikes]), 
    #                    np.random.randint(0, 3,[num_inp_spikes])])

    inp_data = np.array([np.concatenate((np.ones(10, dtype=int) * 30, np.ones(2, dtype=int) * 60)),
                        np.array(list(range(10)) + [0, 1], dtype=int)])
    #print(inp_data.shape, inp_data)

    out_data = train_model(model, inp_data, n_in, 100)   
    print(f'Input 0 fired with a delay of {int(model.delays[0,0])} ms to the hidden unit 0')


if __name__ == '__main__':
    main()