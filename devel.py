import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import delayedmodels as dm

def test_experiment(num_sec=300):
  import time as timer
  tf.random.set_seed(1)
  np.random.seed(0)
  dt = 1.0
  #num_sec = 300

  # Build model
  model = dm.DelayedLIFNeurons(2, 3, delay_max=20, timestep=dt, fgi=13)

  # Prepare data
  freq = 2  # Hz
  patt = np.array([
      [0, 3, 7],
      [0, 1, 2]
  ])
  inp_spikes = np.tile(patt, num_sec * freq) + np.array([(np.arange(num_sec * freq * 3) // 3) * 500, np.zeros(num_sec * freq * 3)])
  inp_spikes = inp_spikes.astype(int)
  time_start_loop = []
  time_call_model = []
  time_model_return = []
  time_end_loop = []

  # Run network
  debug = []
  sim_time = num_sec * 1000 # 300 sec as ms
  num_steps = int(sim_time / dt)
  for step in range(num_steps):
      time_start_loop.append(timer.time())
      #print('TIME: ', step)
      time = step * dt
      #if time % 1000 == 0:
      #print(time)

      spike_idxs = inp_spikes[1, inp_spikes[0, :] == time]
      #print(spike_idxs.shape, spike_idxs)
      step_spikes = np.zeros((3, 1))
      step_spikes[spike_idxs] = 1

      time_call_model.append(timer.time())
      fired = model(step_spikes).numpy()
      time_model_return.append(timer.time())


      idxs = np.array(np.nonzero(np.squeeze(fired))).transpose()  ## there has got to be a neater way to do this...
      ts = np.ones(idxs.shape) * time
      if idxs.size > 0:
          spike_times = np.concatenate((spike_times, np.concatenate((ts, idxs), axis=1)))
      
      debug.append(model.v.numpy())
      time_end_loop.append(timer.time())
  data = np.array(tf.squeeze(debug))

  ## Plot timing info
  # convert all to np arrays
  
  start_loop = np.array(time_start_loop)
  call_model = np.array(time_call_model)
  return_model = np.array(time_model_return)
  end_loop = np.array(time_end_loop)
  
  total_loop_time = end_loop - start_loop
  total_call_time = return_model - call_model

  av_loop_time = np.average(total_loop_time)
  av_call_time = np.average(total_call_time)
  print('-' * 25, 'training loop times', '-'*25)
  print(f'    average time for full loop: {av_loop_time:.4f}')
  print(f'    average call time: {av_call_time:.4f}')
  print(f'    Percent of time spent in call: {av_call_time / av_loop_time :.2f}')
  print(f'    Time for 1 simulated second: {end_loop[-1] - start_loop[0]:.2f} seconds')


  ## Plot call times
  """
  a = np.array(model.a)
  b = np.array(model.b)
  c = np.array(model.c)
  d = np.array(model.d)
  e = np.array(model.e)
  f = np.array(model.f)
  g = np.array(model.g)

  av_first_half = np.average(c - a)
  av_all = np.average(g - a)

  print('-' * 25, 'model times', '-'*25)

  print(f'  First half time: {av_first_half:.4f}, - {av_first_half / av_all :.2f} ')

  av_active = np.average(d - c)
  print(f'  activespikes time: {av_active :.4f}, av: {av_active / av_all :.2f} ')

  av_stdp = np.average(e - d)
  print(f'  STDP time: {av_stdp :.4f}, av: {av_stdp / av_all :.2f} ')

  av_sdvl = np.average(f - e)
  print(f'  SDVL time: {av_stdp :.4f}, av: {av_sdvl / av_all :.2f} ')

  all_above = av_first_half + av_active + av_stdp + av_sdvl
  print(f'  SUM: {all_above / av_all :.2f} % ')


  ## Plot SDVL time
  print('-' * 25, 'SDVL times', '-'*25)
  arrs = [np.array(x) for x in model.sdvl_time]
  avg = lambda x: np.average(x)
  avg_idxs = lambda x, y: avg(arrs[y] - arrs[x])  # Average of time from x to y
  perc = lambda x, y: avg_idxs(x, y) / avg_idxs(0, 7)    # arg should be an index

  print(f'  all delay time:    {avg_idxs(0, 6):.6f}, {perc(0, 6):.2f}')
  for i in range(1, 8):
    print(f'  ({i-1}, {i}):    {avg_idxs(i-1, i):.6f}, {perc(i - 1, i):.2f}')


  

  print(f'  all var time: {avg_idxs(6, 7):.6f}, {perc(6, 7):.2f}')
"""

def plot_intial_implementation():
  sim_secs = [1, 4, 8, 16, 32, 64, 128]
  prealloc_secs =  [5.271489381790161, 20.46659827232361, 41.10378813743591, 107.08386468887329, 190.880713224411, 374.88883996009827, 744.4272258281708]
  real_secs = [8.752479791641235, 35.17718696594238, 73.05648279190063, 145.0080795288086, 285.8125159740448, 1205/2, 1205.7392013072968]
  # Generate linear fit
  orig_fit = np.polyfit(sim_secs, real_secs, 1)
  orig_fit_fn = np.poly1d(orig_fit)
  prea_fit = np.polyfit(sim_secs, prealloc_secs, 1)
  prea_fit_fn = np.poly1d(prea_fit)


  plt.plot(sim_secs, real_secs, 'b', 
    sim_secs, orig_fit_fn(sim_secs), '--r',
    sim_secs, prealloc_secs, 'k',
    sim_secs, prea_fit_fn(sim_secs), '--r')
  plt.xlabel('Simulated seconds')
  plt.ylabel('Real time taken (sec)')
  plt.title(f'With preallocation fixes, Gradients of {orig_fit[0]:.2f} (original) and {prea_fit[0]:.2f} (fixed)')
  plt.legend()
  plt.show()

def generate_speed_plot():
  import time as timer
  timed_results = []

  test_durations = [64]
  #test_durations = [1, 2, 3]
  for sec in test_durations:
    print(f'Sim time: {sec}')
    start = timer.time()
    test_experiment(num_sec=sec)
    timed_results.append(timer.time() - start)

  print(test_durations, timed_results)
  plt.plot(test_durations, timed_results)
  plt.show()
  

def profile_code():
  import cProfile

  cProfile.run('test_experiment(4)')
  

if __name__ == '__main__':
    #main()
    #test_experiment(1)
    #generate_speed_plot()
    #profile_code()
    plot_intial_implementation()