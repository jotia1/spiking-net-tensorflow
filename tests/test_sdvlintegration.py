import delayedmodels as dm
import numpy as np
import tensorflow as tf

class TestSDVLIntegration():
    def test_fig2_behaviour(self):
        """ Recreate the dynamics for fig 2 and verify simulation works.

        TODO: 
            - Set network weights to 1
            - Set up network delays
            - Set up network variances
            - Set up supervisory signal?
            - Check what variables will keep track of the changes 
                which will eventually be used in the comparison.
        """
        
        # TODO : This test is not practical whilst the simulator is so slow
        assert False

        tf.random.set_seed(1)
        np.random.seed(0)
        dt = 1.0
        num_sec = 1

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
        # Run network
        debug = []
        sim_time = num_sec * 1000 # 300 sec as ms
        num_steps = int(sim_time / dt)
        for step in range(num_steps):
            #print('TIME: ', step)
            time = step * dt

            spike_idxs = inp_spikes[1, inp_spikes[0, :] == time]
            #print(spike_idxs.shape, spike_idxs)
            step_spikes = np.zeros((3, 1))
            step_spikes[spike_idxs] = 1

            fired = model(step_spikes).numpy()

            idxs = np.array(np.nonzero(np.squeeze(fired))).transpose()  ## there has got to be a neater way to do this...
            ts = np.ones(idxs.shape) * time
            if idxs.size > 0:
                spike_times = np.concatenate((spike_times, np.concatenate((ts, idxs), axis=1)))
            
            debug.append(model.v.numpy())
        data = np.array(tf.squeeze(debug))

        # Measure results
        