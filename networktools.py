import numpy as np
import tensorflow as tf
import delayedmodels as dm
from matplotlib import pyplot as plt


def train_model(model, input_data, dt=1.0, metrics=None, sim_time=None, 
    step_callback=None, logging=True):
    """
    Train a model supplied data and return requested metrics

    Returns the trained model, the

    Params:
        model (SNN) - A spiking neural network model
        input_data (np.ndarray) - A (2, x) array of input spikes where (0, :)
            represents the spike times and (1, :) are the input indexes.
        dt (float) - The timestep to use for simulation
        metrics () - TODO : how should this be structured?
        sim_time (int) - The time (in ms) that the simulation should run for
        step_callack (function) - A function to call after each step (can be
            used for logging).

    Returns: 
        (SNN, np.ndarray, list<np.array>) - The final model, a (2,s) array 
            of output spikes in the same form as input_data and requested 
            metrics
    """

    # Sanity checks
    assert input_data.shape[0] == 2
    assert model != None

    # Sort input spikes by spike time
    input_spikes = input_data[:, input_data[0, :].argsort()]

    if input_spikes[0, 0] > 1000:
        print("WARN: more than 1 second until first input data")

    if not step_callback:
        step_callback = lambda _1, _2 : None

    if not sim_time:
        sim_time = input_spikes[0, -1]
    num_steps = int(sim_time / dt)

    output_spikes = []

    step_spikes = np.zeros((model.num_inputs, 1))
    if logging:
        print('Start training model ...')
    current_percentile = 0
    for step in range(num_steps):
        step_time = step * dt

        if logging and int(step * 10 / num_steps) > current_percentile:
            print(f'  Finished {step_time}ms, ~{step * 100 / num_steps :.0f}%')
            current_percentile = int(step * 10 / num_steps)


        spike_idxs = input_spikes[1, input_spikes[0, :] == step_time]
        step_spikes[spike_idxs] = 1

        step_output = model(step_spikes)

        step_callback(model, step_time)

        output_spikes.append((step_time, step_output))
    
        # Reset step_spikes
        step_spikes[spike_idxs] = 0

    return model, output_spikes, []

def output_spikes_to_array(spikes):
    """ Give a list of tuples of times and spikes, convert to a (2,s)
        numpy array of spikes.
    """
    res = np.array([]).reshape([2, 0])

    for time_step, fired in spikes:
        idxs = np.array(np.nonzero(np.squeeze(fired.numpy())))  ## there has got to be a neater way to do this...
        ts = np.ones(idxs.shape) * time_step
        if idxs.size > 0:
            res = np.concatenate((res, np.concatenate((ts, idxs), axis=1)))
    return res

def plot_spikes(spikes, fig=None):
    """ Plot the given spikes
    """
    if not fig:
        fig = plt.gca()
    fig.plot(spikes[0, :], spikes[1, :], '.k')


def test():
    print('Running tests / debugging section')
    tf.random.set_seed(2)
    np.random.seed(0)

    model = dm.DelayedLIFNeurons(2, 3, dt=1)

    voltage_trace = []
    metric_fn = lambda model, _ : voltage_trace.append(model.v.numpy())

    inp = np.array([[10, 10, 11, 12, 12, 50],
        [1, 2, 0, 0, 2, 1]])

    _, out_spikes, _ = train_model(model, inp,step_callback=metric_fn, logging=True)

    # plot voltage
    data = np.array(tf.squeeze(voltage_trace))
    fig, axs = plt.subplots(2, 1, figsize = (12, 12))
    axs[0].plot(data[:, 0])

    res = output_spikes_to_array(out_spikes)
    print(res.shape)
    plot_spikes(res, axs[1])
    #plot_spikes(inp)
    plt.show()


if __name__ == '__main__':
    test()
    print('networktools should not be run as main.')