import delayedmodels as dm

class TestDelayedLIFNeurons():
    def test_STDP(self):
        pass

    def test_SDVL(self):
        pass

    def test_timestep_correctly_set(self):
        model = dm.DelayedLIFNeurons(2, 2, timestep=1.0)
        assert model.dt == 1.0

    def test_fgi_intialised(self):
        model = dm.DelayedLIFNeurons(2, 2, fgi=13)
        assert model.fgi == 13