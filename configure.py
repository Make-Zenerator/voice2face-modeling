# model edit
cfg_model = {'spectrogram': [64, 64, 128, 'M', 128, 'M', 128, 'M', 256 'M', 512, 512, 512]}

class Configure():
    def make_configure(cfg_type="spectrogram"):
        cfg = Configure.set_cfg(cfg_type)
        return cfg

    def set_cfg(cfg_type):
        return cfg_model[cfg_type]