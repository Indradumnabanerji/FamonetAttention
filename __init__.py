import config.config as configs

class Setting:
    def __init__(self, configs):
        for a in dir(configs):
            if a.isupper():
                setattr(self, a, getattr(configs, a))

configs = Setting(configs)
