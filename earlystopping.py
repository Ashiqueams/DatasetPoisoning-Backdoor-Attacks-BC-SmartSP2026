class EarlyStopping:
    """
    Early stopping utility that matches code expecting:
      - early_stopping(val_loss)  OR early_stopping.step(val_loss)
      - early_stopping.early_stop (bool)
    """
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = None
        self.counter = 0
        self.early_stop = False

    def step(self, value: float):
        value = float(value)
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def __call__(self, value: float):
        return self.step(value)
