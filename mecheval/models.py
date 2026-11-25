"""Reference models for `python -m mecheval run`.

CopyLast predicts that nothing changes. It scores 0 on every mechanic and makes no false edit, and it shows the
smallest model the runner accepts.
"""


class CopyLast:
    """Predicts the current frame."""

    def __init__(self, game, **kwargs):
        self.game = game

    def predict(self, history, action):
        return history.current

    def update(self, history, action, next_frame, valid):
        pass
