from abc import ABCMeta, abstractmethod

class Prompt(metaclass=ABCMeta):
    def __init__(self):
        self.system_prompt = self.get_system_prompt()
        self.user_prompt = self.get_user_prompt()

    @abstractmethod
    def get_system_prompt(self):
        raise NotImplementedError

    @abstractmethod
    def get_user_prompt(self):
        raise NotImplementedError