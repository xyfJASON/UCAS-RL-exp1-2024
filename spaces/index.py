from spaces.base import BaseSpace


class IndexSpace(BaseSpace):
    def __init__(self, size: int):
        self._size = size

    @property
    def size(self):
        return self._size

    def check_valid(self, element: int):
        if not 0 <= element < self.size:
            raise ValueError(f'Element {element} is out of bounds [0, {self.size}).')
