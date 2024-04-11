from spaces.base import BaseSpace


class ContinuousSpace(BaseSpace):
    def __init__(self, l_end: float, r_end: float, l_included: bool = True, r_included: bool = True):
        self.l_end = l_end
        self.r_end = r_end
        self.l_included = l_included
        self.r_included = r_included

    def check_valid(self, element: float):
        l_cmp = element >= self.l_end if self.l_included else element > self.l_end
        r_cmp = element <= self.r_end if self.r_included else element < self.r_end
        if not (l_cmp and r_cmp):
            l_brace = '[' if self.l_included else '('
            r_brace = ']' if self.r_included else ')'
            raise ValueError(f'Element {element} is not in range {l_brace}{self.l_end}, {self.r_end}{r_brace}')
