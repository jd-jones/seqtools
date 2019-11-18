import math
class ParamDict(dict):    # inherits from built-in dict
    """
    A dictionary used to map parameter names to their numeric values.
    The names are arbitrary objects.  The default value for a parameter
    that has never been updated is normally 0 and is given by the `None` key.

    These dictionaries may be treated like sparse numeric vectors --
    they may be added, subtracted, multiplied by real scalars, etc.
    Also, adding a real number to a dictionary adds it to all values
    (including the default), by analogy with numpy vectors.
    """

    def __init__(self, contents=(), default=None, **kwargs):
        super().__init__(contents, **kwargs)
        if default:    # explicitly specified default: override anything copied from contents
            self[None] = default
        elif None not in self: # no explicitly specified default
            self[None] = 0     # set to 0 since contents didn't specify one either

    def __getitem__(self, key):  # supports self[key]
        try:
            return super().__getitem__(key)
        except KeyError:
            return super().__getitem__(None)  # default

    def copy(self):
        return self.__class__(self)

    def _clean(self):
        """
        Put this ParamDict into canonical form by (destructively) removing any
        redundant entries that match the default.
        """
        default = self[None]
        kk = [k for k,v in self.items() if v==default and k is not None]   # keys of redundant entries
        for k in kk:
            del self[k]

    def __eq__(self, other):      # supports self==other
        self._clean()
        if isinstance(other,ParamDict):
            other._clean()
            return super().__eq__(other)
        else:   # for tests like self==0
            return self[None]==other and len(self)==1   # only contains the default

    # See https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

    def __neg__(self):            # supports -self
        result = self.__class__()
        for k,v in self.items():
            result[k] = -v
        return result

    def __iadd__(self, other):   # supports self += other  (destructive)
        if isinstance(other,dict):
            for k,v in other.items():
                if k not in self: self[k] = self[None]
                self[k] += v
        else:
            for k in self:
                self[k] += other
        return self

    def __isub__(self, other):   # supports self -= other  (destructive)
        if isinstance(other,dict):
            for k,v in other.items():
                if k not in self: self[k] = self[None]
                self[k] -= v
        else:
            for k in self:
                self[k] -= other
        return self

    def __imul__(self, other):   # supports self *= other  (destructive)
        if isinstance(other,dict):
            for k,v in other.items():
                if k not in self: self[k] = self[None]
                self[k] *= v
        else:
            for k in self:
                self[k] *= other
        return self

    def __itruediv__(self, other):   # supports self /= other  (destructive)
        if isinstance(other,dict):
            for k,v in other.items():
                if k not in self: self[k] = self[None]
                self[k] /= v
        else:
            for k in self:
                self[k] /= other
        return self

    def __add__(self, other):  # supports self+other
        result = self.copy()
        result += other
        return result

    def __sub__(self, other):  # supports self-other
        result = self.copy()
        result -= other
        return result

    def __mul__(self, other):  # supports self*other
        result = self.copy()
        result *= other
        return result

    def __truediv__(self, other):  # supports self/other
        result = self.copy()
        result /= other
        return result

    def __radd__(self, other):  # supports other+self (when other is a scalar)
        return self+other       # warning: assumes that + is commutative

    def __rsub__(self, other):  # supports other-self (when other is a scalar)
        return -self+other      # warning: assumes that + is commutative

    def __rmul__(self, scalar):       # supports scalar*self.  Does not assume that * is commutative.
        result = self.__class__()
        for k,v in self.items():
            result[k] = scalar*v
        return result

    def __rtruediv__(self, scalar):
        result = self.__class__()
        for k,v in self.items():
            result[k] = scalar/v
        return result

    def __str__(self):
        self._clean()
        return super().__str__()

    def quantize(self, quantum):   # round keys to multiples of quantum > 0
        result = self.__class__()
        for k,v in self.items():
            if math.isfinite(v):
                result[k] = (v // quantum) * quantum
            else:
                result[k] = v
        return result
