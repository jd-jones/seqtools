# TODO: add formal unit tests
# TODO: coercions from float to (float,0,0) for convenience?  This can be done
# in methods like __add__ and __radd__.

import math
import numpy as np
from mfst import AbstractSemiringWeight
from .paramdict import ParamDict


class ExpectationSemiringWeight(AbstractSemiringWeight):
    """ A weight from the expectation semiring.  Such a weight has the form
    <p,v> where p is a real number and v may be a scalar, vector, matrix, or
    tensor. Methods implement the semiring operations as well as other
    operations.

    Typically p = p1+p2+... >= 0 is a sum of unnormalized probabilities, and
    may be very large or very small (if p1, p2, ... are obtained by
    exponentiating strongly positive or negative values of score/temperature).
    And typically v = p1*v1 + p2*v2 + ... is an unnormalized expectation of
    some value, and is comparably large or small.

    To prevent overflow or underflow, we also include a scaling factor that is
    to be multiplied by the stored p and v numbers.  We store the logarithm of
    this scaling factor, e, which is a signed integer.  Note that integers in
    Python can't overflow.

    Specifically, we store (p,v,e) and take the actual expectation semiring
    weight to be (p*(b**e), v*(b**e)).  By taking b=2**1024, we are in effect
    adding 1024*e to the floating-point exponents that are stored in p and v.
    As a result, we can ensure that the floating-point exponents of p and v
    fall in [-512,511], which allows them to be both represented and safely
    added/multiplied as IEEE 64-bit floating point numbers.

    In practice, we reduce b to 2**512 and only rescale to ensure that the
    exponent of p is in [-256,255].  We do *not* explicitly ensure that the
    exponent of v is also in range, which would be time-consuming and might
    conflict with keeping the exponent of p in range.  But with this more
    aggressive restriction on p, the exponent of v will remain in [-512,511]
    for any "reasonable" value of the normalized expectation v/p (absolute
    value in [2**-256,2**256]).  Thus, we do not have to explicitly ensure that
    v is in range.  In the special case p==0, we do not rescale at all, so the
    exponent of v will be in [-512,511] if v was "reasonable" to begin with.

    (In "unreasonable" cases where v is extreme, computations on v will not
    throw an error, but they may overflow to give inf or underflow to give 0 as
    the v value in the result.  For example, try
    `ExpectationSemiringWeight(1e-240,1)`, or
    `ExpectationSemiringWeight(1e240,1)**2`.  When this happens is sensitive to
    details of the representation.  For example, taking
    `w=ExpectationSemiringWeight(x,1)`, `(w*w)*(w*w)` overflows and `w**4` does
    not for `x==1e-200`, but it's the other way around for `x==1e-100`.)

    An alternative way to prevent overflow/underflow would be the usual
    logarithmic trick: we could store log(p) and log(v).  But then both
    semiring `+` and semiring `*` would require expensive logsumexp operations
    for every element of v.  Moreover, since v can have zero or negative
    elements, we couldn't just store log(v); we would actually have to store
    log(abs(v)) and also sign(v).

    Now, notice that IEEE floating-point representation is a kind of
    generalization of that representation: it stores r as an exponent and
    mantissa, where the exponent is int(log_2(abs(v))) and the mantissa is a
    multiplier such that v = mantissa * 2**exponent.  The mantissa is needed to
    correct for the fact that 2**exponent alone cannot yield v if v <= 0 or v
    is not an exact power of 2.  The previous paragraph allowed non-integer
    exponents, so the only mantissa it needed to store was the sign in
    {-1,0,1}.

    Our `e` trick thus sticks with floating-point representation but simply
    extends the range of the exponent, by using e to store additional
    high-order bits.  This implicitly leverages hardware support for logarithms
    (namely built-in floating-point arithmetic), and it is efficient because we
    share the same e value between p and all elements of v.

    The semiring zero (0,0) is implemented with e=-inf.  This ensures that it
    acts as a true zero, with zero+w == w and zero*w == zero for all w.  (If we
            took e=0 instead, then zero+w where `w._e < 0` would rescale w and
            lose precision.) However, we can represent (0,x) with other values
    of e, and this is useful when x!=0.

    The methods `prob`, `value`, and `expectation` should be used to access p,
    v, and v/p.  These return ordinary mathematical objects unencumbered by e.
    """

    # CLASS ATTRIBUTES

    # should make these read-only
    # limit _p's floating-point exponent to use 9 of the 11 bits allocated by
    # the IEEE 64-bit floating-point format: this leaves one bit for _v > _p,
    # plus one bit to hold the carry when we multiply _p*_v and thus add their
    # exponents
    log_base = 512
    # p,v are given by _p,_v multiplied by base**_e = 2**(log_base*_e)
    base = 2.0**log_base

    # true if we allow _p to have a *signed* integer as exponent
    assert log_base % 2 == 0
    # we'll force abs(_p) to be in [_tiny,_huge)
    _huge = 2.0**(log_base / 2)
    _tiny = 2.0**(-log_base / 2)
    assert _tiny == 1 / _huge
    assert _huge == _tiny * base

    # user can set this temporarily to `True` to change the behavior of `_quantize`
    # in a way that is needed for our modified determinization algorithm.
    aggressive_quantization = False

    # The following constants will be initialized *after* we define the class.
    # Yes, this is the Pythonic way to do it: we can't initialize them here because we can't
    # call the constructor until the class is defined (in contrast to recursive functions).
    # Of course, the user should not change these elements (a safer design would use properties).
    zero = None   # the zero element of the semiring
    one = None    # the one element of the semiring
    nan = None    # a special object used to indicate the result of division by zero

    def __init__(self, prob=1, value=0, exponent=0):
        p = float(prob)
        # p should always be finite for a legal weight (but see nan, isnan())
        assert np.isfinite(p)
        if p != 0:
            # We're working with such a large base that p's exponent will rarely have
            # to wrap around.  But these loops check for that rare case: if we've
            # started using the disallowed high-order bits of the exponent, we
            # move them into e.  The loops should run at most once when the
            # constructor is called internally, since then at most one high-order
            # bit will be set.  If called from outside with a double-precision floating
            # point number, they could potentially run up to 3 times, to clear
            # two high-order bits.  If called with a higher-precision number (e.g.,
            # Decimal), then these loops could potentiallly take longer.  Also note
            # that the loops have some non-obvious overhead: each iteration creates
            # a new array for `value` rather than modifying the old one in place using
            # `/=` or `*=`, to avoid affecting the caller's copy.
            while abs(p) >= self._huge:
                p = p / self.base
                value = value / self.base
                exponent += 1
            while abs(p) < self._tiny:
                p = p * self.base
                value = value * self.base
                exponent -= 1
        self._p = p
        self._v = value   # could contain inf, e.g., ExpectationSemiringWeight(1e-240,1)
        self._e = exponent

    def isnan(self):
        """
        Tests whether we have a nan object.
        (In principle, we could also get such an object from combining nan with
        another weight, but the current constructor has an assertion saying that
        we should never do such a combination.)

        This method only tests `self._p==nan`.  It does *not* test for the possibility
        that `self._v` contains some inf or nan elements, which can arise due to
        numeric overflow when `self._v` is far larger than `self._p`.  Such cases
        are legal semiring weights, just unfortunately not fully known, due to floating-point
        limitations.
        """
        return np.isnan(self._p)

    def member(self):
        """
        Check whether this instance is truly a member of the semiring.
        Required by OpenFST framework.
        """
        return not self.isnan()

    def __add__(self, other):
        if self._e >= other._e:
            # TODO: scaling would be more efficient if we were bit-hacking the floating-point
            # representations in C.  Consider caching the powers of base, or even using
            # math.frexp and math.ldexp to directly increase the exponents by
            # log_base * (other._e - self._e).
            # shows up for zero+zero or zero-zero, and will lead to nan in subtraction below
            if self._e == -math.inf:
                # return quickly; equivalent to continuing with scale = 1
                return self.zero
            scale = self.base**(other._e - self._e)
            p = self._p + other._p * scale
            v = self._v + other._v * scale
            e = self._e
            return self.__class__(p, v, e)
        else:
            return other + self   # swap arguments

    def __neg__(self):
        return self.__class__(-self._p, -self._v, self._e)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        p = self._p * other._p
        v = self._v * other._p + other._v * self._p
        e = self._e + other._e
        return self.__class__(p, v, e)

    def __div__(self, other):
        # the multiplicative inverse is <1/p, -r/p^2>
        # so we are going to multiply ourselves by that weight
        if other._p == 0:
            return self.nan     # not a member of this semiring
        p = self._p / other._p
        v = self._v / other._p - self._p * other._v / (other._p * other._p)
        e = self._e - other._e
        return self.__class__(p, v, e)

    def __pow__(self, pow):
        """
        Arbitrary real-number exponents are allowed.
        However, negative powers of 0 will generate nan; so will fractional
        powers of negative numbers.
        """
        # The "quick" formula for this might overflow or underflow.  In that
        # case, we can fall back on the repeated squaring trick for taking
        # powers, because we have set `base` high enough to support pairwise
        # multiplication.

        # compute new e and remove any fractional part (in case pow is fractional)
        e = self._e * pow
        if np.isfinite(e):
            mult = self.base**(e - int(e))
            e = int(e)
        else:
            # presumably self==zero, with e==-inf
            mult = 1

        try:
            p = mult * self._p**pow
            if type(p) == complex:
                # fractional power of a negative number, but we currently insist that p is real
                return self.nan
            if p == 0 and self._p != 0:    # underflow error
                # hack: just call it an overflow error rather than creating a new error type
                raise OverflowError
            mult *= self._p ** (pow - 1) * pow
            v = mult * self._v
            # underflow error in value (could happen even if p didn't underflow, if pow < 0)
            if mult == 0 and self._p != 0:
                # hack: just call it an overflow error rather than creating a new error type
                raise OverflowError
            return self.__class__(p, v, e)
        except ZeroDivisionError:
            # tried to take p==0 to a negative power, when computing p or when computing v or both
            if pow == 0:
                # Right answer is clearly one.
                # We got here because formula for v includes 0**-1 * 0, which is undefined.
                return self.one
            elif pow > 0 and np.all(self._v == 0):
                # We have something like (0,0)**0.5, which should preserve v of zero
                # (of the appropriate shape). We got here because formula for v
                # includes 0**-0.5 * 0.5, which is undefined but is being
                # multiplied by a zero self._v.
                return self.__class__(p, self._v, e)
            else:
                # We have either something like (0,0)**-1, or something like (0,1)**0.5,
                # Either way it's a division by 0 and we return nan.
                return self.nan
        except OverflowError:
            # Local overflow or underflow due to too large a power.
            # Fall back on the repeated squaring trick.
            # Could we have an infinite regress?  No, because eventually pow should get small
            # enough to avoid an exception.  `*` doesn't throw exceptions; only `**` does,
            # and we should be able to safely take self._p to any power in [-2,2].
            if type(pow) != int:
                return (self * self) ** (pow / 2)
            elif pow % 2 == 0:
                return (self * self) ** (pow // 2)
            elif pow > 0:
                return (self * self) ** ((pow - 1) // 2) * self
            else:
                return (self * self) ** ((pow + 1) // 2) / self

    def __hash__(self):
        return self._e + hash(self._p)

    def __eq__(self, other):
        # Equality testing is pretty easy because our representations are distinctive:
        # a given nonzero `p` is compatible with only one `e` value.
        # However, if `p==0`, then it is possible for the `e` values to mismatch
        # and yet the `v` values to be equal, so we must check for that.
        if self._p != other._p:
            return False   # includes the case of two nans
        if self._e != other._e:
            if self._p != 0:
                return False
            else:
                # We have (0,v,e) and (0,v',e'), which could be equal even though e != e'.
                diff = self - other   # subtraction handles rescaling to make the e values match
                return np.all(diff._v == 0)
                # TODO: this will not accept inf==inf
        # also handles cases where one or both _v are scalar
        return np.all(self._v == other._v)

    def approx_eq(self, other, delta=0.00097652):   # this delta is the same default used by OpenFST
        """
        Approximate equality testing, used to determine whether the forward or backward
        algorithm on a cyclic FST has converged yet, and whether FST minimization should
        treat two states as equivalent.  Required by OpenFST.

        Our test is symmetric and is interested only in relative difference, although
        tiny absolute differences may also vanish because of machine epsilon.

        TODO: The test does require that the signs (1,0,-1) of all components match, which
        might be too strict.  nan elements in the values will also prevent a True result.
        """
        def smaller(a,b):  # local function to test whether a <= (delta/2)*b everywhere
            if isinstance(a,ParamDict):
                # a and b will have the same keys when this is called
                for k,v in a.items():
                    if not abs(v) <= (delta / 2) * abs(b[k]):
                        return False
                return True
            else:
                return np.all(abs(a) <= (delta / 2) * abs(b))

        diff = self - other
        summ = self + other
        # skipping equality case is faster and also avoids nan if both are -inf
        if summ._e != diff._e:
            delta *= self.base**(summ._e - diff._e)   # multiplier on summ values
        return (abs(diff._p) <= (delta / 2) * abs(summ._p)) and smaller(diff._v, summ._v)
        # TODO: this will not accept inf==inf (nor nan==nan, which we might
        #   wish to allow in approx_eq)
        # TODO: When values are high-dim vectors, it would be faster to first compare
        # diff._p and summ._p, and only compute diff._v and summ._v if that succeeds

    def quantize(self, delta=0.00097652):   # this delta is the same default used by OpenFST
        """
        Quantized version of this weight.  Quantized weights are hashed and
        used to help determine whether two states are equal during FST
        determinization.  Coarser quantization results in fewer false negatives
        (i._e., we will detect equal states even if floating-point error makes
        them look different) but also more false positives (i._e., we may merge
        distinct states that are "similar," giving an FST that is slightly
        erroneous but smaller).  Required by OpenFST.

        We'll do relative quantization, so that 2.0000e300 and 2.0001e300 could
        be merged but 2e-300 and 3e-300 could not be merged.  This is
        appropriate when we expect very large or small unnormalized
        probabilities whose relative values are meaningful.

        Specifically, we express p as a power of 2 times a mantissa in [0.5,1),
        and truncate this mantissa to a multiple of delta.  We truncate the
        elements of v similarly, using the *same* power of 2 for all elements
        in v, chosen so that the largest finite mantissa has absolute value in
        [0.5,1).  Much smaller elements in v will thus tend to be rounded to 0.

        The class attribute `aggressive_quantization` can be temporarily set to
        `True` to make this method always quantize v to zero.  In certain
        settings, this does not affect the semantics of determinization but
        results in a much smaller determinized FST.
        """
        if delta < 1e-15:
            # delta=0 specifies "no quantization".  Other very small values (below machine epsilon)
            # wouldn't actually quantize in theory (and would introduce errors in practice,
            # e.g., due to limits of numpy integer range), so treat them as delta=0.
            return self

        def q(x, refval):   # quantize x
            _,exponent = np.frexp(refval)
            quantum = delta * 2.0**exponent
            if isinstance(x, ParamDict):
                return x.quantize(quantum)
            else:
                # astype works because quantum inherited np.float64 type from exponent
                result = (x / quantum).astype(int) * quantum
                # astype(int) does strange things to infinite values, so restore
                # the original unquantized versions of those
                if np.isscalar(x):
                    if not np.isfinite(x):
                        result = x
                else:
                    # would even work in scalar case, but would return a length-1 array
                    result = np.where(np.isfinite(x), result, x)
                return result

        p = q(self._p, refval=self._p)
        if self.aggressive_quantization:
            if np.isscalar(self._v):
                v = 0
            else:
                v = np.zeros_like(self._v)
        elif np.isscalar(self._v):
            v = q(self._v, refval=self._v)
        else:
            if isinstance(self._v, ParamDict):
                # maximum finite absval
                refval = max([abs(val) for val in self._v.values() if np.isfinite(val)])
            else:
                # maximum finite absval
                refval = np.max(np.abs(self._v[np.isfinite(self._v)]))
            v = q(self._v, refval=refval)   # quantize the whole array
        return self.__class__(p, v, self._e)

    def __repr__(self):
        return str(self.__class__.__name__) + str(self)

    def __format__(self, digits):
        """ Makes it possible to write things like f'{w:2}' to print w with 2
        digits after the decimal point.
        """
        return self.__str__(digits=digits)

    def __str__(self, digits=None):
        def scientific(x,e):
            "Return the base-10 mantissa and exponent of x * base**e, for scientific notation."
            if x == 0 or not np.isfinite(x):
                return x,0
            else:
                e10 = e * self.log_base * math.log(2,10)   # rewrite base**e as 10**e10
                # we'll add int(e10) to the exponent; move non-integer part back into x
                x *= 10 ** (e10 - int(e10))
                exponent = math.floor(math.log(abs(x),10))  # scientific notation for x
                mantissa = x / 10 ** exponent
                return mantissa, exponent + int(e10)

        def scistr(mantissa,exponent,digits=None):
            "Format the components of scientific notation as a string."
            if digits:
                m = f'{mantissa:.{digits}f}'
            elif type(mantissa) == int or mantissa.is_integer():
                m = f'{int(mantissa)}'
            else:
                m = f'{mantissa}'
            if exponent == 0:
                return m
            elif exponent > 0:
                return f'{m}e+{exponent}'
            else:
                return f'{m}e{exponent}'

        pstr = scistr(*scientific(self._p, self._e),digits=digits)
        if self._e == 0:
            vstr = str(self._v)
        elif np.isscalar(self._v):
            vstr = scistr(*scientific(self._v, self._e),digits=digits)
        elif self._e == -math.inf:
            # handle this case specially to avoid problems with infinity below
            vstr = str(0 * self._v)
        else:
            # The value isn't just a tensor, so we need to extract a common exponent.
            # We'll make a heuristic aesthetic choice based on the median.
            scale = 1
            if isinstance(self._v, ParamDict):
                # nonzero finite vals
                vals = [val for val in self._v.values() if val != 0 and np.isfinite(val)]
            else:
                # nonzero finite vals
                vals = self._v[np.logical_and(self._v != 0,np.isfinite(self._v))]
            if len(vals) > 0:
                scale = np.median(np.abs(vals))
            mantissa,exponent = scientific(scale, self._e)
            v = mantissa * (self._v / scale)
            # TODO: in future, if isinstance(value, np.ndarray), we will be
            #   able to format it with digits (https://github.com/numpy/numpy/issues/5543)
            vstr = str(v)
            if exponent != 0:
                vstr = f'{scistr(1,exponent)} * {vstr}'   # multiplier like 1e+100 * [...]
        return f'<{pstr}, {vstr}>'

    def sampling_weight(self):
        return self.prob()

    def expectation(self):
        """ Return the normalized expected value v/p, or throw a
        ZeroDivisionError.  The answer is an ordinary value, not in this
        semiring, so there is no `e` field to worry about.
        """
        return self._v / self._p   # ignore self._e, which cancels out.

    def prob(self, normalizer=None, log=False):
        """ Return `p/normalizer.p`.  This is useful in converting computations
        in this semiring back to (normalized) probabilities.  By default, the
        normalizer is 1.  This method should ordinarily be used instead of the
        private `_p` attribute since the result is an ordinary float with no
        `_e` value to worry about.  The result might underflow to 0 or overflow
        to inf.

        If log=True, then log(prob) is returned (base e).  This will generally
        avoid underflow or overflow.
        """
        normalizer = normalizer or self.one
        p = self._p / normalizer._p
        e = self._e
        if log:
            return math.log(p) + (self.log_base * math.log(2)) * (e - normalizer._e)
        else:
            while e > normalizer._e:
                p *= self.base
                e -= 1
            while e < normalizer._e:
                p /= self.base
                e += 1
            return p

    def value(self, normalizer=None):
        """ Return `v/normalizer.p`.  By default, the normalizer is 1.  This is
        useful in converting computations in this semiring back to normalized
        vectors.  It should ordinarily be used instead of the `value` attribute
        since the result is an ordinary value with no `e` value to worry about,
        although its components might underflow to 0 or overflow to int or
        -int.
        """
        normalizer = normalizer or self.one
        v = self._v / normalizer._p
        e = self._e
        while e > normalizer._e:
            v *= self.base
            e -= 1
        while e < normalizer._e:
            v /= self.base
            e += 1
        return v

    def dropvalue(self):
        """
        Return a version of `self` in which `v` has been replaced with 0.
        """
        return self.__class__(self._p, 0, self._e)

################################
# Now that the class is defined, we can give it some attributes that specify
# useful constants of the class. These are used within the class definition
# itself.
################################


# see docstring for the class
ExpectationSemiringWeight.zero = ExpectationSemiringWeight(prob=0,exponent=-math.inf)
ExpectationSemiringWeight.one = ExpectationSemiringWeight(prob=1)
# constructor will not allow p=math.nan so we'll fix it below
ExpectationSemiringWeight.nan = ExpectationSemiringWeight(prob=0)
ExpectationSemiringWeight.nan._p = math.nan

################################
# Some basic though incomplete tests.
################################

EWeight = ExpectationSemiringWeight
z = EWeight.zero
o = EWeight.one
v = EWeight(0,1)
n = EWeight.nan
x = EWeight(1e240,1e235)
y = EWeight(10,10.0**np.array(range(1,6)))
p = EWeight(10,ParamDict(a=1e1,f=1e5))  # similar to like y but with a param dict


def test(w, delta=1e-12):    # test various identities
    def small(a):
        if isinstance(a._v, ParamDict):
            return all([abs(val) < 1e-10 for val in a._v.values()])
        else:
            return np.all(abs(a._v) < 1e-10)

    def eq(a, b, delta=delta):
        assert a.approx_eq(b, delta=delta) or small(a) and small(b)
        # if a != b:  warnings.warn(f'Not exactly equal:\t{a}\n\t{b}')

    eq(z + w, w)
    eq(w + z, w)
    eq(o * w, w)
    eq(w * o, w)
    eq(z * w, z)
    eq(w * z, z)
    eq(w + w, w * (o + o))
    eq(w + w, (o + o) * w)
    eq(w - w, z)
    eq(w / w, o)

    if not isinstance(w._v, ParamDict):  # since ParamDict is not compatible with y
        # allow more floating - point tolerance for test(o / x)
        eq((w * y) / y, w, delta=1e-6)
        eq((w / y) * y, w, delta=1e-6)
    eq(w * w * w * w, w ** 4)
    eq(o / (w * w), w ** -2)
    eq(o / (w * w * w * w), w ** -4)
    eq(w, (w ** 100) ** (1 / 100))
    # have to quantize finely compared to approx_eq because of wide dynamic range in x.value
    eq(w, w.quantize(delta=1e-10), delta=1e-6)


test(x)
test(o / x)
test(y)
test(o / y)
test(p)
test(o / p)

a = EWeight(2.0 ** -1024)   # check that .prob() can return the full range
assert a._e != o._e   # different exponents
assert a.prob(o) == 2.0 ** -1024
assert o.prob(a + a) == 2.0 ** 1023
assert o.prob(-a - a) == -2.0 ** 1023
assert a.prob(a) == 1
assert a.prob(o / a) == 0
assert (o / a).prob(a) == math.inf
