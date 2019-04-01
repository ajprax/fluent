import importlib
import inspect


_enabled_experimental_features = {
    "FloatRange": False,
}


def enable_experimental_feature(feature):
    assert feature in _enabled_experimental_features, f"no such experimental feature: {feature}"
    _enabled_experimental_features[feature] = True


def experimental_feature_enabled(feature):
    assert feature in _enabled_experimental_features, f"no such experimental feature: {feature}"
    return _enabled_experimental_features[feature]


class Underscore:
    """
    Syntactic sugar for anonymous functions.
    Supports various binary operators (e.g. +, -, *, / and their "right-handed" versions) as well as calling methods,
    accessing properties, and indexing

    Example:
         lambda foo: foo[0].bar().baz + 5
         _[0].bar().baz + 5
    This is considerably slower than builtin lambdas, but you can make it comparably fast by ending with ._f
    Example:
        lambda foo: foo.bar()
        _.bar()._f

    NOTE: Unless using ._f, only works with higher order functions from the fluent module. This is because the code has
    to be able to distinguish between calls made to extend the chain of operations inside the lambda and calls made to
    evaluate the lambda and it does so by checking the module of the caller.
    """
    def __init__(self, ops=()):
        self.ops = ops

    def __getattr__(self, attr):
        return Underscore(self.ops + (lambda s: getattr(s, attr),))

    def __getitem__(self, item):
        return Underscore(self.ops + (lambda s: s[item],))

    def __add__(self, other):
        return Underscore(self.ops + (lambda s: s + other,))

    def __sub__(self, other):
        return Underscore(self.ops + (lambda s: s - other,))

    def __mul__(self, other):
        return Underscore(self.ops + (lambda s: s * other,))

    def __matmul__(self, other):
        return Underscore(self.ops + (lambda s: s @ other,))

    def __truediv__(self, other):
        return Underscore(self.ops + (lambda s: s / other,))

    def __floordiv__(self, other):
        return Underscore(self.ops + (lambda s: s // other,))

    def __mod__(self, other):
        return Underscore(self.ops + (lambda s: s % other,))

    def __divmod__(self, other):
        return Underscore(self.ops + (lambda s: divmod(s, other),))

    def __pow__(self, other):
        return Underscore(self.ops + (lambda s: pow(s, other),))

    def __lshift__(self, other):
        return Underscore(self.ops + (lambda s: s << other,))

    def __rshift__(self, other):
        return Underscore(self.ops + (lambda s: s >> other,))

    def __and__(self, other):
        return Underscore(self.ops + (lambda s: s & other,))

    def __xor__(self, other):
        return Underscore(self.ops + (lambda s: s ^ other,))

    def __or__(self, other):
        return Underscore(self.ops + (lambda s: s | other,))

    def __radd__(self, other):
        return Underscore(self.ops + (lambda s: other + s,))

    def __rsub__(self, other):
        return Underscore(self.ops + (lambda s: other - s,))

    def __rmul__(self, other):
        return Underscore(self.ops + (lambda s: other * s,))

    def __rmatmul__(self, other):
        return Underscore(self.ops + (lambda s: other @ s,))

    def __rtruediv__(self, other):
        return Underscore(self.ops + (lambda s: other + s,))

    def __rfloordiv__(self, other):
        return Underscore(self.ops + (lambda s: other // s,))

    def __rmod__(self, other):
        return Underscore(self.ops + (lambda s: other % s,))

    def __rdivmod__(self, other):
        return Underscore(self.ops + (lambda s: divmod(other, s),))

    def __rpow__(self, other):
        return Underscore(self.ops + (lambda s: pow(other, s),))

    def __rlshift__(self, other):
        return Underscore(self.ops + (lambda s: other << s,))

    def __rrshift__(self, other):
        return Underscore(self.ops + (lambda s: other >> s,))

    def __rand__(self, other):
        return Underscore(self.ops + (lambda s: other & s,))

    def __rxor__(self, other):
        return Underscore(self.ops + (lambda s: other ^ s,))

    def __ror__(self, other):
        return Underscore(self.ops + (lambda s: other | s,))

    def __neg__(self):
        return Underscore(self.ops + (lambda s: -s,))

    def __pos__(self):
        return Underscore(self.ops + (lambda s: +s,))

    def __invert__(self):
        return Underscore(self.ops + (lambda s: ~s,))

    def __call__(self, *a, **kw):
        if inspect.getmodule(inspect.currentframe().f_back) is importlib.import_module("fluent"):
            return self._f(a[0])
        else:
            return Underscore(self.ops + (lambda s: s(*a, **kw),))

    def _f(self, a):
        for op in self.ops:
            a = op(a)
        return a


_ = Underscore()


class FloatRange:
    """
    Similar to builtin range, but allows values to be floats. Because of precision limitations, floats behave much less
    predictably than integers. Take care when using this class that you understand the behavior of adding and
    multiplying floats.
    """
    def __init__(self, *a):
        if len(a) == 1:
            self.start = 0
            self.stop, = a
            self.step = 1
        elif len(a) == 2:
            self.start, self.stop = a
            self.step = 1
        elif len(a) == 3:
            self.start, self.stop, self.step = a

        # TODO consider cases where we're within some epsilon of self.stop
        if self.step > 0:
            self._past_end = lambda n: n >= self.stop
        else:
            self._past_end = lambda n: n <= self.stop

    def __contains__(self, item):
        return (self.stop - self.start) % self.step == 0  # TODO: precision?

    def __getitem__(self, item):
        if isinstance(item, int):
            if item >= 0:
                return self.start + self.step * item
            else:
                return self.start + self.step * (len(self) + item)

        if item.start is not None:
            if item.start >= 0:
                start = self.start + item.start * self.step
            else:
                start = self.start + (len(self) + item.start) * self.step
        else:
            start = self.start

        if item.stop is not None:
            if item.stop >= 0:
                stop = self.start + item.stop * self.step
            else:
                stop = self.start + (len(self) + item.stop) * self.step
        else:
            stop = self.stop

        if item.step is not None:
            step = self.step * item.step
        else:
            step = self.step

        return FloatRange(start, stop, step)

    def __iter__(self):
        n = self.start
        while not self._past_end(n):
            yield n
            n += self.step

    def __len__(self):
        return int((self.stop - self.start) / self.step) + 1

    def __reversed__(self):
        return FloatRange(self.start + (len(self) - 1) * self.step, self.start - self.step, -self.step)


