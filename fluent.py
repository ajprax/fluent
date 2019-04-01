from collections import Iterator
import itertools
from functools import wraps, partial
from operator import itemgetter

from experimental import enable_experimental_feature, experimental_feature_enabled


# Basically the same as None, but distinct so that None can be a valid value. Typically used as a default value for
# function arguments.
UNSET = object()


class EmptyReduceException(Exception):
    def __init__(self, msg=UNSET):
        if msg is UNSET:
            super().__init__("Consider using fold(default_value, reducer_function)")
        else:
            super().__init__(msg)


class NoFirstItem(Exception):
    pass


class RequirementException(Exception):
    pass


def require(cond, msg=None):
    """functionally the same as `assert cond, msg` but does not raise an AssertionError to avoid conflicts in tests"""
    if not cond:
        if msg is None:
            raise RequirementException()
        raise RequirementException(msg)


def t(fn):
    """
    Convert a function of N arguments to a function of one N-degree tuple

    Useful with higher order functions that expect single argument functions as inputs. This is given as an alternative
    to implementing a star_map, star_for_each, star_flat_map, etc functions for all collections.

    Example:
        def add(a, b):
            return a + b

        pairs = fit([(1, 2), (2, 4), (3, 6)])
        pairs.map(lambda pair: add(pair[0], pair[1]))
        # write more readable lambdas
        pairs.map(t(lambda a, b: a + b))
        # no manual decomposition of the pair, use functions that weren't written for use with HOFs
        pairs.map(t(add))
    """
    @wraps(fn)
    def tuplized(args):
        return fn(*args)
    return tuplized


def keep_every_nth(n, offset=0):
    def f(it):
        return it.enumerate().filter(t(lambda i, e: (i + offset) % n == 0)).map(itemgetter(1))
    return f


def fit(iterable):
    """
    Convert an iterable into a fluent iterable. Converts built in types to the corresponding fluent type, and all other
    iterables to FluentIterator.
    """
    if isinstance(iterable, tuple):
        return FluentTuple(iterable)
    elif isinstance(iterable, list):
        return FluentList(iterable)
    elif isinstance(iterable, set):
        return FluentSet(iterable)
    elif isinstance(iterable, dict):
        return FluentDict(iterable)
    elif isinstance(iterable, (range, FluentRange)):
        return FluentRange(iterable)
    elif experimental_feature_enabled("FloatRange"):
        from experimental import FloatRange
        if isinstance(iterable, FloatRange):
            return FluentRange(iterable)
    return FluentIterator(iterable)


class FluentIterator(Iterator):
    def __init__(self, it=()):
        self._it = iter(it)
        self._next = UNSET  # in case we have to read an extra item from the underlying iterator without returning it

    def __next__(self):
        if self._next is UNSET:
            return next(self._it)
        else:
            _next = self._next
            self._next = UNSET
            return _next

    def __iter__(self):
        return self

    def __contains__(self, item):
        for e in self:
            if e == item:
                return True
        return False

    def all(self, key=UNSET):
        return all(self if key is UNSET else self.map(key))

    def any(self, key=UNSET):
        return any(self if key is UNSET else self.map(key))

    def apply(self, fn):
        """
        Apply the given function to this iterator. Useful for bridging the gap between fluent and non-fluent operations
        by allowing functions which accept iterables to be used fluently.

        Example:
            # defined before introduction of fluent collections
            def sum(items):
                return reduce(lambda x, y: x + y, items, 0)

            FluentIterator((1, 2, 3)).apply(sum) == sum((1, 2, 3))

            fluent_iterator.filter(is_even).map(double).map(square).apply(sum)
        """
        return fn(self)

    def apply_and_fit(self, fn):
        return fit(fn(self))

    def batch(self, size):
        """
        Turn this iterator of items into an iterator of iterators each of size size (the last batch may have fewer items)
        """
        require(size > 0, f"cannot create batches with fewer than 1 item, got {size}")

        return (
            self.enumerate()
                # use itertools' groupby instead of self.group_by because it does not materialize the entire collection
                .apply_and_fit(partial(itertools.groupby, key=t(lambda i, e: i // size)))
                .map(itemgetter(1))
                .map(FluentIterator)
                .map(lambda b: b.map(itemgetter(1)))
        )

    def chain(self, *its):
        return FluentIterator(itertools.chain(self, *its))

    def cycle(self):
        return FluentIterator(itertools.cycle(self))

    def debug(self, fn, every_n=1):
        """
        similar to self.do(fn), but allows skipping elements to avoid spam
        every_n = 0 will never call fn to allow a pattern like:
            iter.debug(print, verbose and 10 or 0)...
        """
        require(every_n >= 0, f"every_n must be greater than or equal to 0, but got {every_n}")
        def gen():
            if every_n == 0:
                return self
            elif every_n == 1:
                for item in self:
                    fn(item)
                    yield item
            else:
                i = 0
                for item in self:
                    i += 1
                    if i == every_n:
                        fn(item)
                        i = 0
                    yield item
        return FluentIterator(gen())

    def dict(self):
        """requires that items be length 2 iterables"""
        return FluentDict(self)

    def distinct(self, key=lambda x: x):
        def gen():
            seen = FluentSet()
            for item in self:
                if seen.add(key(item)):
                    yield item
        return FluentIterator(gen())

    def do(self, fn=UNSET, tween_fn=UNSET):
        """
        equivalent to self.map(lambda e: fn(e); return e) if that were possible in python
        if tween_fn is present, it is called on sequential pairs of elements between when fn is called on each
        """
        def gen():
            if fn is UNSET and tween_fn is UNSET:
                for item in self:
                    yield item
            elif fn is UNSET:
                s = UNSET
                for f, s in self.sliding(2):
                    tween_fn(f, s)
                    yield f
                if s is not UNSET:
                    yield s
            elif tween_fn is UNSET:
                for item in self:
                    fn(item)
                    yield item
            else:
                s = UNSET
                for f, s in self.sliding(2):
                    fn(f)
                    yield f
                    tween_fn(f, s)
                if s is not UNSET:
                    fn(s)
                    yield s
        return FluentIterator(gen())

    def drop(self, n):
        """
        equivalent to self[n:]
        if n < 0, consumes the iterator and materializes up to -n elements in memory at a time
        """
        if n >= 0:
            for _ in range(n):
                try:
                    self.next()
                except StopIteration:
                    break
            return self
        else:
            window = None
            for window in self.sliding(-n):
                pass
            return FluentIterator() if window is None else FluentIterator(window)

    def drop_while(self, pred):
        return FluentIterator(itertools.dropwhile(pred, self))

    def enumerate(self):
        return FluentIterator(enumerate(self))

    def filter(self, pred=UNSET, for_each_discarded=UNSET):
        """
        If no predicate is provided, passes truthy items.
        If for_each_discarded is provided, it will be called on each item which does not pass
        """
        def gen():
            if pred is UNSET and for_each_discarded is UNSET:
                for item in self:
                    if item:
                        yield item
            elif pred is UNSET:
                for item in self:
                    if item:
                        yield item
                    else:
                        for_each_discarded(item)
            elif for_each_discarded is UNSET:
                for item in self:
                    if pred(item):
                        yield item
            else:
                for item in self:
                    if pred(item):
                        yield item
                    else:
                        for_each_discarded(item)
        return FluentIterator(gen())

    def first(self, pred=UNSET, if_none_pass=UNSET):
        """
        If no predicate is provided, returns the first item.
        If if_none_oass is provided, it will be returned when no items pass the predicate;
            if not provided, raises NoFirstItem if none pass
        """
        try:
            if pred is UNSET:
                return self.next()
            else:
                return self.filter(pred).next()
        except StopIteration:
            if if_none_pass is UNSET:
                raise NoFirstItem()
            else:
                return if_none_pass

    def flat_map(self, fn):
        def gen():
            for item in self:
                yield from fn(item)
        return FluentIterator(gen())

    def flatten(self):
        """requires that items are iterable"""
        return self.flat_map(lambda x: x)

    def fold(self, initial, fn):
        acc = initial
        for item in self:
            acc = fn(acc, item)
        return acc

    def for_each(self, fn=UNSET, tween_fn=UNSET):
        if fn is UNSET and tween_fn is UNSET:
            for item in self:
                pass
        elif fn is UNSET:
            for f, s in self.sliding(2):
                tween_fn(f, s)
        elif tween_fn is UNSET:
            for item in self:
                fn(item)
        else:
            s = UNSET
            for f, s in self.sliding(2):
                fn(f)
                tween_fn(f, s)
            if s is not UNSET:
                fn(s)

    def group_by(self, key=lambda x: x, group_factory=UNSET):
        """
        group_factory must be a 2 length iterable with a factory function and the name of the insertion method
        e.g. list, "append" or set, "add"
        """
        # not set as a default value because FluentList doesn't exist yet when this declaration is interpreted
        factory, insert_name = FluentList, "append" if group_factory is UNSET else group_factory
        out = FluentDict()
        for item in self:
            getattr(out.setdefault(key(item), factory()), insert_name)(item)
        return out

    def has_next(self):
        try:
            if self._next is UNSET:
                self._next = self.next()
            return True
        except StopIteration:
            return False

    def iter(self):
        # unlike the other collections, there's no need to create a copy here since the underlying iterator can only be
        # consumed once anyway
        return self

    def list(self):
        return FluentList(self)

    def map(self, fn):
        return FluentIterator(map(fn, self))

    def max(self, key=UNSET):
        """returns the first item with the maximum value if multiple items are equal"""
        # TODO: maybe allow a comparator exclusively with key
        if key is UNSET:
            best = self.next()
            for item in self:
                if item > best:
                    best = item
            return best
        else:
            best = self.next()
            best_key = key(best)
            for item in self:
                item_key = key(item)
                if item_key > best_key:
                    best = item
                    best_key = item_key
            return best

    def min(self, key=UNSET):
        """returns the first item with the minimum value if multiple items are equal"""
        # TODO: maybe allow a comparator exclusively with key
        if key is UNSET:
            best = self.next()
            for item in self:
                if item < best:
                    best = item
            return best
        else:
            best = self.next()
            best_key = key(best)
            for item in self:
                item_key = key(item)
                if item_key < best_key:
                    best = item
                    best_key = item_key
            return best

    def next(self):
        return next(self)

    def partition(self, pred):
        """
        Materializes multiple items in memory which belong to the partition that is not being consumed.

        i.e.
            passing, failing = items.partition(predicate)
            for passing_item in passing:
                ...
            # failing is now a materialized collection containing all elements which failed the predicate
        """
        t, f = self.tee()
        return t.filter(pred), f.filter(lambda e: not pred(e))

    def permutations(self, r=None):
        return FluentIterator(itertools.permutations(self, r))

    def product(self, *its, repeat=1):
        return FluentIterator(itertools.product(self, *its, repeat=repeat))

    def reduce(self, fn):
        try:
            return self.fold(next(self), fn)
        except StopIteration:
            raise EmptyReduceException("reduce on empty iterator")

    def set(self):
        return FluentSet(self)

    def size(self):
        count = 0
        for _ in self:
            count += 1
        return count

    def sliding(self, size, step=1):
        """
        All windows are guaranteed to have exactly size items
        When step > 1, may truncate the end of the iterator if a window cannot be filled
        """
        require(size > 0, f"window size must be greater than 0, got {size}")
        require(step > 0, f"step size must be greater than 0, got {step}")
        def gen():
            window = self.take(size).list()
            while window.size() == size:
                yield window
                window = window.extended(self.take(step)).drop(step)
        return FluentIterator(gen())

    def take(self, n):
        """
        equivalent to self[:n]
        if n < 0, consumes the iterator and materializes up to -n elements in memory at a time
        """
        def gen():
            if n >= 0:
                for _ in range(n):
                    try:
                        yield self.next()
                    except StopIteration:
                        break
            else:
                windows = self.sliding(-n)
                _next = windows.next()[0]
                while windows.has_next():
                    yield _next
                    _next = windows.next()[0]
        return FluentIterator(gen())

    def take_while(self, pred):
        def gen():
            for item in self:
                if pred(item):
                    yield item
                else:
                    self._next = item
                    break
        return FluentIterator(gen())

    def tee(self, n=2):
        return tuple(map(FluentIterator, itertools.tee(self, n)))

    def tuple(self):
        return FluentTuple(self)

    def zip(self, other=None, *others):
        return FluentIterator(zip(self, other, *others))


class FluentTuple(tuple):
    def __add__(self, other):
        return FluentTuple(super().__add__(other))

    def __getitem__(self, item):
        if isinstance(item, int):
            return super().__getitem__(item)
        else:
            return FluentTuple(super().__getitem__(item))

    def __repr__(self):
        return "FluentTuple({})".format(super().__repr__())

    def all(self, key=UNSET):
        return self.iter().all(key)

    def any(self, key=UNSET):
        return self.iter().any(key)

    def apply(self, fn):
        return fn(self)

    def apply_and_fit(self, fn):
        return fit(fn(self))

    def batch(self, size):
        return self.iter().batch(size).map(FluentTuple).tuple()

    def debug(self, fn, every_n=1):
        return self.iter().debug(fn, every_n)

    def dict(self):
        """requires that items be 2-tuples"""
        return FluentDict(self)

    def distinct(self, key=lambda x: x):
        return self.iter().distinct(key).tuple()

    def do(self, fn=UNSET, tween_fn=UNSET):
        self.iter().for_each(fn, tween_fn)
        return self

    def drop(self, n):
        return FluentTuple(self[n:])

    def drop_while(self, pred):
        return self.iter().drop_while(pred).tuple()

    def enumerate(self):
        return FluentTuple(enumerate(self))

    def filter(self, pred=UNSET, for_each_discarded=UNSET):
        return self.iter().filter(pred, for_each_discarded).tuple()

    def first(self, pred=UNSET, if_none_pass=UNSET):
        return self.iter().first(pred, if_none_pass)

    def flat_map(self, fn):
        return self.iter().flat_map(fn).tuple()

    def flatten(self):
        """requires that items are iterable"""
        return self.iter().flatten().tuple()

    def fold(self, initial, fn):
        return self.iter().fold(initial, fn)

    def for_each(self, fn=UNSET, tween_fn=UNSET):
        return self.iter().for_each(fn, tween_fn)

    def group_by(self, key, group_factory=UNSET):
        return self.iter().group_by(key, group_factory)

    def iter(self):
        return FluentIterator(self)

    def list(self):
        return FluentList(self)

    def map(self, fn):
        return self.iter().map(fn).tuple()

    def max(self, key=UNSET):
        return self.iter().max(key)

    def min(self, key=UNSET):
        return self.iter().min(key)

    def partition(self, pred):
        t, f = self.iter().partition(pred)
        return t.tuple(), f.tuple()

    def permutations(self, r=None):
        return self.iter().permutations(r)

    def product(self, *its, repeat=1):
        return self.iter().product(*its, repeat=repeat)

    def reduce(self, fn):
        return self.iter().reduce(fn)

    def reversed(self):
        return FluentTuple(reversed(self))

    def set(self):
        return FluentSet(self)

    def size(self):
        return len(self)

    def sliding(self, size, step=1):
        return self.iter().sliding(size, step).tuple()

    def sorted(self, key=None, reverse=False):
        return FluentTuple(sorted(self, key=key, reverse=reverse))

    def take(self, n):
        return FluentTuple(self[:n])

    def take_while(self, pred):
        return self.iter().take_while(pred).tuple()

    def tuple(self):
        return FluentTuple(self)

    def zip(self, other=None, *others):
        return self.iter().zip(other, *others).tuple()


class FluentList(list):
    def __add__(self, other):
        return FluentList(super().__add__(other))

    def __getitem__(self, item):
        if isinstance(item, int):
            return super().__getitem__(item)
        else:
            return FluentList(super().__getitem__(item))

    def __repr__(self):
        return "FluentList({})".format(super().__repr__())

    def all(self, key=UNSET):
        return self.iter().all(key)

    def any(self, key=UNSET):
        return self.iter().any(key)

    def append(self, item):
        """equivalent to ls.append(item), but returns self to allow chaining"""
        super().append(item)
        return self

    def appended(self, item):
        """returns a copy of the list with the new item appended"""
        ls = self.list()
        ls.append(item)
        return ls

    def apply(self, fn):
        return fn(self)

    def apply_and_fit(self, fn):
        return fit(fn(self))

    def batch(self, size):
        return self.iter().batch(size).map(FluentList).list()

    def debug(self, fn, every_n=1):
        return self.iter().debug(fn, every_n).list()

    def dict(self):
        """requires that items be 2 element iterables"""
        return FluentDict(self)

    def distinct(self, key=lambda x: x):
        return self.iter().distinct(key).list()

    def do(self, fn=UNSET, tween_fn=UNSET):
        self.iter().for_each(fn, tween_fn)
        return self

    def drop(self, n):
        return FluentList(self[n:])

    def drop_while(self, pred):
        return self.iter().drop_while(pred).list()

    def enumerate(self):
        return FluentList(enumerate(self))

    def extended(self, iterable):
        ls = self.list()
        ls.extend(iterable)
        return ls

    def filter(self, pred=UNSET, for_each_discarded=UNSET):
        return self.iter().filter(pred, for_each_discarded).list()

    def first(self, pred=UNSET, if_none_pass=UNSET):
        return self.iter().first(pred, if_none_pass)

    def flat_map(self, fn):
        return self.iter().flat_map(fn).list()

    def flatten(self):
        """requires that items are iterable"""
        return self.iter().flatten().list()

    def fold(self, initial, fn):
        return self.iter().fold(initial, fn)

    def for_each(self, fn=UNSET, tween_fn=UNSET):
        return self.iter().for_each(fn, tween_fn)

    def group_by(self, key):
        return self.iter().group_by(key)

    def iter(self):
        return FluentIterator(self)

    def list(self):
        return FluentList(self)

    def map(self, fn):
        return self.iter().map(fn).list()

    def max(self, key=UNSET):
        return self.iter().max(key)

    def min(self, key=UNSET):
        return self.iter().min(key)

    def partition(self, pred):
        t, f = self.iter().partition(pred)
        return t.list(), f.list()

    def permutations(self, r=None):
        return self.iter().permutations(r)

    def product(self, *its, repeat=1):
        return self.iter().product(*its, repeat=repeat)

    def reduce(self, fn):
        return self.iter().reduce(fn)

    def reversed(self):
        return FluentList(reversed(self))

    def set(self):
        return FluentSet(self)

    def size(self):
        return len(self)

    def sliding(self, size, step=1):
        return self.iter().sliding(size, step).list()

    def sorted(self, key=None, reverse=False):
        return FluentList(sorted(self, key=key, reverse=reverse))

    def take(self, n):
        return FluentList(self[:n])

    def take_while(self, pred):
        return self.iter().take_while(pred).list()

    def tuple(self):
        return FluentTuple(self)

    def zip(self, other=None, *others):
        return self.iter().zip(other, *others).list()


class FluentSet(set):
    def __add__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersection(other)

    def __or__(self, other):
        return self.union(other)

    def __sub__(self, other):
        return self.difference(other)

    def __xor__(self, other):
        return self.symmetric_difference(other)

    def add(self, item):
        """same as default set.add but returns True if the item was added or False if it was already present"""
        new = item not in self
        super().add(item)
        return new

    def all(self, key=UNSET):
        return self.iter().all(key)

    def any(self, key=UNSET):
        return self.iter().any(key)

    def apply(self, fn):
        return fn(self)

    def apply_and_fit(self, fn):
        return fit(fn(self))

    def batch(self, size):
        """
        Unlike other batch methods which return the same type, does not return a FluentSet because the individual
        batches are not hashable
        """
        return self.iter().batch(size).map(FluentSet)

    def debug(self, fn, every_n=1):
        return self.iter().debug(fn, every_n)

    def dict(self):
        """requires that items be 2-tuples"""
        return FluentDict(self)

    def difference(self, *s):
        return FluentSet(super().difference(*s))

    def discard(self, item):
        removed = item in self
        super().discard(item)
        return removed

    def do(self, fn=UNSET, tween_fn=UNSET):
        self.iter().for_each(fn, tween_fn)
        return self

    def enumerate(self):
        return self.iter().enumerate()

    def filter(self, pred=UNSET, for_each_discarded=UNSET):
        return self.iter().filter(pred, for_each_discarded).set()

    def first(self, pred=UNSET, if_none_pass=UNSET):
        return self.iter().first(pred, if_none_pass)

    def flat_map(self, fn):
        return self.iter().flat_map(fn).set()

    def flatten(self):
        """requires that items are iterable"""
        return self.iter().flatten().set()

    def fold(self, initial, fn):
        return self.iter().fold(initial, fn)

    def for_each(self, fn=UNSET, tween_fn=UNSET):
        return self.iter().for_each(fn, tween_fn)

    def group_by(self, key):
        return self.iter().group_by(key)

    def intersection(self, *s):
        return FluentSet(super().intersection(*s))

    def iter(self):
        return FluentIterator(self)

    def list(self):
        return FluentList(self)

    def map(self, fn):
        return self.iter().map(fn).set()

    def max(self, key=UNSET):
        return self.iter().max(key)

    def min(self, key=UNSET):
        return self.iter().min(key)

    def partition(self, pred):
        t, f = self.iter().partition(pred)
        return t.set(), f.set()

    def permutations(self, r=None):
        return self.iter().permutations(r)

    def product(self, *its, repeat=1):
        return self.iter().product(*its, repeat=repeat)

    def reduce(self, fn):
        return self.iter().reduce(fn)

    def set(self):
        return FluentSet(self)

    def size(self):
        return len(self)

    def symmetric_difference(self, s):
        return FluentSet(super().symmetric_difference(s))

    def tuple(self):
        return FluentTuple(self)

    def union(self, *s):
        return FluentSet(super().union(*s))

    def zip(self, other=None, *others):
        return self.iter().zip(other, *others)


class FluentDict(dict):
    def __repr__(self):
        return "FluentDict({})".format(super().__repr__())

    def all(self, key=UNSET):
        return self.iter().all(key)

    def any(self, key=UNSET):
        return self.iter().any(key)

    def apply(self, fn):
        return fn(self)

    def apply_and_fit(self, fn):
        return fit(fn(self))

    def batch(self, size):
        """
        Unlike other batch methods which return the same type, does not return a FluentDict because the individual
        batches are not hashable
        """
        return self.items().batch(size).map(FluentDict)

    def copy(self):
        return FluentDict(super().copy())

    def debug(self, fn, every_n=1):
        return self.items().debug(fn, every_n)

    def dict(self):
        return FluentDict(self)

    def do(self, fn=UNSET, tween_fn=UNSET):
        self.iter().for_each(fn, tween_fn)
        return self

    def filter(self, pred=UNSET, for_each_discarded=UNSET):
        """both pred and for_each_discarded are called with tuples of (key, value)"""
        return self.items().filter(pred, for_each_discarded).dict()

    def filter_keys(self, pred=UNSET, for_each_discarded=UNSET):
        """pred is called with key only, but for_each_discarded is called with tuples of (key, value)"""
        def gen():
            if pred is UNSET and for_each_discarded is UNSET:
                for k, v in self.items():
                    if k:
                        yield k, v
            elif pred is UNSET:
                for k, v in self.items():
                    if k:
                        yield k, v
                    else:
                        for_each_discarded((k, v))
            elif for_each_discarded is UNSET:
                for k, v in self.items():
                    if pred(k):
                        yield k, v
            else:
                for k, v in self.items():
                    if pred(k):
                        yield k, v
                    else:
                        for_each_discarded((k, v))
        return FluentDict(gen())

    def filter_values(self, pred=UNSET, for_each_discarded=UNSET):
        """pred is called with value only, but for_each_discarded is called with tuples of (key, value)"""
        def gen():
            if pred is UNSET and for_each_discarded is UNSET:
                for k, v in self.items():
                    if v:
                        yield k, v
            elif pred is UNSET:
                for k, v in self.items():
                    if v:
                        yield k, v
                    else:
                        for_each_discarded((k, v))
            elif for_each_discarded is UNSET:
                for k, v in self.items():
                    if pred(v):
                        yield k, v
            else:
                for k, v in self.items():
                    if pred(v):
                        yield k, v
                    else:
                        for_each_discarded((k, v))
        return FluentDict(gen())

    def first(self, pred=UNSET, if_none_pass=UNSET):
        return self.iter().first(pred, if_none_pass)

    def flat_map(self, fn):
        return self.items().flat_map(fn).dict()

    # TODO: consider adding flatten

    def fold(self, initial, fn):
        return self.items().fold(initial, fn)

    def for_each(self, fn=UNSET, tween_fn=UNSET):
        return self.items().for_each(fn, tween_fn)

    def group_by(self, key=lambda x: x):
        return self.items().group_by(key).map_values(FluentDict)

    def invert(self):
        """reverse all key, value pairs to value, key pairs"""
        return FluentDict({v: k for k, v in self.items()})

    def invert_flatten(self):
        """reverse key, [values] pairs to [values], key pairs and then flatten to value, key pairs"""
        return FluentDict({v: k for k, vs in self.items() for v in vs})

    def items(self):
        return FluentIterator(super().items())

    def iter(self):
        return FluentIterator(iter(self))

    def keys(self):
        return FluentIterator(super().keys())

    def list(self):
        """returns a list of key, value pairs instead of a list of keys as with list({...})"""
        return FluentList(self.items())

    def map(self, fn):
        return self.items().map(fn).dict()

    def map_keys(self, fn):
        return FluentDict({fn(k): v for k, v in self.items()})

    def map_values(self, fn):
        return FluentDict({k: fn(v) for k, v in self.items()})

    def max(self, key=UNSET):
        return self.iter().max(key)

    def min(self, key=UNSET):
        return self.iter().min(key)

    def partition(self, pred):
        t, f = self.items().partition(pred)
        return t.dict(), f.dict()

    def partition_keys(self, pred):
        t, f = FluentDict(), FluentDict()
        for k, v in self.items():
            if pred(k):
                t[k] = v
            else:
                f[k] = v
        return t, f

    def partition_values(self, pred):
        t, f = FluentDict(), FluentDict()
        for k, v in self.items():
            if pred(v):
                t[k] = v
            else:
                f[k] = v
        return t, f

    def put(self, k, v):
        """equivalent to d[k] = v, but returns self to allow chaining"""
        self[k] = v
        return self

    def tuple(self):
        return FluentTuple(self.items())

    def update(self, E=None, **F):
        """equivalent to d.update(...), but returns self to allow chaining"""
        super().update(E, F)
        return self

    def updated(self, E=None, **F):
        d = self.dict()
        d.update(E, **F)
        return d

    def reduce(self, fn):
        return self.items().reduce(fn)

    def size(self):
        return len(self)

    def values(self):
        return FluentIterator(super().values())


class FluentRange:
    def __init__(self, *a):
        require(0 < len(a) <= 3)
        if len(a) == 1 and isinstance(a[0], (range, FluentRange)):
            self._range = a[0]
        elif all(isinstance(_, int) for _ in a):
            self._range = range(*a)
        else:
            if experimental_feature_enabled("FloatRange"):
                from experimental import FloatRange
                self._range = FloatRange(*a)
            else:
                raise Exception()
        self.start = self._range.start
        self.stop = self._range.stop
        self.step = self._range.step

    def __contains__(self, item):
        return item in self._range

    def __getitem__(self, item):
        r = self._range[item]
        if isinstance(item, int):
            return r
        return FluentRange(r)

    def __iter__(self):
        return iter(self._range)

    def __len__(self):
        return len(self._range)

    def __repr__(self):
        return "FluentRange({}, {}, {})".format(self.start, self.stop, self.step)

    def all(self, key=UNSET):
        return self.iter().all(key)

    def any(self, key=UNSET):
        return self.iter().any(key)

    def apply(self, fn):
        return fn(self)

    def apply_and_fit(self, fn):
        return fit(fn(self))

    def batch(self, size):
        return self.iter().batch(size)

    def debug(self, fn, every_n=1):
        return self.iter().debug(fn, every_n)

    def distinct(self, key=lambda x: x):
        return self.iter().distinct(key)

    def do(self, fn=UNSET, tween_fn=UNSET):
        self.iter().for_each(fn, tween_fn)
        return self

    def drop(self, n):
        return FluentRange(self.start + self.step * n, self.stop, self.step)

    def drop_while(self, pred):
        # TODO: maybe iterate until we reach the boundary and return a new range starting there
        return self.iter().drop_while(pred)

    def enumerate(self):
        return FluentList(enumerate(self))

    def filter(self, pred=UNSET, for_each_discarded=UNSET):
        return self.iter().filter(pred, for_each_discarded)

    def first(self, pred=UNSET, if_none_pass=UNSET):
        return self.iter().first(pred, if_none_pass)

    def flat_map(self, fn):
        return self.iter().flat_map(fn)

    def fold(self, initial, fn):
        return self.iter().fold(initial, fn)

    def for_each(self, fn=UNSET, tween_fn=UNSET):
        return self.iter().for_each(fn, tween_fn)

    def group_by(self, key):
        return self.iter().group_by(key)

    def iter(self):
        return FluentIterator(self._range)

    def list(self):
        return FluentList(self)

    def map(self, fn):
        return self.iter().map(fn)

    def max(self, key=UNSET):
        return self.iter().max(key)

    def min(self, key=UNSET):
        return self.iter().min(key)

    def partition(self, pred=UNSET):
        return self.iter().partition(pred)

    def permutations(self, r=None):
        return self.iter().permutations(r)

    def product(self, *its, repeat=1):
        return self.iter().product(*its, repeat=repeat)

    def reduce(self, fn):
        return self.iter().reduce(fn)

    def reversed(self):
        if isinstance(self._range, range):
            return FluentRange(range(self._range.start + (len(self._range) - 1) * self._range.step, self._range.start - self._range.step, -self._range.step))
        else:
            return FluentRange(reversed(self._range))

    def set(self):
        return FluentSet(self)

    def size(self):
        return len(self)

    def sliding(self, size, step=1):
        return self.iter().sliding(size, step)

    def sorted(self, key=None, reverse=False):
        return FluentIterator(sorted(self, key=key, reverse=reverse))

    def take(self, n):
        return FluentRange(self.start, min(self.stop, self.start + self.step * n), self.step)

    def take_while(self, pred):
        # TODO: maybe iterate until we reach the boundary and return a new range ending there
        return self.iter().take_while(pred)

    def tuple(self):
        return FluentTuple(self)

    def zip(self, other=None, *others):
        return self.iter().zip(other, *others)
