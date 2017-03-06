from abc import abstractproperty, ABCMeta
from functools import wraps
from typing import Any, Dict, Callable, Iterable, Iterator, Generic, List, Optional, Set, Tuple, TypeVar

T = TypeVar("T", covariant=True)
U = TypeVar("U")

K = TypeVar("K", covariant=True)
V = TypeVar("V", covariant=True)
K2 = TypeVar("K2")
V2 = TypeVar("V2")

# Define types we're going to use so that they can be referenced in type annotations before they're properly defined
FluentIterable = Iterable
FluentIterator = Iterator
FluentList = List
FluentSet = Set
FluentDict = Dict
FluentOption = Optional
FluentTry = Optional


class FluentIterable(Generic[T], Iterable[T]):
    __metaclass__ = ABCMeta

    @abstractproperty
    def _repr_fmt(self) -> str:
        ...

    def size(self) -> int:
        return len(self)

    def map(self, fn: Callable[[T], U]) -> FluentIterable[U]:
        return self.__class__(map(fn, self))

    def flat_map(self, fn: Callable[[T], Iterable[U]]) -> FluentIterable[U]:
        def yielder():
            for item in self:
                yield from fn(item)
        return self.__class__(yielder())

    def filter(self, pred: Callable[[T], bool]) -> FluentIterable[T]:
        return self.__class__(filter(pred, self))

    def for_each(self, fn: Callable[[T], None]) -> None:
        for item in self:
            fn(item)

    def fold(self, zero: U, fn: Callable[[U, T], U]) -> U:
        acc = zero
        for item in self:
            acc = fn(acc, item)
        return acc

    def group_by(self, fn: Callable[[T], U]) -> FluentDict[U, FluentList[T]]:
        out = FluentDict()
        for item in self:
            out.setdefault(fn(item), FluentList()).append(item)
        return out

    def flatten(self):
        return self.flat_map(lambda i: i)

    def to_list(self) -> FluentList[T]:
        return FluentList(self)

    def to_set(self) -> FluentSet[T]:
        return FluentSet(self)

    def to_dict(self) -> FluentDict[Any, Any]:
        return FluentDict(self)

    def reduce(self, fn: Callable[[T, T], T]) -> T:
        return self.tail().fold(self.head(), fn)

    def zip(self, other: Iterable[U]) -> FluentIterable[Tuple[T, U]]:
        return self.__class__(zip(self, other))

    def __contains__(self, item) -> bool:
        for e in self:
            if e == item:
                return True
        return False

    def __repr__(self) -> str:
        return self._repr_fmt.format(", ".join(self.map(repr)))


class FluentIterator(Generic[T], FluentIterable[T], Iterator[T]):
    def _repr_fmt(self) -> str:
        raise NotImplementedError

    def __init__(self, _repr: Iterable[T]=()):
        self._repr = iter(_repr)

    def __repr__(self) -> str:
        return super(FluentIterable, self).__repr__()

    def __next__(self) -> T:
        return next(self._repr)

    def take_while(self, pred: Callable[[T], bool]) -> FluentIterator[T]:
        def yielder():
            for item in self:
                if not pred(item):
                    break
                yield item
        return FluentIterator(yielder())

    def drop_while(self, pred: Callable[[T], bool]) -> FluentIterator[T]:
        def yielder():
            for item in self:
                if not pred(item):
                    break
            yield item # the first item that doesn't pass the predicate should be included
            yield from self
        return FluentIterator(yielder())


class FluentDict(Generic[K, V], Dict[K, V]):
    def size(self) -> int:
        return len(self)

    def map(self, fn: Callable[[Tuple[K, V]], Tuple[K2, V2]]) -> FluentDict[K2, V2]:
        return FluentDict(map(fn, self.items()))

    def flat_map(self, fn: Callable[[Tuple[K, V]], Iterable[Tuple[K2, V2]]]) -> FluentDict[K2, V2]:
        def yielder():
            for kv in self.items():
                yield from fn(kv)
        return FluentDict(yielder())

    def filter(self, pred: Callable[[Tuple[K, V]], bool]) -> FluentDict[K, V]:
        return FluentDict([kv for kv in self.items() if pred(kv)])

    def to_list(self) -> FluentList[Tuple[K, V]]:
        return FluentList(self.items())

    def to_set(self) -> FluentSet[Tuple[K, V]]:
        return FluentSet(self.items())

    def to_dict(self) -> FluentDict[K, V]:
        return self

    def __repr__(self) -> str:
        def pair_format(kv):
            k, v = kv
            return "({!r}, {!r})".format(k, v)
        return "FluentDict([{}])".format(", ".join(self.to_list().map(pair_format)))


class FluentList(Generic[T], FluentIterable[T], List[T]):
    @property
    def _repr_fmt(self) -> str:
        return "FluentList([{}])"

    def __iter__(self) -> FluentIterator[T]:
        return FluentIterator(super(List, self).__iter__())

    def take(self, n: int) -> FluentList[T]:
        return self[:n]

    def take_while(self, pred: Callable[[T], bool]) -> FluentList[T]:
        return iter(self).take_while(pred).to_list()

    def drop(self, n: int) -> FluentList[T]:
        return self[n:]

    def drop_while(self, pred: Callable[[T], bool]) -> FluentList[T]:
        return iter(self).drop_while(pred).to_list()

    def head(self) -> T:
        return self[0]

    def head_option(self) -> FluentOption[T]:
        try:
            return Something(self.head())
        except IndexError:
            return Nothing()

    def tail(self) -> FluentList[T]:
        return self[1:]

    def init(self) -> FluentList[T]:
        return self[:-1]

    def last(self) -> FluentList[T]:
        return self[-1]

    def last_option(self) -> FluentOption[T]:
        try:
            return Something(self.last())
        except IndexError:
            return Nothing()

    def __getitem__(self, item):
        if isinstance(item, int):
            return super(List, self).__getitem__(item)
        elif isinstance(item, slice):
            return FluentList(super(List, self).__getitem__(item))


class FluentSet(Generic[T], FluentIterable[T], Set[T]):
    @property
    def _repr_fmt(self) -> str:
        return "FluentSet({{{}}})"

    def __iter__(self) -> FluentIterator[T]:
        return FluentIterator(super(Set, self).__iter__())


class FluentOption(Generic[T], FluentIterable[T]):
    @classmethod
    def from_optional(cls, opt: Optional[T]) -> FluentOption[T]:
        if opt is None:
            return Nothing()
        else:
            return Something(opt)


class Something(Generic[T], FluentOption[T]):
    def _repr_fmt(self):
        raise NotImplementedError()

    def __init__(self, t: T):
        self._t = t

    def __iter__(self) -> Iterator[T]:
        yield self._t

    @property
    def size(self):
        return 1

    def get(self) -> T:
        return self._t

    def get_or_else(self, default: T) -> T:
        return self._t

    def is_something(self) -> bool:
        return True

    def map(self, fn: Callable[[T], U]) -> FluentOption[U]:
        return Something(fn(self._t))

    def flat_map(self, fn: Callable[[T], FluentOption[U]]) -> FluentOption[U]:
        return fn(self._t)

    def filter(self, pred: Callable[[T], bool]) -> FluentOption[T]:
        if pred(self._t):
            return self
        else:
            return Nothing()

    def __len__(self) -> int:
        return 1

    def __repr__(self) -> str:
        return "Something({})".format(self._t)


class Nothing(Generic[T], FluentOption[T]):
    def _repr_fmt(self) -> str:
        raise NotImplementedError()

    def __iter__(self):
        return iter(())

    @property
    def size(self) -> int:
        return 0

    def map(self, fn: Callable[[T], U]) -> FluentOption[U]:
        return self

    def flat_map(self, fn: Callable[[T], FluentOption[U]]) -> FluentOption[U]:
        return self

    def filter(self, pred: Callable[[T], bool]) -> FluentOption[T]:
        return self

    def __len__(self) -> int:
        return 0

    def for_each(self, fn: Callable[[T], None]):
        pass

    def __repr__(self) -> str:
        return "Nothing()"


class ExceptionFromSuccess(Generic[T], Exception):
    def __init__(self, t: T):
        super().__init__("{!r}".format(t))
        self.t = t


class FilterFailed(Generic[T], Exception):
    def __init__(self, t: T, pred: Callable[[T], bool]):
        super().__init__("{!r} failed {!r}".format(t, pred))
        self.t = t
        self.pred = pred


class FluentTry(Generic[T]):
    @classmethod
    def safely(cls, fn: Callable[[], T]) -> FluentTry[T]:
        try:
            return Success(fn())
        except Exception as e:
            return Failure(e)

    @classmethod
    def safe(cls, fn: Callable[..., T]) -> Callable[..., FluentTry[T]]:
        @wraps(fn)
        def decorated(*a, **kw):
            return FluentTry.safely(lambda: fn(*a, **kw))
        return decorated


class Success(Generic[T], FluentTry[T]):
    def __init__(self, t: T):
        self._t = t

    def get(self) -> T:
        return self._t

    def exc(self) -> Exception:
        raise ExceptionFromSuccess(self._t)

    def map(self, fn: Callable[[T], U]) -> FluentTry[U]:
        return Success(fn(self._t))

    def flat_map(self, fn: Callable[[T], FluentTry[U]]) -> FluentTry[U]:
        return fn(self._t)

    def filter(self, pred: Callable[[T], bool]) -> FluentTry[U]:
        if pred(self._t):
            return self
        else:
            raise FilterFailed(self._t, pred)

    def for_each(self, fn: Callable[[T], None]):
        fn(self._t)

    def __repr__(self) -> str:
        return "Success({!r})".format(self._t)


class Failure(Generic[T], FluentTry[Any]):
    def __init__(self, exc: Exception):
        self._exc = exc

    def get(self) -> T:
        raise self._exc

    def exc(self) -> Exception:
        return self._exc

    def map(self, fn: Callable[[T], U]) -> FluentTry[U]:
        return self

    def flat_map(self, fn: Callable[[T], FluentTry[U]]) -> FluentTry[U]:
        return self

    def filter(self, pred: Callable[[T], bool]) -> FluentTry[T]:
        return self

    def for_each(self, fn: Callable[[T], None]):
        pass

    def __repr__(self) -> str:
        return "Failure({!r})".format(self._exc)


# short names
fit = FluentIterable
fiter = FluentIterator
fdict = FluentDict
flist = FluentList
fset = FluentSet
fopt = FluentOption
some = Something
none = Nothing
ftry = FluentTry
success = Success
failure = Failure
