from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from queue import Empty, PriorityQueue, Queue
from typing import Generic, Iterable, Optional, TypeVar

ConstantsT = TypeVar('ConstantsT')


class State(ABC, Generic[ConstantsT]):
    T = TypeVar('T', bound='State')

    def __init__(self, constants: ConstantsT, prev: Optional[T] = None, **_kwargs):
        self.c = constants
        self.prev = prev

    @abstractmethod
    def move(self: T, distance: int = 1, **kwargs) -> T:
        pass

    @property
    @abstractmethod
    def is_finished(self: T) -> bool:
        pass

    @property
    @abstractmethod
    def next_states(self: T) -> Iterable[T]:
        pass

    @abstractmethod
    def __hash__(self: T) -> int:
        """Should return a property suitable for keeping track of visited states etc., such as position."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return hash(self) == hash(other)


class BFSState(State[ConstantsT], ABC, Generic[ConstantsT]):
    T = TypeVar('T', bound='BFSState')

    def move(self: T, _distance: int = 1, **kwargs) -> T:
        return self.__class__(prev=self, constants=self.c, **kwargs)


class DijkstraState(State[ConstantsT], ABC, Generic[ConstantsT]):
    T = TypeVar('T', bound='DijkstraState')

    def __init__(self, cost: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.cost = cost

    def move(self: T, distance: int = 1, **kwargs) -> T:
        return self.__class__(cost=self.cost + distance, prev=self, constants=self.c, **kwargs)

    def __lt__(self, other: object) -> bool:
        """By defining this, states with a lower cost will have priority in the queue."""
        if not isinstance(other, DijkstraState):
            return NotImplemented
        return self.cost < other.cost


class AStarState(DijkstraState[ConstantsT], ABC, Generic[ConstantsT]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = self.cost + self.heuristic

    def __lt__(self, other: object) -> bool:
        """By defining this, states with a lower score (cost + heuristic) will have priority in the queue."""
        if not isinstance(other, AStarState):
            return NotImplemented
        return self.score < other.score

    @property
    @abstractmethod
    def heuristic(self) -> int:
        """Basically turns the Dijkstra algo into A*"""


class ShortestPath(ABC, Generic[State.T]):
    visited: set[State.T]
    end_state: State.T
    performance: int

    def __init__(self):
        self.visited = set()

    @abstractmethod
    def from_queue(self) -> State.T:
        pass

    @abstractmethod
    def to_queue(self, state: State.T) -> None:
        pass

    def find(self, initial_state: State.T) -> ShortestPath[State.T]:
        counter = 0
        state = initial_state
        self.to_queue(state)
        try:
            while state := self.from_queue():
                counter += 1
                if state.is_finished:
                    break
                # update costs for next possible states
                for next_state in state.next_states:
                    if next_state not in self.visited and self._handle_state(next_state):
                        self.to_queue(next_state)
                self._on_state_processed(state)
                # raise Empty
        except Empty:
            pass
        self.end_state = state
        self.performance = counter
        return self

    def _on_state_processed(self, state: State.T) -> None:
        pass

    @abstractmethod
    def _handle_state(self, state: State.T) -> bool:
        """Determine wether a state should be added to the queue."""
        pass

    @property
    def states(self) -> list[State.T]:
        """Get the complete path by traversing back to the start."""
        state = self.end_state
        path = [state]
        while state := state.prev:  # type: ignore
            path = [state] + path  # type: ignore
        return path

    @property
    @abstractmethod
    def length(self) -> int:
        pass


class ShortestPathBFS(ShortestPath[BFSState.T], Generic[BFSState.T]):
    def __init__(self) -> None:
        super().__init__()
        self._queue: deque[BFSState.T] = deque()

    def from_queue(self) -> BFSState.T:
        try:
            return self._queue.popleft()
        except IndexError as exc:
            raise Empty() from exc

    def to_queue(self, state: BFSState.T) -> None:
        self._queue.append(state)

    def _handle_state(self, state: BFSState.T) -> bool:
        self.visited.add(state)
        return True

    @property
    def length(self) -> int:
        return len(self.states) - 1


class ShortestPathDijkstra(ShortestPath[DijkstraState.T], Generic[DijkstraState.T]):
    def __init__(self) -> None:
        super().__init__()
        self._queue: Queue[DijkstraState.T] = PriorityQueue()
        self._lowest_costs: dict[DijkstraState.T, int] = {}

    def from_queue(self) -> DijkstraState.T:
        return self._queue.get_nowait()

    def to_queue(self, state: DijkstraState.T) -> None:
        self._queue.put_nowait(state)

    def _on_state_processed(self, state: DijkstraState.T) -> None:
        self.visited.add(state)

    def _handle_state(self, state: DijkstraState.T) -> bool:
        if old_cost := self._lowest_costs.get(state):
            if state.cost >= old_cost:
                # you can do better than that
                return False
        # most efficient route to this state so far: update cost
        self._lowest_costs[state] = state.cost
        return True

    @property
    def length(self) -> int:
        return self.end_state.cost


def shortest_path(initial_state: State.T) -> ShortestPath[State.T]:
    path: ShortestPath[State.T]
    if isinstance(initial_state, BFSState):
        path = ShortestPathBFS()  # type: ignore
    elif isinstance(initial_state, DijkstraState):
        path = ShortestPathDijkstra()  # type: ignore
    else:
        raise NotImplementedError
    return path.find(initial_state)
