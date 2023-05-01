import random
from collections import deque

from src.util_cls import Experience


class ReplayMemory(object):
    """
    A cyclic buffer of bounded size that holds the experiences observed recently.

    Methods:
    push: Adds a new experience to the memory.
    sample: Retrieves several random experiences from the memory.
    """

    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, experience: Experience) -> None:
        """
        Save an experience. The deque will automatically remove items when
        full, so no need to check capacity.
        Args:
          experience: Experience. Tuple of arguments that should fit into an experience.
        """
        self.memory.append(experience)

    def sample(self, batch_size: int) -> Experience:
        """
        Retrieve batch_size number of random experiences from the memory.
        Args:
          batch_size: Number of experiences to retrieve
        """
        x = random.sample(self.memory, batch_size)
        return Experience(*zip(*(x)))

    def __len__(self) -> int:
        return len(self.memory)


# class PrioritizedReplayBuffer(object):
#     """
#     A prioritized cyclic buffer of bounded size that holds the experiences
#     observed recently.


#     Methods:
#     push: Adds a new experience to the memory.
#     sample: Retrieves several random experiences from the memory.
#     update_priority: Update the priority of a stored experience.
#     """

#     def __init__(
#         self,
#         capacity: int,
#         alpha: float = 0.6,
#         beta: float = 0.4,
#         beta_annealing: float = 0.99,
#     ) -> None:
#         self.capacity = capacity
#         self.memory = deque([], maxlen=capacity)
#         self.priorities = deque([], maxlen=capacity)
#         self.alpha = alpha
#         self.beta = beta
#         self.beta_annealing = beta_annealing
#         self.experience = namedtuple(
#             "Experience",
#             field_names=["state", "action", "reward", "next_state", "done", "priority"],
#         )

#     def push(self, experience: Experience) -> None:
#         """
#         Save an experience. The deque will automatically remove items when
#         full, so no need to check capacity.
#         Args:
#           experience: Experience. Tuple of arguments that should fit into an experience.
#         """
#         max_priority = max(self.priorities) if self.memory else 1.0
#         self.memory.append(experience)
#         self.priorities.append(max_priority)

#     def sample(self, batch_size: int) -> Experience:
#         """
#         Retrieve batch_size number of experiences from the memory based on priority.
#         Args:
#           batch_size: Number of experiences to retrieve
#         """
#         priorities = self.priorities
#         probs = [p**self.alpha for p in priorities]
#         probs /= sum(probs)
#         indices = random.choices(range(len(self.memory)), k=batch_size, weights=probs)
#         samples = [self.memory[idx] for idx in indices]
#         total_priority = (len(self.memory) ** self.beta) * self.beta_annealing
#         weights = [
#             (p / total_priority) ** -self.beta
#             for p in [self.priorities[idx] for idx in indices]
#         ]
#         weights /= max(weights)
#         experiences = [
#             self.experience(*exp, weight) for exp, weight in zip(samples, weights)
#         ]
#         return self.experience(*zip(*experiences))

#     def update_priority(self, indices: list, priorities: list) -> None:
#         """
#         Update the priority of a stored experience.
#         Args:
#           indices: List of indices corresponding to the experiences whose priorities are to be updated.
#           priorities: List of updated priorities.
#         """
#         for i, priority in zip(indices, priorities):
#             self.priorities[i] = priority

#     def __len__(self) -> int:
#         return len(self.memory)
