import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, state_size, action_size=None):
        self.states = np.empty((capacity, *state_size))
        self.actions = np.empty(capacity) if action_size is None else np.empty((capacity, *action_size))
        self.rewards = np.empty(capacity)
        self.dones = np.empty(capacity, dtype=np.bool)
        self.next_states = np.empty((capacity, *state_size))
        self.index = 0
        self.capacity = capacity
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.index

    def store(self, transition):
        s, a, r, d, n_s = transition

        self.states[self.index] = s
        self.actions[self.index] = a
        self.rewards[self.index] = r
        self.dones[self.index] = d
        self.next_states[self.index] = n_s

        transition_index = self.index
        self.index = (self.index + 1) % self.capacity
        if not self.full:
            self.full = self.index == 0

        return transition_index

    def sample(self, batch_size):
        size = len(self)
        if size < batch_size:
            raise Exception('Batch Size is too large')

        indices = np.random.randint(0, size, size=batch_size)

        return self.get_data(indices)

    def get_data(self, indices):
        return (self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.dones[indices],
                self.next_states[indices])


class PriorityBuffer:
    eps = 0.01
    alpha = 0.6
    beta = 0.4
    beta_inc = 0.001
    err_clip = 1

    def __init__(self, capacity, state_size):
        self.tree = SumTree(capacity)
        self.data = ReplayBuffer(capacity, state_size)
        self.max = 0

    def store(self, transition):
        priority = self.err_clip if self.max == 0 else self.max
        data_index = self.data.store(transition)
        self.tree.add(data_index, priority)

    def sample(self, batch_size):
        indices = np.empty(batch_size, dtype=np.int32)
        priorities = np.empty(batch_size)
        segment_size = self.tree.total_priority / batch_size

        self.beta = min(1, self.beta + self.beta_inc)

        for i in range(batch_size):
            seg_start = segment_size * i
            seg_end = seg_start + segment_size
            priority = np.random.uniform(seg_start, seg_end)
            indices[i], priorities[i] = self.tree.get_leaf(priority)

        sampling_probabilities = priorities / self.tree.total_priority
        is_weights = np.power(sampling_probabilities * len(self.data), -self.beta)
        is_weights /= is_weights.max()

        update_fn = lambda td_err: self.update_weights(indices, td_err)
        return self.data.get_data(indices), is_weights, update_fn

    def update_weights(self, data_indices, errs):
        for i, err in zip(data_indices, errs):
            self.tree.update(i, self.get_priority(err))

    def get_priority(self, err):
        clipped = np.minimum(np.abs(err), self.err_clip)
        return np.power(clipped + self.eps, self.alpha)


class SumTree:
    def __init__(self, capacity):
        self.tree = np.zeros(2 * capacity - 1)
        self.capacity = capacity

    def add(self, data_index, priority):
        self.update(data_index, priority)

    def get_leaf(self, priority):
        curr = 0
        while curr < self.tree.size:
            left = 2 * curr + 1
            right = curr + 1

            if left >= self.tree.size:
                break

            if priority <= self.tree[left]:
                curr = left
            else:
                priority -= self.tree[left]
                curr = right

        data_index = curr - self.capacity + 1
        return data_index, self.tree[curr]

    def update(self, data_index, new_priority):
        tree_index = self.get_tree_index(data_index)
        change = new_priority - self.tree[tree_index]
        self.tree[tree_index] = new_priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_tree_index(self, data_index):
        return data_index + self.capacity - 1

    @property
    def total_priority(self):
        return self.tree[0]


if __name__ == '__main__':
    size = 8
    tree = SumTree(size)
    for i in range(2 * size):
        tree.add(i % size, i)

    print(tree.tree)
