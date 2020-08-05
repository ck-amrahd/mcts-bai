import numpy as np

number_of_leaves = 9
epsilon = 0.15
delta = 0.15
uniform_samples = 500
iterations = 50


def beta(s):
    return np.log(number_of_leaves / delta) + np.log(np.log(s) + 1)


class Node:
    def __init__(self, name, left_child, middle_child, right_child, parent, is_root=False, is_max_node=False):
        self.left_child = left_child
        self.middle_child = middle_child
        self.right_child = right_child
        self.parent = parent
        self.is_root = is_root
        self.is_max_node = is_max_node
        self.name = name

    def update_left_child(self, left_child):
        self.left_child = left_child

    def update_middle_child(self, middle_child):
        self.middle_child = middle_child

    def update_right_child(self, right_child):
        self.right_child = right_child

    def value(self):
        values = []
        values.append(self.left_child.empirical_mean())
        values.append(self.middle_child.empirical_mean())
        values.append(self.right_child.empirical_mean())
        values.sort()
        return min(values)

    def representative_node(self):
        left_value = self.left_child.value()
        middle_value = self.middle_child.value()
        right_value = self.right_child.value()

        if left_value >= middle_value and left_value >= right_value:
            return self.left_child
        elif middle_value >= left_value and middle_value >= right_value:
            return self.middle_child
        else:
            return self.right_child

    def representative_leaf(self):
        # For our case the node is the min node - so let's have a fixed rule
        left_lower, _ = self.left_child.confidence_interval()
        middle_lower, _ = self.middle_child.confidence_interval()
        right_lower, _ = self.right_child.confidence_interval()
        if left_lower <= middle_lower and left_lower <= middle_lower:
            return self.left_child
        elif middle_lower <= left_lower and middle_lower <= right_lower:
            return self.middle_child
        else:
            return self.right_child

    def confidence_interval(self):
        lowers = []
        uppers = []

        left_lower, left_upper = self.left_child.confidence_interval()
        lowers.append(left_lower)
        uppers.append(left_upper)

        middle_lower, middle_upper = self.middle_child.confidence_interval()
        lowers.append(middle_lower)
        uppers.append(middle_upper)

        right_lower, right_upper = self.right_child.confidence_interval()
        lowers.append(right_lower)
        uppers.append(right_upper)

        lowers.sort()
        uppers.sort()
        return [min(lowers), min(uppers)]

    def find_best_arm_and_competitor_by_value(self):
        b_c = {}
        left_lower, left_upper = self.left_child.confidence_interval()
        b_c[self.left_child] = self.left_child.value() + np.sqrt(
            2 * (left_upper - left_lower) / self.left_child.representative_leaf().number_sampled)
        middle_lower, middle_upper = self.middle_child.confidence_interval()
        b_c[self.middle_child] = self.middle_child.value() + np.sqrt(
            2 * (middle_upper - middle_lower) / self.middle_child.representative_leaf().number_sampled)
        right_lower, right_upper = self.right_child.confidence_interval()
        b_c[self.right_child] = self.right_child.value() + np.sqrt(
            2 * (right_upper - right_lower) / self.right_child.representative_leaf().number_sampled)

        best_arm, competitor = [k for k, v in sorted(b_c.items(), key=lambda item: item[1], reverse=True)][:2]
        return best_arm, competitor

    def find_best_arm_and_competitor(self):
        left_lower, left_upper = self.left_child.confidence_interval()
        middle_lower, middle_upper = self.middle_child.confidence_interval()
        right_lower, right_upper = self.right_child.confidence_interval()

        B_left_middle = middle_upper - left_lower
        B_left_right = right_upper = left_lower
        B_left = None
        if B_left_middle > B_left_right:
            B_left = B_left_middle
        else:
            B_left = B_left_right

        B_middle_left = left_upper - middle_lower
        B_middle_right = right_upper - middle_lower
        B_middle = None
        if B_middle_left > B_middle_right:
            B_middle = B_middle_left
        else:
            B_middle = B_middle_right

        B_right_left = left_upper - right_lower
        B_right_middle = middle_upper - right_lower
        B_right = None
        if B_right_left > B_right_middle:
            B_right = B_right_left
        else:
            B_right = B_right_middle

        b_t = None
        c_t_list = []
        if B_left < B_middle and B_left < B_right:
            b_t = self.left_child
            c_t_list.append(self.middle_child)
            c_t_list.append(self.right_child)
        elif B_middle < B_left and B_middle < B_right:
            b_t = self.middle_child
            c_t_list.append(self.left_child)
            c_t_list.append(self.right_child)
        else:
            b_t = self.right_child
            c_t_list.append(self.left_child)
            c_t_list.append(self.middle_child)

        c_t = None
        _, ub_first = c_t_list[0].confidence_interval()
        _, ub_second = c_t_list[1].confidence_interval()

        if ub_first > ub_second:
            c_t = c_t_list[0]
        else:
            c_t = c_t_list[1]

        return b_t, c_t


class LeafNode:
    def __init__(self, name, parent, true_mean_val):
        self.parent = parent
        self.true_mean_val = true_mean_val
        self.values = []
        self.number_sampled = 0
        self.name = name

    def sample(self):
        random_value = np.random.normal(self.true_mean_val, 0.2)
        self.values.append(random_value)
        self.number_sampled += 1
        return random_value

    def empirical_mean(self):
        return np.mean(self.values)

    def lower_bound(self):
        confidence_gap = np.sqrt(beta(self.number_sampled) / self.number_sampled)
        emp_mean = self.empirical_mean()
        return emp_mean - confidence_gap

    def upper_bound(self):
        confidence_gap = np.sqrt(beta(self.number_sampled) / self.number_sampled)
        emp_mean = self.empirical_mean()
        return emp_mean + confidence_gap

    def confidence_interval(self):
        lower_bound = self.lower_bound()
        upper_bound = self.upper_bound()
        return [lower_bound, upper_bound]

    def true_mean(self):
        return self.true_mean_val

    def sample_complexity(self):
        return self.number_sampled

    def reset(self):
        self.number_sampled = 0
        self.values = []


leaves = []

root = Node('root', None, None, None, None, is_root=True, is_max_node=True)

leaf_1 = LeafNode('leaf_1', root, 0.45)
leaves.append(leaf_1)
leaf_2 = LeafNode('leaf_2', root, 0.50)
leaves.append(leaf_2)
leaf_3 = LeafNode('leaf_3', root, 0.55)
leaves.append(leaf_3)
node_left = Node('node_left', leaf_1, leaf_2, leaf_3, root, is_root=False, is_max_node=False)
root.update_left_child(node_left)

leaf_4 = LeafNode('leaf_4', root, 0.35)
leaves.append(leaf_4)
leaf_5 = LeafNode('leaf_5', root, 0.40)
leaves.append(leaf_5)
leaf_6 = LeafNode('leaf_6', root, 0.60)
leaves.append(leaf_6)
node_middle = Node('node_middle', leaf_4, leaf_5, leaf_6, root, is_root=False, is_max_node=False)
root.update_middle_child(node_middle)

leaf_7 = LeafNode('leaf_7', root, 0.30)
leaves.append(leaf_7)
leaf_8 = LeafNode('leaf_8', root, 0.47)
leaves.append(leaf_8)
leaf_9 = LeafNode('leaf_9', root, 0.52)
leaves.append(leaf_9)
node_right = Node('node_right', leaf_7, leaf_8, leaf_9, root, is_root=False, is_max_node=False)
root.update_right_child(node_right)


def total_sample_complexity():
    samples = 0
    for leaf in leaves:
        samples += leaf.sample_complexity()

    return samples


def get_best_arm():
    node_left_vlaue = node_left.value()
    node_middle_value = node_middle.value()
    node_right_value = node_right.value()

    if node_left_vlaue >= node_middle_value and node_left_vlaue >= node_right_value:
        return 'node_left'
    elif node_middle_value >= node_left_vlaue and node_middle_value >= node_right_value:
        return 'node_middle'
    else:
        return 'node_right'


def root_value():
    values = []
    values.append(root.left_child.value())
    values.append(root.middle_child.value())
    values.append(root.right_child.value())
    # root is a max node
    values.sort()
    return max(values)


def reset():
    leaf_1.reset()
    leaf_2.reset()
    leaf_3.reset()
    leaf_4.reset()
    leaf_5.reset()
    leaf_6.reset()
    leaf_7.reset()
    leaf_8.reset()
    leaf_9.reset()


# MCTS-BAI
print('Running MCTS-BAI ... ')
sample_complexity_list = []
best_arm_list = []
root_values_list = []

for i in range(iterations):
    reset()
    leaf_1.sample()
    leaf_2.sample()
    leaf_3.sample()
    leaf_4.sample()
    leaf_5.sample()
    leaf_6.sample()
    leaf_7.sample()
    leaf_8.sample()
    leaf_9.sample()

    while True:
        b_t, c_t = root.find_best_arm_and_competitor()
        b_lower, b_upper = b_t.confidence_interval()
        b_diff = b_upper - b_lower
        c_lower, c_upper = c_t.confidence_interval()
        c_diff = c_upper - c_lower

        if c_upper - b_lower <= epsilon:
            break

        next_move = None
        if b_diff > c_diff:
            next_move = b_t
        else:
            next_move = c_t

        representative_leaf = next_move.representative_leaf()
        representative_leaf.sample()

    samples = total_sample_complexity()
    best_arm = get_best_arm()
    sample_complexity_list.append(samples)
    best_arm_list.append(best_arm)
    root_values_list.append(root_value())
averate_complexity = 0
for sample in sample_complexity_list:
    averate_complexity += sample
averate_complexity = averate_complexity / iterations

average_root_value = 0
for root_val in root_values_list:
    average_root_value += root_val
average_root_value = average_root_value / iterations

percentage_correct = None
average_best_arm = None
count_left = 0
count_middle = 0
count_right = 0

for b_arm in best_arm_list:
    if b_arm == 'node_left':
        count_left += 1
    elif b_arm == 'node_middle':
        count_middle += 1
    else:
        count_right += 1

if count_left > count_middle and count_left > count_right:
    average_best_arm = 'node_left'
    percentage_correct = (count_left / iterations) * 100
elif count_middle > count_left and count_middle > count_right:
    average_best_arm = 'node_middle'
    percentage_correct = (count_middle / iterations) * 100
else:
    average_best_arm = 'node_right'
    percentage_correct = (count_right / iterations) * 100

print('MCTS-BAI')
print(f'Average Best Arm: {average_best_arm}')
print(f'Average Complexity: {averate_complexity}')
print(f'Average Root Value: {round(average_root_value, 2)}')
print(f'Percentage Correct: {percentage_correct} %')

# MCTS-BAI - Dharma
print('\n')
print('Running MCTS-BAI-Dharma ... ')
sample_complexity_list = []
best_arm_list = []
root_values_list = []
for i in range(iterations):
    reset()

    leaf_1.sample()
    leaf_2.sample()
    leaf_3.sample()
    leaf_4.sample()
    leaf_5.sample()
    leaf_6.sample()
    leaf_7.sample()
    leaf_8.sample()
    leaf_9.sample()

    while True:
        b_t, c_t = root.find_best_arm_and_competitor_by_value()
        b_lower, b_upper = b_t.confidence_interval()
        b_diff = b_upper - b_lower
        c_lower, c_upper = c_t.confidence_interval()
        c_diff = c_upper - c_lower

        if c_upper - b_lower <= epsilon:
            break

        next_move = None
        if b_diff > c_diff:
            next_move = b_t
        else:
            next_move = c_t

        representative_leaf = next_move.representative_leaf()
        representative_leaf.sample()

    samples = total_sample_complexity()
    best_arm = get_best_arm()
    sample_complexity_list.append(samples)
    best_arm_list.append(best_arm)
    root_values_list.append(root_value())

averate_complexity = 0
for sample in sample_complexity_list:
    averate_complexity += sample
averate_complexity = averate_complexity / iterations

average_root_value = 0
for root_val in root_values_list:
    average_root_value += root_val
average_root_value = average_root_value / iterations

percentage_correct = None
average_best_arm = None
count_left = 0
count_middle = 0
count_right = 0

for b_arm in best_arm_list:
    if b_arm == 'node_left':
        count_left += 1
    elif b_arm == 'node_middle':
        count_middle += 1
    else:
        count_right += 1

if count_left > count_middle and count_left > count_right:
    average_best_arm = 'node_left'
    percentage_correct = (count_left / iterations) * 100
elif count_middle > count_left and count_middle > count_right:
    average_best_arm = 'node_middle'
    percentage_correct = (count_middle / iterations) * 100
else:
    average_best_arm = 'node_right'
    percentage_correct = (count_right / iterations) * 100

print('MCTS-BAI-Dharma')
print(f'Average Best Arm: {average_best_arm}')
print(f'Average Complexity: {averate_complexity}')
print(f'Average Root Value: {round(average_root_value, 2)}')
print(f'Percentage Correct: {percentage_correct} %')

# Uniform-Sampling
reset()
for i in range(uniform_samples):
    leaf_1.sample()
    leaf_2.sample()
    leaf_3.sample()
    leaf_4.sample()
    leaf_5.sample()
    leaf_6.sample()
    leaf_7.sample()
    leaf_8.sample()
    leaf_9.sample()

samples = total_sample_complexity()
best_arm = get_best_arm()

print('\n')
print('Uniform Sampling')
print(f'Best Arm: {best_arm}')
print(f'Samples: {samples}')
print(f'Value at root: {round(root_value(), 2)}')
