from collections import deque
from Problem import AthletePerformanceProblem

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = 0 if parent is None else parent.depth + 1

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

class DFSSearch:
    def __init__(self, problem):
        self.problem = problem
        self.expanded_nodes = 0
        self.max_stack_size = 0

        # Set target day and performance (needed for is_goal)
        self.problem.target_day = 10
        self.problem.target_perf = 6.5
        self.problem.max_fatigue = 4
        self.problem.max_risk = 0.5

    def search(self, max_depth=10):
        start_node = Node(state=self.problem.initial_state)
        stack = deque([start_node])
        explored = set()

        while stack:
            current_node = stack.pop()

            day, fatigue, risk, performance, _ = current_node.state
            if day >= self.problem.target_day and performance >= self.problem.target_perf:
                print(f"Goal found! Day: {day}, Performance: {performance:.2f}, Fatigue: {fatigue:.2f}, Risk: {risk:.2f}")
                return current_node

            rounded_state = self._round_state(current_node.state)
            if rounded_state in explored:
                continue

            explored.add(rounded_state)

            for action in self.problem.actions():
                current_state = current_node.state
                new_state = self.problem.apply_action(current_state, action)

                if not self.is_valid(new_state):
                    continue

                child_node = Node(new_state, parent=current_node, action=action)

                if child_node.depth > max_depth:
                    continue

                stack.append(child_node)

            self.expanded_nodes += 1
            self.max_stack_size = max(self.max_stack_size, len(stack))
            
            # Progress indicator
            if self.expanded_nodes % 100 == 0:
                print(f"Explored {self.expanded_nodes} nodes, stack size: {len(stack)}, " 
                      f"Current state: Day {day}, F={fatigue:.2f}, R={risk:.2f}, P={performance:.2f}, Depth={current_node.depth}")

        return None

    def is_valid(self, state):
        _, fatigue, risk, _, _ = state
        return fatigue <= self.problem.max_fatigue and risk <= self.problem.max_risk

    def _round_state(self, state):
        day, fatigue, risk, performance, _ = state
        return (
            day,
            round(fatigue, 1),
            round(risk, 1),
            round(performance, 1)
        )

    def reconstruct_path(self, node):
        path = []
        while node and node.parent:
            path.append(node.action)
            node = node.parent
        return path[::-1]

def test_dfs_search():
    print("Testing DFS Algorithm")
    print("-----------------------------------------")

    problem = AthletePerformanceProblem(
        initial_state=(0, 1.5, 0.2, 6.0)
    )

    searcher = DFSSearch(problem)

    print("Starting search...")
    goal_node = searcher.search()
    print(f"Search completed. Nodes explored: {searcher.expanded_nodes}")

    if goal_node is None:
        print("No solution found.")
    else:
        path = searcher.reconstruct_path(goal_node)

        print("\nTraining Plan:")
        print("Day | Intensity | Duration | Fatigue | Risk | Performance")
        print("----|-----------|----------|---------|------|------------")

        state = problem.initial_state
        day = 0
        print(f"{day:3d} | {'-':9} | {'-':8} |  {state[1]:.2f}   | {state[2]:.2f} | {state[3]:.2f}")

        for action in path:
            state = problem.apply_action(state, action)
            day += 1
            intensity, duration = action
            print(f"{day:3d} | {intensity:9.1f} | {duration:8.1f} |  {state[1]:.2f}   | {state[2]:.2f} | {state[3]:.2f}")

        final_day, final_fatigue, final_risk, final_perf, _ = state
        print("\nFinal State:")
        print(f"Day: {final_day}")
        print(f"Fatigue: {final_fatigue:.2f}/5.00")
        print(f"Risk: {final_risk:.2f}/1.00")
        print(f"Performance: {final_perf:.2f}/10.00")

        if final_day >= searcher.problem.target_day and final_perf >= searcher.problem.target_perf:
            print("\nGoal achieved!")
        else:
            print("\nGoal not achieved.")

if __name__ == "__main__":
    import time

    start_time = time.time()
    test_dfs_search()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")