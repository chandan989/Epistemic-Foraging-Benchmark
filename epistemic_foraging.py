import random
import re
import json

class BlackBoxDiagnostic:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.max_turns = 10
        self.turns_taken = 0
        self.score = 0
        self.max_score = 10
        self.transcript = []
        self.profile = None

        self.core_hubs = ['C1', 'C2']
        self.subnet_switches = ['S1', 'S2', 'S3', 'S4']
        self.endpoints = [f'E{i}' for i in range(1, 15)]

        self.nodes = self.core_hubs + self.subnet_switches + self.endpoints
        self.graph = {node: [] for node in self.nodes}
        self._generate_topology()

        self.anomalous_node = random.choice(self.nodes)

    def _generate_topology(self):
        # Connect C1, C2 to S1..S4
        for s in self.subnet_switches:
            parent = random.choice(self.core_hubs)
            self.graph[parent].append(s)

        # Ensure every Core Hub has at least one Subnet Switch (optional but good practice)
        for c in self.core_hubs:
            if not self.graph[c]:
                s = random.choice(self.subnet_switches)
                # Move s to c
                for c_other in self.core_hubs:
                    if s in self.graph[c_other]:
                        self.graph[c_other].remove(s)
                self.graph[c].append(s)

        # Connect Endpoints to S1..S4
        for e in self.endpoints:
            parent = random.choice(self.subnet_switches)
            self.graph[parent].append(e)

    def _get_downstream(self, node):
        downstream = set([node])
        queue = [node]
        while queue:
            current = queue.pop(0)
            for child in self.graph[current]:
                if child not in downstream:
                    downstream.add(child)
                    queue.append(child)
        return downstream

    def _get_path_between(self, n1, n2):
        # Returns nodes on the path between n1 and n2 if it exists, else empty
        # Directed graph: check n1 -> n2 and n2 -> n1
        for start, end in [(n1, n2), (n2, n1)]:
            path = self._bfs_path(start, end)
            if path:
                return path
        return []

    def _bfs_path(self, start, target):
        queue = [[start]]
        visited = set([start])
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node == target:
                return path
            for child in self.graph[node]:
                if child not in visited:
                    visited.add(child)
                    queue.append(path + [child])
        return []

    def _get_action_cost(self, action_dict):
        if 'solution' in action_dict:
            return 1 # Final action, cost doesn't matter much as it terminates
        if action_dict.get('query_type') == 'check_status':
            target = action_dict.get('target')
            if target in self.endpoints:
                return 3
            return 1
        if action_dict.get('query_type') == 'check_connection':
            t1 = action_dict.get('target_1')
            t2 = action_dict.get('target_2')
            if t1 in self.endpoints or t2 in self.endpoints:
                return 3
            return 1
        return 1

    def step(self, response_text):
        if self.turns_taken >= self.max_turns:
            return {"status": "error", "message": "Max turns exceeded."}

        self.transcript.append({"role": "model", "text": response_text})

        # Parse JSON
        match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if not match:
            self.turns_taken += 1
            msg = "Fail: No valid JSON action found."
            self.transcript.append({"role": "environment", "text": msg})
            return {"status": "error", "message": msg}

        try:
            action = json.loads(match.group(0))
        except json.JSONDecodeError:
            self.turns_taken += 1
            msg = "Fail: Invalid JSON formatting."
            self.transcript.append({"role": "environment", "text": msg})
            return {"status": "error", "message": msg}

        cost = self._get_action_cost(action)
        self.turns_taken += cost

        if self.turns_taken > self.max_turns:
            self.profile = 'brute_force'
            msg = "Fail: Turn limit exceeded."
            self.transcript.append({"role": "environment", "text": msg})
            return {"status": "terminated", "result": "fail"}

        if 'solution' in action:
            solution = action['solution']
            if solution == self.anomalous_node:
                self.score = self.max_turns - self.turns_taken
                if self.turns_taken <= 5:
                    self.profile = 'deductive'
                else:
                    self.profile = 'brute_force'
                msg = f"Pass: Correct node identified. Score: {self.score}"
                self.transcript.append({"role": "environment", "text": msg})
                return {"status": "terminated", "result": "pass"}
            else:
                self.score = 0
                self.profile = 'hallucination'
                msg = "Fail: Incorrect node."
                self.transcript.append({"role": "environment", "text": msg})
                return {"status": "terminated", "result": "fail"}

        elif action.get('query_type') == 'check_status':
            target = action.get('target')
            if target not in self.nodes:
                msg = f"Error: Node {target} does not exist."
            else:
                downstream = self._get_downstream(target)
                result_bool = self.anomalous_node in downstream
                msg = f"Result: Anomaly downstream of {target}? {result_bool}"

        elif action.get('query_type') == 'check_connection':
            t1 = action.get('target_1')
            t2 = action.get('target_2')
            if t1 not in self.nodes or t2 not in self.nodes:
                msg = f"Error: Invalid nodes provided."
            else:
                path = self._get_path_between(t1, t2)
                if not path:
                    msg = "Result: No direct topological path exists between these nodes."
                else:
                    result_bool = self.anomalous_node in path
                    msg = f"Result: Anomaly on path? {result_bool}"
        else:
            msg = "Error: Unknown action."

        self.transcript.append({"role": "environment", "text": msg})
        return {"status": "active", "message": msg}

    def run(self, mock_agent=None):
        # A simple run loop to demonstrate interaction
        # If mock_agent is provided, it's a function that takes transcript and returns text
        if not mock_agent:
            return self

        while self.turns_taken < self.max_turns:
            response = mock_agent(self.transcript)
            result = self.step(response)
            if result['status'] == 'terminated':
                break

        if self.profile is None:
            self.profile = 'brute_force' # Ran out of turns

        return self

if __name__ == "__main__":
    env = BlackBoxDiagnostic(seed=42)
    print("Graph:", env.graph)
    print("Anomaly:", env.anomalous_node)
