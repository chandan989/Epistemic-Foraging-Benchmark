import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

nb.cells.extend([
    new_markdown_cell("# Epistemic-Foraging-Benchmark\n\nA Measurement of Epistemic Foraging and Executive Function in Frontier Models. This notebook demonstrates the environment, evaluation metric, and example strategies."),

    new_markdown_cell("## 1. Environment Code\nBelow is the implementation of the `BlackBoxDiagnostic` environment."),
    new_code_cell("""import random
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
        for s in self.subnet_switches:
            parent = random.choice(self.core_hubs)
            self.graph[parent].append(s)

        for c in self.core_hubs:
            if not self.graph[c]:
                s = random.choice(self.subnet_switches)
                for c_other in self.core_hubs:
                    if s in self.graph[c_other]:
                        self.graph[c_other].remove(s)
                self.graph[c].append(s)

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
            return 1
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
        return {"status": "active", "message": msg}"""),

    new_markdown_cell("## 2. Visualization\nLet's generate the topology diagram to understand the DAG."),
    new_code_cell("""import networkx as nx
import matplotlib.pyplot as plt

env = BlackBoxDiagnostic(seed=42)
G = nx.DiGraph()

for c in env.core_hubs: G.add_node(c, layer=0)
for s in env.subnet_switches: G.add_node(s, layer=1)
for e in env.endpoints: G.add_node(e, layer=2)

for node, children in env.graph.items():
    for child in children:
        G.add_edge(node, child)

pos = nx.multipartite_layout(G, subset_key="layer")

plt.figure(figsize=(10, 6))
node_colors = ['#ff9999' if n in env.core_hubs else '#66b3ff' if n in env.subnet_switches else '#ff4d4d' if n == env.anomalous_node else '#99ff99' for n in G.nodes()]

nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1500, font_size=12, arrows=True)
plt.title(f"Black Box Diagnostic DAG Topology\\nAnomaly at: {env.anomalous_node}")
plt.show()"""),

    new_markdown_cell("## 3. Mock Runs\nWe will simulate two types of models to show how scoring works: a **deductive** model that queries from top-down, and a **brute-force** model that randomly guesses endpoints."),
    new_code_cell("""print("===== DEDUCTIVE MODEL MOCK RUN =====")
env_deductive = BlackBoxDiagnostic(seed=42)
# Anomaly is E9 in seed 42

print("Turn 1: Check Core Hub 1")
print(env_deductive.step('{"query_type": "check_status", "target": "C1"}'))

print("\\nTurn 2: Check Subnet Switch 2")
print(env_deductive.step('{"query_type": "check_status", "target": "S2"}'))

print("\\nTurn 3: Check Endpoint 9 directly")
print(env_deductive.step('{"query_type": "check_status", "target": "E9"}'))

print("\\nTurn 4: Declare Solution")
print(env_deductive.step('{"solution": "E9"}'))

print(f"\\nFinal Profile: {env_deductive.profile}")
print(f"Final Score: {env_deductive.score}")
print(f"Turns taken: {env_deductive.turns_taken}")"""),

    new_code_cell("""print("===== BRUTE FORCE MODEL MOCK RUN =====")
env_brute = BlackBoxDiagnostic(seed=42)

print("Turn 1: Check E1")
print(env_brute.step('{"query_type": "check_status", "target": "E1"}'))

print("\\nTurn 2: Check E2")
print(env_brute.step('{"query_type": "check_status", "target": "E2"}'))

print("\\nTurn 3: Check E3")
print(env_brute.step('{"query_type": "check_status", "target": "E3"}'))

print("\\nTurn 4: Check E4")
print(env_brute.step('{"query_type": "check_status", "target": "E4"}'))

print(f"\\nFinal Profile: {env_brute.profile}")
print(f"Final Score: {env_brute.score}")
print(f"Turns taken: {env_brute.turns_taken}")""")
])

with open('benchmark_notebook.ipynb', 'w') as f:
    nbformat.write(nb, f)
print("Notebook generated successfully.")
