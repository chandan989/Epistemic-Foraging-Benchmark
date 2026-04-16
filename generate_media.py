import networkx as nx
import matplotlib.pyplot as plt
import os
from epistemic_foraging import BlackBoxDiagnostic

os.makedirs('media', exist_ok=True)

# 1. Generate Topology Graph
env = BlackBoxDiagnostic(seed=42)
G = nx.DiGraph()

# Add nodes
for c in env.core_hubs:
    G.add_node(c, layer=0)
for s in env.subnet_switches:
    G.add_node(s, layer=1)
for e in env.endpoints:
    G.add_node(e, layer=2)

# Add edges
for node, children in env.graph.items():
    for child in children:
        G.add_edge(node, child)

pos = nx.multipartite_layout(G, subset_key="layer")

# Customize appearance
plt.figure(figsize=(12, 8))
node_colors = []
for node in G.nodes():
    if node in env.core_hubs:
        node_colors.append('#ff9999')
    elif node in env.subnet_switches:
        node_colors.append('#66b3ff')
    elif node == env.anomalous_node:
        node_colors.append('#ff4d4d') # Red for anomaly
    else:
        node_colors.append('#99ff99')

nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1500, font_size=12, font_weight='bold', arrows=True, arrowsize=20)
plt.title("Black Box Diagnostic DAG Topology\nRed Node = Anomaly", fontsize=16)

# Add a legend
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='#ff9999', label='Core Hubs (Layer 1)')
blue_patch = mpatches.Patch(color='#66b3ff', label='Subnet Switches (Layer 2)')
green_patch = mpatches.Patch(color='#99ff99', label='Endpoints (Layer 3)')
anomaly_patch = mpatches.Patch(color='#ff4d4d', label='Anomaly')
plt.legend(handles=[red_patch, blue_patch, green_patch, anomaly_patch], loc='lower right')

plt.tight_layout()
plt.savefig('media/topology.png', dpi=300)
plt.close()

# 2. Generate Flowchart/Bar Chart for Strategies
strategies = ['Systematic Deduction', 'Brute Force / Retrieval', 'Hallucination']
scores = [6, 0, 0] # Example scores
turns = [4, 10, 0] # Example turns

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Reasoning Profile', fontsize=12)
ax1.set_ylabel('Efficiency Score (Max 10)', color=color, fontsize=12)
bars = ax1.bar(strategies, scores, color=color, alpha=0.7, label='Score')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(-1, 10)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Turns Taken', color=color, fontsize=12)
line = ax2.plot(strategies, turns, color=color, marker='o', linewidth=2, label='Turns Taken')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(-1, 11)

plt.title('Performance Comparison: Active vs Passive Strategies', fontsize=14)

# Add text labels on bars
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval}', va='bottom', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('media/performance_comparison.png', dpi=300)
plt.close()

print("Media generated successfully.")
