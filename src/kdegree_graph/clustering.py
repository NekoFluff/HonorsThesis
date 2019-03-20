import collections
import matplotlib.pyplot as plt
import networkx as nx


fname = "edge_lists/Enron_Output20.txt"
# fname = "edge_lists/test1.txt"
with open(fname) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
lines = [x.strip() for x in content]
lines = list(map(lambda deg: deg.split(' '), lines))

for i in range(len(lines)):
    lines[i] = list(filter(lambda deg: deg != "" and deg != ":", lines[i]))
    lines[i] = list(map(lambda deg: deg.replace("[", ""), lines[i]))
    lines[i] = list(map(lambda deg: deg.replace("]", ""), lines[i]))

# degrees_int = list(map(lambda deg: int(deg), lines))
# print(lines)

#G = nx.complete_graph(5)
G = nx.Graph()
for line in lines:
    G.add_node(line[0])
    for i in range(1, len(line)):
        G.add_edge(line[0], line[i])
        # G.add_edge(line[i], line[0])

# print(nx.clustering(G, 0))
clustering_coefficient = nx.clustering(G)


def Average(lst):
    return sum(lst) / len(lst)


avg_cc = Average(list(clustering_coefficient.values()))
print("Average CC: ", avg_cc)
