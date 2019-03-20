import collections
import matplotlib.pyplot as plt
import networkx as nx


def Average(lst):
    return sum(lst) / len(lst)


text_file = open("Enron_Output10.txt", "r")
# text_file = open("test1.txt", "r")
lines = text_file.read().split(' ')
lines = map(lambda deg: deg.replace("\n", ""), lines)
lines = list(filter(lambda deg: deg != "", lines))
degrees_int = list(map(lambda deg: int(deg), lines))

print(degrees_int[0:10])
degree_sequence = lines
degree_sequence.reverse()

# print("Degree sequence", degree_sequence[0:10])
print("AVG:", Average(degrees_int))
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
deg = deg[99:]
cnt = cnt[99:]
fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color='b')

plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")

ax.set_xticks([deg[0], deg[int(len(deg)/2)], deg[len(deg)-1]])
ax.set_xticklabels([deg[0], deg[int(len(deg)/2)], deg[len(deg)-1]])

# draw graph in inset
plt.axes([1, 1, 10, 20])
# Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
# pos = nx.spring_layout(G)
plt.axis('off')
# nx.draw_networkx_nodes(G, pos, node_size=20)
# nx.draw_networkx_edges(G, pos, alpha=0.4)

plt.savefig('degree_histogram_enron_k10.png')
plt.show()
