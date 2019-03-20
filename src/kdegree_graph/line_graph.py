import numpy as np
import matplotlib.pyplot as plt

# k=20 -> Avg cc = 0.5954908740464774
# k=15 -> Avg cc = 0.5596959244975125
# k=10 -> Avg cc = 0.5230318171147785
# k=5 -> Avg cc = 0.5097620404363641
# k=1 -> Avg cc = 0.49698255959950277
x = [1, 5, 10, 15, 20]
y = [0.49698255959950277, 0.5097620404363641,
     0.5230318171147785, 0.5596959244975125, 0.5954908740464774]


plt.figure(1)
plt.plot(x, y, 'bo', x, y, 'k')
plt.title("CC vs k-degree")
plt.ylabel("clustering coefficient")
plt.xlabel("k-degree")
plt.gca().set_ylim([0.4, 0.7])
plt.gca().set_xticks(x)

plt.gca().set_xticklabels(x)
plt.savefig('cc_vs_k-deg.png')
plt.show()
