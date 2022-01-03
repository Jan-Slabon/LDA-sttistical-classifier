import logistic_regresion as lr
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
data = [[1, 1, 1], [1, 1, 2], [1, 1.5, 1.5], [1, 2, 1], [1, 3, 0.5], [0, 1.5, 3.5], [0, 3.5, 3], [0, 2.5, 2.5],
        [0, 3, 1.5], [0, 4.5, 1]]
sb.set()
plt.scatter([x[1] for x in data[:5]], [x[2] for x in data[:5]], marker="x", color="black")
plt.scatter([x[1] for x in data[5:]], [x[2] for x in data[5:]], marker="o", color="green")
Result = lr.log_reg(data, 2)
Clasifier = Result[0]
probe = [[np.random.uniform(0, 4), np.random.uniform(0, 4)] for i in range(20)]
class1 = []
class0 = []
for x in probe:
    if Clasifier(x) >= 0.5:
        class1.append(x)
    else:
        class0.append(x)
plt.scatter([x[0] for x in class1], [x[1] for x in class1], marker="x", color="red")
plt.scatter([x[0] for x in class0], [x[1] for x in class0], marker="o", color="blue")
plt.show()
