import matplotlib.pyplot as plt
import numpy as np
C = [[1784, 103, 384, 1, 8,],
                 [28, 14, 27, 0, 3,],
                 [67, 69, 884, 44, 96,],
                 [1, 2, 19, 22, 6,],
                 [0, 1, 30, 4, 162,]]
C = np.array(C)
plt.matshow(C, cmap=plt.cm.Reds) # 根据最下面的图按自己需求更改颜色
# plt.colorbar()

for i in range(len(C)):
    for j in range(len(C)):
        plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
