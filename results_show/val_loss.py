import matplotlib.pyplot as plt

# 数据
global_steps = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4030]
precision = [11.17, 47.09, 44.00, 45.90, 47.88, 45.69, 46.67, 47.61, 48.21]
recall = [33.33, 46.34, 43.86, 46.40, 46.40, 45.42, 46.43, 46.93, 47.43]
f1 = [16.73, 42.49, 41.28, 44.53, 45.13, 44.77, 45.79, 46.47, 46.92]
loss = [101.05, 99.01, 97.96, 95.94, 97.99, 98.07, 96.13, 97.01, 97.00]

# 创建图形
fig, ax1 = plt.subplots()

# 画出 precision, recall, f1
ax1.set_xlabel('Global Step')
ax1.set_ylabel('Precision, Recall, F1 Score', color='tab:blue')
ax1.plot(global_steps, precision, 'o-', label='Precision', color='tab:blue')
ax1.plot(global_steps, recall, 's-', label='Recall', color='tab:orange')
ax1.plot(global_steps, f1, '^-', label='F1 Score', color='tab:green')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')

# 创建第二个y轴，画出loss
ax2 = ax1.twinx()
ax2.set_ylabel('Loss', color='tab:red')
ax2.plot(global_steps, loss, 'd-', label='Loss', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc='upper right')

fig.tight_layout()
plt.title('Precision, Recall, F1 Score and Loss over Global Steps')
plt.show()
