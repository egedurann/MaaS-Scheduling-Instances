import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Read the Excel file
# df = pd.read_excel("all_methods_obj.xlsx",sheet_name='Sheet1')  # replace with your filename
#
# # Assume first column is X-axis
# x = df.iloc[:, 0]
#
# # Color + marker combinations for clarity
# colors = ['C0', 'C1', 'C2', 'C3', 'C4']  # Matplotlib default color cycle
# markers = ['o', 's', '^', 'D', 'v']  # circle, square, triangle_up, diamond, triangle_down

plt.figure(figsize=(8, 6))

# for i, col in enumerate(df.columns[1:]):
#     plt.plot(
#         x, df[col],
#         marker=markers[i % len(markers)],
#         color=colors[i % len(colors)],
#         label=col,
#         linewidth=1.5,
#         markersize=7
#     )
#
#
# plt.ylabel("Runtime (sec)")
# plt.xlabel("Instances")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

data=pd.read_excel("all_methods_obj.xlsx",sheet_name='Benders')

sns.catplot(
    data=data, y="LBBD Methods", hue="LBBD Methods", kind="count",
    palette="pastel", edgecolor=".6",
)


sns.catplot(data=data, x="Instances", y="LBBD Methods")
plt.show()

sns.catplot(
    data=data, y="LBBD Methods", hue="Instance Size", kind="count",
    palette="pastel", edgecolor=".6",
)
plt.show()

CP=pd.read_excel("all_methods_obj.xlsx",sheet_name='CP')
sns.catplot(data=CP, x="CP Search Phase", y="Instances", hue="Instance Size", kind="swarm",palette="pastel", edgecolor=".8")
plt.show()

methods=pd.read_excel("all_methods_obj.xlsx",sheet_name='comparison')
sns.catplot(data=methods, x="Instances", y="Methods", hue="Instance Size", kind="swarm",palette="pastel", edgecolor=".8")
plt.show()