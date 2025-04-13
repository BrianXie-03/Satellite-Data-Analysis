import matplotlib.pyplot as plt

# Data dictionary
data = {
    "Good": {"Reference": 622493, "New": 961497, "Matching": 622469},
    "Snow": {"Reference": 0, "New": 0, "Matching": 0},
    "Heavy Aerosol": {"Reference": 22936, "New": 178330, "Matching": 22936}
}

# Extract categories and subcategories
categories = list(data.keys())  # ["Good", "Snow", "Heavy Aerosol"]
subcategories = list(data["Good"].keys())  # ["Reference", "New", "Matching"]
colors = {"Reference": "blue", "New": "red", "Matching": "green"}
markers = {"Reference": "o", "New": "s", "Matching": "^"}

# Assign x-axis positions for each category
x_positions = range(len(categories))

# Plot scatter points
fig, ax = plt.subplots(figsize=(8, 5))
for sub in subcategories:
    y_values = [data[cat][sub] for cat in categories]
    ax.scatter(x_positions, y_values, label=sub, color=colors[sub], marker=markers[sub], s=100)

# Formatting
ax.set_xticks(x_positions)
ax.set_xticklabels(categories)
ax.set_ylabel("Pixel Count")
ax.set_title("Scatter Plot of Reference, New, and Matching Values")
ax.legend(title="Legend")

plt.show()