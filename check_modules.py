import matplotlib
matplotlib.use("Agg")  # ensures Matplotlib can save files without GUI
import matplotlib.pyplot as plt
import os

# Print current working directory
print("Current working directory:", os.getcwd())

# Try saving a very simple plot
plt.plot([1, 2, 3], [4, 5, 6])
test_file = os.path.abspath("test_plot.png")
plt.savefig(test_file)
plt.close()

print("Saved test plot at:", test_file)

# Check if file exists
if os.path.exists(test_file):
    print("✅ File successfully saved!")
else:
    print("❌ File was NOT saved.")
