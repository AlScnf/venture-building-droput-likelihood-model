import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load updated dataset
df = pd.read_csv("oof_predictions_with_updated_segments.csv")

# Set style
sns.set(style="whitegrid", font_scale=1.25)
plt.figure(figsize=(11, 8))

# Scatterplot: dropout probability vs friction index, colored by segment
ax = sns.scatterplot(
    data=df,
    x="dropout_proba",
    y="friction_index",
    hue="segment",
    palette="RdYlGn_r",       # red = high risk, green = low risk
    edgecolor="black",
    s=110,
    alpha=0.85
)

# Draw quadrant threshold lines
plt.axvline(x=0.6, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
plt.axhline(y=8.0, color="black", linestyle="--", linewidth=1.2, alpha=0.6)

# Refined quadrant labels with more constructive wording
plt.text(0.13, 15.5, "Ideal Founders", fontsize=11, weight="bold", color="green")
plt.text(0.63, 15.5, "High Priority Follow-Up", fontsize=11, weight="bold", color="red")
plt.text(0.63, 2.0, "Low-touch / Observe", fontsize=11, weight="bold", color="gray")
plt.text(0.13, 2.0, "Worth Nudging", fontsize=11, weight="bold", color="orange")

# Axis and title
plt.xlabel("Predicted Dropout Likelihood", fontsize=13)
plt.ylabel("Candidate Value (Friction Index)", fontsize=13)
plt.title("Strategic Segmentation of Venture Applicants", fontsize=16, pad=20)

# Legend formatting
plt.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

# Output
plt.tight_layout()
plt.savefig("strategic_segmentation_matrix_final.png", dpi=300)
plt.show()
