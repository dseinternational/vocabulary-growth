# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dse-vocab-growth
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Vocabulary data variables

# %% [markdown]
# ## Preparation

# %%
# %config InlineBackend.figure_format = "retina"

# %%
import os

import dse_research_utils.environment.setup as setup
import dse_research_utils.metadata.packages as package_metadata
import dse_research_utils.plot.styles as plot_styles
import dse_research_utils.statistics.descriptive as descriptive_stats
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

setup.init_workbook()

# %%
WORKBOOK = "001-vocab-data-variables"
OUTPUT_DIR = f"../output/notebooks/{WORKBOOK}"
REPORT_FIGS_DIR = f"../docs/report/figures/{WORKBOOK}"
SAVE_PLOTS = True

RANDOM_SEED = 47
np.random.seed(RANDOM_SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_FIGS_DIR, exist_ok=True)

print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print()

package_list = [
    "duckdb",
    "matplotlib",
    "numpy",
    "pandas",
]

print()

package_metadata.report_package_versions(package_list)


# %% [markdown]
# ### Helper functions

# %%
def plot_histogram(data : pd.DataFrame, x_name: str, x_label: str, output_filename: str | None = None):
    plt.figure(figsize=(6, 4))
    sns.histplot(data, x=x_name, bins=10, kde=True)
    median = data[x_name].median()
    mean = data[x_name].mean()
    std = data[x_name].std()
    plt.axvline(median, color="red", label=f"Median: {median:.1f}")
    plt.axvline(mean, color="#0F447A", linestyle="--", label=f"Mean: {mean:.1f}")
    # shade mean +/- 1 std
    plt.fill_betweenx(y=[0, plt.gca().get_ylim()[1]], x1=mean - std, x2=mean + std, color="#0F447A", alpha=0.05, label=f"Mean ± 1 Std Dev: [{mean - std:.1f}, {mean + std:.1f}]")
    # show legend to right of the plot
    plt.legend()

    plt.xlabel(x_label)

    if output_filename:
        plt.savefig(os.path.join(OUTPUT_DIR, f"{output_filename}.png"), dpi=300)
        plt.savefig(os.path.join(OUTPUT_DIR, f"{output_filename}.svg"))


# %% [markdown]
# ## Data

# %%
con = duckdb.connect("../data/vocabulary.duckdb")
vocab_df = con.execute(
    """
    SELECT
        study,
        survey_vocab_max,
        subject_id,
        sex,
        age,
        understood,
        spoken,
        signed
    FROM vocab_combined
    """
).df()
con.close()

# %% [markdown]
# ## Descriptive statistics

# %%
desc = descriptive_stats.describe_all(
    vocab_df[["survey_vocab_max", "age", "understood", "spoken", "signed"]], alpha=0.05
)
desc

# %% [markdown]
# ### By study

# %%
study_df = vocab_df.groupby("study")[["survey_vocab_max", "age", "understood", "spoken", "signed"]]
study_df.describe().T

# %% [markdown]
# #### By survey maximum score

# %%
survey_max_df = vocab_df.groupby("survey_vocab_max")[["age", "understood", "spoken", "signed"]]
survey_max_df.describe().T

# %% [markdown]
# One observation is at the maximum score of the instrument used:

# %%
vocab_df[vocab_df[["understood", "spoken", "signed"]].max(axis=1) == vocab_df["survey_vocab_max"]]


# %% [markdown]
# ## Variables

# %% [markdown]
# ### Words understood

# %%
plt.figure(figsize=plot_styles.FIGSIZE_XXL)
plt.scatter(
    x=vocab_df[vocab_df["study"] == "uk_01"]["age"],
    y=vocab_df[vocab_df["study"] == "uk_01"]["understood"],
    label="UK 01",
    alpha=0.5,
)
plt.scatter(
    x=vocab_df[vocab_df["study"] == "uk_02"]["age"],
    y=vocab_df[vocab_df["study"] == "uk_02"]["understood"],
    label="UK 02",
    alpha=0.5,
)
plt.scatter(
    x=vocab_df[vocab_df["study"] == "ie_01"]["age"],
    y=vocab_df[vocab_df["study"] == "ie_01"]["understood"],
    label="Ireland 01",
    alpha=0.5,
)
plt.scatter(
    x=vocab_df[vocab_df["study"] == "us_01"]["age"],
    y=vocab_df[vocab_df["study"] == "us_01"]["understood"],
    label="US 01",
    alpha=0.5,
)
plt.scatter(
    x=vocab_df[vocab_df["study"] == "uk_03"]["age"],
    y=vocab_df[vocab_df["study"] == "uk_03"]["understood"],
    label="UK 03",
    alpha=0.5,
)
plt.scatter(
    x=vocab_df[vocab_df["study"] == "it_01"]["age"],
    y=vocab_df[vocab_df["study"] == "it_01"]["understood"],
    label="Italy 01",
    alpha=0.5,
)
plt.scatter(
    x=vocab_df[vocab_df["study"] == "uk_04"]["age"],
    y=vocab_df[vocab_df["study"] == "uk_04"]["understood"],
    label="UK 04",
    alpha=0.5,
)
plt.scatter(
    x=vocab_df[vocab_df["study"] == "uk_05"]["age"],
    y=vocab_df[vocab_df["study"] == "uk_05"]["understood"],
    label="UK 05",
    alpha=0.5,
)
plt.scatter(
    x=vocab_df[vocab_df["study"] == "us_02"]["age"],
    y=vocab_df[vocab_df["study"] == "us_02"]["understood"],
    label="US 02",
    alpha=0.5,
)
plt.scatter(
    x=vocab_df[vocab_df["study"] == "uk_06"]["age"],
    y=vocab_df[vocab_df["study"] == "uk_06"]["understood"],
    label="UK 06",
    alpha=0.5,
)


plt.legend(loc="upper left")
plt.xlabel("Age (months)")
plt.ylabel("Count of checklist words understood")

if SAVE_PLOTS:
    plt.savefig(os.path.join(OUTPUT_DIR, "scatter-words-age-understood.png"), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, "scatter-words-age-understood.svg"))

plt.show()

# %%
plot_histogram(
    data=vocab_df,
    x_name="understood",
    x_label="Words Understood",
    output_filename="histogram-words-understood",
)

plt.show()



# %% [markdown]
# ### Words spoken

# %%
plt.figure(figsize=plot_styles.FIGSIZE_XXL)
plt.scatter(
    x=vocab_df[vocab_df["study"] == "uk_01"]["age"],
    y=vocab_df[vocab_df["study"] == "uk_01"]["spoken"],
    label="UK 01",
    alpha=0.5,
)
plt.scatter(
    x=vocab_df[vocab_df["study"] == "uk_02"]["age"],
    y=vocab_df[vocab_df["study"] == "uk_02"]["spoken"],
    label="UK 02",
    alpha=0.5,
)
plt.scatter(
    x=vocab_df[vocab_df["study"] == "ie_01"]["age"],
    y=vocab_df[vocab_df["study"] == "ie_01"]["spoken"],
    label="Ireland 01",
    alpha=0.5,
)
plt.scatter(
    x=vocab_df[vocab_df["study"] == "us_01"]["age"],
    y=vocab_df[vocab_df["study"] == "us_01"]["spoken"],
    label="US 01",
    alpha=0.5,
)
plt.scatter(
    x=vocab_df[vocab_df["study"] == "uk_03"]["age"],
    y=vocab_df[vocab_df["study"] == "uk_03"]["spoken"],
    label="UK 03",
    alpha=0.5,
)
plt.scatter(
    x=vocab_df[vocab_df["study"] == "it_01"]["age"],
    y=vocab_df[vocab_df["study"] == "it_01"]["spoken"],
    label="Italy 01",
    alpha=0.5,
)
plt.scatter(
    x=vocab_df[vocab_df["study"] == "uk_04"]["age"],
    y=vocab_df[vocab_df["study"] == "uk_04"]["spoken"],
    label="UK 04",
    alpha=0.5,
)
plt.scatter(
    x=vocab_df[vocab_df["study"] == "uk_05"]["age"],
    y=vocab_df[vocab_df["study"] == "uk_05"]["spoken"],
    label="UK 05",
    alpha=0.5,
)
plt.scatter(
    x=vocab_df[vocab_df["study"] == "us_02"]["age"],
    y=vocab_df[vocab_df["study"] == "us_02"]["spoken"],
    label="US 02",
    alpha=0.5,
)
plt.scatter(
    x=vocab_df[vocab_df["study"] == "uk_06"]["age"],
    y=vocab_df[vocab_df["study"] == "uk_06"]["spoken"],
    label="UK 06",
    alpha=0.5,
)

regression_df = vocab_df[["age", "spoken"]].dropna()
x = regression_df["age"]
y = regression_df["spoken"]

plt.legend()
plt.xlabel("Age (months)")
plt.ylabel("Count of checklist words spoken")

if SAVE_PLOTS:
    plt.savefig(os.path.join(OUTPUT_DIR, "scatter-words-age-spoken.png"), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, "scatter-words-age-spoken.svg"))

plt.show()

# %%
plot_histogram(
    data=vocab_df,
    x_name="spoken",
    x_label="Words Spoken",
    output_filename="histogram-words-spoken",
)

plt.show()

# %%
under_2y_df = vocab_df[vocab_df["age"] < 24]
under_2y_df.describe().T

# %%
from_2_to_4y_df = vocab_df[(vocab_df["age"] >= 24) & (vocab_df["age"] < 48)]
from_2_to_4y_df.describe().T

# %%
from_4_to_6y_df = vocab_df[(vocab_df["age"] >= 48) & (vocab_df["age"] < 72)]
from_4_to_6y_df.describe().T

# %%
over_6y_df = vocab_df[vocab_df["age"] >= 72]
over_6y_df.describe().T

# %%
plot_histogram(
    data=under_2y_df,
    x_name="spoken",
    x_label="Words Spoken (under 2 years)",
    output_filename="histogram-words-spoken-under-2y",
)

plt.show()

# %%
plot_histogram(
    data=under_2y_df,
    x_name="understood",
    x_label="Words Understood (under 2 years)",
    output_filename="histogram-words-understood-under-2y",
)

plt.show()

# %%
plot_histogram(
    data=from_4_to_6y_df,
    x_name="spoken",
    x_label="Words Spoken (4 to 6 years)",
    output_filename="histogram-words-spoken-4-to-6y",
)

plt.show()

# %%
plot_histogram(
    data=from_4_to_6y_df,
    x_name="spoken",
    x_label="Words Spoken (4 to 6 years)",
    output_filename="histogram-words-spoken-4-to-6y",
)

plt.show()

# %%
plot_histogram(
    data=over_6y_df,
    x_name="spoken",
    x_label="Words Spoken (over 6 years)",
    output_filename="histogram-words-spoken-over-6y",
)

plt.show()
