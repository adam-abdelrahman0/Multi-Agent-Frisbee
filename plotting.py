import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRAIN_LEARNING = "/Users/adam-admin/code/Multi-Agent-Frisbee/learning_curve.json"
TRAIN_EPISODES = "/Users/adam-admin/code/Multi-Agent-Frisbee/training_episode_stats.json"
EVAL_EPISODES  = "/Users/adam-admin/code/Multi-Agent-Frisbee/eval_episode_stats.csv"


# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------

with open(TRAIN_LEARNING, "r") as f:
    learning_curve = json.load(f)

with open(TRAIN_EPISODES, "r") as f:
    train_eps = json.load(f)

df_eval = pd.read_csv(EVAL_EPISODES)


# -------------------------------------------------------------------
# 1. LEARNING CURVE
# -------------------------------------------------------------------

plt.figure(figsize=(10,4))
plt.plot(learning_curve, linewidth=2)
plt.title("Learning Curve: Mean Episode Reward per Iteration")
plt.xlabel("Training Iteration")
plt.ylabel("Mean Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=200)
plt.show()


# -------------------------------------------------------------------
# 2. EPISODE OUTCOME BREAKDOWN (EVALUATION)
# -------------------------------------------------------------------

plt.figure(figsize=(10,4))
df_eval['reason'].value_counts().plot(kind="bar", color="gray")
plt.title("Episode Outcomes (Evaluation)")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outcome_bar.png", dpi=200)
plt.show()

plt.figure(figsize=(5,5))
df_eval['reason'].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Episode Outcome Breakdown")
plt.ylabel("")
plt.tight_layout()
plt.savefig("outcome_pie.png", dpi=200)
plt.show()


# -------------------------------------------------------------------
# 3. THROW / CATCH / INTERCEPTION RATES (EVALUATION)
# -------------------------------------------------------------------

plt.figure(figsize=(10,4))
plt.plot(df_eval["throws"], label="Throws")
plt.plot(df_eval["catches"], label="Catches")
plt.plot(df_eval["intercepts"], label="Interceptions")
plt.title("Throw / Catch / Intercept Count per Episode")
plt.xlabel("Episode #")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("throw_catch_intercept_trends.png", dpi=200)
plt.show()

# Bar chart version
df_eval[["throws","catches","intercepts"]].mean().plot(kind="bar", figsize=(6,4))
plt.title("Average Throw / Catch / Intercept Count (Evaluation)")
plt.ylabel("Average per Episode")
plt.tight_layout()
plt.savefig("throw_catch_intercept_bar.png", dpi=200)
plt.show()


# -------------------------------------------------------------------
# 4. STALL-OUTS
# -------------------------------------------------------------------

plt.figure(figsize=(10,4))
plt.plot(df_eval["stallouts"], label="Stall-outs", color="darkred")
plt.title("Stall-outs per Episode (Evaluation)")
plt.xlabel("Episode #")
plt.ylabel("Stall Count")
plt.grid(True)
plt.tight_layout()
plt.savefig("stallouts_trend.png", dpi=200)
plt.show()


# -------------------------------------------------------------------
# 5. SCORE PROGRESSION
# -------------------------------------------------------------------

plt.figure(figsize=(10,4))
plt.plot(df_eval["score"], marker="o")
plt.title("Episode Score (Evaluation)")
plt.xlabel("Episode #")
plt.ylabel("Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("score_trend.png", dpi=200)
plt.show()


# -------------------------------------------------------------------
# 6. THROW DISTANCE DISTRIBUTION & SCATTER VS OUTCOME
# -------------------------------------------------------------------
# NOTE: This requires that training_episode_stats.json contains throw distances.
# If not, this will skip.

if len(train_eps) > 0 and "throw_distances" in train_eps[0]:
    all_dists = []
    success_flags = []

    for ep in train_eps:
        ds = ep.get("throw_distances", [])
        succ = ep.get("throw_success", [])
        all_dists.extend(ds)
        success_flags.extend(succ)

    all_dists = np.asarray(all_dists)
    success_flags = np.asarray(success_flags)

    plt.figure(figsize=(7,4))
    plt.hist(all_dists, bins=20)
    plt.title("Distribution of Throw Distances (Training)")
    plt.xlabel("Throw Distance (units)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("throw_distance_hist.png", dpi=200)
    plt.show()

    plt.figure(figsize=(7,4))
    plt.scatter(all_dists, success_flags + 0.01*np.random.randn(len(all_dists)))
    plt.title("Throw Distance vs Success")
    plt.xlabel("Throw Distance")
    plt.ylabel("Success (1=yes,0=no)")
    plt.tight_layout()
    plt.savefig("throw_distance_vs_success.png", dpi=200)
    plt.show()


# -------------------------------------------------------------------
# 7. EPISODE LENGTH DISTRIBUTION
# -------------------------------------------------------------------

if "episode_length" in df_eval.columns:
    plt.figure(figsize=(7,4))
    plt.hist(df_eval["episode_length"], bins=10, color="gray")
    plt.title("Episode Length Distribution (Evaluation)")
    plt.xlabel("Steps per Episode")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("episode_length_dist.png", dpi=200)
    plt.show()


# -------------------------------------------------------------------
# 8. DEFENSIVE IMPACT: Interceptions vs Score
# -------------------------------------------------------------------

plt.figure(figsize=(7,4))
plt.scatter(df_eval["intercepts"], df_eval["score"])
plt.title("Interceptions vs Score (Evaluation)")
plt.xlabel("Interception Count")
plt.ylabel("Episode Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("intercepts_vs_score.png", dpi=200)
plt.show()


print("All analysis plots generated.")
