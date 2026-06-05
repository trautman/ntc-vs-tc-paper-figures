import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

OUTDIR = "paper-figures-3-agent/"
os.makedirs(OUTDIR, exist_ok=True)

np.random.seed(7)

# IMPORTANT: 3-agent joint is N^3, so keep N much smaller than the 2-agent case.
N = 45
N_JOINT_TRIPLES_TO_DISPLAY = 100
T = 100

AGENT_COLORS = ["orange", "blue", "black"]
MODE_NAMES = ["".join(bits) for bits in itertools.product(["L", "R"], repeat=3)]


def logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


def softmax_from_logweights(logw):
    return np.exp(logw - logsumexp(logw))


def build_gp_like_library(start, goal, T=100, n_samples=90, seed=0):
    """
    Same spirit as the 2-agent version, but now the lateral deviation is relative
    to the agent's own start-goal heading. This is important for triangle geometry.
    """
    rng = np.random.default_rng(seed)
    tau = np.linspace(0.0, 1.0, T)

    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    heading = goal - start
    heading_norm = np.linalg.norm(heading)
    if heading_norm < 1e-12:
        raise ValueError("start and goal must be different")

    left_normal = np.array([-heading[1], heading[0]]) / heading_norm
    baseline = (1.0 - tau)[:, None] * start + tau[:, None] * goal

    trajs = []
    for _ in range(n_samples):
        side = rng.choice([-1.0, 1.0])

        # Smooth density: many samples near centerline, fewer far away.
        amp = abs(rng.normal(loc=0.0, scale=0.23))
        amp = min(amp, 0.85)

        base = np.sin(np.pi * tau)

        wiggle = np.zeros_like(tau)
        for k in range(1, 4):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            coeff = rng.normal(0.0, 1.0 / (k ** 2.3))
            wiggle += coeff * np.sin(2.0 * np.pi * k * tau + phase)

        wiggle = wiggle / max(np.max(np.abs(wiggle)), 1e-9)
        wiggle_amp = rng.uniform(0.00, 0.025)

        lateral = side * amp * base + wiggle_amp * wiggle * base
        traj = baseline + lateral[:, None] * left_normal[None, :]
        traj[0] = start
        traj[-1] = goal
        trajs.append(traj)

    return np.array(trajs)


def local_lateral_values(traj):
    start = traj[0]
    goal = traj[-1]
    heading = goal - start
    heading_norm = np.linalg.norm(heading)
    left_normal = np.array([-heading[1], heading[0]]) / heading_norm
    baseline = np.linspace(start, goal, len(traj))
    return (traj - baseline) @ left_normal


def preference_cost(traj):
    lateral = local_lateral_values(traj)
    dlateral = np.diff(lateral)
    ddlateral = np.diff(lateral, n=2)
    max_dev = np.max(np.abs(lateral))

    return (
        0.35 * max_dev ** 2
        + 2.0 * max(0.0, max_dev - 0.65) ** 2
        + 0.35 * np.sum(dlateral ** 2)
        + 0.60 * np.sum(ddlateral ** 2)
    )


def compute_marginal_weights(trajs):
    costs = np.array([preference_cost(tr) for tr in trajs])
    return softmax_from_logweights(-0.9 * costs)


def pairwise_collision_cost(tr_a, tr_b):
    d = np.linalg.norm(tr_a - tr_b, axis=1)
    d_min = np.min(d)

    collision_wall = 60.0 * np.exp(-(d_min / 0.24) ** 2)
    comfort_wall = 12.0 / (1.0 + np.exp(22.0 * (d_min - 0.70)))
    return collision_wall + comfort_wall


def triple_cost(tr0, tr1, tr2):
    # Main interaction term: all pairwise separation costs.
    interaction = (
        pairwise_collision_cost(tr0, tr1)
        + pairwise_collision_cost(tr0, tr2)
        + pairwise_collision_cost(tr1, tr2)
    )

    # Light effort term, analogous to the 2-agent code.
    effort = 0.20 * (
        np.mean(local_lateral_values(tr0) ** 2)
        + np.mean(local_lateral_values(tr1) ** 2)
        + np.mean(local_lateral_values(tr2) ** 2)
    )

    return interaction + effort


def solve_joint_kl(gamma_ind, C, lam=0.9):
    log_gamma = np.log(gamma_ind + 1e-300) - C / lam
    return np.exp(log_gamma - logsumexp(log_gamma.ravel()))


def local_lateral_side(traj, eps=1e-6):
    mid = len(traj) // 2
    signed_lateral = local_lateral_values(traj)[mid]

    if signed_lateral > eps:
        return 1
    if signed_lateral < -eps:
        return -1
    return 0


def mode_from_sides(sides):
    if any(s == 0 for s in sides):
        return "center"
    return "".join("L" if s > 0 else "R" for s in sides)


def sample_triples_from_gamma(gamma, n_samples=1000, seed=0):
    rng = np.random.default_rng(seed)
    flat_probs = gamma.ravel()
    flat_probs = flat_probs / flat_probs.sum()

    flat_indices = rng.choice(
        len(flat_probs),
        size=n_samples,
        replace=True,
        p=flat_probs,
    )

    triples = []
    n0, n1, n2 = gamma.shape
    for flat_idx in flat_indices:
        i, j, k = np.unravel_index(flat_idx, (n0, n1, n2))
        triples.append((i, j, k, gamma[i, j, k]))

    return triples


def classify_triples(trajs, triples):
    groups = {name: [] for name in MODE_NAMES}
    groups["center"] = []

    side_tables = [np.array([local_lateral_side(tr) for tr in T_a]) for T_a in trajs]

    for i, j, k, mass in triples:
        mode = mode_from_sides([side_tables[0][i], side_tables[1][j], side_tables[2][k]])
        groups[mode].append((i, j, k, mass))

    return groups


def compute_mode_masses(gamma, trajs):
    masses = {name: 0.0 for name in MODE_NAMES}
    masses["center"] = 0.0

    side_tables = [np.array([local_lateral_side(tr) for tr in T_a]) for T_a in trajs]

    n0, n1, n2 = gamma.shape
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                mode = mode_from_sides([side_tables[0][i], side_tables[1][j], side_tables[2][k]])
                masses[mode] += gamma[i, j, k]

    return masses


def setup_axis(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.25, 1.20)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_traj(ax, traj, color, lw, alpha):
    ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=lw, alpha=alpha)


def plot_marginal(ax, trajs, color):
    for traj in trajs:
        plot_traj(ax, traj, color, lw=0.95, alpha=0.22)


def plot_triple_group(ax, trajs, triples, colors, alpha_scale=1.0):
    for i, j, k, mass in triples:
        lw = 0.7 + 30.0 * mass
        plot_traj(ax, trajs[0][i], colors[0], lw=lw, alpha=0.66 * alpha_scale)
        plot_traj(ax, trajs[1][j], colors[1], lw=lw, alpha=0.62 * alpha_scale)
        plot_traj(ax, trajs[2][k], colors[2], lw=lw, alpha=0.58 * alpha_scale)


def label_start_goal(ax, start, goal, color):
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    dx, dy = goal - start
    theta = np.degrees(np.arctan2(dy, dx))

    triangle_marker = MarkerStyle(">")
    triangle_marker._transform = Affine2D().rotate_deg(theta - 90.0)

    triangle_offset = 0.095
    triangle_x = start[0] + triangle_offset * np.cos(np.radians(theta))
    triangle_y = start[1] + triangle_offset * np.sin(np.radians(theta))

    ax.scatter(
        [triangle_x],
        [triangle_y],
        s=95,
        marker=triangle_marker,
        facecolors=color,
        edgecolors="black",
        linewidths=1.5,
        zorder=21,
    )

    ax.scatter(
        [goal[0]],
        [goal[1]],
        s=120,
        marker="x",
        color=color,
        linewidths=3.4,
        zorder=22,
    )


def add_agent_markers(ax, starts, goals, colors):
    for start, goal, color in zip(starts, goals, colors):
        label_start_goal(ax, start, goal, color)


def add_title(ax, title):
    ax.text(
        0.0,
        1.08,
        title,
        ha="center",
        va="center",
        fontsize=10,
        zorder=100,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=1.5),
    )


def save_fig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)


def main():
    # ---------------------------------------------------------------------
    # Set 3-agent start and goal locations here.
    # This default is a symmetric triangle: each agent heads toward the
    # opposite side, so the agents cross near the center.
    # ---------------------------------------------------------------------
    starts = [
        np.array([0.00,  1.00]),
        np.array([-1.00, -0.58]),
        np.array([1.00, -0.58]),
    ]

    goals = [
        np.array([0.00, -1.00]),
        np.array([1.00,  0.58]),
        np.array([-1.00, 0.58]),
    ]

    seeds = [3, 11, 19]

    trajs = [
        build_gp_like_library(starts[a], goals[a], T=T, n_samples=N, seed=seeds[a])
        for a in range(3)
    ]

    ps = [compute_marginal_weights(T_a) for T_a in trajs]

    C = np.zeros((N, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j, k] = triple_cost(trajs[0][i], trajs[1][j], trajs[2][k])

    gamma_ind = ps[0][:, None, None] * ps[1][None, :, None] * ps[2][None, None, :]
    gamma = solve_joint_kl(gamma_ind, C, lam=0.9)

    sampled_triples = sample_triples_from_gamma(
        gamma,
        n_samples=N_JOINT_TRIPLES_TO_DISPLAY,
        seed=42,
    )

    triple_groups = classify_triples(trajs, sampled_triples)
    mode_masses = compute_mode_masses(gamma, trajs)

    print("Sampled triple counts:")
    for name in MODE_NAMES + ["center"]:
        print(name, len(triple_groups[name]))

    print("Full gamma mass:")
    for name in MODE_NAMES + ["center"]:
        print(f"{name:6s}  {mode_masses[name]:.6f}  ({100*mode_masses[name]:.2f}%)")
    print(f"total   {sum(mode_masses.values()):.6f}")

    # ----------------------------
    # Figure 1: overlaid marginals
    # ----------------------------
    fig_marg = plt.figure(figsize=(7.2, 4.2))
    ax_marg = fig_marg.add_subplot(1, 1, 1)
    setup_axis(ax_marg)

    for a in range(3):
        plot_marginal(ax_marg, trajs[a], AGENT_COLORS[a])

    add_agent_markers(ax_marg, starts, goals, AGENT_COLORS)
    add_title(ax_marg, "Marginals")

    marg_png_path = os.path.join(OUTDIR, "ot_3agent_marginals_only.png")
    save_fig(fig_marg, marg_png_path)

    # ----------------------------
    # Figures 2: one figure per mode
    # ----------------------------
    for mode_name in MODE_NAMES:
        fig = plt.figure(figsize=(7.2, 4.2))
        ax = fig.add_subplot(1, 1, 1)
        setup_axis(ax)

        plot_triple_group(ax, trajs, triple_groups[mode_name], AGENT_COLORS)
        add_agent_markers(ax, starts, goals, AGENT_COLORS)
        add_title(
            ax,
            f"{mode_name} Mode ({100*mode_masses[mode_name]:.1f}% of $\\gamma^*$ mass)",
        )

        out_path = os.path.join(OUTDIR, f"ot_3agent_joint_{mode_name}.png")
        save_fig(fig, out_path)

    print(f"Wrote {marg_png_path}")
    for mode_name in MODE_NAMES:
        print(f"Wrote {os.path.join(OUTDIR, f'ot_3agent_joint_{mode_name}.png')}")


if __name__ == "__main__":
    main()
