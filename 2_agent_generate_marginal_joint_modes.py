import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

from ot_models import (
    logsumexp,
    solve_joint_kl,
    solve_sinkhorn_ot,
    solve_marginal_kl,
    solve_balanced_ot,
)

OUTDIR = "paper-figures-2-agent/"

SEPARATED_DIR = os.path.join(OUTDIR, "separated_plots")
COMBINED_DIR = os.path.join(OUTDIR, "combined_plots")

os.makedirs(SEPARATED_DIR, exist_ok=True)
os.makedirs(COMBINED_DIR, exist_ok=True)

np.random.seed(7)

N = 300
N_JOINT_PAIRS_TO_DISPLAY = 300
T = 100
S = 2.5

# HUMAN_COLOR = "green"
# ROBOT_COLOR = "firebrick"
HUMAN_COLOR = "orange"
ROBOT_COLOR = "blue"
BAD_COLOR = "#d64b7f"





def softmax_from_logweights(logw):
    return np.exp(logw - logsumexp(logw))


def build_gp_like_library(start, goal, T=100, n_samples=90, seed=0):
    rng = np.random.default_rng(seed)
    tau = np.linspace(0.0, 1.0, T)
    x = np.linspace(start[0], goal[0], T)

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

        # y = side * amp * base + wiggle_amp * wiggle * base
        baseline_y = np.linspace(start[1], goal[1], T)
        y = baseline_y + side * amp * base + wiggle_amp * wiggle * base


        traj = np.column_stack([x, y])
        traj[0] = start
        traj[-1] = goal
        trajs.append(traj)

    return np.array(trajs)


def preference_cost(traj):
    y = traj[:, 1]
    dy = np.diff(y)
    ddy = np.diff(y, n=2)
    max_dev = np.max(np.abs(y))

    return (
        0.35 * max_dev ** 2
        + 2.0 * max(0.0, max_dev - 0.65) ** 2
        + 0.35 * np.sum(dy ** 2)
        + 0.60 * np.sum(ddy ** 2)
    )


def compute_marginal_weights(trajs):
    costs = np.array([preference_cost(tr) for tr in trajs])
    return softmax_from_logweights(-0.9 * costs)


def pairwise_cost(tr_h, tr_r):
    d = np.linalg.norm(tr_h - tr_r, axis=1)
    d_min = np.min(d)

    collision_wall = 60.0 * np.exp(-(d_min / 0.24) ** 2)
    comfort_wall = 12.0 / (1.0 + np.exp(22.0 * (d_min - 0.70)))

    mid = len(tr_h) // 2
    yh = tr_h[mid, 1] - 0.5 * (tr_h[0, 1] + tr_h[-1, 1])
    yr = tr_r[mid, 1] - 0.5 * (tr_r[0, 1] + tr_r[-1, 1])

    same_world_side_penalty = 0.0
    if np.sign(yh) == np.sign(yr) and abs(yh) > 0.05 and abs(yr) > 0.05:
        same_world_side_penalty = 7.0

    effort = 0.20 * (np.mean(tr_h[:, 1] ** 2) + np.mean(tr_r[:, 1] ** 2))

    return collision_wall + comfort_wall + same_world_side_penalty + effort





def top_joint_pairs(gamma, k=1000):
    idx = np.argsort(gamma.ravel())[::-1]
    n_r = gamma.shape[1]

    pairs = []
    for flat_idx in idx:
        i = flat_idx // n_r
        j = flat_idx % n_r
        mass = gamma[i, j]
        if mass <= 1e-10:
            continue
        pairs.append((i, j, mass))
        if len(pairs) >= k:
            break

    return pairs


def side_at_midpoint(traj, eps=1e-6):
    mid = len(traj) // 2

    y_mid = traj[mid, 1]

    y_baseline_mid = 0.5 * (traj[0, 1] + traj[-1, 1])
    y_rel = y_mid - y_baseline_mid

    if y_rel > eps:
        return 1
    if y_rel < -eps:
        return -1
    return 0


def local_lateral_side(traj, eps=1e-6):
    mid = len(traj) // 2

    start = traj[0]
    goal = traj[-1]
    point = traj[mid]

    heading = goal - start
    heading_norm = np.linalg.norm(heading)

    if heading_norm < 1e-12:
        return 0

    # left normal of this agent's own start-goal direction
    left_normal = np.array([-heading[1], heading[0]]) / heading_norm

    baseline_mid = 0.5 * (start + goal)
    signed_lateral = np.dot(point - baseline_mid, left_normal)

    if signed_lateral > eps:
        return 1
    if signed_lateral < -eps:
        return -1
    return 0


def split_pairs(H, R, gamma, k_scan=1000, k_each=18):
    all_pairs = top_joint_pairs(gamma, k=k_scan)

    ll = []
    rr = []
    lr = []
    rl = []
    center = []

    for i, j, mass in all_pairs:
        h_side = local_lateral_side(H[i])
        r_side = local_lateral_side(R[j])

        if h_side == 0 or r_side == 0:
            center.append((i, j, mass))
        elif h_side > 0 and r_side > 0:
            ll.append((i, j, mass))
        elif h_side < 0 and r_side < 0:
            rr.append((i, j, mass))
        elif h_side > 0 and r_side < 0:
            lr.append((i, j, mass))
        else:
            rl.append((i, j, mass))
 
    return {
        "LL": ll[:k_each],
        "RR": rr[:k_each],
        "LR": lr[:k_each],
        "RL": rl[:k_each],
        "center": center[:k_each],
    }


def setup_axis(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.80, 1.00)

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def add_arrowhead(ax, traj, color, alpha=1.0, size=12):
    p0 = traj[-28]
    p1 = traj[-20]
    ax.annotate(
        "",
        xy=(p1[0], p1[1]),
        xytext=(p0[0], p0[1]),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=0.8,
            alpha=alpha,
            mutation_scale=2.2 * size,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=6,
    )


def plot_traj(ax, traj, color, lw, alpha):
    ax.plot(
        traj[:, 0],
        traj[:, 1],
        color=color,
        linewidth=lw,
        alpha=alpha,
    )

def plot_marginal(ax, trajs, weights, color):
    for traj, w in zip(trajs, weights):
        plot_traj(
            ax,
            traj,
            color,
            lw=0.95,
            alpha=0.25,
        )




def plot_pair_group(ax, H, R, pairs, h_color, r_color, alpha_scale=1.0):
    for i, j, mass in pairs:
        lw = 0.8 + 22.0 * mass
        plot_traj(ax, H[i], h_color, lw=lw, alpha=0.68 * alpha_scale)
        plot_traj(ax, R[j], r_color, lw=lw, alpha=0.62 * alpha_scale)

def label_start_goal(ax, start, goal, color, label, label_side):
    label_offset = 0.20

    x_start, y_start = start
    x_goal, y_goal = goal

    if color == HUMAN_COLOR:
        start_label_y = y_start + label_offset
        goal_label_y = y_goal + label_offset
    else:
        start_label_y = y_start - label_offset
        goal_label_y = y_goal - label_offset

    direction = np.sign(x_goal - x_start)

    dx = x_goal - x_start
    dy = y_goal - y_start
    theta = np.degrees(np.arctan2(dy, dx))

    triangle_marker = MarkerStyle(">")
    triangle_marker._transform = Affine2D().rotate_deg(theta - 90.0)

    triangle_offset = 0.095
    triangle_x = x_start + triangle_offset * np.cos(np.radians(theta))
    triangle_y = y_start + triangle_offset * np.sin(np.radians(theta))

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

    # Goal marker: thick X at exact sample goal
    ax.scatter(
        [x_goal],
        [y_goal],
        s=120,
        marker="x",
        color=color,
        linewidths=3.4,
        zorder=22,
    )

def add_agent_labels(ax, start_h, goal_h, start_r, goal_r):
    label_start_goal(ax, start_h, goal_h, HUMAN_COLOR, "Human", "left")
    label_start_goal(ax, start_r, goal_r, ROBOT_COLOR, "Robot", "right")

def add_s_label(ax, start_h, start_r):
    ax.annotate(
        "",
        xy=(start_h[0], -0.88),
        xytext=(start_r[0], -0.88),
        arrowprops=dict(arrowstyle="<->", linewidth=1.0, color="black"),
    )
    ax.text(
        0.0,
        -0.98,
        f"$s={S:.1f}\\,\\mathrm{{m}}$",
        ha="center",
        va="top",
        fontsize=12,
    )

def add_centerline_arrow(ax, x_start, x_goal, y, color):
    margin = 0.35
    direction = np.sign(x_goal - x_start)

    x0 = x_start + direction * margin
    x1 = x_goal - direction * margin

    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        arrowprops=dict(
            arrowstyle="-|>",
            linewidth=1.8,
            color=color,
            alpha=0.85,
        ),
        zorder=6,
    )


def sample_pairs_from_gamma(gamma, n_samples=1000, seed=0):
    rng = np.random.default_rng(seed)

    flat_probs = gamma.ravel()
    flat_probs = flat_probs / flat_probs.sum()

    flat_indices = rng.choice(
        len(flat_probs),
        size=n_samples,
        replace=True,
        p=flat_probs,
    )

    n_r = gamma.shape[1]

    pairs = []
    for flat_idx in flat_indices:
        i = flat_idx // n_r
        j = flat_idx % n_r
        pairs.append((i, j, gamma[i, j]))

    return pairs


def classify_pairs(H, R, pairs):
    groups = {
        "LL": [],
        "RR": [],
        "LR": [],
        "RL": [],
        "center": [],
    }

    for i, j, mass in pairs:
        h_side = local_lateral_side(H[i])
        r_side = local_lateral_side(R[j])

        if h_side == 0 or r_side == 0:
            groups["center"].append((i, j, mass))
        elif h_side > 0 and r_side > 0:
            groups["LL"].append((i, j, mass))
        elif h_side < 0 and r_side < 0:
            groups["RR"].append((i, j, mass))
        elif h_side > 0 and r_side < 0:
            groups["LR"].append((i, j, mass))
        else:
            groups["RL"].append((i, j, mass))

    return groups

def add_mode_title(ax, mode_name, mass):
    ax.text(
        0.0,
        0.94,
        f"{mode_name} Mode ({100*mass:.1f}% of $\\gamma^*$ mass)",
        ha="center",
        va="center",
        fontsize=10,
        zorder=100,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.90,
            pad=1.5,
        ),
    )


def save_combined_mode_page(
    mode_name,
    model_results,
    model_order,
    H,
    R,
    start_h,
    goal_h,
    start_r,
    goal_r,
):
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.8))

    for ax, (model_name, label) in zip(axes, model_order):

        setup_axis(ax)

        pairs = model_results[model_name]["pair_groups"][mode_name]
        mass = model_results[model_name]["masses"][mode_name]

        plot_pair_group(
            ax,
            H,
            R,
            pairs,
            HUMAN_COLOR,
            ROBOT_COLOR,
        )

        add_agent_labels(
            ax,
            start_h,
            goal_h,
            start_r,
            goal_r,
        )

        ax.text(
            0.0,
            0.94,
            f"{label}\n({100*mass:.1f}% mass)",
            ha="center",
            va="center",
            fontsize=8,
            zorder=100,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.90,
                pad=1.5,
            ),
        )

    out_path = os.path.join(
        COMBINED_DIR,
        f"{mode_name}_comparison.png",
    )

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    print(f"Wrote {out_path}")



def main():
 
    start_h = np.array([-1.25,  0.0])
    goal_h  = np.array([ 1.25,  0.18])

    start_r = np.array([ 1.25, -0.3])
    goal_r  = np.array([-1.25, 0.28])

    H = build_gp_like_library(start_h, goal_h, T=T, n_samples=N, seed=3)
    R = build_gp_like_library(start_r, goal_r, T=T, n_samples=N, seed=11)

    p_h = compute_marginal_weights(H)
    p_r = compute_marginal_weights(R)

    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            C[i, j] = pairwise_cost(H[i], R[j])



    # ----------------------------
    # Figure 1: overlaid marginals
    # ----------------------------

    fig_marg = plt.figure(figsize=(7.2, 3.8))
    ax_marg = fig_marg.add_subplot(1, 1, 1)

    setup_axis(ax_marg)

    plot_marginal(ax_marg, H, p_h, HUMAN_COLOR)
    plot_marginal(ax_marg, R, p_r, ROBOT_COLOR)

    add_agent_labels(ax_marg, start_h, goal_h, start_r, goal_r)


    # Direction labels

    legend_y = -0.9


    ax_marg.text(
        start_h[0],
        start_h[1] - 0.25,
        r"$p_h$",
        color=HUMAN_COLOR,
        fontsize=15,
        ha="center",
    )

    ax_marg.text(
        start_r[0],
        start_r[1] - 0.25,
        r"$p_r$",
        color=ROBOT_COLOR,
        fontsize=15,
        ha="center",
    )

    ax_marg.text(
        0.0,
        0.94,
        "Marginals",
        ha="center",
        va="center",
        fontsize=10,
        zorder=100,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.90,
            pad=1.5,
        ),
    )
 

    marg_png_path = os.path.join(OUTDIR, "ot_marginals_only.png")
    marg_pdf_path = os.path.join(OUTDIR, "ot_marginals_only.pdf")

    fig_marg.savefig(
        marg_png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )


    gamma_ind = np.outer(p_h, p_r)

    models = {
        "ot_kl_joint": solve_joint_kl(p_h, p_r, C, lam=0.9),
        "ot_sinkhorn": solve_sinkhorn_ot(p_h, p_r, C, reg=0.9),
        "ot_kl_marg": solve_marginal_kl(p_h, p_r, C, lam_h=0.9, lam_r=0.9, reg=1e-2),
        "ot_balanced": solve_balanced_ot(p_h, p_r, C),
    }

    model_results = {}
    for model_name, gamma in models.items():

        sampled_pairs = sample_pairs_from_gamma(
            gamma,
            n_samples=N_JOINT_PAIRS_TO_DISPLAY,
            seed=42,
        )

        pair_groups = classify_pairs(H, R, sampled_pairs)

        print(f"\nModel: {model_name}")
        print("Sampled pair counts:")
        for name, pairs in pair_groups.items():
            print(name, len(pairs))

        total_ll = 0.0
        total_rr = 0.0
        total_lr = 0.0
        total_rl = 0.0
        total_center = 0.0

        for i in range(len(H)):
            for j in range(len(R)):
                h_side = local_lateral_side(H[i])
                r_side = local_lateral_side(R[j])

                if h_side == 0 or r_side == 0:
                    total_center += gamma[i, j]
                elif h_side > 0 and r_side > 0:
                    total_ll += gamma[i, j]
                elif h_side < 0 and r_side < 0:
                    total_rr += gamma[i, j]
                elif h_side > 0 and r_side < 0:
                    total_lr += gamma[i, j]
                else:
                    total_rl += gamma[i, j]

        print("Full gamma mass:")
        print(f"LL      {total_ll:.6f}  ({100*total_ll:.2f}%)")
        print(f"RR      {total_rr:.6f}  ({100*total_rr:.2f}%)")
        print(f"LR      {total_lr:.6f}  ({100*total_lr:.2f}%)")
        print(f"RL      {total_rl:.6f}  ({100*total_rl:.2f}%)")
        print(f"center  {total_center:.6f}  ({100*total_center:.2f}%)")
        print(f"total   {total_ll + total_rr + total_lr + total_rl + total_center:.6f}")

        model_results[model_name] = {
            "pair_groups": pair_groups,
            "masses": {
                "LL": total_ll,
                "RR": total_rr,
                "LR": total_lr,
                "RL": total_rl,
            },
        }




        # ----------------------------
        # LL mode
        # ----------------------------

        fig_ll = plt.figure(figsize=(7.2, 3.8))
        ax_ll = fig_ll.add_subplot(1, 1, 1)

        setup_axis(ax_ll)

        plot_pair_group(
            ax_ll,
            H,
            R,
            pair_groups["LL"],
            HUMAN_COLOR,
            ROBOT_COLOR,
        )

        add_agent_labels(ax_ll, start_h, goal_h, start_r, goal_r)
        add_mode_title(ax_ll, "LL", total_ll)

        ll_png_path = os.path.join(
            SEPARATED_DIR,
            f"{model_name}_LL.png",
        )

        fig_ll.savefig(
            ll_png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )

        # ----------------------------
        # RR mode
        # ----------------------------

        fig_rr = plt.figure(figsize=(7.2, 3.8))
        ax_rr = fig_rr.add_subplot(1, 1, 1)

        setup_axis(ax_rr)

        plot_pair_group(
            ax_rr,
            H,
            R,
            pair_groups["RR"],
            HUMAN_COLOR,
            ROBOT_COLOR,
        )

        add_agent_labels(ax_rr, start_h, goal_h, start_r, goal_r)
        add_mode_title(ax_rr, "RR", total_rr)

        rr_png_path = os.path.join(
            SEPARATED_DIR,
            f"{model_name}_RR.png",
        )

        fig_rr.savefig(
            rr_png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )

        # ----------------------------
        # LR mode
        # ----------------------------

        fig_lr = plt.figure(figsize=(7.2, 3.8))
        ax_lr = fig_lr.add_subplot(1, 1, 1)

        setup_axis(ax_lr)

        plot_pair_group(
            ax_lr,
            H,
            R,
            pair_groups["LR"],
            HUMAN_COLOR,
            ROBOT_COLOR,
        )

        add_agent_labels(ax_lr, start_h, goal_h, start_r, goal_r)
        add_mode_title(ax_lr, "LR", total_lr)

        lr_png_path = os.path.join(
            SEPARATED_DIR,
            f"{model_name}_LR.png",
        )

        fig_lr.savefig(
            lr_png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )

        # ----------------------------
        # RL mode
        # ----------------------------

        fig_rl = plt.figure(figsize=(7.2, 3.8))
        ax_rl = fig_rl.add_subplot(1, 1, 1)

        setup_axis(ax_rl)

        plot_pair_group(
            ax_rl,
            H,
            R,
            pair_groups["RL"],
            HUMAN_COLOR,
            ROBOT_COLOR,
        )

        add_agent_labels(ax_rl, start_h, goal_h, start_r, goal_r)
        add_mode_title(ax_rl, "RL", total_rl)

        rl_png_path = os.path.join(
            SEPARATED_DIR,
            f"{model_name}_RL.png",
        )

        fig_rl.savefig(
            rl_png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )


    model_order = [
        ("ot_balanced", "Balanced"),
        ("ot_sinkhorn", "Sinkhorn"),
        ("ot_kl_marg", "Marginal KL"),
        ("ot_kl_joint", "Joint KL"),
    ]

    for mode_name in ["LL", "RR", "LR", "RL"]:
        save_combined_mode_page(
            mode_name,
            model_results,
            model_order,
            H,
            R,
            start_h,
            goal_h,
            start_r,
            goal_r,
        )

    print(f"Wrote {marg_png_path}")
    print(f"Wrote {ll_png_path}")
    print(f"Wrote {rr_png_path}")
    print(f"Wrote {lr_png_path}")
    print(f"Wrote {rl_png_path}")





if __name__ == "__main__":
    main()