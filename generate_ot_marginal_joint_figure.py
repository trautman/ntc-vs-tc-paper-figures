import os
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = "paper-figures/"
os.makedirs(OUTDIR, exist_ok=True)

np.random.seed(7)

N = 200
# N = 200
N_JOINT_PAIRS_TO_DISPLAY = 150
T = 100
S = 2.5

HUMAN_COLOR = "green"
ROBOT_COLOR = "firebrick"
BAD_COLOR = "#d64b7f"


def logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


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

        y = side * amp * base + wiggle_amp * wiggle * base

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
    yh = tr_h[mid, 1]
    yr = tr_r[mid, 1]

    same_world_side_penalty = 0.0
    if np.sign(yh) == np.sign(yr) and abs(yh) > 0.05 and abs(yr) > 0.05:
        same_world_side_penalty = 7.0

    effort = 0.20 * (np.mean(tr_h[:, 1] ** 2) + np.mean(tr_r[:, 1] ** 2))

    return collision_wall + comfort_wall + same_world_side_penalty + effort


def solve_joint_kl(gamma_ind, C, lam=0.9):
    log_gamma = np.log(gamma_ind + 1e-300) - C / lam
    return np.exp(log_gamma - logsumexp(log_gamma.ravel()))


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
    y = traj[len(traj) // 2, 1]
    if y > eps:
        return 1
    if y < -eps:
        return -1
    return 0


def split_pairs(H, R, gamma, k_scan=1000, k_each=18):
    all_pairs = top_joint_pairs(gamma, k=k_scan)

    ll = []
    rr = []
    lr_rl = []
    center = []

    for i, j, mass in all_pairs:
        h_side = side_at_midpoint(H[i])
        r_side = side_at_midpoint(R[j])

        if h_side == 0 or r_side == 0:
            center.append((i, j, mass))
        elif h_side > 0 and r_side < 0:
            ll.append((i, j, mass))
        elif h_side < 0 and r_side > 0:
            rr.append((i, j, mass))
        else:
            lr_rl.append((i, j, mass))

    return {
        "LL": ll[:k_each],
        "RR": rr[:k_each],
        "LR_RL": lr_rl[:k_each],
        "center": center[:k_each],
    }


def setup_axis(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.05, 1.05)
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


# def plot_marginal(ax, trajs, weights, color):
#     for traj, w in zip(trajs, weights):
#         plot_traj(
#             ax,
#             traj,
#             color,
#             lw=0.8 + 40.0 * w,
#             alpha=0.35,
#         )
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


def label_start_goal(ax, x_start, x_goal, y, color, label, label_side):
    label_offset = 0.20

    if color == HUMAN_COLOR:
        label_y = y + label_offset
    else:
        label_y = y - label_offset

    # Filled start marker
    ax.scatter(
        [x_start],
        [y],
        s=70,
        facecolors=color,
        edgecolors=color,
        linewidths=2.4,
        zorder=8,
    )

    # Hollow goal marker: always drawn above filled markers
    ax.scatter(
        [x_goal],
        [y],
        s=70,
        facecolors="none",
        edgecolors=color,
        linewidths=2.8,
        zorder=12,
    )

    ax.text(
        x_start,
        label_y,
        "start",
        color=color,
        fontsize=14,
        ha="center",
        va="center",
        zorder=13,
    )

    ax.text(
        x_goal,
        label_y,
        "goal",
        color=color,
        fontsize=14,
        ha="center",
        va="center",
        zorder=13,
    )

# def add_agent_labels(ax, start_h, goal_h, start_r, goal_r):
#     label_start_goal(ax, start_h[0], goal_h[0], 0.05, HUMAN_COLOR, "Human", "left")
#     label_start_goal(ax, start_r[0], goal_r[0], -0.08, ROBOT_COLOR, "Robot", "right")
def add_agent_labels(ax, start_h, goal_h, start_r, goal_r):
    label_start_goal(ax, start_h[0], goal_h[0], 0.0, HUMAN_COLOR, "Human", "left")
    label_start_goal(ax, start_r[0], goal_r[0], 0.0, ROBOT_COLOR, "Robot", "right")


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


def main():
    start_h = np.array([-S / 2.0, 0.0])
    goal_h = np.array([S / 2.0, 0.0])
    start_r = np.array([S / 2.0, 0.0])
    goal_r = np.array([-S / 2.0, 0.0])

    H = build_gp_like_library(start_h, goal_h, T=T, n_samples=N, seed=3)
    R = build_gp_like_library(start_r, goal_r, T=T, n_samples=N, seed=11)

    p_h = compute_marginal_weights(H)
    p_r = compute_marginal_weights(R)

    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            C[i, j] = pairwise_cost(H[i], R[j])

    gamma_ind = np.outer(p_h, p_r)
    gamma = solve_joint_kl(gamma_ind, C, lam=0.9)

    pair_groups = split_pairs(
        H,
        R,
        gamma,
        k_scan=3000,
        k_each=N_JOINT_PAIRS_TO_DISPLAY,
    )

    total_ll = 0.0
    total_rr = 0.0
    total_bad = 0.0

    # for i in range(N):
    #     for j in range(N):
    #         h_side = side_at_midpoint(H[i])
    #         r_side = side_at_midpoint(R[j])

    #         if h_side > 0 and r_side < 0:
    #             total_ll += gamma[i,j]
    #         elif h_side < 0 and r_side > 0:
    #             total_rr += gamma[i,j]
    #         else:
    #             total_bad += gamma[i,j]

    # print(total_ll, total_rr, total_bad)

    # print("Mass by group:")
    # for name, pairs in pair_groups.items():
    #     print(name, sum(mass for _, _, mass in pairs))
    total_ll = 0.0
    total_rr = 0.0
    total_lr_rl = 0.0
    total_center = 0.0

    for i in range(len(H)):
        for j in range(len(R)):
            h_side = side_at_midpoint(H[i])
            r_side = side_at_midpoint(R[j])

            if h_side == 0 or r_side == 0:
                total_center += gamma[i, j]
            elif h_side > 0 and r_side < 0:
                total_ll += gamma[i, j]
            elif h_side < 0 and r_side > 0:
                total_rr += gamma[i, j]
            else:
                total_lr_rl += gamma[i, j]

    print("Full gamma mass:")
    print(f"LL      {total_ll:.6f}  ({100*total_ll:.2f}%)")
    print(f"RR      {total_rr:.6f}  ({100*total_rr:.2f}%)")
    print(f"LR/RL   {total_lr_rl:.6f}  ({100*total_lr_rl:.2f}%)")
    print(f"center  {total_center:.6f}  ({100*total_center:.2f}%)")
    print(f"total   {total_ll + total_rr + total_lr_rl + total_center:.6f}")


  
    # ----------------------------
    # Figure 1: overlaid marginals
    # ----------------------------

    fig_marg = plt.figure(figsize=(7.2, 3.8))
    ax_marg = fig_marg.add_subplot(1, 1, 1)

    setup_axis(ax_marg)

    plot_marginal(ax_marg, H, p_h, HUMAN_COLOR)
    plot_marginal(ax_marg, R, p_r, ROBOT_COLOR)

    label_start_goal(ax_marg, start_h[0], goal_h[0], 0.0, HUMAN_COLOR, "Human", "left")
    label_start_goal(ax_marg, start_r[0], goal_r[0], 0.0, ROBOT_COLOR, "Robot", "right")

    # Direction labels

    legend_y = -0.9

    ax_marg.text(
        -0.35,
        legend_y,
        r"$p_h \rightarrow$",
        color=HUMAN_COLOR,
        fontsize=22,
        ha="center",
        va="center",
    )

    ax_marg.text(
        0.35,
        legend_y,
        r"$p_r \leftarrow$",
        color=ROBOT_COLOR,
        fontsize=22,
        ha="center",
        va="center",
    )




    marg_png_path = os.path.join(OUTDIR, "ot_marginals_only.png")
    marg_pdf_path = os.path.join(OUTDIR, "ot_marginals_only.pdf")

    fig_marg.savefig(marg_png_path, dpi=300, bbox_inches="tight")
    # fig_marg.savefig(marg_pdf_path, bbox_inches="tight")




    # ----------------------------
    # Figure 2: joint pairs only
    # ----------------------------

    # fig_joint = plt.figure(figsize=(7.2, 6.8))

    # gs_joint = fig_joint.add_gridspec(
    #     2,
    #     1,
    #     hspace=0.20,
    # )

    # ax_ll = fig_joint.add_subplot(gs_joint[0, 0])
    # ax_rr = fig_joint.add_subplot(gs_joint[1, 0])

   
    # setup_axis(ax_ll)
    # plot_pair_group(ax_ll, H, R, pair_groups["LL"], HUMAN_COLOR, ROBOT_COLOR)
    # add_agent_labels(ax_ll, start_h, goal_h, start_r, goal_r)
    # # add_s_label(ax_ll, start_h, start_r)
    # ax_ll.text(
    #     0.0,
    #     0.92,
    #     r"Joint KL OT: Both Agents Go Left",
    #     ha="center",
    #     fontsize=10,
    # )

    # setup_axis(ax_rr)
    # plot_pair_group(ax_rr, H, R, pair_groups["RR"], HUMAN_COLOR, ROBOT_COLOR)
    # add_agent_labels(ax_rr, start_h, goal_h, start_r, goal_r)
    # # add_s_label(ax_rr, start_h, start_r)
    # ax_rr.text(
    #     0.0,
    #     0.92,
    #     r"Joint KL OT: Both Agents Go Right",
    #     ha="center",
    #     fontsize=10,
    # )





    # ----------------------------
    # Figure 2a: LL mode
    # ----------------------------

    fig_ll = plt.figure(figsize=(3.8, 3.2))
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

    ax_ll.text(
        0.0,
        0.92,
        r"LL mode",
        ha="center",
        fontsize=10,
    )

    ll_png_path = os.path.join(OUTDIR, "ot_joint_LL.png")

    fig_ll.savefig(
        ll_png_path,
        dpi=300,
        bbox_inches="tight",
    )

    # ----------------------------
    # Figure 2b: RR mode
    # ----------------------------

    fig_rr = plt.figure(figsize=(3.8, 3.2))
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

    ax_rr.text(
        0.0,
        0.92,
        r"RR mode",
        ha="center",
        fontsize=10,
    )

    rr_png_path = os.path.join(OUTDIR, "ot_joint_RR.png")

    fig_rr.savefig(
        rr_png_path,
        dpi=300,
        bbox_inches="tight",
    )





    print(f"Wrote {marg_png_path}")
    print(f"Wrote {ll_png_path}")
    print(f"Wrote {rr_png_path}")





    # joint_png_path = os.path.join(OUTDIR, "ot_joint_pair_modes.png")
    # joint_pdf_path = os.path.join(OUTDIR, "ot_joint_pair_modes.pdf")

    # fig_joint.savefig(joint_png_path, dpi=300, bbox_inches="tight")
    # # fig_joint.savefig(joint_pdf_path, bbox_inches="tight")

    # plt.show()

    # print(f"Wrote {marg_png_path}")
    # # print(f"Wrote {marg_pdf_path}")
    # print(f"Wrote {joint_png_path}")
    # # print(f"Wrote {joint_pdf_path}")


if __name__ == "__main__":
    main()