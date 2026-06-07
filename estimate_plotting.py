def plot_estimate_pair(
    ax,
    H,
    R,
    pair,
    h_color="black",
    r_color="black",
    lw=3.0,
    linestyle="-",
    alpha=1.0,
    zorder=80,
):
    """
    Overlay a single estimated trajectory pair.
    """
    i, j = pair

    ax.plot(
        H[i][:, 0],
        H[i][:, 1],
        color=h_color,
        linewidth=lw,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
    )

    ax.plot(
        R[j][:, 0],
        R[j][:, 1],
        color=r_color,
        linewidth=lw,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
    )