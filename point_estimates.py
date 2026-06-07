import numpy as np


def compute_independent_point_estimate(p_h, p_r):
    """
    Pure individual trajectory point estimate.

    Solves:
        h_ind = argmax_h p_h(h)
        r_ind = argmax_r p_r(r)

    This ignores the joint task cost.
    """
    i = int(np.argmax(p_h))
    j = int(np.argmax(p_r))
    return i, j


def compute_collaborative_point_estimate(p_h, p_r, C, lam=0.9, eps=1e-300):
    """
    Collaborative trajectory point estimate.

    Solves:
        argmax_{h,r} log p_h(h) + log p_r(r) - C(h,r)/lam

    Equivalently:
        argmax_{h,r} p_h(h) p_r(r) exp(-C(h,r)/lam)

    This is the single best joint trajectory pair under the
    preference-weighted task objective.
    """
    score = (
        np.log(p_h[:, None] + eps)
        + np.log(p_r[None, :] + eps)
        - C / lam
    )

    flat_idx = int(np.argmax(score))
    i, j = np.unravel_index(flat_idx, C.shape)

    return i, j, score[i, j]


def compute_cost_only_point_estimate(C):
    """
    Pure task-cost point estimate.

    Solves:
        argmin_{h,r} C(h,r)

    This ignores the marginal preference models.
    """
    flat_idx = int(np.argmin(C))
    i, j = np.unravel_index(flat_idx, C.shape)

    return i, j, C[i, j]


def compute_joint_argmax(gamma):
    """
    Joint MAP estimate from a coupling.

    Solves:
        argmax_{h,r} gamma(h,r)

    This extracts a single point estimate from the full joint model.
    """
    flat_idx = int(np.argmax(gamma))
    i, j = np.unravel_index(flat_idx, gamma.shape)

    return i, j, gamma[i, j]