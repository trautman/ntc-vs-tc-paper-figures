import numpy as np
import ot


def logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


def solve_joint_kl(p_h, p_r, C, lam=0.9):
    """
    Analytic joint-KL coupling.

    Solves:
        argmin_gamma <C,gamma> + lam KL(gamma || p_h p_r)

    with no marginal constraints.
    """
    gamma_ind = np.outer(p_h, p_r)
    log_gamma = np.log(gamma_ind + 1e-300) - C / lam
    return np.exp(log_gamma - logsumexp(log_gamma.ravel()))


def solve_sinkhorn_ot(p_h, p_r, C, reg=0.9):
    """
    Entropic / Sinkhorn balanced OT.

    Solves:
        argmin_{gamma in Pi(p_h,p_r)}
            <C,gamma> + reg KL(gamma || p_h p_r)
    """
    gamma = ot.sinkhorn(p_h, p_r, C, reg=reg)
    return gamma / gamma.sum()


def solve_balanced_ot(p_h, p_r, C):
    """
    Balanced OT.

    Solves:
        argmin_{gamma in Pi(p_h,p_r)} <C,gamma>
    """
    gamma = ot.emd(p_h, p_r, C)
    return gamma / gamma.sum()


def solve_marginal_kl(p_h, p_r, C, lam_h=0.9, lam_r=0.9, reg=1e-2):
    """
    Marginal-KL / unbalanced OT.

    Solves:
        argmin_gamma
            <C,gamma>
            + lam_h KL(gamma_h || p_h)
            + lam_r KL(gamma_r || p_r)
    """
    gamma = ot.unbalanced.sinkhorn_unbalanced(
        p_h,
        p_r,
        C,
        reg=reg,
        reg_m=(lam_h, lam_r),
    )
    return gamma / gamma.sum()