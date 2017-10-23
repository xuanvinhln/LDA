"""
LDA inferrence
"""
import numpy as np
from scipy.special import psi, gammaln
from scipy import log, exp

# Compute log likelihood
def compute_lower_bound(doc, model, phi, var_gamma, beta_ref_doc):
    """
    Compute Lower Bound
    """
    dig = psi(var_gamma)
    var_gamma_sum = var_gamma.sum()
    digsum = psi(var_gamma_sum)

    # Tinh tong 1
    likelihood = gammaln(model.alpha * model.num_topics)\
                - model.num_topics * gammaln(model.alpha)\
                - gammaln(var_gamma_sum)
    # Tinh tong 2
    likelihood += ((model.alpha - 1) * (dig - digsum)\
                + gammaln(var_gamma)\
                - (var_gamma-1) * (dig-digsum)).sum()
    # Tinh tong 3
    likelihood += (doc.counts\
                * np.nan_to_num(phi * ((dig - digsum) - log(phi)\
                + beta_ref_doc.T)).sum(axis=1)).sum()
    return likelihood

# variational inference
def run_inference(doc, model, var_gamma, phi):
    """
    Inference gamma and phi
    """
    var_max_iter = 50
    var_converged = 1e-6

    converged = 1.0
    lower_bound_old = 0.0

    # compute posterior dirichlet
    # Khoi tao gia tri cho gamma va phi
    # Ma tran phi[Nd, K]
    # phi.fill(1.0/model.num_topics)
    # Mang gamma[K]
    var_gamma.fill(model.alpha + doc.total/float(model.num_topics))
    digamma_gam = psi(var_gamma)

    # Tao logbeta tham chieu cua tung doc
    beta_ref_doc = np.empty((model.num_topics, doc.length))
    for n in xrange(doc.length):
        beta_ref_doc[:, n] = model.log_prob_w[:, doc.words[n]]

    # Cap nhat gamma va phi
    var_iter = 0
    while (converged > var_converged) and (var_iter < var_max_iter or var_max_iter == -1):
        var_iter += 1
        # Cach moi cap nhat phi
        phi_new = exp(digamma_gam + beta_ref_doc.T)
        phisum = phi_new.sum(axis=1)
        phi_new = (phi_new.T / phisum).T

        # gamma chuan la the nay, nhung cap nhat trong vong lap n se hoi tu nhanh hon
        var_gamma = model.alpha + (doc.counts * phi_new.T).sum(axis=1)
        digamma_gam = psi(var_gamma)

        lower_bound = compute_lower_bound(doc, model, phi_new, var_gamma, beta_ref_doc)
        assert not np.isnan(lower_bound)
        converged = (lower_bound_old - lower_bound)/lower_bound_old
        lower_bound_old = lower_bound
        # print "[LDA INF] iter={:2d}\t{:8.5f}\t{:1.3e}".format(var_iter, lower_bound, converged)

    for n in xrange(doc.length):
        phi[n] = phi_new[n]

    return lower_bound
