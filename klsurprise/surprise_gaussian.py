import numpy as np


class SurpriseGauss:
    """
    Compute the Gaussian Surprise statistic between two distributions.

    Given two distributions (as MCMC chains or as (mean, covariance) tuples),
    this class computes analytical KLD and Surprise assuming both are
    multivariate Gaussians.

    Parameters
    ----------
    chain_2 : ndarray or tuple
        Samples from the posterior p(theta|D2), or a (mean, cov) tuple.
    chain_1 : ndarray or tuple
        Samples from the posterior p(theta|D1), or a (mean, cov) tuple.
    """

    def __init__(self, chain_2, chain_1):
        self.chain_1 = chain_1
        self.mu_1, self.cov_1 = self._infer_stats(chain_1)
        self.chain_2 = chain_2
        self.mu_2, self.cov_2 = self._infer_stats(chain_2)

    @staticmethod
    def _infer_stats(chain):
        """Infer mean and covariance from a chain or (mean, cov) tuple."""
        if isinstance(chain, tuple) and len(chain) == 2:
            mu = chain[0]
            cov = chain[1]
            return mu, cov
        else:
            mu = np.mean(chain, axis=0)
            cov = np.cov(chain, rowvar=False)
        return mu, cov

    def calculate_surprise(self, Nsamples=10000):
        """
        Calculate the Surprise statistic between chain_2 and chain_1: S(p2|p1).

        Parameters
        ----------
        Nsamples : int
            Number of samples to draw from the Surprise distribution.

        Returns
        -------
        Sval : float
            Value of the Surprise statistic.
        Sdist : ndarray
            Surprise distribution (array of length Nsamples).
        """
        Sval = self.surprise((self.mu_2, self.cov_2), (self.mu_1, self.cov_1))
        Sdist = self.S_dist(self.cov_2, self.cov_1, Nsamples)
        return Sval, Sdist

    def calculate_kld(self, Nsamples=10000):
        """
        Calculate the KLD between chain_2 and chain_1: KLD(p2|p1).

        Parameters
        ----------
        Nsamples : int
            Number of samples to draw from the KLD distribution.

        Returns
        -------
        kld_val : float
            Value of the Kullback-Leibler divergence.
        kld_exp : float
            Expected value of the KLD (Eq. A38, Seehars et al. 2014).
        kld_dist : ndarray
            KLD distribution (array of length Nsamples).
        """
        kld_val = self.kld((self.mu_2, self.cov_2), (self.mu_1, self.cov_1))
        kld_exp = self.expected_kld(self.cov_2, self.cov_1)
        Sdist = self.S_dist(self.cov_2, self.cov_1, Nsamples)
        kld_dist = Sdist + kld_exp
        return kld_val, kld_exp, kld_dist

    @staticmethod
    def kld(chain_2, chain_1):
        """
        Compute the analytical KLD between two multivariate Gaussians.

        Parameters
        ----------
        chain_2 : tuple or ndarray
            (mean_2, cov_2) or chain from which to infer them.
        chain_1 : tuple or ndarray
            (mean_1, cov_1) or chain from which to infer them.

        Returns
        -------
        kld : float
            The KLD from distribution 2 to distribution 1.
        """
        mu2, cov2 = SurpriseGauss._infer_stats(chain_2)
        mu1, cov1 = SurpriseGauss._infer_stats(chain_1)

        cov1_inv = np.linalg.inv(cov1)
        k = len(mu2)
        term1 = np.trace(cov1_inv.dot(cov2))
        term2 = (mu2 - mu1).T.dot(cov1_inv).dot(mu2 - mu1)
        term3 = -k
        term4 = np.linalg.slogdet(cov1)[1] - np.linalg.slogdet(cov2)[1]
        kld = 0.5 * (term1 + term2 + term3 + term4)
        return kld

    @staticmethod
    def expected_kld(cov2, cov1):
        """
        Calculate the expected KLD based on Equation A38 from
        Seehars et al. 2014, "Information Gains from Cosmic Microwave
        Background Experiments".

        Parameters
        ----------
        cov2 : ndarray
            Covariance matrix of the second distribution.
        cov1 : ndarray
            Covariance matrix of the first distribution.

        Returns
        -------
        expected_kld : float
            The expected value of the KLD.
        """
        invC1 = np.linalg.inv(cov1)
        term1 = -0.5 * (np.linalg.slogdet(cov2)[1] - np.linalg.slogdet(cov1)[1])
        term2 = np.diag(np.matmul(cov2, invC1)).sum()
        return term1 + term2

    @staticmethod
    def surprise(chain_2, chain_1):
        """
        Calculate the Gaussian Surprise between two multivariate normals.
        S(p(theta|D2), p(theta|D1))

        Parameters
        ----------
        chain_2 : tuple or ndarray
            (mean_2, cov_2) or chain for distribution 2.
        chain_1 : tuple or ndarray
            (mean_1, cov_1) or chain for distribution 1.

        Returns
        -------
        S : float
            The calculated Surprise.
        """
        mu2, cov2 = SurpriseGauss._infer_stats(chain_2)
        mu1, cov1 = SurpriseGauss._infer_stats(chain_1)

        dim = mu2.shape[0]
        cov_inv_1 = np.linalg.inv(cov1)
        var = np.identity(dim) + np.matmul(cov2, cov_inv_1)
        tr_var = np.diag(var).sum()
        dMu = mu2 - mu1
        S = 0.5 * (np.matmul(dMu, np.matmul(cov_inv_1, dMu)) - tr_var)
        return S

    @staticmethod
    def S_dist(cov2, cov1, Nsamples=10000):
        """
        Generate a distribution of the Surprise metric by sampling from
        a chi-square distribution weighted by eigenvalues.

        Parameters
        ----------
        cov2 : ndarray
            Covariance matrix of distribution 2.
        cov1 : ndarray
            Covariance matrix of distribution 1.
        Nsamples : int
            Number of samples to generate.

        Returns
        -------
        dist : ndarray
            Array of sampled Surprise values.
        """
        cov_inv_1 = np.linalg.inv(cov1)
        var = np.identity(cov1.shape[0]) + np.matmul(cov2, cov_inv_1)

        eigenvals = np.linalg.eig(var)[0]
        dist = np.zeros(Nsamples)
        for eig in eigenvals:
            dist += 0.5 * eig * (np.random.chisquare(1, Nsamples) - 1)
        return dist
