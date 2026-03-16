import numpy as np
import jax.numpy as jnp

from .likelihood import logL2_jitted
from .sampling import load_create_NS_file
from .kld import KLD_numerical, compute_kld_distribution
from .ppd import create_ppd_chain
from .io import save_dict_to_hdf5
from .utils import sampler, find_pval, sigma_discordance


class surprise_statistics:
    """
    Computes the surprise statistics between two datasets or models using Nested Sampling.

    This function calculates the surprise (S) distribution and, optionally, the S value between two datasets or models
    by performing Nested Sampling. It allows for comparing the likelihoods and data models within a specified parameter
    domain using a top-hat prior.

    Computes the Surprise statistics S(p(x|D2)||p(x|D1))

    Args:
        logL1 (callable):
            A function that computes the log-likelihood of data_1 given parameters (theta).
            Signature: logL1(theta) -> float
        data_2_model_fun (callable):
            A function that maps parameters (theta) to the data space (D) for data_2.
            Signature: data_2_model_fun(theta) -> array_like
        covariance_matrix_2 (array_like):
            The covariance matrix of data_2 in the data space. The likelihood is assumed to be Gaussian.
        domain (array_like):
            The parameter space boundaries defining the top-hat prior for Nested Sampling.
            Should be a Nx2 array, e.g. np.array([[0.6,0.9],[0.2, 1.5]])
        data_2 (array_like, optional):
            The data vector for data_2. If provided, the function will perform Nested Sampling on likelihood 2
            using a top-hat prior defined by domain. Necessary to compute S.If None, only the S distribution
            will be provided. Default is None.
        data_1_name (str, optional):
            The name identifier for data_1. If provided, the Nested Sampling results for data_1 will be saved
            using this name. If None, results will not be saved. Default is None.
        data_2_name (str, optional):
            The name identifier for data_2. If provided, the Nested Sampling results for data_2 will be saved
            using this name. If None, results will not be saved. Default is None.

    Notes:
        - The code assumes Gaussian likelihoods.
        - The code assumes top-hat priors defined by domain.
        - For now, it's necessary that the likelihoods provided are jax compatible. So the code can't work with some
          external likelihoods like Planck.
    """

    def __init__(
        self,
        logL1,
        data_2_model_fun,
        covariance_matrix_2,
        domain,
        data_2=None,
        data_1_name=None,
        data_2_name=None,
        init_NS=False,
        Nppd=None,
    ):
        """
        Initializes the SurpriseStatistics class with the provided parameters.
        """
        self.logL1 = logL1
        self.data_2_model_fun = data_2_model_fun
        self.covariance_matrix_2 = covariance_matrix_2
        self.domain = domain
        self.data_2 = data_2
        self.data_1_name = data_1_name
        self.data_2_name = data_2_name
        self.ndim = domain.shape[0]

        self.chol_cov_2 = jnp.linalg.cholesky(covariance_matrix_2)
        self.res_1, self.res_2 = None, None
        self.PPD_chain = None

        if init_NS:
            self.__initialize_NS__()

    def __initialize_NS__(self):
        self.res_1 = load_create_NS_file(
            self.data_1_name, self.logL1, self.ndim, self.domain
        )
        try:
            self.res_2 = load_create_NS_file(
                self.data_2_name, self.logP2, self.ndim, self.domain
            )
        except Exception:
            print("Could neither load or create the posterior NS estimate for data D2.")

    def __initialize_PPD__(self, Nppd, n_jobs=1, sample_size=1):
        if self.res_1 is None:
            self.res_1 = load_create_NS_file(
                self.data_1_name, self.logL1, self.ndim, self.domain
            )
        self.th1_samples = sampler(
            self.res_1.samples_equal(), Nppd
        )  # we take a subset of samples with size Nkld
        self.PPD_chain = create_ppd_chain(
            th1_samples=self.th1_samples,
            data_model_fun=self.data_2_model_fun,
            cov_matrix=self.covariance_matrix_2,
            sample_size=sample_size,
            n_jobs=n_jobs,
        )

    def logL2(self, theta, D):
        """
        Compute the log-likelihood of theta for the multivariate normal distribution.

        Args:
        - theta: parameter space vector (n-dimensional vector).
        - D: Data space vector

        Returns:
        - The log-likelihood of theta, D.
        """
        data_2_model = self.data_2_model_fun(theta)
        return logL2_jitted(data_2_model, self.chol_cov_2, D)

    def logP2(self, theta):
        return self.logL2(theta, self.data_2)

    def surprise_function_call(
        self, Nkld, result_path=None, n_effective=15000, n_jobs=-1, verbose=1
    ):
        """
        Compute the Kullback-Leibler Divergence (KLD) distribution and optionally calculate the surprise statistic
        if a second dataset is provided.

        This function loads or creates posterior samples using Nested Sampling (NS) for a first dataset (D1)
        and then load or create a Posterior Predictive Distribution (PPD). If a second dataset (D2) is provided,
        it also computes the KLD between the posterior distributions of D1 and D2, returning the surprise statistic.

        Parameters:
        -----------
        Nkld : int
            The number of KLD samples to be drawn from the Posterior Predictive Distribution (PPD).

        result_path : str
            The path where the results will be saved (in HDF5 format).

        n_effective : int, optional (default=15000)
            The number of effective samples to target for the nested sampling run.

        n_jobs : int, optional (default=-1)
            The number of parallel jobs to run. Set to -1 to use all available cores.

        verbose : int, optional (default=1)
            Level of verbosity. Set to 0 for silent mode, higher values for more verbose output.

        Returns:
        --------
        results_dic : dict
            A dictionary containing the computed KLD values and the surprise statistic (if applicable).
            The contents of the dictionary vary depending on whether a second dataset is provided. The keys include:
            - 'S': The computed surprise statistic (if D2 is provided).
            - 'S_dist': The distribution of surprise statistics.
            - 'kld21': KLD(p2 | p1), i.e., the KLD between the posterior distributions of D1 and D2 (if D2 is provided).
            - 'kld_exp': The expected KLD (mean value of the KLD distribution).
            - 'kld_dist': The distribution of KLD samples.
            - 'p_value': The p-value associated with the surprise statistic (if D2 is provided).
            - 'sigma_discordance': The sigma-level discordance between the two datasets (if D2 is provided).

        Notes:
        ------
        - The function first computes the KLD between the posterior distribution and the PPD for dataset D1.
        - If a second dataset (D2) is provided, the KLD between the posterior distributions of D1 and D2 is also computed.
        - The function calculates the surprise statistic using the expected KLD and compares it to the KLD of D2.
        """

        print("Handling dataset 1...")
        print(70 * "_")
        ############ loading/creating mock 1 and 2 ############
        if self.res_1 is None:
            self.__initialize_NS__()
        print("Done!")

        print("")
        print("Handling posterior predictive distribution PPD(D2|D1) ...")
        print(70 * "_")

        ############ create posterior predictive distribution ############
        if self.PPD_chain is None:
            self.__initialize_PPD__(Nkld)
        else:
            Nkld = self.PPD_chain.shape[0]
            print("Will sample KLD the same size as PPD.\nNkld = ", Nkld)

        print("Handling KLD distribution...")
        print(70 * "_")
        kld_samples = compute_kld_distribution(
            self.PPD_chain,
            self.logL2,
            self.logL1,
            self.ndim,
            self.domain,
            self.res_1,
            logP_1=self.logL1,
            n_jobs=n_jobs,
            n_effective=n_effective,
        )

        kld_array = jnp.array(kld_samples)
        # kld_array = np.array(kld_samples)
        kld_exp = kld_array.mean()
        S_dist = kld_array - kld_exp

        # if data 2 is provided
        if self.res_2 is not None:
            kld_value = KLD_numerical(
                self.res_2, self.logP2, self.res_1, self.logL1, domain=self.domain
            )
            S = kld_value - kld_exp
            p_value = find_pval(S_dist, S, verbose=0)
            sigma_disc = sigma_discordance(p_value)
            if verbose > 0:
                print("S = {:.2f} nats".format(S))
                print("<KLD> = {:.2f} nats".format(kld_exp))
                print("KLD = {:.2f} nats".format(kld_value))
                print("p-val = {:.2f} nats".format(p_value))
            results_dic = {
                "domain": self.domain,
                "S": S,
                "S_dist": S_dist,
                "kld21": kld_value,
                "kld_exp": kld_exp,
                "kld_dist": kld_array,
                "p_value": p_value,
                "sigma_discordance": sigma_disc,
            }
        else:
            if verbose > 0:
                print("<KLD> = {:.2f} nats".format(kld_exp))
            results_dic = {
                "domain": self.domain,
                "S_dist": S_dist,
                "kld_exp": kld_exp,
                "kld_dist": kld_array,
            }
        if result_path is not None:
            print("Saving results to ", result_path)
            save_dict_to_hdf5(result_path, results_dic)

        return results_dic
