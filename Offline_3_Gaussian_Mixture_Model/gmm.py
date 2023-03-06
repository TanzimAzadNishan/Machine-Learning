import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
import matplotlib.transforms as transforms
from scipy.stats import multivariate_normal
from sklearn.decomposition import TruncatedSVD

sns.set()


def gaussian(x, mu, sigma):

    mu = mu + 1e-6

    # add 1e-6 to diagonal elements of sigma
    # for i in range(sigma.shape[0]):
    #     sigma[i][i] = sigma[i][i] + 1e-6
    sigma = sigma + 1e-6

    pdf_prob = multivariate_normal.pdf(x, mean=mu, cov=sigma, allow_singular=True)

    return pdf_prob


def log_likelihood(data, K, weights, means, covariances):
    n, m = data.shape
    log_likelihood = 0.0

    pdf_prob = [gaussian(data, means[j], covariances[j]) for j in range(K)]

    for i in range(n):
        ith_log_likelihood = sum([weights[j] * pdf_prob[j][i] for j in range(K)])
        log_likelihood += np.log(ith_log_likelihood)

    
    return log_likelihood



def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2.25 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2.25 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ellipse = Ellipse(xy=position, width=nsig * width, height=nsig * height, angle=angle, fill=True, 
                fc='royalblue', ec='black', zorder=2, **kwargs)

        ellipse.set_alpha(0.4)
        ax.add_patch(ellipse)



def plot_gmm(data, weights, means, covariances):

    ax = plt.gca()
    ax.scatter(data[:, 0], data[:, 1], s=10, zorder=2, edgecolor='red', alpha=0.8)
    ax.axis('equal')

    max_weight = np.max(weights)

    w_factor = 0.2 / max_weight
    
    for pos, covar, w in zip(means, covariances, weights):
        draw_ellipse(pos, covar, alpha=w * w_factor)



def plot_contour(df, means, covariances, K, prob):

    plt.scatter(df[:, 0], df[:, 1] , c=prob.argmax(axis=1), cmap='viridis', s=40, edgecolor='k', alpha=0.5)
    x, y = np.mgrid[np.min(df[:, 0]):np.max(df[:, 0]):.01, np.min(df[:, 1]):np.max(df[:, 1]):.01]
    positions = np.dstack((x, y))
    
    for j in range(K):
        rv = gaussian(positions, means[j], covariances[j])
        plt.contour(x, y, rv, colors='black', alpha=0.5, linewidths=1)

    

def EM_algorithm(data, K, max_itr=100, tol=1e-3, plot=False):
    n, m = data.shape

    # Initialize parameters
    weights = np.ones(K) / K
    # means = data[np.random.choice(n, K, replace=False)]
    means = np.random.randn(K, m)

    # np.eye(m) is a m*m identity matrix
    covariances = np.array([np.eye(m) for _ in range(K)])

    # Initialize log likelihood
    log_likelihoods = log_likelihood(data, K, weights, means, covariances)

    itr_count = 0
    all_weights = []
    all_means = []
    all_covariances = []

    # repeat until convergence
    while True:
        
        if plot:
            all_weights.append([x for x in weights])
            all_means.append([[x for x in means[j]] for j in range(K)])
            all_covariances.append([[[y for y in x] for x in covariances[j]] for j in range(K)])

        # Expectation step(E)
        latents = np.zeros((n, K))
        pdf_prob = [gaussian(data, means[j], covariances[j]) for j in range(K)]

        for i in range(n):
            for j in range(K):
                latents[i, j] = weights[j] * pdf_prob[j][i]
            
            latents[i] /= sum(latents[i])

        # Maximization step(M)
        for j in range(K):
            weights[j] = sum(latents[:, j]) / n
            means[j] = 0.0

            current_latent = sum(latents[:, j])

            for i in range(n):
                means[j] += latents[i, j] * data[i]
            means[j] /= current_latent

            covariances[j] = 0.0
            for i in range(n):
                covariances[j] += latents[i, j] * (data[i] - means[j]).reshape(m, 1).dot((data[i] - means[j]).reshape(1, m))
            
            covariances[j] /= current_latent


        # Check for convergence
        log_likelihoods_new = log_likelihood(data, K, weights, means, covariances)

        if abs(log_likelihoods_new - log_likelihoods) < tol:
            break
        
        log_likelihoods = log_likelihoods_new

        itr_count += 1
        if itr_count > max_itr:
            break
        
    
    if plot:
        return all_weights, all_means, all_covariances, log_likelihoods, itr_count
    
    return weights, means, covariances, log_likelihoods, itr_count




if __name__ == '__main__':

    fileName = 'data2D_B1.txt'
    # data = pd.read_csv(f'./Datasets/{fileName}', sep=' ', header=None)
    data = pd.read_csv(f'./Datasets/{fileName}', delimiter=r'\s+', header=None)
    # data = (data - data.mean()) / data.std()

    data = data.values
    print(data.shape)

    K_max = 10
    em_itr = 5 # Number of times EM algorithm is run for each K
    max_itr = 30 # Maximum number of iterations for EM algorithm

    converged_log_likelihoods = []
    K_values = [x for x in range(1, K_max + 1)]

    for j in K_values:

        log_likelihoods = -np.inf

        for i in range(em_itr):
            weights, means, covariances, log, itr_count = EM_algorithm(data, j, max_itr=max_itr)

            if log > log_likelihoods:
                log_likelihoods = log

        # weights, means, covariances, log_likelihoods, itr_count = EM_algorithm(data, j)
        log_likelihoods /= data.shape[0]
        converged_log_likelihoods.append(log_likelihoods)

        print(f'For k = {j}: Log Likelihood: {log_likelihoods}, Iterations: {itr_count}')

    
    # Plot the converged log likelihoods against K
    plt.figure(figsize=(10, 8))
    plt.plot(K_values, converged_log_likelihoods)
    plt.title(f'Converged Log Likelihoods vs K for dataset {fileName}')
    plt.xlabel('K')
    plt.ylabel('Converged Log Likelihood')
    plt.xticks(K_values)
    plt.savefig('converged_log_likelihoods.png')
    plt.close()

    k_star = int(input('Enter the value of k*: '))

    n,m = data.shape

    if m > 2:
        print("Before reduction", data.shape)
        pca = PCA(n_components=2, whiten=True) 
        data = pca.fit_transform(data)
        print("After reduction", data.shape)

    # Plot the GMMs for k = k*
    plt.figure(figsize=(7, 5))


    log_likelihoods = -np.inf
    k_star_weights = []
    k_star_means = []
    k_star_covariances = []

    for i in range(em_itr):
        weights, means, covariances, log, itr_count = EM_algorithm(data, k_star, max_itr=max_itr, plot=True)
        
        if log > log_likelihoods:
            log_likelihoods = log
            k_star_weights = weights
            k_star_means = means
            k_star_covariances = covariances

    # weights, means, covariances, log_likelihoods, itr_count = EM_algorithm(data, k_star, plot=True)
    log_likelihoods /= n

    total_itr = len(k_star_weights)

    for i in range(total_itr):

        weights = np.array(k_star_weights[i])
        means = np.array(k_star_means[i])
        covariances = np.array(k_star_covariances[i])

        # plt.ion()
        plt.clf()
        plot_gmm(data, weights, means, covariances)
        # plot_contour(data, means, covariances, K, latents)
        plt.draw()
        plt.pause(0.5)
        # plt.ioff()
        

    plt.title(f'GMM-EM {k_star} Gaussians')
    plt.savefig(f'clusters_online_{m}D.png')

    weights = k_star_weights[-1]
    means = k_star_means[-1]
    covariances = k_star_covariances[-1]
    itr_count = total_itr

    # print('Weights: ', weights)
    # print('Means: ', means)
    # print('Covariances: ', covariances)
    print('Log Likelihood: ', log_likelihoods)
    print('Iteration Count: ', itr_count)

