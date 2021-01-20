import matplotlib.pyplot as plt
from scipy.stats import expon, rv_continuous


if __name__ == '__main__':
    lambda_ = 1  # Once per year, on average
    X = expon.rvs(scale=1/lambda_, size=100, random_state=1)

    plt.hist(X)
    plt.show()

    _, expon_mean = expon.fit(X)
    print("Guessed Lambda:", 1/expon_mean)
