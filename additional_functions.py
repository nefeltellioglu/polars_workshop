import numpy as np


def death_cdf(propn_die):
    """
    Return an array that contains the cumulative death proportion.

    >>> death_cdf([0.5, 0.5, 0.5, 0.5])
    array([0.    , 0.5   , 0.75  , 0.875 , 0.9375])
    """
    intervals = np.zeros(len(propn_die) + 1)
    for ix, propn in enumerate(propn_die):
        prev_propn = intervals[ix]
        intervals[ix + 1] = prev_propn + propn * (1 - prev_propn)
    return intervals


def pick_age_death_given_cdf(ages, cdf_bins, rng):
    """
    Sample the age at death for individuals of any current age.

    Note that:

    (a) We set the lower bound for each roll to the person's current age;

    (b) We clip the samples so that people who would live to be 100 instead
    die at 99, after which we add a random number of days, ensuring everyone
    dies before the age of 100.
    """
    age_locns = rng.uniform(low=cdf_bins[ages.astype(int)]).clip(
        max=cdf_bins[:-1].max()
    )
    dead_mask = age_locns.reshape((-1, 1)) <= cdf_bins.reshape((1, -1))
    (_, death_ages) = np.nonzero(np.diff(dead_mask))
    return death_ages


def pick_ages_at_death_given_ages(death_rates_csv, ages, rng):
    """
    Pick date of death given age of each agent, using
    age-specific yearly death rates from ABS
    """
    with open(death_rates_csv) as f:
        _, proportions = np.genfromtxt(f, delimiter=",", skip_header=1, unpack=True)

    cdf_bins = death_cdf(proportions)
    death_ages = pick_age_death_given_cdf(ages, cdf_bins, rng)

    return death_ages
