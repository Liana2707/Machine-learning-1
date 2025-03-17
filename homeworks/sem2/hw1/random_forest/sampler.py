import numpy as np


class BaseSampler:
    def __init__(self, max_samples=1.0, bootstrap=False, random_state=None):
        """
        Parameters
        ----------
        bootstrap : Boolean
            if True then use bootstrap sampling
        max_samples : float in [0;1]
            proportion of sampled examples
        """
        self.random_state = np.random.RandomState(random_state)
        self.bootstrap = bootstrap
        self.max_samples = max_samples

    def sample_indices(self, n_objects):
        """
        Parameters
        ----------
        n_objects : int > 0
            number of sampling objects
        """
        n_size = int(n_objects * self.max_samples)
        indices = self.random_state.choice(n_objects, size=n_size, replace=self.bootstrap)
        return indices

    def sample(self, x, y=None):
        # abstract method
        raise NotImplementedError


class ObjectSampler(BaseSampler):
    def __init__(self, max_samples=1.0, bootstrap=True, random_state=None):
        super().__init__(max_samples=max_samples, bootstrap=bootstrap, random_state=random_state)

    def sample(self, x, y=None):
        """
        Parameters
        ----------
        x : numpy ndarray of shape (n_objects, n_features)
        y : numpy ndarray of shape (n_objects,)

        Returns
        -------
        x_sampled, y_sampled : numpy ndarrays of shape (n_samples, n_features) and (n_samples,)
        """
        indices = self.sample_indices(x.shape[0])
        x_sampled = x[indices]
        if y is not None:
            y_sampled = y[indices]
        else:
            y_sampled = None

        return x_sampled, y_sampled


class FeatureSampler(BaseSampler):
    def __init__(self, max_samples=1.0, bootstrap=True, random_state=None):
        super().__init__(max_samples=max_samples, bootstrap=bootstrap, random_state=random_state)

    def sample(self, x, y=None):
        """
        Parameters
        ----------
        x : numpy ndarray of shape (n_objects, n_features)
        y : numpy ndarray of shape (n_objects,)

        Returns
        -------
        x_sampled : numpy ndarrays of shape (n_objects, n_features_sampled)
        """
        indices = self.sample_indices(x.shape[1])
        return x[:, indices]
    