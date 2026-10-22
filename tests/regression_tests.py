import sys
import unittest

import numpy as np


STUDENT = {}


class TestRegressionModels(unittest.TestCase):

    def setUp(self):
        """
        Set up mock data for testing purposes.
        """
        self.X_linear = np.array([1, 2, 3, 4, 5])
        self.y_linear = 2 * self.X_linear + 3

    def student(self, name):
        return STUDENT[name]

    def test_linear_regression_fit(self):
        """
        Test the LinearRegression `fit` method.
        """
        model = self.student('LinearRegression')()
        model.fit(self.X_linear, self.y_linear)

        # Check if the coefficients are close to the expected values [3, 2]
        expected_coefficients = np.array([3, 2])
        np.testing.assert_allclose(model.coefficients, expected_coefficients, rtol=1e-2, atol=1e-2,
                                   err_msg="LinearRegression coefficients do not match the expected values.")

    def test_linear_regression_predict(self):
        """
        Test the LinearRegression `predict` method.
        """
        model = self.student('LinearRegression')()
        model.fit(self.X_linear, self.y_linear)

        # Predictions should closely match the original labels
        predictions = model.predict(self.X_linear)
        np.testing.assert_allclose(predictions, self.y_linear, rtol=1e-2, atol=1e-2,
                                   err_msg="LinearRegression predictions do not match the expected values.")

    def test_linear_regression_shape_robustness(self):
        """
        LinearRegression should handle both 1D and column-vector inputs.
        """
        X_col = self.X_linear.reshape(-1, 1)
        model = self.student('LinearRegression')()
        model.fit(X_col, self.y_linear)
        preds = model.predict(X_col)

        np.testing.assert_allclose(preds, self.y_linear, rtol=1e-2, atol=1e-2,
                                   err_msg="LinearRegression failed on column-vector input shape.")

    def test_linear_regression_non_uniform_data(self):
        """
        Test LinearRegression on non-uniformly spaced data.
        """
        X = np.array([0, 2, 5, 7, 9])
        y = 4 * X - 1

        model = self.student('LinearRegression')()
        model.fit(X, y)

        np.testing.assert_allclose(model.coefficients, np.array([-1, 4]), rtol=1e-2, atol=1e-2,
                                   err_msg="LinearRegression failed on non-uniform input data.")

    def test_ridge_regression_fit(self):
        """
        Test the RidgeRegression `fit` method.
        """
        model = self.student('RidgeRegression')(beta=0.1)
        model.fit(self.X_linear, self.y_linear)

        # Check if the coefficients are close to expected values (with regularization)
        expected_coefficients = np.array([2.758436, 2.062132])
        np.testing.assert_allclose(model.coefficients, expected_coefficients, rtol=1e-2, atol=1e-2,
                                   err_msg="RidgeRegression coefficients do not match the expected values.")

    def test_ridge_regression_predict(self):
        """
        Test the RidgeRegression `predict` method.
        """
        model = self.student('RidgeRegression')(beta=0.1)
        model.fit(self.X_linear, self.y_linear)

        predictions = model.predict(self.X_linear)
        np.testing.assert_allclose(predictions, [4.820568, 6.8827, 8.944831, 11.006963, 13.069095],
                                   rtol=1e-2, atol=1e-2,
                                   err_msg="RidgeRegression predictions do not match the expected values.")

    def test_ridge_regression_shrinks_coefficients(self):
        """
        RidgeRegression should shrink coefficients compared to LinearRegression.
        """
        lin = self.student('LinearRegression')()
        lin.fit(self.X_linear, self.y_linear)

        ridge = self.student('RidgeRegression')(beta=1.0)
        ridge.fit(self.X_linear, self.y_linear)

        self.assertLess(np.linalg.norm(ridge.coefficients), np.linalg.norm(lin.coefficients),
                        msg="RidgeRegression does not shrink coefficients as expected.")

    def test_make_lasso_uses_beta(self):
        """
        make_lasso should pass beta to scikit-learn's alpha.
        """
        model = self.student('make_lasso')(beta=0.5)
        self.assertAlmostEqual(model.alpha, 0.5, msg="make_lasso should set alpha=beta.")

    def test_evaluate_cv_runs(self):
        """
        evaluate_cv should return one row per candidate with nonnegative validation MSE.
        """
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 2))
        y = 1 + X[:, 0] - 2 * X[:, 1]
        results = self.student('evaluate_cv')(
            X, y, ['a', 'b'], {'both': ['a', 'b']}, degrees=[1], betas=[0.1, 1.0])

        self.assertEqual(len(results), 4, msg="Expected one row per model and beta.")
        self.assertTrue((results['mean_validation_mse'] >= 0).all(),
                        msg="Validation MSE should be nonnegative.")


def run_all_tests(ns):
    """Run the supplied tests against your notebook code.

    Pass `globals()` from your notebook so the tests can find your implementations.
    """
    STUDENT.clear()
    STUDENT.update(ns)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestRegressionModels)
    return unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
