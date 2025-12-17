from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["monthly_payment"] = X["amount"] / X["duration"]

        rate_map = {
            1: 0.2,
            2: 0.35,
            3: 0.5,
            4: 0.75
        }
        X["installment_rate_pct"] = X["installment_rate"].map(rate_map)

        X["income_proxy"] = X["monthly_payment"] / X["installment_rate_pct"]

        return X
