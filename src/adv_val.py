import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection
from sklearn import preprocessing

def adv_val(
    train,
    test,
    feats,
    cat_feats,
):
    target = np.hstack([np.zeros(train.shape[0]), np.ones(test.shape[0])])
    train = pd.concat([train, test], axis=0)
    train = train[feats]

    for col in cat_feats:
        if train[col].dtype == object:
            encoder = preprocessing.LabelEncoder()
            train[col] = encoder.fit_transform(train[col].to_list())

    train, test, y_train, y_test = model_selection.train_test_split(
        train, target, test_size=0.33, random_state=1, shuffle=True
    )

    train = lgb.Dataset(train, label=y_train, categorical_feature=cat_feats)

    test = lgb.Dataset(test, label=y_test, categorical_feature=cat_feats)

    param = {
        "objective": "binary",
        "learning_rate": 0.01,
        "boosting": "gbdt",
        "metric": "auc",
        "verbosity": -1,
    }

    clf = lgb.train(
        param,
        train,
        num_boost_round=200,
        valid_sets=[train, test],
        verbose_eval=50,
        early_stopping_rounds=50,
    )

    feature_imp = pd.DataFrame(
        sorted(
            zip(
                clf.feature_importance(
                    importance_type="gain",
                ),
                clf.feature_name(),
            )
        ),
        columns=["Feature Split Total Gain", "Feature"],
    )

    plt.figure(figsize=(8, 8))
    sns.barplot(
        x="Feature Split Total Gain",
        y="Feature",
        data=feature_imp.sort_values(
            by="Feature Split Total Gain", ascending=False
        ).head(100),
    )

    plt.title("LightGBM - Feature Importance")
    plt.tight_layout()
    plt.show()

    return clf, feature_imp
