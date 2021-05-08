from sklearn.metrics import balanced_accuracy_score

from prepare import *

if __name__ == '__main__':
    # TRAIN DATA
    df, df_unknown, le = prepare_data(high_level_pred=False)

    X_train, X_test, y_train, y_test, vect, ori_test_txt = create_vecs(df, label_enc=le)

    log_reg = fit_model(X_train, y_train, X_test, y_test, use_weights=True)
    #
    test_preds = le.inverse_transform(log_reg.predict(X_test))
    test_actuals = le.inverse_transform(y_test)

    dff = pd.DataFrame(
        {'preds': test_preds,
         'actuals': test_actuals,
         'txt': ori_test_txt
         })

    print(f'Balanced Accuracy: {balanced_accuracy_score(y_true=y_test, y_pred=log_reg.predict(X_test))}')
    # dff[dff['preds'] != dff['actuals']]






















