from sklearn.ensemble import RandomForestClassifier
import neptune.new.integrations.sklearn as npt_utils

run = neptune.init(project='common/sklearn-integration',
                   api_token='ANONYMOUS',
                   name='other-options')

rfc = RandomForestClassifier()

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=28743)

rfc.fit(X_train, y_train)

# Log parameters from the model
run['estimator/parameters'] = npt_utils.get_estimator_params(rfc)

# Log pickled model
run['estimator/pickled-model'] = npt_utils.get_pickled_model(rfc)

# Log confusion matrix
run['confusion-matrix'] = npt_utils.create_confusion_matrix_chart(rfc, X_train, X_test, y_train, y_test)

