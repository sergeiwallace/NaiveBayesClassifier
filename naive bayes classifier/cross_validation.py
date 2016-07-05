import numpy as np
import pandas as pd


def cross_validation(model, data, k):
    """
    function to perform cross-validation.
    :param model: a model as a python class. It must have fit, predict, and score methods
    :param data: training data as pandas dataframe
    :param k: numpy of folds for cross-validation
    :return: the best performing model, it's score, and the average score of the folds
    """

    # split the data into k separate data frames
    k_folds = np.array_split(data, k)

    # initialize the score and model variables
    scores = list()
    best_score = 0
    best_model = model

    for i in range(len(k_folds)):
        # create new list for the k - 1 folds
        k_min_1_folds = list(k_folds)

        # pop off the ith dataframe and assign it as the validation fold
        test_fold = k_min_1_folds.pop(i)

        # concatenate the remaining folds and assign it as the training fold
        train_folds = pd.concat(k_min_1_folds)

        # train the algorithm
        model.fit(train_folds)

        # make predictions on validation set
        labels = model.predict(test_fold.ix[:, :-1])

        # get the validation score
        score = model.score(test_fold.ix[:, -1], labels)

        print("\nscore with fold " + str(i+1) + " as validation is:", score)
        scores.append(score)

        # if the validation score is higher than the previous best score, assign it as new best score and save the model
        if score > best_score:
            best_score = score
            best_model = model

    # take the average of the validation scores
    average_score = float(sum(scores))/float(len(scores))
    print("\naverage of scores = ", average_score)

    return best_model, best_score, average_score

