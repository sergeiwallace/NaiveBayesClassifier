import pandas as pd
import operator
import math
from cross_validation import cross_validation


class NaiveBayes:
    """
    The Naive Bayes algorithm fits the posterior given training data using the fit method, predict new classes for a set
    of test data using the predict method, and scores those classes against the true classes using the score method.
    
    Bayes Rule: P(C_k, x) = P(C_k) * P(x, C_k) / P(x) => posterior =  prior*likelihood/evidence, we drop the evidence
    since it is constant.
    
    This implementation takes categorical variables and calculates the likelihood directly, using Laplace correction for
    zero probability values.
    """
    def __init__(self):
        self.prior = dict()
        self.likelihood = dict()
        self.posterior = dict()

    def class_probabilities(self, data):
        """
        Given Bayes theorem: P(C_k, x) = P(C_k) * P(x, C_k) / P(x) => posterior =  prior * likelihood / evidence
        This function calculates the prior.

        :param data: the training data as a pandas dataframe of categorical variables
        :return:
        1) the unique class in a dictionary
        2) the number of examples as an integer
        """

        classes = data.ix[:, -1]
        num_examples = data.size
        self.prior = classes.value_counts().to_dict()
        for key in self.prior:
            self.prior[key] /= num_examples
        return classes

    @staticmethod
    def feature_values(data):
        """
        helper function that creates a dictionary of all the unique values for a feature

        :param data: training data as pandas data frame
        :return: dictionary of all the possible values for a feature
        """
        unique_values = dict()
        for name in list(data.columns.values):
            values = data[name].unique()
            unique_values[name] = values
        return unique_values

    def fit(self, tr_data):
        """
        this function fits the naive bayes estimator with some training data (of categorical variables)

        :param tr_data: training data of categorical variables as pandas data frame
        :return: no return value
        """

        # calculate prior
        classes = self.class_probabilities(tr_data)

        # unique values is a dictionary containing all unique values for each feature
        unique_values = self.feature_values(tr_data.ix[:, :-1])

        # calculate posterior
        for label in classes.unique():
            # for each class, separate the data by that class
            data = tr_data.loc[tr_data.ix[:, -1] == label]
            
            # remove the class from the data set
            data = data.ix[:, :-1]
            
            # dictionary to store the likelihoods for each class
            likelihoods = dict()

            for column in data:
                # take each feature separately
                feature = data[column]

                # number of examples with certain class
                num_examples = feature.size

                # use pandas method value_counts to count all the values in that column and put it in a dictionary
                counts = feature.value_counts().to_dict()

                # for each value in that feature, divide by number of examples to get the likelihood of that value
                for key in counts:
                    counts[key] /= num_examples

                # get unique values array for the feature
                values = unique_values[column]

                # use Laplace correction for unique values missing from that feature (for that class)
                # This addresses zero probability issue.
                for attribute in values:
                    if attribute not in counts:
                        counts[attribute] = 1 / (num_examples + 1)

                likelihoods[feature.name] = counts
            self.likelihood[label] = likelihoods
        return

    def predict(self, data):
        """
        predict the labels for the test data

        :param data: test data as pandas dataframe
        :return: predicted labels as a pandas series
        """

        labels = list()
        class_prob = dict()

        # iterate over the rows in the test dataframe
        for index, row in data.iterrows():

            # iterate over the classes
            for key1 in self.prior:
                # the probability initially equals the natural log of the prior for that class
                posterior = math.log(self.prior[key1], math.e)

                # create variable for the likelihood for that class
                likelihood = self.likelihood[key1]

                # calculate the natural log of the posterior
                for key2 in likelihood:

                    # for each feature, add the the natural log of the likelihood for test example's value
                    posterior += math.log(likelihood[key2][row[key2]], math.e)

                # create new class-posterior pair in dictionary for each class
                class_prob[key1] = posterior

            # get the class with the greatest posterior
            label = max(class_prob.items(), key=operator.itemgetter(1))[0]

            # assign that class to the test example
            labels.append(label)
        return pd.Series(labels)

    @staticmethod
    def score(true_labels, predicted_labels):
        """
        calculate the percentage of correctly predicted labels

        :param true_labels: true labels as pandas series/dataframe
        :param pred_labels: predicted labels as pandas series
        :return: return the score as a float
        """
        score = 0

        for i in range(predicted_labels.size):
            if true_labels.iloc[i] == predicted_labels.iloc[i]:
                score += 1
        score /= predicted_labels.size

        return score


def read_data(file, names=0):
    """
    helper function that reads the data into a pandas data frame

    :param file: file path as a string
    :param names: names of column headers as a list of strings
    :return: pandas dataframe
    """
    data = pd.read_csv(file, names=names)
    return data


def main():
    # column names for car dataset
    names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "classes"]

    # read the training and test data into a pandas dataframe
    df_train = read_data("./data/car1.data.csv", names)
    df_test = read_data("./data/car1.test.csv", names)

    # initialize the naive Bayes algorithm
    naive_bayes = NaiveBayes()

    # run naive bayes with cross validation first
    best_model, best_score, average_score = cross_validation(naive_bayes, df_train, 4)
    labels = best_model.predict(df_test.ix[:, :-1])
    score = best_model.score(df_test.ix[:, -1], labels)
    print("cross validated naive bayes - test score =", score)

    # run naive bayes on whole training set
    naive_bayes.fit(df_train)
    labels1 = naive_bayes.predict(df_test.ix[:, :-1])
    score1 = naive_bayes.score(df_test.ix[:, -1], labels1)
    print("naive bayes - test score =", score1)


if __name__ == "__main__":
    main()
