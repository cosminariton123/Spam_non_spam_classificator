from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import numpy as np
import random
import copy


class Bag_of_words:

    def __init__(self):
        self.vocabulary = {}
        self.words = []
        self.vocabulary_length = 0

    def build_vocabulary(self, data):
        for document in data:
            for word in document:
                if word not in self.vocabulary.keys():
                    self.vocabulary[word] = len(self.vocabulary)
                    self.words.append(word)

        self.vocabulary_length = len(self.vocabulary)
        self.words = np.array(self.words)
        
    def get_features(self, data):
        features = np.zeros((len(data), self.vocabulary_length))

        for document_idx, document in enumerate(data):
            for word in document:
                if word in self.vocabulary.keys():
                    features[document_idx, self.vocabulary[word]] += 1
        return features



def normalize_data(train_data,  validation_data, type=None):
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler()

    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()

    elif type == 'l1':
        scaler = preprocessing.Normalizer(norm='l1')

    elif type == 'l2':
        scaler = preprocessing.Normalizer(norm='l2')

    if scaler is not None:
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_validation_data = scaler.transform(validation_data) 
        return (scaled_train_data, scaled_validation_data)
    else:
        print("No scaling was performed. Raw data is returned.")
        return (train_data, validation_data)



def precision(true_pos, true_neg, false_pos, false_neg):
    return true_pos / (true_pos + false_pos)


def recall(true_pos, true_neg, false_pos, false_neg):
    return true_pos / (true_pos + false_neg)

def accuracy(true_pos, true_neg, false_pos, false_neg):
    return (true_pos + true_neg) / (false_pos + false_neg + true_pos + true_neg)

def f1_score(true_pos, true_neg, false_pos, false_neg):
    return 2 * recall(true_pos , true_neg, false_pos, false_neg) * precision(true_pos, true_neg, false_pos, false_neg) / (recall(true_pos , true_neg, false_pos, false_neg) + precision(true_pos, true_neg, false_pos, false_neg))

def automatizare_gasire_neuroni():



    train_samples = list()
    train_labels = list()

    validation_labels = list()
    validation_samples = list()



    with open("train_samples.txt" , "r") as f:


        fisier = f.readline()
        fisier = fisier.strip("\n")
        fisier =fisier.split("\t")
        fisier = fisier[1].split(" ")
        train_samples.append(fisier)


        while fisier != [""]:

            fisier = f.readline()
            fisier = fisier.strip("\n")
            fisier =fisier.split("\t")

            if fisier != [""]:
                fisier = fisier[1].split(" ")
                train_samples.append(fisier)




    with open("validation_samples.txt", "r") as f:

        fisier = f.readline()
        fisier = fisier.strip("\n")
        fisier =fisier.split("\t")
        fisier = fisier[1].split(" ")
        validation_samples.append(fisier)


        while fisier != [""]:

            fisier = f.readline()
            fisier = fisier.strip("\n")
            fisier =fisier.split("\t")

            if fisier != [""]:
                fisier = fisier[1].split(" ")
                validation_samples.append(fisier)






    with open("train_labels.txt" , "r") as f:
        fisier = f.readline()
        fisier = fisier.strip("\n")
        fisier =fisier.split("\t")
        fisier = int(fisier[1])
        train_labels.append(fisier)


        while fisier != [""]:

            fisier = f.readline()
            fisier = fisier.strip("\n")
            fisier =fisier.split("\t")

            if fisier != [""]:
                fisier = int(fisier[1])
                train_labels.append(fisier)


    with open("validation_labels.txt" , "r") as f:
        fisier = f.readline()
        fisier = fisier.strip("\n")
        fisier =fisier.split("\t")
        fisier = int(fisier[1])
        validation_labels.append(fisier)


        while fisier != [""]:

            fisier = f.readline()
            fisier = fisier.strip("\n")
            fisier =fisier.split("\t")

            if fisier != [""]:
                fisier = int(fisier[1])
                validation_labels.append(fisier)



    bow = Bag_of_words()
    bow.build_vocabulary(train_samples)


    train_features = bow.get_features(train_samples)
    validation_features = bow.get_features(validation_samples)


    train_features,  validation_features = normalize_data(train_features, validation_features, type='l2')



    formatie_neuroni = [random.randint(3, 10) for i in range(random.randint(1, 10))]

    clf = MLPClassifier(solver='adam', alpha=10**-5,
                         hidden_layer_sizes=(formatie_neuroni), random_state=5)

    clf.fit(train_features, train_labels)




    raspuns  = clf.predict(validation_features)

    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    bune = 0
    rele = 0

    for ghicit, adevar in zip(raspuns, validation_labels):

        if ghicit == 1 and adevar == 1:
            true_pos += 1

        if ghicit == 1 and adevar == 0:
            false_pos += 1

        if ghicit == 0 and adevar == 1:
            false_neg += 1

        if ghicit == 0 and adevar == 0:
            true_neg += 1


        if ghicit == adevar:
            bune += 1

        else:
            rele += 1



    return f1_score(true_pos, true_neg, false_pos, false_neg), formatie_neuroni, true_pos, true_neg, false_pos, false_neg






if __name__ == "__main__":

    f1,  formatie_optima, true_pos_optim, true_neg_optim, false_pos_optim, false_neg_optim = automatizare_gasire_neuroni()
    max = f1


    

    for i in range(1):
        try:
            f1, formatie_neuroni, true_pos, true_neg, false_pos, false_neg = automatizare_gasire_neuroni()

            print("\n############" + str(i)  + "#############")

            if f1 > max:

                formatie_optima, true_pos_optim, true_neg_optim, false_pos_optim, false_neg_optim = formatie_neuroni, true_pos, true_neg, false_pos, false_neg


                max = f1

        except KeyboardInterrupt:
            print("Execution halted, showing results for the best so far\n")
            break

        except:
            continue


    print('\nFromatia neuronilor:')
    print(formatie_optima)
    print("\nTRUE POS: " + str(true_pos_optim) + "           "  + "FALSE POS: " + str(false_pos_optim))
    print("FALSE NEG: " + str(false_neg_optim) + "         " + "TRUE NEG: " + str(true_neg_optim))
    print("\nBune: " + str(true_pos_optim + true_neg_optim))
    print("Rele: " + str(false_pos_optim + false_neg_optim))
    print("Acuratete: " + str(accuracy(true_pos_optim, true_neg_optim, false_pos_optim, false_neg_optim)))
    print("F1_score: " + str(f1_score(true_pos_optim, true_neg_optim, false_pos_optim, false_neg_optim)))