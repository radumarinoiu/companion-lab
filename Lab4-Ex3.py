import pandas as pd
from scipy.stats import entropy
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import metrics


def attribute_information_gain(base_entropy, attribute_name, attribute_values):
    attribute_positive = titanic[titanic[attribute_name] == attribute_values[0]]
    attribute_negative = titanic[titanic[attribute_name] == attribute_values[1]]
    count_positive_survivors = sum(attribute_positive["Survivors"])
    count_positive_deaths = sum(attribute_positive["Passengers"]) - sum(attribute_positive["Survivors"])
    count_negative_survivors = sum(attribute_negative["Survivors"])
    count_negative_deaths = sum(attribute_negative["Passengers"]) - sum(attribute_negative["Survivors"])
    return base_entropy - (
            sum(attribute_positive["Passengers"]) /
            sum(titanic["Passengers"]) *
            entropy([count_positive_survivors, count_positive_deaths], base=2) +

            sum(attribute_negative["Passengers"]) /
            sum(titanic["Passengers"]) *
            entropy([count_negative_survivors, count_negative_deaths], base=2)
    )


def sub1():
    attributes = ["Class", "Gender", "Age"]
    ig_list = [
        attribute_information_gain(base_entropy, "Class", ["Upper", "Lower"]),
        attribute_information_gain(base_entropy, "Gender", ["Male", "Female"]),
        attribute_information_gain(base_entropy, "Age", ["Adult", "Child"])
    ]
    return attributes[ig_list.index(max(ig_list))]


def sub2(attribute_name):
    attribute_values_dict = {
        "Class": ["Upper", "Lower"],
        "Gender": ["Male", "Female"],
        "Age": ["Adult", "Child"]
    }
    attribute_positive = titanic[titanic[attribute_name] == attribute_values_dict[attribute_name][0]]
    attribute_negative = titanic[titanic[attribute_name] == attribute_values_dict[attribute_name][1]]

    return((sum(attribute_positive["Survivors"]) + sum(attribute_negative["Survivors"])) /
           (sum(attribute_positive["Passengers"]) + sum(attribute_negative["Passengers"])))


def sub3(titanic2):
    titanic2["Class"] = [1 if element == "Upper" else 0 for element in titanic["Class"]]
    titanic2["Gender"] = [1 if element == "Male" else 0 for element in titanic["Gender"]]
    titanic2["Age"] = [1 if element == "Adult" else 0 for element in titanic["Age"]]
    new_list = []
    for index, line in titanic2.iterrows():
        survivors_count = line[4]
        deaths_count = line[3] - survivors_count
        for _ in range(survivors_count):
            new_list.append((line[0], line[1], line[2], 1))
        for _ in range(deaths_count):
            new_list.append((line[0], line[1], line[2], 0))
    titanic2 = pd.DataFrame(new_list,
                            columns=["Class", "Gender", "Age", "Survivor"])
    dt = tree.DecisionTreeClassifier(criterion='entropy').fit(titanic2[["Class", "Gender", "Age"]], titanic2["Survivor"])
    predicted = dt.predict(titanic2[["Class", "Gender", "Age"]])
    print("Ex. 3: Accuracy:", metrics.accuracy_score(titanic2["Survivor"], predicted))
    fig, ax = plt.subplots(figsize=(16, 6))
    f = tree.plot_tree(dt, ax=ax, fontsize=8, feature_names=titanic2[["Class", "Gender", "Age"]].columns)
    plt.show()


if __name__ == '__main__':
    titanic = pd.DataFrame([
        ('Upper', 'Male', 'Child', 5, 5),
        ('Upper', 'Male', 'Adult', 175, 57),
        ('Upper', 'Female', 'Child', 1, 1),
        ('Upper', 'Female', 'Adult', 144, 140),
        ('Lower', 'Male', 'Child', 59, 24),
        ('Lower', 'Male', 'Adult', 1492, 281),
        ('Lower', 'Female', 'Child', 44, 27),
        ('Lower', 'Female', 'Adult', 281, 176)
    ],
        columns=['Class', 'Gender', 'Age', 'Passengers', 'Survivors'])
    count_survivors = sum(titanic["Survivors"])
    count_deaths = sum(titanic["Passengers"]) - sum(titanic["Survivors"])
    base_entropy = entropy([count_survivors, count_deaths], base=2)
    attribute = sub1()
    print("Ex. 1:", attribute)
    print("Ex. 2:", sub2(attribute))
    sub3(titanic)

