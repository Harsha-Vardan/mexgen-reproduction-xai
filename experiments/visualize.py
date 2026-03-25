import matplotlib.pyplot as plt

def plot_faithfulness_vs_cost(cost, faithfulness):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(cost, faithfulness)

    plt.xlabel("Cost")
    plt.ylabel("Faithfulness")
    plt.title("Faithfulness vs Cost")

    plt.show()
    
def plot_importance(scores):
    """
    Plot normalized importance scores
    """
    indices = [idx for idx, _, _, _ in scores]
    values = [norm for _, _, _, norm in scores]

    plt.figure()
    plt.bar(indices, values)

    plt.xlabel("Sentence Index")
    plt.ylabel("Normalized Importance")
    plt.title("Sentence Importance (MExGen)")

    plt.show()