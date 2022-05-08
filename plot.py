import matplotlib.pyplot as plt


def show_plot(time_array, result, title=None, x_label=None, y_label=None):
    fig, ax = plt.subplots()
    fig.tight_layout()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    for res in result:
        plt.plot(time_array, res)
    plt.show()
