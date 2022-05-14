import matplotlib.pyplot as plt


def show_plot(x_array, y_array, title=None, x_label=None, y_label=None):
    fig, ax = plt.subplots()
    fig.tight_layout()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    for y_arr in  y_array:
        plt.plot(x_array, y_arr)
    plt.show()
