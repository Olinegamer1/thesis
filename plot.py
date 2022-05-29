import matplotlib.pyplot as plt


def show_plot(x_array, y_array, title=None, x_label=None, y_label=None, labels=None):
    fig, ax = plt.subplots()
    fig.tight_layout()
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.grid()
    for y_arr, label in zip(y_array, labels):
        plt.plot(x_array, y_arr, label=f"{label} - радиальная скорость")
    plt.legend()
    plt.show()


def show_multiplot(x_array, y_array, title=None, x_label=None, y_label=None, labels=None):
    fig, ax = plt.subplots()
    fig.tight_layout()
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.grid()
    for y_arr, x_arr, label in zip(y_array, x_array, labels):
        plt.plot(x_arr, y_arr, label=f"{label} - радиальная скорость")
    plt.legend()
    plt.show()
