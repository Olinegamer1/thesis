import matplotlib.pyplot as plt


def show_plot(time_array, result):
    fig, ax = plt.subplots()
    fig.tight_layout()
    plt.grid()
    for res in result:
        plt.plot(time_array, res)
    plt.show()
