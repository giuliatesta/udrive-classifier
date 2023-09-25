from matplotlib.pyplot import figure, plot, legend, title, ylabel, xlabel, savefig, grid

from convolutional_neural_network import K, EPOCHS, BATCH_SIZE

CNN_PLOT_PATH = "./plots/cnn"
FF_PLOT_PATH = "./plots/ff"


def _average(data):
    return sum(data) / len(data)


def plot_accuracy(histories, epochs=EPOCHS, batch_size=BATCH_SIZE, cnn=True):
    _plot(histories, "accuracy", epochs=epochs, batch_size=batch_size, cnn=cnn)


def plot_loss(histories, epochs=EPOCHS, batch_size=BATCH_SIZE, cnn=True):
    _plot(histories, "loss", epochs=epochs, batch_size=batch_size, cnn=cnn)


def _plot(histories, name, epochs=EPOCHS, batch_size=BATCH_SIZE, cnn=True):
    train = []
    validation = []

    for i in range(K-1):
        train.append(_average(histories[i].history.get(name)))     #average
        validation.append(_average(histories[i].history.get(f"val_{name}")))  # average

    figure(figsize=(12, 8))
    grid()
    plot(range(1, K), train, 'o-g', label=f"Training {name}")
    plot(range(1, K), validation, 'x-r', label=f"Validation {name}")
    legend(loc='best')
    title(f"Average {name}")
    ylabel(name)
    xlabel("Fold")
    savefig(f"{CNN_PLOT_PATH if cnn else FF_PLOT_PATH}/{name}-{epochs}-{batch_size}.png")
