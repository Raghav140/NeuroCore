import numpy as np

from . import config
from .activations import ReLU, Sigmoid, Tanh
from .dataset import make_xor, make_binary_classification, make_regression
from .layers import Dense, Dropout
from .losses import MeanSquaredError, BinaryCrossEntropy
from .model import NeuralNetwork
from .optimizer import GradientDescent, ReduceLROnPlateau, StepLR
from .utils import set_random_seed, accuracy, train_test_split


def build_xor_model() -> NeuralNetwork:
    model = NeuralNetwork()
    model.add(Dense(2, 8, weight_init="he"))
    model.add(ReLU())
    model.add(Dense(8, 1, weight_init="xavier"))
    model.add(Sigmoid())
    model.compile(loss=BinaryCrossEntropy(), optimizer=GradientDescent(lr=config.LR_XOR))
    return model


def build_binary_classification_model(input_dim: int) -> NeuralNetwork:
    model = NeuralNetwork()
    model.add(Dense(input_dim, 16, weight_init="he", l2_lambda=1e-4))
    model.add(ReLU())
    model.add(Dropout(0.1))
    model.add(Dense(16, 8, weight_init="he", l2_lambda=1e-4))
    model.add(ReLU())
    model.add(Dense(8, 1, weight_init="xavier"))
    model.add(Sigmoid())
    model.compile(
        loss=BinaryCrossEntropy(), optimizer=GradientDescent(lr=config.LR_BINARY)
    )
    return model


def build_regression_model(input_dim: int) -> NeuralNetwork:
    model = NeuralNetwork()
    model.add(Dense(input_dim, 32, weight_init="he", l2_lambda=1e-4))
    model.add(Tanh())
    model.add(Dense(32, 16, weight_init="he", l2_lambda=1e-4))
    model.add(Tanh())
    model.add(Dense(16, 1, weight_init="xavier"))
    model.compile(
        loss=MeanSquaredError(),
        optimizer=GradientDescent(lr=config.LR_REGRESSION, momentum=0.9),
    )
    return model


def run_xor_experiment():
    print("\n=== XOR Problem ===")
    set_random_seed(config.SEED)
    X, y = make_xor(n_samples=400)

    model = build_xor_model()
    model.train(X, y, epochs=config.EPOCHS_XOR, batch_size=len(X), verbose=False)

    preds = model.predict(X)
    loss = BinaryCrossEntropy().forward(y, preds)
    acc = accuracy(y, preds)
    print(f"Final loss: {loss:.4f} - accuracy: {acc:.4f}")


def run_binary_classification_experiment():
    print("\n=== Binary Classification ===")
    set_random_seed(config.SEED)
    X, y = make_binary_classification(n_samples=600, noise=0.2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, seed=config.SEED
    )
    model = build_binary_classification_model(input_dim=X.shape[1])
    scheduler = StepLR(
        model.optimizer,
        step_size=config.STEP_LR_BINARY_STEP_SIZE,
        gamma=config.STEP_LR_BINARY_GAMMA,
    )

    model.train(
        X_train,
        y_train,
        epochs=config.EPOCHS_BINARY,
        batch_size=config.BATCH_SIZE,
        verbose=False,
        scheduler=scheduler,
    )

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_loss = BinaryCrossEntropy().forward(y_train, train_preds)
    test_loss = BinaryCrossEntropy().forward(y_test, test_preds)
    train_acc = accuracy(y_train, train_preds)
    test_acc = accuracy(y_test, test_preds)

    print(
        f"Train loss: {train_loss:.4f} - acc: {train_acc:.4f} | "
        f"Test loss: {test_loss:.4f} - acc: {test_acc:.4f}"
    )

    # Demonstrate model save/load.
    save_prefix = config.BINARY_MODEL_SAVE_PREFIX
    model.save(save_prefix)
    loaded_model = NeuralNetwork.load(save_prefix)
    loaded_test_preds = loaded_model.predict(X_test)
    loaded_test_acc = accuracy(y_test, loaded_test_preds)
    print(f"Loaded model test accuracy: {loaded_test_acc:.4f}")


def run_regression_experiment():
    print("\n=== Regression ===")
    set_random_seed(config.SEED)
    X, y = make_regression(n_samples=600, noise=0.1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, seed=config.SEED
    )

    model = build_regression_model(input_dim=X.shape[1])
    scheduler = ReduceLROnPlateau(
        model.optimizer,
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=config.REDUCE_LR_MIN_LR,
        min_delta=config.REDUCE_LR_MIN_DELTA,
    )
    model.train(
        X_train,
        y_train,
        epochs=config.EPOCHS_REGRESSION,
        batch_size=config.BATCH_SIZE,
        verbose=False,
        X_val=X_test,
        y_val=y_test,
        scheduler=scheduler,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=config.EARLY_STOPPING_MIN_DELTA,
        early_stopping_monitor="val_loss",
    )

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_loss = MeanSquaredError().forward(y_train, train_preds)
    test_loss = MeanSquaredError().forward(y_test, test_preds)

    print(f"Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")


if __name__ == "__main__":
    run_xor_experiment()
    run_binary_classification_experiment()
    run_regression_experiment()

