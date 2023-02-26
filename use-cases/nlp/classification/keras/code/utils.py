def extract_files(source: str, destination: str) -> None:
    """Extracts files from the source archive to the destination path

    Args:
        source (str): Archive file path
        destination (str): Extract destination path
    """

    import tarfile

    print("Extracting data...")
    with tarfile.open(source) as f:
        f.extractall(destination)


def prep_data(imdb_folder: str, dest_path: str) -> None:
    """Removes unnecessary folders/files and renames source folder

    Args:
        imdb_folder (str): Path of the aclImdb folder
        dest_name (str): Destination folder to which the aclImdb folder has to be renamed to
    """
    import os
    import shutil

    shutil.rmtree(f"{imdb_folder}/train/unsup")
    os.remove(f"{imdb_folder.rsplit('/', maxsplit=1)[0]}/aclImdb_v1.tar.gz")

    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    os.rename(imdb_folder, dest_path)
    print(f"{imdb_folder} renamed to {dest_path}")


def build_model(model_params: dict, data_params: dict):
    """Accepts model and data parameters to build and compile a keras model

    Args:
        model_params (dict): Model parameters
        data_params (dict): Data parameters

    Returns:
        A compiled keras model
    """

    import tensorflow as tf
    from tensorflow.keras import layers

    # A integer input for vocab indices.
    inputs = tf.keras.Input(shape=(None,), dtype="int64")

    # Next, we add a layer to map those vocab indices into a space of dimensionality
    # 'embedding_dim'.
    x = layers.Embedding(data_params["max_features"], data_params["embedding_dim"])(inputs)
    x = layers.Dropout(model_params["dropout"])(x)

    # Conv1D + global max pooling
    x = layers.Conv1D(
        data_params["embedding_dim"],
        model_params["kernel_size"],
        padding="valid",
        activation=model_params["activation"],
        strides=model_params["strides"],
    )(x)
    x = layers.Conv1D(
        data_params["embedding_dim"],
        model_params["kernel_size"],
        padding="valid",
        activation=model_params["activation"],
        strides=model_params["strides"],
    )(x)
    x = layers.GlobalMaxPooling1D()(x)

    # We add a vanilla hidden layer:
    x = layers.Dense(data_params["embedding_dim"], activation=model_params["activation"])(x)
    x = layers.Dropout(model_params["dropout"])(x)

    # We project onto a single unit output layer, and squash it with a sigmoid:
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    keras_model = tf.keras.Model(inputs, predictions)

    # Compile the model with binary crossentropy loss and an adam optimizer.
    keras_model.compile(
        loss=model_params["loss"],
        optimizer=model_params["optimizer"],
        metrics=model_params["metrics"],
    )

    return keras_model
