import numpy as np

from PIL import Image


def random_interpolation():
    """
    Returns:
        a random interpolation filter for the Image library
    """

    filters = [Image.NEAREST, Image.BOX, Image.BILINEAR, Image.HAMMING, Image.BICUBIC]
    return np.random.choice(filters)


class Augmentation:
    def __init__(self, ratio):
        """
        Args:
            ratio: The probability to apply the augmentation
        """
        self.ratio = ratio

    def _perform_augmentation(self, data):
        """Perform augmentation if needed

        Args:
            data: The spectrogram to use
        """
        if np.random.random() < self.ratio:
            return self._apply(data)
        return data

    def _apply(self, data):
        """
        Args:
            data: The spectrogram to use
        """
        raise NotImplementedError("This is an abstract class")

    def __call__(self, data):
        """
        Args:
            data: The spectrogram to use
        """
        return self._perform_augmentation(data)


class FractalTimeStretch(Augmentation):
    def __init__(self, ratio, intra_ratio: float = 0.3, rate: tuple = (0.8, 1.2),
                 min_column_size: int = None, max_column_size: int = None):
        """
        Args:
            ratio: The probability of applying the augmentation of the file
            intra_ratio (float): The probability to apply time stretch on one column
            rate (tuple): The min and max stretch value to use
            min_column_size (int): the minimum size of a column
            max_column_size (int): The maximum size of a column
        """
        super().__init__(ratio)

        # This values, if set to None, will be automatically computed when needed
        self.min_column_size = min_column_size  # 1% of the total length
        self.max_column_size = max_column_size  # 10% of the total length

        # ratio to apply or not the stretching on each columns (independantly)
        self.intra_ratio = intra_ratio
        self.rate = rate

    def _apply(self, data):
        S_im = Image.fromarray(data)
        (w, h) = S_im.size

        # Compute min and max column size if needed
        self.min_column_size = int(w * 0.01) if self.min_column_size is None else self.min_column_size
        self.max_column_size = int(w * 0.1) if self.max_column_size is None else self.max_column_size

        # Split the spectro into many small columns (random size)
        freq, temps = data.shape
        columns_width = []
        columns = []

        while sum(columns_width) < w:
            width = np.random.randint(self.min_column_size, self.max_column_size)

            current_index = sum(columns_width)
            columns.append(data[:, current_index:current_index+width])

            columns_width.append(width)

        # Apply time stretch to each column (intra_ratio)
        ratios = np.random.uniform(0, 1, len(columns))
        stretched_columns = []

        print(columns_width)
        print(ratios)
        print(len(columns))
        for ratio, width, column in zip(ratios, columns_width, columns):
            if ratio <= self.intra_ratio:
                rate = np.random.uniform(*self.rate)
                column = Image.fromarray(column)

                print(width)
                stretched_column = Image.Image.resize(
                    column,
                    (int(width * rate), h),
                    random_interpolation()
                )

                stretched_columns.append(np.array(stretched_column))

            else:
                stretched_columns.append(column)

        # Final resized to original dimension
        stretched_S = np.concatenate(stretched_columns, axis=1)

        tmp = Image.fromarray(stretched_S)
        tmp = Image.Image.resize(tmp, (temps, freq), Image.LANCZOS)

        final_S = np.array(tmp)

        return final_S


class FractalFreqStretch(Augmentation):
    def __init__(self, ratio, intra_ratio: float = 0.3, rate: tuple = (0.8, 1.2),
                 min_column_size: int = None, max_column_size: int = None):
        """
        Args:
            ratio: The probability of applying the augmentation of the file
            intra_ratio (float): The probability to apply time stretch on one column
            rate (tuple): The min and max stretch value to use
            min_column_size (int): the minimum size of a column
            max_column_size (int): The maximum size of a column
        """
        super().__init__(ratio)

        # This values, if set to None, will be automatically computed when needed
        self.min_column_size = min_column_size  # 1% of the total length
        self.max_column_size = max_column_size  # 10% of the total length

        # ratio to apply or not the stretching on each columns (independantly)
        self.intra_ratio = intra_ratio
        self.rate = rate

    def _apply(self, data):
        S_im = Image.fromarray(data)
        (w, h) = S_im.size

        # Compute min and max column size if needed
        self.min_column_size = int(h * 0.01) if self.min_column_size is None else self.min_column_size
        self.max_column_size = int(h * 0.1) if self.max_column_size is None else self.max_column_size

        # Split the spectro into many small chunks (random size)
        freq, temps = data.shape
        chunk_width = []
        chunks = []

        while sum(chunk_width) < h:
            width = np.random.randint(self.min_column_size, self.max_column_size)

            current_index = sum(chunk_width)
            chunks.append(data[current_index:current_index+width, :])

            chunk_width.append(width)

        # Apply time stretch to each column (intra_ratio)
        ratios = np.random.uniform(0, 1, len(chunks))
        stretched_columns = []

        print(chunk_width)
        print(ratios)
        print(len(chunks))
        for ratio, width, column in zip(ratios, chunk_width, chunks):
            if ratio <= self.intra_ratio:
                rate = np.random.uniform(*self.rate)
                column = Image.fromarray(column)

                print(width)
                stretched_column = Image.Image.resize(
                    column,
                    (w, int(width * rate)),
                    random_interpolation()
                )

                stretched_columns.append(np.array(stretched_column))

            else:
                stretched_columns.append(column)

        # Final resized to original dimension
        stretched_S = np.concatenate(stretched_columns, axis=0)

        tmp = Image.fromarray(stretched_S)
        tmp = Image.Image.resize(tmp, (temps, freq), Image.LANCZOS)

        final_S = np.array(tmp)

        return final_S


if __name__ == '__main__':
    from datasetManager import DatasetManager

    audio_root = "../dataset/audio"
    metadata_root = "../dataset/metadata"
    dataset = DatasetManager(metadata_root, audio_root, train_fold = [1], val_fold=[])
