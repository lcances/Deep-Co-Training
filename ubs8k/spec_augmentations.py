import numpy as np
from augmentations import SpecAugmentation

from PIL import Image


def random_interpolation():
    """
    Returns:
        a random interpolation filter for the Image library
    """

    filters = [Image.NEAREST, Image.BOX, Image.BILINEAR, Image.HAMMING, Image.BICUBIC]
    return np.random.choice(filters)


class VerticalFlip(SpecAugmentation):
    def __init__(self, ratio):
        super().__init__(ratio)

    def apply_helper(self, data):
        return np.flipud(data)


class HorizontalFlip(SpecAugmentation):
    def __init__(self, ratio):
        super().__init__(ratio)

    def apply_helper(self, data):
        return np.fliplr(data)


class Noise(SpecAugmentation):
    def __init__(self, ratio, snr, mini=-80, maxi=0):
        super().__init__(ratio)
        self.snr = snr
        self.mini = mini
        self.maxi = maxi

    def apply_helper(self, data):
        random_min = self.maxi
        random_max = - self.snr

        noise = np.random.uniform(random_min, random_max, size=data.shape)
        noisy_data = data + noise
        return np.clip(noisy_data, self.mini, self.maxi)


class FractalTimeStretch(SpecAugmentation):
    def __init__(self, ratio, intra_ratio: float = 0.3, rate: tuple = (0.8, 1.2),
                 min_chunk_size: int = None, max_chunk_size: int = None):
        """
        Args:
            ratio: The probability of applying the augmentation of the file
            intra_ratio (float): The probability to apply time stretch on one column
            rate (tuple): The min and max stretch value to use
            min_chunk_size (int): the minimum size of a column
            max_chunk_size (int): The maximum size of a column
        """
        super().__init__(ratio)

        # This values, if set to None, will be automatically computed when needed
        self.min_chunk_size = min_chunk_size  # 1% of the total length
        self.max_chunk_size = max_chunk_size  # 10% of the total length

        # ratio to apply or not the stretching on each columns (independantly)
        self.intra_ratio = intra_ratio
        self.rate = rate

    def apply_helper(self, data):
        (h, w) = data.shape

        # Compute min and max column size if needed
        self.min_chunk_size = int(w * 0.01) if self.min_chunk_size is None else self.min_chunk_size
        self.max_chunk_size = int(w * 0.1) if self.max_chunk_size is None else self.max_chunk_size

        # Split the spectro into many small chunks (random size)
        freq, temps = data.shape
        chunks_width = []
        chunks = []

        while sum(chunks_width) < w:
            width = np.random.randint(self.min_chunk_size, self.max_chunk_size)

            current_index = sum(chunks_width)
            chunks.append(data[:, current_index:current_index+width])

            chunks_width.append(width)

        # Apply time stretch to each column (intra_ratio)
        ratios = np.random.uniform(0, 1, len(chunks))
        stretched_chunks = []

        for ratio, width, column in zip(ratios, chunks_width, chunks):
            if ratio <= self.intra_ratio:
                rate = np.random.uniform(*self.rate)
                column = Image.fromarray(column)

                stretched_column = Image.Image.resize(
                    column,
                    (int(width * rate), h),
                    random_interpolation()
                )

                stretched_chunks.append(np.array(stretched_column))

            else:
                stretched_chunks.append(column)

        # Final resized to original dimension
        stretched_S = np.concatenate(stretched_chunks, axis=1)

        tmp = Image.fromarray(stretched_S)
        tmp = Image.Image.resize(tmp, (temps, freq), Image.LANCZOS)

        final_S = np.array(tmp)

        return np.float32(final_S)


class FractalFreqStretch(SpecAugmentation):
    def __init__(self, ratio, intra_ratio: float = 0.3, rate: tuple = (0.8, 1.2),
                 min_chunk_size: int = None, max_chunk_size: int = None):
        """
        Args:
            ratio: The probability of applying the augmentation of the file
            intra_ratio (float): The probability to apply time stretch on one column
            rate (tuple): The min and max stretch value to use
            min_chunk_size (int): the minimum size of a column
            max_chunk_size (int): The maximum size of a column
        """
        super().__init__(ratio)

        # This values, if set to None, will be automatically computed when needed
        self.min_chunk_size = min_chunk_size  # 1% of the total length
        self.max_chunk_size = max_chunk_size  # 10% of the total length

        # ratio to apply or not the stretching on each columns (independantly)
        self.intra_ratio = intra_ratio
        self.rate = rate

    def apply_helper(self, data):
        (h, w) = data.shape

        # Compute min and max chunk size if needed
        self.min_chunk_size = int(h * 0.01) if self.min_chunk_size is None else self.min_chunk_size
        self.max_chunk_size = int(h * 0.1) if self.max_chunk_size is None else self.max_chunk_size

        # Split the spectro into many small chunks (random size)
        freq, temps = data.shape
        chunk_width = []
        chunks = []

        while sum(chunk_width) < h:
            width = np.random.randint(self.min_chunk_size, self.max_chunk_size)

            current_index = sum(chunk_width)
            chunks.append(data[current_index:current_index+width, :])

            chunk_width.append(width)

        # Apply time stretch to each chunk (intra_ratio)
        ratios = np.random.uniform(0, 1, len(chunks))
        stretched_chunks = []

        for ratio, width, chunk in zip(ratios, chunk_width, chunks):
            if ratio <= self.intra_ratio:
                rate = np.random.uniform(*self.rate)
                chunk = Image.fromarray(chunk)

                stretched_column = Image.Image.resize(
                    chunk,
                    (w, int(width * rate)),
                    random_interpolation()
                )

                stretched_chunks.append(np.array(stretched_column))

            else:
                stretched_chunks.append(chunk)

        # Final resized to original dimension
        stretched_S = np.concatenate(stretched_chunks, axis=0)

        tmp = Image.fromarray(stretched_S)
        tmp = Image.Image.resize(tmp, (temps, freq), Image.LANCZOS)

        final_S = np.array(tmp)

        return np.float32(final_S)


class FractalTimeDropout(SpecAugmentation):
    def __init__(self, ratio, intra_ratio: float = 0.1,
                 min_chunk_size: int = None, max_chunk_size: int = None,
                 void: bool = True):
        super().__init__(ratio)

        # This values, if set to None, will be automatically computed when needed
        self.min_chunk_size = min_chunk_size  # 1% of the total length
        self.max_chunk_size = max_chunk_size  # 10% of the total length

        # ratio to apply or not the stretching on each columns (independantly)
        self.intra_ratio = intra_ratio
        self.void = void

    def apply_helper(self, data):
        (h, w) = data.shape
        mini = data.min()

        # Compute min and max column size if needed
        self.min_chunk_size = int(h * 0.01) if self.min_chunk_size is None else self.min_chunk_size
        self.max_chunk_size = int(h * 0.1) if self.max_chunk_size is None else self.max_chunk_size

        # Split the spectro into many small chunks (random size)
        chunk_width = []
        chunks = []

        while sum(chunk_width) < w:
            width = np.random.randint(self.min_chunk_size, self.max_chunk_size)

            current_index = sum(chunk_width)
            chunks.append(data[:, current_index:current_index+width])

            chunk_width.append(width)

        # each chunk have an <intra_ratio> chance of disapearing.
        valid_mask = [0 if x <= self.intra_ratio else 1 for x in np.random.uniform(0, 1, size=len(chunks))]

        # minimum one chunk have to disapear
        if sum(valid_mask) == len(chunks):
            valid_mask[np.random.choice(range(len(chunks)))] = 0

        # TODO add max limit

        # reconstruct the signal using void or compacting it
        reconstructed_S = []
        for valid, chunk in zip(valid_mask, chunks):
            if valid:
                reconstructed_S.append(chunk)
            else:
                reconstructed_S.append(list(np.ones(chunk.shape) * mini))

        reconstructed_S = np.concatenate(reconstructed_S, axis=1)

        return np.float32(reconstructed_S)


class FractalFrecDropout(SpecAugmentation):
    def __init__(self, ratio, intra_ratio: float = 0.1,
                 min_chunk_size: int = None, max_chunk_size: int = None,
                 void: bool = True):
        super().__init__(ratio)

        # This values, if set to None, will be automatically computed when needed
        self.min_chunk_size = min_chunk_size  # 1% of the total length
        self.max_chunk_size = max_chunk_size  # 10% of the total length

        # ratio to apply or not the stretching on each columns (independantly)
        self.intra_ratio = intra_ratio
        self.void = void

    def apply_helper(self, data):
        (h, w) = data.shape
        mini = data.min()

        # Compute min and max column size if needed
        self.min_chunk_size = int(h * 0.01) if self.min_chunk_size is None else self.min_chunk_size
        self.max_chunk_size = int(h * 0.1) if self.max_chunk_size is None else self.max_chunk_size

        # Split the spectro into many small chunks (random size)
        chunk_width = []
        chunks = []

        while sum(chunk_width) < h:
            width = np.random.randint(self.min_chunk_size, self.max_chunk_size)

            current_index = sum(chunk_width)
            chunks.append(data[current_index:current_index + width :])

            chunk_width.append(width)

        # each chunk have an <intra_ratio> chance of disapearing.
        valid_mask = [0 if x <= self.intra_ratio else 1 for x in np.random.uniform(0, 1, size=len(chunks))]

        # minimum one chunk have to disapear
        if sum(valid_mask) == len(chunks):
            valid_mask[np.random.choice(range(len(chunks)))] = 0

        # TODO add max limit

        # reconstruct the signal using void or compacting it
        reconstructed_S = []
        for valid, chunk in zip(valid_mask, chunks):
            if valid:
                reconstructed_S.append(chunk)
            else:
                reconstructed_S.append(list(np.ones(chunk.shape) * mini))

        reconstructed_S = np.concatenate(reconstructed_S, axis=0)

        return np.float32(reconstructed_S)


class RandomTimeDropout(SpecAugmentation):
    def __init__(self, ratio, dropout: float = 0.5):
        super().__init__(ratio)

        self.dropout = dropout

    def apply_helper(self, data):
        out = data.copy()

        (h, w) = out.shape
        mini = out.min()
        valid_mask = [0 if x <= self.dropout else 1 for x in np.random.uniform(0, 1, size=w)]

        for valid, idx in zip(valid_mask, range(w-1)):
            if not valid:
                out[:, idx] = mini

        return np.float32(out)


class RandomFreqDropout(SpecAugmentation):
    def __init__(self, ratio, dropout: float = 0.5):
        super().__init__(ratio)

        self.dropout = dropout

    def apply_helper(self, data):
        out = data.copy()

        (h, w) = out.shape
        mini = out.min()
        valid_mask = [0 if x <= self.dropout else 1 for x in np.random.uniform(0, 1, size=h)]

        for valid, idx in zip(valid_mask, range(w-1)):
            if not valid:
                out[idx, :] = mini

        return np.float32(out)
