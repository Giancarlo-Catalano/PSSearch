#!/usr/bin/env python3
import pickle
import gzip
import sys


def load_with_gzip_custom(path):
    try:
        with gzip.open(path, "rb") as f:
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # remap old numpy module name to current
                    if module == 'numpy._core':
                        module = 'numpy.core'
                    return super().find_class(module, name)

            data = CustomUnpickler(f).load()
        print("Success with gzip using custom unpickler.")
        print(data)

        return data
    except Exception as e:
        print("load_with_gzip_custom failed:", e)
        return None


def load_with_plain_open_custom(path):
    try:
        with open(path, "rb") as f:
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == 'numpy._core':
                        module = 'numpy.core'
                    return super().find_class(module, name)

            data = CustomUnpickler(f).load()
        print("Success with plain open using custom unpickler.")
        return data
    except Exception as e:
        print("load_with_plain_open_custom failed:", e)
        return None


def load_with_gzip_standard(path):
    try:
        with gzip.open(path, "rb") as f:
            # fix_imports=True sometimes helps with module remapping
            data = pickle.load(f, fix_imports=True)
        print("Success with gzip using standard pickle.load.")
        return data
    except Exception as e:
        print("load_with_gzip_standard failed:", e)
        return None


def load_with_plain_open_standard(path):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f, fix_imports=True)
        print("Success with plain open using standard pickle.load.")
        return data
    except Exception as e:
        print("load_with_plain_open_standard failed:", e)
        return None


def main(path):
    print("Trying to load file:", path)

    # Try gzipped with custom unpickler
    data = load_with_gzip_custom(path)
    if data is not None:
        return

    # Try plain binary open with custom unpickler
    data = load_with_plain_open_custom(path)
    if data is not None:
        return

    # Try gzipped with standard pickle.load
    data = load_with_gzip_standard(path)
    if data is not None:
        return

    # Try plain binary open with standard pickle.load
    data = load_with_plain_open_standard(path)
    if data is not None:
        return

    print("All methods failed to load the file.")


path = r"/data/retail_forecasting/cluster_info_20_qmc.pkl"
main(path)


