"""Minimal torch-free reader for a torch.save'd state dict of float tensors."""
import zipfile, pickle, collections
import numpy as np

_DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64,
       "IntStorage": np.int32, "HalfStorage": np.float16, "BoolStorage": np.bool_}

class _StorageType:
    def __init__(self, name): self.name = name

def load_pth(path):
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]
    cache = {}
    def persistent_load(pid):
        _, stype, key, loc, numel = pid
        if key not in cache:
            dt = _DT[stype.name]
            cache[key] = np.frombuffer(z.read(f"{root}/data/{key}"), dtype=dt).copy()
        return cache[key]
    def rebuild(storage, offset, size, stride, *a):
        if len(size) == 0:
            return storage[offset]
        return np.lib.stride_tricks.as_strided(storage[offset:], shape=size,
                                               strides=[s * storage.itemsize for s in stride]).copy()
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "torch._utils" and name == "_rebuild_tensor_v2": return rebuild
            if mod == "torch" and name.endswith("Storage"): return _StorageType(name)
            if mod == "collections" and name == "OrderedDict": return collections.OrderedDict
            return super().find_class(mod, name)
        def persistent_load(self, pid): return persistent_load(pid)
    return U(z.open(f"{root}/data.pkl")).load()

if __name__ == "__main__":
    import sys
    d = load_pth(sys.argv[1])
    for k, v in d.items():
        print(k, type(v))
        if isinstance(v, dict):
            for kk, vv in v.items(): print("   ", kk, getattr(vv, "shape", vv))
