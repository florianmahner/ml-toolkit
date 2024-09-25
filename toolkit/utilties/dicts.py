class IndexedDict(dict):
    def __getitem__(self, key):
        if isinstance(key, int):
            keys = list(self.keys())
            if key < 0:
                key += len(keys)
            if key >= len(keys) or key < 0:
                raise IndexError("Index out of range")
            return self[keys[key]]
        else:
            return super().__getitem__(key)
