
import threading


class ThreadSafeDict:
    def __init__(self):
        self.lock = threading.Lock()
        self.d = {}
    def __getitem__(self, key):
        with self.lock:
            return self.d[key]
    def __setitem__(self, key, value):
        with self.lock:
            self.d[key] = value
    
    def __delitem__(self, key):
        with self.lock:
            del self.d[key]
    def __len__(self):
        with self.lock:
            return len(self.d)
    def __iter__(self):
        with self.lock:
            return iter(self.d)
    def clear(self):
        with self.lock:
            self.d = {}

    def get_or_add(self, key, default_callback):
        with self.lock:
            if key in self.d:
                return self.d[key]
            else:
                self.d[key] = default_callback()
                return self.d[key]

    def insert_or_update(self, key, insert_callback, update_callback):
        with self.lock:
            if key in self.d:
                self.d[key] = update_callback(self.d[key])
            else:
                self.d[key] = insert_callback()
            return self.d[key]
        