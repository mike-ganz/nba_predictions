#!/usr/bin/env python3
import importlib.util as u
import os

print(os.getcwd())
for name in ["analysis", "game", "config", "data"]:
    try:
        print(f"{name}:", u.find_spec(name) is not None)
    except Exception as e:
        print(f"{name}: ERROR {e}")


