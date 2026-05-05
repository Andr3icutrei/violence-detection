import re
import os
with open("repositories/videos_repository.py", "rb") as f:
    text = f.read().decode("utf-8")
print(text[:20])
