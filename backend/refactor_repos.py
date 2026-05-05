import os
import re
def refactor_repo(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # Add init if missing
    if "def __init__" not in content:
        content = re.sub(
            r"(class \w+Repository:)",
            r"\1\n    def __init__(self, db: AsyncSession):\n        self.db = db\n",
            content
        )
    # Remove db: AsyncSession from signatures
    content = re.sub(r",\s*db:\s*AsyncSession", "", content)
    content = re.sub(r"db:\s*AsyncSession\s*,?", "", content)
    # Replace db. with self.db. inside methods
    content = re.sub(r"(\s+)await db\.", r"\1await self.db.", content)
    content = re.sub(r"(\s+)db\.", r"\1self.db.", content)
    content = re.sub(r"self\.db\s*=\s*self\.db", "self.db = db", content) # fix
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
for root, _, files in os.walk('repositories'):
    for file in files:
        if file.endswith('_repository.py'):
            refactor_repo(os.path.join(root, file))
print("Repos refactored!")
