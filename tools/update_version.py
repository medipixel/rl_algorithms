"""Update package version.
    - reference: https://github.com/ubuntu/ubuntu-make/blob/master/confs/githooks/update_version

"""

import os

if __name__ == "__main__":
    version_file_path = os.path.join(
        os.path.dirname(__file__), "..", "rl_algorithms", "version.py"
    )
    version_tempfile_path = f"{version_file_path}.tmp"

    old_version = (
        open(version_file_path, "r", encoding="utf-8").read().strip().split('"')[1]
    )

    minor = old_version[-1]
    # Update minor version
    version = old_version
    if minor:
        minor = int(minor) + 1
        version = old_version[:-1] + f"{minor}"

    # Write new version
    with open(version_tempfile_path, "w", encoding="utf-8") as f:
        f.write(f'__version__ = "{version}"\n')
        f.flush()
        os.fsync(f.fileno())

    os.rename(version_tempfile_path, version_file_path)
    print(f"Update version {old_version} -> {version}")
