"""Update package version.
    - reference: https://github.com/ubuntu/ubuntu-make/blob/master/confs/githooks/update_version

"""

import datetime
import os
import sys

if __name__ == "__main__":
    version_file_path = os.path.join(
        os.path.dirname(__file__), "..", "rl_algorithms", "version.py"
    )
    version_tempfile_path = f"{version_file_path}.tmp"

    version = f"{datetime.datetime.now():%y.%m}"

    old_version = (
        open(version_file_path, "r", encoding="utf-8").read().strip().split('"')[1]
    )
    if old_version[0:5] == version:
        minor = old_version[6:]
        # If need minor version
        if minor:
            try:
                minor = int(minor) + 1
                version += f".{minor}"
            except ValueError:
                print(f"{old_version} hasn't the expected format: YY.MM.<minor>")
                sys.exit(1)
        # If there was no minor version, add 0
        else:
            version += ".0"

    # Write new version
    with open(version_tempfile_path, "w", encoding="utf-8") as f:
        f.write(f'__version__ = "{version}"\n')
        f.flush()
        os.fsync(f.fileno())

    os.rename(version_tempfile_path, version_file_path)
    print(f"Update version {version}")
