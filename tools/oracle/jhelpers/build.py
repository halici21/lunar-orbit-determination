"""Build ``oracle-helpers.jar`` from the three helper sources against the jars shipped in orekit-jpype.

    <python with orekit-jpype> tools/oracle/jhelpers/build.py

Needs ``JAVA_HOME`` (JDK >= 21). Output: ``tools/oracle/jhelpers/build/oracle-helpers.jar`` (gitignored
through ``build/``; never committed). The sources are carried unchanged from the 2026-10-21 trial.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    import orekit_jpype

    jars = Path(orekit_jpype.__file__).parent / "jars"
    cp = os.pathsep.join(str(p) for p in sorted(jars.glob("*.jar"))
                         if p.name.startswith(("orekit-", "hipparchus-core", "hipparchus-geometry")))
    java_home = Path(os.environ["JAVA_HOME"])
    build = HERE / "build"
    classes = build / "classes"
    shutil.rmtree(build, ignore_errors=True)
    classes.mkdir(parents=True)
    sources = sorted(str(p) for p in (HERE / "oracle").glob("*.java"))
    subprocess.run([str(java_home / "bin" / "javac"), "--release", "21", "-cp", cp, "-d", str(classes), *sources],
                   check=True)
    jar = build / "oracle-helpers.jar"
    subprocess.run([str(java_home / "bin" / "jar"), "cf", str(jar), "-C", str(classes), "."], check=True)
    print(f"built {jar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
