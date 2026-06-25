#!/usr/bin/env bash
# Compile the WikiGraph pyprod production from its split modules.
#
# pyprod compiles only the classes physically present in the file it is handed (imports are NOT
# followed), so each module that defines interoperability classes must be compiled on its own.
# common.py defines no such classes - it is a plain helper module - so it is intentionally skipped.
# wikiGraph.py (the Production) is compiled LAST, after its components exist, so item validation
# can resolve every config-item class.
#
# Run inside the IRIS container with the venv active and the connection vars set, exactly as the
# Dockerfile does:
#   . /opt/.venv/bin/activate
#   IRISUSERNAME=_SYSTEM IRISPASSWORD=SYS IRISNAMESPACE=GRAPHRAG bash loadProduction.sh
# Inherit the image's default environment (IRISINSTALLDIR / LD_LIBRARY_PATH / PYTHONPATH) as-is -
# pyprod's embedded-iris callin needs those library paths, so do NOT unset them.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

for module in messages services processes operations wikiGraph; do
    echo ">> compiling ${module}.py"
    intersystems_pyprod "${DIR}/${module}.py"
done

echo ">> WikiGraph production compiled."
