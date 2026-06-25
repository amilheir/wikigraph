ARG IRIS_IMAGE
FROM ${IRIS_IMAGE}

COPY --chown="${ISC_PACKAGE_MGRUSER}:${ISC_PACKAGE_IRISGROUP}" ./src-iris /opt/irisbuild

USER root
WORKDIR /opt
RUN python3 -m venv .venv
RUN . .venv/bin/activate && pip install --no-warn-script-location --break-system-packages --ignore-installed -r /opt/irisbuild/requirements_iris.txt

WORKDIR /opt/irisbuild
USER irisowner

RUN --mount=type=bind,src=./src-iris,dst=/opt/irisbuild \
    export IRISUSERNAME=_SYSTEM IRISPASSWORD=SYS IRISNAMESPACE=GRAPHRAG \
    && pip install --target /usr/irissys/mgr/python -r /opt/irisbuild/requirements_iris.txt \
    && iris start IRIS \
    && iris session IRIS < /opt/irisbuild/iris.script \
    && . /opt/.venv/bin/activate \
    && IRISINSTALLDIR=/usr/irissys LD_LIBRARY_PATH=/usr/irissys/bin:/usr/irissys/dev COMLIB=/usr/irissys/bin PYTHONPATH=/usr/irissys/lib/python \
       bash /opt/irisbuild/production/loadProduction.sh \
    && iris session IRIS < /opt/irisbuild/autostart.script \
    && iris stop IRIS quietly

# RUN --mount=type=secret,id=iris_pw,mode=0444 /usr/irissys/dev/Container/changePassword.sh /run/secrets/iris_pw
# RUN rm -r /opt/irisbuild/*
