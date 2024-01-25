FROM nasanccs/tensorflow-caney:latest
LABEL maintainer="jordan.a.caraballo-vega@nasa.gov"

ENV TFC_VERSION=0.1.1
ENV VHR_CLOUDMASK_VERSION=main

RUN pip install git+https://github.com/dshean/pygeotools.git@master \
        git+https://github.com/nasa-nccs-hpda/vhr-cloudmask.git@main

HEALTHCHECK NONE
ENTRYPOINT [""]
