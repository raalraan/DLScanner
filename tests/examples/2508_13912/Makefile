version=6.1.2
MGversion=3.6.2
MGversion_us=$(subst .,_,${MGversion})
NMSSMTools=NMSSMTools_${version}
MadGraph=MG5_aMC_v${MGversion}
MadGraph_us=MG5_aMC_v${MGversion_us}
workdir ?= ${PWD}
reNMSSMTools=${NMSSMTools}_${INDEX_OPT}

.PHONY: all

all: ${workdir}/${NMSSMTools} ${workdir}/${MadGraph_us}/output_dir_mg

${workdir}/${NMSSMTools}:
	@if [ "${workdir}" != "${PWD}" ]; then mkdir -p ${workdir}; fi
	cp ${NMSSMTools}.tgz ${workdir}
	cd ${workdir} && tar xvf ${NMSSMTools}.tgz
	@echo "NMSSMTools will work in ${workdir}/${NMSSMTools}"
	cd ${workdir}/${NMSSMTools} && make init && make

${workdir}/${MadGraph_us}/output_dir_mg:
	@if [ "${workdir}" != "${PWD}" ]; then mkdir -p ${workdir}; fi
	cp ${MadGraph}.tar.gz ${workdir}
	@echo "MadGraph will work in ${workdir}/${MadGraph_us}"
	cd ${workdir} && tar xvf ${MadGraph}.tar.gz
	cp mg_create_output me_calculate_xsec ${workdir}/${MadGraph_us}
	cd ${workdir}/${MadGraph_us} && ./bin/mg5_aMC mg_create_output

${NMSSMTools}.tgz:
	wget https://www.lupm.in2p3.fr/users/nmssm/$@

${MadGraph}.tar.gz:
	wget https://launchpad.net/mg5amcnlo/3.0/3.6.x/+download/$@
