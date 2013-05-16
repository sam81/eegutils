#!/bin/sh

cd ../doc
./mkdoc.sh

cd ../prep-release
./mkupdate.sh
./distbuild.sh 
